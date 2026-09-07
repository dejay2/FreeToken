"""Reusable pinned host-bank primitives shared by the fast expert-load paths.

Two ideas the parallel read of the original checkpoint and FTW (read a repacked
contiguous cache) paths both rely on:

* **pin-after-fill** -- allocate the bank as a *lazy* anonymous ``mmap`` (no pages
  resident, instant), fill it with real data, and only THEN ``cudaHostRegister`` it.
  Registering already-resident pages just page-locks them; registering a lazy mmap first
  faults+zero-fills every page (~137 GiB -> ~47 s for DSV4) and that zero-fill is then
  immediately overwritten by the read. So pin-after-fill removes a whole redundant pass.
* **chunked multi-threaded O_DIRECT** -- DMA straight from disk into the (page-aligned)
  bank, bypassing the page cache, with many concurrent ``preadv`` on one fd (scales to the
  device's queue-depth ceiling even for a single file).

The mmaps are held for the process lifetime (the banks live as long as the offload cache).
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import mmap
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import torch

from freetoken.utils import init_logger

from . import win_io

logger = init_logger(__name__)

_BLK = 4096  # O_DIRECT alignment (page size)


def _win_unbuffered() -> bool:
    """Route the readers below through the Windows ``FILE_FLAG_NO_BUFFERING`` backend.

    False on POSIX (O_DIRECT is already there) and whenever ``FREETOKEN_WIN_UNBUFFERED_IO=0``
    asks for the old buffered behaviour, so the Linux paths stay byte-identical."""
    return not hasattr(os, "O_DIRECT") and win_io.enabled()


class HostResidency(str, Enum):
    """Residency class of a host bank layer.

    Only PINNED (cudaHostRegister'd) memory can feed the GPU movement paths; LOCKED (mlock'd, no device address) and PAGEABLE layers must decode on the CPU executor.
    The non-pinned classes exist for hosts that cap CUDA pin quota (WSL/WDDM: ~half of RAM).
    GPU_OWNED is the odd one out: there is no host bank at all -- the layer's experts live in
    VRAM for the process lifetime, so it neither spends pin quota nor holds host pages.
    DISK has neither host nor device banks permanently allocated; weights live in an on-disk copy
    and rows are gathered into staging during decode or materialized for prefill.
    """

    PINNED = "pinned"
    LOCKED = "locked"
    PAGEABLE = "pageable"
    GPU_OWNED = "gpu_owned"
    DISK = "disk"


_DEFAULT_CHUNK = 8 << 20

# Hold the mmaps for the process lifetime; the offload cache reads from these banks forever.
_LIVE_BUFFERS: list[mmap.mmap] = []

def _env_born_pinned() -> bool | None:
    """``FREETOKEN_BANK_CUDA_ALLOC`` tri-state: unset -> ``None`` (default applies), else the parsed boolean."""
    v = os.environ.get("FREETOKEN_BANK_CUDA_ALLOC", "").strip().lower()
    if not v:
        return None
    return v in ("1", "true", "yes", "on")


def born_pinned_default() -> bool:
    """Whether PINNED serving banks use cudaHostAlloc instead of mmap + register-after-fill.

    Off by default: registered mmaps already read at the PCIe roofline and lazy mmaps commit pages only on fill. ``FREETOKEN_BANK_CUDA_ALLOC`` overrides."""
    env = _env_born_pinned()
    if env is not None:
        return env
    return False


class HostBank:
    """A page-aligned host buffer + its torch view, page-locked on demand: allocate -> fill -> ``pin()``/``lock()``.

    * ``"mmap"`` (default) -- lazy anonymous mmap; pages materialize on fill, then ``pin()`` registers or ``lock()`` OS-locks it.
    * ``"cuda"`` -- cudaHostAlloc, born pinned+mapped; ``pin()``/``lock()``/``release()`` are no-ops and it never takes LOCKED. See :func:`born_pinned_default`.

    The buffer is rounded up to the O_DIRECT block; ``tensor`` views exactly ``nbytes``. ``backing=None`` follows ``FREETOKEN_BANK_CUDA_ALLOC``."""

    __slots__ = ("tensor", "addr", "nbytes", "_buf", "_pinned", "_locked", "_backing", "_raw")

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype,
                 *, backing: str | None = None):
        if backing is None:
            plan = _requested_residency
            # a plan with non-pinned labels vetoes born-pinned: cudaHostAlloc spends the pin quota the plan exists to save
            born = _env_born_pinned() and (plan is None or not plan.has_unpinned)
            backing = "cuda" if born else "mmap"
        assert backing in ("mmap", "cuda"), backing
        self._backing = backing
        elsize = torch.empty((), dtype=dtype).element_size()
        self.nbytes = math.prod(shape) * elsize
        asize = ((self.nbytes + _BLK - 1) // _BLK) * _BLK
        if backing == "cuda":
            from freetoken.kernel.pinned import alloc_pinned_tensor

            # direct-IO readers need page alignment, but cudaHostAlloc only guarantees ~512 in practice
            # over-allocate one block and carve the aligned window; the numpy slice keeps the pinned storage alive via .base
            raw = alloc_pinned_tensor(asize + _BLK, dtype=torch.uint8)  # cudaMallocHost
            raw.zero_()  # keep the anonymous-mmap guarantee: unwritten regions stay zero
            off = (-raw.data_ptr()) % _BLK
            self._raw = raw
            self._buf = raw.numpy()[off:off + asize]
            self.addr = raw.data_ptr() + off
            assert self.addr % _BLK == 0
            self._pinned = True  # born pinned+mapped; pin() is a no-op
        else:
            self._raw = None
            self._buf = mmap.mmap(-1, asize)  # lazy: address space only, no resident pages yet
            _LIVE_BUFFERS.append(self._buf)
            self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
            self._pinned = False
        self.tensor = torch.frombuffer(self._buf, dtype=dtype, count=self.nbytes // elsize).view(*shape)
        self._locked = False

    @property
    def residency(self) -> HostResidency:
        if self._pinned:
            return HostResidency.PINNED
        if self._locked:
            return HostResidency.LOCKED
        return HostResidency.PAGEABLE

    @property
    def fill(self) -> torch.Tensor:
        """Where a loader writes this bank's rows.

        Identical to :attr:`tensor` here, and also for :class:`GpuOwnedBank` (whose tensor
        already lives on the device): the attribute exists so the assignment loops stay one
        code path whichever kind of bank a layer got."""
        return self.tensor

    def memoryview(self) -> memoryview:
        return memoryview(self._buf)

    def pin(self) -> None:
        """cudaHostRegister the (now-filled) buffer -- pin-after-fill.

        ``FREETOKEN_SKIP_BANK_PIN=1`` makes this a no-op for CPU-only tooling (the FTW converter); never set it when serving, the GPU paths need registered banks."""
        if self._pinned:
            return
        if os.environ.get("FREETOKEN_SKIP_BANK_PIN", "").strip().lower() in ("1", "true", "yes", "on"):
            return
        from freetoken.kernel.pinned import host_register

        try:
            host_register(self.addr, len(self._buf))
        except RuntimeError as exc:
            raise RuntimeError(
                f"cudaHostRegister failed for {len(self._buf) / 2**30:.1f} GiB"
            ) from exc
        self._pinned = True

    def release(self) -> None:
        """Drop the resident pages; the address space stays valid, the contents become undefined.

        For buffers that are done being read (the converter). No-op for born-pinned banks: registered pages cannot be dropped."""
        if self._pinned:
            return
        self._buf.madvise(mmap.MADV_DONTNEED)

    def free(self) -> None:
        """Drop references to the underlying buffer and tensors.

        For born-pinned (cudaHostAlloc) banks, dropping tensor, _buf, and the raw
        pinned tensor allows the PyTorch storage deleter to run cudaFreeHost.
        mmap-backed banks keep release() semantics (MADV_DONTNEED).
        """
        if getattr(self, "_backing", None) == "cuda":
            self.tensor = None
            self._buf = None
            self._raw = None
            self.addr = 0
            self._pinned = False
        else:
            self.release()

    def lock(self) -> None:
        """mlock the (now-filled) buffer: resident without CUDA pin quota, but no device address -- only the CPU executor can serve a locked layer.

        Lock after fill, or the lazy mmap faults+zero-fills every page. A failed lock (RLIMIT_MEMLOCK) warns once and leaves the bank PAGEABLE, which every consumer treats the same."""
        if self._locked or self._pinned:  # cudaHostRegister already page-locks
            return
        global _os_lock_failed
        if _os_lock_failed:
            return  # the quota is exhausted for good; skip the syscall spam
        try:
            _os_lock(self.addr, len(self._buf))
        except (OSError, ImportError) as exc:
            _os_lock_failed = True
            logger.warning(f"bank lock failed; leaving this and later banks pageable: {exc}")
            return
        self._locked = True


_os_locked_total = 0  # bytes locked so far; the OS lock ceiling is a per-process quota
_os_lock_failed = False  # sticky: once over quota, later (bigger-total) locks fail too


def _os_lock(addr: int, nbytes: int) -> None:
    global _os_locked_total
    import resource

    # grow the soft RLIMIT_MEMLOCK (defaults to a few MiB); the hard limit needs privilege, past it mlock fails below
    want = _os_locked_total + nbytes + (256 << 20)
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < want:
        new_soft = want if hard == resource.RLIM_INFINITY else min(want, hard)
        if new_soft > soft:
            try:
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (new_soft, hard))
            except (OSError, ValueError):
                pass  # keep the old limit; mlock below reports the real ceiling
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)):
        err = ctypes.get_errno()
        raise OSError(
            err,
            f"mlock({nbytes / 2**30:.1f} GiB): {os.strerror(err)} "
            f"(RLIMIT_MEMLOCK / `ulimit -l` caps OS-locked bytes; raise it or "
            f"shrink --moe-cpu-layers)",
        )
    _os_locked_total += nbytes


class GpuOwnedBank:
    """One bank kind of a GPU-owned MoE layer: the device tensor consumers read, which is
    also the tensor the loader writes.

    Duck-types the two attributes the loaders touch on a :class:`HostBank` -- ``tensor``
    (what the bank IS, here already on the device) and ``fill`` (where to write) -- so the
    placement loops stay one code path. They are the SAME tensor here: every
    ``fill[expert] = row`` is a synchronous pageable H2D copy issued by the placement
    thread itself. Six owned layers of Qwen3.8 are 7.9 GiB of such copies, a few seconds
    of boot, and that is the price of two properties nothing else gave us:

    * The engine loads weights inside ``torch.inference_mode()``, which is THREAD-LOCAL.
      The device banks are therefore inference tensors, and only the loading thread may
      write them. The first design staged through a pinned host layer and flushed it on
      the :class:`PinPipeline` drain thread, which is not in inference mode: every flush
      raised ``"Inplace update to inference tensor outside InferenceMode is not allowed"``.
    * The NVFP4 placement loop is single-threaded (the parallel reader parallelises the
      byte reads and yields to one consumer), so ANY bounded per-layer resource it has to
      wait for deadlocks as soon as the reader interleaves more owned layers than the
      bound: the only thread that could release a slot is the one that is blocked.

    Both were observed live on 2026-09-02
    (``docs/research/measurements-gpu-owned-layers-2026-09-02.md``). There are no host
    pages, so ``pin``/``lock``/``release`` have no meaning: the layer-completion sink
    settles nothing for this label (see :meth:`PinPipeline.__call__`) and :func:`_settle`
    refuses it outright.
    """

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    @property
    def fill(self) -> torch.Tensor:
        return self.tensor

    @property
    def residency(self) -> HostResidency:
        return HostResidency.GPU_OWNED


def alloc_banks(specs: dict[str, tuple[tuple[int, ...], torch.dtype]]) -> dict[str, HostBank]:
    """Allocate (lazy, unpinned) host banks from ``{name: (shape, dtype)}``."""
    return {name: HostBank(shape, dtype) for name, (shape, dtype) in specs.items()}


def alloc_layer_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    num_layers: int,
    *,
    gpu_owned: "frozenset[int] | None" = None,
    device: "torch.device | None" = None,
) -> dict[str, list]:
    """Allocate per-layer host banks: ``{name: ([num_experts, ...] row shape, dtype)}``
    -> one independently allocated (page-aligned, independently pin/lock-able)
    ``HostBank`` per layer per name.

    ``gpu_owned`` layer ids get NO host bank at all: their entry is a :class:`GpuOwnedBank`
    wrapping the ``[num_experts, ...]`` tensor on ``device``, which the loader fills in
    place (see that class for why there is no staging). The per-layer list keeps length
    ``num_layers`` and every entry still has ``size(0) == num_experts``, so downstream
    consumers are unchanged. ``None`` (the default) reads the set and the device from the
    ambient :func:`requested_residency` plan, so every provider honors
    ``--moe-gpu-owned-layers`` without a new parameter; pass ``frozenset()`` to force plain
    host banks.
    """
    if gpu_owned is None:
        gpu_owned, device = plan_gpu_owned()
    banks: dict[str, list] = {name: [] for name in specs}
    for layer_id in range(num_layers):
        for name, (shape, dtype) in specs.items():
            if layer_id in gpu_owned:
                banks[name].append(GpuOwnedBank(torch.empty(shape, dtype=dtype, device=device)))
            else:
                banks[name].append(HostBank(shape, dtype))
    return banks


class _ResidencyPlan:
    """Per-layer ``HostResidency`` labels, ambiently visible to the bank settle points.

    Installed by ``load_expert_banks`` around the provider dispatch so every loader honors --moe-cpu-layers without a new parameter in each signature. ``applied`` flips once a settle point consults the plan."""

    __slots__ = (
        "labels", "applied", "has_unpinned", "actual", "gpu_owned", "device", "expert_quant",
    )

    def __init__(self, labels: list[str], device=None, expert_quant: str | None = None):
        self.labels = list(labels)
        self.applied = False
        self.has_unpinned = any(r != HostResidency.PINNED.value for r in labels)
        self.actual: dict[int, str] = {}
        # GPU_OWNED layers are resolved once here so alloc_layer_banks can consult the plan
        # ambiently, exactly like pin_banks/PinPipeline consult it for LOCKED.
        self.gpu_owned = frozenset(
            i for i, r in enumerate(labels) if r == HostResidency.GPU_OWNED.value
        )
        self.device = device
        # The checkpoint's expert quant format, when the caller declared one. Only providers
        # reviewed for the owned-layer device-fill contract may write these tensors; every other
        # provider writes straight through .fill/.tensor since 8a63977 removed the staging
        # indirection, so a wrong-geometry load may silently appear to work.
        # ``None`` = undeclared (hand-built plans in tests and the shadow tooling), unchecked.
        self.expert_quant = expert_quant

    def residency_for(self, layer_id: int) -> str:
        self.applied = True
        return self.labels[layer_id]

    def record(self, layer_id: int, achieved: str) -> None:
        """One pageable bank downgrades the whole layer (a failed lock settles PAGEABLE)."""
        if self.actual.get(layer_id) != HostResidency.PAGEABLE.value:
            self.actual[layer_id] = achieved


_requested_residency: _ResidencyPlan | None = None


#: Expert quant formats whose providers are known to fill a GPU-owned layer's DEVICE banks
#: correctly. Everything else is refused before a device tensor exists -- see
#: :attr:`_ResidencyPlan.expert_quant`.
GPU_OWNED_EXPERT_QUANTS = frozenset({"nvfp4", "exl3"})


@contextlib.contextmanager
def requested_residency(labels: list[str] | None, device=None, expert_quant: str | None = None):
    """Install the ambient per-layer residency plan for the enclosed bank load (``None`` = no plan, everything pins).
    ``device`` is where GPU_OWNED layers' banks are allocated. ``expert_quant`` is the
    checkpoint's expert quant format, checked against :data:`GPU_OWNED_EXPERT_QUANTS` the
    first time a loader asks for the owned set."""
    global _requested_residency
    if labels is None:
        yield None
        return
    plan = _ResidencyPlan(labels, device, expert_quant)
    prev, _requested_residency = _requested_residency, plan
    try:
        yield plan
    finally:
        _requested_residency = prev


def plan_gpu_owned() -> "tuple[frozenset[int], torch.device | None]":
    """The ambient plan's GPU_OWNED layer ids and their target device (empty without a plan).

    Consulting the plan for owned layers counts as applying it (``_echo_residency`` keys its
    "this loader ignored the request" failure on that), but only when there is an owned set
    to honor -- a LOCKED-only plan is still only applied by a settle point.

    This is the narrowest gate every provider's owned-bank allocation passes through, so an
    unreviewed expert format is refused: since 8a63977 an owned bank IS its device tensor, so
    a provider writing through ``.fill``/``.tensor`` without a reviewed geometry may silently
    appear to work on a row layout the owned path was never checked against (status doc,
    residual risk 6).
    """
    plan = _requested_residency
    if plan is None:
        return frozenset(), None
    if plan.gpu_owned:
        if plan.expert_quant is not None and plan.expert_quant not in GPU_OWNED_EXPERT_QUANTS:
            raise ValueError(
                f"--moe-gpu-owned-layers needs a reviewed expert-bank format; this checkpoint's "
                f"expert quant format is {plan.expert_quant!r}. Only reviewed providers fill "
                f"owned-layer device banks in place (supported: {sorted(GPU_OWNED_EXPERT_QUANTS)}); "
                f"drop the flag for this model"
            )
        plan.applied = True
    return plan.gpu_owned, plan.device


def _settle(bank, residency: str) -> None:
    """Route a filled bank to its residency class (PAGEABLE = leave the plain mmap)."""
    if residency == HostResidency.GPU_OWNED.value:
        raise RuntimeError(
            "a GPU-owned MoE layer reached the host settle path: this checkpoint's bank "
            "loader has no per-layer completion sink, so its device banks would never be "
            "filled; drop --moe-gpu-owned-layers for this model"
        )
    if residency == HostResidency.DISK.value:
        raise RuntimeError(
            "a DISK MoE layer reached the host settle path: disk layers have no host banks"
        )
    if residency == HostResidency.PINNED.value:
        bank.pin()
    elif residency == HostResidency.LOCKED.value:
        bank.lock()


def pin_banks(banks: dict[str, HostBank | list[HostBank]]) -> None:
    """Settle every bank after it has been filled -- pin-after-fill by default.
    List-valued entries are per-layer and honor the ambient :func:`requested_residency` plan; scalar banks always pin."""
    plan = _requested_residency
    if plan is not None and plan.gpu_owned:
        # Refuse BEFORE settling anything: reaching this function at all means the loader has
        # no per-layer completion sink, so the owned layers' device banks would never be
        # filled. Checked here (not only in _settle) so the failure does not depend on which
        # layer happens to be settled first. _settle keeps the same guard.
        raise RuntimeError(
            "a GPU-owned MoE layer reached the host settle path: this checkpoint's bank "
            "loader has no per-layer completion sink, so its device banks would never be "
            "filled; drop --moe-gpu-owned-layers for this model"
        )
    for bank in banks.values():
        if isinstance(bank, list):
            for layer_id, layer_bank in enumerate(bank):
                residency = (
                    HostResidency.PINNED.value if plan is None
                    else plan.residency_for(layer_id)
                )
                _settle(layer_bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, layer_bank.residency.value)
        else:
            bank.pin()


class PinPipeline:
    """Settle (pin or lock) filled banks while other banks are still being read.

    cudaHostRegister is driver-serialized, so one background thread drains a queue and submitters never block: load time ~= max(read, settle).
    LOCKED banks mlock on the same thread (the quota bookkeeping in ``_os_lock`` is not thread-safe).
    A clean context-manager exit drains the queue and re-raises the first settle failure.
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._exc: BaseException | None = None
        # the current device is thread-local: a fresh thread sits on device 0 and cudaHostRegister would build its context there -- carry the creator's (bound) device into the worker
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self._device is not None:
            torch.cuda.set_device(self._device)
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._exc is not None:
                # Drain without settling after a failure. Safe to swallow the rest only
                # because NOTHING blocks on this thread: submitters never wait for a slot,
                # a token or an event, so a stored exception can strand no one and is
                # simply re-raised by wait()/__exit__. (It could, before GPU-owned layers
                # stopped staging through here; that made a settle failure a silent hang --
                # docs/research/measurements-gpu-owned-layers-2026-09-02.md.)
                continue
            try:
                bank, residency, plan, layer_id = item
                _settle(bank, residency)
                if plan is not None and residency == HostResidency.LOCKED.value:
                    plan.record(layer_id, bank.residency.value)
            except BaseException as exc:  # surfaced by wait()/__exit__
                self._exc = exc

    def submit(self, bank: HostBank, residency: str = HostResidency.PINNED.value,
               plan=None, layer_id: int | None = None) -> None:
        self._q.put((bank, residency, plan, layer_id))

    def __call__(self, layer_id: int, banks: dict[str, HostBank]) -> None:
        """Layer-completion sink: queue every bank of the completed layer at its ambient :func:`requested_residency` label.
        A GPU_OWNED layer is already finished when it gets here -- the loader filled its device tensor in place -- and it has no host pages, so there is nothing to settle and nothing to queue. Consulting the plan still marks it applied, and the tracker still counted the layer, so the loaders' ``placed`` asserts are unaffected."""
        plan = _requested_residency
        residency = (
            HostResidency.PINNED.value if plan is None else plan.residency_for(layer_id)
        )
        if residency == HostResidency.GPU_OWNED.value:
            return
        for bank in banks.values():
            self.submit(bank, residency, plan, layer_id)

    def _join(self) -> None:
        self._q.put(None)
        self._thread.join()

    def wait(self) -> None:
        self._join()
        if self._exc is not None:
            raise self._exc

    def __enter__(self) -> "PinPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._join()  # no thread leak; the in-flight exception wins
            return
        self.wait()


class LayerCompletionTracker:
    """Fire a sink once per layer, when all of that layer's writes have landed.

    ``note(layer_id)`` is called after each write; at ``expected_per_layer``
    notes the layer's banks are handed to ``on_layer(layer_id, {name: bank})``
    exactly once. Thread-safe (shard-driven loaders write layers from many
    threads in arbitrary order).
    """

    def __init__(
        self,
        expected_per_layer: int,
        banks: dict[str, list],
        on_layer,
    ) -> None:
        assert expected_per_layer > 0
        self._expected = expected_per_layer
        self._banks = banks
        self._on_layer = on_layer
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def note(self, layer_id: int) -> None:
        with self._lock:
            n = self._counts.get(layer_id, 0) + 1
            self._counts[layer_id] = n
            fire = n == self._expected
        if fire:
            self._on_layer(layer_id, {name: per[layer_id] for name, per in self._banks.items()})


def read_file_into(buf: memoryview | mmap.mmap, path: str, *, workers: int = 8,
                   chunk: int = _DEFAULT_CHUNK, drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of the whole file ``path`` into ``buf``
    (page-aligned). Returns the file size. The buffer must be >= the rounded-up file size.

    On Windows the same read runs through the ``FILE_FLAG_NO_BUFFERING`` reader
    (:mod:`freetoken.moe.win_io`), which bypasses the cache the same way; ``drop_cache``
    is then moot (nothing was cached to drop)."""
    if _win_unbuffered():
        return win_io.read_file_into(buf, path, workers=workers, chunk=chunk)
    size = os.path.getsize(path)
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    mv = buf if isinstance(buf, memoryview) else memoryview(buf)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    offs = list(range(0, size, chunk))

    def rd(o):
        want = min(chunk, len(mv) - o)
        want = min(want, ((size - o + _BLK - 1) // _BLK) * _BLK)
        os.preadv(fd, [mv[o:o + want]], o)

    try:
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return size


def _preadv_all(fd: int, dst: memoryview, offset: int, need: int) -> None:
    """preadv into ``dst`` until ``need`` bytes have landed; O_DIRECT may return a short count."""
    done = 0
    while done < need:
        if done % _BLK:  # a continuation read has to stay block-aligned on both sides
            raise OSError(f"unaligned short O_DIRECT read: {done} of {need} bytes at {offset}")
        got = os.preadv(fd, [dst[done:]], offset + done)
        if got <= 0:
            raise OSError(f"short O_DIRECT read: {done} of {need} bytes at {offset}")
        done += got


def read_range_into(buf: memoryview | mmap.mmap, path: str, *, file_offset: int, nbytes: int,
                    dest_offset: int = 0, workers: int = 8, chunk: int = _DEFAULT_CHUNK,
                    drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of ``path[file_offset : file_offset + nbytes]`` into ``buf`` at ``dest_offset``. Returns ``nbytes``.

    Byte-range counterpart of :func:`read_file_into`, for one tensor inside a shard. O_DIRECT needs the file offset AND the destination address block-aligned at the same time, which only holds when the two share their offset mod 4096 -- a safetensors data offset practically never lines up with the tensor's slot in the bank. Chunks that do line up DMA straight into ``buf``; the rest DMA into a page-aligned bounce (source window rounded out to whole blocks) and are copied into place, which also covers the unaligned head and tail.

    On Windows the same read (and the same bounce arithmetic, against the volume's sector
    size) runs through :mod:`freetoken.moe.win_io`.
    """
    if _win_unbuffered():
        return win_io.read_range_into(buf, path, file_offset=file_offset, nbytes=nbytes,
                                      dest_offset=dest_offset, workers=workers, chunk=chunk)
    mv = (buf if isinstance(buf, memoryview) else memoryview(buf)).cast("B")
    if dest_offset + nbytes > len(mv):
        raise ValueError(f"destination holds {len(mv)} bytes, need {dest_offset + nbytes}")
    base = ctypes.addressof(ctypes.c_char.from_buffer(mv))
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, file_offset, nbytes, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    scratch = threading.local()

    def rd(i: int) -> None:
        n = min(chunk, nbytes - i)
        src, dst = file_offset + i, dest_offset + i
        if src % _BLK == 0 and (base + dst) % _BLK == 0 and n % _BLK == 0:
            _preadv_all(fd, mv[dst:dst + n], src, n)
            return
        head = src % _BLK
        span = ((head + n + _BLK - 1) // _BLK) * _BLK
        bounce = getattr(scratch, "buf", None)
        if bounce is None or len(bounce) < span:
            bounce = scratch.buf = mmap.mmap(-1, span)  # anonymous mmaps are page-aligned
        bmv = memoryview(bounce)
        _preadv_all(fd, bmv[:span], src - head, head + n)
        mv[dst:dst + n] = bmv[head:head + n]

    try:
        offs = list(range(0, nbytes, chunk))
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return nbytes


__all__ = [
    "HostBank",
    "HostResidency",
    "LayerCompletionTracker",
    "PinPipeline",
    "alloc_banks",
    "alloc_layer_banks",
    "born_pinned_default",
    "pin_banks",
    "read_file_into",
    "read_range_into",
    "requested_residency",
]

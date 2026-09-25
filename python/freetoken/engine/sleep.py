"""Sleep and wake: give the graphics card back while the model stays in host RAM.

Design: docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md (section 3.2 is the sleep
order, section 3.3 the wake order). Everything here runs at the scheduler's idle safe point, as
rebuild_runtime_cache does, and composes primitives the memory governor has run live since
2026-09-07: ``Engine._move_layer`` (owned layers to and from the SSD expert copy),
``OffloadMoeCache.release_slots`` / ``rebuild``, the KV and GDN pool rebuilds, and graph
teardown and re-capture. Nothing here touches the host expert banks (about 53 GiB pinned), the
dense weights (phase 1 keeps them on the card) or the CUDA context.

Duck-typed on purpose: the functions take the engine and read only the attributes named here,
so tests drive them on the CPU ``FakeDiskEngine`` harness (tests/engine/test_engine_sleep.py).
"""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import dataclass, field

import torch

from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.utils import init_logger, mem_GB

logger = init_logger(__name__)

# Wake asks for the bytes sleep released plus this much before it allocates anything. The
# promote guard in rebuild_runtime_cache wants 256 MiB for one layer; a wake also re-creates
# the graph pools and (MTP on) the draft head's scratch, so it asks for twice that. Capped at
# what the card had free before the sleep (see _wake_margin).
WAKE_MARGIN_BYTES = 512 << 20

# Failed wakes in a row (each one went back to sleep cleanly) before the next is treated as an
# engine fault: three OOMs with the preflight passing each time means the free-bytes probe and
# the real allocations disagree, and a restart (the helper's watchdog on "failed") is the only
# thing that resets that; retrying forever would hold chats in a loop that never ends.
MAX_FAILED_WAKES = 3


class SleepRefused(CacheRebuildRejected):
    """Sleep or wake refused before anything was freed or allocated, or a failed wake that went
    back to sleep: the engine is in a known state and the scheduler answers "rejected"."""


class WakeFailed(RuntimeError):
    """A wake failed AND going back to sleep failed too: the engine state is unknown and the
    scheduler latches failed (the helper's watchdog then restarts the server)."""


@dataclass
class SleepSnapshot:
    moe_cache_size: int
    owned_layers: tuple[int, ...]
    ram_parked: tuple[int, ...]
    num_pages: int
    linear_slots: int | None  # physical slots, padding sink included
    graph_bs: list[int]
    had_spec_draft: bool
    free_before: int
    # The draft head's private KV width at sleep time. SpecDraftHead otherwise sizes it from
    # engine.max_seq_len, which after an idle sleep reflects the shrunk dynamic pool: a narrower
    # draft KV than the target's would take out-of-bounds writes on the first long chat.
    draft_num_pages: int | None = None
    draft_seed: int | None = None
    failed_wakes: int = 0
    free_after: int = 0
    slept_at: float = 0.0
    spilled_while_asleep: list[int] = field(default_factory=list)

    @property
    def released_bytes(self) -> int:
        return max(0, self.free_after - self.free_before)


def _disk_copy(engine):
    cache = engine.moe_offload_cache
    return getattr(engine, "expert_disk_copy", None) or (
        getattr(cache, "expert_disk_copy", None) if cache is not None else None
    )


def _sleep_kv_pages(engine) -> int:
    """The smallest pool the family allows (one page for every family but DSV4, whose floor is
    its window working set: kvcache/dsv4_paged_pool.py min_kv_tokens)."""
    config = engine.config
    min_tokens = int(engine._pool_cls.min_kv_tokens(config))
    return max(1, -(-min_tokens // int(config.page_size)))


def _sleep_linear_slots(engine) -> int:
    # Slot 0 is the padding sink; the MTP ladder takes one more slot in rebind().
    return 2 if getattr(engine, "spec_state_ladder", None) is not None else 1


def _wake_margin(snap: SleepSnapshot) -> int:
    """Headroom a wake asks for on top of the released bytes.

    Capped at what was free before the sleep: the boot's auto-sized KV pool (KVCacheTokens=0
    fills the card, memory project-262k-context-settings) can leave less than 512 MiB free, and
    that geometry served fine, so asking for more would refuse a wake with no game running at
    all. A wake that then runs short mid-way still goes back to sleep and answers "rejected"."""
    return min(WAKE_MARGIN_BYTES, max(0, int(snap.free_before)))


def _gb(n: int) -> str:
    return f"{n / (1 << 30):.1f} GB"


def _wake_refusal(free_now: int, need: int, card_free: int | None) -> str:
    """The wake preflight's refusal. cudaMemGetInfo (free_now) under WSL/WDDM does not see
    other processes' allocations: live 2026-09-25 it read 17.7 GB free while a 25 GB game
    allocation held the card and nvidia-smi showed 30714 of 32607 MiB used (about 1.9 GB
    free). The number shown is the smaller of it and NVML's card-wide free (what nvidia-smi
    reports); when NVML cannot be read, no number is shown rather than a wrong one."""
    tail = "; close the game or program using it, then try again"
    if card_free is None:
        return f"the graphics card does not have enough free memory to wake (waking needs {_gb(need)})" + tail
    return f"the graphics card has {_gb(min(free_now, card_free))} free and waking needs {_gb(need)}" + tail


def _card_free_bytes(engine) -> int | None:
    """Card-wide free bytes from NVML for the engine's device (every process counted), or
    None when it cannot be read (no CUDA device, no NVML library, a lookup error)."""
    device = getattr(engine, "device", None)
    if device is None or getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return None
    try:
        uuid = str(torch.cuda.get_device_properties(device).uuid)
    except Exception:  # noqa: BLE001
        return None
    return _nvml_free_bytes(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")


def _nvml_free_bytes(uuid: str) -> int | None:
    import ctypes
    import os

    names = ["nvml.dll"] if os.name == "nt" else ["libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1"]
    lib = None
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return None

    class _Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    try:
        if lib.nvmlInit_v2() != 0:
            return None
        try:
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByUUID(uuid.encode("ascii"), ctypes.byref(handle)) != 0:
                return None
            mem = _Memory()
            if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
                return None
            if not (0 < mem.total and mem.free <= mem.total):
                return None
            return int(mem.free)
        finally:
            lib.nvmlShutdown()
    except (OSError, AttributeError):
        return None


def _report(snap: SleepSnapshot | None, *, asleep: bool, elapsed: float, free: int, note=None) -> dict:
    return {
        "asleep": asleep,
        "released_bytes": snap.released_bytes if (snap is not None and asleep) else 0,
        "vram_free_bytes": int(free),
        "elapsed_s": round(float(elapsed), 2),
        "note": note,
    }


def check_can_sleep(engine) -> None:
    """Every refusal sleep can make, with nothing freed. The scheduler calls this before it
    parks conversations, so a refused sleep parks nothing either."""
    if getattr(engine, "mtp_shadow_observer", None) is not None:
        raise SleepRefused("sleep is not available while the MTP shadow observer runs (a diagnostic boot)")
    cache = engine.moe_offload_cache
    owned = sorted(engine._gpu_owned_layer_ids) if cache is not None else []
    if owned:
        disk = _disk_copy(engine)
        if disk is None:
            raise SleepRefused(
                f"expert layers {owned} live on the graphics card and no SSD expert copy is "
                "configured, so sleep has nowhere to put them"
            )
        missing = [l for l in owned if not disk.layer_complete(l)]
        if missing:
            raise SleepRefused(
                f"expert layers {missing} have no finished copy on the SSD yet (the first boot "
                "writes it in about four minutes), so sleep would have to keep them in PC memory"
            )


def sleep_engine(engine) -> dict:
    t0 = time.monotonic()
    if getattr(engine, "sleep_snapshot", None) is not None:
        return _report(engine.sleep_snapshot, asleep=True, elapsed=0.0,
                       free=engine.sleep_snapshot.free_after, note="already asleep")
    check_can_sleep(engine)
    cache = engine.moe_offload_cache
    pool = engine.linear_state_pool
    free_before = int(engine._sync_get_memory()[0])
    snap = SleepSnapshot(
        moe_cache_size=int(cache.cache_size) if cache is not None else 0,
        owned_layers=tuple(sorted(engine._gpu_owned_layer_ids)) if cache is not None else (),
        ram_parked=tuple(getattr(engine, "_ram_parked_layers", None) or ()),
        num_pages=int(engine.num_pages),
        linear_slots=int(pool.num_slots) if pool is not None else None,
        # The boot-resolved sizes (or the deferred copy while a layer is already on the SSD);
        # also parks them in _deferred_graph_bs for a later governor recall's re-capture.
        graph_bs=list(engine._graph_bs_for_recapture()),
        had_spec_draft=getattr(engine, "spec_draft", None) is not None,
        free_before=free_before,
    )
    draft = getattr(engine, "spec_draft", None)
    if draft is not None:
        snap.draft_num_pages = getattr(draft, "num_pages", None)
        snap.draft_seed = getattr(draft, "seed", None)
    # No local reference may outlive release_to_sleep: the head's weights and private KV
    # (2.17-2.56 GiB) are freed by refcount, and a live `draft` here would keep them through
    # the flush below, so released_bytes would undercount (PR #19 review).
    del draft
    # Point of no return for the scheduler's verdict: a failure from here is "after teardown".
    engine.rebuild_teardown_started = True
    release_to_sleep(engine, snap)
    snap.free_after = int(engine._sync_get_memory()[0])
    snap.slept_at = time.monotonic()
    engine.sleep_snapshot = snap
    elapsed = time.monotonic() - t0
    logger.info_rank0(
        f"Asleep in {elapsed:.1f} s: released {mem_GB(snap.released_bytes)} of VRAM, "
        f"{mem_GB(snap.free_after)} free on the card"
    )
    return _report(snap, asleep=True, elapsed=elapsed, free=snap.free_after)


def release_to_sleep(engine, snap: SleepSnapshot, *, force_pools: bool = False) -> None:
    """Free everything sleep frees. Idempotent, so a failed wake can call it to get back to a
    consistent asleep state from anywhere in the wake sequence.

    ``force_pools`` (the failed-wake path) rebuilds the KV and GDN pools even when their counts
    already read the sleep size: ``rebuild_from_config`` / ``LinearStatePool.rebuild`` free the
    old tensors first and set the count last, so an OOM inside one leaves a pool whose count
    says "1" and whose tensors are gone."""
    progress = engine._report_maintenance_progress
    # A marker, not a flag anyone decodes on: while it is set, _graph_bs_for_recapture reads
    # the parked boot sizes and the spec verify path skips its runner. _recapture_graphs
    # clears it (or re-marks it "deferred until no disk layers") on wake.
    engine._graphs_deferred = getattr(engine, "_graphs_deferred", None) or "asleep"
    # 1. Graphs first: they bake pool, page-table and slot addresses (rebuild step 1). The
    #    spec verify graphs before reset_capture, which drops the QSA verify metadata they
    #    replay against (same order as rebuild_runtime_cache).
    if getattr(engine, "spec_graph_runner", None) is not None:
        engine.spec_graph_runner.destroy()
        engine.spec_graph_runner = None
    engine.attn_backend.reset_capture()
    engine.graph_runner.destroy_cuda_graphs()
    progress("sleep:graphs")
    # 2. The MTP draft head (2.17-2.56 GiB resident, measured 2026-09-02 / 09-07): no host copy
    #    exists, so it is dropped and wake rebuilds it with the boot constructor. close() frees
    #    its own graphs and expert runner; the weights go with the last reference.
    draft = getattr(engine, "spec_draft", None)
    if draft is not None:
        draft.close()
        engine.spec_draft = None
        progress("sleep:draft")
    del draft  # the last reference: gc.collect / empty_cache below must be able to free it
    # 3. Owned (and RAM-parked) layers go to the SSD copy, never to pinned RAM: 1.32 GiB of
    #    Windows RAM per layer is not there to spend (control-panel acceptance 2026-09-25:
    #    Windows free fell to 2.7 GB during a boot). _move_layer suspends prefill overlap on
    #    the first disk layer and drops the layer from _ram_parked_layers.
    cache = engine.moe_offload_cache
    if cache is not None:
        # Prefill overlap off BEFORE any layer goes to the SSD or the slots go: its double
        # buffers alias slots [0, 2E), and rebind_layer refuses a DISK layer while it is on
        # ("prefill overlap DMAs from registered banks"). release_slots alone leaves the flag
        # set, which an asleep SSD spill with no owned layers would then trip over. Marked
        # suspended with _move_layer's own flag, so the wake (or the governor's recall of the
        # last disk layer) turns it back on through _resume_prefill_overlap_if_clear.
        if cache.prefill_overlap:
            cache.set_prefill_overlap(False)
            engine._prefill_overlap_suspended = True
        for layer_id in snap.owned_layers:
            if cache.layer_residency[layer_id] == "gpu_owned":
                engine._move_layer(layer_id, "disk")
                progress("sleep:layer", f"layer {layer_id} -> SSD")
        # 4. The shared slot cache, after the moves (a disk rebind reads the cache geometry).
        #    EXL3's packed pointer tables hold references to the slot tensors: drop them first
        #    or release_slots frees nothing (same as _resize_pools, engine.py; the next eager
        #    call after the wake rebuilds them for the new bank addresses).
        scratch = getattr(cache, "exl3_scratch", None)
        tables = getattr(scratch, "mgemm_tables", None)
        if tables is not None:
            tables.clear()
        if cache.cache_size or force_pools:
            cache.release_slots()
            progress("sleep:slots")
    # 5. KV to the family's minimum, GDN to its padding slot (+ the ladder's). The scheduler has
    #    already parked every eligible conversation (cache_manager.prepare_rebuild) and
    #    re-threads its page managers after this returns.
    pages = _sleep_kv_pages(engine)
    if force_pools or engine.num_pages != pages:
        engine._resize_kv_pool(engine.config, pages, None)
    pool = engine.linear_state_pool
    if pool is not None:
        slots = _sleep_linear_slots(engine)
        if force_pools or pool.num_slots != slots:
            pool.rebuild(slots)
            if getattr(engine, "spec_state_ladder", None) is not None:
                # rebuild reset the free list and replaced the state tensors: the ladder takes
                # a fresh snapshot slot and drops its recorded replays.
                engine.spec_state_ladder.rebind()
    engine._refresh_seq_state(engine.config)
    progress("sleep:pools")
    gc.collect()
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None and engine.device.type == "cuda" and torch.cuda.is_available():
        clear()


def wake_engine(engine) -> dict:
    t0 = time.monotonic()
    snap = getattr(engine, "sleep_snapshot", None)
    if snap is None:
        return _report(None, asleep=False, elapsed=0.0, free=int(engine._sync_get_memory()[0]),
                       note="already awake")
    # Preflight, nothing allocated yet (review focus 1): a game holding the card gets a plain
    # refusal and the model stays exactly as it slept.
    free_now = int(engine._sync_get_memory()[0])
    need = snap.released_bytes + _wake_margin(snap)
    # The gate takes NVML's card-wide figure too when it can read it: cudaMemGetInfo under
    # WSL misses other processes (see _wake_refusal), and a wake it waved through would OOM,
    # go back to sleep and count toward MAX_FAILED_WAKES, so a game could force a restart.
    card_free = _card_free_bytes(engine)
    if min(free_now, card_free if card_free is not None else free_now) < need:
        raise SleepRefused(_wake_refusal(free_now, need, card_free))
    engine.rebuild_teardown_started = True
    try:
        _restore(engine, snap)
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            # Not a shortage the game can fix by quitting: a CUDA fault, a kernel error or a
            # bug. The engine state is unknown, so no way back is attempted; the scheduler
            # latches failed and the helper's watchdog restarts the server.
            raise WakeFailed(f"wake failed ({exc!r}); not an out-of-memory, restart needed") from exc
        logger.error(f"wake failed ({exc!r}); putting the model back to sleep")
        # The traceback's frames (SpecDraftHead / GraphRunner constructors, a layer move) still
        # hold whatever they allocated before the OOM; clear them or the flush below cannot
        # hand those bytes back (PR #19 review).
        _drop_frames(exc)
        try:
            release_to_sleep(engine, snap, force_pools=True)
            engine._sync_get_memory()  # empty_cache: hand the partial allocations back
        except Exception as exc2:  # noqa: BLE001
            raise WakeFailed(
                f"wake failed ({exc!r}) and going back to sleep failed too ({exc2!r})"
            ) from exc2
        snap.failed_wakes += 1
        if snap.failed_wakes >= MAX_FAILED_WAKES:
            raise WakeFailed(
                f"{snap.failed_wakes} wakes in a row ran out of memory after the free-memory "
                f"check passed (last: {exc!r}); restart needed"
            ) from exc
        raise SleepRefused(f"wake failed and the model went back to sleep: {exc!r}") from exc
    engine.sleep_snapshot = None
    engine.snapshot_pool_budget()
    free_after = int(engine._sync_get_memory()[0])
    elapsed = time.monotonic() - t0
    logger.info_rank0(f"Awake in {elapsed:.1f} s, {mem_GB(free_after)} free on the card")
    return _report(None, asleep=False, elapsed=elapsed, free=free_after)


def _drop_frames(exc: BaseException) -> None:
    """Clear the locals of every finished frame in ``exc``'s traceback (and its chained
    causes) and detach the tracebacks, so tensors a failing constructor held become garbage.
    The message and type survive for the SleepRefused / WakeFailed chain."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        tb = exc.__traceback__
        if tb is not None:
            traceback.clear_frames(tb)  # skips the still-executing frame (wake_engine)
            exc.__traceback__ = None
        exc = exc.__cause__ or exc.__context__


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _restore(engine, snap: SleepSnapshot) -> None:
    progress = engine._report_maintenance_progress
    config = engine.config
    # 1. GDN and KV back to their pre-sleep sizes (the draft head below gets its private KV
    #    width from the snapshot, not from engine.max_seq_len; _refresh_seq_state still runs
    #    first so the page table and max_seq_len match the restored pool).
    pool = engine.linear_state_pool
    if pool is not None and snap.linear_slots is not None and pool.num_slots != snap.linear_slots:
        pool.rebuild(snap.linear_slots)
        if getattr(engine, "spec_state_ladder", None) is not None:
            engine.spec_state_ladder.rebind()
    if engine.num_pages != snap.num_pages:
        engine._resize_kv_pool(config, snap.num_pages, None)
    engine._refresh_seq_state(config)
    progress("wake:pools")
    cache = engine.moe_offload_cache
    if cache is not None:
        # 2. Slot cache BEFORE the layers come home: resuming prefill overlap (inside
        #    _move_layer, _resume_prefill_overlap_if_clear) needs cache_size >= 2 * num_experts
        #    and live banks.
        if cache.cache_size != snap.moe_cache_size:
            cache.rebuild(snap.moe_cache_size)
            progress("wake:slots")
        # 3. Owned layers SSD -> card. A layer the governor spilled WHILE asleep was pinned,
        #    not owned, so it stays on the SSD until the governor recalls it (design D8).
        for layer_id in snap.owned_layers:
            if cache.layer_residency[layer_id] == "disk":
                engine._move_layer(layer_id, "gpu_owned")
                progress("wake:layer", f"layer {layer_id} -> card")
        # _move_layer dropped each layer from the RAM-parked list on its way to the SSD.
        engine._ram_parked_layers = [l for l in snap.ram_parked if cache.is_gpu_owned_layer(l)]
        # Overlap back on now the slots are back (a no-op while a layer spilled during sleep
        # is still on the SSD: the governor's recall resumes it then, as after a live spill).
        # Explicit because a sleep with no owned layers makes no _move_layer call here.
        from .engine import _resume_prefill_overlap_if_clear

        _resume_prefill_overlap_if_clear(engine)
    # 4. Graphs, with the boot-resolved sizes (deferred while any layer is on the SSD).
    gc.collect()
    free_min = engine._sync_get_memory()[0]
    progress("wake:capture")
    engine._recapture_graphs(config, snap.graph_bs, free_min)
    # 5. MTP: the draft head, then the verify/draft/ladder graphs captured as at boot.
    spec = getattr(config, "spec_decode", None)
    if snap.had_spec_draft and getattr(engine, "spec_draft", None) is None:
        from .spec_draft import SpecDraftHead

        engine.spec_draft = SpecDraftHead(engine, spec, seed=snap.draft_seed,
                                          num_pages=snap.draft_num_pages)
        progress("wake:draft")
    engine._rearm_spec_graphs()
    wants_spec_capture = spec is not None and (
        getattr(spec, "enabled", False) or getattr(spec, "graph_widths", ())
    )
    if wants_spec_capture and not getattr(engine, "_graphs_deferred", None):
        # Skipped while a layer spilled during sleep keeps the graphs deferred: the verify path
        # does not replay graphs then (speculative_decode_batch), a disk layer's gather needs a
        # host sync no capture can hold, and a live spill's rebuild only re-arms too. The
        # widths stay lazily capturable after the governor's recall.
        engine._capture_spec_graphs_at_boot()
    progress("wake:captured")
    # 6. Self-check: two short prefills through the fresh pools and kernels, so a sticky CUDA
    #    fault surfaces as a failed wake, not inside Jay's next chat. Harmless to the KV: the
    #    pool is freshly zeroed and parked prefixes restore after the wake, on their next turn.
    # The boot's own lengths (backend- and format-dependent, incl. _SMALL_PREFILL_ROWS): nothing
    # a wake runs here was not compiled and proven at boot. None at boot means none here.
    lengths = engine._boot_warmup_lengths(config)
    if lengths:
        engine._warmup_prefill(lengths=lengths)
    progress("wake:checked")


def asleep_rebuild(
    engine, *, moe_cache_size=None, num_pages=None, num_mamba_slots=None, num_swa_pages=None,
    layer_moves=None,
) -> None:
    """The ``rebuild`` step_memory runs while the model sleeps: SSD spills and nothing else.

    With the slot cache empty the RAM ladder's park-on-card rung cannot fire (it needs
    num_experts slots to give up), so a RAM-down step reaches the SSD rung and lands here. A
    game squeezing Windows memory then still gets expert layers out of the way; the governor's
    recall brings them back after the wake (design D8). No graph work: there are none asleep,
    and wake's re-capture sees the disk layer and defers them as a live spill does."""
    snap = getattr(engine, "sleep_snapshot", None)
    if snap is None:
        raise SleepRefused("not asleep: governor steps use the normal rebuild while awake")
    if any(v is not None for v in (moe_cache_size, num_pages, num_mamba_slots, num_swa_pages)):
        raise SleepRefused("asleep: only expert-layer spills to the SSD run while the model sleeps")
    disk = _disk_copy(engine)
    moves = list(layer_moves or ())
    # Validate every move before the first one runs, as rebuild_runtime_cache's step 0a does.
    for layer_id, target in moves:
        if target != "disk":
            raise SleepRefused(f"asleep: layer {layer_id} cannot move to {target} until the model wakes")
        if disk is None or not disk.layer_complete(layer_id):
            raise SleepRefused(f"layer {layer_id} has no finished SSD copy")
    for layer_id, _target in moves:
        if engine.moe_offload_cache.layer_residency[layer_id] == "disk":
            continue  # already there: nothing moved, nothing for the wake to remember
        _spill_or_roll_back(engine, layer_id)
        snap.spilled_while_asleep.append(layer_id)
        engine._report_maintenance_progress("asleep:spill", f"layer {layer_id} -> SSD")


def _spill_or_roll_back(engine, layer_id: int) -> None:
    """One asleep pinned -> disk move that either lands or leaves the layer as it was.

    rebind_layer switches the residency to ``disk`` before it builds the layer's reader (the
    SSD staging buffer is allocated lazily there), so a failed staging allocation used to leave
    a half-moved layer: residency ``disk``, no reader, host bank still allocated. Later spills
    skipped it as "already there" and the wake could not bring it back. Codex round 2 on PR #19.
    Here the layer is rebound to its old residency and banks and the step reports rejected; if
    even that fails the engine is torn (rebuild_teardown_started) and the step latches failed,
    so the watchdog restarts the server instead of serving on a broken layer."""
    cache = engine.moe_offload_cache
    before = cache.layer_residency[layer_id]
    banks = {n: cache.bank_sources[n][layer_id] for n in cache.bank_schema}
    try:
        engine._move_layer(layer_id, "disk")
    except Exception as exc:
        if cache.layer_residency[layer_id] == before:
            raise  # nothing moved: the old layer is intact
        try:
            cache.rebind_layer(layer_id, before, banks)
            engine._gpu_owned_layer_ids = cache.gpu_owned_layer_ids
            engine._stash_vram_ledger_inputs(cache.bank_sources, engine._gpu_owned_layer_ids)
            spilled = getattr(engine, "_ram_spilled_layers", None)
            if spilled and layer_id in spilled:
                spilled.remove(layer_id)
        except Exception:
            engine.rebuild_teardown_started = True
            raise exc from None
        raise SleepRefused(
            f"asleep: layer {layer_id} could not spill to the SSD ({exc!r}); left {before}"
        ) from exc

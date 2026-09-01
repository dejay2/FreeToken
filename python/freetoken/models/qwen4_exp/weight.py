"""Qwen3.8-Flash-Next (RadixArk NVFP4) checkpoint reader.

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_FUSIONS``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`load_nvfp4_expert_sources` -- the routed NVFP4 experts, into the offload cache's source banks.

Dropped unconditionally: ``mtp.*`` (speculative head, including its stacked
``mtp.layers.0.mlp.experts.*``). ``model.visual.*`` is retained only when picture loading
is explicitly enabled.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import ctypes
import json
import mmap
import os
import re
import struct
import threading
from dataclasses import dataclass
from typing import BinaryIO, Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import vision_execution_mode, vision_load_enabled
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import download_hf_weight, init_logger
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

logger = init_logger(__name__)

# Routed NVFP4 experts (nvidia modelopt layout): per-expert, un-fused. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP head's
# stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# Fused projections: concat the checkpoint parts along dim 0 in this exact order. A nonzero pad
# rounds the merged row count up; the model splits the result back with the same sizes.
_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    # q carries the output gate, so its half is twice the attention width: [2*qo | kv | kv].
    ".self_attn.qkv_proj.weight": ((
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ), 0),
    ".linear_attn.in_proj.weight": ((
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ), 0),
    ".mlp.shared_expert.gate_up_proj.weight": ((
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ), 0),
    # HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM
    # pads the merged output to a multiple of 16 rows for cuBLAS (hyperconnection.py pad_size).
    # The top-level hyper_connection_mixer has no injection and so never fuses.
    ".attn_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".attn_hyper_connection.input_mix_weight_down.weight",
        ".attn_hyper_connection.block_inject_weight.weight",
    ), 16),
    ".mlp_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".mlp_hyper_connection.input_mix_weight_down.weight",
        ".mlp_hyper_connection.block_inject_weight.weight",
    ), 16),
}


def _rename(raw_name: str, *, include_vision: bool = False) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    if raw_name.startswith("model.visual."):
        return "visual." + raw_name[len("model.visual.") :] if include_vision else None
    if raw_name.startswith("visual."):
        return raw_name if include_vision else None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; return the merged ``(name, tensor)`` once all parts arrive, ``()`` while incomplete, ``None`` if ``name`` is not a fusion part."""
    for fused_suffix, (parts, pad_to) in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if not name.endswith(part):
                continue
            key = name[: -len(part)] + fused_suffix
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(parts):
                return ()
            del buf[key]
            rows = [slots[i] for i in range(len(parts))]
            pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
            if pad:
                rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
            return key, torch.cat(rows, dim=0)
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the
    model's state dict minus the routed experts. Nothing here is quantized: the modelopt
    ``ignore`` list covers everything except those experts, so attention, GDN, HC, PLE, the shared
    expert and lm_head are all plain bf16 (the n-gram hash constants stay int64). Fusions:
    attention q|k|v -> ``qkv_proj``, GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, shared-expert
    gate|up -> ``gate_up_proj``, and each per-layer HC's ``input_mix_weight_down`` |
    ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.

    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the
    routed experts are NVFP4 and always come from :func:`load_nvfp4_expert_sources`.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    include_vision = vision_load_enabled()
    stream_vision = include_vision and vision_execution_mode() == "layer-stream"
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with ExitStack() as stack:
            engine_file = stack.enter_context(
                safetensors.safe_open(file, framework="pt", device=str(device))
            )
            raw_names = engine_file.keys()
            has_vision = stream_vision and any(
                raw.startswith(("model.visual.", "visual.")) for raw in raw_names
            )
            vision_file = engine_file
            if has_vision and device.type != "cpu":
                # Read picture tensors directly into their persistent CPU home. Loading
                # them on CUDA first would leave allocator reservations behind and make
                # automatic expert sizing undercount the memory this mode is meant to free.
                vision_file = stack.enter_context(
                    safetensors.safe_open(file, framework="pt", device="cpu")
                )
            for raw_name in raw_names:
                name = _rename(raw_name, include_vision=include_vision)
                if name is None:
                    continue
                source = vision_file if name.startswith("visual.") else engine_file
                tensor = source.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused
                    continue
                yield name, tensor

    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor FP8 scale."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` float8_e4m3fn view of the bank."""
        return self.bank.tensor


@dataclass(frozen=True)
class PleShard:
    """A PLE byte range in a safetensor file."""

    path: str
    offset: int
    nbytes: int
    rows: int
    cols: int


@dataclass(frozen=True)
class PleLayout:
    """Validated on-disk PLE layout."""

    shards: tuple[PleShard, ...]
    weight_scale: torch.Tensor
    rows_per_shard: int
    head_dim: int


# A gather splits into _PLE_GATHER_WORKERS chunks, one per NVMe queue slot. Measured on this
# box (47.7 GiB table, uniformly random ids) 4 is the peak at every batch size -- 1/2/4/8/16
# chunks cost 8.8/4.9/3.3/3.6/4.2 ms for 64 rows -- past 4 the fault path stops scaling and
# the extra dispatch is pure loss. This is the fallback: where PrefetchVirtualMemory is
# available (see _prefetch_rows) it is both faster and simpler than the fan-out.
_PLE_GATHER_WORKERS = 4
_PLE_MIN_GATHER_CHUNK = 4  # below this the pool round-trip costs more than the faults it hides
_PLE_ROW_CACHE_ENV = "FREETOKEN_PLE_ROW_CACHE"
# 1 Mi rows = 160 MiB of slab, +8 MiB for the slot->id list, +~110 MiB for the id->slot dict
# at full occupancy (1 M boxed int keys and their slots). This box has ~8 GiB free with the
# server up, so ~280 MiB buys 16x the old residency for ~3.5% of the headroom; the slab is
# allocated up front, so shrink FREETOKEN_PLE_ROW_CACHE on tighter hosts.
_PLE_ROW_CACHE_DEFAULT = 1_048_576
# Below two rows the PrefetchVirtualMemory syscall costs more than the fault it hides
# (measured: 1 row 0.15 ms plain vs 0.17 ms prefetched; 2 rows 0.29 vs 0.20).
_PLE_MIN_PREFETCH_ROWS = 2

_gather_pool: ThreadPoolExecutor | None = None
_gather_pool_lock = threading.Lock()


def _resolve_prefetch_virtual_memory():
    """``(PrefetchVirtualMemory, GetCurrentProcess())``, or ``(None, None)`` off Windows.

    Resolved once at import: ``ctypes.wintypes`` does not exist on POSIX and the symbol is
    Win8+/Server-2012+, so either step may raise. The entries pointer is typed ``c_void_p``
    because the array is built as a torch tensor, not a ctypes array -- see _range_entries.
    """
    try:
        import ctypes.wintypes as wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        prefetch = kernel32.PrefetchVirtualMemory
        prefetch.argtypes = [
            wintypes.HANDLE,     # hProcess
            ctypes.c_size_t,     # NumberOfEntries (ULONG_PTR)
            ctypes.c_void_p,     # PWIN32_MEMORY_RANGE_ENTRY
            wintypes.DWORD,      # Flags (reserved, must be 0)
        ]
        prefetch.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p  # else the -1 pseudo-handle
        return prefetch, kernel32.GetCurrentProcess()         # comes back as a signed int
    except Exception:  # non-Windows, or a kernel too old to export the symbol
        return None, None


_prefetch_virtual_memory, _current_process = _resolve_prefetch_virtual_memory()
_prefetch_failed = False  # set once if the call ever returns FALSE; then never retried


def _ple_gather_pool() -> ThreadPoolExecutor:
    """The process-wide pool that fans random row faults out over the NVMe queue."""
    global _gather_pool
    with _gather_pool_lock:
        if _gather_pool is None:
            _gather_pool = ThreadPoolExecutor(
                max_workers=_PLE_GATHER_WORKERS, thread_name_prefix="ple-gather"
            )
        return _gather_pool


class _PleRowCache:
    """Bounded cache of table rows: one ``uint8 [capacity, head_dim]`` slab, a dict id -> slot
    and FIFO (round-robin) slot reuse.

    The dict-of-cloned-row-tensors this replaces paid a tensor allocation per cached row and
    a Python-level ``out[i].copy_(row)`` per hit. Here both directions are one vectorized
    ``index_copy_``, which on this box costs (ms per gather, all rows resident):

        rows      OrderedDict   slab (loop)   slab (index_copy)
          16          0.027        0.042           0.009
          96          0.162        0.249           0.022
        1120          1.982        3.197           0.238

    and on the insert side 0.062/0.310/3.49 ms -> 0.021/0.040/0.636 ms. The per-row loop over
    the slab is the *slowest* of the three (every hit builds a row view), so the batched
    index_copy_ is what makes the slab win, not the slab itself.

    FIFO, not LRU: recency bookkeeping is what made the OrderedDict path expensive, and the
    n-gram ids are near-uniform over 320 M rows, so a hit is luck rather than recency.

    Only the thread coordinating a gather touches it -- hits are served before the fault and
    insertions happen after it -- so it needs no lock.
    """

    def __init__(self, capacity: int, head_dim: int) -> None:
        self.capacity = capacity
        self._slab = torch.empty((capacity, head_dim), dtype=torch.uint8)
        self._slot_of: dict[int, int] = {}
        # slot -> row id (-1 free). A plain list, not a tensor/array: it is read once per
        # inserted row and ``int(tensor[i])`` is ~50x a list index. It holds no int objects
        # of its own -- the ids are the very objects ``_slot_of`` already keys on.
        self._id_at: list[int] = [-1] * capacity
        self._next = 0  # FIFO hand

    def take_hits(
        self, valid_ids: list[int], positions: list[int], out: torch.Tensor
    ) -> tuple[list[int], list[int]]:
        """Copy cached rows into ``out``; return the (ids, positions) still to be faulted."""
        slot_of = self._slot_of.get
        hit_slots: list[int] = []
        hit_positions: list[int] = []
        miss_ids: list[int] = []
        miss_positions: list[int] = []
        for row_id, position in zip(valid_ids, positions):
            slot = slot_of(row_id, -1)
            if slot < 0:
                miss_ids.append(row_id)
                miss_positions.append(position)
            else:
                hit_slots.append(slot)
                hit_positions.append(position)
        if hit_slots:
            out.index_copy_(
                0,
                torch.tensor(hit_positions, dtype=torch.int64),
                self._slab.index_select(0, torch.tensor(hit_slots, dtype=torch.int64)),
            )
        return miss_ids, miss_positions

    def insert(self, row_ids: list[int], positions: list[int], out: torch.Tensor) -> None:
        """Copy the freshly faulted rows out of ``out`` into the slab (never an mmap view)."""
        slot_of = self._slot_of
        id_at = self._id_at
        hand = self._next
        capacity = self.capacity
        writes: dict[int, int] = {}  # slot -> out position; a dict so a slot reused twice in
        for row_id, position in zip(row_ids, positions):  # one batch keeps only its last row
            if row_id in slot_of:
                continue  # duplicate id within this batch
            slot = hand
            hand += 1
            if hand == capacity:
                hand = 0
            evicted = id_at[slot]
            if evicted >= 0:
                del slot_of[evicted]
            id_at[slot] = row_id
            slot_of[row_id] = slot
            writes[slot] = position
        self._next = hand
        if writes:
            self._slab.index_copy_(
                0,
                torch.tensor(list(writes), dtype=torch.int64),
                out.index_select(0, torch.tensor(list(writes.values()), dtype=torch.int64)),
            )


class MmapPleStorage:
    """Memory-map PLE safetensor ranges and gather selected rows."""

    def __init__(self, layout: PleLayout) -> None:
        self.rows_per_shard = layout.rows_per_shard
        self.head_dim = layout.head_dim
        self.num_rows = len(layout.shards) * self.rows_per_shard
        self.nbytes = sum(shard.nbytes for shard in layout.shards)
        self._files: dict[str, BinaryIO] = {}
        self._maps: dict[str, mmap.mmap] = {}
        self._shards: list[torch.Tensor] = []
        capacity = int(os.environ.get(_PLE_ROW_CACHE_ENV, _PLE_ROW_CACHE_DEFAULT))
        self._row_cache = _PleRowCache(capacity, self.head_dim) if capacity > 0 else None
        try:
            for shard in layout.shards:
                mapping = self._maps.get(shard.path)
                if mapping is None:
                    fh = open(shard.path, "rb")
                    mapping = mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_COPY)
                    if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                        mapping.madvise(mmap.MADV_RANDOM)
                    self._files[shard.path] = fh
                    self._maps[shard.path] = mapping
                tensor = torch.frombuffer(
                    mapping,
                    dtype=torch.uint8,
                    count=shard.nbytes,
                    offset=shard.offset,
                ).reshape(shard.rows, shard.cols)
                self._shards.append(tensor)
        except Exception:
            self.close()
            raise
        # Virtual address of each shard's row 0, for the prefetch range array. int64, not
        # uint64: Windows user-mode addresses live below 0x8000_0000_0000, so the signed
        # arithmetic torch actually supports cannot overflow.
        self._shard_bases = torch.tensor(
            [shard.data_ptr() for shard in self._shards], dtype=torch.int64
        )

    def gather(self, row_ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Copy valid rows into a CPU buffer; invalid IDs produce zeros."""
        ids = row_ids.reshape(-1)
        if ids.device.type != "cpu" or ids.dtype != torch.int64:
            raise ValueError("mmap PLE row ids must be a CPU int64 tensor")
        expected = (ids.numel(), self.head_dim)
        if out.device.type != "cpu" or out.dtype != torch.uint8 or tuple(out.shape) != expected:
            raise ValueError(
                f"mmap PLE output must be CPU uint8 {expected}, got {out.device} "
                f"{out.dtype} {tuple(out.shape)}"
            )
        out.zero_()
        if ids.numel() == 0:
            return out

        valid = (ids >= 0) & (ids < self.num_rows)
        positions = valid.nonzero().reshape(-1)
        if positions.numel() == 0:
            return out
        valid_ids = ids.index_select(0, positions)

        cache = self._row_cache
        if cache is None:
            self._fault(valid_ids, positions, out)
            return out
        miss_ids, miss_positions = cache.take_hits(
            valid_ids.tolist(), positions.tolist(), out
        )
        if not miss_ids:
            return out
        self._fault(
            torch.tensor(miss_ids, dtype=torch.int64),
            torch.tensor(miss_positions, dtype=torch.int64),
            out,
        )
        cache.insert(miss_ids, miss_positions, out)
        return out

    def _range_entries(self, valid_ids: torch.Tensor) -> torch.Tensor:
        """``int64 [n, 2]`` laid out exactly as ``WIN32_MEMORY_RANGE_ENTRY[n]``.

        The struct is ``{PVOID VirtualAddress; SIZE_T NumberOfBytes;}`` -- two pointer-sized
        fields, 16 B per entry on x64 -- so a contiguous ``[n, 2]`` int64 tensor is the array,
        and its ``data_ptr()`` is the pointer the API wants. Built with vectorized address
        math rather than a Python loop over a ``(_Range * n)()``: measured 0.011/0.018/0.057/
        0.26 ms here at 16/1120/11200/112 000 entries against 0.003/0.27/2.2/29.0 ms for the
        loop, which at prefill sizes would have eaten most of the win.

        No coalescing of adjacent rows: the ids are near-uniform over 320 M rows, so runs are
        vanishingly rare, and the API is happy to be handed 10 rows of the same 4 KiB page.
        """
        shard_ids = torch.div(valid_ids, self.rows_per_shard, rounding_mode="floor")
        local_ids = valid_ids - shard_ids * self.rows_per_shard
        entries = torch.empty((valid_ids.numel(), 2), dtype=torch.int64)
        torch.add(
            self._shard_bases.index_select(0, shard_ids),
            local_ids,
            alpha=self.head_dim,
            out=entries[:, 0],
        )
        entries[:, 1] = self.head_dim
        return entries

    def _prefetch_rows(self, valid_ids: torch.Tensor) -> bool:
        """Ask Windows to fault every row in one syscall; False if that is not available.

        One PrefetchVirtualMemory over the whole miss set, then a plain serial copy, beats the
        4-worker fan-out at every batch size measured on this box (47.7 GiB table, uniformly
        random ids, cold, row cache off; ms per gather, mean of 40-60 batches at 16/96/1120
        and 6-12 at 11200/112000):

            rows     pool fan-out   prefetch+serial   prefetch+pool   of which the syscall
              16          0.90            0.44            0.78              0.18
              96          3.92            1.53            3.39              0.68
            1120         37.2            10.9            29.7               7.8
           11200        275              79              83                68
          112000       1691             731             721               642

        So: serial copy, because it wins by 2-3x up to 1120 rows and only ties (within the
        ~10% run-to-run spread) at 11200 and above -- fanning out after the prefetch buys
        nothing once the pages are already resident, it just adds dispatch.

        And a single call, not several: chunking the entries into 16 384-entry calls measured
        0.41/1.53/12.0/84/808 ms against 0.44/1.54/10.9/86/731 for one call -- indistinguishable
        everywhere, so take the simpler shape. The array itself is one allocation either way.

        A FALSE return disables the path for the process: the documented failure is a bad
        parameter or a working-set/quota refusal, neither of which a retry fixes.
        """
        global _prefetch_failed
        if _prefetch_virtual_memory is None or _prefetch_failed:
            return False
        n = valid_ids.numel()
        if n < _PLE_MIN_PREFETCH_ROWS:
            return False
        entries = self._range_entries(valid_ids)
        if _prefetch_virtual_memory(
            _current_process, n, ctypes.c_void_p(entries.data_ptr()), 0
        ):
            return True
        _prefetch_failed = True
        logger.warning(
            "PrefetchVirtualMemory failed for %d PLE rows (error %d); falling back to "
            "thread-fanned page faults for the rest of this process",
            n,
            ctypes.get_last_error(),
        )
        return False

    def _fault(
        self, valid_ids: torch.Tensor, positions: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Read rows from the maps: one prefetch syscall where Windows offers it, otherwise
        by spreading the page faults over the gather pool."""
        if self._prefetch_rows(valid_ids):
            self._read_rows(valid_ids, positions, out)  # pages are resident: copy serially
            return
        n = valid_ids.numel()
        chunk = max(_PLE_MIN_GATHER_CHUNK, -(-n // _PLE_GATHER_WORKERS))
        if chunk >= n:
            self._read_rows(valid_ids, positions, out)
            return
        pool = _ple_gather_pool()
        futures = [
            pool.submit(
                self._read_rows_in_worker,
                valid_ids[start : start + chunk],
                positions[start : start + chunk],
                out,
            )
            for start in range(chunk, n, chunk)
        ]
        try:
            self._read_rows(valid_ids[:chunk], positions[:chunk], out)
        finally:
            # the workers write into ``out``: join them all before returning or propagating
            for future in futures:
                future.result()

    def _read_rows_in_worker(
        self, valid_ids: torch.Tensor, positions: torch.Tensor, out: torch.Tensor
    ) -> None:
        # Inference mode is thread-local; the staging tensors were created under it.
        with torch.inference_mode():
            self._read_rows(valid_ids, positions, out)

    def _read_rows(
        self, valid_ids: torch.Tensor, positions: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Rows go to disjoint ``out`` positions, so concurrent calls need no lock."""
        shard_ids = torch.div(valid_ids, self.rows_per_shard, rounding_mode="floor")
        for shard_id in torch.unique(shard_ids).tolist():
            in_shard = (shard_ids == shard_id).nonzero().reshape(-1)
            dst_positions = positions.index_select(0, in_shard)
            local_ids = valid_ids.index_select(0, in_shard).remainder(self.rows_per_shard)
            rows = self._shards[shard_id].index_select(0, local_ids)
            out.index_copy_(0, dst_positions, rows)

    def close(self) -> None:
        self._row_cache = None
        self._shards.clear()
        for mapping in self._maps.values():
            mapping.close()
        self._maps.clear()
        for fh in self._files.values():
            fh.close()
        self._files.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class MmapPleTable:
    """Mapped PLE storage and its FP8 scale."""

    storage: MmapPleStorage
    weight_scale: torch.Tensor


_PLE_ST_DTYPE = "F8_E4M3"


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def _ple_layout(model_path: str, qwen4_args) -> PleLayout:
    """Parse and validate the PLE shards."""
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")

    shards = tuple(
        PleShard(path=parts[i][0], offset=parts[i][1], nbytes=parts[i][2], rows=rows, cols=cols)
        for i in range(expected)
    )
    shard_bytes = rows * cols
    for shard_id, shard in enumerate(shards):
        if shard.nbytes != shard_bytes:
            raise ValueError(
                f"PLE shard {shard_id} is {shard.nbytes} B, expected {shard_bytes}"
            )
    return PleLayout(
        shards=shards,
        weight_scale=scale,
        rows_per_shard=rows,
        head_dim=cols,
    )


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB and must not also sit in the page cache while the bank holds the same bytes.
    """
    layout = _ple_layout(model_path, qwen4_args)

    bank = HostBank((len(layout.shards) * layout.rows_per_shard, layout.head_dim),
                    torch.float8_e4m3fn)
    bar = byte_bar(sum(shard.nbytes for shard in layout.shards), "Loading PLE table")
    try:
        buf = bank.memoryview()
        dest_offset = 0
        for shard in layout.shards:
            read_range_into(buf, shard.path, file_offset=shard.offset, nbytes=shard.nbytes,
                            dest_offset=dest_offset, workers=workers, chunk=chunk)
            dest_offset += shard.nbytes
            bar.update(shard.nbytes)
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=layout.weight_scale)


def load_mmap_ple_table(model_path: str, qwen4_args) -> MmapPleTable:
    """Map the PLE safetensor ranges."""
    layout = _ple_layout(model_path, qwen4_args)
    return MmapPleTable(
        storage=MmapPleStorage(layout),
        weight_scale=layout.weight_scale,
    )


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None) -> dict:
    """Build the CPU NVFP4 expert source banks for the offload cache (gate/up fused on the output-row axis, down separate; weight_scale_2 carried as the per-row global scale)."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
) -> dict:
    """parallel: same NVFP4 source banks via the common chunked multi-threaded reader."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
    )


__all__ = [
    "MmapPleStorage",
    "MmapPleTable",
    "PleTable",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_mmap_ple_table",
    "load_ple_table",
]

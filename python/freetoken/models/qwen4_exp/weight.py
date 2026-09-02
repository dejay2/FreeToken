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
from freetoken.models.config import (
    vision_execution_mode,
    vision_load_enabled,
    vision_weights_backing,
)
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.models.weight import _ST_DTYPE
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
    # ``mmap``: install the picture tensors as zero-copy views over a copy-on-write mapping
    # of their shard extent instead of reading 856 MiB into process RAM. ``None`` means
    # today's resident behaviour -- either the flag says ``ram`` or the mapping was refused.
    mapped_vision = (
        open_mmap_vision_weights(model_path)
        if stream_vision and vision_weights_backing() == "mmap"
        else None
    )
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
            if has_vision and mapped_vision is None and device.type != "cpu":
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
                if mapped_vision is not None and name.startswith("visual."):
                    tensor = mapped_vision.tensor(name)
                else:
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
_advise_failed = False  # the POSIX twin: set once if madvise(WILLNEED) ever raises


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
        # (mapping, byte offset of the shard's row 0 inside it), per shard: what the POSIX
        # madvise path addresses rows by (_advise_rows); the Windows path uses _shard_bases.
        self._shard_maps: list[tuple[mmap.mmap, int]] = []
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
                self._shard_maps.append((mapping, shard.offset))
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

    def _advise_rows(self, valid_ids: torch.Tensor) -> bool:
        """The POSIX twin of ``_prefetch_rows``: ``madvise(MADV_WILLNEED)`` on every row's page
        range, so the kernel issues the reads together and the serial copy behind them finds
        resident pages; False where the platform mmap cannot advise (Windows).

        UNMEASURED on a real table -- this branch has only ever run on Windows -- so the shape
        simply follows the measured Windows path: one advice per row, no coalescing (the ids
        are near-uniform over 320 M rows), skipped below ``_PLE_MIN_PREFETCH_ROWS`` where the
        syscall costs more than the fault it hides. Unlike PrefetchVirtualMemory this is one
        syscall per row rather than one per batch, so a 100k-row prefill pays ~100k calls;
        still far under the serial-fault cost it replaces, but a coalescing pass is the first
        thing to try if a Linux profile shows it. An OSError disables the path for the
        process, like the Windows FALSE return.
        """
        global _advise_failed
        willneed = getattr(mmap, "MADV_WILLNEED", None)
        if willneed is None or _advise_failed:
            return False
        n = valid_ids.numel()
        if n < _PLE_MIN_PREFETCH_ROWS:
            return False
        maps = self._shard_maps
        if not maps or not hasattr(maps[0][0], "madvise"):
            return False
        page = mmap.PAGESIZE
        dim = self.head_dim
        shard_ids = torch.div(valid_ids, self.rows_per_shard, rounding_mode="floor")
        offsets = (valid_ids - shard_ids * self.rows_per_shard) * dim
        try:
            for shard_id, offset in zip(shard_ids.tolist(), offsets.tolist()):
                mapping, base = maps[shard_id]
                start = base + offset
                aligned = start - start % page
                mapping.madvise(willneed, aligned, start + dim - aligned)
        except OSError as exc:
            _advise_failed = True
            logger.warning(
                "madvise(MADV_WILLNEED) failed for PLE rows (%s); falling back to "
                "thread-fanned page faults for the rest of this process",
                exc,
            )
            return False
        return True

    def _fault(
        self, valid_ids: torch.Tensor, positions: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Read rows from the maps: one prefetch syscall where Windows offers it, a batch of
        madvise(WILLNEED) where POSIX does, otherwise by spreading the page faults over the
        gather pool."""
        if self._prefetch_rows(valid_ids) or self._advise_rows(valid_ids):
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
        self._shard_maps.clear()
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
# Picture weights served from the mapped shard extent
# ======================================================================================
#
# With FREETOKEN_VISION_WEIGHTS=mmap the 333 picture tensors (897,862,112 B of bf16, one
# unbroken extent at the end of model-bf16-00001.safetensors) are never read into process
# memory. Their extent is mapped copy-on-write and each tensor becomes a zero-copy
# torch.frombuffer view installed exactly where the resident tensor used to go, so the
# ~856 MiB stays on the SSD until a picture request faults it in -- and goes back to the
# standby list that feeds the 47.7 GiB mapped PLE table when the OS wants it.
#
# This reuses MmapPleStorage's mechanics (header-derived ranges, ACCESS_COPY, frombuffer
# views, batched PrefetchVirtualMemory / madvise(WILLNEED)) but none of its caching or
# fault fan-out: PLE gathers a few random rows out of 320 M per decoded token, while a
# picture reads 100% of its tensors, in the same order, once. The only cache it wants is
# the OS page cache, and one prefetch over one sequential extent beats any fan-out.

_VISION_RAW_PREFIXES = ("model.visual.", "visual.")
# ``visual.pos_embed.weight`` (5,308,416 B) is the one picture tensor the streamed encode
# uses as a CPU compute operand -- ``F.embedding`` in ``Qwen4VisionModel._position_data`` --
# rather than a memcpy source, and its mapped view would be 2-byte misaligned like every
# other bf16 tensor in this shard. 0.6% of the extent buys an ordinary aligned tensor and
# removes an entire class of question about misaligned CPU kernels.
_VISION_RESIDENT_KEYS = frozenset({"visual.pos_embed.weight"})


@dataclass(frozen=True)
class VisionTensorSpec:
    """One picture tensor's byte range in a safetensor file."""

    name: str  # FreeToken state-dict key, e.g. ``visual.blocks.0.attn.qkv.weight``
    raw_name: str  # the checkpoint's own key
    offset: int  # absolute file offset of its first byte
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class VisionShardLayout:
    """One shard's picture tensors and the file window that covers them."""

    path: str
    start: int  # lowest file offset of any picture tensor here
    end: int  # one past the highest
    tensors: tuple[VisionTensorSpec, ...]

    @property
    def span(self) -> int:
        return self.end - self.start

    @property
    def nbytes(self) -> int:
        return sum(spec.nbytes for spec in self.tensors)


@dataclass(frozen=True)
class VisionLayout:
    """Validated on-disk layout of the picture weights."""

    shards: tuple[VisionShardLayout, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        """Bytes the picture tensors actually occupy."""
        return sum(shard.nbytes for shard in self.shards)

    @property
    def span(self) -> int:
        """Bytes the windows cover -- equal to ``nbytes`` when the extent is unbroken."""
        return sum(shard.span for shard in self.shards)

    @property
    def count(self) -> int:
        return sum(len(shard.tensors) for shard in self.shards)


def _vision_shard_files(folder: str) -> list[str]:
    """Shards holding picture tensors, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {
        shard for name, shard in weight_map.items() if name.startswith(_VISION_RAW_PREFIXES)
    }
    return sorted(os.path.join(folder, shard) for shard in files)


def _vision_layout(model_path: str) -> VisionLayout:
    """Parse and validate the picture tensors' byte ranges out of the shard headers.

    Which shards, which ranges, which dtype: all of it comes from the headers, so a
    re-exported checkpoint with a different layout is either read correctly or rejected,
    never misread. Contiguity is NOT assumed -- the window per shard is
    ``[min offset, max end)``, which for this checkpoint is exactly the 897,862,112-byte run
    at the end of ``model-bf16-00001.safetensors`` and in general is a superset of the
    picture bytes.
    """
    folder = download_hf_weight(model_path)
    shards: list[VisionShardLayout] = []
    dtypes: dict[torch.dtype, str] = {}
    for path in _vision_shard_files(folder):
        header, base = _safetensors_header(path)
        specs: list[VisionTensorSpec] = []
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            name = _rename(key, include_vision=True)
            if name is None or not name.startswith("visual."):
                continue
            dtype = _ST_DTYPE.get(meta["dtype"])
            if dtype is None:
                raise ValueError(f"picture tensor {name} has unsupported dtype {meta['dtype']}")
            shape = tuple(int(extent) for extent in meta["shape"])
            begin, end = meta["data_offsets"]
            count = 1
            for extent in shape:
                count *= extent
            if end - begin != count * dtype.itemsize:
                raise ValueError(
                    f"picture tensor {name} header says {list(shape)} {meta['dtype']} "
                    f"but reserves {end - begin} bytes"
                )
            dtypes.setdefault(dtype, name)
            specs.append(
                VisionTensorSpec(
                    name=name,
                    raw_name=key,
                    offset=base + begin,
                    nbytes=end - begin,
                    shape=shape,
                    dtype=dtype,
                )
            )
        if not specs:
            continue
        shards.append(
            VisionShardLayout(
                path=path,
                start=min(spec.offset for spec in specs),
                end=max(spec.offset + spec.nbytes for spec in specs),
                tensors=tuple(specs),
            )
        )
    if not shards:
        raise ValueError(f"{folder} has no picture tensors (model.visual.*) to map")
    if len(dtypes) > 1:
        named = ", ".join(
            f"{name} is {dtype}" for dtype, name in sorted(dtypes.items(), key=lambda kv: kv[1])
        )
        raise ValueError(f"picture tensors have mixed dtypes: {named}")
    return VisionLayout(shards=tuple(shards), dtype=next(iter(dtypes)))


@dataclass(frozen=True)
class MappedVisionWindow:
    """One ``mmap`` window: where it starts in the file, how long, and where it landed."""

    path: str
    file_offset: int  # aligned down to mmap.ALLOCATIONGRANULARITY
    span: int
    address: int  # virtual address of ``file_offset``


# A one-shot disable per mechanism, separate from the PLE table's: a FALSE return for a
# 112,000-entry PLE row prefetch is a working-set/quota refusal that says nothing about a
# single-entry request over one extent, so one subsystem must not silence the other.
_vision_prefetch_failed = False
_vision_advise_failed = False


class MmapVisionWeights:
    """The picture weights as zero-copy views over a copy-on-write mapping of their shard.

    One mapping per shard that carries picture tensors, starting at the extent rounded down
    to ``mmap.ALLOCATIONGRANULARITY`` and only as long as the extent needs, so the commit
    charge is the extent (~857 MiB here) rather than the whole 1.27 GiB file.

    ``ACCESS_COPY``, not ``ACCESS_READ``: over a read-only buffer ``torch.frombuffer`` warns
    on every call and then hands back a tensor it believes is writable, while copy-on-write
    is silent and correct. Clean copy-on-write pages are still file-backed and still
    reclaimable with no pagefile write.

    NEVER write through one of these views. The copy-on-write fault would make the page
    private and dirty and silently hand back the RAM this mode exists to save. They are only
    ever ``copy_`` sources (``vision._copy_component_state_``).
    """

    def __init__(self, layout: VisionLayout) -> None:
        self.layout = layout
        self.resident_names = frozenset(
            spec.name
            for shard in layout.shards
            for spec in shard.tensors
            if spec.name in _VISION_RESIDENT_KEYS
        )
        self._files: dict[str, BinaryIO] = {}
        self._maps: dict[str, mmap.mmap] = {}
        self._views: dict[str, torch.Tensor] = {}
        # uint8 view over each whole window: what turns a mapping into the virtual address
        # the prefetch API wants, and what ``contains`` measures against.
        self._window_views: list[torch.Tensor] = []
        self._windows: tuple[MappedVisionWindow, ...] = ()
        self._entries = torch.empty((0, 2), dtype=torch.int64)
        self._prefetch_issued = False
        windows: list[MappedVisionWindow] = []
        try:
            for shard in layout.shards:
                fh = open(shard.path, "rb")
                self._files[shard.path] = fh
                aligned = shard.start - shard.start % mmap.ALLOCATIONGRANULARITY
                span = shard.end - aligned
                mapping = mmap.mmap(
                    fh.fileno(), length=span, access=mmap.ACCESS_COPY, offset=aligned
                )
                self._maps[shard.path] = mapping
                window = torch.frombuffer(mapping, dtype=torch.uint8, count=span)
                self._window_views.append(window)
                windows.append(
                    MappedVisionWindow(
                        path=shard.path,
                        file_offset=aligned,
                        span=span,
                        address=window.data_ptr(),
                    )
                )
                for spec in shard.tensors:
                    if spec.name in _VISION_RESIDENT_KEYS:
                        self._views[spec.name] = self._read_resident(fh, spec)
                        continue
                    self._views[spec.name] = (
                        torch.frombuffer(
                            mapping,
                            dtype=torch.uint8,
                            count=spec.nbytes,
                            offset=spec.offset - aligned,
                        )
                        .view(spec.dtype)
                        .reshape(spec.shape)
                    )
        except Exception:
            self.close()
            raise
        self._windows = tuple(windows)
        # int64 [n, 2] laid out exactly as ``WIN32_MEMORY_RANGE_ENTRY[n]`` --
        # ``{PVOID VirtualAddress; SIZE_T NumberOfBytes;}``, 16 B per entry on x64 -- so the
        # tensor's data_ptr is the array pointer the API wants. Same trick as
        # ``MmapPleStorage._range_entries``, minus the vectorized row arithmetic: there is
        # one entry per shard, not one per gathered row.
        self._entries = torch.tensor(
            [[window.address, window.span] for window in self._windows], dtype=torch.int64
        )

    @staticmethod
    def _read_resident(fh: BinaryIO, spec: VisionTensorSpec) -> torch.Tensor:
        """One picture tensor as an ordinary aligned heap tensor (the ``pos_embed`` carve-out)."""
        buffer = bytearray(spec.nbytes)
        fh.seek(spec.offset)
        got = fh.readinto(buffer)
        if got != spec.nbytes:
            raise ValueError(
                f"picture tensor {spec.name}: read {got} of {spec.nbytes} bytes from {fh.name}"
            )
        return torch.frombuffer(buffer, dtype=torch.uint8).view(spec.dtype).reshape(spec.shape)

    def tensor(self, name: str) -> torch.Tensor:
        """The installed tensor for a state-dict key."""
        view = self._views.get(name)
        if view is None:
            raise KeyError(f"{name} is not a picture tensor of this checkpoint")
        return view

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._views)

    @property
    def nbytes(self) -> int:
        """Bytes of picture weights this holder serves."""
        return self.layout.nbytes

    @property
    def mapped_bytes(self) -> int:
        """Of those, the bytes that live in the mapping rather than in process RAM."""
        return self.layout.nbytes - sum(
            spec.nbytes
            for shard in self.layout.shards
            for spec in shard.tensors
            if spec.name in self.resident_names
        )

    @property
    def windows(self) -> tuple[MappedVisionWindow, ...]:
        return self._windows

    def contains(self, address: int) -> bool:
        """Is ``address`` inside a mapped window? The containment check the acceptance
        criteria are written against -- a picture tensor that has left the mapping has been
        copied, and the saving with it."""
        return any(
            window.address <= address < window.address + window.span for window in self._windows
        )

    def prefetch(self) -> bool:
        """Ask the OS to start reading the whole picture extent; ``False`` if it cannot.

        The call is asynchronous, and that is the property this design leans on: measured on
        this box it returned in 84-156 ms while 260-820 ms of cold faulting still had to
        happen, so the ~1.5 s encode that follows consumes the extent while the drive is
        still filling it.

        At most one outstanding prefetch per picture: the scheduler issues it at admission
        and ``forward_layer_streamed`` issues it defensively at entry, and the encode
        releases it when it finishes, so a picture never pays the syscall (40 ms even warm)
        twice. Never raises -- a failed prefetch only costs latency.
        """
        if self._prefetch_issued:
            return True
        if self._prefetch_windows() or self._advise_windows():
            self._prefetch_issued = True
            return True
        return False

    def release_prefetch(self) -> None:
        """Let the next picture issue its own prefetch."""
        self._prefetch_issued = False

    def _prefetch_windows(self) -> bool:
        """Windows: one ``PrefetchVirtualMemory`` covering every window."""
        global _vision_prefetch_failed
        if _prefetch_virtual_memory is None or _vision_prefetch_failed:
            return False
        count = int(self._entries.shape[0])
        if count == 0:
            return False
        if _prefetch_virtual_memory(
            _current_process, count, ctypes.c_void_p(self._entries.data_ptr()), 0
        ):
            return True
        _vision_prefetch_failed = True
        logger.warning(
            "PrefetchVirtualMemory failed for the %d-byte picture extent (error %d); "
            "the encode's own copies will fault the pages in for the rest of this process",
            self.mapped_bytes,
            ctypes.get_last_error(),
        )
        return False

    def _advise_windows(self) -> bool:
        """POSIX: one ``madvise(MADV_WILLNEED)`` per window.

        One call for the whole window, not the per-row loop ``MmapPleStorage._advise_rows``
        needs: the picture extent is a single sequential run.
        """
        global _vision_advise_failed
        willneed = getattr(mmap, "MADV_WILLNEED", None)
        if willneed is None or _vision_advise_failed or not self._windows:
            return False
        try:
            for window in self._windows:
                mapping = self._maps[window.path]
                if not hasattr(mapping, "madvise"):
                    return False
                mapping.madvise(willneed, 0, window.span)
        except OSError as exc:
            _vision_advise_failed = True
            logger.warning(
                "madvise(MADV_WILLNEED) failed for the picture extent (%s); the encode's own "
                "copies will fault the pages in for the rest of this process",
                exc,
            )
            return False
        return True

    def close(self) -> None:
        self._views.clear()
        self._window_views.clear()
        self._windows = ()
        self._entries = torch.empty((0, 2), dtype=torch.int64)
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


_vision_sources: dict[str, MmapVisionWeights] = {}
_vision_sources_lock = threading.Lock()
_vision_mmap_warned = False


def open_mmap_vision_weights(model_path: str) -> MmapVisionWeights | None:
    """Map this checkpoint's picture extent, or ``None`` to fall back to resident RAM.

    Process-scoped and idempotent per checkpoint folder: ``iter_weights`` builds the holder
    while it installs the views, and the model claims the same object afterwards, so there
    is never a second mapping or a second 857 MiB of commit charge.

    A layout the reader cannot trust -- no picture tensors at all, a dtype it cannot view, a
    header that disagrees with itself -- raises, because that means the picture weights are
    not where the caller thinks they are. A mapping the OS refuses costs only the
    optimization, so it warns once and falls back.
    """
    global _vision_mmap_warned
    folder = download_hf_weight(model_path)
    with _vision_sources_lock:
        existing = _vision_sources.get(folder)
        if existing is not None:
            return existing
        layout = _vision_layout(folder)
        try:
            source = MmapVisionWeights(layout)
        except Exception as exc:  # noqa: BLE001 - an optimization must never fail a boot
            if not _vision_mmap_warned:
                _vision_mmap_warned = True
                logger.warning(
                    "picture weights: mapping the %d-byte extent of %s failed (%s); serving "
                    "them from resident RAM for the rest of this process",
                    layout.nbytes,
                    folder,
                    exc,
                )
            return None
        logger.info(
            "Picture weights: mapped %d tensors, %d bytes in %d window(s) of %s",
            layout.count,
            layout.nbytes,
            len(source.windows),
            ", ".join(os.path.basename(shard.path) for shard in layout.shards),
        )
        _vision_sources[folder] = source
        return source


def mmap_vision_weights(model_path: str) -> MmapVisionWeights | None:
    """The holder built for this checkpoint, or ``None`` if the weights are resident."""
    with _vision_sources_lock:
        return _vision_sources.get(download_hf_weight(model_path))


def close_mmap_vision_weights(model_path: str | None = None) -> None:
    """Unmap the picture extent.

    The engine holds the mapping for its whole life, exactly as it holds the PLE one, so
    nothing on the serving path calls this; it exists for tests and explicit teardown.
    """
    with _vision_sources_lock:
        keys = list(_vision_sources) if model_path is None else [download_hf_weight(model_path)]
        for key in keys:
            source = _vision_sources.pop(key, None)
            if source is not None:
                source.close()


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
    "MmapVisionWeights",
    "PleTable",
    "VisionLayout",
    "close_mmap_vision_weights",
    "iter_weights",
    "load_mmap_ple_table",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
    "mmap_vision_weights",
    "open_mmap_vision_weights",
]

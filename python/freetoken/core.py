from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Tuple

import torch

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend, BaseAttnMetadata
    from freetoken.attention.linear import FLAMetadata
    from freetoken.kvcache import BaseCacheHandle, BaseKVCachePool
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.moe import BaseMoeBackend
    from freetoken.moe.offload_cache import OffloadMoeCache


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
    # Stop strings (OpenAI `stop` / Anthropic `stop_sequences`). Generation finishes when one
    # appears in the decoded output; the matched substring (and anything after) is trimmed.
    stop_strs: list[str] = field(default_factory=list)

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass
class SpecInflight:
    """The undo record of one speculative decode step, written by
    ``Scheduler._prepare_spec_batch`` and consumed by ``Scheduler._rollback_spec_tokens``.

    A speculative step forwards ``width`` rows on the request's REAL page table and KV, then
    keeps only the accepted prefix. Rejected QSA rows need no KV rewind (the K/V write is
    position-addressed and the compressed-slab scorer clamps visibility to
    ``sequence_length // index_ratio``), but the lengths, the pages and the page-table cells
    the step stamped all do.
    """

    width: int          # w = 1 + drafts
    cached_len: int     # the request's KV-valid length before the step
    device_len: int     # the request's device length before the step (== cached_len + 1)
    first_page: int     # first page index the step's allocate_paged considered
    last_page: int      # one past the last
    pages: torch.Tensor       # page-base token indices the step allocated, allocation order
    page_row: torch.Tensor    # page_table[table_idx, first_page*ps : last_page*ps] before it


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
    # Optional precomputed multimodal soft-token embeddings (GPU, [num_image_tokens,
    # hidden]) scattered at image-token positions during this request's prefill.
    mm_embeds: torch.Tensor | None = None
    # Picture requests remain private after the one-shot soft embeddings are released.
    cache_private: bool = False
    # Full-prompt three-axis Qwen rotary coordinates plus the generated-token offset.
    # Logical/cache addressing continues to use the scalar lengths above.
    mrope_position_ids: torch.Tensor | None = None
    mrope_position_delta: int = 0

    # --- hybrid-radix (GDN linear-state) per-request slots; None for non-hybrid models or
    # until allocated from LinearStatePool. Set by the scheduler (P2). ---
    linear_slot_idx: int | None = None              # live GDN state slot (sglang mamba_pool_idx)
    mamba_ping_pong: tuple[int, int] | None = None  # 2 donatable track slots under overlap
    mamba_next_track_idx: int = 0                   # which ping-pong slot is the next snapshot dst (0/1)
    mamba_last_track_seqlen: int | None = None      # chunk-aligned committed len of the last snapshot
    mamba_restore_src: int | None = None            # on a prefix hit: tree snapshot slot to COW into the live slot (first chunk only)
    swa_evicted_seqlen: int = 0                      # SWA radix: positions < this had their swa KV freed (slid out of window) during decode
    decode_batch_idx: int = 0                        # SWA radix: # of decode forwards done; the proactive free_swa skips the first (overlap guard)
    # Set once, at the first sampled tool-call opener token (scheduler detection): the state
    # length just after that token (its index + 1). A client-side rewrite of the echoed tool
    # call diverges strictly after this point, so it is the deepest reuse boundary that
    # survives such a rewrite. GDN: the state is frozen into a ping-pong slot when cached_len
    # reaches it (snapshot_toolcall_anchor) and donated at finish. SWA: caps the proactive
    # out-of-window eviction so the window ending here stays resumable.
    toolcall_anchor_len: int | None = None
    # Non-None between _prepare_spec_batch and _rollback_spec_tokens: this request has
    # forwarded speculative rows that are not yet settled. A radix commit under one is
    # forbidden (CacheManager.cache_req) -- it re-points the page-table row, which no
    # device_len rewind can undo.
    spec_inflight: SpecInflight | None = None
    # Abort arrived while this request's forward was in flight (overlap scheduling). The abort
    # handler must not free resources under an in-flight forward; it sets this flag and
    # _process_last_data frees the request when the batch drains (after copy_done.synchronize).
    aborted: bool = False

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        self._alloc_ids_buf()

    def _alloc_ids_buf(self) -> None:
        self._ids_buf = torch.empty(self.max_device_len, dtype=self.input_ids.dtype)
        self._ids_buf[: self.device_len] = self.input_ids
        self.input_ids = self._ids_buf[: self.device_len]

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.complete_many(1)

    def complete_many(self, n: int) -> None:
        """Advance the device length by a step that emitted ``n`` tokens. The rows this
        forward wrote become the cached prefix, exactly as for a one-token step."""
        assert n >= 1
        self.cached_len = self.device_len
        self.device_len += n

    def append_host(self, tokens: torch.Tensor) -> None:
        """Append a step's sampled tokens; already generic over the run's width."""
        n = self.input_ids.numel()
        m = n + tokens.numel()
        assert m <= self.max_device_len
        self._ids_buf[n:m] = tokens
        self.input_ids = self._ids_buf[:m]

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )



@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # Private Qwen3.8 MTP verification keeps causal prefill semantics while the
    # offloaded MoE reads only routed experts through its decode cache. Ordinary
    # scheduler batches leave this disabled.
    mtp_verify: bool = field(default=False, init=False)
    # Tokens this step emits per request. Plain decode emits one; a wider step reserves
    # emit_width sampled-token slots per request in the write mapping.
    emit_width: int = field(default=1, init=False)
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    # Optional Qwen temporal/height/width positions [3, padded_tokens]. Scalar
    # ``positions`` remains authoritative for causal masks, cache writes, and QSA rows.
    rope_positions: torch.Tensor | None = field(default=None, init=False)
    out_loc: torch.Tensor | None = field(init=False)
    # Per-(padded-)request table_idx as a GPU int64 tensor, used by GatedDeltaNet
    # decode to gather/scatter recurrent+conv state without host-side loops (so the
    # decode step is CUDA-graph capturable). Set by the scheduler / graph buffer.
    linear_table_idx: torch.Tensor | None = field(default=None, init=False)
    # Per-forward GatedDeltaNet metadata (cu_seqlens / cache_indices / continuation
    # flags), built once and shared by all GDN layers. Lazily built by the GDN op if
    # the scheduler/graph didn't set it.
    fla_metadata: "FLAMetadata | None" = field(default=None, init=False)
    padded_reqs: List[Req] = field(init=False)
    # DSV4 paged-KV out-locations for this batch (None for non-DSV4 models). Set by the scheduler.
    # This decode batch's padded per-row page-table rows. Attention backends that must read
    # positions anywhere in a request's history snapshot those rows before a captured replay
    # (DSV4), since the next batch's allocate_paged mutates the live table.
    active_table_idx: "torch.Tensor | None" = None
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    # concatenated multimodal soft-token embeddings for a prefill batch (or None)
    mm_embeds: torch.Tensor | None = field(default=None, init=False)
    # Prefill log stats snapshotted at schedule time (before forward's complete_one()
    # advances cached_len), so the prefill log reports the tokens actually forwarded and
    # the prefix-cache hit -- matching SGLang's #new-token / #cached-token. Set by the
    # PrefillManager; 0 on decode batches.
    log_new_tokens: int = field(default=0, init=False)
    log_cached_tokens: int = field(default=0, init=False)
    # (uid, complete prompt length, prefix-cache hit) for requests entering their first
    # prepared prefill batch. The scheduler turns these into PromptAdmittedMsg only AFTER
    # _prepare_batch succeeds. Continuation chunks leave this empty, so accounting is
    # exactly-once.
    prompt_admissions: List[Tuple[int, int, int]] = field(default_factory=list, init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    moe_offload_cache: OffloadMoeCache | None = None
    kv_cache: BaseKVCachePool = field(init=False)
    # Per-request recurrent state for GatedDeltaNet layers; set by the engine for
    # hybrid linear-attention models, otherwise None.
    linear_state_pool: LinearStatePool | None = None
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX

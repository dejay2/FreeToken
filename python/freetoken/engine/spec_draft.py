"""The MTP draft head, standalone -- design section 6.3.

Integrated speculation runs WITHOUT the shadow observer (the two flags are mutually
exclusive), so the draft cannot ride the observer's machinery. This module rebuilds the draft
half of ``MTPShadowObserver.__init__`` out of the same pieces and nothing else:

  * ``derive_mtp_model_config`` + ``Qwen4ExpMTPModel`` -- the native one-layer scheme-A head;
  * ``MTPWeightStore`` + ``build_mtp_weight_plan`` -- its 31 bf16 tensors, read from the
    TARGET CHECKPOINT. ``FREETOKEN_MTP_PRIVATE_ROOT`` backs only the observer's trace file and
    the pre-converted NVFP4 expert banks, and integrated mode needs neither;
  * ``MTPStagedModelRunner(resident=True)`` + ``MTPGPUExpertRunner`` -- dense weights and both
    expert banks device-resident, the placement the spike measured at 17.4 ms/draft. The GPU
    runner drops the host bank copy once it has uploaded, which is what keeps the ~5 GB
    transient from becoming a ~5 GB resident.

The chain also gets its OWN LM head (``engine/spec_lmhead.py``,
``FREETOKEN_MTP_SPEC_DRAFT_LMHEAD``): the target's is bf16 and 1.27 GB, which a depth-5 chain
would stream out of HBM five times for five one-row projections. The default is a weight-only
int8 copy; ``bf16`` shares the target's head exactly as before. The TARGET's own logits are
untouched either way -- this is a separate object over a separate copy of the weight.

``FREETOKEN_MTP_SPEC_EXPERT_FORMAT=nvfp4`` (with ``FREETOKEN_MTP_SPEC_NVFP4_MANIFEST``) swaps
that last piece for ``MTPNVFP4GPUExpertRunner``: the same 512 experts held quantized (~1.42 GB
plus a fixed dequant scratch) instead of exact (5.03 GB), returning ~3.2 GB to the TARGET's
expert cache, whose hit rate is what decode speed actually turns on. Quantized draft weights
move the draft's logits slightly, so acceptance may dip; correctness cannot -- the target
still decides every token. The default is unchanged.

WHAT THE HEAD REMEMBERS
-----------------------
The MTP head is a one-layer QSA model that attends its OWN running context, so it needs its
own KV. That context is the sequence of shifted pairs ``(target_hidden[i], embed(token[i+1]))``
-- one row per accepted token, so its committed length tracks the target's ``cached_len``
exactly. The private state is therefore just: an identity page table (logical position ==
physical slot, no allocator), one ``QSAKVCache`` whose pending ring is widened to
``index_ratio + depth`` (a narrower ring aliases a draft row onto an open compression group's
still-needed members -- silently wrong keys), one ``QSASparseAttnBackend``, and the scalars
``committed_len`` / ``pending_hidden``. No recurrent state, no PLE, no leases, no free list.

Fallback-decode pairs are BUFFERED rather than committed. Committing one is a whole eager
one-row draft forward, and an adaptive cooldown spends long stretches in which no proposal
ever reads its result; the buffer is drained in a single multi-row commit when the head is
next needed. ``committed_len`` plus the buffered rows is therefore what tracks the target's
``cached_len``, and it is what ``is_ready`` compares.

A PROPOSAL IS SPECULATIVE IN THE DRAFT'S OWN KV TOO
--------------------------------------------------
Drafting ``k`` tokens writes ``k - 1`` recursive rows into that KV. They are undone by
rewinding ``committed_len`` (the identity page table means the next real row overwrites the
same slots) plus restoring the two pending rings, which have no epoch tag and would otherwise
survive the rewind.

THE DISTRIBUTION CONTRACT
-------------------------
``SpecSampler``'s acceptance divides by the draft's own ``q``, so the proposal must be drawn
from ``filtered_probs(draft_logits, args)`` -- the same filter the server samples the target
with. ``MTPDraftSampler.sample`` is exactly that filter's one-row form, pinned by
``tests/engine/test_spec_draft.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch, Req

# The GPU expert runner's per-call row cap; priming chunks are split to it.
_MAX_DRAFT_ROWS = 128

_EXPERT_FORMAT_ENV = "FREETOKEN_MTP_SPEC_EXPERT_FORMAT"
_NVFP4_MANIFEST_ENV = "FREETOKEN_MTP_SPEC_NVFP4_MANIFEST"
_CONF_LOG_ENV = "FREETOKEN_MTP_SPEC_CONF_LOG"


def spec_conf_log_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the confidence-cut diagnosis log is armed (``FREETOKEN_MTP_SPEC_CONF_LOG``).

    Read fresh on every call rather than frozen into a module constant: the flag arms an
    instrumentation-only side channel, so a test that sets it must be able to see it take
    effect without reimporting the module. The cost of an armed proposal is a topk over the
    logit rows it already keeps plus one readback; the cost of an unarmed one is this dict
    lookup and nothing else -- in particular no device sync.
    """
    env = os.environ if environ is None else environ
    return bool((env.get(_CONF_LOG_ENV, "") or "").strip())


def resolve_spec_expert_placement(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, Path | None]:
    """Which expert banks the draft head puts on the card, and where they come from.

    Parsed here rather than on ``EngineConfig``: the placement is a private bank layout of
    this head, not a serving flag. The default is the exact bf16 banks read from the target
    checkpoint -- the placement that shipped, unchanged.
    """
    env = os.environ if environ is None else environ
    placement = (env.get(_EXPERT_FORMAT_ENV, "") or "bf16").strip().lower() or "bf16"
    if placement not in {"bf16", "nvfp4"}:
        raise ValueError(f"{_EXPERT_FORMAT_ENV} must be bf16 or nvfp4, got {placement!r}")
    if placement == "bf16":
        return "bf16", None
    raw = (env.get(_NVFP4_MANIFEST_ENV, "") or "").strip()
    if not raw:
        raise ValueError(
            f"{_EXPERT_FORMAT_ENV}=nvfp4 needs {_NVFP4_MANIFEST_ENV} to name the "
            "pre-converted expert manifest"
        )
    manifest = Path(raw).expanduser()
    if not manifest.is_file():
        raise ValueError(f"{_NVFP4_MANIFEST_ENV} is not a manifest file: {manifest}")
    return "nvfp4", manifest.resolve()


def resolve_draft_cut_mode(spec_decode) -> str:
    """Which cut strategy the chain runs (``SpecDecodeConfig.draft_cut_mode``).

    A ``getattr`` with a default, exactly as ``conf_cut`` is read: a hand-built stub config
    (tests, the shadow tooling) that predates the field gets the shipped mode. Validated here
    rather than trusted -- an unrecognized value would otherwise silently mean ``chain``, and
    the point of the flag is that an operator can tell which arm of an A/B they are on.
    """
    mode = str(getattr(spec_decode, "draft_cut_mode", "chain"))
    if mode not in ("chain", "step"):
        raise ValueError(f"the draft cut mode must be chain or step, got {mode!r}")
    return mode


def spec_expert_runner_type(placement: str):
    from freetoken.models.qwen4_exp.mtp_spike import (
        MTPGPUExpertRunner,
        MTPNVFP4GPUExpertRunner,
    )

    if placement == "bf16":
        return MTPGPUExpertRunner
    if placement == "nvfp4":
        return MTPNVFP4GPUExpertRunner
    raise ValueError(f"MTP draft expert placement must be bf16 or nvfp4, got {placement!r}")


def load_spec_expert_banks(placement: str, manifest: Path | None, store):
    """The banks the placement asks for: the checkpoint's exact rows, or the NVFP4 six."""

    from freetoken.models.qwen4_exp.mtp_spike import (
        MTPBF16ExpertBanks,
        MTPNVFP4ExpertBanks,
    )

    if placement == "bf16":
        return MTPBF16ExpertBanks.from_store(store)
    if manifest is None:
        raise ValueError(f"the nvfp4 draft placement needs {_NVFP4_MANIFEST_ENV}")
    return MTPNVFP4ExpertBanks.from_manifest(manifest, validate_hashes=True)


def build_shifted_pairs(
    pending_hidden: torch.Tensor | None,
    hidden: torch.Tensor,
    inputs_embeds: torch.Tensor,
    *,
    next_embedding: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build exact hidden[i]/following-embedding pairs across target prefill chunks."""
    if hidden.ndim != 2 or inputs_embeds.ndim != 2 or hidden.shape[0] != inputs_embeds.shape[0]:
        raise ValueError("target hidden and embeddings must have the same row count")
    pair_hidden = []
    pair_embeds = []
    if pending_hidden is not None:
        pair_hidden.append(pending_hidden)
        pair_embeds.append(inputs_embeds[:1])
    if hidden.shape[0] > 1:
        pair_hidden.append(hidden[:-1])
        pair_embeds.append(inputs_embeds[1:])
    pending = hidden[-1:].clone()
    if next_embedding is not None:
        pair_hidden.append(pending)
        pair_embeds.append(next_embedding)
        pending = None
    if not pair_hidden:
        return hidden[:0], inputs_embeds[:0], pending
    return torch.cat(pair_hidden), torch.cat(pair_embeds), pending


def build_shifted_rope_positions(
    pending_rope: torch.Tensor | None,
    current_rope: torch.Tensor | None,
    *,
    had_pending_hidden: bool,
    final_chunk: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Mirror shifted hidden rows for Qwen three-axis picture positions."""
    if current_rope is None:
        if pending_rope is not None:
            raise RuntimeError("MTP picture positions disappeared within one request")
        return None, None
    parts = []
    if had_pending_hidden:
        if pending_rope is None:
            raise RuntimeError("MTP picture position state is incomplete")
        parts.append(pending_rope)
    current_rows = current_rope.shape[1] if final_chunk else max(current_rope.shape[1] - 1, 0)
    if current_rows:
        parts.append(current_rope[:, :current_rows])
    paired = torch.cat(parts, dim=1) if parts else current_rope[:, :0]
    pending = None if final_chunk else current_rope[:, -1:]
    return paired, pending


@dataclass(frozen=True)
class DraftProposal:
    """One cycle's drafts and the FULL logit rows they were drawn from.

    Rejection sampling corrects from the residual ``(p - q)+``, so the sampler needs all of
    ``q``, not just the proposed tokens' probabilities.

    ``draft_top1`` / ``draft_top1_gap`` are diagnosis only and are None unless
    ``spec_conf_log_enabled()``. They are the draft head's RAW softmax top-1 probability and
    its top1-minus-top2 margin per drafted token -- the head's own confidence, which is what a
    confidence cut would have to decide on. The verdict's ``draft_probabilities`` cannot serve:
    that is ``q`` under the REQUEST's sampling filter, and a greedy request's filter makes it
    degenerate (1.0 for every draft, accepted or not). Timing-only fields, hence
    ``compare=False`` -- the same treatment ``SpecDecision`` gives its own instrumentation.
    """

    tokens: tuple[int, ...]
    logits: torch.Tensor  # [k, vocab]
    draft_top1: tuple[float, ...] | None = field(default=None, compare=False)
    draft_top1_gap: tuple[float, ...] | None = field(default=None, compare=False)


class SpecDraftHead:
    """The resident MTP head plus the running context it attends."""

    def __init__(
        self,
        engine,
        spec_decode,
        *,
        seed: int | None = None,
        num_pages: int | None = None,
    ) -> None:
        from freetoken.engine.spec_lmhead import (
            build_draft_lm_head,
            resolve_draft_lmhead_placement,
        )
        from freetoken.engine.spec_sample import resolve_spec_seed
        from freetoken.models.qwen4_exp.mtp_spike import (
            Qwen4ExpMTPModel,
            derive_mtp_model_config,
        )
        from freetoken.utils.torch_utils import torch_dtype

        self.engine = engine
        self.device = engine.device
        self.depth = int(spec_decode.depth)
        # The confidence cut arrives on the SAME resolved object ``depth`` does, so nothing in
        # engine.py has to learn about it (``SpecDraftHead(self, config.spec_decode)``). The
        # getattr keeps a hand-built stub config (tests, the shadow tooling) working: a config
        # that predates the field simply drafts the full depth, which is the old behaviour.
        self.conf_cut = float(getattr(spec_decode, "conf_cut", 0.0))
        # ...and so does WHEN the cut is applied. Same getattr rule, same reason: a config
        # that predates the field drafts the whole chain and truncates once, which is the
        # default. See ``SpecDecodeConfig.draft_cut_mode`` for the cost either mode pays.
        self.draft_cut_mode = resolve_draft_cut_mode(spec_decode)
        self.seed = resolve_spec_seed() if seed is None else int(seed)
        self.target_ctx = engine.ctx
        self.target_model = engine.model
        self.mtp_config = derive_mtp_model_config(engine.config.model_config)
        # The chain's own LM head: a private quantized copy of the target's, or the target's
        # own object under ``FREETOKEN_MTP_SPEC_DRAFT_LMHEAD=bf16``. The TARGET's logits go on
        # coming out of ``engine.model.lm_head``, untouched either way.
        self.lmhead_placement = resolve_draft_lmhead_placement()
        self.draft_lm_head = build_draft_lm_head(
            self.target_model.lm_head,
            placement=self.lmhead_placement,
            device=self.device,
        )
        # ...and the widest flush that must stay off the expert-major loop: a cold request
        # buffers a whole backed-off cooldown of plain decode observations, then commits them
        # in one multi-row forward at the head of the next ``propose``.
        self._max_flush_rows = min(
            _MAX_DRAFT_ROWS,
            int(getattr(spec_decode, "cooldown_cap", 64)) + self.depth + 8,
        )

        page_size = 64  # pinned by QSA, exactly as the target's pool is
        self.num_pages = (
            num_pages
            if num_pages is not None
            else -(-int(engine.max_seq_len) // page_size) + 1
        )
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            self.model = Qwen4ExpMTPModel(self.mtp_config)
        self._load_weights()
        self._init_private_state(page_size)

        self._uid: int | None = None
        self.committed_len = 0
        self._pending_hidden: torch.Tensor | None = None
        self._pending_rope: torch.Tensor | None = None
        self._sample: torch.Tensor | None = None
        self._recursive: torch.Tensor | None = None
        self._saved_blocks: dict[int, torch.Tensor] | None = None
        self._buffered: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        self._sampler = None

    # ------------------------------------------------------------------------ construction

    def _load_weights(self) -> None:
        from freetoken.models.qwen4_exp.mtp_spike import (
            MTPStagedModelRunner,
            MTPWeightStore,
            build_mtp_weight_plan,
        )

        self.expert_placement, manifest = resolve_spec_expert_placement()
        model_state = self.model.state_dict()
        with MTPWeightStore(self.engine.config.model_path) as store:
            plan = build_mtp_weight_plan(store.keys, model_state.keys())
            cpu_weights = {
                entry.model_name: store.materialize(entry, device=torch.device("cpu"))
                for entry in plan.entries
                if not entry.expert
            }
            banks = load_spec_expert_banks(self.expert_placement, manifest, store)
        self.staged_model = MTPStagedModelRunner(
            self.model, cpu_weights, device=self.device, resident=True
        )
        extra = (
            # ``max_gather_rows`` is the widest CALL that stays on the (chunked) gather path,
            # so a buffered flush never pays the expert-major loop's host round trip; the
            # gather WORKSPACE is unchanged by it. The quantized gather's scratch is resident,
            # so its pass width is the widest per-cycle call -- the accepted run, ``1 + depth``
            # rows -- and nothing wider.
            {"max_gather_rows": self._max_flush_rows}
            if self.expert_placement == "bf16"
            else {
                "max_gather_tokens": self.depth + 1,
                "max_gather_rows": self._max_flush_rows,
            }
        )
        self.expert_runner = spec_expert_runner_type(self.expert_placement)(
            banks,
            top_k=self.mtp_config.num_experts_per_tok,
            activation=self.mtp_config.hidden_act,
            renormalize=self.mtp_config.norm_topk_prob,
            max_tokens=_MAX_DRAFT_ROWS,
            num_threads=1,
            device=self.device,
            **extra,
        )
        # the runner uploaded the banks and kept only their geometry; this was the last
        # reference pinning the host copy
        del banks
        self.model.layers.op_list[0].mlp.experts.attach_runner(self.expert_runner)

    def _init_private_state(self, page_size: int) -> None:
        from freetoken.attention.qsa_sparse import QSASparseAttnBackend
        from freetoken.kvcache import create_kvcache_pool

        width = self.num_pages * page_size
        self.page_table = torch.zeros((2, width), dtype=torch.int32, device=self.device)
        # identity: logical position i lives in physical slot i, so out_loc == positions and
        # the compression group-closing test (out_loc % ratio == ratio - 1) is exact
        self.page_table[0] = torch.arange(width, dtype=torch.int32, device=self.device)
        self.page_table[1].fill_(width - page_size)
        self.kv_cache = create_kvcache_pool(
            model_config=self.mtp_config,
            num_pages=self.num_pages,
            page_size=page_size,
            dtype=torch.bfloat16,
            device=self.device,
            num_req_slots=2,
            num_speculative_tokens=self.depth,
        )
        self.kv_cache.attach_page_table(self.page_table)
        with self._private_context_fields():
            self.attn_backend = QSASparseAttnBackend(self.mtp_config)
            # The chain is ungraphed, so nothing else would ever give this backend static
            # buffers: without this every step took the fully allocating eager path (three
            # pinned host tensors and ~5 device transients per forward). Armed inside the
            # private context so the workspace is sized from THIS head's page table.
            self.attn_backend.enable_step_workspace()

    @property
    def resident_bytes(self) -> int:
        """Everything the head charges the card: dense weights, expert banks, KV, LM head.

        The private LM head copy counts only when there IS one: ``bf16`` hands back the
        target's own object, whose bytes the target already paid for.
        """
        pool = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.kv_cache._kv_buffer,
                self.kv_cache._cmp_k_buffer,
                self.kv_cache._pending_ring,
                self.kv_cache._pending_position_ring,
                self.page_table,
            )
        )
        head = getattr(self, "draft_lm_head", None)
        private_head = (
            0
            if head is None or head is self.target_model.lm_head
            else int(head.resident_bytes)
        )
        return (
            int(self.staged_model.resident_bytes)
            + int(self.expert_runner.resident_bytes)
            + int(pool)
            + private_head
        )

    # ------------------------------------------------------------------ private context

    def _private_context_fields(self):
        from contextlib import contextmanager

        @contextmanager
        def _swap():
            ctx = self.target_ctx
            old = (
                getattr(ctx, "kv_cache", None),
                getattr(ctx, "page_table", None),
                getattr(ctx, "attn_backend", None),
            )
            try:
                ctx.kv_cache = self.kv_cache
                ctx.page_table = self.page_table
                yield
            finally:
                ctx.kv_cache, ctx.page_table, ctx.attn_backend = old

        return _swap()

    def _private_forward(self, batch):
        from contextlib import contextmanager

        @contextmanager
        def _swap():
            ctx = self.target_ctx
            old = (ctx.kv_cache, ctx.page_table, ctx.attn_backend)
            assert ctx._batch is None, "the target context is still in a forward"
            try:
                ctx.kv_cache = self.kv_cache
                ctx.page_table = self.page_table
                ctx.attn_backend = self.attn_backend
                with ctx.forward_batch(batch):
                    yield
            finally:
                ctx.kv_cache, ctx.page_table, ctx.attn_backend = old

        return _swap()

    def _step_tensors(self, rows: int):
        """This head's reusable per-row-count ``(positions, out_loc, active_table_idx)``.

        A chain step used to allocate all three on the device every forward (an ``arange``, a
        ``contiguous()`` of a page-table slice and a ``zeros``). They are pure functions of
        the row count and the committed length, so one buffer per row count is enough: both
        refills below are stream-ordered kernels, so a buffer cannot be rewritten out from
        under the forward that is still reading it. Built lazily -- the row counts a request
        actually uses are 1 (a draft step) and whatever its flushes come to.
        """
        cache = getattr(self, "_step_cache", None)
        if cache is None:
            cache = self._step_cache = {}
        buffers = cache.get(rows)
        if buffers is None:
            buffers = cache[rows] = (
                torch.empty(rows, dtype=torch.int32, device=self.device),
                torch.empty(rows, dtype=torch.int32, device=self.device),
                torch.zeros(1, dtype=torch.int32, device=self.device),
            )
        return buffers

    def _batch(self, rows: int, *, rope_positions=None, capture=None, saved=None):
        from types import SimpleNamespace

        start = self.committed_len
        req = SimpleNamespace(
            table_idx=0, cached_len=start, device_len=start + rows, extend_len=rows
        )
        positions, out_loc, active_table_idx = self._step_tensors(rows)
        torch.arange(start, start + rows, out=positions)
        out_loc.copy_(self.page_table[0, start : start + rows])
        batch = SimpleNamespace(
            reqs=[req],
            padded_reqs=[req],
            phase="decode" if rows == 1 else "prefill",
            size=1,
            padded_size=1,
            is_prefill=rows != 1,
            is_decode=rows == 1,
            positions=positions,
            rope_positions=rope_positions,
            out_loc=out_loc,
            attn_metadata=None,
            active_table_idx=active_table_idx,
        )
        if capture is not None:
            batch.mtp_qsa_capture_blocks = capture
        if saved is not None:
            batch.mtp_qsa_saved_blocks = saved
        # Under the head's OWN context: ``prepare_metadata``'s non-decode arm gathers the page
        # row through ``get_global_ctx().page_table``, which outside this swap is the TARGET's
        # table -- a different (and for a private pool, out-of-range) set of physical pages.
        with self._private_context_fields():
            self.attn_backend.prepare_metadata(batch)
        return batch

    def _run_rows(self, embeddings, hidden, *, rope_positions=None, capture=None, saved=None):
        """Commit ``embeddings.shape[0]`` shifted pairs, returning the LAST row's outputs."""
        outputs = None
        total = embeddings.shape[0]
        for lo in range(0, total, _MAX_DRAFT_ROWS):
            hi = min(lo + _MAX_DRAFT_ROWS, total)
            local_capture = capture if hi == total else None
            rope = None if rope_positions is None else rope_positions[:, lo:hi]
            batch = self._batch(
                hi - lo, rope_positions=rope, capture=local_capture, saved=saved
            )
            with self._private_forward(batch):
                outputs = self.staged_model.forward(
                    embeddings[lo:hi].to(self.device), hidden[lo:hi].to(self.device), batch
                )
            self.committed_len += hi - lo
        assert outputs is not None
        return outputs[0][-1:], outputs[1][-1:]

    # ---------------------------------------------------------------------- request lifecycle

    def reset_request(self, uid: int) -> None:
        from freetoken.models.qwen4_exp.mtp_spike import MTPDraftSampler

        self._uid = int(uid)
        self.committed_len = 0
        self._pending_hidden = None
        self._pending_rope = None
        self._sample = None
        self._recursive = None
        self._saved_blocks = None
        self._buffered = []
        self.kv_cache._kv_buffer.zero_()
        self.kv_cache._cmp_k_buffer.zero_()
        self.kv_cache._pending_ring.zero_()
        self.kv_cache._pending_position_ring.zero_()
        self._sampler = MTPDraftSampler(
            seed=_draft_seed(self.seed, self._uid), device=self.device
        )

    def is_ready(self, req: "Req") -> bool:
        """True once the head has consumed every token the request has emitted.

        A request admitted before speculation was enabled, or one still mid-chunked-prefill,
        is not ready and the loop falls back to a plain one-row decode for that step.

        Buffered decode pairs count as consumed -- ``propose`` flushes before it reads
        ``_sample``, so a buffered row is already this head's context. The scheduler calls this
        every iteration, so it must not itself flush. ``_sample`` is written only by a flush,
        hence the second arm: a head whose every pair is still buffered has no ``_sample`` yet
        and would have one the moment a proposal asked for it.
        """
        return (
            self._uid == req.uid
            and (self._sample is not None or bool(self._buffered))
            and self._pending_hidden is None
            and self.committed_len + self._buffered_rows == req.cached_len
        )

    # ---------------------------------------------------------------- the engine capture seam

    def observe_forward(self, batch: "Batch", capture, next_tokens) -> None:
        """Consume one ordinary (prefill or fallback-decode) forward's capture.

        ``capture`` is ``(logits, multi_stream, inputs_embeds)`` from
        ``Qwen4ExpForCausalLM.forward_mtp_capture`` -- the same tensors the observer takes,
        minus every observer-only artifact: no CPU snapshot completion, no lease, no trace.
        A non-final prompt chunk only carries its last hidden row forward; the final chunk (or
        a decode step) closes the pair with the token the target just sampled.
        """
        if len(batch.reqs) != 1:
            raise RuntimeError("integrated speculation observes one request at a time")
        req = batch.reqs[0]
        if self._uid != req.uid:
            self.reset_request(req.uid)
        _, hidden, embeds = capture
        chunked = type(req).__name__ == "ChunkedReq"
        next_embedding = None
        if not chunked:
            next_embedding = self.target_model.model.embed_tokens.forward(
                next_tokens.to(self.device).reshape(1)
            )
        had_pending = self._pending_hidden is not None
        pending_rope = self._pending_rope
        rope = None if batch.rope_positions is None else batch.rope_positions
        paired_hidden, paired_embeds, self._pending_hidden = build_shifted_pairs(
            self._pending_hidden, hidden, embeds, next_embedding=next_embedding
        )
        paired_rope, self._pending_rope = build_shifted_rope_positions(
            pending_rope,
            rope,
            had_pending_hidden=had_pending,
            final_chunk=next_embedding is not None,
        )
        if paired_hidden.shape[0] == 0:
            return
        if batch.is_decode:
            # A plain decode step's pair is BUFFERED, not committed. Committing it costs a
            # whole eager draft forward per fallback token, and during an adaptive cooldown
            # nothing reads the result before the buffer is flushed anyway.
            self._buffered.append((paired_hidden, paired_embeds, paired_rope))
            return
        # prefill stays eager -- it is once per prompt, not per token -- and the flush comes
        # first because any buffered pair sits at an EARLIER position than this batch's rows
        self._flush_pairs()
        self._commit_pairs(paired_hidden, paired_embeds, paired_rope)

    def commit(self, req: "Req", *, hidden: torch.Tensor, token_ids: Sequence[int]) -> None:
        """Consume a speculative step's accepted rows.

        Row ``i`` of the step read position ``cached_len + i`` and its emitted token occupies
        ``cached_len + i + 1``, so the pairs are ``(hidden[i], embed(token_ids[i]))`` -- the
        same shifted alignment a plain decode step commits, ``n`` at a time. The corrected
        token is NOT the draft token staged at that row, so the embeddings are looked up from
        the emitted ids rather than reused from the forward's own inputs.
        """
        # the rope base below is read off ``committed_len``, which buffered pairs have not
        # advanced yet (in the served order ``propose`` already emptied the buffer)
        self._flush_pairs()
        n = len(token_ids)
        if hidden.shape[0] < n:
            # ``hidden`` is the whole step's ``w`` rows; the accepted prefix is sliced here,
            # because which prefix that is is only known after the scheduler's stop scan.
            raise RuntimeError(
                f"a {n}-token speculative run needs {n} target hidden rows, got "
                f"{hidden.shape[0]}"
            )
        ids = torch.tensor(
            [int(t) for t in token_ids], dtype=torch.int32, device=self.device
        )
        embeds = self.target_model.model.embed_tokens.forward(ids)
        rope = None
        if self._pending_rope is not None or getattr(req, "mrope_position_ids", None) is not None:
            base = self.committed_len + int(getattr(req, "mrope_position_delta", 0))
            rope = (
                torch.arange(base, base + n, dtype=torch.int64, device=self.device)
                .expand(3, -1)
                .contiguous()
            )
        self._commit_pairs(hidden[:n], embeds, rope)

    @property
    def _buffered_rows(self) -> int:
        return sum(int(hidden.shape[0]) for hidden, _, _ in self._buffered)

    def _flush_pairs(self) -> None:
        """Commit the buffered pairs, one call per CONSECUTIVE RUN of like-roped rows.

        Batching them is identical in effect to committing them one at a time. The rows occupy
        the same sequential positions either way, so their K/V, their closing groups'
        compressed keys and the pending ring's final state all match: within one call a group's
        earlier members are read from this call's raw rows rather than from the ring row they
        would have been stored in first, and the ring's keep-mask drops exactly the rows a
        row-at-a-time run would have overwritten. ``_commit_pairs`` already takes
        ``_sample``/``_recursive``/``_saved_blocks`` from the LAST row, which is the only row
        whose outputs survive a row-at-a-time run.

        A buffer can MIX kinds: on a vision boot the decode observation that closes a prompt's
        carried pair takes the prefill's picture coordinates, while the decode steps after it
        carry none. ``rope=None`` and ``rope=<tensor>`` are each valid per call and mean
        different things downstream, so they cannot be concatenated -- and no rope may be
        synthesized for the None rows, because eager committed exactly None there. Splitting at
        the seam costs one extra call and preserves order, which is what keeps the positions
        sequential.
        """
        buffered = self._buffered
        if not buffered:
            return
        self._buffered = []
        runs: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]] = []
        for entry in buffered:
            if not runs or (entry[2] is None) is not (runs[-1][0][2] is None):
                runs.append([])
            runs[-1].append(entry)
        for run in runs:
            self._commit_pairs(
                torch.cat([hidden for hidden, _, _ in run]),
                torch.cat([embeds for _, embeds, _ in run]),
                None
                if run[0][2] is None
                else torch.cat([segment for _, _, segment in run], dim=1),
            )

    def _commit_pairs(self, hidden, embeds, rope) -> None:
        capture: dict[int, torch.Tensor] = {}
        self._sample, self._recursive = self._run_rows(
            embeds, hidden, rope_positions=rope, capture=capture
        )
        # step 0's QSA block selection, frozen for the recursive draft rows
        self._saved_blocks = (
            {slot: blocks[-1:].clone() for slot, blocks in capture.items()} or None
        )

    # ------------------------------------------------------------------------- the proposal

    def propose(self, req: "Req", depth: int) -> DraftProposal:
        """Draft up to ``depth`` tokens, leaving the private KV exactly as it was.

        The recursive rows are written at ``committed_len + i`` and undone by rewinding the
        length; the two pending rings have no epoch tag, so they are restored explicitly.

        THE CONFIDENCE CUT
        ------------------
        With ``conf_cut > 0`` the proposal comes back SHORT (``k < depth``): it is truncated
        before the first token whose raw softmax top-1 probability falls under the bar. That is
        not an error anywhere downstream: ``_prepare_spec_batch`` accepts 1..depth drafts,
        verify width is ``1 + k``, and every width in that range is graph-capturable.

        The cut gates CONTINUING, never STARTING: row 0 is drafted and proposed whatever its
        confidence. A cycle exists to verify at least one draft -- refusing to propose would
        just be a plain decode step taken the expensive way -- and a doubtful row 0 costs only
        the jump from w=1 to w=2, which is the cheapest row a cycle can buy. Every row after it
        is priced against a chain that has already shown it is guessing.

        WHEN THE CUT IS APPLIED (``draft_cut_mode``)
        -------------------------------------------
        The two modes propose exactly the same thing; they differ only in what the chain
        spends getting there, and the flag exists so a live A/B can price them on one boot.

        ``step`` breaks out of the loop the moment a row comes back doubtful, which means
        reading that row's confidence back to the host -- a device synchronization inside the
        chain, once per drafted token. Each one drains the queue: the host cannot enqueue step
        i+1's dozen-odd kernels until step i's have all retired, so a five-step chain pays five
        launch ramps instead of one, and no draft step can ever be graph-captured.

        ``chain`` (the default) runs the FULL depth with every step's confidence written into a
        device buffer, and applies the cut after the single readback below -- truncating to the
        prefix before the first sub-cut row, which is the prefix the breaking loop produced.
        The rows after the cut are computed and thrown away (draft forwards, the cheap side)
        in exchange for the whole chain issuing as one uninterrupted stream.

        Neither mode changes the verify width, which is ``1 + k`` rows of the TARGET and the
        expensive side by an order of magnitude.
        """
        from freetoken.engine.spec_sample import request_filter_params

        # the head is needed now, so the deferred decode-step observations are paid for here,
        # in one batched commit, ahead of every read of ``_sample``/``_recursive``
        self._flush_pairs()
        if self._sample is None or self._sampler is None:
            raise RuntimeError("the draft head has no primed context for this request")
        if not 1 <= depth <= self.depth:
            raise ValueError(f"a proposal drafts 1..{self.depth} tokens, got {depth}")
        # the SAME filter acceptance will divide by -- read off the request the way
        # Sampler.prepare would, not off its raw params (design 5.2's precondition)
        temperature, top_k, top_p = request_filter_params(req.sampling_params)

        sample, recursive = self._sample, self._recursive
        saved = self._saved_blocks
        base_len = self.committed_len
        ring = self.kv_cache._pending_ring.clone()
        position_ring = self.kv_cache._pending_position_ring.clone()
        mrope = getattr(req, "mrope_position_ids", None) is not None
        delta = int(getattr(req, "mrope_position_delta", 0))
        # the drafted ids stay on device for the whole chain: a per-step ``int()`` would
        # sync the stream once per token, and only the returned tuple needs host ints
        drafted: list[torch.Tensor] = []
        rows: list[torch.Tensor] = []
        cut = self.conf_cut
        stepwise = cut > 0.0 and self.draft_cut_mode == "step"
        # the cut's evidence, accumulated on the DEVICE: one scalar store per step, no sync.
        # ``step`` mode does not want it -- it has already decided, row by row, on the host.
        conf = self._confidence_buffer(depth) if cut > 0.0 and not stepwise else None
        try:
            for index in range(depth):
                logits = self.draft_lm_head.forward_all(sample)[0]
                token = self._sampler.sample_device(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                rows.append(logits.detach().clone())
                drafted.append(token)
                if conf is not None:
                    _record_row_top1(logits, conf, index)
                if index + 1 == depth:
                    break
                # ``step``: the readback is taken only where it can save a draft forward --
                # after the LAST token there is nothing left to stop
                if stepwise and _row_top1(logits) < cut:
                    break
                embedding = self.target_model.model.embed_tokens.forward(
                    token.reshape(1).to(torch.int32)
                )
                self.committed_len = base_len + index
                rope = (
                    torch.full(
                        (3, 1),
                        self.committed_len + delta,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    if mrope
                    else None
                )
                sample, recursive = self._run_rows(
                    embedding, recursive, rope_positions=rope, saved=saved
                )
        finally:
            self.kv_cache._pending_ring.copy_(ring)
            self.kv_cache._pending_position_ring.copy_(position_ring)
            self.committed_len = base_len
        # THE one readback of the drafted ids -- and, in ``chain`` mode, the only point at
        # which the host waits for the device at all. The ids and the cut's confidences travel
        # together in one float64 tensor rather than in two calls: a float64 holds an int64
        # token id exactly to 2^53, so the pack is free and ``chain`` synchronizes exactly once
        # whether the cut is armed or not. In ``step`` mode the loop already truncated itself,
        # so there is nothing left to decide and no confidence to carry.
        packed = [token.reshape(1).to(torch.float64) for token in drafted]
        if conf is not None:
            packed.append(conf.to(torch.float64))
        values = torch.cat(packed).tolist()
        drawn = len(drafted)
        keep = drawn if conf is None else _confidence_prefix(values[drawn:], cut)
        stacked = torch.stack(rows[:keep])
        top1, gap = _draft_confidence(stacked)
        return DraftProposal(
            tokens=tuple(int(value) for value in values[:keep]),
            logits=stacked,
            draft_top1=top1,
            draft_top1_gap=gap,
        )

    def _confidence_buffer(self, depth: int) -> torch.Tensor:
        """The chain's ``[depth]`` device slots for per-step top-1 probabilities.

        Preallocated at the configured ceiling and reused, so an armed cut adds no allocation
        to a proposal -- only ``depth`` scalar stores and one slice.
        """
        buffer = getattr(self, "_conf_buffer", None)
        if buffer is None or buffer.numel() < depth:
            buffer = self._conf_buffer = torch.zeros(
                max(depth, self.depth), dtype=torch.float32, device=self.device
            )
        return buffer[:depth]

    def close(self) -> None:
        runner = getattr(self, "expert_runner", None)
        if runner is not None:
            runner.close()


def _row_top1(logits: torch.Tensor) -> float:
    """ONE drafted row's raw softmax top-1 probability, as a host float.

    The per-step form of ``_draft_confidence``'s ``top1`` -- same detach, same float32 softmax,
    same maximum -- so the value a cut decides on and the value the diagnosis log records for
    that row are the same number. Raw, not filtered: under a greedy request's filter ``q`` is
    1.0 for every draft, sure or not, which is exactly the signal the cut needs to keep.

    The readback ``draft_cut_mode="step"`` takes per drafted token, and the reference the
    diagnosis log is compared against. ``chain`` mode writes the same number straight to the
    device instead (:func:`_record_row_top1`) so that no step has to wait for it.

    Deliberately a module-level function rather than an inlined expression: it is the single
    place ``step`` mode's readback happens, so a test can count it.
    """
    return float(torch.softmax(logits.detach().float(), dim=-1).max().item())


def _record_row_top1(logits: torch.Tensor, out: torch.Tensor, index: int) -> None:
    """Write one row's raw softmax top-1 probability into a device slot -- no readback.

    Same expression as :func:`_row_top1` minus the ``.item()``: the store is a device-to-device
    copy into a preallocated buffer, so the chain never stops to look at it. Module level for
    the same reason ``_row_top1`` is -- it is the single place the confidence is produced, so a
    test can count the productions independently of the readbacks.
    """
    out[index] = torch.softmax(logits.detach().float(), dim=-1).max()


def _confidence_prefix(top1: Sequence[float], cut: float) -> int:
    """How many of a full-depth chain's drafts survive the cut.

    Row 0 is always kept (the cut gates continuing, never starting) and the LAST row's
    confidence is never consulted -- after the final draft there is no further row for it to
    stop. Both are the pre-existing semantics; only the moment the decision is taken moved.
    """
    keep = 1
    while keep < len(top1) and top1[keep - 1] >= cut:
        keep += 1
    return keep


def _draft_confidence(
    logits: torch.Tensor,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    """Per drafted token: the raw softmax top-1 probability and the top1-top2 margin.

    ``(None, None)`` unless the confidence log is armed, and then nothing is computed at all --
    the proposal path must stay byte for byte what it was when the flag is unset. Armed, this
    is one softmax and one width-2 topk over rows ``propose`` already holds, plus the single
    readback that turns them into host floats (a sync, which the flag licenses).
    """
    if not spec_conf_log_enabled():
        return None, None
    probabilities = torch.softmax(logits.detach().float(), dim=-1)
    width = min(2, probabilities.shape[-1])
    best = probabilities.topk(width, dim=-1).values
    top1 = best[:, 0]
    runner_up = best[:, 1] if width == 2 else torch.zeros_like(top1)
    # clamped because a float32 softmax can put the two within an ulp of each other
    margin = (top1 - runner_up).clamp_min(0.0)
    return (
        tuple(float(v) for v in top1.tolist()),
        tuple(float(v) for v in margin.tolist()),
    )


def _draft_seed(seed: int, uid: int, *, purpose: int = 1) -> int:
    """The shadow observer's private-stream seed shape, so a request replays identically."""
    modulus = (1 << 63) - 1
    return int((int(seed) + int(uid) * 10_000_019 + purpose * 1_000_003) % modulus)


__all__ = [
    "DraftProposal",
    "SpecDraftHead",
    "build_shifted_pairs",
    "build_shifted_rope_positions",
    "load_spec_expert_banks",
    "resolve_draft_cut_mode",
    "resolve_spec_expert_placement",
    "spec_conf_log_enabled",
    "spec_expert_runner_type",
]

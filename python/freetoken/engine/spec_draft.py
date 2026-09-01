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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch, Req

# The GPU expert runner's per-call row cap; priming chunks are split to it.
_MAX_DRAFT_ROWS = 128

_EXPERT_FORMAT_ENV = "FREETOKEN_MTP_SPEC_EXPERT_FORMAT"
_NVFP4_MANIFEST_ENV = "FREETOKEN_MTP_SPEC_NVFP4_MANIFEST"


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
    """

    tokens: tuple[int, ...]
    logits: torch.Tensor  # [k, vocab]


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
        from freetoken.engine.spec_sample import resolve_spec_seed
        from freetoken.models.qwen4_exp.mtp_spike import (
            Qwen4ExpMTPModel,
            derive_mtp_model_config,
        )
        from freetoken.utils.torch_utils import torch_dtype

        self.engine = engine
        self.device = engine.device
        self.depth = int(spec_decode.depth)
        self.seed = resolve_spec_seed() if seed is None else int(seed)
        self.target_ctx = engine.ctx
        self.target_model = engine.model
        self.mtp_config = derive_mtp_model_config(engine.config.model_config)

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
            {}
            if self.expert_placement == "bf16"
            # the quantized gather's scratch is resident, so it is sized for the widest
            # per-cycle call -- the accepted run, ``1 + depth`` rows -- and nothing wider
            else {"max_gather_tokens": self.depth + 1}
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

    @property
    def resident_bytes(self) -> int:
        """Everything the head charges the card: dense weights, both expert banks, its KV."""
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
        return (
            int(self.staged_model.resident_bytes)
            + int(self.expert_runner.resident_bytes)
            + int(pool)
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

    def _batch(self, rows: int, *, rope_positions=None, capture=None, saved=None):
        from types import SimpleNamespace

        start = self.committed_len
        req = SimpleNamespace(
            table_idx=0, cached_len=start, device_len=start + rows, extend_len=rows
        )
        positions = torch.arange(
            start, start + rows, dtype=torch.int32, device=self.device
        )
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
            out_loc=self.page_table[0, start : start + rows].contiguous(),
            attn_metadata=None,
            active_table_idx=torch.zeros(1, dtype=torch.int32, device=self.device),
        )
        if capture is not None:
            batch.mtp_qsa_capture_blocks = capture
        if saved is not None:
            batch.mtp_qsa_saved_blocks = saved
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
        """
        return (
            self._uid == req.uid
            and self._sample is not None
            and self._pending_hidden is None
            and self.committed_len == req.cached_len
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
        self._commit_pairs(paired_hidden, paired_embeds, paired_rope)

    def commit(self, req: "Req", *, hidden: torch.Tensor, token_ids: Sequence[int]) -> None:
        """Consume a speculative step's accepted rows.

        Row ``i`` of the step read position ``cached_len + i`` and its emitted token occupies
        ``cached_len + i + 1``, so the pairs are ``(hidden[i], embed(token_ids[i]))`` -- the
        same shifted alignment a plain decode step commits, ``n`` at a time. The corrected
        token is NOT the draft token staged at that row, so the embeddings are looked up from
        the emitted ids rather than reused from the forward's own inputs.
        """
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
        """Draft ``depth`` tokens, leaving the private KV exactly as it was.

        The recursive rows are written at ``committed_len + i`` and undone by rewinding the
        length; the two pending rings have no epoch tag, so they are restored explicitly.
        """
        from freetoken.engine.spec_sample import request_filter_params

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
        tokens: list[int] = []
        rows: list[torch.Tensor] = []
        try:
            for index in range(depth):
                logits = self.target_model.lm_head.forward_all(sample)[0]
                token = self._sampler.sample(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                rows.append(logits.detach().clone())
                tokens.append(int(token))
                if index + 1 == depth:
                    break
                embedding = self.target_model.model.embed_tokens.forward(
                    torch.tensor([token], dtype=torch.int32, device=self.device)
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
        return DraftProposal(tokens=tuple(tokens), logits=torch.stack(rows))

    def close(self) -> None:
        runner = getattr(self, "expert_runner", None)
        if runner is not None:
            runner.close()


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
    "resolve_spec_expert_placement",
    "spec_expert_runner_type",
]

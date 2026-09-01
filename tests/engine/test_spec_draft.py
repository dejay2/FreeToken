"""The standalone draft head: its running context, its proposals, and its RNG.

``SpecDraftHead`` is the draft half of the shadow observer with the observer removed -- the
same ``Qwen4ExpMTPModel`` under the same derived config, the same resident staged runner and
GPU expert runner, the same private one-layer QSA state. Its construction needs the real
checkpoint and a GPU, so what is pinned here is everything ABOVE the weights: the shifted-pair
bookkeeping that decides which rows the head attends, the proposal loop's undo, and the
distribution the proposal is drawn from.

Only Phase 6 can prove the head's weights load and its logits are the real MTP head's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Context, Req, SamplingParams, set_global_ctx
from freetoken.engine.sample import BatchSamplingArgs
from freetoken.engine.spec_draft import DraftProposal, SpecDraftHead, _draft_seed
from freetoken.engine.spec_sample import filtered_probs
from freetoken.models.qwen4_exp.mtp_spike import MTPDraftSampler

CPU = torch.device("cpu")
VOCAB = 32
HIDDEN = 4
WIDTH = 4  # hc_count * hidden of the fake head
RING = 8
PAGES = 4
PAGE_SIZE = 64


@pytest.fixture(autouse=True)
def _no_ctx_leak():
    import freetoken.core as core

    yield
    core._GLOBAL_CTX = None


# ------------------------------------------------------------------------------- the fakes


class _FakeKVCache:
    def __init__(self):
        self._kv_buffer = torch.zeros(2, 2)
        self._cmp_k_buffer = torch.zeros(2, 2)
        self._pending_ring = torch.zeros(1, 1, RING, 2)
        self._pending_position_ring = torch.zeros(1, 1, RING, 3, dtype=torch.int64)


class _FakeStaged:
    """Records the rows it is asked to commit and scribbles on the ring, as QSA would."""

    def __init__(self, kv_cache):
        self.kv_cache = kv_cache
        self.calls: list[tuple[int, int]] = []  # (rows, cached_len)

    def forward(self, embeds, hidden, batch):
        rows = embeds.shape[0]
        self.calls.append((rows, batch.reqs[0].cached_len))
        start = batch.reqs[0].cached_len
        for i in range(rows):
            self.kv_cache._pending_ring[0, 0, (start + i) % RING] = float(start + i + 1)
        capture = getattr(batch, "mtp_qsa_capture_blocks", None)
        if isinstance(capture, dict):
            # QSA's shape: one selection row per committed row, of which _commit_pairs keeps
            # the last -- the row a one-at-a-time run would have left behind
            capture[0] = torch.arange(start, start + rows, dtype=torch.int32).reshape(rows, 1)
        sample = (embeds.sum(-1, keepdim=True) + hidden.sum(-1, keepdim=True)).expand(
            rows, HIDDEN
        ).contiguous()
        recursive = sample.repeat(1, WIDTH // HIDDEN)
        return sample, recursive


class _FakeTargetModel:
    def __init__(self, *, logits_for=None):
        self.logits_for = logits_for or (lambda h: torch.arange(VOCAB, dtype=torch.float32))
        self.model = SimpleNamespace(
            embed_tokens=SimpleNamespace(
                forward=lambda ids: torch.stack(
                    [torch.full((HIDDEN,), float(i) + 0.5) for i in ids.tolist()]
                )
            )
        )
        self.lm_head = SimpleNamespace(
            forward_all=lambda hidden: torch.stack(
                [self.logits_for(row) for row in hidden]
            )
        )


def _head(*, depth=3, seed=1729, logits_for=None) -> SpecDraftHead:
    import freetoken.core as core

    core._GLOBAL_CTX = None
    ctx = Context(page_size=PAGE_SIZE)
    ctx.page_table = torch.zeros((2, PAGES * PAGE_SIZE), dtype=torch.int32)
    ctx.kv_cache = None
    ctx.attn_backend = None
    set_global_ctx(ctx)

    head = object.__new__(SpecDraftHead)
    head.device = CPU
    head.depth = depth
    head.seed = seed
    head.num_pages = PAGES
    head.target_ctx = ctx
    head.target_model = _FakeTargetModel(logits_for=logits_for)
    head.page_table = torch.arange(
        2 * PAGES * PAGE_SIZE, dtype=torch.int32
    ).reshape(2, PAGES * PAGE_SIZE)
    head.kv_cache = _FakeKVCache()
    head.prepared: list = []
    head.attn_backend = SimpleNamespace(prepare_metadata=head.prepared.append)
    head.staged_model = _FakeStaged(head.kv_cache)
    head._uid = None
    head.committed_len = 0
    head._pending_hidden = None
    head._pending_rope = None
    head._sample = None
    head._recursive = None
    head._saved_blocks = None
    head._buffered = []
    head._sampler = None
    return head


def _req(uid=1, *, cached_len=0, temperature=0.0, top_k=-1, top_p=1.0) -> Req:
    req = Req(
        input_ids=torch.arange(1, 6, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=uid,
        sampling_params=SamplingParams(
            temperature=temperature, top_k=top_k, top_p=top_p, max_tokens=8
        ),
        cache_handle=None,
    )
    req.cached_len = cached_len
    req.device_len = cached_len + 1
    return req


def _capture(rows: int, *, base: float = 0.0):
    """``(logits, multi_stream, inputs_embeds)`` -- the engine's capture triple."""
    hidden = torch.stack(
        [torch.full((WIDTH,), base + float(i) + 1.0) for i in range(rows)]
    )
    embeds = torch.stack(
        [torch.full((HIDDEN,), base + float(i) + 10.0) for i in range(rows)]
    )
    return None, hidden, embeds


def _prefill_batch(rows: int, *, chunked: bool = False, uid: int = 1, rope=None):
    cls = type("ChunkedReq", (Req,), {}) if chunked else Req
    req = cls(
        input_ids=torch.arange(1, rows + 1, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=8),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = req.input_ids
    batch.rope_positions = rope
    return batch, req


def _decode_batch(*, cached_len: int, uid: int = 1, rope=None):
    req = Req(
        input_ids=torch.arange(1, cached_len + 2, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=8),
        cache_handle=None,
    )
    req.cached_len = cached_len
    req.device_len = cached_len + 1
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.input_ids = req.input_ids[-1:]
    batch.rope_positions = rope
    return batch, req


def _decode_run(head, steps: int, *, start: int, eager: bool, rope_base: int | None = None):
    """``steps`` fallback decode observations; ``eager`` flushes after each, as the head used
    to commit."""
    for step in range(steps):
        rope = (
            None
            if rope_base is None
            else torch.full((3, 1), rope_base + step, dtype=torch.int64)
        )
        batch, _ = _decode_batch(cached_len=start + step, rope=rope)
        head.observe_forward(batch, _capture(1, base=100.0 * (step + 1)), torch.tensor(step + 2))
        if eager:
            head._flush_pairs()


# ---------------------------------------------------------------------- the running context


def test_a_prompt_chunk_commits_its_interior_pairs_and_carries_its_last_row():
    head = _head()
    batch, _ = _prefill_batch(5, chunked=True)
    head.observe_forward(batch, _capture(5), torch.tensor(7))

    # rows 0..3 pair with embeddings 1..4; row 4 has no following token yet
    assert head.staged_model.calls == [(4, 0)]
    assert head.committed_len == 4
    assert head._pending_hidden is not None


def test_the_final_chunk_closes_the_carried_pair_with_the_sampled_token():
    head = _head()
    first, _ = _prefill_batch(5, chunked=True)
    head.observe_forward(first, _capture(5), torch.tensor(7))
    second, _ = _prefill_batch(3)
    head.observe_forward(second, _capture(3), torch.tensor(7))

    # the carried row + rows 0..1 interior + the last row closed by the sampled token
    assert head.staged_model.calls == [(4, 0), (4, 4)]
    assert head.committed_len == 8
    assert head._pending_hidden is None


def test_a_committed_length_that_tracks_the_targets_cached_length_is_what_ready_means():
    head = _head()
    batch, req = _prefill_batch(5)
    head.observe_forward(batch, _capture(5), torch.tensor(7))
    assert head.committed_len == 5

    req.cached_len = 5
    assert head.is_ready(req) is True
    req.cached_len = 6  # the target moved on without the head
    assert head.is_ready(req) is False


def test_an_unprimed_or_foreign_request_is_never_ready():
    head = _head()
    assert head.is_ready(_req(uid=1)) is False
    batch, _ = _prefill_batch(5, uid=1)
    head.observe_forward(batch, _capture(5), torch.tensor(7))
    assert head.is_ready(_req(uid=1, cached_len=5)) is True
    assert head.is_ready(_req(uid=2, cached_len=5)) is False


def test_a_mid_chunk_head_is_not_ready():
    head = _head()
    batch, _ = _prefill_batch(5, chunked=True)
    head.observe_forward(batch, _capture(5), torch.tensor(7))
    assert head.is_ready(_req(cached_len=4)) is False  # a pair is still open


def test_a_new_request_resets_the_context():
    head = _head()
    first, _ = _prefill_batch(5, uid=1)
    head.observe_forward(first, _capture(5), torch.tensor(7))
    second, _ = _prefill_batch(3, uid=2)
    head.observe_forward(second, _capture(3), torch.tensor(7))
    assert head.committed_len == 3  # not 8: the second request started from zero
    assert head._uid == 2


# ------------------------------------------------------------- lazy decode observation


def _hidden_sensitive_logits(hidden: torch.Tensor) -> torch.Tensor:
    """A distribution the committed rows actually move, so equal proposals mean equal state."""
    return torch.sin(hidden.sum() * 0.37 + torch.arange(VOCAB, dtype=torch.float32)) * 3.0


def _primed(*, eager: bool, steps: int = 4) -> SpecDraftHead:
    head = _head(logits_for=_hidden_sensitive_logits)
    batch, _ = _prefill_batch(5)
    head.observe_forward(batch, _capture(5), torch.tensor(7))
    _decode_run(head, steps, start=5, eager=eager)
    return head


def test_buffered_decode_observations_land_the_state_committing_them_one_at_a_time_does():
    """The whole point of deferring: K buffered pairs flushed as one K-row commit leave the
    head exactly where K one-row commits left it -- same private KV, same sampled proposal."""
    eager, lazy = _primed(eager=True), _primed(eager=False)

    assert eager.committed_len == 9 and eager.staged_model.calls[1:] == [
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
    ]
    assert lazy.committed_len == 5 and lazy._buffered_rows == 4

    before_flush = lazy.kv_cache._pending_ring.clone()
    first = eager.propose(_req(cached_len=9, temperature=1.0), 3)
    second = lazy.propose(_req(cached_len=9, temperature=1.0), 3)

    assert lazy.staged_model.calls[1] == (4, 5)  # ONE four-row commit, not four one-row ones
    assert not torch.equal(lazy.kv_cache._pending_ring, before_flush)
    assert first.tokens == second.tokens
    assert torch.equal(first.logits, second.logits)
    assert eager.committed_len == lazy.committed_len == 9
    assert lazy._buffered_rows == 0
    assert torch.equal(eager._sample, lazy._sample)
    assert torch.equal(eager._recursive, lazy._recursive)
    assert torch.equal(eager._saved_blocks[0], lazy._saved_blocks[0])
    assert torch.equal(eager.kv_cache._pending_ring, lazy.kv_cache._pending_ring)
    assert torch.equal(
        eager.kv_cache._pending_position_ring, lazy.kv_cache._pending_position_ring
    )


def test_a_buffer_mixing_roped_and_ropeless_rows_flushes_in_consecutive_runs():
    """A vision boot mixes the two kinds inside one buffer: the decode observation that closes
    a prompt's carried pair takes picture coordinates, the steps after it carry none.
    ``rope=None`` and ``rope=<tensor>`` are each valid per commit and mean different things
    downstream, so the flush splits at the seam instead of concatenating across it."""
    ropes = [torch.full((3, 1), 5, dtype=torch.int64), None, None]
    heads = {}
    for eager in (True, False):
        head = _head(logits_for=_hidden_sensitive_logits)
        prompt, _ = _prefill_batch(5)
        head.observe_forward(prompt, _capture(5), torch.tensor(7))
        head.staged_model.calls.clear()
        head.prepared.clear()
        for step, rope in enumerate(ropes):
            decode, _ = _decode_batch(cached_len=5 + step, rope=rope)
            head.observe_forward(
                decode, _capture(1, base=100.0 * (step + 1)), torch.tensor(step + 2)
            )
            if eager:
                head._flush_pairs()
        head._flush_pairs()
        heads[eager] = head

    eager, lazy = heads[True], heads[False]
    assert eager.staged_model.calls == [(1, 5), (1, 6), (1, 7)]
    # the seam costs ONE extra call; the two ropeless rows still share a commit
    assert lazy.staged_model.calls == [(1, 5), (2, 6)]
    assert torch.equal(lazy.prepared[0].rope_positions, ropes[0])
    assert lazy.prepared[1].rope_positions is None  # never synthesized for the None rows

    assert eager.committed_len == lazy.committed_len == 8
    assert torch.equal(eager._sample, lazy._sample)
    assert torch.equal(eager._recursive, lazy._recursive)
    assert torch.equal(eager._saved_blocks[0], lazy._saved_blocks[0])
    assert torch.equal(eager.kv_cache._pending_ring, lazy.kv_cache._pending_ring)

    first = eager.propose(_req(cached_len=8, temperature=1.0), 3)
    second = lazy.propose(_req(cached_len=8, temperature=1.0), 3)
    assert first.tokens == second.tokens
    assert torch.equal(first.logits, second.logits)


def test_a_buffered_pair_counts_as_consumed_for_readiness():
    head = _primed(eager=False, steps=3)
    assert head.committed_len == 5
    assert head.is_ready(_req(cached_len=8)) is True
    assert head.is_ready(_req(cached_len=5)) is False  # the buffer is context, not a gap
    assert head.is_ready(_req(cached_len=9)) is False
    assert head.is_ready(_req(uid=2, cached_len=8)) is False


def test_readiness_does_not_wait_for_a_flush_to_write_the_sample():
    """``_sample`` is written only by a flush, and ``is_ready`` runs every scheduler iteration
    and must not flush; a head holding only buffered pairs is nonetheless primed."""
    head = _head()
    _decode_run(head, 1, start=0, eager=False)
    assert head._sample is None
    assert head.is_ready(_req(cached_len=1)) is True
    head._flush_pairs()
    assert head._sample is not None
    assert head.is_ready(_req(cached_len=1)) is True


def test_buffered_mrope_segments_flush_to_the_positions_the_eager_commits_saw():
    """The picture coordinates are the target's own rows, not derived from ``committed_len``,
    so concatenating buffered segments must reproduce them column for column."""
    prepared = {}
    for eager in (True, False):
        head = _head()
        rope = torch.arange(5, dtype=torch.int64).expand(3, -1).contiguous()
        batch, _ = _prefill_batch(5, rope=rope)
        head.observe_forward(batch, _capture(5), torch.tensor(7))
        head.prepared.clear()
        _decode_run(head, 4, start=5, eager=eager, rope_base=5)
        head._flush_pairs()
        prepared[eager] = torch.cat([b.rope_positions for b in head.prepared], dim=1)

    assert torch.equal(prepared[True], prepared[False])
    assert torch.equal(prepared[True], torch.arange(5, 9, dtype=torch.int64).expand(3, -1))


def test_a_reset_drops_the_buffer():
    head = _primed(eager=False, steps=3)
    assert head._buffered_rows == 3
    head.reset_request(2)
    assert head._buffered_rows == 0
    assert head.committed_len == 0


def test_a_prefill_observation_commits_the_buffer_before_its_own_rows():
    head = _primed(eager=False, steps=2)
    head.staged_model.calls.clear()
    batch, _ = _prefill_batch(3, uid=1)
    head.observe_forward(batch, _capture(3), torch.tensor(7))
    assert head.staged_model.calls == [(2, 5), (3, 7)]


def test_a_cycle_commit_pays_for_the_buffer_first():
    """``commit`` reads its rope base off ``committed_len``, which buffered pairs have not
    advanced."""
    head = _primed(eager=False, steps=2)
    head.staged_model.calls.clear()
    head.commit(_req(cached_len=7), hidden=torch.zeros(2, WIDTH), token_ids=(11, 12))
    assert head.staged_model.calls == [(2, 5), (2, 7)]


# ---------------------------------------------------------------------------- the spec feed


def test_a_cycle_commits_one_pair_per_emitted_token():
    head = _head()
    batch, _ = _prefill_batch(5)
    head.observe_forward(batch, _capture(5), torch.tensor(7))
    head.staged_model.calls.clear()

    hidden = torch.stack([torch.full((WIDTH,), float(i)) for i in range(4)])
    head.commit(_req(cached_len=5), hidden=hidden, token_ids=(11, 12))

    assert head.staged_model.calls == [(2, 5)]
    assert head.committed_len == 7


def test_the_cycle_embeds_the_EMITTED_ids_not_the_drafts_the_forward_read():
    """Row j's token is the CORRECTION, which is not the draft staged at that row -- so the
    pair's embedding has to come from the emitted id, never from the forward's own inputs."""
    head = _head()
    batch, _ = _prefill_batch(5)
    head.observe_forward(batch, _capture(5), torch.tensor(7))
    seen: list[list[int]] = []
    head.target_model.model.embed_tokens.forward = lambda ids: (
        seen.append(ids.tolist())
        or torch.zeros(ids.numel(), HIDDEN)
    )

    hidden = torch.zeros(4, WIDTH)
    head.commit(_req(cached_len=5), hidden=hidden, token_ids=(11, 12, 13))
    assert seen == [[11, 12, 13]]


def test_a_run_longer_than_its_hidden_rows_is_refused():
    head = _head()
    batch, _ = _prefill_batch(5)
    head.observe_forward(batch, _capture(5), torch.tensor(7))
    with pytest.raises(RuntimeError, match="hidden rows"):
        head.commit(_req(cached_len=5), hidden=torch.zeros(1, WIDTH), token_ids=(11, 12))


# ------------------------------------------------------------------------------ proposals


def test_a_proposal_returns_full_logit_rows_for_every_draft():
    head = _head()
    batch, _ = _prefill_batch(5)
    head.observe_forward(batch, _capture(5), torch.tensor(7))

    proposal = head.propose(_req(cached_len=5), 3)
    assert isinstance(proposal, DraftProposal)
    assert len(proposal.tokens) == 3
    assert proposal.logits.shape == (3, VOCAB)  # rejection sampling needs all of q


def test_a_proposal_leaves_the_private_kv_exactly_as_it_found_it():
    """The recursive draft rows are speculative in the DRAFT's own context too: the length
    rewind unmakes the K/V (identity page table -> the next real row overwrites the slot), but
    the pending rings have no epoch tag and must be restored explicitly."""
    head = _head()
    batch, _ = _prefill_batch(5)
    head.observe_forward(batch, _capture(5), torch.tensor(7))

    before_len = head.committed_len
    before_ring = head.kv_cache._pending_ring.clone()
    before_positions = head.kv_cache._pending_position_ring.clone()

    head.propose(_req(cached_len=before_len), 3)

    assert head.committed_len == before_len
    assert torch.equal(head.kv_cache._pending_ring, before_ring)
    assert torch.equal(head.kv_cache._pending_position_ring, before_positions)


def test_a_proposal_writes_its_recursive_rows_at_the_committed_length():
    head = _head()
    batch, _ = _prefill_batch(5)
    head.observe_forward(batch, _capture(5), torch.tensor(7))
    head.staged_model.calls.clear()

    head.propose(_req(cached_len=5), 3)
    # depth 3 = three lm-head samples but only two recursive rows, both at the committed tail
    assert head.staged_model.calls == [(1, 5), (1, 6)]


def test_a_proposal_deeper_than_the_configured_depth_is_refused():
    head = _head(depth=2)
    batch, _ = _prefill_batch(5)
    head.observe_forward(batch, _capture(5), torch.tensor(7))
    with pytest.raises(ValueError, match="drafts 1..2"):
        head.propose(_req(cached_len=5), 3)


def test_an_unprimed_head_refuses_to_propose():
    head = _head()
    with pytest.raises(RuntimeError, match="primed"):
        head.propose(_req(), 3)


# --------------------------------------------------------------- the distribution contract


@pytest.mark.parametrize(
    "temperature,top_k,top_p",
    [(0.0, -1, 1.0), (1.0, -1, 1.0), (0.7, 5, 1.0), (1.0, -1, 0.8), (0.5, 8, 0.9)],
)
def test_the_draft_samples_from_the_same_filtered_distribution_acceptance_divides_by(
    temperature, top_k, top_p
):
    """Phase 4's theorem needs the proposal drawn from ``q = filtered_probs(draft_logits)``.
    ``MTPDraftSampler`` is that filter's one-row form; pinned empirically, not by reading."""
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(VOCAB, generator=generator) * 3
    args = BatchSamplingArgs(
        temperatures=None if temperature <= 0 else torch.tensor([temperature]),
        top_k=None if top_k < 1 else torch.tensor([top_k]),
        top_p=None if top_p >= 1 else torch.tensor([top_p]),
    )
    expected = filtered_probs(logits.unsqueeze(0), args)[0]

    sampler = MTPDraftSampler(seed=11, device=CPU)
    counts = torch.zeros(VOCAB)
    draws = 4000
    for _ in range(draws):
        counts[sampler.sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)] += 1
    empirical = counts / draws

    # never proposes a token the server's filter zeroed (q = 0 there, and acceptance would
    # divide by it), and matches the surviving mass
    assert not bool(((empirical > 0) & (expected == 0)).any())
    assert float((empirical - expected).abs().max()) < 0.05


def test_the_draft_stream_is_per_request_and_replays():
    head = _head()
    head.reset_request(5)
    first = head._sampler.generator.get_state().clone()
    head.reset_request(5)
    assert torch.equal(head._sampler.generator.get_state(), first)
    head.reset_request(6)
    assert not torch.equal(head._sampler.generator.get_state(), first)


def test_the_draft_seed_keeps_the_observers_private_stream_shape():
    assert _draft_seed(1729, 0) == 1729 + 1_000_003
    assert _draft_seed(1729, 3) == 1729 + 3 * 10_000_019 + 1_000_003


@pytest.mark.parametrize(
    "temperature,top_k,top_p",
    [
        (0.0, -1, 1.0),      # greedy
        (0.0, -1, 0.8),      # NOT greedy: the server floors T at 1e-6 and SAMPLES this
        (0.0, 1, 1.0),       # greedy by top_k
        (1.0, -1, 1.0),
        (0.7, 5, 1.0),
        (1.0, -1, 0.8),
        (0.5, 8, 0.9),
        (0.5, 0, 0.5),       # top_k 0 means "no top-k", not "keep nothing"
    ],
)
def test_the_drafts_filter_is_the_one_the_server_would_have_prepared(
    temperature, top_k, top_p
):
    """The draft picks its filter before the speculative batch exists, so it reads the request
    instead of the batch's ``BatchSamplingArgs``. Reading the RAW params would disagree
    wherever ``Sampler.prepare`` rewrites them -- pinned here against the real sampler."""
    from freetoken.engine.sample import Sampler
    from freetoken.engine.spec_sample import request_filter_params, spec_filter_params
    from freetoken.engine.spec_sample import _sampling_probabilities_batch

    params = SamplingParams(
        temperature=temperature, top_k=top_k, top_p=top_p, max_tokens=4
    )
    req = Req(
        input_ids=torch.arange(4, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=4,
        uid=1,
        sampling_params=params,
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    prepared = spec_filter_params(Sampler(device=CPU, vocab_size=VOCAB).prepare(batch))

    generator = torch.Generator().manual_seed(5)
    logits = (torch.randn(VOCAB, generator=generator) * 3).unsqueeze(0)
    expected = _sampling_probabilities_batch(
        logits, temperature=prepared[0], top_k=prepared[1], top_p=prepared[2]
    )
    drafted = request_filter_params(params)
    actual = _sampling_probabilities_batch(
        logits, temperature=drafted[0], top_k=drafted[1], top_p=drafted[2]
    )
    assert torch.equal(actual, expected)

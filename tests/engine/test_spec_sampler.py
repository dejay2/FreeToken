"""The integrated speculative sampler -- design section 5.2, phase 4.

One speculative step forwards ``w = 1 + k`` rows. Row ``i`` consumes the token at position
``cached_len + i`` and holds the target's distribution for position ``cached_len + i + 1``.
``SpecSampler.step`` turns those ``w`` rows plus the ``k`` drafts into the run the request
actually emits, and into the row count ``Scheduler._rollback_spec_tokens`` settles on.

THE contract, and the one thing a phase-5 wire-up can get wrong silently:

    j accepted drafts  ->  emitted run = drafts[:j] + [bonus sampled from row j]
                       ->  len(run) == j + 1 == accepted_rows

``accepted_rows`` is not ``j``. ``_rollback_spec_tokens`` sets ``cached_len += accepted``, and
row ``j``'s bonus token occupies a position of its own, so a step that accepts nothing still
keeps one row. Passing ``j`` would drop a token from the KV every cycle.

The distribution contract (what the acceptance filter must equal) lives in
``freetoken.engine.sample.sample_impl``; ``tests/engine/test_spec_filter_backend.py`` pins the
equality against the installed backend on real hardware. Here it is pinned against
``_sampling_probabilities_batch``, the filter acceptance itself uses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine.sample import BatchSamplingArgs
from freetoken.engine.spec_sample import (
    SpecSampler,
    default_rng_guard,
    filtered_probs,
    resolve_spec_seed,
    spec_filter_params,
)

CPU = torch.device("cpu")
VOCAB = 8


def _args(temperature=1.0, top_k=None, top_p=None):
    """A ``BatchSamplingArgs`` shaped exactly as ``Sampler.prepare`` builds it for one request:
    ``temperatures=None`` is the whole-batch greedy fast path, and an absent top-k/top-p tensor
    means the filter is off (every request wanted the full vocabulary)."""
    if temperature is None:
        return BatchSamplingArgs(temperatures=None)
    return BatchSamplingArgs(
        temperatures=torch.tensor([temperature], dtype=torch.float32),
        top_k=None if top_k is None else torch.tensor([top_k], dtype=torch.int32),
        top_p=None if top_p is None else torch.tensor([top_p], dtype=torch.float32),
    )


def _peaked(winners, vocab=VOCAB):
    """Logit rows whose argmax is unambiguous."""
    rows = torch.full((len(winners), vocab), -20.0)
    for row, winner in enumerate(winners):
        rows[row, winner] = 20.0
    return rows


def _rows(probabilities):
    return torch.stack(
        [torch.tensor(row, dtype=torch.float32).log() for row in probabilities]
    )


def _sampler(depth=3, seed=1729, **kwargs):
    return SpecSampler(device=CPU, depth=depth, seed=seed, **kwargs)


# ------------------------------------------------- the emitted-run / accepted-rows contract


@pytest.mark.parametrize("accepted_drafts", [0, 1, 2, 3])
def test_the_run_is_the_accepted_drafts_plus_one_bonus_token(accepted_drafts):
    drafts = [1, 2, 3]
    # the target agrees with the drafts up to `accepted_drafts`, then wants 7
    winners = drafts[:accepted_drafts] + [7] * (4 - accepted_drafts)
    decision = _sampler().step(
        uid=1,
        draft_tokens=drafts,
        draft_logits=_peaked(drafts),
        target_logits=_peaked(winners),
        args=_args(temperature=None),
    )

    assert decision.accepted_drafts == accepted_drafts
    assert decision.tokens == tuple(drafts[:accepted_drafts]) + (7,)
    assert decision.accepted_rows == accepted_drafts + 1 == len(decision.tokens)
    assert decision.bonus_token == 7


def test_a_fully_rejected_step_still_emits_one_token():
    decision = _sampler().step(
        uid=1,
        draft_tokens=[1, 2, 3],
        draft_logits=_peaked([1, 2, 3]),
        target_logits=_peaked([6, 6, 6, 6]),
        args=_args(temperature=None),
    )
    assert (decision.accepted_drafts, decision.accepted_rows) == (0, 1)
    assert decision.tokens == (6,)


@pytest.mark.parametrize("accepted_drafts", [0, 1, 2, 3])
def test_the_accepted_row_count_settles_the_step_where_plain_decode_would(accepted_drafts):
    """The row count the sampler returns, handed straight to ``_rollback_spec_tokens``, must
    leave the request where ``len(run)`` ordinary decode steps would have. This is the join
    between phase 2's bookkeeping and phase 4's arithmetic; nothing else checks it."""
    from tests.scheduler.test_spec_batch import (
        _decode_req,
        _plain_decode_steps,
        _scheduler,
    )

    drafts = [11, 12, 13]
    winners = drafts[:accepted_drafts] + [77] * (4 - accepted_drafts)
    decision = _sampler().step(
        uid=1,
        draft_tokens=drafts,
        draft_logits=_peaked(drafts, vocab=128),
        target_logits=_peaked(winners, vocab=128),
        args=_args(temperature=None),
    )

    plain = _scheduler()
    plain_req = _decode_req(plain)
    _plain_decode_steps(plain, plain_req, len(decision.tokens))

    stub = _scheduler()
    req = _decode_req(stub)
    stub._prepare_spec_batch(req, drafts)
    stub._rollback_spec_tokens(req, decision.accepted_rows)

    assert (req.cached_len, req.device_len) == (plain_req.cached_len, plain_req.device_len)
    assert torch.equal(stub.cache_manager.free_slots, plain.cache_manager.free_slots)
    assert req.extend_len == 1


# ----------------------------------------------------------------------------------- greedy


def test_greedy_accepts_exactly_the_rows_whose_argmax_the_draft_guessed():
    # row 1's draft is wrong, so row 2's correct guess is unreachable -- acceptance is a prefix
    decision = _sampler().step(
        uid=1,
        draft_tokens=[1, 5, 3],
        draft_logits=_peaked([1, 5, 3]),
        target_logits=_peaked([1, 2, 3, 4]),
        args=_args(temperature=None),
    )
    assert decision.accepted_drafts == 1
    assert decision.tokens == (1, 2)  # the bonus is row 1's argmax, not row 3's
    assert decision.rejected_at == 1


def test_full_greedy_acceptance_takes_the_bonus_from_the_last_row():
    decision = _sampler().step(
        uid=1,
        draft_tokens=[1, 2, 3],
        draft_logits=_peaked([1, 2, 3]),
        target_logits=_peaked([1, 2, 3, 4]),
        args=_args(temperature=None),
    )
    assert decision.accepted_drafts == 3
    assert decision.tokens == (1, 2, 3, 4)
    assert decision.rejected_at is None


@pytest.mark.parametrize(
    "args",
    [_args(temperature=None), _args(temperature=0.0), _args(temperature=0.7, top_k=1)],
    ids=["batch-greedy", "zero-temperature", "top-k-1"],
)
def test_every_greedy_shape_draws_no_randomness(args):
    sampler = _sampler()
    sampler.reset_request(1)
    before = sampler.generator_state(depth=3)

    decision = sampler.step(
        uid=1,
        draft_tokens=[1, 2, 3],
        draft_logits=_peaked([1, 2, 3]),
        target_logits=_peaked([1, 2, 3, 4]),
        args=args,
    )

    assert decision.greedy is True
    assert decision.tokens == (1, 2, 3, 4)
    assert torch.equal(sampler.generator_state(depth=3), before)


def test_greedy_is_self_consistent_with_this_forward_only():
    """Acceptance compares the draft against THIS forward's row argmax. bf16 GEMMs are not
    row-stable across batch widths, so a plain-decode argmax is not the reference -- the
    sampler must never be asked to reconcile the two."""
    target = _peaked([1, 2, 3, 4])
    decision = _sampler().step(
        uid=1,
        draft_tokens=[1, 2, 3],
        draft_logits=_peaked([6, 6, 6]),  # the draft head's own view is irrelevant when greedy
        target_logits=target,
        args=_args(temperature=None),
    )
    assert decision.tokens == tuple(torch.argmax(target, dim=-1).tolist())


# ------------------------------------------------------------------- the distribution filter


def test_the_filter_triple_is_read_off_the_servers_own_args():
    # the triple comes back through fp32 tensors, so compare it the way the filter uses it
    assert spec_filter_params(_args(temperature=None)) == (0.0, -1, 1.0)
    assert spec_filter_params(_args(temperature=0.7)) == pytest.approx((0.7, -1, 1.0))
    assert spec_filter_params(_args(temperature=0.7, top_k=5, top_p=0.9)) == pytest.approx(
        (0.7, 5, 0.9)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Sampler.prepare pins host memory")
def test_the_triple_survives_the_round_trip_through_sampler_prepare():
    from freetoken.engine.sample import Sampler

    def triple(params):
        batch = SimpleNamespace(reqs=[SimpleNamespace(sampling_params=params)])
        return spec_filter_params(Sampler(device=CPU, vocab_size=VOCAB).prepare(batch))

    assert triple(SamplingParams(temperature=0.0)) == (0.0, -1, 1.0)
    assert triple(SamplingParams(temperature=0.8, top_k=3, top_p=0.9)) == pytest.approx(
        (0.8, 3, 0.9)
    )
    # top_k <= 0 means "the whole vocabulary"; prepare drops the tensor entirely
    assert triple(SamplingParams(temperature=0.8, top_k=-1, top_p=1.0)) == pytest.approx(
        (0.8, -1, 1.0)
    )


def test_the_filter_is_the_one_acceptance_itself_uses():
    from freetoken.engine.spec_sample import _sampling_probabilities_batch

    logits = torch.randn(4, VOCAB, generator=torch.Generator().manual_seed(5))
    for temperature, top_k, top_p in [(1.0, -1, 1.0), (0.7, 3, 1.0), (1.2, -1, 0.8),
                                      (0.9, 4, 0.85), (0.0, -1, 1.0)]:
        args = _args(
            temperature=None if temperature == 0.0 else temperature,
            top_k=None if top_k < 1 else top_k,
            top_p=None if top_p >= 1.0 else top_p,
        )
        expected = _sampling_probabilities_batch(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        assert torch.equal(filtered_probs(logits, args), expected)


def test_greedy_args_filter_to_a_one_hot_row():
    probabilities = filtered_probs(_peaked([3]), _args(temperature=None))
    assert probabilities.tolist() == [[0, 0, 0, 1, 0, 0, 0, 0]]


# --------------------------------------------------------------------- rejection sampling


def test_a_draft_the_target_agrees_with_is_always_accepted():
    row = _rows([[0.2, 0.3, 0.5]])
    sampler = _sampler(depth=1)
    for _ in range(200):
        decision = sampler.step(
            uid=1,
            draft_tokens=[2],
            draft_logits=row,
            target_logits=torch.cat((row, row)),
            args=_args(temperature=1.0),
        )
        assert decision.accepted_drafts == 1


def test_a_draft_the_target_gives_no_mass_is_always_rejected():
    sampler = _sampler(depth=1)
    for _ in range(50):
        decision = sampler.step(
            uid=1,
            draft_tokens=[0],
            draft_logits=_rows([[0.5, 0.5, 0.0]]),
            target_logits=_rows([[0.0, 0.5, 0.5], [0.3, 0.3, 0.4]]),
            args=_args(temperature=1.0),
        )
        assert decision.accepted_drafts == 0
        assert decision.tokens[0] in (1, 2)


@pytest.mark.parametrize(
    "top_k,top_p", [(None, None), (4, None), (None, 0.9), (4, 0.85)],
    ids=["plain", "top-k", "top-p", "top-k-top-p"],
)
def test_the_first_emitted_token_is_distributed_as_the_targets_filtered_row(top_k, top_p):
    """THE property of speculative sampling: whatever the draft head proposes, the emitted
    token's marginal is the target's own filtered distribution. Accept-with-probability
    min(1, p/q) plus residual sampling on rejection is exactly what buys it, and getting either
    half wrong (sampling the correction from p, renormalizing the residual against the wrong
    row) shows up here and essentially nowhere else.

    The draft is drawn from the SAME filtered draft row acceptance uses as ``q`` -- that
    agreement is the theorem's precondition, and phase 5 owes it to this sampler.
    """
    trials = 20000
    draft_logits = _rows([[0.05, 0.30, 0.10, 0.25, 0.20, 0.10]])
    target_logits = _rows(
        [[0.30, 0.05, 0.25, 0.10, 0.20, 0.10], [0.10, 0.20, 0.30, 0.15, 0.15, 0.10]]
    )
    args = _args(temperature=1.0, top_k=top_k, top_p=top_p)
    q = filtered_probs(draft_logits, args)[0]
    expected = filtered_probs(target_logits[:1], args)[0]
    sampler = _sampler(depth=1, seed=20260901)
    proposals = torch.multinomial(
        q, trials, replacement=True, generator=torch.Generator().manual_seed(4242)
    )

    counts = torch.zeros(q.numel())
    for proposal in proposals.tolist():
        decision = sampler.step(
            uid=7,
            draft_tokens=[proposal],
            draft_logits=draft_logits,
            target_logits=target_logits,
            args=args,
        )
        counts[decision.tokens[0]] += 1

    empirical = counts / trials
    # 20k draws over a support of <= 6 puts the sampling error near 0.005 total variation;
    # using q, or the unrenormalized residual, moves it by more than 0.1
    assert float(0.5 * (empirical - expected).abs().sum()) < 0.03


def test_the_marginal_holds_at_full_depth():
    trials = 12000
    draft_logits = _rows([[0.5, 0.2, 0.2, 0.1]] * 3)
    target_logits = _rows([[0.1, 0.4, 0.2, 0.3]] * 4)
    args = _args(temperature=1.0)
    q = filtered_probs(draft_logits[:1], args)[0]
    expected = filtered_probs(target_logits[:1], args)[0]
    sampler = _sampler(depth=3, seed=99)
    proposals = torch.multinomial(
        q, trials * 3, replacement=True, generator=torch.Generator().manual_seed(11)
    ).view(trials, 3)

    counts = torch.zeros(q.numel())
    for row in proposals.tolist():
        decision = sampler.step(
            uid=7,
            draft_tokens=row,
            draft_logits=draft_logits,
            target_logits=target_logits,
            args=args,
        )
        counts[decision.tokens[0]] += 1

    assert float(0.5 * (counts / trials - expected).abs().sum()) < 0.035


# -------------------------------------------------------------------------------------- RNG


def _sampled_run(sampler, uid=3, depth=3):
    return sampler.step(
        uid=uid,
        draft_tokens=[0, 1, 2][:depth],
        draft_logits=_rows([[0.4, 0.3, 0.2, 0.1]] * depth),
        target_logits=_rows([[0.25, 0.25, 0.25, 0.25]] * (depth + 1)),
        args=_args(temperature=1.0),
    ).tokens


def test_the_same_request_replays_exactly():
    first = [_sampled_run(_sampler(seed=5150)) for _ in range(1)]
    replay = [_sampled_run(_sampler(seed=5150)) for _ in range(1)]
    assert first == replay

    sampler = _sampler(seed=5150)
    run = [_sampled_run(sampler) for _ in range(6)]
    sampler.reset_request(3)
    assert [_sampled_run(sampler) for _ in range(6)] == run


def test_a_new_request_resets_the_stream_without_an_explicit_call():
    sampler = _sampler(seed=5150)
    first = [_sampled_run(sampler, uid=3) for _ in range(4)]
    _sampled_run(sampler, uid=4)
    assert [_sampled_run(sampler, uid=3) for _ in range(4)] == first


def test_different_requests_get_different_streams():
    runs = {uid: tuple(_sampled_run(_sampler(seed=17), uid=uid) for _ in range(8))
            for uid in (1, 2, 3)}
    assert len(set(runs.values())) == 3


def test_each_depth_owns_an_independent_stream():
    sampler = _sampler(seed=17)
    sampler.reset_request(3)
    depth_two = sampler.generator_state(depth=2)
    depth_three = sampler.generator_state(depth=3)

    _sampled_run(sampler, depth=1)

    assert torch.equal(sampler.generator_state(depth=2), depth_two)
    assert torch.equal(sampler.generator_state(depth=3), depth_three)


def test_the_seed_is_read_off_the_shared_draft_seed_flag():
    assert resolve_spec_seed({}) == 1729
    assert resolve_spec_seed({"FREETOKEN_MTP_DRAFT_SEED": "4321"}) == 4321


def test_speculative_sampling_never_touches_the_default_rng():
    before = torch.random.get_rng_state().clone()
    sampler = _sampler(seed=8, guard_default_rng=True)
    for _ in range(20):
        _sampled_run(sampler)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_the_guard_catches_a_default_rng_leak(monkeypatch):
    import freetoken.engine.spec_sample as spec_sample

    original = spec_sample.batched_speculative_accept
    before = torch.random.get_rng_state().clone()

    def leaking(**kwargs):
        torch.rand(())
        return original(**kwargs)

    monkeypatch.setattr(spec_sample, "batched_speculative_accept", leaking)
    with pytest.raises(RuntimeError, match="default RNG"):
        _sampled_run(_sampler(seed=8, guard_default_rng=True))
    # the guard restores what the leak advanced, so the next request is still reproducible
    assert torch.equal(torch.random.get_rng_state(), before)


def test_the_guard_is_off_by_default_and_costs_the_hot_path_nothing():
    sampler = _sampler()
    assert sampler.guard_default_rng is False
    with default_rng_guard(CPU):
        _sampled_run(sampler)  # the sampler is clean whether or not it is watched


# ------------------------------------------------------------------------ shapes and stats


def test_the_step_serves_exactly_one_request():
    sampler = _sampler()
    args = BatchSamplingArgs(
        temperatures=torch.tensor([1.0, 1.0]),
        top_k=None,
        top_p=None,
    )
    with pytest.raises(AssertionError, match="one request"):
        sampler.step(
            uid=1,
            draft_tokens=[1],
            draft_logits=_peaked([1]),
            target_logits=_peaked([1, 2]),
            args=args,
        )


def test_the_target_must_bring_one_row_more_than_the_drafts():
    with pytest.raises(ValueError, match="1 \\+ k"):
        _sampler().step(
            uid=1,
            draft_tokens=[1, 2],
            draft_logits=_peaked([1, 2]),
            target_logits=_peaked([1, 2]),
            args=_args(temperature=None),
        )


def test_a_draft_run_deeper_than_the_sampler_is_refused():
    with pytest.raises(ValueError, match="depth"):
        _sampler(depth=2).step(
            uid=1,
            draft_tokens=[1, 2, 3],
            draft_logits=_peaked([1, 2, 3]),
            target_logits=_peaked([1, 2, 3, 4]),
            args=_args(temperature=None),
        )


def test_the_step_reports_the_per_draft_acceptance_probabilities():
    decision = _sampler(depth=1).step(
        uid=1,
        draft_tokens=[0],
        draft_logits=_rows([[0.5, 0.5]]),
        target_logits=_rows([[0.25, 0.75], [0.4, 0.6]]),
        args=_args(temperature=1.0),
    )
    assert decision.acceptance_probabilities == pytest.approx((0.5,))
    assert decision.draft_probabilities == pytest.approx((0.5,))
    assert decision.target_probabilities == pytest.approx((0.25,))


@pytest.mark.parametrize("keep", [1, 2, 3, 4])
def test_a_stop_condition_shrinks_the_run_and_the_row_count_together(keep):
    """Design 6.4: EOS / a stop string / the output budget can cut the accepted run after the
    fact. The row count must follow it, or the KV keeps a token the client never sees."""
    drafts = [1, 2, 3]
    decision = _sampler().step(
        uid=1,
        draft_tokens=drafts,
        draft_logits=_peaked(drafts),
        target_logits=_peaked([1, 2, 3, 4]),
        args=_args(temperature=None),
    )
    truncated = decision.truncated(keep)

    assert truncated.tokens == decision.tokens[:keep]
    assert truncated.accepted_rows == len(truncated.tokens) == keep
    assert truncated.accepted_drafts <= keep
    assert decision.truncated(len(decision.tokens)) is decision


@pytest.mark.parametrize("keep", [0, 5])
def test_a_truncation_outside_the_run_is_refused(keep):
    decision = _sampler(depth=1).step(
        uid=1,
        draft_tokens=[1],
        draft_logits=_peaked([1]),
        target_logits=_peaked([1, 2]),
        args=_args(temperature=None),
    )
    with pytest.raises(ValueError, match="keeps 1"):
        decision.truncated(keep)


def test_the_sampler_takes_its_depth_from_the_resolved_spec_config(monkeypatch):
    from freetoken.engine.config import resolve_spec_decode

    monkeypatch.setenv("FREETOKEN_MTP_DRAFT_SEED", "31337")
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_DEPTH": "2"})
    sampler = SpecSampler.from_config(spec, CPU)

    assert (sampler.depth, sampler.seed) == (2, 31337)
    assert sorted(sampler._generators) == [1, 2]


def test_acceptance_statistics_accumulate_over_a_run():
    sampler = _sampler()
    for accepted in (0, 3, 1):
        drafts = [1, 2, 3]
        winners = drafts[:accepted] + [7] * (4 - accepted)
        sampler.step(
            uid=1,
            draft_tokens=drafts,
            draft_logits=_peaked(drafts),
            target_logits=_peaked(winners),
            args=_args(temperature=None),
        )

    stats = sampler.stats
    assert stats["steps"] == 3
    assert stats["drafts_proposed"] == 9
    assert stats["drafts_accepted"] == 4
    assert stats["tokens_emitted"] == 7  # (0+1) + (3+1) + (1+1)
    assert stats["acceptance_histogram"] == (1, 1, 0, 1)

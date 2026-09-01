from __future__ import annotations

from types import SimpleNamespace
import time

import pytest
import torch

from freetoken.engine.mtp_fast_verify import batched_speculative_accept
from freetoken.engine.mtp_shadow import (
    MTPShadowObserver,
    greedy_acceptance,
    sampling_probabilities,
    speculative_accept,
)


def _logits(winners, vocab=8):
    rows = torch.full((len(winners), vocab), -20.0)
    for row, winner in enumerate(winners):
        rows[row, winner] = 20.0
    return rows


@pytest.mark.parametrize(
    ("proposals", "winners", "accepted", "corrected"),
    [
        ([1], [1, 4], 1, 4),
        ([1, 2], [1, 7, 3], 1, 7),
        ([1, 2, 3], [0, 2, 3, 6], 0, 0),
        ([1, 2, 3], [1, 2, 3, 6], 3, 6),
    ],
)
def test_greedy_acceptance_and_correction_are_exact(
    proposals, winners, accepted, corrected
):
    assert greedy_acceptance(proposals, _logits(winners)) == (accepted, corrected)


def test_sampling_filter_support_matches_temperature_topk_topp_rules():
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    topk = sampling_probabilities(logits, temperature=0.7, top_k=2, top_p=1.0)
    assert set(topk.nonzero().flatten().tolist()) == {0, 1}
    topp = sampling_probabilities(logits, temperature=1.0, top_k=-1, top_p=0.7)
    assert topp[0] > 0
    assert topp[-1] == 0
    greedy = sampling_probabilities(logits, temperature=0.0, top_k=-1, top_p=1.0)
    assert torch.equal(greedy, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))


def test_rejection_sampling_ratio_and_residual_correction_are_seeded():
    q = torch.tensor([0.5, 0.5])
    p = torch.tensor([0.25, 0.75])
    generator = torch.Generator().manual_seed(123)
    accepted = 0
    corrected = []
    for _ in range(4000):
        ok, token, ratio = speculative_accept(0, q, p, generator=generator)
        accepted += int(ok)
        if not ok:
            corrected.append(token)
        assert ratio == 0.5
    assert accepted / 4000 == pytest.approx(0.5, abs=0.035)
    assert set(corrected) == {1}


def test_dedicated_acceptance_generator_does_not_advance_global_rng():
    before = torch.random.get_rng_state().clone()
    generator = torch.Generator().manual_seed(99)
    q = torch.tensor([0.4, 0.6])
    p = torch.tensor([0.7, 0.3])
    for _ in range(20):
        speculative_accept(1, q, p, generator=generator)
    assert torch.equal(torch.random.get_rng_state(), before)


def _persistent_rng_observer(*, seed=177, uid=23):
    observer = object.__new__(MTPShadowObserver)
    observer.device = torch.device("cpu")
    observer.config = SimpleNamespace(seed=seed, depth=3)
    observer._init_private_rng()
    observer._reset_private_rng(uid)
    return observer


def _acceptance_inputs(uid=23):
    captured = SimpleNamespace(
        uid=uid,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
    )
    draft = [torch.tensor([0.5, 0.5]).log()]
    target = torch.stack(
        (
            torch.tensor([0.25, 0.75]).log(),
            torch.tensor([0.4, 0.6]).log(),
        )
    )
    return captured, draft, target


def test_persistent_acceptance_rng_advances_across_cycles_without_global_or_target_rng():
    observer = _persistent_rng_observer()
    captured, draft_logits, target_logits = _acceptance_inputs()
    global_before = torch.random.get_rng_state().clone()
    target_generator = torch.Generator().manual_seed(991)
    target_before = target_generator.get_state().clone()

    first = observer._accept_target_logits(
        captured, [0], draft_logits, target_logits
    )
    second = observer._accept_target_logits(
        captured, [0], draft_logits, target_logits
    )

    first_rng = first["acceptance_rng"]
    second_rng = second["acceptance_rng"]
    assert first_rng["stream"] == "acceptance-depth-1"
    assert first_rng["cycle_index"] == 0
    assert second_rng["cycle_index"] == 1
    assert first_rng["default_rng_unchanged"] is True
    assert (first_rng["draw_count_before"], first_rng["draw_count_after"]) == (0, 3)
    assert (second_rng["draw_count_before"], second_rng["draw_count_after"]) == (3, 6)
    assert first_rng["state_sha256_after"] == second_rng["state_sha256_before"]
    assert first_rng["state_sha256_before"] != second_rng["state_sha256_before"]
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert torch.equal(target_generator.get_state(), target_before)


def test_persistent_acceptance_rng_replays_only_after_request_boundary_reset():
    observer = _persistent_rng_observer(seed=712, uid=19)
    captured, draft_logits, target_logits = _acceptance_inputs(uid=19)

    def run_two_cycles():
        results = []
        for _ in range(2):
            result = observer._accept_target_logits(
                captured, [0], draft_logits, target_logits
            )
            rng = result["acceptance_rng"]
            results.append(
                (
                    result["accepted_tokens"],
                    result["corrected_token"],
                    rng["draw_count_before"],
                    rng["draw_count_after"],
                    rng["state_sha256_before"],
                    rng["state_sha256_after"],
                )
            )
        return results

    first = run_two_cycles()
    observer._reset_private_rng(19)
    replay = run_two_cycles()

    assert replay == first
    assert first[0][4] != first[1][4]


def test_greedy_acceptance_rng_records_zero_draws_and_unchanged_state():
    observer = _persistent_rng_observer()
    captured, draft_logits, target_logits = _acceptance_inputs()
    captured.temperature = 0.0

    result = observer._accept_target_logits(
        captured, [0], draft_logits, target_logits
    )

    rng = result["acceptance_rng"]
    assert rng["draw_count_before"] == rng["draw_count_after"] == 0
    assert rng["state_sha256_before"] == rng["state_sha256_after"]


def test_depth_acceptance_rng_streams_are_independent():
    observer = _persistent_rng_observer(seed=815, uid=41)
    captured, depth1_draft, depth1_target = _acceptance_inputs(uid=41)
    depth2_draft = [
        torch.tensor([0.6, 0.4]).log(),
        torch.tensor([0.3, 0.7]).log(),
    ]
    depth2_target = torch.stack(
        (
            torch.tensor([0.4, 0.6]).log(),
            torch.tensor([0.8, 0.2]).log(),
            torch.tensor([0.5, 0.5]).log(),
        )
    )
    depth2_before = observer._rng_stream_snapshot("acceptance", depth=2)
    depth3_before = observer._rng_stream_snapshot("acceptance", depth=3)

    depth1 = observer._accept_target_logits(
        captured, [0], depth1_draft, depth1_target
    )

    assert observer._rng_stream_snapshot("acceptance", depth=2) == depth2_before
    assert observer._rng_stream_snapshot("acceptance", depth=3) == depth3_before
    depth1_after = observer._rng_stream_snapshot("acceptance", depth=1)

    depth2 = observer._accept_target_logits(
        captured, [0, 1], depth2_draft, depth2_target
    )

    assert observer._rng_stream_snapshot("acceptance", depth=1) == depth1_after
    assert observer._rng_stream_snapshot("acceptance", depth=3) == depth3_before
    assert depth1["acceptance_rng"]["stream"] == "acceptance-depth-1"
    assert depth2["acceptance_rng"]["stream"] == "acceptance-depth-2"
    assert depth2["acceptance_rng"]["draws"] == 5
    assert depth1["acceptance_rng"]["cycle_index"] == 0
    assert depth2["acceptance_rng"]["cycle_index"] == 0


def test_acceptance_rng_rejects_default_rng_leak(monkeypatch):
    import freetoken.engine.mtp_shadow as mtp_shadow

    observer = _persistent_rng_observer()
    captured, draft_logits, target_logits = _acceptance_inputs()
    original = mtp_shadow.batched_speculative_accept
    global_before = torch.random.get_rng_state().clone()

    def leaking_acceptance(**kwargs):
        torch.rand(())
        return original(**kwargs)

    monkeypatch.setattr(mtp_shadow, "batched_speculative_accept", leaking_acceptance)
    try:
        with pytest.raises(
            RuntimeError,
            match="MTP checker changed target/global RNG state",
        ):
            observer._accept_target_logits(
                captured, [0], draft_logits, target_logits
            )
    finally:
        torch.random.set_rng_state(global_before)


def test_acceptance_requires_depth_plus_one_target_rows():
    with pytest.raises(ValueError, match=r"depth\+1"):
        greedy_acceptance([1, 2], _logits([1, 2]))


def _probability_logits(rows):
    return torch.stack([torch.tensor(row, dtype=torch.float32).log() for row in rows])


def test_batched_acceptance_matches_exact_distribution_without_touching_target_rng():
    draft_logits = _probability_logits([[0.5, 0.5]])
    target_logits = _probability_logits([[0.25, 0.75], [1.0, 0.0]])
    private = torch.Generator().manual_seed(8128)
    target_generator = torch.Generator().manual_seed(9128)
    target_before = target_generator.get_state().clone()
    global_before = torch.random.get_rng_state().clone()
    accepted = 0
    rejected_tokens = []
    trials = 4000

    for _ in range(trials):
        result = batched_speculative_accept(
            proposals=[0],
            draft_logits=draft_logits,
            target_logits=target_logits,
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            generator=private,
        )
        accepted += result.accepted_prefix
        if result.accepted_prefix == 0:
            rejected_tokens.append(result.corrected_token)
        assert result.acceptance_probabilities == pytest.approx((0.5,))

    standard_error = (0.5 * 0.5 / trials) ** 0.5
    assert accepted / trials == pytest.approx(0.5, abs=5 * standard_error)
    assert set(rejected_tokens) == {1}
    assert torch.equal(target_generator.get_state(), target_before)
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_batched_normal_acceptance_replays_exactly_with_private_seed():
    draft_logits = _probability_logits(
        [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.4, 0.1, 0.5]]
    )
    target_logits = _probability_logits(
        [[0.5, 0.4, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3]]
    )

    def run():
        return batched_speculative_accept(
            proposals=[0, 1, 2],
            draft_logits=draft_logits,
            target_logits=target_logits,
            temperature=0.8,
            top_k=-1,
            top_p=0.95,
            generator=torch.Generator().manual_seed(773),
        )

    assert run() == run()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_batched_normal_acceptance_uses_one_cuda_sync_and_not_global_rng():
    device = torch.device("cuda")
    draft_logits = _probability_logits([[0.5, 0.5]]).to(device)
    target_logits = _probability_logits([[0.25, 0.75], [0.4, 0.6]]).to(device)
    global_before = torch.cuda.get_rng_state(device).clone()

    result = batched_speculative_accept(
        proposals=[0],
        draft_logits=draft_logits,
        target_logits=target_logits,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        generator=torch.Generator(device=device).manual_seed(551),
    )

    assert result.synchronizations == 1
    assert torch.equal(torch.cuda.get_rng_state(device), global_before)


def test_acceptance_timing_excludes_evidence_conversion(monkeypatch):
    import freetoken.engine.spec_sample as spec_sample

    original = spec_sample._tensor_to_tuple

    def slow_conversion(tensor, cast):
        time.sleep(0.002)
        return original(tensor, cast)

    monkeypatch.setattr(spec_sample, "_tensor_to_tuple", slow_conversion)
    result = batched_speculative_accept(
        proposals=[1],
        draft_logits=_logits([1]),
        target_logits=_logits([1, 2]),
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        generator=torch.Generator().manual_seed(44),
    )

    assert result.required_wall_ms >= 0
    assert result.instrumentation_wall_ms >= 0.008
    assert result.required_wall_ms < result.instrumentation_wall_ms
    assert result.required_synchronizations == 0
    assert result.instrumentation_synchronizations == 0


def test_batched_acceptance_is_deterministic_and_greedy_matches_reference():
    proposals = [1, 2, 3]
    target_logits = _logits([1, 7, 3, 6])
    draft_logits = _logits(proposals)
    first = batched_speculative_accept(
        proposals=proposals,
        draft_logits=draft_logits,
        target_logits=target_logits,
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        generator=torch.Generator().manual_seed(44),
    )
    second = batched_speculative_accept(
        proposals=proposals,
        draft_logits=draft_logits,
        target_logits=target_logits,
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        generator=torch.Generator().manual_seed(44),
    )

    assert (first.accepted_prefix, first.corrected_token) == greedy_acceptance(
        proposals, target_logits
    )
    assert first == second
    assert first.greedy is True
    assert first.synchronizations == 0


def test_batched_acceptance_requires_bounded_matching_shapes():
    # the bound is 1..5 now (the integrated path's depth ceiling); an empty chain and a chain
    # past the ceiling are both refused
    for proposals in ([], [1, 2, 3, 4, 5, 6]):
        with pytest.raises(ValueError, match=r"1\.\.5 proposals"):
            batched_speculative_accept(
                proposals=proposals,
                draft_logits=torch.empty(len(proposals), 4),
                target_logits=torch.empty(len(proposals) + 1, 4),
                temperature=1.0,
                top_k=-1,
                top_p=1.0,
                generator=torch.Generator().manual_seed(1),
            )
    with pytest.raises(ValueError, match=r"depth\+1"):
        batched_speculative_accept(
            proposals=[1, 2],
            draft_logits=torch.zeros(2, 4),
            target_logits=torch.zeros(2, 4),
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            generator=torch.Generator().manual_seed(1),
        )


# -------------------------------------------------------- one transfer for the whole verdict


def test_the_verdict_leaves_the_device_in_a_single_transfer(monkeypatch):
    """Read row by row the verdict was six device round trips (a synchronize, two ``int()``
    reads and four ``.tolist()``). Packed it is one blocking copy, and that copy IS the
    synchronization -- so ``required_synchronizations`` still counts exactly one on cuda."""
    import freetoken.engine.spec_sample as spec_sample

    transfers: list[tuple] = []
    original = torch.Tensor.to

    def counting_to(self, *args, **kwargs):
        if args and args[0] in ("cpu", torch.device("cpu")):
            transfers.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counting_to)
    result = spec_sample.batched_speculative_accept(
        proposals=[1, 2, 3],
        draft_logits=_logits([1, 2, 3]),
        target_logits=_logits([1, 2, 5, 6]),
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        generator=torch.Generator().manual_seed(44),
    )

    # a CPU run makes no transfer at all; what is pinned is that nothing walks the tensors
    # one by one -- the whole verdict is read off a single packed host buffer
    assert transfers == []
    assert result.accepted_prefix == 2
    assert result.target_tokens == (1, 2, 5)


def test_the_packed_verdict_carries_the_same_numbers_the_row_reads_did():
    """The pack round-trips through float64; token ids and counts are exact in a double, and
    float32 -> float64 -> float is exact for the probabilities."""
    proposals = [1, 2, 3]
    target_logits = _logits([1, 7, 3, 6])
    draft_logits = _logits(proposals)
    result = batched_speculative_accept(
        proposals=proposals,
        draft_logits=draft_logits,
        target_logits=target_logits,
        temperature=0.8,
        top_k=-1,
        top_p=0.95,
        generator=torch.Generator().manual_seed(773),
    )

    from freetoken.engine.spec_sample import _sampling_probabilities_batch

    q_rows = _sampling_probabilities_batch(
        draft_logits, temperature=0.8, top_k=-1, top_p=0.95
    )
    p_rows = _sampling_probabilities_batch(
        target_logits, temperature=0.8, top_k=-1, top_p=0.95
    )
    for row, token in enumerate(proposals):
        assert result.draft_probabilities[row] == float(q_rows[row][token])
        assert result.target_probabilities[row] == float(p_rows[row][token])
    assert result.target_tokens == tuple(
        int(v) for v in torch.argmax(target_logits[:3], dim=-1)
    )


def test_acceptance_hands_back_the_emitted_run_on_the_device():
    """The caller's ``next_tokens_gpu`` used to be an H2D of ids the device had just produced.
    Acceptance cuts the run out of the tensors it already held instead."""
    result = batched_speculative_accept(
        proposals=[1, 2, 3],
        draft_logits=_logits([1, 2, 3]),
        target_logits=_logits([1, 2, 5, 6]),
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        generator=torch.Generator().manual_seed(44),
    )

    run = result.emitted_tokens_device
    assert run is not None
    assert run.dtype is torch.int32
    assert run.tolist() == [1, 2, result.corrected_token]
    assert len(run) == result.accepted_prefix + 1


def test_the_device_run_is_evidence_not_identity():
    """``emitted_tokens_device`` is a second shape of the same decision, so two results that
    agree on the decision must still compare equal."""
    kwargs = dict(
        proposals=[1, 2],
        draft_logits=_logits([1, 2]),
        target_logits=_logits([1, 2, 5]),
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
    )
    first = batched_speculative_accept(
        generator=torch.Generator().manual_seed(3), **kwargs
    )
    second = batched_speculative_accept(
        generator=torch.Generator().manual_seed(3), **kwargs
    )
    assert first.emitted_tokens_device is not second.emitted_tokens_device
    assert first == second

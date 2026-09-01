from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from flashlib.kernels.slot_cache import Stat

from freetoken.core import Batch
from freetoken.engine.mtp_fast_verify import (
    MTPFastVerifier,
    MTPGraphCaptureResult,
    MTPProjectionSample,
    MTPVerifyForwardResult,
    compare_verifier_distributions,
    compare_verifier_logits,
    sampling_distribution_divergence,
    sampling_support_order_matches,
)
from freetoken.engine.mtp_shadow import MTPShadowObserver


class _StatsCache:
    collect_stats = True
    decode_target = "gpu"

    def __init__(self):
        self.lru_stats = torch.zeros((1, 3), dtype=torch.int64)
        self.stat_fetched = torch.zeros((), dtype=torch.int64)
        self.bank_caches = {
            "gate_up": torch.zeros((8, 3), dtype=torch.float32),
            "down": torch.zeros((8, 2), dtype=torch.float16),
        }


class _TargetContext:
    def __init__(self):
        self.batch = None

    @contextmanager
    def forward_batch(self, batch):
        assert self.batch is None
        self.batch = batch
        try:
            yield
        finally:
            self.batch = None


class _TargetTextModel:
    def __init__(self, cache, ctx):
        self.cache = cache
        self.ctx = ctx

    def forward(self, input_ids, batch):
        assert self.ctx.batch is batch
        assert batch.mtp_verify is True
        self.cache.lru_stats[0, Stat.ACTIVE] += 20
        self.cache.lru_stats[0, Stat.MISS] += 5
        self.cache.lru_stats[0, Stat.CALLS] += 2
        self.cache.stat_fetched += 3
        return input_ids.float().unsqueeze(1)


class _LMHead:
    @staticmethod
    def forward_all(hidden):
        return torch.cat((hidden, hidden + 1, hidden - 1), dim=1)


def _batch(rows=3):
    req = SimpleNamespace(uid=7, extend_len=rows)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.arange(rows, dtype=torch.int32)
    return batch


def test_scratch_out_loc_requires_lease_owned_page_bases():
    leased = torch.tensor([128, 192], dtype=torch.int32)

    MTPShadowObserver._assert_out_loc_owned_by_lease(
        torch.tensor([134, 135, 192], dtype=torch.int32),
        leased,
        page_size=64,
    )
    with pytest.raises(RuntimeError, match="outside its scratch-page lease"):
        MTPShadowObserver._assert_out_loc_owned_by_lease(
            torch.tensor([64, 134], dtype=torch.int32),
            leased,
            page_size=64,
        )


def test_eager_fast_verifier_returns_all_rows_and_movement_counters():
    cache = _StatsCache()
    ctx = _TargetContext()
    target = SimpleNamespace(model=_TargetTextModel(cache, ctx), lm_head=_LMHead())
    verifier = MTPFastVerifier(
        target_ctx=ctx,
        target_model=target,
        moe_offload_cache=cache,
        device=torch.device("cpu"),
    )
    batch = _batch(rows=3)

    result = verifier.forward_eager(batch)

    assert isinstance(result, MTPVerifyForwardResult)
    assert result.mode == "fast-eager"
    assert result.logits.shape == (3, 3)
    assert torch.equal(result.logits[:, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert result.required_wall_ms >= 0
    assert result.core_cuda_ms >= 0
    assert result.required_synchronizations == 0
    assert result.instrumentation_wall_ms >= 0
    assert result.instrumentation_synchronizations == 0
    assert result.expert_movement == {
        "available": True,
        "layer_calls": 2,
        "active_experts": 20,
        "hit_experts": 15,
        "missing_experts": 5,
        "fetched_experts": 3,
        "cpu_experts": 0,
        "d2d_rows": 0,
        "bytes_per_expert": 16,
        "h2d_bytes": 48,
        "d2d_bytes": 0,
        "transfer_bytes": 48,
        "movement_reconciled": False,
    }
    assert batch.phase == "prefill"
    assert batch.mtp_verify is True
    assert ctx.batch is None


def test_eager_timing_ends_before_validation_and_counter_reduction(monkeypatch):
    events = []
    record_count = 0

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing is True

        def record(self):
            nonlocal record_count
            record_count += 1
            events.append(f"record-{record_count}")

        def elapsed_time(self, other):
            return 2.5

    original_isfinite = torch.isfinite
    original_movement = MTPFastVerifier._movement_counters
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: events.append(("synchronize", str(device))),
    )
    monkeypatch.setattr(
        torch,
        "isfinite",
        lambda value: (events.append("finite"), original_isfinite(value))[1],
    )
    monkeypatch.setattr(
        MTPFastVerifier,
        "_movement_counters",
        staticmethod(
            lambda cache, before: (
                events.append("movement-reduction"),
                original_movement(cache, before),
            )[1]
        ),
    )
    cache = _StatsCache()
    ctx = _TargetContext()
    target = SimpleNamespace(model=_TargetTextModel(cache, ctx), lm_head=_LMHead())

    result = MTPFastVerifier(ctx, target, cache, torch.device("cuda")).forward_eager(
        _batch(rows=2)
    )

    assert events == [
        ("synchronize", "cuda"),
        "record-1",
        "record-2",
        ("synchronize", "cuda"),
        "finite",
        "movement-reduction",
        ("synchronize", "cuda"),
    ]
    assert result.required_synchronizations == 1
    assert result.instrumentation_synchronizations == 2
    assert result.core_cuda_ms == 2.5


def test_eager_fast_verifier_reads_hybrid_fetch_counters():
    cache = _StatsCache()
    cache.decode_target = "hybrid"
    cache.stat_calls = torch.zeros((), dtype=torch.int64)
    cache.stat_active = torch.zeros((), dtype=torch.int64)
    cache.stat_missing = torch.zeros((), dtype=torch.int64)
    cache.stat_fetched = torch.zeros((), dtype=torch.int64)
    ctx = _TargetContext()

    class HybridTextModel(_TargetTextModel):
        def forward(self, input_ids, batch):
            assert self.ctx.batch is batch
            cache.stat_calls += 2
            cache.stat_active += 30
            cache.stat_missing += 7
            cache.stat_fetched += 3
            return input_ids.float().unsqueeze(1)

    target = SimpleNamespace(model=HybridTextModel(cache, ctx), lm_head=_LMHead())
    result = MTPFastVerifier(ctx, target, cache, torch.device("cpu")).forward_eager(
        _batch(rows=2)
    )

    assert result.expert_movement == {
        "available": True,
        "layer_calls": 2,
        "active_experts": 30,
        "hit_experts": 23,
        "missing_experts": 7,
        "fetched_experts": 3,
        "cpu_experts": 4,
        "d2d_rows": 0,
        "bytes_per_expert": 16,
        "h2d_bytes": 48,
        "d2d_bytes": 0,
        "transfer_bytes": 48,
        "movement_reconciled": True,
    }


def test_normal_sampling_comparison_requires_identical_filtered_support_and_order():
    first = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    same = torch.tensor([[4.01, 3.0, 2.0, 1.0]])
    reordered = torch.tensor([[4.0, 2.0, 3.0, 1.0]])

    assert sampling_support_order_matches(
        first, same, temperature=0.8, top_k=3, top_p=0.95
    )
    assert not sampling_support_order_matches(
        first, reordered, temperature=0.8, top_k=3, top_p=0.95
    )


def test_verifier_comparison_uses_declared_numeric_bound_and_exact_greedy_ids():
    oracle = torch.tensor([[5.0, 1.0, -2.0], [0.0, 3.0, 1.0]])
    fast = oracle + torch.tensor([[0.001, -0.001, 0.0], [0.0, 0.001, -0.001]])

    comparison = compare_verifier_logits(oracle, fast, rtol=2e-2, atol=2e-2)

    assert comparison.matches is True
    assert comparison.greedy_ids_match is True
    assert comparison.rows == 2
    assert comparison.max_abs_error == pytest.approx(0.001, abs=1e-6)
    changed_winner = fast.clone()
    changed_winner[0] = torch.tensor([0.0, 6.0, -2.0])
    mismatch = compare_verifier_logits(oracle, changed_winner, rtol=2e-2, atol=2e-2)
    assert mismatch.matches is False
    assert mismatch.greedy_ids_match is False


def test_compare_mode_uses_fast_timing_and_excludes_oracle_from_projection():
    from freetoken.engine.mtp_shadow import MTPShadowObserver

    observer = object.__new__(MTPShadowObserver)
    observer.fast_verifier = SimpleNamespace(forward_eager=lambda batch: None)
    observer._forward_target_oracle = lambda batch: None
    calls = []
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])

    def transaction(captured, candidate_ids, *, forward, mode):
        calls.append(mode)
        return {
            "mode": mode,
            "logits": logits.clone(),
            "target_forward_ms": 4.0 if mode == "fast-eager" else 100.0,
            "target_cuda_ms": 3.5 if mode == "fast-eager" else 100.0,
            "target_synchronizations": 1,
            "expert_movement": {"available": mode == "fast-eager"},
            "state_prepare_ms": 1.0,
            "state_cleanup_ms": 2.0,
            "state_ms": 3.0,
            "state_digest_unchanged": True,
            "pages_conserved": True,
            "recurrent_slots_conserved": True,
            "borrowed_recurrent_snapshot": True,
            "borrowed_snapshot_digest_unchanged": True,
        }

    observer._run_target_transaction = transaction
    observer._accept_target_logits = lambda captured, proposals, drafts, got: {
        "verified_rows": 2,
        "target_logits_sha256": "fast-hash",
        "target_tokens": [0],
        "accepted_tokens": 1,
        "corrected_token": 1,
        "acceptance_ms": 0.5,
        "selected_probabilities": [],
    }
    captured = SimpleNamespace()

    result = observer.verify_candidates(
        captured,
        confirmed_token=9,
        proposals=[4],
        draft_logits=[torch.zeros(2)],
        mode="compare",
    )

    assert calls == ["fast-eager", "oracle"]
    assert result["target_forward_ms"] == 4.0
    assert result["verification_ms"] == 4.5
    assert result["projection_eligible"] is False
    assert result["comparison"]["oracle_target_forward_ms"] == 100.0
    assert result["comparison"]["projection_eligible"] is False


def test_compare_mode_checks_graph_after_eager_and_before_oracle():
    observer = object.__new__(MTPShadowObserver)
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    calls = []

    class Graph:
        replay = object()

        def support(self, width):
            return MTPGraphCaptureResult(width, "captured", "", 1234)

    observer.graph_verifier = Graph()
    observer.fast_verifier = SimpleNamespace(forward_eager=object())

    def transaction(captured, candidate_ids, *, forward, mode):
        calls.append(mode)
        return {
            "mode": mode,
            "logits": logits.clone(),
            "target_forward_ms": 2.0,
            "target_cuda_ms": 1.5,
            "target_synchronizations": 1,
            "expert_movement": {"available": mode != "oracle"},
            "state_prepare_ms": 0.5,
            "state_cleanup_ms": 0.5,
            "state_ms": 1.0,
        }

    observer._run_target_transaction = transaction
    observer._forward_target_oracle = object()
    observer._accept_target_logits = lambda captured, proposals, drafts, got: {
        "verified_rows": 2,
        "target_logits_sha256": "same-hash",
        "target_tokens": [0],
        "accepted_tokens": 1,
        "corrected_token": 1,
        "acceptance_ms": 0.25,
        "selected_probabilities": [],
    }

    result = observer.verify_candidates(
        SimpleNamespace(),
        confirmed_token=9,
        proposals=[4],
        draft_logits=[torch.zeros(2)],
        mode="compare",
    )

    assert calls == ["fast-eager", "fast-graph", "oracle"]
    assert result["projection_eligible"] is False
    assert result["graph"]["support"]["memory_bytes"] == 1234
    assert result["graph"]["eager_comparison"]["matches"] is True
    assert result["graph"]["replay"]["mode"] == "fast-graph"


def test_unsupported_graph_falls_back_to_eager_but_is_not_graph_projection_data():
    observer = object.__new__(MTPShadowObserver)
    calls = []

    class Graph:
        def support(self, width):
            return MTPGraphCaptureResult(
                width, "permanently-unsupported", "BOUNDED_TEST", 0
            )

    observer.graph_verifier = Graph()
    observer.fast_verifier = SimpleNamespace(forward_eager=object())

    def transaction(captured, candidate_ids, *, forward, mode):
        calls.append(mode)
        return {
            "mode": mode,
            "logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            "target_forward_ms": 2.0,
            "target_cuda_ms": 1.5,
            "target_synchronizations": 1,
            "expert_movement": {"available": True},
            "state_prepare_ms": 0.5,
            "state_cleanup_ms": 0.5,
            "state_ms": 1.0,
        }

    observer._run_target_transaction = transaction
    observer._accept_target_logits = lambda captured, proposals, drafts, got: {
        "verified_rows": 2,
        "target_logits_sha256": "eager-hash",
        "target_tokens": [0],
        "accepted_tokens": 1,
        "corrected_token": 1,
        "acceptance_ms": 0.25,
        "selected_probabilities": [],
    }

    result = observer.verify_candidates(
        SimpleNamespace(),
        confirmed_token=9,
        proposals=[4],
        draft_logits=[torch.zeros(2)],
        mode="fast-graph",
    )

    assert calls == ["fast-eager"]
    assert result["mode"] == "fast-eager"
    assert result["projection_eligible"] is False
    assert result["graph"]["support"]["reason"] == "BOUNDED_TEST"


def test_retryable_graph_attempt_falls_back_to_eager_without_projection():
    observer = object.__new__(MTPShadowObserver)
    calls = []
    retryable = MTPGraphCaptureResult(2, "retryable", "MEMORY_ADMISSION", 0)

    class Graph:
        capture_forward = object()

        def support(self, width):
            return None

        def last_attempt(self, width):
            return retryable

    observer.graph_verifier = Graph()
    observer.fast_verifier = SimpleNamespace(forward_eager=object())

    def transaction(captured, candidate_ids, *, forward, mode):
        calls.append(mode)
        return {
            "mode": mode,
            "logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            "target_required_wall_ms": 2.0,
            "target_core_cuda_ms": 1.5,
            "target_required_synchronizations": 1,
            "target_instrumentation_wall_ms": 0.25,
            "target_instrumentation_synchronizations": 1,
            "target_forward_ms": 2.0,
            "target_cuda_ms": 1.5,
            "target_synchronizations": 2,
            "expert_movement": {"available": mode != "graph-capture"},
            "state_required_scope": "scratch-backup-through-cleanup",
            "state_required_synchronizations": 2,
            "state_prepare_ms": 0.5,
            "state_cleanup_ms": 0.5,
            "state_ms": 1.0,
            "state_instrumentation_ms": 0.25,
            "state_instrumentation_synchronizations": 0,
        }

    observer._run_target_transaction = transaction
    observer._accept_target_logits = lambda captured, proposals, drafts, got: {
        "verified_rows": 2,
        "target_logits_sha256": "eager-hash",
        "target_tokens": [0],
        "accepted_tokens": 1,
        "corrected_token": 1,
        "acceptance_ms": 0.25,
        "acceptance_required_wall_ms": 0.25,
        "selected_probabilities": [],
    }

    result = observer.verify_candidates(
        SimpleNamespace(),
        confirmed_token=9,
        proposals=[4],
        draft_logits=[torch.zeros(2)],
        mode="fast-graph",
    )

    assert calls == ["graph-capture", "fast-eager"]
    assert result["mode"] == "fast-eager"
    assert result["projection_eligible"] is False
    assert result["graph"]["support"]["status"] == "retryable"
    assert result["graph"]["capture"]["mode"] == "graph-capture"


def test_projection_components_conserve_total_time():
    sample = MTPProjectionSample(
        prompt_ms=20.0,
        draft_ms=4.0,
        verify_ms=5.0,
        state_ms=2.0,
        acceptance_ms=1.0,
        emitted_tokens=3,
    )

    assert sample.cycle_ms == 12.0
    assert sample.total_ms == 32.0
    assert sample.tokens_per_second == pytest.approx(93.75)
    assert sample.component_total_ms == sample.total_ms
    assert sample.component_total_ms == sum(
        sample.components[name] for name in ("P", "D", "V", "S", "A")
    )
    assert sample.components == {
        "P": 20.0,
        "D": 4.0,
        "V": 5.0,
        "S": 2.0,
        "A": 1.0,
        "E": 3,
    }
    with pytest.raises(ValueError, match="non-negative"):
        MTPProjectionSample(-1.0, 1.0, 1.0, 1.0, 1.0, 1)
    with pytest.raises(ValueError, match="at least one"):
        MTPProjectionSample(1.0, 1.0, 1.0, 1.0, 1.0, 0)
    with pytest.raises(ValueError, match="must be positive"):
        MTPProjectionSample(0.0, 0.0, 0.0, 0.0, 0.0, 1)


@pytest.mark.parametrize("rows", [0, 1, 5])
def test_eager_fast_verifier_rejects_unapproved_width_before_target(rows):
    cache = _StatsCache()
    ctx = _TargetContext()
    target = SimpleNamespace(model=_TargetTextModel(cache, ctx), lm_head=_LMHead())
    verifier = MTPFastVerifier(ctx, target, cache, torch.device("cpu"))
    batch = _batch(rows=rows)

    with pytest.raises(ValueError, match="2, 3, or 4 rows"):
        verifier.forward_eager(batch)

    assert ctx.batch is None
    assert int(cache.lru_stats.sum()) == 0


# --------------------------------------------------------------------------------------
# compare-mode tolerances: the oracle runs the prefill kernel path and the fast checker the
# decode-movement path, so the two bf16 accumulation orders differ by roughly one ulp
# --------------------------------------------------------------------------------------

_COMPARE_BASE = torch.tensor([[9.0, 1.0, -2.0, 0.5], [0.25, 8.0, 1.0, -3.0]])


def _compare_observer(tmp_path, per_mode: dict[str, torch.Tensor]):
    observer = object.__new__(MTPShadowObserver)
    observer.fast_verifier = SimpleNamespace(forward_eager=object())
    observer._forward_target_oracle = object()
    observer.config = SimpleNamespace(private_root=tmp_path, placement="test")
    observer.trace_path = tmp_path / "trace.jsonl"
    (tmp_path / "evidence").mkdir(exist_ok=True)

    def transaction(captured, candidate_ids, *, forward, mode):
        return {
            "mode": mode,
            "logits": per_mode[mode].clone(),
            "target_forward_ms": 2.0,
            "target_cuda_ms": 1.5,
            "target_synchronizations": 1,
            "expert_movement": {"available": mode != "oracle"},
            "state_prepare_ms": 0.5,
            "state_cleanup_ms": 0.5,
            "state_ms": 1.0,
        }

    observer._run_target_transaction = transaction
    observer._accept_target_logits = lambda captured, proposals, drafts, got: {
        "verified_rows": 2,
        "target_logits_sha256": "compare-hash",
        "target_tokens": [0],
        "accepted_tokens": 1,
        "corrected_token": 1,
        "acceptance_ms": 0.25,
        "selected_probabilities": [],
    }
    return observer


def _compare(observer, proposals=(4,), captured=None):
    return observer.verify_candidates(
        SimpleNamespace() if captured is None else captured,
        confirmed_token=9,
        proposals=list(proposals),
        draft_logits=[torch.zeros(4) for _ in proposals],
        mode="compare",
    )


class _CapturedGraph:
    replay = object()

    def support(self, width):
        return MTPGraphCaptureResult(width, "captured", "", 1234)


def test_distribution_comparison_passes_identical_rows():
    comparison = compare_verifier_distributions(_COMPARE_BASE, _COMPARE_BASE.clone())

    assert comparison.matches is True
    assert comparison.greedy_ids_match is True
    assert comparison.rows == 2
    assert comparison.worst_total_variation == pytest.approx(0.0, abs=1e-7)
    assert comparison.max_abs_error == pytest.approx(0.0, abs=1e-7)
    assert comparison.worst_excess_total_variation == pytest.approx(0.0, abs=1e-7)
    assert comparison.max_excess_total_variation == 0.05
    assert comparison.noise_tolerance == 0.5


def test_distribution_comparison_rejects_a_moved_greedy_token():
    moved = _COMPARE_BASE.clone()
    moved[0] = torch.tensor([9.0, 9.4, -2.0, 0.5])

    comparison = compare_verifier_distributions(_COMPARE_BASE, moved)

    assert comparison.greedy_ids_match is False
    assert comparison.matches is False
    # a structural flip fails the near-tie escape from the oracle's side by whole logits
    assert comparison.worst_greedy_logit_gap == pytest.approx(8.0)


def test_distribution_comparison_forgives_an_exact_bf16_tie_argmax_flip():
    # the live signature: the oracle's two-ulp gap (18.625 vs 18.375 at bf16 ulp 0.125)
    # collapses to an exact 18.5/18.5 tie in the fast path, and GPU argmax tie-breaking
    # picks the other token while the distributions stay put
    oracle = torch.tensor([[18.625, 18.375, 13.875, 1.0]])
    fast = torch.tensor([[18.5, 18.5001, 13.8125, 1.0]])
    assert torch.argmax(oracle, -1).item() != torch.argmax(fast, -1).item()

    comparison = compare_verifier_distributions(oracle, fast)

    assert comparison.greedy_ids_match is True
    assert comparison.matches is True
    assert comparison.worst_greedy_logit_gap == pytest.approx(0.25, abs=1e-3)


def test_distribution_comparison_near_tie_escape_needs_both_sides():
    # the fast path scoring the tokens as a tie is not enough when the oracle
    # separates them by whole logits -- that is a wrong-token error, not noise
    oracle = torch.tensor([[9.0, 4.0, -2.0, 0.5]])
    fast = torch.tensor([[9.0, 9.0001, -2.0, 0.5]])

    comparison = compare_verifier_distributions(oracle, fast)

    assert comparison.greedy_ids_match is False
    assert comparison.matches is False
    assert comparison.worst_greedy_logit_gap == pytest.approx(5.0, abs=1e-3)


def test_distribution_comparison_brackets_the_excess_bound():
    reference = torch.tensor([[0.0, 0.0]])
    # a 0.6 move clamps to the 0.5 band leaving |sigmoid(0.6) - sigmoid(0.5)| ~ 0.023
    # of excess; a 1.0 move leaves |sigmoid(1.0) - sigmoid(0.5)| ~ 0.109
    inside = compare_verifier_distributions(reference, torch.tensor([[0.6, 0.0]]))
    outside = compare_verifier_distributions(reference, torch.tensor([[1.0, 0.0]]))

    assert inside.greedy_ids_match is outside.greedy_ids_match is True
    assert inside.worst_excess_total_variation == pytest.approx(0.0232, abs=1e-3)
    assert inside.matches is True
    assert outside.worst_excess_total_variation == pytest.approx(0.1086, abs=1e-3)
    assert outside.matches is False


def test_distribution_comparison_forgives_within_noise_mass_on_a_knife_edge():
    # the second live signature: a near-tied top pair carrying most of the row's mass
    # reshuffles under sub-noise logit moves, so the plain TV (0.18 here, 0.13 live)
    # crosses any bound that would still catch the subtlest structural error (0.26)
    oracle = torch.tensor([[18.625, 18.375, 13.875, 1.0]])
    fast = torch.tensor([[18.25, 18.75, 13.875, 1.0]])

    comparison = compare_verifier_distributions(oracle, fast)

    assert comparison.worst_total_variation > 0.10
    assert comparison.worst_excess_total_variation == pytest.approx(0.0, abs=1e-6)
    assert comparison.greedy_ids_match is True
    assert comparison.matches is True


def test_distribution_comparison_still_stops_the_subtlest_known_structural_error():
    # the 62/38 -> 88/12 mass shift the exact-order check was blind to: the winning
    # token's logit lead grows by 1.5, far beyond cross-path noise
    oracle = torch.log(torch.tensor([[0.62, 0.38]]))
    fast = torch.log(torch.tensor([[0.88, 0.12]]))

    comparison = compare_verifier_distributions(oracle, fast)

    assert comparison.greedy_ids_match is True
    assert comparison.worst_excess_total_variation == pytest.approx(0.0876, abs=1e-3)
    assert comparison.matches is False


def test_distribution_comparison_rejects_mismatched_shapes():
    comparison = compare_verifier_distributions(_COMPARE_BASE, _COMPARE_BASE[:1])

    assert comparison.matches is False
    assert comparison.worst_total_variation == float("inf")


def test_compare_mode_accepts_bf16_ulp_scale_oracle_drift(tmp_path):
    drifted = _COMPARE_BASE + torch.tensor(
        [[0.3, -0.28, 0.31, -0.3], [-0.3, 0.29, -0.31, 0.3]]
    )
    observer = _compare_observer(
        tmp_path, {"fast-eager": _COMPARE_BASE, "oracle": drifted}
    )

    result = _compare(observer)

    assert result["comparison"]["greedy_ids_match"] is True
    assert result["comparison"]["normal_support_order_matches"] is True
    assert result["comparison"]["matches"] is True
    assert result["comparison"]["worst_excess_total_variation"] < 0.05
    assert result["comparison"]["worst_total_variation"] < 0.10
    # the raw-logit bound is still reported, and would have stopped the run on this drift
    assert result["comparison"]["max_abs_error"] == pytest.approx(0.31, abs=1e-4)
    assert compare_verifier_logits(drifted, _COMPARE_BASE).matches is False


def test_compare_mode_accepts_deep_row_drift_that_leaves_the_distribution_intact(tmp_path):
    # the live depth-2 signature: per-row logit drift compounds through candidate-row KV until
    # the tail moves by ~0.8, while the sampled distribution stays put
    base = torch.tensor(
        [[9.0, 1.0, -2.0, 0.5], [0.25, 8.0, 1.0, -3.0], [-1.0, 0.5, 9.0, 0.25]]
    )
    drifted = base + torch.tensor(
        [[0.0, 0.8, -0.8, 0.8], [0.8, 0.0, -0.8, 0.8], [0.8, -0.8, 0.0, 0.8]]
    )
    observer = _compare_observer(tmp_path, {"fast-eager": base, "oracle": drifted})

    result = _compare(observer, proposals=(4, 5))

    assert result["comparison"]["rows"] == 3
    assert result["comparison"]["matches"] is True
    assert result["comparison"]["worst_total_variation"] < 0.01
    assert result["comparison"]["max_abs_error"] == pytest.approx(0.8, abs=1e-4)
    # the fixed-logit bound this replaced could not admit an 0.8 drift at any depth
    assert compare_verifier_logits(drifted, base, rtol=0.05, atol=0.5).matches is False


def test_compare_mode_still_stops_when_the_greedy_token_moves(tmp_path):
    flipped = _COMPARE_BASE.clone()
    flipped[0] = torch.tensor([9.0, 9.4, -2.0, 0.5])
    observer = _compare_observer(
        tmp_path, {"fast-eager": _COMPARE_BASE, "oracle": flipped}
    )

    with pytest.raises(RuntimeError, match="prompt-style oracle"):
        _compare(observer)


def test_compare_mode_still_stops_when_sampling_mass_moves(tmp_path):
    # the greedy token and the support order both survive; only the distribution moves
    diverged = _COMPARE_BASE.clone()
    diverged[0] = torch.tensor([9.0, 8.6, -2.0, 0.5])
    observer = _compare_observer(
        tmp_path, {"fast-eager": _COMPARE_BASE, "oracle": diverged}
    )
    assert compare_verifier_distributions(diverged, _COMPARE_BASE).greedy_ids_match

    with pytest.raises(RuntimeError, match="prompt-style oracle"):
        _compare(observer)


def test_compare_mode_keeps_the_tight_bound_between_graph_and_eager(tmp_path):
    replayed = _COMPARE_BASE + torch.tensor(
        [[0.3, -0.28, 0.31, -0.3], [-0.3, 0.29, -0.31, 0.3]]
    )
    observer = _compare_observer(
        tmp_path,
        {"fast-eager": _COMPARE_BASE, "fast-graph": replayed, "oracle": _COMPARE_BASE},
    )
    observer.graph_verifier = _CapturedGraph()

    with pytest.raises(RuntimeError, match="eager fast checker"):
        _compare(observer)


def test_eager_fast_verifier_rejects_rows_that_disagree_with_extend_len():
    cache = _StatsCache()
    ctx = _TargetContext()
    target = SimpleNamespace(model=_TargetTextModel(cache, ctx), lm_head=_LMHead())
    verifier = MTPFastVerifier(ctx, target, cache, torch.device("cpu"))
    batch = _batch(rows=3)
    # the overlap-skewed capture reconstructs a request one token short of its verify rows
    batch.reqs[0].extend_len = 2

    with pytest.raises(ValueError, match="extend_len"):
        verifier.forward_eager(batch)
    assert ctx.batch is None


# --------------------------------------------------------------------------------------
# sampling-order agreement: exact rank order among a bf16 near-tie is not stable across the
# oracle's prefill kernels and the fast checker's decode-movement kernels
# --------------------------------------------------------------------------------------

_NORMAL_SAMPLING = SimpleNamespace(temperature=0.8, top_k=40, top_p=0.9)
# a flattening temperature raises tail probabilities, so a beyond-noise move on a tail
# token becomes client-visible in the filtered distribution while the unfiltered
# distribution barely registers it
_FLAT_SAMPLING = SimpleNamespace(temperature=2.0, top_k=4, top_p=1.0)


def test_sampling_divergence_is_zero_for_identical_logits():
    assert sampling_distribution_divergence(
        _COMPARE_BASE, _COMPARE_BASE.clone(), temperature=0.8, top_k=40, top_p=0.9
    ) == pytest.approx(0.0, abs=1e-7)


def test_sampling_divergence_is_one_when_the_greedy_token_moves():
    moved = _COMPARE_BASE.clone()
    moved[0] = torch.tensor([9.0, 9.4, -2.0, 0.5])

    assert sampling_distribution_divergence(
        _COMPARE_BASE, moved, temperature=0.0, top_k=1, top_p=1.0
    ) == pytest.approx(1.0, abs=1e-6)


def test_sampling_divergence_matches_the_two_logit_closed_form():
    reference = torch.tensor([[0.0, 0.0]])
    params = dict(temperature=1.0, top_k=2, top_p=1.0)

    # the filtered distribution is a sigmoid, so the total variation is |sigmoid(d) - 0.5|
    assert sampling_distribution_divergence(
        reference, torch.tensor([[0.3, 0.0]]), **params
    ) == pytest.approx(0.0744, abs=1e-3)
    assert sampling_distribution_divergence(
        reference, torch.tensor([[0.45, 0.0]]), **params
    ) == pytest.approx(0.1107, abs=1e-3)


def test_sampling_divergence_rejects_mismatched_shapes():
    assert sampling_distribution_divergence(
        _COMPARE_BASE, _COMPARE_BASE[:1], temperature=0.8, top_k=40, top_p=0.9
    ) == float("inf")


def test_compare_mode_accepts_a_near_tie_rank_swap_inside_the_support(tmp_path):
    # two mid-support tokens exchange rank under ulp noise; the sampled distribution does not move
    base = torch.tensor([[2.0, 1.00, 0.99, 0.5], [0.25, 8.0, 1.0, -3.0]])
    swapped = torch.tensor([[2.0, 0.99, 1.00, 0.5], [0.25, 8.0, 1.0, -3.0]])
    observer = _compare_observer(tmp_path, {"fast-eager": base, "oracle": swapped})
    # the exact-order check this replaced is what stopped the live run
    assert not sampling_support_order_matches(
        swapped, base, temperature=0.8, top_k=40, top_p=0.9
    )

    result = _compare(observer, captured=_NORMAL_SAMPLING)

    assert result["comparison"]["matches"] is True
    assert result["comparison"]["normal_support_order_matches"] is True
    assert result["comparison"]["normal_sampling_divergence"] < 0.01


def test_compare_mode_still_stops_when_the_sampled_distribution_moves(tmp_path):
    # a tail token moves 1.4 logits -- far beyond cross-path noise -- but carries so
    # little unfiltered mass the unfiltered excess gate stays quiet; a flattening
    # temperature makes the move client-visible and the filtered excess gate fires
    base = torch.tensor([[5.0, 2.0, 2.4, -5.0], [0.25, 8.0, 1.0, -3.0]])
    oracle = torch.tensor([[5.0, 2.0, 1.0, -5.0], [0.25, 8.0, 1.0, -3.0]])
    observer = _compare_observer(tmp_path, {"fast-eager": base, "oracle": oracle})
    # only the filtered distribution gate may fire: the unfiltered checks all pass
    unfiltered = compare_verifier_distributions(oracle, base)
    assert unfiltered.matches is True and unfiltered.greedy_ids_match is True
    assert sampling_distribution_divergence(
        oracle, base, temperature=2.0, top_k=4, top_p=1.0, noise_tolerance=0.5
    ) > 0.05

    with pytest.raises(RuntimeError, match="prompt-style oracle"):
        _compare(observer, captured=_FLAT_SAMPLING)


def test_compare_mode_no_longer_stops_on_a_sharpened_within_noise_move(tmp_path):
    # the failure that killed the live matrix: a 0.3-logit move (inside observed
    # cross-path noise) sharpened by a low temperature crossed the old fixed bound;
    # in excess terms it is exactly zero
    base = torch.tensor([[3.0, 2.9, 0.0, -5.0], [0.25, 8.0, 1.0, -3.0]])
    oracle = torch.tensor([[3.0, 2.6, 0.0, -5.0], [0.25, 8.0, 1.0, -3.0]])
    observer = _compare_observer(tmp_path, {"fast-eager": base, "oracle": oracle})
    assert sampling_distribution_divergence(
        oracle, base, temperature=0.2, top_k=2, top_p=1.0
    ) > 0.10
    assert sampling_distribution_divergence(
        oracle, base, temperature=0.2, top_k=2, top_p=1.0, noise_tolerance=0.5
    ) == pytest.approx(0.0, abs=1e-6)

    result = _compare(
        observer, captured=SimpleNamespace(temperature=0.2, top_k=2, top_p=1.0)
    )

    assert result["comparison"]["matches"] is True
    assert result["comparison"]["normal_support_order_matches"] is True


def test_compare_mode_keeps_exact_sampling_order_between_graph_and_eager(tmp_path):
    base = torch.tensor([[2.0, 1.00, 0.99, 0.5], [0.25, 8.0, 1.0, -3.0]])
    swapped = torch.tensor([[2.0, 0.99, 1.00, 0.5], [0.25, 8.0, 1.0, -3.0]])
    observer = _compare_observer(
        tmp_path, {"fast-eager": base, "fast-graph": swapped, "oracle": base}
    )
    observer.graph_verifier = _CapturedGraph()
    # the graph replays the same kernels, so only the exact-order gate may fire here
    assert compare_verifier_logits(base, swapped).matches is True

    with pytest.raises(RuntimeError, match="eager fast checker"):
        _compare(observer, captured=_NORMAL_SAMPLING)

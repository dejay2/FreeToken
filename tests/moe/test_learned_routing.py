"""moe/learned_routing.py: the breadth ranking, the stats file, and the recorder's merge rules.

CPU-only and torch-free apart from the import chain; the one real-data test reads the Qwen3.8
routing captures checked into docs/research.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from freetoken.moe import learned_routing as lr

RESEARCH = Path(__file__).resolve().parents[2] / "docs" / "research" / "routing-skew-2026-09-02"


def _stats(num_layers=3, num_experts=4, rows=None):
    freq = rows or [[0] * num_experts for _ in range(num_layers)]
    return lr.RoutingStats(num_layers, num_experts, freq)


# ------------------------------------------------------------------ ranking


def test_breadth_ranking_reproduces_the_measured_qwen_top_six():
    """The fixed GPU_OWNED_LAYER_RANK came from realized miss rates; the histogram-only score
    must stay a close proxy on the pooled captures (five of the top six, seven of the top
    eight, rank correlation above 0.95) or auto:N would change behaviour on a Qwen box the
    first time a stats file appears. The one swap is layer 5 (ninth) for layer 22 (sixth)."""
    from freetoken.engine.engine import GPU_OWNED_LAYER_RANK

    if not RESEARCH.is_dir():
        pytest.skip("routing captures are not present")
    total = None
    for name in ("code", "prose", "chat8k", "toolcall"):
        freq = json.loads((RESEARCH / f"{name}.json").read_text(encoding="utf-8"))["decode_freq"]
        if total is None:
            total = [[0] * len(freq[0]) for _ in freq]
        for layer, row in enumerate(freq):
            for expert, count in enumerate(row):
                total[layer][expert] += count
    ranked = lr.rank_layers_by_breadth(total)
    assert sorted(ranked) == list(range(48))
    assert set(ranked[:3]) == {0, 1, 2}
    assert len(set(ranked[:6]) & set(GPU_OWNED_LAYER_RANK[:6])) >= 5
    assert len(set(ranked[:8]) & set(GPU_OWNED_LAYER_RANK[:8])) >= 7
    pos_learned = {layer: i for i, layer in enumerate(ranked)}
    pos_fixed = {layer: i for i, layer in enumerate(GPU_OWNED_LAYER_RANK)}
    n = 48
    spearman = 1 - 6 * sum((pos_learned[l] - pos_fixed[l]) ** 2 for l in range(n)) / (n * (n * n - 1))
    assert spearman > 0.95, spearman


def test_breadth_ranking_orders_broad_before_narrow_and_empty_last():
    freq = [
        [10, 10, 10, 10],  # broad: all four experts carry the mass
        [40, 0, 0, 0],  # narrow: one expert
        [0, 0, 0, 0],  # never routed
        [20, 20, 0, 0],  # two experts
    ]
    assert lr.rank_layers_by_breadth(freq) == [0, 3, 1, 2]


def test_breadth_ranking_breaks_ties_by_working_set_then_layer_id():
    freq = [
        [50, 50, 0, 0],  # two experts for 90 %, working set 2
        [50, 50, 1, 0],  # two experts for 90 %, working set 3 -> broader
        [50, 50, 0, 0],
    ]
    assert lr.rank_layers_by_breadth(freq) == [1, 0, 2]


def test_cold_experts_lists_the_never_routed_ids_per_layer():
    assert lr.cold_experts([[0, 3, 0], [1, 1, 1]]) == [[0, 2], []]


# ------------------------------------------------------------------ the file


def test_load_returns_none_without_a_file(tmp_path):
    assert lr.load_routing_stats(tmp_path, num_layers=2, num_experts=2) is None


def test_recorder_round_trips_through_the_file(tmp_path):
    rec = lr.RoutingStatsRecorder(tmp_path, num_layers=2, num_experts=3, top_k=2)
    assert rec.total.boots == 1
    assert rec.note([[1, 2, 0], [0, 0, 5]]) == 8
    assert rec.save() is True
    assert rec.save() is False, "nothing new: no rewrite"
    loaded = lr.load_routing_stats(tmp_path, num_layers=2, num_experts=3)
    assert loaded is not None
    assert loaded.freq == [[1, 2, 0], [0, 0, 5]]
    assert loaded.boots == 1 and loaded.top_k == 2 and loaded.routes == 5
    doc = json.loads(lr.routing_stats_path(tmp_path).read_text(encoding="utf-8"))
    assert doc["layer_rank"] == [0, 1]
    assert doc["cold_experts"] == [[2], [0, 1]]
    assert not list(tmp_path.glob("*.tmp"))


def test_recorder_merges_only_new_counts_and_detects_a_reset(tmp_path):
    rec = lr.RoutingStatsRecorder(tmp_path, num_layers=1, num_experts=2)
    rec.note([[4, 1]])
    assert rec.note([[6, 1]]) == 2, "cumulative histogram: only the delta is merged"
    assert rec.total.freq == [[6, 1]]
    # GET /v1/cache/routing?reset=true zeroed the cache histogram; the next snapshot is smaller
    # in some cell, so the whole current histogram is new.
    assert rec.note([[1, 0]]) == 1
    assert rec.total.freq == [[7, 1]]


def test_a_new_boot_halves_the_prior_and_counts_the_boot(tmp_path):
    first = lr.RoutingStatsRecorder(tmp_path, num_layers=1, num_experts=2)
    first.note([[10, 3]])
    assert first.save()
    prior = lr.load_routing_stats(tmp_path, num_layers=1, num_experts=2)
    second = lr.RoutingStatsRecorder(tmp_path, num_layers=1, num_experts=2, prior=prior)
    assert second.total.boots == 2
    assert second.total.freq == [[5, 1]]
    second.note([[2, 2]])
    assert second.total.freq == [[7, 3]]


@pytest.mark.parametrize(
    "doc, why",
    [
        ({"schema": 99, "decode_freq": [[1, 1]]}, "schema"),
        ({"schema": 1, "decode_freq": [[1, 1], [1, 1]]}, "shape"),
        ({"schema": 1}, "decode_freq"),
    ],
)
def test_a_bad_file_is_ignored_not_fatal(tmp_path, doc, why, caplog):
    lr.routing_stats_path(tmp_path).write_text(json.dumps(doc), encoding="utf-8")
    with caplog.at_level("WARNING", logger="freetoken.moe.learned_routing"):
        assert lr.load_routing_stats(tmp_path, num_layers=1, num_experts=2) is None
    assert any("ignoring routing stats" in r.getMessage() for r in caplog.records)


def test_garbage_in_the_file_is_ignored_too(tmp_path):
    lr.routing_stats_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert lr.load_routing_stats(tmp_path, num_layers=1, num_experts=2) is None


def test_an_unwritable_checkpoint_dir_only_warns_once(tmp_path, caplog):
    rec = lr.RoutingStatsRecorder(tmp_path / "missing" / "dir", num_layers=1, num_experts=1)
    rec.note([[1]])
    with caplog.at_level("WARNING", logger="freetoken.moe.learned_routing"):
        assert rec.save() is False
        rec.note([[2]])
        assert rec.save() is False
    assert sum("cannot save routing stats" in r.getMessage() for r in caplog.records) == 1
    assert rec.last_error


def test_recorder_rejects_a_histogram_of_another_shape(tmp_path):
    rec = lr.RoutingStatsRecorder(tmp_path, num_layers=1, num_experts=2)
    with pytest.raises(ValueError, match="shape"):
        rec.note([[1, 2, 3]])


# ------------------------------------------------------------------ the learned rank


def test_learned_rank_needs_enough_routes(tmp_path):
    rec = lr.RoutingStatsRecorder(tmp_path, num_layers=2, num_experts=2)
    rec.note([[10, 10], [20, 0]])
    rec.save()
    ranked, reason = lr.learned_layer_rank(tmp_path, num_layers=2, num_experts=2)
    assert ranked is None and "below" in reason
    rec.note([[lr.MIN_LEARNED_ROUTES, lr.MIN_LEARNED_ROUTES], [20, 0]])
    rec.save()
    ranked, reason = lr.learned_layer_rank(tmp_path, num_layers=2, num_experts=2)
    assert ranked == [0, 1] and "routes per layer" in reason


def test_learned_rank_without_a_file_says_so(tmp_path):
    ranked, reason = lr.learned_layer_rank(tmp_path, num_layers=2, num_experts=2)
    assert ranked is None and "no routing stats" in reason

"""scripts/diag/analyze_predict.py on synthetic predict-log records."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diag" / "analyze_predict.py"


@pytest.fixture(scope="module")
def analyze_predict():
    spec = importlib.util.spec_from_file_location("diag_analyze_predict", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(layer, actual, pred_next=None, pred_next2=None, positions=(0, 1)):
    return {
        "layer": layer,
        "phase": "prefill",
        "positions": list(positions),
        "tokens": [11, 12],
        "actual_top10": actual,
        "pred_next_top20": pred_next or [],
        "pred_next2_top20": pred_next2 or [],
    }


@pytest.fixture
def log(tmp_path):
    """Two tokens, three layers, with hand-checkable overlaps."""
    records = [
        # layer 0 predicts layer 1 (token 0: 1 of 2 hit at k>=2) and layer 2 (both hit at k=3)
        _record(
            0,
            actual=[[1, 7], [3, 8]],
            pred_next=[[9, 1, 5], [9, 9, 9]],
            pred_next2=[[4, 4, 4], [5, 6, 7]],
        ),
        _record(1, actual=[[1, 2], [3, 4]], pred_next=[[5, 5, 5], [5, 5, 5]]),
        _record(2, actual=[[6, 7], [5, 6]]),
    ]
    path = tmp_path / "predict-log-1.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return tmp_path


def _cells(out, prefix):
    for line in out.splitlines():
        if line.strip().startswith(prefix):
            return line.split()
    raise AssertionError(f"no {prefix!r} row in:\n{out}")


def test_recall_at_k_for_the_next_layer(analyze_predict, log, capsys):
    assert analyze_predict.main([str(log), "--ks", "1,2,3"]) == 0
    out = capsys.readouterr().out
    # layer 0: next@1 sees only {9} -> 0.0; next@2 adds expert 1 -> (0.5 + 0)/2 = 0.25
    row = _cells(out, "0 ")
    assert row[0] == "0" and row[1] == "2"
    assert row[2:5] == ["0.000", "0.250", "0.250"]


def test_recall_for_layer_plus_two_and_the_loyalty_baseline(analyze_predict, log, capsys):
    assert analyze_predict.main([str(log), "--ks", "1,2,3"]) == 0
    out = capsys.readouterr().out
    row = _cells(out, "0 ")
    # L+2 actual [[6,7],[5,6]] against [[4,4,4],[5,6,7]]: token 0 misses at every k;
    # token 1 finds expert 5 at k=1 and both of its experts from k=2 on
    assert row[5:8] == ["0.250", "0.500", "0.500"]
    # loyalty: layer 1's [[1,2],[3,4]] inside layer 0's own [[1,7],[3,8]] -> 0.5
    assert row[8] == "0.500"


def test_layer_without_a_successor_is_absent_and_all_row_averages(analyze_predict, log, capsys):
    assert analyze_predict.main([str(log), "--ks", "10"]) == 0
    out = capsys.readouterr().out
    # layer 1's constant [[5,5,5]] guess only ever finds token 1's expert 5
    assert _cells(out, "1 ")[2] == "0.250"
    assert "\n     2 " not in out  # layer 2 has no successor in the log
    assert _cells(out, "ALL")[1] == "0.250"  # layers 0 and 1 both score 0.25 at k=10


def test_empty_and_unjoinable_inputs(analyze_predict, tmp_path, capsys):
    assert analyze_predict.main([str(tmp_path)]) == 2
    assert "no records" in capsys.readouterr().out

    path = tmp_path / "predict-log-9.jsonl"
    path.write_text(json.dumps(_record(0, actual=[[1, 2], [3, 4]])) + "\n", encoding="utf-8")
    assert analyze_predict.main([str(tmp_path)]) == 1
    assert "nothing to score" in capsys.readouterr().out

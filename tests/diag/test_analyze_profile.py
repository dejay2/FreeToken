"""scripts/diag/analyze_profile.py on a hand-built chrome trace."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diag" / "analyze_profile.py"


@pytest.fixture(scope="module")
def analyze_profile():
    spec = importlib.util.spec_from_file_location("diag_analyze_profile", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _range(name, ts, dur, cat="user_annotation"):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "pid": 1, "tid": 1}


def _kernel(name, ts, dur, cat="kernel"):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "pid": 1, "tid": 7}


@pytest.fixture
def trace(tmp_path):
    events = [
        # step one: three device events inside a 1 ms range
        _range("diag.plain_decode_step", 0, 1000),
        _kernel("gemm_kernel", 100, 200),
        _kernel("moe_expert_gemm", 400, 100),
        _kernel("Memcpy HtoD", 600, 50, cat="gpu_memcpy"),
        # the mirrored device-side annotation must not be counted as a second instance
        _range("diag.plain_decode_step", 100, 100, cat="gpu_user_annotation"),
        # step two: a 3 ms range with a nested gather that owns the same kernel
        _range("diag.plain_decode_step", 2000, 3000),
        _range("diag.ple_gather", 2050, 100),
        _kernel("gemm_kernel", 2100, 300),
        # device work outside every range, and a non-complete event
        _kernel("gemm_kernel", 9000, 1000),
        {"ph": "i", "cat": "kernel", "name": "instant", "ts": 5},
    ]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return tmp_path


def test_per_range_wall_and_kernel_time(analyze_profile, trace, capsys):
    assert analyze_profile.main([str(trace)]) == 0
    out = capsys.readouterr().out

    assert "== diag.plain_decode_step" in out
    # two instances of 1 ms and 3 ms; 200+100+50+300 us of device work inside them
    assert "count 2  mean wall 2.000 ms  max wall 3.000 ms  mean gpu-kernel 0.325 ms" in out
    assert "mean kernels/instance 2.0" in out
    # the nested gather counts its own kernel; the columns are per stage, not a partition
    assert "== diag.ple_gather" in out
    assert "count 1  mean wall 0.100 ms  max wall 0.100 ms  mean gpu-kernel 0.300 ms" in out


def test_top_kernels_and_memory_rows(analyze_profile, trace, capsys):
    assert analyze_profile.main([str(trace), "--top", "5"]) == 0
    out = capsys.readouterr().out
    # 200 + 300 us of gemm across the two steps, ahead of the 100 us expert gemm
    assert "gemm_kernel" in out and "0.500" in out
    assert "moe_expert_gemm" in out
    assert "gpu_memcpy" in out  # memory movement is its own row, not folded into "kernel"


def test_kernels_per_step_summary(analyze_profile, trace, capsys):
    assert analyze_profile.main([str(trace)]) == 0
    out = capsys.readouterr().out
    assert "diag.plain_decode_step: instances 2  mean 2.0  min 1  median 3  max 3" in out
    assert "diag.spec_verify_replay: not in this trace" in out


def test_directory_argument_and_missing_trace(analyze_profile, tmp_path, capsys):
    assert analyze_profile.main([str(tmp_path)]) == 2
    assert "no trace at" in capsys.readouterr().out


def test_trace_without_diag_ranges(analyze_profile, tmp_path, capsys):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [_kernel("gemm_kernel", 0, 10)]}), encoding="utf-8")
    assert analyze_profile.main([str(path)]) == 1
    assert "no diag.* ranges" in capsys.readouterr().out

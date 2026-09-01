"""The kernel-profiler seam (freetoken/diag.py): arming, step counting, and disarm.

CPU only -- the seam's state machine is host-side and the real profiler is exercised once,
on CPU activities, for two iterations.
"""

import json
import os

import pytest
import torch

import freetoken.diag as diag


@pytest.fixture(autouse=True)
def _disarmed():
    diag.disable()
    yield
    diag.disable()


class _FakeProfiler:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.steps = 0
        self.traces = []
        self.tables = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def step(self) -> None:
        self.steps += 1

    def export_chrome_trace(self, path) -> None:
        self.traces.append(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"traceEvents": []}, handle)

    def key_averages(self):
        profiler = self

        class _Averages:
            def table(self, sort_by=None, row_limit=None):
                profiler.tables += 1
                return f"sort_by={sort_by} row_limit={row_limit}\n"

        return _Averages()


def _arm(monkeypatch, tmp_path, steps, skip=0):
    made = []
    monkeypatch.setattr(diag, "_make_profiler", lambda: made.append(_FakeProfiler()) or made[-1])
    diag.enable(str(tmp_path), steps=steps, skip=skip)
    return made


def _busy_iteration(name="diag.plain_decode_step"):
    with diag.region(name):
        pass
    diag.profile_step()


def test_disabled_region_is_the_shared_noop():
    assert diag.ENABLED is False
    assert diag.region("diag.plain_decode_step") is diag._NULL
    with diag.region("diag.plain_decode_step") as handle:
        assert handle is diag._NULL
    diag.profile_step()  # must not raise with nothing armed


def test_armed_but_idle_never_starts(monkeypatch, tmp_path):
    made = _arm(monkeypatch, tmp_path, steps=2)
    for _ in range(50):
        diag.profile_step()  # no traffic: no region was ever entered
    assert made == []
    assert not os.path.exists(tmp_path / "trace.json")


def test_boot_time_ranges_do_not_count_as_traffic(monkeypatch, tmp_path):
    """Graph capture runs diag.ple_gather; the loop must not read that as an iteration."""
    made = _arm(monkeypatch, tmp_path, steps=2)
    with diag.region("diag.ple_gather"):  # boot: capture warm-up
        pass
    diag.profile_loop_begin()
    diag.profile_step()  # first loop iteration, idle
    assert made == []
    _busy_iteration()
    assert len(made) == 1


def test_records_exactly_n_iterations_then_disarms(monkeypatch, tmp_path):
    made = _arm(monkeypatch, tmp_path, steps=2)

    _busy_iteration()  # the iteration that ARMS the profiler runs outside it
    assert len(made) == 1 and made[0].entered and diag._session.counted == 0

    _busy_iteration()
    assert diag._session.counted == 1 and not made[0].exited

    _busy_iteration()  # the second recorded iteration closes the session
    assert diag._session.counted == 2 and made[0].exited
    assert (tmp_path / "trace.json").exists()
    assert (tmp_path / "key_averages.txt").exists()
    assert "row_limit=80" in (tmp_path / "key_averages.txt").read_text(encoding="utf-8")

    # never re-arms
    for _ in range(10):
        _busy_iteration()
    assert len(made) == 1 and diag._session.counted == 2


def test_skip_counts_only_non_idle_iterations(monkeypatch, tmp_path):
    made = _arm(monkeypatch, tmp_path, steps=1, skip=2)

    diag.profile_step()  # idle iterations do not consume the skip budget
    diag.profile_step()
    assert made == []

    _busy_iteration()
    _busy_iteration()
    assert made == [], "the first two non-idle iterations are skipped"

    _busy_iteration()
    assert len(made) == 1 and diag._session.counted == 0

    _busy_iteration()
    assert diag._session.counted == 1 and made[0].exited


def test_a_start_failure_disarms_instead_of_retrying(monkeypatch, tmp_path):
    def _boom():
        raise RuntimeError("no profiler here")

    monkeypatch.setattr(diag, "_make_profiler", _boom)
    diag.enable(str(tmp_path), steps=1)
    for _ in range(5):
        _busy_iteration()
    assert diag._session.finished is True


def test_real_cpu_profiler_two_iterations(tmp_path):
    """No CUDA: two iterations through the real torch profiler, then read the trace back."""
    diag.enable(str(tmp_path), steps=2)
    for _ in range(3):  # one to arm, two recorded
        with diag.region("diag.plain_decode_step"):
            with diag.region("diag.ple_gather"):
                torch.randn(64, 64) @ torch.randn(64, 64)
        diag.profile_step()

    trace = tmp_path / "trace.json"
    assert trace.exists() and (tmp_path / "key_averages.txt").exists()
    events = json.loads(trace.read_text(encoding="utf-8"))["traceEvents"]
    names = [event.get("name") for event in events]
    # exactly the recorded iterations: the arming one ran before the profiler existed,
    # and nothing is lost between them (the profiler runs scheduleless, one cycle)
    assert names.count("diag.plain_decode_step") == 2
    assert names.count("diag.ple_gather") == 2

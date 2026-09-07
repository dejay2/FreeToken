"""The governor loop's HTTP edge and the helper's status path, with the server faked."""

from __future__ import annotations

import io
import json
import logging
import urllib.error
from types import SimpleNamespace

import pytest

from freetoken.daemon.settings import governor
from freetoken.daemon.settings.governor import GIB, Action, GovernorLoop, GovernorPolicy
from freetoken.daemon.settings.process_manager import ProcessManager


class _FakeServer:
    """Answers POST /v1/cache/step from a scripted list of (http_status, body) replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def urlopen(self, req, timeout=None):
        self.requests.append(json.loads(req.data.decode("utf-8")))
        status, body = self.replies.pop(0)
        payload = json.dumps(body).encode("utf-8")
        if status != 200:
            raise urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(payload))

        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return payload

        return _Resp()


def _loop(monkeypatch, server: _FakeServer) -> GovernorLoop:
    monkeypatch.setattr(governor.urllib.request, "urlopen", server.urlopen)
    pm = SimpleNamespace(server_status=lambda: {"reachable": True, "state": "serving"})
    loop = GovernorLoop(pm, GovernorPolicy(vram_cushion=2 * GIB, ram_cushion=4 * GIB), http_port=2020)
    loop.last_moe_cache_size = 6144
    return loop


def test_step_request_shape_and_log_line(monkeypatch, caplog):
    server = _FakeServer([(200, {"status": "ok", "applied": "slots", "moe_cache_size": 5632,
                                 "layers": {"owned": 6, "pinned": 42, "disk": 0},
                                 "vram_free_bytes": int(2.6 * GIB), "at_floor": False})])
    loop = _loop(monkeypatch, server)
    with caplog.at_level(logging.INFO, logger="freetoken.daemon.settings.governor"):
        loop._execute_action(Action("vram", "down", ram_tight=True), int(1.2 * GIB), 8 * GIB)
    assert server.requests == [{"axis": "vram", "direction": "down", "ram_tight": True}]
    assert loop.last_action == "governor: vram down -> slots 6144->5632 (free 1.2->2.6 GiB)"
    assert loop.last_layers == {"owned": 6, "pinned": 42, "disk": 0}
    assert loop.status()["layers"]["pinned"] == 42


def test_ram_axis_logs_the_ram_reading(monkeypatch):
    server = _FakeServer([(200, {"status": "ok", "applied": "pinned->disk", "moe_cache_size": 6144,
                                 "layers": {"owned": 6, "pinned": 41, "disk": 1}, "vram_free_bytes": 0})])
    loop = _loop(monkeypatch, server)
    loop._execute_action(Action("ram", "down", ram_tight=True), 3 * GIB, int(3.5 * GIB))
    assert loop.last_action == "governor: ram down -> pinned->disk (free RAM 3.5 GiB)"


def test_unsupported_reply_idles_the_loop(monkeypatch, caplog):
    server = _FakeServer([(503, {"status": "unsupported", "error": "this model's cache does not support runtime rebuild"})])
    loop = _loop(monkeypatch, server)
    with caplog.at_level(logging.WARNING, logger="freetoken.daemon.settings.governor"):
        loop._execute_action(Action("vram", "down"), GIB, 8 * GIB)
    assert loop.enabled is False
    assert loop.status()["enabled"] is False
    assert "unsupported" in caplog.text
    # The idle loop's tick posts nothing more.
    loop._tick()
    assert len(server.requests) == 1


def test_floor_is_logged_once_per_episode(monkeypatch, caplog):
    floor = (200, {"status": "ok", "applied": None, "at_floor": True, "moe_cache_size": 1024,
                   "layers": {"owned": 0, "pinned": 48, "disk": 0}, "vram_free_bytes": GIB})
    floor_after = (200, {**floor[1], "moe_cache_size": 1536})
    server = _FakeServer([floor, floor, (200, {"status": "ok", "applied": "slots", "moe_cache_size": 1536,
                                               "layers": {"owned": 0, "pinned": 48, "disk": 0},
                                               "vram_free_bytes": GIB}), floor_after])
    loop = _loop(monkeypatch, server)
    loop.last_moe_cache_size = 1024
    with caplog.at_level(logging.INFO, logger="freetoken.daemon.settings.governor"):
        for _ in range(4):
            loop._execute_action(Action("vram", "down"), GIB, 8 * GIB)
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert infos == [
        "governor: vram down -> at floor (free 1.0->1.0 GiB)",
        "governor: vram down -> slots 1024->1536 (free 1.0->1.0 GiB)",
        "governor: vram down -> at floor (free 1.0->1.0 GiB)",
    ]


def test_residency_layer_bytes_updates_rung_and_zero_uses_qwen_fallback(monkeypatch):
    class _ResidencyResponse:
        def __init__(self, body):
            self.payload = json.dumps(body).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.payload

    replies = iter((
        {"layers": {}, "moe_cache_size": 6144, "layer_bytes": 987654321},
        {"layers": {}, "moe_cache_size": 6144, "layer_bytes": 0},
    ))

    def residency(url_request, timeout=None):
        return _ResidencyResponse(next(replies))

    monkeypatch.setattr(governor.urllib.request, "urlopen", residency)
    pm = SimpleNamespace(server_status=lambda: {"reachable": True, "state": "serving"})
    loop = GovernorLoop(pm, GovernorPolicy(vram_cushion=2 * GIB, ram_cushion=4 * GIB), http_port=2020)
    loop._query_residency()
    assert loop.policy.rung_bytes == 987654321
    loop._query_residency()
    assert loop.policy.rung_bytes == governor.DEFAULT_RUNG_BYTES


def test_server_down_or_timeout_never_raises(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(governor.urllib.request, "urlopen", boom)
    pm = SimpleNamespace(server_status=lambda: {"reachable": True, "state": "serving"})
    loop = GovernorLoop(pm, GovernorPolicy(vram_cushion=2 * GIB, ram_cushion=4 * GIB))
    loop._execute_action(Action("vram", "down"), GIB, 8 * GIB)
    assert loop.last_action is None
    # A server that is not serving (down, loading, rebuilding) is left alone.
    pm.server_status = lambda: {"reachable": False, "state": "unreachable"}
    loop._tick()


def test_tick_restamps_the_axis_when_the_post_returns(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(governor.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(governor, "read_free_vram_bytes", lambda environ=None: GIB)
    monkeypatch.setattr(governor, "read_free_windows_ram_bytes", lambda: 8 * GIB)

    class _SlowServer(_FakeServer):
        def urlopen(self, req, timeout=None):
            clock["t"] += 8.0  # the rebuild behind the POST takes 8 s
            return super().urlopen(req, timeout)

    ok = (200, {"status": "ok", "applied": "slots", "moe_cache_size": 5632, "vram_free_bytes": GIB})
    server = _SlowServer([ok, ok])
    monkeypatch.setattr(governor.urllib.request, "urlopen", server.urlopen)
    pm = SimpleNamespace(server_status=lambda: {"reachable": True, "state": "serving"})
    loop = GovernorLoop(pm, GovernorPolicy(vram_cushion=2 * GIB, ram_cushion=4 * GIB))
    loop.last_moe_cache_size = 6144
    loop._last_residency_query = clock["t"]  # this test isolates POST timing from the minute refresh
    loop._serving_since = clock["t"] - governor.BOOT_SETTLE_SECONDS - 1  # past the boot settle window
    loop._tick()
    assert len(server.requests) == 1
    assert loop.policy._state["vram"]["last_step_time"] == 108.0
    clock["t"] = 110.0  # 10 s after the step was chosen, 2 s after it finished: too soon
    loop._tick()
    assert len(server.requests) == 1
    clock["t"] = 113.0
    loop._tick()
    assert len(server.requests) == 2


def test_status_path_reads_the_boot_file_and_never_dials_the_server(tmp_path, monkeypatch):
    def no_network(url, *, timeout):
        raise AssertionError(f"status path dialled {url}")

    monkeypatch.setattr(ProcessManager, "_get_json", staticmethod(no_network))
    boot = tmp_path / "boot.ps1"
    boot.write_text("& $launcher -ModelPath 'x' -MemoryGovernor:$false\n", encoding="utf-8")
    process = ProcessManager(
        boot_file=boot, stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log", lock_path=tmp_path / "gpu.lock",
        readiness=lambda: {"state": "serving"}, stats=lambda: {}, gpu_probe=lambda: {},
        platform_windows=False, linux_stop=lambda *_a, **_k: {"ok": True},
        launch_builder=lambda *_a, **_k: SimpleNamespace(argv=["python"], env={}, notes=(), command_line=lambda: "python"),
        popen=lambda *_a, **_k: SimpleNamespace(poll=lambda: None),
    )
    status = process.governor_status()
    assert status["layers"] == {"owned": 0, "pinned": 0, "disk": 0}
    assert status["last_action"] is None
    assert isinstance(status["enabled"], bool)


def test_start_governor_without_settings_reads_the_boot_file(tmp_path, monkeypatch):
    """The helper entry point starts the governor with no snapshot (adopted server after a
    helper restart); it must read the cushions from the boot file and start the loop."""
    from freetoken.daemon.settings import governor as gov
    from freetoken.daemon.settings.process_manager import ProcessManager

    started = {}

    class FakeLoop:
        def __init__(self, pm, policy, http_port=2020):
            started["policy"] = policy
            started["loop"] = self
            self.policy = policy
            self.enabled = None

        def start(self):
            started["running"] = True

        def stop(self):
            started["running"] = False

        def status(self):
            return {}

    monkeypatch.setattr(gov, "GovernorLoop", FakeLoop)
    import shutil
    from pathlib import Path

    from freetoken.daemon.settings.boot_parser import BootFile

    boot = tmp_path / "boot-2020.ps1"
    shutil.copy2(Path(__file__).parents[2] / "boot-2020.ps1", boot)
    BootFile(boot).save({
        "MemoryGovernor": True,
        "GovernorVRAMFreeGB": 2.5,
        "GovernorRAMFreeGB": 6,
        "GovernorUpMarginGB": 0.75,
        "GovernorStepIntervalS": 7,
        "GovernorUpHoldS": 90,
        "GovernorMaxHoldS": 900,
        "GovernorPostUpGraceS": 15,
        "GovernorRAMRungsBeforeUp": 3,
        "GovernorVRAMRungsBeforeUp": 2,
    })
    pm = ProcessManager(boot_file=boot, stop_script=tmp_path / "stop.ps1", log_path=tmp_path / "log", lock_path=tmp_path / "lock", port=2020)
    pm.start_governor()
    try:
        assert started.get("running") is True
        policy = started["policy"]
        assert policy.vram_cushion == int(round(2.5 * gov.GIB))
        assert policy.ram_cushion == int(round(6 * gov.GIB))
        assert policy.margin == int(round(0.75 * gov.GIB))
        assert policy.step_interval == 7.0
        assert policy.up_hold == 90.0
        assert policy.max_hold == 900.0
        assert policy.post_up_grace == 15.0
        assert policy.ram_rungs_before_up == 3
        assert policy.vram_rungs_before_up == 2
    finally:
        pm.stop_governor()


def test_apply_governor_settings_swaps_running_policy_without_server_call(tmp_path, caplog):
    calls = []
    pm = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        readiness=lambda: calls.append("readiness") or {"state": "serving"},
        stats=lambda: calls.append("stats") or {},
        gpu_probe=lambda: calls.append("gpu") or {},
        platform_windows=False,
    )
    old_policy = GovernorPolicy(vram_cushion=2 * GIB, ram_cushion=4 * GIB)
    loop = GovernorLoop(pm, old_policy)
    loop.enabled = True
    pm._governor = loop

    with caplog.at_level(logging.INFO, logger="freetoken.daemon.settings.process_manager"):
        pm.apply_governor_settings({
            "MemoryGovernor": True,
            "GovernorVRAMFreeGB": 3.0,
            "GovernorRAMFreeGB": 7.0,
            "GovernorUpMarginGB": 1.25,
            "GovernorStepIntervalS": 9.0,
            "GovernorUpHoldS": 12.0,
            "GovernorMaxHoldS": 120.0,
            "GovernorPostUpGraceS": 18.0,
            "GovernorRAMRungsBeforeUp": 4,
            "GovernorVRAMRungsBeforeUp": 3,
        })

    assert loop.policy is not old_policy
    assert loop.policy.vram_cushion == int(3 * GIB)
    assert loop.policy.ram_cushion == int(7 * GIB)
    assert loop.policy.margin == int(1.25 * GIB)
    assert loop.policy.step_interval == 9.0
    assert loop.policy.up_hold == 12.0
    assert loop.policy.max_hold == 120.0
    assert loop.policy.post_up_grace == 18.0
    assert loop.policy.ram_rungs_before_up == 4
    assert loop.policy.vram_rungs_before_up == 3
    assert calls == []
    assert "governor settings applied" in caplog.text


def test_boot_settle_window_drops_down_steps_but_not_up_steps(monkeypatch):
    """Right after the server starts serving, Windows free memory sags for about a minute
    (2026-09-07 21:01:57: seven spills with nothing else running). Down steps wait out the
    settle window; up steps and a squeeze that persists past the window still act."""
    server = _FakeServer([])
    loop = _loop(monkeypatch, server)
    monkeypatch.setattr(governor, "read_free_vram_bytes", lambda: int(2.5 * GIB))  # card: no step either way
    monkeypatch.setattr(governor, "read_free_windows_ram_bytes", lambda: 1 * GIB)  # below the 4 GiB cushion
    executed: list[Action] = []
    monkeypatch.setattr(loop, "_execute_action", lambda a, fv, fr: executed.append(a))
    monkeypatch.setattr(loop, "_query_residency", lambda: None)
    clock = {"t": 1000.0}
    monkeypatch.setattr(governor.time, "monotonic", lambda: clock["t"])

    loop._tick()  # first serving tick starts the window; the RAM axis wants "down" but must wait
    assert executed == []
    clock["t"] += governor.BOOT_SETTLE_SECONDS - 1
    loop._tick()
    assert executed == []
    clock["t"] += 2
    loop._tick()  # past the window the persisting squeeze is acted on
    assert [a.direction for a in executed] == ["down"]

    # up steps are never delayed by the window: fresh loop, memory comfortably free
    executed.clear()
    loop2 = _loop(monkeypatch, server)
    monkeypatch.setattr(loop2, "_execute_action", lambda a, fv, fr: executed.append(a))
    monkeypatch.setattr(loop2, "_query_residency", lambda: None)
    monkeypatch.setattr(governor, "read_free_windows_ram_bytes", lambda: 40 * GIB)
    monkeypatch.setattr(governor, "read_free_vram_bytes", lambda: 12 * GIB)
    clock["t"] = 5000.0
    loop2._tick()
    clock["t"] += loop2.policy.up_hold + 1  # the policy's own hold, well inside the settle window
    loop2._tick()
    assert executed and all(a.direction == "up" for a in executed)

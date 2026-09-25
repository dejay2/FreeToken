"""The settings helper's side of sleep: the proxy, the route, the watchdog, the governor and reclaim."""

from __future__ import annotations

import io
import json
import urllib.error
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.settings import governor
from freetoken.daemon.settings import memory_reclaim as reclaim
from freetoken.daemon.settings import process_manager as pm_module
from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.governor import GIB, GovernorLoop, GovernorPolicy
from freetoken.daemon.settings.process_manager import ProcessManager
from tests.settings.test_crash_watchdog import _wired


class Reply:
    def __init__(self, status, body):
        self.status, self._body = status, json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def manager(tmp_path, readiness=lambda: {"state": "serving"}):
    return ProcessManager(boot_file=tmp_path / "boot.ps1", stop_script=tmp_path / "stop.ps1",
                          log_path=tmp_path / "server.log", lock_path=tmp_path / "gpu.lock",
                          runner=lambda *a, **k: None, readiness=readiness, sleep=lambda _: None,
                          poll_interval=0)


# ---- the proxy ---------------------------------------------------------------


def test_sleep_server_posts_to_the_model_server_and_reports_the_http_status(tmp_path, monkeypatch):
    seen = []

    def urlopen(request, timeout=None):
        seen.append((request.full_url, request.get_method(), timeout))
        if request.full_url.endswith("/v1/wake"):
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, io.BytesIO(
                json.dumps({"status": "rejected", "error": "close the game"}).encode()))
        return Reply(200, {"status": "ok", "asleep": True})

    monkeypatch.setattr(pm_module.urllib.request, "urlopen", urlopen)
    pm = manager(tmp_path)
    assert pm.sleep_server("sleep") == {"status": "ok", "asleep": True, "httpStatus": 200}
    assert pm.sleep_server("wake") == {"status": "rejected", "error": "close the game", "httpStatus": 503}
    assert seen[0] == (f"http://127.0.0.1:{pm.port}/v1/sleep", "POST", 330.0)
    with pytest.raises(ValueError):
        pm.sleep_server("nap")


def test_sleep_server_says_unreachable_when_nothing_answers(tmp_path, monkeypatch):
    def refused(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))

    monkeypatch.setattr(pm_module.urllib.request, "urlopen", refused)
    assert manager(tmp_path).sleep_server("sleep")["httpStatus"] == 0


def test_sleep_server_keeps_a_non_json_error_page_as_failed(tmp_path, monkeypatch):
    def html_error(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 502, "bad gateway", {}, io.BytesIO(b"<html>oops</html>"))

    monkeypatch.setattr(pm_module.urllib.request, "urlopen", html_error)
    assert manager(tmp_path).sleep_server("wake") == {"status": "failed", "httpStatus": 502}


# ---- the route ---------------------------------------------------------------


def test_the_server_route_proxies_sleep_and_wake_without_a_job(tmp_path):
    pm = manager(tmp_path)
    pm.sleep_server = lambda action: {"status": "ok" if action == "sleep" else "rejected",
                                      "httpStatus": 200 if action == "sleep" else 503}
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -Port 2020\n", encoding="utf-8")
    client = TestClient(create_app(boot_file=boot, process_manager=pm, log_path=tmp_path / "server.log",
                                   static_path=tmp_path / "missing.html"))
    sleep = client.post("/api/server/sleep")
    assert sleep.status_code == 200 and sleep.json()["status"] == "ok"
    wake = client.post("/api/server/wake")
    assert wake.status_code == 503 and wake.json()["status"] == "rejected"
    assert pm.current_job() is None
    assert client.post("/api/server/nap").status_code == 422


def test_the_server_route_answers_503_when_the_model_server_is_unreachable(tmp_path):
    pm = manager(tmp_path)
    pm.sleep_server = lambda action: {"status": "unreachable", "error": "refused", "httpStatus": 0}
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -Port 2020\n", encoding="utf-8")
    client = TestClient(create_app(boot_file=boot, process_manager=pm, log_path=tmp_path / "server.log",
                                   static_path=tmp_path / "missing.html"))
    response = client.post("/api/server/sleep")
    assert response.status_code == 503 and response.json()["status"] == "unreachable"


# ---- the watchdog ------------------------------------------------------------


def test_the_watchdog_treats_a_sleeping_server_as_alive(tmp_path):
    state = {"state": "sleeping"}
    manager_, dog = _wired(tmp_path, lambda: dict(state))
    for _ in range(5):
        dog.tick()
    assert dog.armed is True and dog.misses == 0 and manager_.current_job() is None


def test_the_watchdog_still_restarts_a_server_that_dies_asleep(tmp_path, monkeypatch):
    from freetoken.daemon.settings import watchdog as wd

    monkeypatch.setattr(wd, "_server_pids", lambda pm: set())
    monkeypatch.setattr(wd, "capture_incident", lambda **kwargs: None, raising=False)
    state = {"state": "sleeping"}
    manager_, dog = _wired(tmp_path, lambda: dict(state))
    dog.tick()
    assert dog.armed is True, "a sleeping server is adopted"
    state["state"] = "unreachable"
    for _ in range(3):
        dog.tick()
    job = manager_.current_job()
    assert job is not None and job["action"] == "start"


# ---- the governor ------------------------------------------------------------


def _asleep_loop(monkeypatch, free_ram):
    pm = SimpleNamespace(server_status=lambda: {"reachable": True, "state": "sleeping"})
    loop = GovernorLoop(pm, GovernorPolicy(vram_cushion=2 * GIB, ram_cushion=4 * GIB), http_port=2020)
    monkeypatch.setattr(governor, "_is_wsl", lambda: True)
    monkeypatch.setattr(governor, "_read_proc_meminfo_available", lambda: 80 * GIB)
    monkeypatch.setattr(governor, "read_free_windows_ram_bytes", lambda **_kwargs: free_ram)
    monkeypatch.setattr(governor, "read_free_vram_bytes",
                        lambda: pytest.fail("the governor must not look at the card while asleep"))
    executed = []
    monkeypatch.setattr(loop, "_execute_action", lambda action, fv, fr: executed.append(action))
    # The real controller reads this host's /proc/meminfo and may spawn a probe subprocess.
    reclaim_ticks = []
    loop.reclaimer = SimpleNamespace(tick=lambda **kw: reclaim_ticks.append(kw), close=lambda: None)
    return loop, executed, reclaim_ticks


def test_asleep_the_governor_only_spills_to_the_ssd_under_ram_pressure(monkeypatch):
    loop, executed, _ = _asleep_loop(monkeypatch, free_ram=1 * GIB)  # a game squeezes Windows RAM
    loop._tick()
    assert [(a.axis, a.direction) for a in executed] == [("ram", "down")]


def test_asleep_the_governor_never_recalls_even_with_ram_to_spare(monkeypatch):
    loop, executed, _ = _asleep_loop(monkeypatch, free_ram=60 * GIB)
    for _ in range(3):
        loop._tick()
    assert executed == []


def test_asleep_the_governor_keeps_reclaim_running_and_restarts_the_settle_window(monkeypatch):
    loop, _, reclaim_ticks = _asleep_loop(monkeypatch, free_ram=60 * GIB)
    loop._serving_since = 1.0
    loop._tick()
    assert loop._serving_since is None, "a wake is treated like a boot: the settle window starts again"
    assert [t["enabled"] for t in reclaim_ticks] == [True]
    assert reclaim_ticks[0]["windows"] == 60 * GIB


# ---- the temporary reclaim watcher -------------------------------------------


def test_the_temporary_reclaim_watcher_stays_enabled_while_sleeping(monkeypatch):
    records = [
        {"governor": {"enabled": True, "free_windows_ram_gb": 4, "free_linux_ram_gb": 30},
         "server": {"reachable": True, "state": "sleeping"}},
        {"settings": {"GovernorRAMFreeGB": 4, "GovernorRAMRungsBeforeUp": 1, "GovernorUpMarginGB": 0.5}},
        {"governor": {"reclaim": {"state": "idle"}}},
    ]
    monkeypatch.setattr(reclaim.urllib.request, "urlopen",
                        lambda *a, **kw: nullcontext(io.BytesIO(json.dumps(records.pop(0)).encode())))
    calls = []

    class Controller:
        def __init__(self, port):
            pass

        def tick(self, **kw):
            calls.append(kw)

        def status(self):
            return {"state": "reclaiming", "running": True}

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(reclaim, "ReclaimController", Controller)
    monkeypatch.setattr(reclaim.time, "sleep", lambda s: None)
    monkeypatch.setattr(reclaim, "is_wsl", lambda: True)
    monkeypatch.delenv("FREETOKEN_CACHE_RECLAIM", raising=False)
    reclaim.watch(2031, 2020)
    assert calls[0]["enabled"] is True
    assert calls[-1] == "closed"

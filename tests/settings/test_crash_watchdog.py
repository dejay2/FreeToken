"""Crash watchdog: restart a server that dies unasked, never one the page stopped, capped per hour."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from freetoken.daemon.settings import linux_launch as ll
from freetoken.daemon.settings import watchdog as wd
from freetoken.daemon.settings.boot_parser import BootFile
from freetoken.daemon.settings.process_manager import ProcessManager
from freetoken.daemon.settings.watchdog import CrashWatchdog

REAL_BOOT = Path(__file__).parents[2] / "boot-2020.ps1"
_REAL_SERVER_PIDS = wd._server_pids  # captured before the autouse fixture stubs it


class FakeProcess:
    pid = 1234
    returncode = None

    def poll(self):
        return self.returncode


class FakeLaunch:
    argv = ["py", "-m", "freetoken.cli", "serve"]
    env: dict = {}
    notes = ()

    def command_line(self):
        return " ".join(self.argv)


class ManualExecutor:
    def __init__(self):
        self.pending = []

    def submit(self, function, *args):
        self.pending.append((function, args))

    def run_next(self):
        function, args = self.pending.pop(0)
        function(*args)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def _no_server_processes(monkeypatch):
    """By default the dead server left no process behind (the shipped LimitCORE=0 case)."""
    monkeypatch.setattr(wd, "_server_pids", lambda pm: set())


def _manager(tmp_path, readiness, executor=None):
    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        popen=lambda *args, **kwargs: FakeProcess(),
        readiness=readiness,
        sleep=lambda _: None,
        poll_interval=0,
        platform_windows=False,
        executor=executor or ManualExecutor(),
        launch_builder=lambda settings, **kwargs: FakeLaunch(),
        linux_stop=lambda *a, **k: {"ok": True, "killed": []},
    )
    return manager


def _wired(tmp_path, readiness, executor=None, clock=None):
    """The production wiring: the manager owns the watchdog, so its own jobs go through disarm()."""
    manager = _manager(tmp_path, readiness, executor)
    clock = clock or Clock()
    dog = CrashWatchdog(manager, interval=0, misses_needed=3, max_restarts_per_hour=3, monotonic=clock, wall_now=clock)
    manager._watchdog = dog
    return manager, dog


def test_starts_the_server_after_three_dead_probes_once_armed(tmp_path):
    state = {"state": "serving"}
    manager, dog = _wired(tmp_path, lambda: dict(state))
    dog.tick()
    assert dog.armed is True, "a serving server is adopted"
    state["state"] = "unreachable"
    dog.tick(); dog.tick()
    assert manager.current_job() is None, "two misses are not enough"
    dog.tick()
    job = manager.current_job()
    assert job is not None and job["action"] == "start", "nothing left on the port: a plain start"
    assert dog.status()["restarts_last_hour"] == 1
    assert "starting it" in dog.status()["last_reason"]
    assert dog.armed is True, "the watchdog's own job keeps it armed so a failed reboot is still watched"


def test_leftover_processes_mean_a_restart_after_twice_the_patience(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "_server_pids", lambda pm: {4242})
    state = {"state": "failed"}
    manager, dog = _wired(tmp_path, lambda: dict(state))
    dog.armed = True
    for _ in range(5):
        dog.tick()
    assert manager.current_job() is None, "a server with processes gets six probes, not three"
    dog.tick()
    job = manager.current_job()
    assert job is not None and job["action"] == "restart", "the stop path clears the leftovers first"


def test_a_page_stop_disarms_and_a_restart_rearms_on_serving(tmp_path):
    state = {"state": "serving"}
    executor = ManualExecutor()
    manager, dog = _wired(tmp_path, lambda: dict(state), executor)
    dog.tick()
    assert dog.armed
    manager.start("stop")
    assert dog.armed is False and dog.last_reason == "stopped by the page"
    executor.run_next()
    assert manager._jobs[next(iter(manager._jobs))].stage == "stopped"
    state["state"] = "unreachable"
    for _ in range(6):
        dog.tick()
    assert manager.current_job() is None, "never restart behind a Stop"
    state["state"] = "serving"
    manager.start("start", settings={"ModelPath": "/models/demo"})
    executor.run_next()  # runs the start job to serving, which arms the watchdog
    assert manager.current_job() is None, manager._jobs
    assert dog.armed is True


def test_a_stop_pressed_during_the_watchdogs_own_reboot_wins(tmp_path):
    state = {"state": "unreachable"}
    executor = ManualExecutor()
    manager, dog = _wired(tmp_path, lambda: dict(state), executor)
    dog.armed = True
    for _ in range(3):
        dog.tick()
    job_id = manager.current_job()["jobId"]
    assert dog.armed is True
    assert manager.start("stop") == job_id, "the Stop cancels the in-flight reboot"
    assert dog.armed is False and dog.last_reason == "stopped by the page"
    executor.run_next()
    assert manager._jobs[job_id].stage == "stopped"
    for _ in range(6):
        dog.tick()
    assert manager.current_job() is None, "never reboot behind a Stop"


def test_server_pids_asks_proc_for_this_port_and_is_unknown_on_windows(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ll, "find_server_pids", lambda port: seen.append(port) or {77})
    manager = _manager(tmp_path, lambda: {"state": "serving"})
    assert _REAL_SERVER_PIDS(manager) == {77} and seen == [2020]
    manager.platform_windows = True
    assert _REAL_SERVER_PIDS(manager) is None, "the Windows helper cannot tell; the watchdog then always restarts"


def test_a_rejected_restart_request_does_not_disarm(tmp_path):
    manager, dog = _wired(tmp_path, lambda: {"state": "serving"})
    dog.armed = True
    manager.start("start", settings={"ModelPath": "/models/demo"})  # queued, stays active
    with pytest.raises(Exception):
        manager.start("restart")
    assert dog.armed is True, "only an accepted request disarms"


def test_a_human_job_resets_the_count(tmp_path):
    calls = []
    state = {"state": "unreachable"}

    def readiness():
        calls.append(1)
        return dict(state)

    manager, dog = _wired(tmp_path, readiness)
    dog.armed = True
    dog.tick(); dog.tick()
    assert dog.misses == 2
    manager.start("start", settings={"ModelPath": "/models/demo"})  # queued on the manual executor: active
    n = len(calls)
    dog.tick(); dog.tick(); dog.tick()
    assert len(calls) == n, "the running job owns readiness"
    assert dog.misses == 0, "a failed human Start must not be re-run ten seconds later"


def test_busy_states_are_not_deaths_and_reset_the_count(tmp_path):
    state = {"state": "unreachable"}
    manager, dog = _wired(tmp_path, lambda: dict(state))
    dog.armed = True
    dog.tick(); dog.tick()
    state["state"] = "rebuilding"
    dog.tick()
    assert dog.misses == 0
    state["state"] = "unreachable"
    dog.tick(); dog.tick()
    assert manager.current_job() is None


def test_three_failed_reboots_in_an_hour_give_up_and_the_budget_returns(tmp_path):
    clock = Clock()
    executor = ManualExecutor()
    state = {"state": "unreachable"}
    manager, dog = _wired(tmp_path, lambda: dict(state), executor, clock)
    dog.armed = True
    for n in range(3):
        for _ in range(3):
            dog.tick()
        assert manager.current_job() is not None, f"restart {n + 1}"
        # the boot fails: the job ends failed, nobody re-arms, and the watchdog is still armed
        manager._jobs[manager._active_id].stage = "failed"
        manager._active_id = None
        executor.pending.clear()
        assert dog.armed is True
        clock.now += 60
    for _ in range(3):
        dog.tick()
    assert manager.current_job() is None
    assert dog.gave_up is True and "gave up" in dog.last_reason
    clock.now += 3601
    for _ in range(3):
        dog.tick()
    assert manager.current_job() is None, "gave up stays until the server is seen serving again"
    state["state"] = "serving"
    dog.tick()
    assert dog.gave_up is False and dog.armed is True
    state["state"] = "unreachable"
    for _ in range(3):
        dog.tick()
    assert manager.current_job() is not None, "an hour later the budget is back"


def test_disabled_watchdog_never_restarts_and_save_flips_it(tmp_path):
    manager = _manager(tmp_path, lambda: {"state": "unreachable"})
    manager.start_watchdog({"FREETOKEN_AUTO_RESTART": "0"}, interval=3600)
    dog = manager._watchdog
    assert dog.enabled is False
    dog.armed = True
    for _ in range(4):
        dog.tick()
    assert manager.current_job() is None
    dog.misses = 2
    manager.apply_watchdog_settings({"FREETOKEN_AUTO_RESTART": True})
    assert dog.enabled is True and dog.misses == 0, "re-enabling starts the count afresh"
    assert manager.watchdog_status()["enabled"] is True
    manager.stop_watchdog()
    assert manager.watchdog_status()["armed"] is False


def test_auto_restart_dial_round_trips_through_a_real_boot_file(tmp_path):
    target = tmp_path / "boot-2020.ps1"
    shutil.copy2(REAL_BOOT, target)
    boot = BootFile(target)
    before = boot.load()
    assert before["FREETOKEN_AUTO_RESTART"] == "1", "absent from an older boot file means on"
    assert ProcessManager._auto_restart_enabled(before) is True
    saved = boot.save({"FREETOKEN_AUTO_RESTART": False, "FREETOKEN_DIAGNOSTIC_MODE": True})
    reloaded = BootFile(target).load()
    assert reloaded["FREETOKEN_AUTO_RESTART"] == "0" and reloaded["FREETOKEN_DIAGNOSTIC_MODE"] == "1", "env toggles read back as text"
    assert ProcessManager._auto_restart_enabled(reloaded) is False
    content = target.read_text()
    assert "FREETOKEN_AUTO_RESTART = '0'" in content and "FREETOKEN_DIAGNOSTIC_MODE = '1'" in content
    assert saved["FREETOKEN_AUTO_RESTART"] in (False, "0")


def _facts():
    return ll.ModelFacts(is_moe=True, expert_count=24576, max_context=262144, has_ple=True, has_vision=True,
                         has_mtp=True, parking_supported=True, model_type="qwen4_exp")


def test_diagnostic_mode_maps_to_blocking_launches_and_graphs_off():
    plan = ll.build_launch({"ModelPath": "/models/demo", "CudaGraphMaxBS": 4, "FREETOKEN_DIAGNOSTIC_MODE": True,
                            "FREETOKEN_MTP_SPECULATE": True},
                           python="py", base_env={}, facts=_facts(), wsl=False)
    argv = plan.argv
    assert argv[argv.index("--cuda-graph-max-bs") + 1] == "0"
    assert plan.env["CUDA_LAUNCH_BLOCKING"] == "1" and plan.env["FREETOKEN_MTP_SPEC_GRAPH"] == "0"
    assert any("Diagnostic mode" in note for note in plan.notes)
    plain = ll.build_launch({"ModelPath": "/models/demo", "CudaGraphMaxBS": 4}, python="py", base_env={}, facts=_facts(), wsl=False)
    assert plain.argv[plain.argv.index("--cuda-graph-max-bs") + 1] == "4"
    assert "CUDA_LAUNCH_BLOCKING" not in plain.env and plain.env["FREETOKEN_DIAGNOSTIC_MODE"] == "0"


def test_running_server_release_uses_the_engines_own_layer_account(tmp_path, monkeypatch):
    manager = _manager(tmp_path, lambda: {"state": "serving"})
    manager._stats = lambda: {"vram_bytes": 30 << 30}
    monkeypatch.setattr(ProcessManager, "_get_json", staticmethod(lambda url, *, timeout: {"pinned": 47, "layer_bytes": 1419509760} if "residency" in url else {}))
    release = manager.running_server_release()
    assert release == {"ram_bytes": 47 * 1419509760, "vram_bytes": 30 << 30}
    monkeypatch.setattr(ProcessManager, "_get_json", staticmethod(lambda url, *, timeout: (_ for _ in ()).throw(OSError("down"))))
    manager._stats = lambda: (_ for _ in ()).throw(OSError("down"))
    assert manager.running_server_release() == {"ram_bytes": 0, "vram_bytes": 0}


# ---- stuck maintenance (2026-09-09 16:25: 11 min 44 s of "rebuilding" with GPU at 0%) ------


def test_a_server_that_reports_its_operation_stuck_is_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "_server_pids", lambda pm: {4242})  # the processes are still there
    doc = {"state": "serving"}
    manager, dog = _wired(tmp_path, lambda: dict(doc))
    dog.tick()
    assert dog.armed
    doc.update(state="rebuilding", maintenance={"stuck": False, "age_s": 12.0})
    dog.tick(); dog.tick()
    assert dog.misses == 0, "a rebuild in progress is not a death"
    doc["maintenance"] = {"stuck": True, "age_s": 400.0}
    for _ in range(5):
        dog.tick()
    assert manager.current_job() is None
    dog.tick()
    job = manager.current_job()
    assert job is not None and job["action"] == "restart"
    assert "stuck in maintenance" in dog.status()["last_reason"]


def test_rebuilding_for_longer_than_the_limit_counts_as_dead_even_without_a_verdict(tmp_path):
    clock = Clock()
    doc = {"state": "rebuilding"}  # an older API with no maintenance block, or a wedged loop
    manager, dog = _wired(tmp_path, lambda: dict(doc), clock=clock)
    dog.armed = True
    for _ in range(10):
        clock.now += 59.0
        dog.tick()
    assert dog.misses == 0 and manager.current_job() is None, "under the limit it is just busy"
    clock.now += 80.0  # past 600 s of continuous rebuilding (the clock started at the first tick)
    dog.tick(); dog.tick()
    assert dog.misses == 2
    dog.tick()
    job = manager.current_job()
    assert job is not None and job["action"] == "start"
    assert "rebuilding for" in dog.status()["last_reason"]


def test_a_rebuild_that_finishes_resets_the_rebuilding_clock(tmp_path):
    clock = Clock()
    doc = {"state": "rebuilding"}
    manager, dog = _wired(tmp_path, lambda: dict(doc), clock=clock)
    dog.armed = True
    clock.now += 500.0
    dog.tick()
    doc["state"] = "serving"
    dog.tick()
    doc["state"] = "rebuilding"
    clock.now += 500.0
    dog.tick()
    assert dog.misses == 0 and manager.current_job() is None

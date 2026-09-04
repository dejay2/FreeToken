from __future__ import annotations

import signal
import threading

import pytest

from freetoken.daemon.settings.process_manager import ProcessManager


class FakeProcess:
    def __init__(self, returncode=None):
        self.pid = 1234
        self.returncode = returncode
        self.stdout = None
        self.stderr = None

    def poll(self):
        return self.returncode


def test_restart_is_serialized_and_returns_progress_job(tmp_path):
    calls = []
    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        popen=lambda *args, **kwargs: FakeProcess(),
        readiness=lambda: {"state": "serving", "geometry": {}},
        sleep=lambda _: None,
        poll_interval=0,
    )
    first = manager.start("restart")
    with pytest.raises(RuntimeError, match="active"):
        manager.start("stop")
    assert first.startswith("job-")
    assert manager.job(first)["stage"] in {"serving", "booting", "stopping", "waiting_for_gpu_lock"}


def test_readiness_uses_cache_status_and_reports_failure(tmp_path):
    statuses = iter([{"state": "loading"}, {"state": "serving"}])
    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *args, **kwargs: None,
        popen=lambda *args, **kwargs: FakeProcess(),
        readiness=lambda: next(statuses),
        sleep=lambda _: None,
        poll_interval=0,
    )
    assert manager.wait_until_serving(timeout=1)["state"] == "serving"


def test_foreign_gpu_lock_is_waited_on(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text("other-job", encoding="utf-8")
    checks = []
    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=lock,
        runner=lambda *args, **kwargs: None,
        popen=lambda *args, **kwargs: FakeProcess(),
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: checks.append(True),
        poll_interval=0,
    )
    assert manager.acquire_gpu_lock("mine", timeout=0.001) is False
    assert checks


def test_stop_command_contains_timeout(tmp_path):
    commands = []
    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *args, **kwargs: commands.append(args),
        sleep=lambda _: None,
    )
    manager.run_stop()
    command = commands[0][0]
    assert "-TimeoutSeconds" in command
    assert "-Port" in command


def test_failure_signatures_are_detected():
    assert ProcessManager.failure_from_log("RuntimeError: broken")
    assert ProcessManager.failure_from_log("CUDA out of memory")
    assert ProcessManager.failure_from_log("ordinary progress") is None


def test_lock_is_released_after_context(tmp_path):
    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *args, **kwargs: None,
    )
    with manager.gpu_lock("mine"):
        assert (tmp_path / "gpu.lock").read_text() == "mine"
    assert not (tmp_path / "gpu.lock").exists()

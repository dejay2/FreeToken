from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from freetoken.daemon.settings.process_manager import LifecycleError, ProcessManager


class FakeProcess:
    def __init__(self, returncode=None):
        self.pid = 1234
        self.returncode = returncode
        self.stdout = None
        self.stderr = None

    def poll(self):
        return self.returncode


@dataclass
class FakeLaunch:
    argv: list[str]
    env: dict[str, str]
    notes: tuple[str, ...] = ()

    def command_line(self) -> str:
        return " ".join(self.argv)


class ManualExecutor:
    def __init__(self):
        self.pending = []

    def submit(self, function, *args):
        self.pending.append((function, args))

    def run_next(self):
        function, args = self.pending.pop(0)
        function(*args)


def _wait_for_idle(manager: ProcessManager, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while manager.current_job() is not None:
        if time.monotonic() >= deadline:
            raise AssertionError(f"lifecycle job did not finish: {manager.current_job()!r}")
        time.sleep(0.001)


def test_restart_is_serialized_and_stop_returns_the_same_progress_job(tmp_path):
    stop_entered = threading.Event()
    release_stop = threading.Event()
    calls = []

    def runner(*args, **kwargs):
        if "-File" in args[0]:
            calls.append((args, kwargs))
        stop_entered.set()
        assert release_stop.wait(1.0)
        return type("Result", (), {"returncode": 0, "stdout": "[]", "stderr": ""})()

    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=runner,
        popen=lambda *args, **kwargs: FakeProcess(),
        readiness=lambda: {"state": "serving", "geometry": {}},
        sleep=lambda _: None,
        poll_interval=0,
        platform_windows=True,
    )
    first = manager.start("restart")
    assert stop_entered.wait(1.0)
    assert manager.start("stop") == first
    assert manager.start("stop") == first
    with pytest.raises(LifecycleError, match="active"):
        manager.start("start")
    assert first.startswith("job-")
    assert manager.job(first)["stage"] == "stopping"
    release_stop.set()
    _wait_for_idle(manager)
    assert manager.job(first)["stage"] == "stopped"
    assert len(calls) == 1


def test_stop_during_gpu_lock_wait_cancels_without_queueing(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text("foreign-job", encoding="utf-8")
    lock_waiting = threading.Event()
    release_wait = threading.Event()

    def sleep(_):
        lock_waiting.set()
        assert release_wait.wait(1.0)

    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=lock,
        runner=lambda *args, **kwargs: None,
        readiness=lambda: {"state": "serving"},
        sleep=sleep,
        poll_interval=0,
        lock_poll_interval=0,
        platform_windows=True,
    )
    first = manager.start("start")
    assert lock_waiting.wait(1.0)
    assert manager.start("stop") == first
    with pytest.raises(LifecycleError, match="active"):
        manager.start("restart")
    release_wait.set()
    _wait_for_idle(manager)
    assert manager.job(first)["stage"] == "stopped"


def test_stop_during_spawn_stops_a_late_spawned_process(tmp_path):
    spawn_entered = threading.Event()
    release_spawn = threading.Event()
    stop_calls = []

    def popen(*args, **kwargs):
        spawn_entered.set()
        assert release_spawn.wait(1.0)
        return FakeProcess()

    def runner(*args, **kwargs):
        if "-File" in args[0]:
            stop_calls.append((args, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": "[]", "stderr": ""})()

    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=runner,
        popen=popen,
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: None,
        poll_interval=0,
        platform_windows=True,
    )
    first = manager.start("start")
    assert spawn_entered.wait(1.0)
    assert manager.start("stop") == first
    release_spawn.set()
    _wait_for_idle(manager)
    assert manager.job(first)["stage"] == "stopped"
    assert len(stop_calls) == 1


@pytest.mark.parametrize("failure", ["stop path failed", "surviving process", "probe failed"])
def test_cancelled_start_retains_ownership_until_scoped_stop_confirmed(tmp_path, failure):
    spawn_entered = threading.Event()
    release_spawn = threading.Event()

    def popen(*args, **kwargs):
        spawn_entered.set()
        assert release_spawn.wait(1.0)
        return FakeProcess()

    def runner(command, **kwargs):
        if "-Command" in command:
            if failure == "probe failed":
                raise RuntimeError(failure)
            stdout = "[1234]" if failure == "surviving process" else "[]"
        else:
            if failure == "stop path failed":
                raise RuntimeError(failure)
            stdout = ""
        return type("Result", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=runner,
        popen=popen,
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: None,
        poll_interval=0,
        platform_windows=True,
    )
    first = manager.start("start")
    assert spawn_entered.wait(1.0)
    assert manager.start("stop") == first
    release_spawn.set()
    manager._executor.submit(lambda: None).result(timeout=1)

    job = manager.job(first)
    assert job["stage"] == "failed"
    assert job["error"]
    assert manager.current_job()["jobId"] == first
    assert (tmp_path / "gpu.lock").read_text() == first
    with pytest.raises(LifecycleError, match="active"):
        manager.start("start")

    failure = None  # A later scoped Stop can now confirm the process set is empty.
    assert manager.start("stop") == first
    _wait_for_idle(manager)
    assert manager.job(first)["stage"] == "stopped"
    assert manager.job(first)["error"] is None
    assert not (tmp_path / "gpu.lock").exists()
    manager.close()


def test_cancelled_restart_retains_active_job_when_initial_stop_fails(tmp_path):
    executor = ManualExecutor()

    def runner(*_args, **_kwargs):
        manager.start("stop")
        raise RuntimeError("initial stop failed")

    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1", stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log", lock_path=tmp_path / "gpu.lock",
        executor=executor, platform_windows=True, runner=runner,
    )
    job_id = manager.start("restart")
    executor.run_next()
    assert manager.job(job_id)["stage"] == "failed"
    assert manager.current_job()["jobId"] == job_id
    with pytest.raises(LifecycleError, match="active"):
        manager.start("start")


def test_stop_during_readiness_cannot_become_serving(tmp_path):
    readiness_entered = threading.Event()
    release_readiness = threading.Event()
    stop_calls = []

    def readiness():
        readiness_entered.set()
        assert release_readiness.wait(1.0)
        return {"state": "serving"}

    def runner(*args, **kwargs):
        if "-File" in args[0]:
            stop_calls.append((args, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": "[]", "stderr": ""})()

    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=runner,
        popen=lambda *args, **kwargs: FakeProcess(),
        readiness=readiness,
        sleep=lambda _: None,
        poll_interval=0,
        platform_windows=True,
    )
    first = manager.start("start")
    assert readiness_entered.wait(1.0)
    assert manager.start("stop") == first
    release_readiness.set()
    _wait_for_idle(manager)
    assert manager.job(first)["stage"] == "stopped"
    assert manager.job(first)["error"] is None
    assert len(stop_calls) == 1


def test_stop_racing_with_serving_commit_runs_a_real_stop(tmp_path):
    stop_calls = []

    def runner(*args, **kwargs):
        if "-File" in args[0]:
            stop_calls.append((args, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": "[]", "stderr": ""})()

    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=runner,
        popen=lambda *args, **kwargs: FakeProcess(),
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: None,
        poll_interval=0,
        platform_windows=True,
    )
    original_finish_serving = manager._finish_serving

    def cancel_before_commit(job_id):
        assert manager.start("stop") == job_id
        return original_finish_serving(job_id)

    manager._finish_serving = cancel_before_commit
    first = manager.start("start")
    _wait_for_idle(manager)

    assert manager.job(first)["stage"] == "stopped"
    assert len(stop_calls) == 1


def test_accepted_linux_launch_snapshot_is_not_reread(tmp_path):
    executor = ManualExecutor()
    accepted = []
    started = []

    def build_launch(settings, *, base_env):
        accepted.append(dict(settings))
        return FakeLaunch(
            argv=["python", "-m", "freetoken.cli", "serve", "--model", settings["ModelPath"]],
            env=dict(base_env),
        )

    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        popen=lambda argv, **kwargs: started.append((argv, kwargs)) or FakeProcess(),
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: None,
        poll_interval=0,
        platform_windows=False,
        executor=executor,
        launch_builder=build_launch,
    )
    settings = {"ModelPath": "old-model"}
    first = manager.start("start", settings=settings, force=True)
    settings["ModelPath"] = "new-model"
    with pytest.raises(LifecycleError, match="active"):
        manager.start("start")
    assert accepted == [{"ModelPath": "old-model"}]
    executor.run_next()
    assert manager.job(first)["stage"] == "serving"
    assert started[0][0][-1] == "old-model"


def test_launch_preparation_does_not_hold_lifecycle_lock(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    entered, release = threading.Event(), threading.Event()
    executor = ManualExecutor()

    def build(settings, **_kwargs):
        entered.set()
        assert release.wait(2)
        return FakeLaunch(["python"], {})

    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1", stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log", lock_path=tmp_path / "gpu.lock",
        executor=executor, platform_windows=False, launch_builder=build,
        popen=lambda *_args, **_kw: pytest.fail("cancelled preparation must not spawn"),
    )
    with ThreadPoolExecutor(max_workers=2) as callers:
        accepted = callers.submit(manager.start, settings={"ModelPath": "accepted"})
        try:
            assert entered.wait(1)
            current = callers.submit(manager.current_job).result(timeout=0.5)
            assert current is not None
            assert callers.submit(manager.start, "stop").result(timeout=0.5) == current["jobId"]
            with pytest.raises(LifecycleError, match="active"):
                manager.start("restart")
        finally:
            release.set()
        job_id = accepted.result(timeout=1)
    executor.run_next()
    assert manager.job(job_id)["stage"] == "stopped"


def test_accepted_windows_snapshot_uses_staged_boot_script(tmp_path):
    boot = tmp_path / "boot.ps1"
    original = (
        "$launcher = Join-Path $PSScriptRoot 'launcher.ps1'\n"
        "& $launcher `\n"
        "    -ModelPath 'old-model' `\n"
        "    -Port 2020\n"
    )
    boot.write_text(original, encoding="utf-8")
    executor = ManualExecutor()
    started = []
    staged_contents = []

    def popen(command, **kwargs):
        staged_path = Path(command[-1])
        staged_contents.append((staged_path, staged_path.read_text(encoding="utf-8")))
        started.append((command, kwargs))
        return FakeProcess()

    manager = ProcessManager(
        boot_file=boot,
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        popen=popen,
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: None,
        poll_interval=0,
        platform_windows=True,
        executor=executor,
    )

    job_id = manager.start("start", settings={"ModelPath": "new-model", "Port": 2021})
    other = tmp_path / "other-profile" / "boot.ps1"
    other.parent.mkdir()
    other.write_text(original.replace("launcher.ps1", "unchecked.ps1"), encoding="utf-8")
    manager.boot_file = other
    executor.run_next()

    staged, staged_text = staged_contents[0]
    assert staged.parent == boot.parent
    assert "'launcher.ps1'" in staged_text and "unchecked.ps1" not in staged_text
    assert staged != boot
    assert "-ModelPath new-model" in staged_text
    assert "-Port 2021" in staged_text
    assert boot.read_text(encoding="utf-8") == original
    assert manager.job(job_id)["stage"] == "serving"
    assert not staged.exists()


@pytest.mark.parametrize("force", [False, True])
def test_job_record_exposes_override_without_settings(tmp_path, force):
    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1", stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log", lock_path=tmp_path / "gpu.lock",
        executor=ManualExecutor(), platform_windows=False,
        launch_builder=lambda settings, **_: FakeLaunch(["python"], {}),
    )
    job_id = manager.start(settings={"ModelPath": "not-for-job-record"}, force=force)
    record = manager.job(job_id)
    assert record["force"] is force
    assert "settings" not in record and "settings_snapshot" not in record
    assert "not-for-job-record" not in str(record)


def test_failed_windows_start_cleans_staged_boot_script(tmp_path):
    boot = tmp_path / "boot.ps1"
    boot.write_text(
        "$launcher = Join-Path $PSScriptRoot 'launcher.ps1'\n"
        "& $launcher `\n"
        "    -ModelPath 'old-model' `\n"
        "    -Port 2020\n",
        encoding="utf-8",
    )
    executor = ManualExecutor()
    manager = ProcessManager(
        boot_file=boot,
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        popen=lambda *args, **kwargs: FakeProcess(),
        readiness=lambda: {"state": "loading"},
        sleep=lambda _: None,
        poll_interval=0,
        readiness_timeout=0,
        platform_windows=True,
        executor=executor,
    )

    job_id = manager.start("start", settings={"ModelPath": "new-model", "Port": 2021})
    executor.run_next()

    assert manager.job(job_id)["stage"] == "failed"
    assert not list(tmp_path.glob(".boot.launch-*.ps1"))
    assert not manager._temporary_boot_files


def test_internal_start_typeerror_is_not_retried(tmp_path):
    calls = []
    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        popen=lambda *args, **kwargs: FakeProcess(),
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: None,
        poll_interval=0,
        platform_windows=True,
    )

    def broken_start(*, launch=None, settings=None):
        calls.append((launch, settings))
        raise TypeError("internal launch planning failure")

    manager.run_start = broken_start
    job_id = manager.start("start")
    _wait_for_idle(manager)

    assert len(calls) == 1
    assert manager.job(job_id)["stage"] == "failed"
    assert "internal launch planning failure" in manager.job(job_id)["error"]


def test_failed_readiness_does_not_stop_the_spawned_process(tmp_path):
    stop_calls = []

    def runner(*args, **kwargs):
        if "-File" in args[0]:
            stop_calls.append((args, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": "[]", "stderr": ""})()

    manager = ProcessManager(
        boot_file=tmp_path / "boot.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=runner,
        popen=lambda *args, **kwargs: FakeProcess(),
        readiness=lambda: {"state": "loading"},
        sleep=lambda _: None,
        poll_interval=0,
        readiness_timeout=0,
        platform_windows=True,
    )

    job_id = manager.start("start")
    _wait_for_idle(manager)

    assert manager.job(job_id)["stage"] == "failed"
    assert len(stop_calls) == 0


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
        platform_windows=True,
    )
    manager.run_stop()
    command = commands[0][0]
    assert "-TimeoutSeconds" in command
    assert "-Port" in command


def test_linux_branch_stops_through_proc_and_starts_the_mapped_command(tmp_path, monkeypatch):
    """On Linux the helper never calls PowerShell: stop walks /proc, start execs ft serve."""
    from freetoken.daemon.settings import linux_launch

    boot = tmp_path / "boot.ps1"
    boot.write_text(
        "$env:FREETOKEN_MTP_SPECULATE = '0'\n$launcher = Join-Path $PSScriptRoot 'x.ps1'\n& $launcher `\n"
        "    -ModelPath '/models/demo' `\n    -Port 2020 `\n    -ContextTokens 4096 `\n    -MoECacheSize 100\n",
        encoding="utf-8",
    )
    facts = linux_launch.ModelFacts(is_moe=True, expert_count=1000, max_context=8192, has_ple=False)
    monkeypatch.setattr(linux_launch, "read_model_facts", lambda path: facts)
    started = []
    stops = []
    manager = ProcessManager(
        boot_file=boot,
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("PowerShell must not run on Linux")),
        popen=lambda argv, **kwargs: started.append((argv, kwargs)) or FakeProcess(),
        linux_stop=lambda port, timeout: stops.append(port) or {"ok": True, "killed": [], "port": port},
        sleep=lambda _: None,
        platform_windows=False,
    )
    manager.run_stop()
    assert stops == [2020]
    manager.run_start()
    argv, kwargs = started[0]
    assert argv[1:5] == ["-m", "freetoken.cli", "serve", "--model"]
    assert "--moe-cache-size" in argv and argv[argv.index("--moe-cache-size") + 1] == "100"
    assert kwargs["env"]["FREETOKEN_MTP_SPECULATE"] == "0"
    assert kwargs.get("start_new_session") is True
    log = (tmp_path / "server.log").read_text(encoding="utf-8")
    assert "linux stop" in log and "freetoken.cli serve" in log


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

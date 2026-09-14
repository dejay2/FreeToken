"""Torch-free Windows lifecycle control for the settings helper."""

from __future__ import annotations

import contextlib
import copy
import datetime as _datetime
import inspect
import json
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


logger = logging.getLogger("freetoken.daemon.settings.process_manager")


FAILURE_SIGNATURES = (
    "Traceback (most recent call last):",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "cudaHostRegister failed",
    "ZMQError: Address in use",
    "RuntimeError",
)


def _supported_kwargs(function: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop optional test-seam kwargs without retrying a call after an internal TypeError.

    The production subprocess callables accept the complete keyword set. Tiny injected fakes in
    settings tests often expose only ``argv``/``stdout``; inspect their signature before invoking
    them so a TypeError raised by the callable itself can never cause a duplicate spawn/stop.
    """
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {name: value for name, value in kwargs.items() if name in parameters}


# The helper boots servers through PowerShell on Windows and through linux_launch elsewhere
# (WSL included). Tests pin the branch with ProcessManager(platform_windows=...).
IS_WINDOWS = os.name == "nt"


class LifecycleError(RuntimeError):
    """A lifecycle operation could not be completed."""


class _StartCancelled(LifecycleError):
    """The page requested Stop while a start/restart job was in flight."""


@dataclass
class LifecycleJob:
    job_id: str
    action: str
    stage: str
    progress: str
    started_at: str
    completed_at: str | None = None
    error: str | None = None
    settings_snapshot: dict[str, Any] | None = field(default=None, repr=False)
    launch_snapshot: Any = field(default=None, repr=False)
    force: bool = field(default=False, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    cleanup_pending: bool = field(default=False, repr=False)
    _started_monotonic: float = field(default=0.0, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "action": self.action,
            "stage": self.stage,
            "progress": self.progress,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "error": self.error,
            "force": self.force,
        }


class ProcessManager:
    """Run one start/stop/restart job at a time and keep the helper independent of the server."""

    def __init__(
        self,
        *,
        boot_file: str | os.PathLike[str],
        stop_script: str | os.PathLike[str],
        log_path: str | os.PathLike[str],
        lock_path: str | os.PathLike[str],
        port: int = 2020,
        runner: Callable[..., Any] | None = None,
        popen: Callable[..., Any] | None = None,
        readiness: Callable[[], dict[str, Any]] | None = None,
        stats: Callable[[], dict[str, Any]] | None = None,
        gpu_probe: Callable[[], dict[str, Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_now: Callable[[], float] = time.time,
        poll_interval: float = 1.5,
        readiness_timeout: float = 600.0,
        platform_windows: bool | None = None,
        linux_stop: Callable[..., Any] | None = None,
        lock_poll_interval: float = 60.0,
        lock_timeout: float = 900.0,
        failure_scan_interval: float = 2.0,
        owner_id: str | None = None,
        executor=None,
        launch_builder: Callable[..., Any] | None = None,
    ) -> None:
        self.boot_file = Path(boot_file)
        self.stop_script = Path(stop_script)
        self.log_path = Path(log_path)
        self.lock_path = Path(lock_path)
        self.port = int(port)
        self._runner = runner or subprocess.run
        self._popen = popen or subprocess.Popen
        self.platform_windows = IS_WINDOWS if platform_windows is None else bool(platform_windows)
        self._linux_stop = linux_stop
        self._readiness = readiness or self._default_readiness
        self._stats = stats or self._default_stats
        self._gpu_probe = gpu_probe or self._default_gpu_probe
        self._sleep = sleep
        self._monotonic = monotonic
        self._wall_now = wall_now
        self.poll_interval = float(poll_interval)
        self.readiness_timeout = float(readiness_timeout)
        self.lock_poll_interval = float(lock_poll_interval)
        self.lock_timeout = float(lock_timeout)
        self.failure_scan_interval = float(failure_scan_interval)
        self.owner_id = owner_id or os.environ.get("FREETOKEN_SETTINGS_JOB_ID")
        self._executor = executor
        self._launch_builder = launch_builder
        self._owned_executor = executor is None
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="settings-lifecycle")
        self._jobs: dict[str, LifecycleJob] = {}
        self._active_id: str | None = None
        self._process: Any = None
        self._start_log_offset = 0
        self._temporary_boot_files: set[Path] = set()
        self._governor: Any = None
        self._watchdog: Any = None
        self._lock = threading.RLock()

    # ---- job API ---------------------------------------------------------

    def start(
        self,
        action: str = "start",
        *,
        settings: dict[str, Any] | None = None,
        force: bool = False,
    ) -> str:
        """Queue a lifecycle action and return its job id immediately.

        Reserve the job atomically, then prepare its immutable launch outside the lifecycle lock.
        Stop/status remain responsive during metadata/probe I/O. Stop during preparation or an
        active start/restart cancels the same job rather than queueing a second job.
        """
        if action not in {"start", "stop", "restart"}:
            raise ValueError("action must be start, stop, or restart")
        with self._lock:
            if self._active_id is not None:
                active = self._jobs[self._active_id]
                if action == "stop" and active.action in {"start", "restart"}:
                    if self._watchdog is not None:
                        # An accepted Stop against a boot in flight, the watchdog's own included:
                        # the page's wish wins over the reboot.
                        self._watchdog.disarm("stopped by the page")
                    active.cancel_event.set()
                    if active.cleanup_pending and active.stage == "failed":
                        active.stage = "stopping"
                        active.completed_at = None
                        active.progress = "Retrying scoped Stop; GPU ownership is retained."
                        self._executor.submit(self._retry_cancelled_stop, active.job_id)
                    elif not active.cleanup_pending:
                        active.progress = "Stopping the in-progress start..."
                    return active.job_id
                if action == "stop" and active.action == "stop":
                    return active.job_id
                raise LifecycleError(
                    f"Job {active.job_id} is currently active (stage: {active.stage}). "
                    "Cannot start another action."
                )

            if action in {"stop", "restart"} and self._watchdog is not None:
                # Only an accepted request disarms: the page asked for the server to go down,
                # and a restart re-arms when serving returns. (The watchdog's own jobs keep it
                # armed; see CrashWatchdog.disarm.)
                self._watchdog.disarm("stopped by the page" if action == "stop" else None)
            settings_snapshot = copy.deepcopy(settings) if settings is not None else None
            source_boot = self.boot_file
            initial = "stopping" if action in {"stop", "restart"} else "booting"
            now = self._monotonic()
            job = LifecycleJob(
                job_id=f"job-{uuid.uuid4().hex[:12]}",
                action=action,
                stage=initial,
                progress=self._initial_progress(action),
                started_at=self._iso_now(),
                settings_snapshot=settings_snapshot,
                force=bool(force),
                _started_monotonic=now,
            )
            self._jobs[job.job_id] = job
            self._active_id = job.job_id
        try:
            if settings_snapshot is not None:
                job.launch_snapshot = (
                    self._snapshot_windows_boot(settings_snapshot, source_boot)
                    if self.platform_windows else self._build_launch_snapshot(settings_snapshot)
                )
            self._executor.submit(self._run_job, job.job_id)
        except Exception as exc:
            with self._lock:
                self._finish(job.job_id, "failed", "Launch preparation failed.", error=str(exc))
                if self._active_id == job.job_id:
                    self._active_id = None
            raise
        return job.job_id

    submit = start
    start_job = start

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.as_dict() if job is not None else None

    get_job = job

    def current_job(self) -> dict[str, Any] | None:
        with self._lock:
            if self._active_id is None:
                return None
            job = self._jobs.get(self._active_id)
            return job.as_dict() if job is not None else None

    def close(self) -> None:
        self._cleanup_temporary_boot_files()
        if self._owned_executor:
            self._executor.shutdown(wait=False, cancel_futures=False)
            self._owned_executor = False

    def _build_launch_snapshot(self, settings: dict[str, Any]) -> Any:
        builder = self._launch_builder
        if builder is None:
            from .linux_launch import build_launch

            builder = build_launch
        try:
            return builder(settings, base_env=dict(os.environ))
        except Exception as exc:  # noqa: BLE001 - surface launch normalization as lifecycle input error
            raise LifecycleError(f"could not normalize accepted launch: {exc}") from exc

    def _snapshot_windows_boot(self, settings: dict[str, Any], source_boot: Path) -> tuple[Path, str]:
        from .boot_parser import BootFile

        try:
            source = BootFile(source_boot)
            return source.path, source._serialize(source.read_document(), dict(settings))
        except Exception as exc:
            raise LifecycleError(f"could not snapshot accepted Windows launch: {exc}") from exc

    def _stage_windows_boot(self, settings: dict[str, Any], *, launch: Any = None) -> Path:
        """Materialize an accepted settings snapshot without rewriting the active boot file.

        The Windows launcher accepts settings only through its PowerShell boot document. A unique
        sibling script keeps an accepted start/restart immutable even if the active profile changes
        before the worker reaches PowerShell. It is removed after the worker finishes (or by the
        scoped stop path), while the original file remains byte-for-byte untouched.
        """
        try:
            source_path, staged = launch if launch is not None else self._snapshot_windows_boot(settings, self.boot_file)
            descriptor, path = tempfile.mkstemp(
                prefix=f".{source_path.stem}.launch-",
                suffix=source_path.suffix or ".ps1",
                dir=source_path.parent,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as fh:
                    fh.write(staged)
                    fh.flush()
                    os.fsync(fh.fileno())
            except Exception:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                with contextlib.suppress(OSError):
                    os.unlink(path)
                raise
            staged_path = Path(path)
            with self._lock:
                self._temporary_boot_files.add(staged_path)
            return staged_path
        except Exception as exc:  # noqa: BLE001 - surface accepted launch materialization
            if isinstance(exc, LifecycleError):
                raise
            raise LifecycleError(f"could not materialize accepted Windows launch: {exc}") from exc

    def _discard_temporary_boot(self, path: Path) -> None:
        with self._lock:
            self._temporary_boot_files.discard(path)
        with contextlib.suppress(FileNotFoundError, OSError):
            path.unlink()

    def _cleanup_temporary_boot_files(self) -> None:
        with self._lock:
            paths = tuple(self._temporary_boot_files)
            self._temporary_boot_files.clear()
        for path in paths:
            with contextlib.suppress(FileNotFoundError, OSError):
                path.unlink()

    # ---- direct process operations (also useful to tests) ----------------

    def stop(self) -> None:
        self.run_stop()

    def run_stop(self) -> Any:
        self.stop_governor()
        if not self.platform_windows:
            return self._run_stop_linux()
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.stop_script),
            "-Port",
            str(self.port),
            "-TimeoutSeconds",
            "120",
        ]
        try:
            result = self._runner(
                command,
                **_supported_kwargs(
                    self._runner,
                    {
                        "capture_output": True,
                        "text": True,
                        "timeout": 130.0,
                        "check": False,
                    },
                ),
            )
            self._append_process_output(result)
            if getattr(result, "returncode", 0) not in (0, None):
                raise LifecycleError(f"stop script exited with code {result.returncode}")
            return result
        finally:
            # A staged start script has served its purpose once the scoped stop path returns,
            # including a reported stop failure; never leave accepted snapshots accumulating.
            self._cleanup_temporary_boot_files()

    def _run_stop_linux(self) -> Any:
        """Stop through /proc instead of the PowerShell stop script (WSL and native Linux)."""
        from .linux_launch import stop_servers

        stop = self._linux_stop or stop_servers
        report = stop(self.port, timeout=120.0)
        text = json.dumps(report, sort_keys=True)
        self._append_process_output(type("Result", (), {"stdout": f"linux stop: {text}\n", "stderr": ""})())
        if not report.get("ok", False):
            raise LifecycleError(f"server on port {self.port} did not stop cleanly: {text}")
        return report

    def _confirm_cancelled_stop(self) -> None:
        """Require an empty scoped process set, not just free ports or low VRAM."""
        if not self.platform_windows:
            from .linux_launch import find_server_pids

            remaining = find_server_pids(self.port)
        else:
            # Reuse the stop script's read-only selectors; do not invent a second kill scope.
            script = str(self.stop_script).replace("'", "''")
            command = [
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                f"$ErrorActionPreference = 'Stop'; . '{script}' -DotSourceOnly; "
                f"$ids = @(Select-FreeTokenKillSet -Processes (Get-FreeTokenProcessSnapshot) -Port {self.port}); "
                "ConvertTo-Json -InputObject @($ids) -Compress",
            ]
            result = self._runner(command, **_supported_kwargs(self._runner, {
                "capture_output": True, "text": True, "timeout": 10.0, "check": False,
            }))
            if getattr(result, "returncode", None) != 0:
                raise LifecycleError("could not confirm the stopped process set")
            remaining = json.loads(result.stdout)
            if not isinstance(remaining, list) or any(type(pid) is not int for pid in remaining):
                raise LifecycleError("invalid stopped-process confirmation")
        if remaining:
            raise LifecycleError("scoped server processes remain after Stop")

    def _retry_cancelled_stop(self, job_id: str) -> None:
        job = self._get_job_object(job_id)
        try:
            self.run_stop()
            self._confirm_cancelled_stop()
        except Exception as exc:
            self._finish(job_id, "failed", "Stop could not be confirmed; retry Stop.", error=str(exc))
            return
        with self._lock:
            job.cleanup_pending = False
            self._finish(job_id, "stopped", "Start cancelled; server stopped.")
            self.release_gpu_lock(self.owner_id or job_id)
            if self._active_id == job_id:
                self._active_id = None

    def _run_start_linux(
        self,
        log: Any,
        *,
        launch: Any = None,
        settings: dict[str, Any] | None = None,
    ) -> Any:
        """Map the accepted snapshot onto ``ft serve`` and start it detached."""
        from .boot_parser import BootFile

        if launch is not None:
            plan = launch
        elif settings is not None:
            plan = self._build_launch_snapshot(settings)
        else:
            from .linux_launch import build_launch

            plan = build_launch(BootFile(self.boot_file).load())
        for note in plan.notes:
            log.write(f"  Note: {note}\n".encode())
        log.write(f"  {plan.command_line()}\n".encode())
        kwargs: dict[str, Any] = {
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "env": plan.env,
            "start_new_session": True,  # the helper may restart; the server must outlive it
        }
        return self._popen(plan.argv, **_supported_kwargs(self._popen, kwargs))

    def run_start(
        self,
        *,
        launch: Any = None,
        settings: dict[str, Any] | None = None,
    ) -> Any:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab", buffering=0) as log:
            # The retained log can contain failures from earlier boots. On WSL this made a
            # fresh override fail in the same second as spawn (J5, 2026-09-07). Keep history
            # for the log viewer, but attribute readiness failures only to this start.
            self._start_log_offset = log.tell()
            separator = f"\n===== settings helper start {_datetime.datetime.now().isoformat()} =====\n".encode()
            log.write(separator)
            if not self.platform_windows:
                process = self._run_start_linux(log, launch=launch, settings=settings)
                with self._lock:
                    self._process = process
                return process
            temporary_boot = None
            boot_path = self.boot_file
            if launch is not None or settings is not None:
                temporary_boot = self._stage_windows_boot(settings or {}, launch=launch)
                boot_path = temporary_boot
            command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(boot_path),
            ]
            kwargs = {
                "stdout": log,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
            }
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if creationflags:
                kwargs["creationflags"] = creationflags
            try:
                process = self._popen(command, **_supported_kwargs(self._popen, kwargs))
            except Exception:
                if temporary_boot is not None:
                    self._discard_temporary_boot(temporary_boot)
                raise
        with self._lock:
            self._process = process
        return process

    def wait_until_serving(
        self,
        timeout: float | None = None,
        *,
        job_id: str | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        timeout = self.readiness_timeout if timeout is None else float(timeout)
        started = self._monotonic()
        deadline = started + timeout
        last_scan = -float("inf")
        while True:
            if cancel is not None and cancel():
                raise _StartCancelled("start cancelled by Stop")
            try:
                document = self._readiness()
            except Exception as exc:  # noqa: BLE001 — an unreachable server is still booting
                document = {"state": "unreachable", "error": str(exc)}
            if isinstance(document, dict) and document.get("state") == "serving":
                if cancel is not None and cancel():
                    raise _StartCancelled("start cancelled by Stop")
                return document

            if job_id is not None:
                elapsed = int(max(0.0, self._monotonic() - started))
                self._set_progress(
                    job_id,
                    "booting",
                    f"Waiting for server to report serving state on port {self.port} "
                    f"(elapsed {elapsed}s)...",
                )

            process = self._process
            if process is not None:
                returncode = process.poll()
                if returncode is not None and returncode != 0:
                    raise LifecycleError(f"server process exited with code {returncode}")
            now = self._monotonic()
            if now - last_scan >= self.failure_scan_interval:
                last_scan = now
                error = self.failure_from_log(self._read_log_tail())
                if error:
                    raise LifecycleError(error)
            if now >= deadline:
                raise LifecycleError(f"server did not report serving within {int(timeout)}s")
            remaining = max(0.0, deadline - now)
            self._sleep(min(self.poll_interval, remaining))

    def acquire_gpu_lock(
        self,
        owner_id: str,
        timeout: float | None = None,
        *,
        cancel: Callable[[], bool] | None = None,
    ) -> bool:
        """Acquire our lock, or wait on a different owner's lock without deleting it."""
        timeout = self.lock_timeout if timeout is None else float(timeout)
        deadline = self._monotonic() + timeout
        owner_id = str(owner_id)
        while True:
            if cancel is not None and cancel():
                return False
            existing = self._read_lock()
            if existing and existing != owner_id:
                if self._monotonic() >= deadline:
                    return False
                self._sleep(min(self.lock_poll_interval, max(0.0, deadline - self._monotonic())))
                continue
            if existing == owner_id:
                return True
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.lock_path.open("x", encoding="utf-8") as fh:
                    fh.write(owner_id)
                    fh.flush()
                    os.fsync(fh.fileno())
                return True
            except FileExistsError:
                # Another process won the create race; read it on the next loop.
                if self._monotonic() >= deadline:
                    return False

    @contextlib.contextmanager
    def gpu_lock(self, owner_id: str) -> Iterator[None]:
        if not self.acquire_gpu_lock(owner_id):
            raise LifecycleError("timed out waiting for the GPU lock")
        try:
            yield
        finally:
            self.release_gpu_lock(owner_id)

    def release_gpu_lock(self, owner_id: str) -> None:
        if self._read_lock() != str(owner_id):
            return
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # A foreign writer may have replaced it; never remove that owner's lock.
            pass

    @classmethod
    def failure_from_log(cls, text: str) -> str | None:
        for signature in FAILURE_SIGNATURES:
            index = text.find(signature)
            if index >= 0:
                start = text.rfind("\n", 0, index) + 1
                end = text.find("\n", index)
                if end < 0:
                    end = len(text)
                snippet = text[start:end].strip()
                return snippet or signature
        return None

    # ---- status and log helpers -----------------------------------------

    def server_status(self) -> dict[str, Any]:
        try:
            cache = self._readiness()
        except Exception as exc:  # noqa: BLE001
            cache = {"state": "unreachable", "error": str(exc)}
        try:
            stats = self._stats()
        except Exception:
            stats = {}
        result = self._merge_server_documents(cache, stats)
        if not result["vramUsedMb"] or not result["vramTotalMb"]:
            try:
                result.update(self._gpu_probe())
            except Exception:
                pass
        return result

    def tail_log(self, limit: int = 100) -> list[str]:
        limit = max(1, min(int(limit), 2000))
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except FileNotFoundError:
            lines = []
        except OSError:
            lines = []
        return lines[-limit:]

    def log_total_lines(self) -> int:
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as fh:
                return sum(1 for _ in fh)
        except (FileNotFoundError, OSError):
            return 0

    # ---- job worker ------------------------------------------------------

    def _run_job(self, job_id: str) -> None:
        lock_owner = self.owner_id or job_id
        acquired = False
        spawned = False
        stop_attempted = False
        try:
            job = self._get_job_object(job_id)
            cancelled = job.cancel_event.is_set
            if job.action in {"stop", "restart"}:
                self._set_progress(job_id, "stopping", "Running the server stop path for port 2020...")
                stop_attempted = True
                self.run_stop()
            if job.action == "stop":
                self._finish(job_id, "stopped", "Server stopped.")
                return
            if cancelled():
                raise _StartCancelled("start cancelled by Stop")
            self._set_progress(job_id, "waiting_for_gpu_lock", "Waiting for the GPU lock before starting port 2020...")
            acquired = self.acquire_gpu_lock(lock_owner, cancel=cancelled)
            if not acquired:
                if cancelled():
                    raise _StartCancelled("start cancelled while waiting for the GPU lock")
                raise LifecycleError("timed out waiting for the GPU lock")
            if cancelled():
                raise _StartCancelled("start cancelled before spawn")
            self._set_progress(job_id, "booting", "Starting the accepted launch and waiting for serving state...")
            self.run_start(launch=job.launch_snapshot, settings=job.settings_snapshot)
            spawned = True
            if cancelled():
                raise _StartCancelled("start cancelled after spawn")
            self.wait_until_serving(job_id=job_id, cancel=cancelled)
            if cancelled():
                raise _StartCancelled("start cancelled as serving state arrived")
            self._finish_serving(job_id)
        except _StartCancelled as exc:
            # Stop the same accepted job. This is deliberately inside the worker so a late spawn
            # cannot race a second queued stop and readiness can never overwrite ``stopped``.
            if spawned or stop_attempted:
                try:
                    if spawned:
                        self.run_stop()
                    self._confirm_cancelled_stop()
                except Exception as stop_exc:  # noqa: BLE001 - retain ownership until confirmed cleanup
                    with self._lock:
                        job.cleanup_pending = True
                        self._finish(job_id, "failed", "Stop could not be confirmed; retry Stop.",
                                     error=f"{exc}; stop path: {stop_exc}")
                    return
            self._finish(job_id, "stopped", "Start cancelled; server stopped.")
        except Exception as exc:  # noqa: BLE001 — a failed job must not kill the helper
            error = str(exc)
            self._cleanup_temporary_boot_files()
            with self._lock:
                if stop_attempted and job.cancel_event.is_set():
                    job.cleanup_pending = True
                progress = "Stop could not be confirmed; retry Stop." if job.cleanup_pending else "Lifecycle action failed."
                self._finish(job_id, "failed", progress, error=error)
        finally:
            # A staged Windows launch is needed only until the accepted worker has crossed its
            # spawn/readiness boundary. Never retain profile snapshots after this job terminates.
            self._cleanup_temporary_boot_files()
            with self._lock:
                if not job.cleanup_pending:
                    if acquired:
                        self.release_gpu_lock(lock_owner)
                    if self._active_id == job_id:
                        self._active_id = None

    # ---- internal helpers ------------------------------------------------

    def _get_job_object(self, job_id: str) -> LifecycleJob:
        with self._lock:
            return self._jobs[job_id]

    def _set_progress(self, job_id: str, stage: str, progress: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.stage = stage
                job.progress = progress

    def _finish_serving(self, job_id: str) -> None:
        """Commit serving atomically with the last cancellation check.

        Stop can arrive after readiness returns but before the worker publishes its terminal
        state. Keep the check and the terminal transition under the same lock; if the transition
        wins first, Stop queues a normal stop job instead of losing the request in the worker's
        finalizer.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.cancel_event.is_set():
                raise _StartCancelled("start cancelled before serving state was committed")
            if job.stage == "stopped":
                return
            job.stage = "serving"
            job.progress = "Server is serving on port 2020."
            job.error = None
            job.completed_at = self._iso_now()
            if self._active_id == job_id:
                self._active_id = None
        self.start_governor(settings=job.settings_snapshot)
        if self._watchdog is not None:
            # A profile Start carries its own flag; re-read it the way the governor is rebuilt.
            self.apply_watchdog_settings(job.settings_snapshot, quiet=True)
            self._watchdog.arm()

    def _finish(self, job_id: str, stage: str, progress: str, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.stage == "stopped" and stage != "stopped":
                return
            job.stage = stage
            job.progress = progress
            job.error = error
            job.completed_at = self._iso_now()

    def _initial_progress(self, action: str) -> str:
        if action == "stop":
            return "Stopping the server on port 2020..."
        if action == "restart":
            return "Stopping the server on port 2020..."
        return "Starting the server on port 2020..."

    def _read_lock(self) -> str:
        try:
            return self.lock_path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return ""

    def _read_log_tail(self, max_bytes: int = 65536) -> str:
        try:
            with self.log_path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                start = self._start_log_offset if size >= self._start_log_offset else 0
                fh.seek(max(start, size - max_bytes), os.SEEK_SET)
                return fh.read().decode("utf-8", "replace")
        except (FileNotFoundError, OSError):
            return ""

    def _append_process_output(self, result: Any) -> None:
        output = getattr(result, "stdout", None) or getattr(result, "stderr", None)
        if not output:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(str(output))
            if not str(output).endswith("\n"):
                fh.write("\n")

    @staticmethod
    def _default_gpu_probe() -> dict[str, Any]:
        """Read the card without importing torch; an absent driver simply reports zeroes."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return {"vramUsedMb": 0, "vramTotalMb": 0, "gpuUtilPercent": 0}
        if result.returncode != 0:
            return {"vramUsedMb": 0, "vramTotalMb": 0, "gpuUtilPercent": 0}
        rows = []
        for line in result.stdout.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) == 3 and all(item.isdigit() for item in parts):
                rows.append(tuple(int(item) for item in parts))
        if not rows:
            return {"vramUsedMb": 0, "vramTotalMb": 0, "gpuUtilPercent": 0}
        return {
            "vramUsedMb": sum(row[0] for row in rows),
            "vramTotalMb": sum(row[1] for row in rows),
            "gpuUtilPercent": max(row[2] for row in rows),
        }

    def _default_readiness(self) -> dict[str, Any]:
        return self._get_json(f"http://127.0.0.1:{self.port}/v1/cache/status", timeout=3.0)

    def _default_stats(self) -> dict[str, Any]:
        return self._get_json(f"http://127.0.0.1:{self.port}/v1/stats", timeout=3.0)

    @staticmethod
    def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _merge_server_documents(cache: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(cache, dict):
            cache = {}
        if not isinstance(stats, dict):
            stats = {}
        geometry = cache.get("geometry") or {}
        parking = cache.get("parking") or {}
        requests = stats.get("requests") or {}
        vram = stats.get("vram_bytes", stats.get("vramBytes", 0)) or 0
        gpus = stats.get("gpus") or []
        first_gpu = gpus[0] if isinstance(gpus, list) and gpus and isinstance(gpus[0], dict) else {}
        total_vram = stats.get("vram_total_bytes", stats.get("vramTotalBytes", 0)) or first_gpu.get("total_bytes", first_gpu.get("totalBytes", 0)) or 0
        util = stats.get("gpu_util_percent", stats.get("gpuUtilPercent", 0)) or first_gpu.get("utilization_percent", first_gpu.get("utilizationPercent", 0)) or 0
        state = cache.get("state", "unreachable")
        result = {
            "reachable": cache.get("reachable", state not in {"unreachable", None}) is not False,
            "state": state,
            "activeRequests": requests.get("active", requests.get("activeRequests", 0)) or 0,
            "uptimeS": stats.get("uptime_s", stats.get("uptimeS", 0)) or 0,
            "geometry": _camelize(geometry),
            "parking": _camelize(parking),
            # Left snake_case (not camelized like geometry/parking): the settings page renders
            # these keys verbatim, matching /v1/cache/status's own field names (Task 9).
            "kv_dynamic": cache.get("kv_dynamic"),
            "vramUsedMb": int(vram / (1024 * 1024)) if isinstance(vram, (int, float)) else 0,
            "vramTotalMb": int(total_vram / (1024 * 1024)) if isinstance(total_vram, (int, float)) else 0,
            "gpuUtilPercent": int(util) if isinstance(util, (int, float)) else 0,
        }
        return result

    def _governor_policy(self, settings: dict[str, Any]) -> tuple[Any, bool]:
        """Build the helper policy and enabled flag without importing torch or the engine."""
        from .governor import GIB, GovernorPolicy

        enabled_val = settings.get("MemoryGovernor", True)
        enabled = (
            str(enabled_val).strip().lower() in {"1", "true", "yes", "on"}
            if not isinstance(enabled_val, bool)
            else enabled_val
        )

        def _number(name: str, default: float) -> float:
            value = settings.get(name)
            return float(default if value is None else value)

        vram_gb = _number("GovernorVRAMFreeGB", 1.5)
        ram_gb = _number("GovernorRAMFreeGB", 4.0)
        margin_gb = _number("GovernorUpMarginGB", 0.5)
        policy = GovernorPolicy(
            vram_cushion=int(round(vram_gb * GIB)),
            ram_cushion=int(round(ram_gb * GIB)),
            margin=int(round(margin_gb * GIB)),
            step_interval=_number("GovernorStepIntervalS", 5.0),
            up_hold=_number("GovernorUpHoldS", 60.0),
            max_hold=_number("GovernorMaxHoldS", 600.0),
            post_up_grace=_number("GovernorPostUpGraceS", 10.0),
            ram_rungs_before_up=int(round(_number("GovernorRAMRungsBeforeUp", 2))),
            vram_rungs_before_up=int(round(_number("GovernorVRAMRungsBeforeUp", 1))),
        )
        return policy, enabled

    def start_governor(self, settings: dict[str, Any] | None = None) -> None:
        try:
            from .governor import GovernorLoop
        except ImportError:
            return
        if settings is None:
            try:
                from .boot_parser import BootFile

                settings = BootFile(self.boot_file).load()
            except Exception:
                settings = {}
        policy, enabled = self._governor_policy(settings)
        self.stop_governor()
        self._governor = GovernorLoop(self, policy, http_port=self.port)
        self._governor.enabled = enabled
        self._governor.start()

    def apply_governor_settings(self, settings: dict[str, Any]) -> None:
        """Swap policy values in the running watcher without restarting the model server."""
        from .governor import GIB

        policy, enabled = self._governor_policy(settings)
        with self._lock:
            loop = self._governor
            if loop is not None:
                loop.policy = policy
                loop.enabled = enabled
        logger.info(
            "governor settings applied: enabled=%s vram_cushion=%.2fGiB ram_cushion=%.2fGiB "
            "margin=%.2fGiB interval=%.1fs up_hold=%.1fs max_hold=%.1fs grace=%.1fs "
            "ram_rungs=%d vram_rungs=%d",
            enabled,
            policy.vram_cushion / GIB,
            policy.ram_cushion / GIB,
            policy.margin / GIB,
            policy.step_interval,
            policy.up_hold,
            policy.max_hold,
            policy.post_up_grace,
            policy.ram_rungs_before_up,
            policy.vram_rungs_before_up,
        )

    def stop_governor(self) -> None:
        if self._governor is not None:
            try:
                self._governor.stop()
            except Exception:
                pass
            self._governor = None

    def governor_status(self) -> dict[str, Any]:
        if self._governor is not None:
            return self._governor.status()
        try:
            from .boot_parser import BootFile

            settings = BootFile(self.boot_file).load()
        except Exception:
            settings = {}
        enabled_val = settings.get("MemoryGovernor", True)
        enabled = str(enabled_val).strip().lower() in {"1", "true", "yes", "on"} if not isinstance(enabled_val, bool) else enabled_val
        # No loop, no numbers. Never touch the server port from here: GET /api/status runs
        # on every page poll and must answer within the 0.5 s the memory-fit tests hold it
        # to, and this path bypasses the injected readiness/stats probes (a stray listener
        # on the port, such as an ssh tunnel, made every status call block for a second).
        return {
            "enabled": enabled,
            "last_action": None,
            "layers": {"owned": 0, "pinned": 0, "disk": 0, "parked": 0},
            "free_vram_gb": 0.0,
            "free_ram_gb": 0.0,
        }

    # ---- what a restart frees --------------------------------------------

    def running_server_release(self) -> dict[str, int]:
        """The running server's own RAM and VRAM, as the fit check should add back for a restart.

        RAM is the engine's own account, pinned layers x bytes per layer from /v1/cache/residency,
        not the processes' RSS: on the WSL box the scheduler's pinned banks (63 GB) showed as a
        24 MiB RSS on 2026-09-08 (the dxg-backed pinned pages are not counted there), while an
        earlier boot had reported 71 GB. VRAM is the server's /v1/stats figure. Zero when nothing
        is serving or a figure cannot be read (see MemoryFitService._release_for).
        """
        ram = 0
        vram = 0
        try:
            residency = self._get_json(f"http://127.0.0.1:{self.port}/v1/cache/residency", timeout=3.0)
            if isinstance(residency, dict):
                layers = int(residency.get("pinned", 0) or 0)
                ram = layers * int(residency.get("layer_bytes", 0) or 0)
        except Exception:  # noqa: BLE001 - an unreachable server frees nothing
            ram = 0
        try:
            stats = self._stats()
            value = stats.get("vram_bytes", stats.get("vramBytes", 0)) if isinstance(stats, dict) else 0
            vram = int(value or 0)
        except Exception:  # noqa: BLE001
            vram = 0
        return {"ram_bytes": max(0, ram), "vram_bytes": max(0, vram)}

    # ---- crash watchdog --------------------------------------------------

    @staticmethod
    def _auto_restart_enabled(settings: dict[str, Any]) -> bool:
        # An env toggle: absent from the boot file reads back as its default "1" (on).
        value = settings.get("FREETOKEN_AUTO_RESTART", "1")
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def start_watchdog(self, settings: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Start the crash watchdog with the helper; it adopts a serving server on its first tick."""
        from .watchdog import CrashWatchdog

        if settings is None:
            try:
                from .boot_parser import BootFile

                settings = BootFile(self.boot_file).load()
            except Exception:
                settings = {}
        self.stop_watchdog()
        watchdog = CrashWatchdog(self, **kwargs)
        watchdog.enabled = self._auto_restart_enabled(settings)
        self._watchdog = watchdog
        watchdog.start()

    def apply_watchdog_settings(self, settings: dict[str, Any] | None, *, quiet: bool = False) -> None:
        """A Save or profile change flips the watchdog on or off without touching the server."""
        if settings is None:
            try:
                from .boot_parser import BootFile

                settings = BootFile(self.boot_file).load()
            except Exception:
                settings = {}
        enabled = self._auto_restart_enabled(settings)
        watchdog = self._watchdog
        if watchdog is not None and watchdog.enabled != enabled:
            watchdog.enabled = enabled
        if not quiet:
            logger.info("auto-restart %s", "enabled" if enabled else "disabled")

    def stop_watchdog(self) -> None:
        watchdog = self._watchdog
        if watchdog is not None:
            try:
                watchdog.stop()
                if watchdog.is_alive():
                    watchdog.join(timeout=5.0)  # a tick past its checks must not queue a job as the helper exits
            except Exception:
                pass
            self._watchdog = None

    def watchdog_status(self) -> dict[str, Any]:
        if self._watchdog is not None:
            return self._watchdog.status()
        try:
            from .boot_parser import BootFile

            enabled = self._auto_restart_enabled(BootFile(self.boot_file).load())
        except Exception:
            enabled = True
        return {
            "enabled": enabled,
            "armed": False,
            "misses": 0,
            "restarts_last_hour": 0,
            "last_restart_at": None,
            "last_reason": None,
            "gave_up": False,
        }

    def _iso_now(self) -> str:
        stamp = _datetime.datetime.fromtimestamp(self._wall_now(), tz=_datetime.timezone.utc)
        return stamp.isoformat(timespec="seconds").replace("+00:00", "Z")


# The old name is useful to callers that describe this component as the server controller.
SettingsProcessManager = ProcessManager


def _camelize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key if "_" not in key else _snake_to_camel(key): _camelize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_camelize(item) for item in value]
    return value


def _snake_to_camel(value: str) -> str:
    pieces = value.split("_")
    return pieces[0] + "".join(piece[:1].upper() + piece[1:] for piece in pieces[1:])


__all__ = [
    "FAILURE_SIGNATURES",
    "LifecycleError",
    "LifecycleJob",
    "ProcessManager",
    "SettingsProcessManager",
]

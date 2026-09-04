"""Torch-free Windows lifecycle control for the settings helper."""

from __future__ import annotations

import contextlib
import datetime as _datetime
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


FAILURE_SIGNATURES = (
    "Traceback (most recent call last):",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "cudaHostRegister failed",
    "ZMQError: Address in use",
    "RuntimeError",
)


class LifecycleError(RuntimeError):
    """A lifecycle operation could not be completed."""


@dataclass
class LifecycleJob:
    job_id: str
    action: str
    stage: str
    progress: str
    started_at: str
    completed_at: str | None = None
    error: str | None = None
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
        lock_poll_interval: float = 60.0,
        lock_timeout: float = 900.0,
        failure_scan_interval: float = 2.0,
        owner_id: str | None = None,
        executor=None,
    ) -> None:
        self.boot_file = Path(boot_file)
        self.stop_script = Path(stop_script)
        self.log_path = Path(log_path)
        self.lock_path = Path(lock_path)
        self.port = int(port)
        self._runner = runner or subprocess.run
        self._popen = popen or subprocess.Popen
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
        self._owned_executor = executor is None
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="settings-lifecycle")
        self._jobs: dict[str, LifecycleJob] = {}
        self._active_id: str | None = None
        self._process: Any = None
        self._lock = threading.RLock()

    # ---- job API ---------------------------------------------------------

    def start(self, action: str = "start") -> str:
        """Queue a lifecycle action and return its job id immediately."""
        if action not in {"start", "stop", "restart"}:
            raise ValueError("action must be start, stop, or restart")
        with self._lock:
            if self._active_id is not None:
                active = self._jobs[self._active_id]
                raise LifecycleError(
                    f"Job {active.job_id} is currently active (stage: {active.stage}). "
                    "Cannot start another action."
                )
            initial = "stopping" if action in {"stop", "restart"} else "booting"
            now = self._monotonic()
            job = LifecycleJob(
                job_id=f"job-{uuid.uuid4().hex[:12]}",
                action=action,
                stage=initial,
                progress=self._initial_progress(action),
                started_at=self._iso_now(),
                _started_monotonic=now,
            )
            self._jobs[job.job_id] = job
            self._active_id = job.job_id
        self._executor.submit(self._run_job, job.job_id)
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
        if self._owned_executor:
            self._executor.shutdown(wait=False, cancel_futures=False)
            self._owned_executor = False

    # ---- direct process operations (also useful to tests) ----------------

    def stop(self) -> None:
        self.run_stop()

    def run_stop(self) -> Any:
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
                capture_output=True,
                text=True,
                timeout=130.0,
                check=False,
            )
        except TypeError:
            # Keep the seam usable with tiny test doubles that only accept argv.
            result = self._runner(command)
        self._append_process_output(result)
        if getattr(result, "returncode", 0) not in (0, None):
            raise LifecycleError(f"stop script exited with code {result.returncode}")
        return result

    def run_start(self) -> Any:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab", buffering=0) as log:
            separator = f"\n===== settings helper start {_datetime.datetime.now().isoformat()} =====\n".encode()
            log.write(separator)
            command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.boot_file),
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
                process = self._popen(command, **kwargs)
            except TypeError:
                # Small test doubles and non-Windows launchers may only accept argv/stdout.
                kwargs.pop("creationflags", None)
                process = self._popen(command, **kwargs)
        with self._lock:
            self._process = process
        return process

    def wait_until_serving(
        self, timeout: float | None = None, *, job_id: str | None = None
    ) -> dict[str, Any]:
        timeout = self.readiness_timeout if timeout is None else float(timeout)
        started = self._monotonic()
        deadline = started + timeout
        last_scan = -float("inf")
        while True:
            try:
                document = self._readiness()
            except Exception as exc:  # noqa: BLE001 — an unreachable server is still booting
                document = {"state": "unreachable", "error": str(exc)}
            if isinstance(document, dict) and document.get("state") == "serving":
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

    def acquire_gpu_lock(self, owner_id: str, timeout: float | None = None) -> bool:
        """Acquire our lock, or wait on a different owner's lock without deleting it."""
        timeout = self.lock_timeout if timeout is None else float(timeout)
        deadline = self._monotonic() + timeout
        owner_id = str(owner_id)
        while True:
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
        try:
            job = self._get_job_object(job_id)
            if job.action in {"stop", "restart"}:
                self._set_progress(job_id, "stopping", "Running the Windows stop script for port 2020...")
                self.run_stop()
            if job.action == "stop":
                self._finish(job_id, "stopped", "Server stopped.")
                return
            self._set_progress(job_id, "waiting_for_gpu_lock", "Waiting for the GPU lock before starting port 2020...")
            acquired = self.acquire_gpu_lock(lock_owner)
            if not acquired:
                raise LifecycleError("timed out waiting for the GPU lock")
            self._set_progress(job_id, "booting", "Starting boot-2020.ps1 and waiting for serving state...")
            self.run_start()
            self.wait_until_serving(job_id=job_id)
            self._finish(job_id, "serving", "Server is serving on port 2020.")
        except Exception as exc:  # noqa: BLE001 — a failed job must not kill the helper
            self._finish(job_id, "failed", "Lifecycle action failed.", error=str(exc))
        finally:
            if acquired:
                self.release_gpu_lock(lock_owner)
            with self._lock:
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

    def _finish(self, job_id: str, stage: str, progress: str, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
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
                fh.seek(max(0, size - max_bytes), os.SEEK_SET)
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
            "vramUsedMb": int(vram / (1024 * 1024)) if isinstance(vram, (int, float)) else 0,
            "vramTotalMb": int(total_vram / (1024 * 1024)) if isinstance(total_vram, (int, float)) else 0,
            "gpuUtilPercent": int(util) if isinstance(util, (int, float)) else 0,
        }
        return result

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

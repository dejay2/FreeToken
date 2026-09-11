"""Bounded, private crash evidence collected before the watchdog restarts workers.

No torch import, environment dump, request bodies or debugger locals. Diagnostics are
best effort: a broken driver command must never prevent recovery.
"""

from __future__ import annotations

import datetime as dt
import base64
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def _incident_worker(send, kwargs) -> None:
    try:
        send.send((str(capture_incident(**kwargs)), None))
    except Exception as exc:
        send.send((None, str(exc)))
    finally:
        send.close()


def capture_incident_bounded(*, timeout: float = 10.0, **kwargs) -> Path:
    """Bound the entire capture, including a filesystem that stops answering."""
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    worker = context.Process(target=_incident_worker, args=(send, kwargs), daemon=True)
    try:
        worker.start()
        send.close()
        if not receive.poll(timeout):
            raise TimeoutError("incident capture exceeded its deadline")
        path, error = receive.recv()
        if error is not None:
            raise RuntimeError(error)
        return Path(path)
    finally:
        if worker.pid is not None:
            worker.join(timeout=0.1)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=0.2)
        receive.close()
        send.close()


def command_output(argv: list[str], *, timeout: float = 1.0, max_bytes: int = 65536) -> dict:
    """Capture at most max_bytes, killing only this diagnostic's process group on timeout."""
    result = {"output": "", "truncated": False, "timed_out": False}
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        return {**result, "error": str(exc)}
    data = bytearray()

    def read() -> None:
        try:
            while len(data) <= max_bytes:
                chunk = proc.stdout.read(min(8192, max_bytes + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
        except (OSError, ValueError):
            pass

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    reader.join(timeout=max(0, timeout))
    result["timed_out"] = reader.is_alive()
    result["truncated"] = len(data) > max_bytes
    # A child can close stdout and keep running: its exit is also part of the deadline.
    try:
        proc.wait(timeout=max(0, deadline - time.monotonic()) if not result["truncated"] else 0)
    except subprocess.TimeoutExpired:
        if not result["truncated"]:
            result["timed_out"] = True
    finally:
        if result["timed_out"] or result["truncated"]:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
        reader.join(timeout=0.2)
    # errors='ignore' keeps the encoded result within the byte limit too.
    result.update(output=bytes(data[:max_bytes]).decode("utf-8", errors="ignore"), exitcode=proc.poll())
    if not reader.is_alive():
        proc.stdout.close()
    return result


def _write(path: Path, data: str | bytes) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out:
        out.write(data.encode() if isinstance(data, str) else data)


def capture_incident(
    *, document: dict, reason: str, pids: set[int] | None, log_path: str | Path,
    directory: str | Path | None = None, retain: int = 5,
) -> Path:
    """Keep five small bundles by default, with a shared eight-second command budget."""
    root = Path(directory or os.environ.get("FREETOKEN_INCIDENT_DIR") or Path(log_path).parent / "incidents")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / f"incident-{stamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir(mode=0o700)
    manifest: dict[str, Any] = {"time": stamp, "reason": reason, "document": document, "pids": sorted(pids or ())}
    deadline = time.monotonic() + 8.0

    def capture(name: str, argv: list[str], timeout: float = 1.0) -> None:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            manifest[name] = command_output(argv, timeout=min(timeout, remaining))

    try:
        with Path(log_path).open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 131072))
            _write(path / "server-tail.log", log.read(131072))
    except OSError as exc:
        manifest["log_error"] = str(exc)
    capture("gpu", ["nvidia-smi", "--query-gpu=timestamp,name,driver_version,memory.used,memory.total,utilization.gpu,power.draw", "--format=csv"], 1.5)
    if os.name == "posix":
        capture("memory", ["free", "-b"])
        capture("processes", ["ps", "-L", "-p", ",".join(str(p) for p in sorted(pids or ())) or "0", "-o", "pid,tid,stat,wchan:40,comm"])
        since = (dt.datetime.now().astimezone() - dt.timedelta(minutes=20)).isoformat(timespec="seconds")
        capture("kernel", ["dmesg", "--level=err,warn", "--time-format=iso", "--since", since], 1.0)
        powershell = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
        if powershell.exists():
            script = (
                "$ErrorActionPreference='Stop'; @(Get-WinEvent -FilterHashtable "
                "@{LogName='System';ProviderName='nvlddmkm';StartTime=(Get-Date).AddMinutes(-20)} "
                "-MaxEvents 30 -ErrorAction SilentlyContinue | "
                "Select-Object TimeCreated,Id,Message) | ConvertTo-Json -Compress"
            )
            encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
            capture("windows_gpu_events", [str(powershell), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], 2.0)
        spy = shutil.which("py-spy") or str(Path(sys.executable).parent / "py-spy")
        for pid in sorted(pids or ())[:8]:
            capture(f"stack_{pid}", [spy, "dump", "--pid", str(pid), "--native"], 1.0)
    _write(path / "manifest.json", json.dumps(manifest, indent=2, default=str))
    # Only remove bundles we own; leave other files or symlinks alone.
    bundles = sorted(p for p in root.glob("incident-*") if p.is_dir() and not p.is_symlink())
    for old in bundles[:-max(1, retain)]:
        shutil.rmtree(old)
    return path

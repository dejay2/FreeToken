"""Bounded, torch-free WSL cache reclamation, independent of expert placement.

The actuator is a disposable subprocess: reclaim can block in the kernel. The
governor only polls it and never waits for it before protecting host/GPU memory.
Older kernels must NOT fall back to reclaim that can swap anonymous memory.
"""

from __future__ import annotations

import ctypes
import errno
import json
import logging
import os
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

MIB = 1024**2
GIB = 1024**3
PROBE_TIMEOUT_S = 10
logger = logging.getLogger(__name__)


class ReclaimUnavailable(RuntimeError):
    pass


class ReclaimPolicy:
    def __init__(self):
        self.batch = 256 * MIB
        self.next_at = 0.0
        self.failures = 0
        self.pending = None
        self.state = "idle"
        self.windows_gain = 0
        self.cache_drop = 0

    def finish(self, now, *, windows, cache, requested):
        self.pending = (windows, cache, requested)
        self.next_at = now + 5
        self.state = "measuring"

    def request(self, now, *, windows, linux, cache, target):
        if now < self.next_at:
            return 0
        if self.pending is not None:
            before_win, before_cache, requested = self.pending
            self.pending = None
            self.windows_gain = windows - before_win
            self.cache_drop = before_cache - cache
            # A successful syscall or Linux cache drop alone is insufficient.
            benefit = min(64 * MIB, requested // 4)
            if self.windows_gain >= benefit and self.cache_drop >= benefit:
                self.failures = 0
                self.batch = min(GIB, self.batch * 2)
            else:
                self.failures += 1
                self.batch = 256 * MIB
                self.next_at = now + (10, 30, 60)[min(self.failures - 1, 2)]
                self.state = "backoff"
                return 0
        if windows >= target:
            self.state = "headroom restored"
            self.batch = 256 * MIB
            return 0
        if linux < target or cache < 768 * MIB:
            self.state = "insufficient reclaimable cache"
            return 0
        self.state = "reclaiming"
        return min(self.batch, cache - 512 * MIB)

    def status(self, now):
        return {
            "state": self.state,
            "batch_bytes": self.batch,
            "retry_in_s": round(max(0, self.next_at - now), 1),
            "windows_gain_bytes": self.windows_gain,
            "cache_drop_bytes": self.cache_drop,
        }


def inactive_cache_bytes():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    # Cached includes shmem (including expert banks). Never count that as disposable.
    return max(
        0,
        values.get("Inactive(file)", 0)
        - values.get("Dirty", 0)
        - values.get("Writeback", 0),
    )


def discover_cgroup(
    proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup"), *, uid=None
):
    uid = os.getuid() if uid is None else uid
    try:
        for line in proc.read_text().splitlines():
            if not line.startswith("0::/"):
                continue
            parts = Path(line[4:]).parts
            name = f"user@{uid}.service"
            if name not in parts or ".." in parts:
                continue
            group = root.joinpath(*parts[: parts.index(name) + 1])
            if os.access(group / "memory.reclaim", os.W_OK):
                return group
    except OSError:
        pass
    return None


def _write_reclaim(path, payload):
    with path.open("w") as stream:
        stream.write(payload)


def reclaim_cgroup(group, amount):
    try:
        _write_reclaim(group / "memory.reclaim", f"{amount} swappiness=0")
    except OSError as exc:
        if exc.errno == errno.EAGAIN:
            return "partial"
        raise ReclaimUnavailable(
            f"file-only cgroup reclaim unavailable: {exc}"
        ) from exc
    return "ok"


def release_completed_file(path):
    """Flush this completed regular file then advise away its cache; never delete it."""
    if os.name != "posix" or not hasattr(os, "posix_fadvise"):
        return False
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return False
            os.fdatasync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            return True
        finally:
            os.close(fd)
    except OSError:
        return False


def inactive_weight_files(active_models):
    """Only sibling model directories; no symlinks, partial files or active roots."""
    active = {Path(p).resolve() for p in active_models}
    active_inodes = {
        (s.st_dev, s.st_ino)
        for folder in active
        for p in folder.glob("*.safetensors")
        for s in [p.stat()]
    }
    files = []
    for parent in sorted({p.parent for p in active}):
        # Bound discovery, including on a misconfigured model path.
        for folder in sorted(parent.iterdir())[:256]:
            if folder.is_symlink() or not folder.is_dir() or folder.resolve() in active:
                continue
            if not (folder / "config.json").is_file():
                continue
            for path in sorted(folder.glob("*.safetensors"))[:1024]:
                if not path.is_symlink() and path.is_file():
                    info = path.stat()
                    if (info.st_dev, info.st_ino) not in active_inodes:
                        files.append(path)
    return files


def reclaim_files(files, amount, cursor=0):
    """Inspect at most 8 GiB per probe without faulting pages into memory.

    Rotate across files so repeated probes do not get stuck on already-cold
    regions. DONTNEED is advisory: count observed eviction, not bytes requested.
    """
    lib = ctypes.CDLL(None, use_errno=True)
    lib.mmap.restype = ctypes.c_void_p
    lib.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    page = os.sysconf("SC_PAGE_SIZE")
    block = 16 * MIB
    regions = [(p, off) for p in files for off in range(0, p.stat().st_size, block)]
    if not regions:
        return {"released_bytes": 0, "cursor": 0}
    released = scanned = 0
    visited = 0
    for i in range(min(len(regions), 8 * GIB // block)):
        index = (cursor + i) % len(regions)
        path, off = regions[index]
        visited += 1
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                continue
            n = min(block, info.st_size - off, amount - released)
            if n <= 0:
                continue
            ptr = lib.mmap(None, n, 1, 2, fd, off)  # PROT_READ, MAP_PRIVATE
            if ptr == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), "mmap")
            try:
                vector = (ctypes.c_ubyte * ((n + page - 1) // page))()
                if lib.mincore(ptr, n, vector):
                    raise OSError(ctypes.get_errno(), "mincore")
                before = sum(x & 1 for x in vector) * page
                if before:
                    # No fsync here: skip dirty pages rather than forcing download I/O.
                    os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
                    if lib.mincore(ptr, n, vector):
                        raise OSError(ctypes.get_errno(), "mincore")
                    released += max(0, before - sum(x & 1 for x in vector) * page)
            finally:
                lib.munmap(ptr, n)
            scanned += n
        finally:
            os.close(fd)
        if released >= amount:
            break
    return {
        "released_bytes": released,
        "scanned_bytes": scanned,
        "cursor": (cursor + visited) % len(regions),
    }


def probe(amount, port, cursor, skip_cgroup):
    group = discover_cgroup()
    reason = "no writable delegated user cgroup"
    if group is not None and not skip_cgroup:
        try:
            result = reclaim_cgroup(group, amount)
            return {"method": "cgroup", "result": result, "cursor": cursor}
        except ReclaimUnavailable as exc:
            reason = str(exc)
    elif skip_cgroup:
        reason = "kernel file-only reclaim unavailable (cached capability result)"
    # Resolve the SERVED model, not the boot file which may describe the next boot.
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/v1/models", timeout=2
    ) as response:
        models = json.load(response)["data"]
    roots = [Path(m["root"]) for m in models if m.get("root")]
    if not roots or not all(p.is_absolute() and p.is_dir() for p in roots):
        raise ReclaimUnavailable("cannot verify active model directories")
    result = reclaim_files(inactive_weight_files(roots), amount, cursor)
    return dict(result, method="inactive model files", cgroup_unavailable=reason)


class ReclaimController:
    """One subprocess at a time; no engine mutations or synchronous waits."""

    def __init__(self, port):
        self.port = port
        self.policy = ReclaimPolicy()
        self.process = None
        self.started = 0.0
        self.baseline = None
        self.cursor = 0
        self.skip_cgroup = False
        self.last_result = {}
        self.killed = False
        self._lock = threading.RLock()
        self._closed = False

    def cancel(self):
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                self.process.kill()
                self.killed = True

    def close(self):
        with self._lock:
            self._closed = True
            self.cancel()

    def _watch(self, process):
        # Independent of slow GPU probes and the governor's synchronous cache steps.
        # wait() reaps on shutdown too. A D-state task retains the one-worker slot.
        try:
            process.wait(timeout=PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with self._lock:
                if self.process is process:
                    self.cancel()
            process.wait()

    def tick(self, *, windows, linux, target, enabled=True):
        with self._lock:
            self._tick(
                windows=windows,
                linux=linux,
                target=target,
                enabled=enabled and not self._closed,
            )

    def _tick(self, *, windows, linux, target, enabled):
        now = time.monotonic()
        if self.process is not None:
            if self.process.poll() is None:
                if not enabled or now - self.started > 10:
                    self.cancel()
                    self.last_result = {"error": "probe cancelled or timed out"}
                return
            out, err = self.process.communicate()
            try:
                self.last_result = (
                    json.loads(out)
                    if not self.killed
                    else {"error": "probe cancelled or timed out"}
                )
            except (ValueError, TypeError):
                self.last_result = {"error": err[-300:] or "probe failed"}
            self.cursor = self.last_result.get("cursor", self.cursor)
            self.skip_cgroup |= bool(self.last_result.get("cgroup_unavailable"))
            self.process = None
            self.policy.finish(now, **self.baseline)
            logger.info("cache reclaim: %s", self.last_result)
        if not enabled:
            self.policy.state = "disabled"
            return
        if windows is None or linux is None:
            self.policy.state = "memory probe unavailable"
            return
        cache = inactive_cache_bytes()
        amount = self.policy.request(
            now, windows=windows, linux=linux, cache=cache, target=target
        )
        if not amount:
            return
        args = [
            sys.executable,
            "-m",
            "freetoken.daemon.settings.memory_reclaim",
            "--probe",
            str(amount),
            str(self.port),
            str(self.cursor),
            str(int(self.skip_cgroup)),
        ]
        try:
            self.process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        except OSError as exc:
            self.last_result = {"error": str(exc)}
            self.policy.finish(now, windows=windows, cache=cache, requested=amount)
            return
        self.baseline = {"windows": windows, "cache": cache, "requested": amount}
        self.started = now
        self.killed = False
        threading.Thread(
            target=self._watch,
            args=(self.process,),
            name="cache-reclaim-watchdog",
            daemon=True,
        ).start()

    def status(self):
        return dict(
            self.policy.status(time.monotonic()),
            running=self.process is not None,
            last_result=dict(self.last_result),
        )


def is_wsl():
    return os.name == "posix" and "microsoft" in os.uname().release.lower()


def watch(settings_port, engine_port):
    """Temporary activation without restarting the existing helper/model service.

    Retires as soon as the helper exposes the integrated reclaim controller.
    This process never calls a cache-step, settings-write or restart endpoint.
    """
    if not is_wsl():
        raise ReclaimUnavailable("automatic reclaim currently requires WSL")
    controller = ReclaimController(engine_port)
    target = 6 * GIB
    next_settings = 0
    last = None

    def get(route):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{settings_port}/api/{route}", timeout=3
        ) as response:
            return json.load(response)

    try:
        while True:
            try:
                status = get("status")
                governor = status.get("governor", {})
                if "reclaim" in governor:
                    logger.info(
                        "integrated governor detected; retiring temporary reclaim watcher"
                    )
                    return
                if time.monotonic() >= next_settings:
                    config = get("settings")
                    settings = config["settings"]
                    rung = config.get("model", {}).get("bytesPerLayer", int(1.33 * GIB))
                    target = int(
                        float(settings.get("GovernorRAMFreeGB", 4)) * GIB
                        + int(settings.get("GovernorRAMRungsBeforeUp", 2)) * rung
                        + float(settings.get("GovernorUpMarginGB", 0.5)) * GIB
                        + 256 * MIB
                    )
                    next_settings = time.monotonic() + 60
                windows = governor.get("free_windows_ram_gb")
                linux = governor.get("free_linux_ram_gb")
                server = status.get("server", {})
                controller.tick(
                    windows=None if windows is None else int(windows * GIB),
                    linux=None if linux is None else int(linux * GIB),
                    target=target,
                    enabled=(
                        governor.get("enabled", False)
                        and server.get("reachable", False)
                        and server.get("state") in ("serving", "rebuilding")
                        and os.environ.get("FREETOKEN_CACHE_RECLAIM", "1") != "0"
                    ),
                )
                current = controller.status()
                summary = (current["state"], current["running"])
                if summary != last:
                    logger.info("cache reclaim status: %s", current)
                    last = summary
            except Exception as exc:  # noqa: BLE001 - isolate optional watcher from API failures
                controller.cancel()
                logger.warning("cache reclaim watcher: %s", exc)
            time.sleep(2)
    finally:
        controller.close()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--watch":
        logging.basicConfig(level=logging.INFO)
        watch(int(sys.argv[2]), int(sys.argv[3]))
        sys.exit(0)
    try:
        if len(sys.argv) != 6 or sys.argv[1] != "--probe":
            raise ValueError("expected --probe BYTES PORT CURSOR SKIP_CGROUP")
        amount, port, cursor, skip = map(int, sys.argv[2:])
        if not 0 < amount <= GIB or not 0 < port < 65536:
            raise ValueError("probe outside bounds")
        print(json.dumps(probe(amount, port, cursor, bool(skip))))
    except Exception as exc:  # noqa: BLE001 - return a bounded diagnostic to the supervisor
        print(json.dumps({"error": str(exc)}))

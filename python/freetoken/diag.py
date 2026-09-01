"""Throwaway kernel-level profiling seam (decode-step breakdown study).

Nobody has ever seen a kernel-level breakdown of one decode step on this box: ~12.5 ms
of GPU compute plus ~4-6 ms of serialized PCIe expert fetch, and no idea which kernels
those are. This module is the seam that answers it, and it is default-off.

Enabled only when ``FREETOKEN_DIAG_PROFILE_DIR`` names a directory; the single cached
module-level bool below is the entire cost when it is unset, and ``torch.profiler`` is
not imported at all. Companion knobs::

    FREETOKEN_DIAG_PROFILE_DIR=<dir>     # arm; write <dir>/trace.json + key_averages.txt
    FREETOKEN_DIAG_PROFILE_STEPS=N       # scheduler iterations to record (default 400)
    FREETOKEN_DIAG_PROFILE_SKIP=M        # non-idle iterations to skip first (default 0)

The recorder arms itself at the first NON-IDLE scheduler iteration (an armed but
trafficless server simply waits), records ``STEPS`` of them -- prefill batches, plain
decode steps, speculative cycles, whatever arrives -- writes its two files, and never
re-arms.

Why the ranges device-sync on exit: a kernel's GPU timestamp only falls inside the CPU
range of the stage that LAUNCHED it if the host waits for that stage's device work
before closing the range. Without the sync every kernel piles up in whichever range
happened to be open when the queue drained, and per-stage attribution is impossible.
That sync is also why this is a diagnosis tool and never a serving one.
"""

from __future__ import annotations

import atexit
import os

import torch
from freetoken.utils import init_logger

logger = init_logger(__name__)

DEFAULT_STEPS = 400

# The one thing the hot path pays while this is unset.
ENABLED = False
_session: "_ProfileSession | None" = None


# ----------------------------------------------------------------------
# Ranges
# ----------------------------------------------------------------------


class _NullRegion:
    """The disabled (and pre-arm) range: three trivial calls and no allocation."""

    __slots__ = ()

    def __enter__(self) -> "_NullRegion":
        return self

    def __exit__(self, *exc) -> bool:
        return False


_NULL = _NullRegion()


def _sync() -> None:
    """Device-sync, unless syncing is illegal or meaningless here."""
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return
    if torch.cuda.is_current_stream_capturing():
        return  # a sync inside CUDA graph capture would abort the capture
    torch.cuda.synchronize()


class _Region:
    """A ``torch.profiler.record_function`` range that device-syncs before it closes."""

    __slots__ = ("_name", "_rf")

    def __init__(self, name: str) -> None:
        self._name = name
        self._rf = None

    def __enter__(self) -> "_Region":
        session = _session
        if session is None:
            return self
        # Entering ANY range is what tells the session this iteration was not idle --
        # true before the profiler starts, which is how the skip phase is counted.
        session.worked = True
        if session.recording:
            from torch.profiler import record_function

            self._rf = record_function(self._name)
            self._rf.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        rf, self._rf = self._rf, None
        if rf is None:
            return False
        try:
            _sync()
        finally:
            rf.__exit__(*exc)
        return False


def region(name: str | None):
    """The profiler range named ``name`` -- a no-op singleton while disabled.

    ``name=None`` is also a no-op, so a call site whose stage name depends on the batch
    can pass the choice straight in.
    """
    if not ENABLED or name is None:
        return _NULL
    return _Region(name)


# ----------------------------------------------------------------------
# The recording session
# ----------------------------------------------------------------------


def _make_profiler():
    """Build the profiler. Imported HERE so a disabled process never touches torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
        with_flops=False,
    )


def _key_averages_text(prof) -> str:
    """The per-kernel table, sorted by self device time across torch's renames of it."""
    averages = prof.key_averages()
    for key in ("self_device_time_total", "self_cuda_time_total"):
        try:
            return averages.table(sort_by=key, row_limit=80)
        except Exception:  # the other spelling, or a CPU-only run with no device column
            continue
    return averages.table(row_limit=80)


class _ProfileSession:
    """Arm -> skip -> record N iterations -> write -> done. Never re-arms."""

    def __init__(self, directory: str, steps: int, skip: int) -> None:
        self.directory = directory
        self.steps = max(1, int(steps))
        self.skip = max(0, int(skip))
        self.prof = None
        self.recording = False
        self.finished = False
        self.counted = 0
        self.skipped = 0
        self.worked = False

    def iteration(self) -> None:
        """One scheduler loop iteration has ended."""
        if self.finished:
            return
        worked, self.worked = self.worked, False
        if not worked:
            return  # armed, but nothing ran: an idle server must not burn the budget
        if self.prof is None:
            if self.skipped < self.skip:
                self.skipped += 1
                return
            self._start()
            return  # the iteration that starts the profiler ran outside it
        # No ``prof.step()``: the profiler runs scheduleless (one cycle, every event kept),
        # where step() emits no ProfilerStep marker and only warns about cycle clearing.
        self.counted += 1
        if self.counted >= self.steps:
            self._stop()

    def _start(self) -> None:
        try:
            self.prof = _make_profiler()
            self.prof.__enter__()
            self.recording = True
            logger.info(
                "diag profiler armed: recording %d scheduler iterations into %s",
                self.steps,
                self.directory,
            )
        except Exception as error:  # a diagnostic never takes the server down
            self.prof = None
            self.recording = False
            self.finished = True
            logger.warning("diag profiler could not start: %s", error)

    def _stop(self) -> None:
        prof, self.prof = self.prof, None
        self.recording = False
        self.finished = True  # before the writes: a failed write must not re-arm either
        if prof is None:
            return
        try:
            prof.__exit__(None, None, None)
            os.makedirs(self.directory, exist_ok=True)
            trace = os.path.join(self.directory, "trace.json")
            prof.export_chrome_trace(trace)
            table = os.path.join(self.directory, "key_averages.txt")
            with open(table, "w", encoding="utf-8") as handle:
                handle.write(_key_averages_text(prof))
            logger.info(
                "diag profiler: recorded %d iterations -> %s, %s", self.counted, trace, table
            )
        except Exception as error:
            logger.warning("diag profiler could not write its trace: %s", error)

    def close(self) -> None:
        """Salvage a partial recording at exit (a server killed mid-run still gets a trace)."""
        if self.prof is not None:
            self._stop()


def profile_step() -> None:
    """Called once per scheduler loop iteration; a single global read while disabled."""
    session = _session
    if session is not None:
        session.iteration()


def profile_loop_begin() -> None:
    """The serving loop is about to start.

    Boot-time graph capture runs the model, and the model's own ranges (``diag.ple_gather``)
    fire during it -- so without this the recorder would see the whole boot as one unit of
    traffic and arm itself on the first, idle, iteration. Only what the loop does counts.
    """
    session = _session
    if session is not None:
        session.worked = False


def enable(directory: str, steps: int = DEFAULT_STEPS, skip: int = 0) -> None:
    """Arm the seam. The env bootstrap below is the only production caller; tests use it too."""
    global ENABLED, _session
    _session = _ProfileSession(directory, steps, skip)
    ENABLED = True


def disable() -> None:
    global ENABLED, _session
    ENABLED = False
    _session = None


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def _close_atexit() -> None:
    session = _session
    if session is not None:
        try:
            session.close()
        except Exception:
            pass


def _bootstrap() -> None:
    directory = (os.getenv("FREETOKEN_DIAG_PROFILE_DIR") or "").strip()
    if not directory:
        return
    enable(
        directory,
        steps=_int_env("FREETOKEN_DIAG_PROFILE_STEPS", DEFAULT_STEPS),
        skip=_int_env("FREETOKEN_DIAG_PROFILE_SKIP", 0),
    )
    atexit.register(_close_atexit)


_bootstrap()


__all__ = [
    "ENABLED", "disable", "enable", "profile_loop_begin", "profile_step", "region",
]

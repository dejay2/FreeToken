"""Crash watchdog for the settings helper: start the model server again when it dies unasked.

Torch-free, like the rest of ``freetoken.daemon``. Seen live 2026-09-07: the scheduler process died
on a card fault at 23:32 and port 2020 stayed dead until a hand-pressed Start at 04:20, because the
API server shuts itself down ("Backend worker is gone and cannot be restarted") and nothing above it
watched. The watchdog arms itself whenever the server is seen serving (a page Start or an adopted
server), disarms on a page Stop, and after ``misses_needed`` consecutive unreachable probes queues
a restart job from the saved boot file. Restarts are capped per hour so a boot-time crash
cannot loop the card forever.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable

logger = logging.getLogger(__name__)

# "unreachable": nothing answers on the port. "failed": the API server's maintenance state after
# a backend worker died (api_server._on_failure, which then exits the process) or after a cache
# step/rebuild failed and left the gate closed; the governor's next successful step clears the
# latter, so a "failed" server that still has processes gets twice the patience below.
DEAD_STATES = frozenset({"unreachable", "failed"})
# "rebuilding" used to be alive-just-busy for ever: on 2026-09-09 16:25 the 5090's scheduler
# stopped answering a governor step, the API left its gate shut, /health kept saying ok and
# three requests sat at 0% GPU for 11 min 44 s before clients gave up, with no restart. The
# API now tracks the one operation holding its gate, restarts a clock on every unit of work
# the scheduler reports for it (a drained batch or prompt chunk ahead of it, each rebuild
# phase) and after MAINTENANCE_STUCK_S (600 s, api_server) of no progress reports
# ``maintenance.stuck``. That verdict is the ONLY thing that makes "rebuilding" count as dead
# here. The helper keeps no timer of its own over a reporting server: a healthy step can sit
# queued behind a long decode for longer than any fixed limit (review F2, 2026-09-09), and
# only the API knows whether work is still completing. The one exception below is a report
# that is plainly stale (no progress for STALE_PROGRESS_FACTOR times the API's own deadline
# without a verdict), which cannot happen while the API judges itself and so costs nothing.
# A wedged API answers nothing at all, which is "unreachable" and already counted dead.
STALE_PROGRESS_FACTOR = 2.0


def _server_pids(process_manager: Any) -> set[int] | None:
    """The server's processes on this port, or None when the helper cannot tell (Windows)."""
    if getattr(process_manager, "platform_windows", False):
        return None
    try:
        from .linux_launch import find_server_pids

        return find_server_pids(int(process_manager.port))
    except Exception:  # noqa: BLE001 - /proc scanning is best effort
        return None


class CrashWatchdog(threading.Thread):
    """Poll the server's readiness document and restart it when it dies while armed."""

    def __init__(
        self,
        process_manager: Any,
        *,
        interval: float = 10.0,
        misses_needed: int = 3,
        max_restarts_per_hour: int = 3,
        monotonic: Callable[[], float] = time.monotonic,
        wall_now: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(name="crash-watchdog", daemon=True)
        self.process_manager = process_manager
        self.interval = float(interval)
        self.misses_needed = int(misses_needed)
        self.max_restarts_per_hour = int(max_restarts_per_hour)
        self._legacy_rebuilding_warned = False
        self._monotonic = monotonic
        self._wall_now = wall_now
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._enabled: bool = True
        self.armed: bool = False
        self.misses: int = 0
        self.gave_up: bool = False
        self.last_reason: str | None = None
        self.last_restart_at: str | None = None
        self._restarts: deque[float] = deque()
        self._own_thread: int | None = None  # the watchdog thread while it queues its own job

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = bool(value)
            # A flip either way starts the count afresh: stale misses must not fire on the
            # first dead probe after a re-enable.
            self.misses = 0

    # ---- control ---------------------------------------------------------

    def stop(self) -> None:
        self._stop_event.set()

    def arm(self) -> None:
        """The server reached serving state (page Start, restart, or adopted): watch it."""
        with self._lock:
            self.armed = True
            self.misses = 0
            self.gave_up = False

    def disarm(self, reason: str | None = "stopped by the page") -> None:
        """A page Stop is a wish for the server to be down; never restart behind it.

        A lifecycle job the watchdog queued itself keeps it armed: if that reboot fails, the
        next dead probes count toward the hourly budget instead of leaving nobody watching.
        """
        with self._lock:
            if self._own_thread is not None and self._own_thread == threading.get_ident():
                return
            self.armed = False
            self.misses = 0
            if reason is not None:
                self.last_reason = reason

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval):
                break
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - never raise out of the thread
                logger.warning("crash watchdog tick error: %s", exc)

    # ---- one probe -------------------------------------------------------

    def tick(self) -> None:
        if not self.enabled:
            return
        if self.process_manager.current_job() is not None:
            # A start, stop or restart is in flight; its own readiness wait owns the outcome,
            # and a human's Start that fails must not be re-run by stale misses ten seconds later.
            with self._lock:
                self.misses = 0
            return
        # The raw /v1/cache/status document, as the boot wait reads it: its only liveness signal
        # is "state" (the merged server_status() adds "reachable" and an nvidia-smi call per tick).
        try:
            document = self.process_manager._readiness()
        except Exception as exc:  # noqa: BLE001 - unreachable is the signal we watch for
            document = {"state": "unreachable", "error": str(exc)}
        state = document.get("state") if isinstance(document, dict) else "unreachable"
        if state == "serving":
            with self._lock:
                if not self.armed:
                    logger.info("crash watchdog: adopted a serving model server")
                self.armed = True
                self.misses = 0
                self.gave_up = False
            return
        if state == "rebuilding":
            state = self._rebuilding_verdict(document)
            if state is None:
                with self._lock:
                    self.misses = 0
                return
        if state not in DEAD_STATES and not state.startswith("stuck"):
            # loading, stopping: alive, just busy
            with self._lock:
                self.misses = 0
            return
        with self._lock:
            if not self.armed or self.gave_up:
                return
            self.misses += 1
        # Second opinion before anything is killed: with no server process left the port is
        # simply dead; with processes still there (a hung server, or a stray listener such as an
        # ssh tunnel blocking the probe) wait twice as long.
        pids = _server_pids(self.process_manager)
        needed = self.misses_needed if not pids else 2 * self.misses_needed
        with self._lock:
            if not self.armed or self.gave_up or self.misses < needed:
                return  # a disarm or give-up may have landed while the lock was down
            now = self._monotonic()
            while self._restarts and now - self._restarts[0] > 3600.0:
                self._restarts.popleft()
            if len(self._restarts) >= self.max_restarts_per_hour:
                self.gave_up = True
                self.last_reason = (
                    f"gave up: {len(self._restarts)} restarts in the last hour; "
                    "start it by hand, the budget returns an hour after the first restart"
                )
                logger.error("crash watchdog: %s", self.last_reason)
                return
            self._restarts.append(now)
            self.misses = 0
            self.last_restart_at = self._iso(self._wall_now())
            # A plain start when nothing of the server is left; otherwise a restart, whose stop
            # path clears what remains (a shutting-down API process, orphaned tokenizer workers)
            # off the port and the card before the boot.
            action = "start" if pids is not None and not pids else "restart"
            self.last_reason = f"server stopped answering ({state}); {action}ing it"
            self._own_thread = threading.get_ident()
        logger.warning("crash watchdog: %s", self.last_reason)
        try:
            self.process_manager.start(action)
        except Exception as exc:  # noqa: BLE001 - report, try again on a later tick
            with self._lock:
                self.last_reason = f"{action} could not be queued: {exc}"
            logger.error("crash watchdog: %s", self.last_reason)
        finally:
            with self._lock:
                self._own_thread = None

    def _rebuilding_verdict(self, document: dict) -> str | None:
        """None while a rebuild is allowed to continue; otherwise the reason it counts as dead.

        The server's ``maintenance`` report is authoritative: ``stuck`` is its verdict that the
        open operation reported no progress for its deadline. Fresh progress (``progress_idle_s``
        under the deadline, whatever the operation's total age) is alive, and each operation
        is judged on its own record, so successive short operations never add up. A server
        that does not report at all (a build older than 2026-09-09) is treated the way it
        always was, alive-just-busy, and said so once: without progress data there is no
        limit that cannot kill a healthy long decode, and Jay's bar is no false restarts.
        """
        maintenance = document.get("maintenance")
        if not isinstance(maintenance, dict):
            if not self._legacy_rebuilding_warned:
                self._legacy_rebuilding_warned = True
                logger.warning(
                    "crash watchdog: the server reports no maintenance progress (older build); "
                    "a stuck cache operation will not be restarted until it runs this build"
                )
            return None
        if maintenance.get("stuck"):
            phase = maintenance.get("phase")
            return (
                "stuck in maintenance: the server reports its cache operation made no progress"
                + (f" (phase {phase})" if phase else "")
            )
        # Defence in depth for a report the server has stopped judging (its own check runs
        # inside the status handler, so this is unreachable on a healthy API): no progress for
        # STALE_PROGRESS_FACTOR times its deadline with no verdict counts as dead.
        try:
            idle = float(maintenance.get("progress_idle_s") or 0.0)
            deadline = float(maintenance.get("deadline_s") or 0.0)
        except (TypeError, ValueError):
            return None
        if deadline > 0 and idle > STALE_PROGRESS_FACTOR * deadline:
            return (
                f"stuck in maintenance: no progress reported for {int(idle)} s and the "
                f"server has not judged it (deadline {int(deadline)} s)"
            )
        return None

    # ---- reporting -------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self._monotonic()
            recent = sum(1 for t in self._restarts if now - t <= 3600.0)
            return {
                "enabled": self._enabled,
                "armed": self.armed,
                "misses": self.misses,
                "restarts_last_hour": recent,
                "last_restart_at": self.last_restart_at,
                "last_reason": self.last_reason,
                "gave_up": self.gave_up,
            }

    @staticmethod
    def _iso(stamp: float) -> str:
        import datetime as _dt

        return _dt.datetime.fromtimestamp(stamp, tz=_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

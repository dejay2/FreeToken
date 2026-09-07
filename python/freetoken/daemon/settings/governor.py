"""Torch-free memory governor for the FreeToken settings daemon.

Watches free VRAM and host RAM, steps expert layers down the VRAM/RAM ladder and shrinks
pools when free memory drops below cushion, and steps them back up when memory returns.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from .memory_fit import _read_vram_snapshot

logger = logging.getLogger("freetoken.daemon.settings.governor")

GIB = 1024 ** 3
DEFAULT_RUNG_BYTES = int(1.33 * GIB)  # 1.33 GiB per Qwen MoE layer
DEFAULT_MARGIN_BYTES = int(0.5 * GIB)  # 0.5 GiB step-up margin
DEFAULT_STEP_INTERVAL = 5.0  # 5 s
DEFAULT_UP_HOLD = 60.0  # 60 s
DEFAULT_MAX_HOLD = 600.0  # 600 s (10 min)


@dataclass(frozen=True)
class Action:
    axis: str  # "vram" | "ram"
    direction: str  # "down" | "up"
    ram_tight: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def read_free_vram_bytes(environ: Mapping[str, str] | None = None) -> int:
    """Read free VRAM in bytes using memory_fit._read_vram_snapshot."""
    env = os.environ if environ is None else environ
    free_bytes, total_bytes, uuid, name, source = _read_vram_snapshot(env)
    return free_bytes


_last_win_ram: tuple[float, int] | None = None
_last_win_ram_lock = threading.Lock()
_fallback_logged = False


def _read_proc_meminfo_available() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                parts = line.split()
                return int(parts[1]) * 1024
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def read_free_windows_ram_bytes() -> int:
    """Read free physical RAM in bytes.

    Uses powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"
    (reported in KB, cached for 2s, 3s timeout). Falls back to /proc/meminfo MemAvailable with a logged note.
    """
    global _last_win_ram, _fallback_logged
    now = time.monotonic()
    with _last_win_ram_lock:
        if _last_win_ram is not None and (now - _last_win_ram[0]) < 2.0:
            return _last_win_ram[1]

    try:
        proc = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            kb = int(proc.stdout.strip())
            free_bytes = kb * 1024
            with _last_win_ram_lock:
                _last_win_ram = (now, free_bytes)
            return free_bytes
    except Exception:  # noqa: BLE001
        pass

    if not _fallback_logged:
        logger.info("Windows RAM interop unavailable, falling back to /proc/meminfo MemAvailable")
        _fallback_logged = True

    try:
        free_bytes = _read_proc_meminfo_available()
        with _last_win_ram_lock:
            _last_win_ram = (now, free_bytes)
        return free_bytes
    except Exception as exc:
        raise RuntimeError(f"failed to read free RAM from both PowerShell and /proc/meminfo: {exc}") from exc


class GovernorPolicy:
    """Pure decision function governing cache steps on the VRAM and RAM ladders."""

    def __init__(
        self,
        vram_cushion: int,
        ram_cushion: int,
        rung_bytes: int = DEFAULT_RUNG_BYTES,
        margin: int = DEFAULT_MARGIN_BYTES,
        step_interval: float = DEFAULT_STEP_INTERVAL,
        up_hold: float = DEFAULT_UP_HOLD,
        max_hold: float = DEFAULT_MAX_HOLD,
    ) -> None:
        self.vram_cushion = int(vram_cushion)
        self.ram_cushion = int(ram_cushion)
        self.rung_bytes = int(rung_bytes)
        self.margin = int(margin)
        self.step_interval = float(step_interval)
        self.up_hold = float(up_hold)
        self.max_hold = float(max_hold)
        self._state: dict[str, Any] = {}

    def decide(
        self,
        now: float,
        free_vram: int,
        free_ram: int,
        last_actions: Any = None,
    ) -> list[Action]:
        """Decide next action(s) based on current free memory and timing."""
        state = last_actions if isinstance(last_actions, dict) else self._state
        ram_tight = free_ram < self.ram_cushion
        actions: list[Action] = []

        for axis in ("vram", "ram"):
            free = free_vram if axis == "vram" else free_ram
            cushion = self.vram_cushion if axis == "vram" else self.ram_cushion
            up_threshold = cushion + self.rung_bytes + self.margin

            axis_state = state.setdefault(
                axis,
                {
                    "last_step_time": -1e9,
                    "last_step_direction": None,
                    "up_hold": self.up_hold,
                    "high_since": None,
                    "doubled": False,
                },
            )
            if isinstance(last_actions, list):
                for act in reversed(last_actions):
                    if isinstance(act, tuple) and len(act) == 2:
                        t_val, act_obj = act
                    else:
                        t_val, act_obj = None, act
                    act_axis = getattr(act_obj, "axis", None) or (act_obj.get("axis") if isinstance(act_obj, dict) else None)
                    if act_axis == axis:
                        direction = getattr(act_obj, "direction", None) or (act_obj.get("direction") if isinstance(act_obj, dict) else None)
                        if axis_state["last_step_direction"] is None:
                            axis_state["last_step_direction"] = direction
                            if t_val is not None:
                                axis_state["last_step_time"] = t_val
                        break

            last_time = axis_state["last_step_time"]
            last_dir = axis_state["last_step_direction"]
            cur_hold = axis_state["up_hold"]
            high_since = axis_state["high_since"]
            doubled = axis_state.get("doubled", False)

            if free < cushion:
                axis_state["high_since"] = None
                # Check for step-up tripping cushion within step_interval -> double hold-off
                if last_dir == "up" and (now - last_time <= self.step_interval) and not doubled:
                    cur_hold = min(self.max_hold, cur_hold * 2.0)
                    axis_state["up_hold"] = cur_hold
                    axis_state["doubled"] = True

                if now - last_time >= self.step_interval:
                    actions.append(Action(axis=axis, direction="down", ram_tight=ram_tight))
                    axis_state["last_step_time"] = now
                    axis_state["last_step_direction"] = "down"
                    axis_state["doubled"] = False

            elif free >= up_threshold:
                if high_since is None:
                    high_since = now
                    axis_state["high_since"] = now

                if (now - high_since >= cur_hold) and (now - last_time >= self.step_interval):
                    actions.append(Action(axis=axis, direction="up", ram_tight=ram_tight))
                    axis_state["last_step_time"] = now
                    axis_state["last_step_direction"] = "up"
                    axis_state["high_since"] = now
                    axis_state["doubled"] = False
            else:
                axis_state["high_since"] = None

        return actions


class GovernorLoop(threading.Thread):
    """Periodic governor loop running every 2 s while the server is up."""

    def __init__(
        self,
        process_manager: Any,
        policy: GovernorPolicy,
        http_port: int = 2020,
    ) -> None:
        super().__init__(name="governor-loop", daemon=True)
        self.process_manager = process_manager
        self.policy = policy
        self.http_port = int(http_port)
        self._stop_event = threading.Event()
        self.enabled: bool = True
        self.last_action: str | None = None
        self.last_layers: dict[str, int] = {"owned": 0, "pinned": 0, "disk": 0}
        self.last_free_vram: int | None = None
        self.last_free_ram: int | None = None
        self.last_moe_cache_size: int | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(2.0):
                break
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 - never raise out of thread
                logger.warning("governor loop tick error: %s", exc)

    def _tick(self) -> None:
        if not self.enabled:
            return
        status = self.process_manager.server_status()
        if not status.get("reachable") or status.get("state") != "serving":
            return

        try:
            free_vram = read_free_vram_bytes()
        except Exception as exc:
            logger.debug("governor failed to probe VRAM: %s", exc)
            return

        try:
            free_ram = read_free_windows_ram_bytes()
        except Exception as exc:
            logger.debug("governor failed to probe RAM: %s", exc)
            return

        self.last_free_vram = free_vram
        self.last_free_ram = free_ram

        if self.last_moe_cache_size is None:
            self._query_residency()

        now = time.monotonic()
        actions = self.policy.decide(now, free_vram, free_ram)
        for action in actions:
            self._execute_action(action, free_vram, free_ram)

    def _execute_action(self, action: Action, free_vram: int, free_ram: int) -> None:
        url = f"http://127.0.0.1:{self.http_port}/v1/cache/step"
        body = {
            "axis": action.axis,
            "direction": action.direction,
            "ram_tight": action.ram_tight,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                reply = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                reply = json.loads(exc.read().decode("utf-8"))
            except Exception:
                reply = {"status": "failed", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("governor: failed to post step: %s", exc)
            return

        if reply.get("reason") == "disk rung not built" or "disk rung not built" in str(reply.get("error", "")):
            logger.info("governor: %s %s -> disk rung not built, retrying later", action.axis, action.direction)
            return

        if isinstance(reply.get("layers"), dict):
            self.last_layers = {
                "owned": reply["layers"].get("owned", 0),
                "pinned": reply["layers"].get("pinned", 0),
                "disk": reply["layers"].get("disk", 0),
            }
        old_slots = self.last_moe_cache_size
        new_slots = reply.get("moe_cache_size")
        if new_slots is not None:
            self.last_moe_cache_size = new_slots

        applied = reply.get("applied")
        vram_free_bytes = reply.get("vram_free_bytes", free_vram)
        old_vram_gb = free_vram / GIB
        new_vram_gb = vram_free_bytes / GIB

        if old_slots is not None and new_slots is not None and old_slots != new_slots:
            change_desc = f"slots {old_slots}->{new_slots}"
        elif applied:
            change_desc = str(applied)
        elif reply.get("at_floor"):
            change_desc = "at floor"
        else:
            change_desc = reply.get("status", "none")

        log_line = f"governor: {action.axis} {action.direction} -> {change_desc} (free {old_vram_gb:.1f}->{new_vram_gb:.1f} GiB)"
        logger.info(log_line)
        self.last_action = log_line

    def _query_residency(self) -> None:
        try:
            url = f"http://127.0.0.1:{self.http_port}/v1/cache/residency"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                rep = json.loads(resp.read().decode("utf-8"))
                if isinstance(rep, dict):
                    self.last_layers = {
                        "owned": rep.get("owned", 0),
                        "pinned": rep.get("pinned", 0),
                        "disk": rep.get("disk", 0),
                    }
                    if "moe_cache_size" in rep:
                        self.last_moe_cache_size = rep["moe_cache_size"]
        except Exception:  # noqa: BLE001
            pass

    def status(self) -> dict[str, Any]:
        free_vram_gb = round(self.last_free_vram / GIB, 2) if self.last_free_vram is not None else 0.0
        free_ram_gb = round(self.last_free_ram / GIB, 2) if self.last_free_ram is not None else 0.0
        return {
            "enabled": self.enabled,
            "last_action": self.last_action,
            "layers": dict(self.last_layers),
            "free_vram_gb": free_vram_gb,
            "free_ram_gb": free_ram_gb,
        }

"""Torch-free memory governor for the FreeToken settings daemon.

Watches free VRAM and host RAM, steps expert layers down the VRAM/RAM ladder and shrinks
pools when free memory drops below cushion, and steps them back up when memory returns.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Collection, Mapping

from .memory_fit import _read_vram_snapshot

logger = logging.getLogger("freetoken.daemon.settings.governor")

GIB = 1024 ** 3
# Fallback only when /v1/cache/residency cannot report the live model size. The 1.33 GiB
# value is the measured Qwen3.8 MoE-layer number (2026-09-07), not a universal rung size.
DEFAULT_RUNG_BYTES = int(1.33 * GIB)
DEFAULT_MARGIN_BYTES = int(0.5 * GIB)  # 0.5 GiB step-up margin
DEFAULT_STEP_INTERVAL = 5.0  # 5 s
DEFAULT_POST_UP_GRACE_MULTIPLIER = 2.0  # two intervals after a recall before ordinary down
DEFAULT_UP_HOLD = 60.0  # 60 s
DEFAULT_MAX_HOLD = 600.0  # 600 s (10 min)
BOOT_SETTLE_SECONDS = 120.0  # no down steps this long after the server starts serving (boot dip)


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


# The helper runs as a systemd --user service whose PATH has no /mnt/c/Windows entries, so a
# bare "powershell.exe" fails with "No such file" and the reader silently fell back to the VM's
# MemAvailable (seen live 2026-09-07 18:00: governor said 16.5 GiB free while Windows had 2.8 GB,
# so the RAM axis never stepped). The absolute path works from a service without any interop
# env; measured with systemd-run on the serving box.
_POWERSHELL_ABS = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def _powershell_candidates() -> list[str]:
    found = shutil.which("powershell.exe")
    out = [found] if found else []
    if os.path.exists(_POWERSHELL_ABS) and _POWERSHELL_ABS not in out:
        out.append(_POWERSHELL_ABS)
    return out


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

    for exe in _powershell_candidates():
        try:
            proc = subprocess.run(
                [
                    exe,
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except Exception:  # noqa: BLE001
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            kb = int(proc.stdout.strip())
            free_bytes = kb * 1024
            with _last_win_ram_lock:
                _last_win_ram = (now, free_bytes)
            return free_bytes

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
    """Decide cache steps while shielding recalls from their own memory cost.

    The default RAM and VRAM rung counts preserve the measured Qwen behaviour, while the
    live model report can replace ``rung_bytes`` for other expert-bank layouts.
    """

    def __init__(
        self,
        vram_cushion: int,
        ram_cushion: int,
        rung_bytes: int = DEFAULT_RUNG_BYTES,
        margin: int = DEFAULT_MARGIN_BYTES,
        step_interval: float = DEFAULT_STEP_INTERVAL,
        up_hold: float = DEFAULT_UP_HOLD,
        max_hold: float = DEFAULT_MAX_HOLD,
        post_up_grace: float | None = None,
        ram_rungs_before_up: int = 2,
        vram_rungs_before_up: int = 1,
    ) -> None:
        self.vram_cushion = int(vram_cushion)
        self.ram_cushion = int(ram_cushion)
        self.rung_bytes = int(rung_bytes)
        self.margin = int(margin)
        self.step_interval = float(step_interval)
        self.post_up_grace = (
            float(post_up_grace)
            if post_up_grace is not None
            else DEFAULT_POST_UP_GRACE_MULTIPLIER * self.step_interval
        )
        self.up_hold = float(up_hold)
        self.max_hold = float(max_hold)
        self.ram_rungs_before_up = int(ram_rungs_before_up)
        self.vram_rungs_before_up = int(vram_rungs_before_up)
        self._state: dict[str, Any] = {}

    def note_step_done(self, axis: str, now: float) -> None:
        """Re-stamp ``axis``'s last step at the moment its POST returned.

        ``decide`` stamps the step when it is chosen, but a step is a rebuild that takes
        seconds (graph recapture, a 1.33 GiB layer move). Counting the 5 s interval and the
        flap window from the *start* meant a step-up whose rebuild took 6 s could never be
        seen "tripping the cushion within the interval" (the next tick was already past it),
        and the next step could fire the moment a long rebuild finished."""
        axis_state = self._state.get(axis)
        if axis_state is not None:
            axis_state["last_step_time"] = float(now)

    def decide(
        self,
        now: float,
        free_vram: int,
        free_ram: int,
        last_actions: Any = None,
        allow_down: bool = True,
        exhausted_up: Collection[str] = (),
    ) -> list[Action]:
        """Decide next action(s) based on current free memory and timing.

        ``exhausted_up`` names axes whose last recall reply said nothing is left to recall;
        an up step there is neither chosen nor stamped (a phantom "up" stamp would put the
        next real down inside the post-up grace), while the high-memory hold keeps running
        so the first recall after the axis is cleared is still immediate.
        """
        state = last_actions if isinstance(last_actions, dict) else self._state
        ram_tight = free_ram < self.ram_cushion
        actions: list[Action] = []

        for axis in ("vram", "ram"):
            free = free_vram if axis == "vram" else free_ram
            cushion = self.vram_cushion if axis == "vram" else self.ram_cushion
            rungs_before_up = (
                self.ram_rungs_before_up if axis == "ram" else self.vram_rungs_before_up
            )
            up_threshold = cushion + rungs_before_up * self.rung_bytes + self.margin

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
            since_last = now - last_time
            in_post_up_grace = last_dir == "up" and since_last <= self.post_up_grace

            if free < cushion:
                axis_state["high_since"] = None
                # The measured recall cost can trip the cushion during this window. Keep the
                # old tripped-cushion hold doubling, but key it to the longer grace window.
                if in_post_up_grace and not doubled:
                    cur_hold = min(self.max_hold, cur_hold * 2.0)
                    axis_state["up_hold"] = cur_hold
                    axis_state["doubled"] = True

                # A real second squeeze (more than one rung below the cushion) overrides the
                # grace; the ordinary post-recall dip must wait until the grace expires.
                hard_squeeze = free < cushion - self.rung_bytes
                # allow_down=False (the loop's boot settle window) must leave no trace: a dropped
                # step must not stamp last_step_time, or the next tick waits a full interval.
                if allow_down and since_last >= self.step_interval and (not in_post_up_grace or hard_squeeze):
                    actions.append(Action(axis=axis, direction="down", ram_tight=ram_tight))
                    axis_state["last_step_time"] = now
                    axis_state["last_step_direction"] = "down"
                    axis_state["doubled"] = False

            elif free >= up_threshold:
                if high_since is None:
                    high_since = now
                    axis_state["high_since"] = now

                if (
                    (now - high_since >= cur_hold)
                    and (since_last >= self.step_interval)
                    and axis not in exhausted_up
                ):
                    actions.append(Action(axis=axis, direction="up", ram_tight=ram_tight))
                    axis_state["last_step_time"] = now
                    axis_state["last_step_direction"] = "up"
                    # Keep high_since: once the first 60 s hold is served, high memory allows
                    # burst recalls every step interval instead of another full hold each time.
                    axis_state["doubled"] = False
            else:
                # Any dip below the generous-zone threshold starts a fresh hold next time.
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
        self.last_layers: dict[str, int] = {"owned": 0, "pinned": 0, "disk": 0, "parked": 0}
        self.last_free_vram: int | None = None
        self.last_free_ram: int | None = None
        self._serving_since: float | None = None
        self.last_moe_cache_size: int | None = None
        self.last_num_pages: int | None = None
        self._last_residency_query = -float("inf")
        self._floor_logged: dict[str, bool] = {}
        # Axes whose last "up" reply was exhausted (nothing left to recall or promote). Until
        # 2026-09-09 the loop re-asked every step interval once the high-memory hold had been
        # served: one live boot on the 5090 answered 847 of 861 governor steps as no-ops, each
        # closing the API gate and syncing the card for nothing. The engine's exhausted verdict
        # depends on layer placement (owned/pinned/disk/parked), the slot count and the KV pool
        # size, so EVERY axis leaves this set as soon as any of those is seen to differ from
        # the last snapshot -- in a step reply (an applied step on either axis: a VRAM spill
        # creates a layer for RAM to recall, review F3) or in the minute residency refresh (a
        # hand-driven rebuild that shrank slots or KV pages, review F4) -- and when the server
        # goes away. The comparison runs before the snapshot is overwritten.
        self._up_exhausted: set[str] = set()

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
            self._serving_since = None
            self._up_exhausted.clear()
            return
        now_mono = time.monotonic()
        if self._serving_since is None:
            self._serving_since = now_mono
        # Boot dip: Windows free memory sags for about a minute right after the model server
        # starts serving, while WSL hands the weight-loading cache back. On 2026-09-07 21:01:53
        # the API came up and the RAM axis spilled seven layers between 21:01:57 and 21:02:32
        # with no other program running, then recalled them all once Windows read 13 GB free.
        # Down steps wait out this settle window; a real squeeze that persists past it is
        # still acted on, and up steps are never delayed by it.
        settling = now_mono - self._serving_since < BOOT_SETTLE_SECONDS

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

        now = time.monotonic()
        # The model's expert-bank size is stable for a boot, but query it periodically so a
        # late-starting server (or a transiently unavailable first reply) can correct the
        # Qwen fallback without adding a request on every 2 s governor tick.
        if now - self._last_residency_query >= 60.0:
            self._last_residency_query = now
            self._query_residency()
            # Do not charge the residency HTTP round trip to the policy's step interval.
            now = time.monotonic()

        actions = self.policy.decide(
            now, free_vram, free_ram, allow_down=not settling, exhausted_up=self._up_exhausted
        )
        for action in actions:
            self._execute_action(action, free_vram, free_ram)
            # The POST blocks for the whole rebuild; the interval and the flap window count
            # from its completion, not from when the step was chosen.
            self.policy.note_step_done(action.axis, time.monotonic())

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
        if reply.get("status") == "unsupported":
            # TP > 1 or a cache without runtime rebuild (a dense model): permanent for this
            # boot. Every retry would flip the API's maintenance state for nothing, so stop
            # here; the next server start builds a fresh loop.
            logger.warning(
                "governor: server reports cache steps unsupported (%s); governor idle until the next start",
                reply.get("error", ""),
            )
            self.enabled = False
            self.last_action = "off: " + str(reply.get("error", "steps unsupported"))
            return

        # Rejected/busy/failed replies carry default slots=0, not a memory snapshot. J11
        # reproduced those defaults clearing RAM exhaustion after a rejected VRAM step.
        ok = reply.get("status") == "ok"
        applied = reply.get("applied") if ok else None
        layers = None
        if ok and isinstance(reply.get("layers"), dict):
            layers = {
                "owned": reply["layers"].get("owned", 0),
                "pinned": reply["layers"].get("pinned", 0),
                "disk": reply["layers"].get("disk", 0),
                "parked": reply["layers"].get("parked", self.last_layers.get("parked", 0)),
            }
        old_slots = self.last_moe_cache_size
        new_slots = reply.get("moe_cache_size") if ok else None
        # Compare before overwriting: a change on EITHER axis invalidates every remembered
        # "nothing to do" (a VRAM spill leaves a disk layer for RAM to recall, review F3).
        self._forget_exhausted_if_changed(layers=layers, slots=new_slots, applied=bool(applied))
        if layers is not None:
            self.last_layers = layers
        if new_slots is not None:
            self.last_moe_cache_size = new_slots

        if ok and action.direction == "up" and reply.get("exhausted"):
            self._up_exhausted.add(action.axis)
        vram_free_bytes = reply.get("vram_free_bytes", free_vram)
        old_vram_gb = free_vram / GIB
        new_vram_gb = vram_free_bytes / GIB

        at_floor_only = False
        slots_changed = old_slots is not None and new_slots is not None and old_slots != new_slots
        if applied and "->" in str(applied) and slots_changed:
            # A park or unpark moves a layer and the slot cache together; say both.
            change_desc = f"{applied} layer {reply.get('layer')}, slots {old_slots}->{new_slots}"
        elif slots_changed:
            change_desc = f"slots {old_slots}->{new_slots}"
        elif applied:
            change_desc = str(applied)
        elif reply.get("at_floor"):
            change_desc = "at floor"
            at_floor_only = True
        elif reply.get("reason") or reply.get("error"):
            # e.g. "park rejected, slots restored: ..." — the one line that explains a no-op step
            change_desc = str(reply.get("reason") or reply.get("error"))
        else:
            change_desc = reply.get("status", "none")

        if action.axis == "ram":
            # The step reply carries no RAM figure; log the reading the decision used.
            free_desc = f"free RAM {free_ram / GIB:.1f} GiB"
        else:
            free_desc = f"free {old_vram_gb:.1f}->{new_vram_gb:.1f} GiB"
        log_line = f"governor: {action.axis} {action.direction} -> {change_desc} ({free_desc})"
        # At the floor the policy keeps asking every 5 s (the KV rung opens when the server
        # goes idle), so say it once per floor episode and whisper the repeats.
        if at_floor_only and self._floor_logged.get(action.axis):
            logger.debug(log_line)
        else:
            logger.info(log_line)
        self._floor_logged[action.axis] = at_floor_only
        self.last_action = log_line

    def _query_residency(self) -> None:
        try:
            url = f"http://127.0.0.1:{self.http_port}/v1/cache/residency"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                rep = json.loads(resp.read().decode("utf-8"))
                if isinstance(rep, dict):
                    layers = {
                        "owned": rep.get("owned", 0),
                        "pinned": rep.get("pinned", 0),
                        "disk": rep.get("disk", 0),
                        "parked": len(rep.get("ram_parked") or []),
                    }
                    # Geometry moved behind the loop's back (a hand-driven rebuild between two
                    # polls, review F4): every fact the exhausted verdict rests on is compared
                    # before the snapshot is replaced. An older server reports no num_pages.
                    self._forget_exhausted_if_changed(
                        layers=layers,
                        slots=rep.get("moe_cache_size"),
                        pages=rep.get("num_pages"),
                    )
                    self.last_layers = layers
                    if "moe_cache_size" in rep:
                        self.last_moe_cache_size = rep["moe_cache_size"]
                    if rep.get("num_pages") is not None:
                        self.last_num_pages = rep["num_pages"]
                    try:
                        layer_bytes = int(rep.get("layer_bytes", 0) or 0)
                    except (TypeError, ValueError, OverflowError):
                        layer_bytes = 0
                    self.policy.rung_bytes = layer_bytes if layer_bytes > 0 else DEFAULT_RUNG_BYTES
        except Exception:  # noqa: BLE001
            pass

    def _forget_exhausted_if_changed(
        self,
        *,
        layers: dict[str, int] | None = None,
        slots: int | None = None,
        pages: int | None = None,
        applied: bool = False,
    ) -> None:
        """Drop every remembered exhausted verdict when a fact it rests on has changed.

        Called BEFORE the caller stores the new snapshot, with only the facts the reply or
        report carried (None = not reported, never a change). If exhaustion was remembered
        before a geometry field became known, recheck once: failed residency queries may
        have hidden a size change. Later identical reports preserve suppression.
        """
        if not self._up_exhausted:
            return
        changed = applied
        if layers is not None and layers != self.last_layers:
            changed = True
        if slots is not None and slots != self.last_moe_cache_size:
            changed = True
        if pages is not None and pages != self.last_num_pages:
            changed = True
        if changed:
            logger.info(
                "governor: memory geometry changed (layers %s->%s, slots %s->%s, kv pages %s->%s%s); "
                "recall may have work again on %s",
                self.last_layers, layers if layers is not None else self.last_layers,
                self.last_moe_cache_size, slots if slots is not None else self.last_moe_cache_size,
                self.last_num_pages, pages if pages is not None else self.last_num_pages,
                ", step applied" if applied else "",
                ",".join(sorted(self._up_exhausted)),
            )
            self._up_exhausted.clear()

    def status(self) -> dict[str, Any]:
        free_vram_gb = round(self.last_free_vram / GIB, 2) if self.last_free_vram is not None else 0.0
        free_ram_gb = round(self.last_free_ram / GIB, 2) if self.last_free_ram is not None else 0.0
        return {
            "enabled": self.enabled,
            "last_action": self.last_action,
            "layers": dict(self.last_layers),
            "free_vram_gb": free_vram_gb,
            "free_ram_gb": free_ram_gb,
            "up_exhausted": sorted(self._up_exhausted),
        }

"""The control panel's server side (own model system part 2, Stage A).

Every change goes through PanelService._save:
  1. check the page's revision;
  2. apply the change to a copy of the registry and validate it (also against each FreeToken
     model's own limits);
  3. render the switcher config for the registry before and after (no holds) and compare each
     loaded model's entry, and for a loaded FreeToken model also its profile settings;
  4. ask "restart now / next time" when a loaded model is affected;
  5. let the switcher check the new file (--check-config);
  6. save the registry, then swap the file in.
A refused check saves nothing. "Next time" keeps the loaded model's old entry in the file (a
hold), because patch P5 stops a loaded model whose entry changes; the hold watcher lets it go
once the model is no longer loaded. The hold is released within 5 s of the unload; a reload of
the same model inside that window still gets the old settings. "Restart now" unloads first,
writes, waits until the switcher reports the new file's hash (P5), then loads again through
P6. The wait is bounded (restart_wait_s): a reload that fails for a reason --check-config does
not catch never changes the hash, so the panel gives up with a plain message instead of loading
the model on the old settings.

"Affected" compares render_config(current, {}) with render_config(proposed, {}) rather than the
file on disk, because the file still carries held entries: comparing against it would ask about
an already-held model on every later save. Existing holds carry over untouched. A FreeToken
entry holds only the folder and the --profile id, so a FreeToken model also counts as affected
when freetoken_profile_settings changes (freetoken.sh pushes the profile at every load, so
"restart" is unload plus load and "next time" needs no hold).
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import ninfer_dials, ninfer_fit
from .boot_parser import BootFile, BootParseError
from .dials import DIAL_BY_NAME, DIALS, GROUP_INFO, Dial, dial_value_for_display, validate_settings
from .memory_fit import EstimateUnavailable, SettingsValidationError
from .model_info import read_model
from .profiles_manager import ProfileError, ProfileValidationError
from .registry import (
    ACTIVE, ENGINE_LABELS, ENGINES, RegistryCorrupt, RegistryError, RegistryMissing, RegistryValidationError,
    StaleRevision, base_settings, canonical_engine_settings, canonicalize, differences, effective_settings,
    engine_defaults, expand, find_model, freetoken_profile_settings, idle_minutes, preset_values, same,
    validate_registry,
)
from .registry_import import ImportRefused, import_live
from .swap_config import HEADER, SwitcherRefused, config_sha256, extract_model_blocks, profile_id, render_config
from .switcher import LOADED_STATES, SwitcherError, is_down

GIB = 1024 ** 3
PROFILE_NOTE = "Control panel settings for this model"
IMPORT_UNKNOWN_MESSAGE = ("Can't tell whether a model is loaded right now, so the import would risk "
                          "restarting it. Try again in a moment.")
REWRITE_UNKNOWN_MESSAGE = ("Can't tell whether a model is loaded right now, so this could restart it. "
                           "Try again in a moment.")
# llama-swap polls its config file every 2 s; give it this long after a write before the
# Right-now strip says it is still on older settings (final review item 5).
STALE_GRACE_S = 10.0
# A failed restart stays on the Right-now strip this long, or until the next save/load/unload.
RESTART_SHOWN_S = 600.0
RESTART_STALE_MESSAGE = "The switcher didn't pick up the new settings; the old ones are still in use. Check the switcher log."

IDENTITY_GROUP = "This model"
IDENTITY_DIALS: tuple[Dial, ...] = (
    Dial("model.name", "text", "", "", "The name chat apps show.", IDENTITY_GROUP,
         plain="Name in apps", blurb="The name chat apps show for this model."),
    Dial("model.aliases", "text", "", "", "Other names, separated by commas.", IDENTITY_GROUP,
         plain="Other names", blurb="Extra names apps can ask for, separated by commas."),
    Dial("model.ramNeedGB", "number", 0, "GB", "PC memory this model needs to load.", IDENTITY_GROUP,
         minimum=0, maximum=512, numeric_kind="float", plain="PC memory it needs",
         blurb="How much PC memory loading it takes. The switcher waits until this much is free."),
    Dial("model.idleMinutes", "number", -1, "minutes", "Unload after this many idle minutes.", IDENTITY_GROUP,
         minimum=-1, maximum=1440, auto_value=-1, auto_label="Same as the System tab", plain="Unload when idle",
         blurb="Minutes without chats before it unloads. 0 keeps it loaded."),
)
SYSTEM_GROUP = "System"
SYSTEM_DIALS: tuple[Dial, ...] = (
    Dial("floorGB", "number", 6, "GB", "Windows free memory kept after a load.", SYSTEM_GROUP,
         minimum=0, maximum=64, numeric_kind="float", plain="PC memory cushion",
         blurb="Keep at least this much PC memory free for Windows after a model loads.",
         info="Measured 2026-09-24: a FreeToken load takes Windows from 57.5 GB free to 0.3 GB without "
              "the cushion. 6 GB keeps Windows responsive."),
    Dial("waitSeconds", "number", 300, "seconds", "How long a load waits for memory.", SYSTEM_GROUP,
         minimum=0, maximum=3600, plain="Wait for memory",
         blurb="How long a model waits for PC memory to free up before giving up."),
    Dial("latestWins", "toggle", True, "switch", "A new pick cancels a half-finished load.", SYSTEM_GROUP,
         plain="Newest pick wins", blurb="Picking another model cancels a half-finished load."),
    Dial("defaultIdleMinutes", "number", 0, "minutes", "Idle minutes before a model unloads.", SYSTEM_GROUP,
         minimum=0, maximum=1440, plain="Unload when idle (all models)",
         blurb="Minutes without chats before a model unloads. 0 keeps it loaded. A model can set its own."),
)


class ChooseRestart(RuntimeError):
    def __init__(self, affected: list[dict[str, str]], next_time_allowed: bool = True) -> None:
        super().__init__("a loaded model is affected")
        self.affected, self.next_time_allowed = affected, next_time_allowed


class PanelError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.payload = status, {"code": code, "message": message}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _number(value: Any) -> int | float:
    if isinstance(value, bool):
        raise ValueError("not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("not a number")
    return int(number) if number.is_integer() else number


def _whole(value: Any) -> int:
    number = _number(value)
    if not isinstance(number, int):
        raise ValueError("not whole")
    return number


def _split_aliases(value: Any) -> list[str]:
    items = value if isinstance(value, list) else str(value or "").split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def _display(engine: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    if engine != "freetoken":
        return dict(settings)
    return {name: dial_value_for_display(DIAL_BY_NAME[name], value)
            for name, value in settings.items() if name in DIAL_BY_NAME and name != "ModelPath"}


def _engine_dials(engine: str, values: Mapping[str, Any], runtime: str | None, model_info: Any = None) -> tuple[list, list]:
    if engine == "ninfer":
        return ninfer_dials.dial_dicts(values, runtime), ninfer_dials.group_list()
    dials = [dial.as_dict(values.get(dial.name, dial.default), model_info) for dial in DIALS if dial.name != "ModelPath"]
    groups = [{"name": name, "plain": info.get("plain", name), "info": info.get("info", "")} for name, info in GROUP_INFO.items()]
    return dials, groups


def _unknown_save_message(names: list[str]) -> str:
    if len(names) == 1:
        who, pronoun = f"{names[0]} is", "it"
    else:
        who, pronoun = f"{', '.join(names[:-1])} or {names[-1]} is", "one of them"
    return f"Can't tell whether {who} loaded right now, so saving could restart {pronoun}. Try again in a moment."


def _default_card_probe() -> dict[str, int] | None:
    from .memory_fit import probe_machine

    try:
        snapshot = probe_machine()
    except Exception:  # noqa: BLE001 - no card reading is shown as a dash
        return None
    total = int(snapshot.get("vram_total_bytes") or 0)
    if total <= 0:
        return None
    return {"totalBytes": total, "usedBytes": max(0, total - int(snapshot.get("vram_free_bytes") or 0))}


def _default_windows_free() -> int | None:
    from .governor import read_free_windows_ram_bytes

    try:
        return read_free_windows_ram_bytes()
    except Exception:  # noqa: BLE001
        return None


def _default_spawn(fn: Callable[..., Any], *args: Any) -> None:
    threading.Thread(target=fn, args=args, name="panel-restart", daemon=True).start()


class PanelService:
    def __init__(self, *, store, writer, switcher, profiles, boot_file: Callable[[], BootFile],
                 default_boot: Callable[[], Path], estimate_service, card_probe=None, windows_free_probe=None,
                 holds_path=None, artifact_size=None, spawn=None, clock=time.monotonic, sleep=time.sleep,
                 restart_wait_s: float = 30.0) -> None:
        self.store, self.writer, self.switcher, self.profiles = store, writer, switcher, profiles
        self._boot_file, self._default_boot, self.estimate_service = boot_file, default_boot, estimate_service
        self._card_probe = card_probe or _default_card_probe
        self._windows_free = windows_free_probe or _default_windows_free
        self.holds_path = Path(holds_path) if holds_path else store.path.with_name("held-models.json")
        self._artifact_size = artifact_size or os.path.getsize
        self._spawn = spawn or _default_spawn
        self._clock, self._sleep, self.restart_wait_s = clock, sleep, restart_wait_s
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None
        self.last_restart: dict[str, Any] | None = None
        self._last_restart_at: float | None = None
        self._last_write: float | None = None

    # ---- small helpers ----
    def _ask(self) -> tuple[list[str] | None, bool]:
        """(loaded model ids, down).

        Loaded is None when the switcher's state is unknown (timeout, HTTP error, bad body):
        that is not "nothing loaded", because a single failed /running call must not release a
        "Next time" hold (P5 then stops the still-loaded model, fix round 1). A refused
        connection is "down": nothing is loaded, so saves go through (final review, ruling
        2026-09-24); holds are kept while it is down, which is harmless."""
        running = self.switcher.running()
        if running is None:
            return None, False
        return sorted(model_id for model_id, state in running.items() if state in LOADED_STATES), is_down(running)

    def _loaded_or_unknown(self) -> list[str] | None:
        return self._ask()[0]

    def _loaded(self) -> list[str]:
        return self._loaded_or_unknown() or []

    def _live_holds(self, loaded: list[str] | None, down: bool = False) -> dict[str, str]:
        """Holds whose model is still loaded; every hold when the state is unknown or the
        switcher is down."""
        holds = self._read_holds()
        return holds if loaded is None or down else {m: t for m, t in holds.items() if m in loaded}

    def _mark_written(self) -> None:
        self._last_write = self._clock()

    def _clear_restart(self) -> None:
        self.last_restart, self._last_restart_at = None, None

    @staticmethod
    def _named(doc: Mapping[str, Any], ids: list[str]) -> list[dict[str, str]]:
        names = {m["id"]: m["name"] for m in doc.get("models") or []}
        return [{"id": model_id, "name": names.get(model_id, model_id)} for model_id in ids]

    def _read_holds(self) -> dict[str, str]:
        try:
            data = json.loads(self.holds_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _write_holds(self, holds: Mapping[str, str]) -> None:
        self.holds_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.holds_path.with_name(self.holds_path.name + ".tmp")
        temporary.write_text(json.dumps(dict(holds), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.holds_path)

    def _ensure_profiles(self, doc: Mapping[str, Any]) -> None:
        for model in doc["models"]:
            if model["engine"] != "freetoken":
                continue
            try:
                self.profiles.upsert(profile_id(model["id"]), name=model["name"], description=PROFILE_NOTE,
                                     settings=freetoken_profile_settings(doc, model), create_only=True)
            except (ProfileError, ProfileValidationError):
                pass  # freetoken.sh --profile pushes the profile again at every load

    def _model_limit_errors(self, doc: Mapping[str, Any]) -> list[dict[str, str]]:
        errors: list[dict[str, str]] = []
        for model in doc["models"]:
            if model["engine"] != "freetoken":
                continue
            effective = effective_settings(doc, model)
            info = read_model(effective.pop("ModelPath"))
            errors += [{**error, "where": model["id"]} for error in validate_settings(effective, info)]
        return errors

    # ---- the one save path ----
    def _save(self, mutate: Callable[[dict], dict], revision: str | None, when_loaded: str | None,
              extra: Callable[[dict, dict], dict] | None = None) -> dict[str, Any]:
        if when_loaded not in (None, "restart", "next-time"):
            raise RegistryValidationError([{"field": "whenLoaded", "message": "Choose restart or next-time."}])
        with self._lock:
            current, current_revision = self.store.load()
            if revision != current_revision:
                raise StaleRevision("The model list was changed somewhere else.")
            proposed = canonicalize(mutate(copy.deepcopy(current)))
            errors = validate_registry(proposed) or self._model_limit_errors(proposed)
            if errors:
                raise RegistryValidationError(errors)
            old_text = self.writer.current_text() or ""
            known, down = self._ask()
            # Unknown switcher state (fix rounds 1-2): keep every hold as it is (the watcher
            # releases them once the state is known again). A change to a held model is safe:
            # the hold pins the running entry in the file and the change applies at the next
            # load. A change to any other model's entry is refused, because that model may be
            # loaded and P5 would stop it. Saves that move no entry (most System fields, a
            # FreeToken profile-only change, which the adapter applies at the next load) go through.
            loaded = known or []
            holds = self._live_holds(known, down)
            before = extract_model_blocks(render_config(current, {}))
            after = extract_model_blocks(render_config(proposed, {}))
            if known is None:
                moved = [m for m in after if m not in holds and before.get(m) != after[m]]
                if moved:
                    raise PanelError(503, "switcher_unknown", _unknown_save_message(
                        [row["name"] for row in self._named(proposed, moved)]))
            affected = self._affected(current, proposed, loaded, before, after)
            if affected and when_loaded is None:
                raise ChooseRestart(self._named(proposed, affected))
            file_blocks = extract_model_blocks(old_text)
            restarting: list[str] = []
            for model_id in affected:
                running_block = holds.get(model_id) or file_blocks.get(model_id) or before.get(model_id)
                entry_changed = before.get(model_id) != after.get(model_id)
                if when_loaded == "next-time":
                    # A FreeToken profile-only change needs no hold: the adapter pushes the
                    # profile at the next load. A changed entry keeps what is running now.
                    if entry_changed and running_block is not None:
                        holds[model_id] = running_block
                else:
                    holds.pop(model_id, None)
                    restarting.append(model_id)
            new_text = render_config(proposed, holds)
            staged = self.writer.check(new_text) if new_text != old_text else None
            try:
                self._unload_for_restart(proposed, restarting)
                new_revision = self.store.save(proposed, expected_revision=current_revision)
            except BaseException:
                if staged is not None:
                    self.writer.discard(staged)
                raise
            self._write_holds(holds)
            if staged is not None:
                self.writer.commit(staged)
                self._mark_written()
            self._clear_restart()
            self._ensure_profiles(proposed)
            if restarting:
                self._spawn(self._reload_after_write, restarting, new_text)
            result = {"status": "saved", "revision": new_revision, "restarting": restarting, "held": sorted(holds)}
            if extra is not None:
                result.update(extra(current, proposed))
            return result

    @staticmethod
    def _affected(current: Mapping[str, Any], proposed: Mapping[str, Any], loaded: list[str],
                  before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
        """Loaded models whose switcher entry (before/after rendered without holds) or, for
        FreeToken, whose profile settings change."""
        affected = []
        for model_id in loaded:
            if model_id not in after:
                continue
            if before.get(model_id) != after[model_id]:
                affected.append(model_id)
                continue
            try:
                old_model, new_model = find_model(current, model_id), find_model(proposed, model_id)
            except KeyError:
                continue
            if new_model["engine"] == "freetoken" and freetoken_profile_settings(current, old_model) != \
                    freetoken_profile_settings(proposed, new_model):
                affected.append(model_id)
        return affected

    def _unload_for_restart(self, doc: Mapping[str, Any], model_ids: list[str]) -> None:
        """"Restart now" puts each model away first. If the switcher does not confirm, nothing
        is saved: before this, the save went on and the page said "restarting" while the old
        model kept running on the old settings (final review item 4)."""
        for row in self._named(doc, model_ids):
            if not self.switcher.unload(row["id"]):
                raise PanelError(503, "restart_failed", f"Couldn't put {row['name']} away to restart it, "
                                                        "so nothing was saved. Try again in a moment.")

    def _plan_rewrite(self, doc: Mapping[str, Any], state: tuple[list[str] | None, bool] | None = None
                      ) -> tuple[str, dict[str, str]]:
        """The config text and holds for writing doc without moving a loaded model's entry.

        Holds carry over while their model is loaded (all of them when the state is unknown or
        the switcher is down). Restore and the helper's start-up sync rewrite every entry, and
        P5 stops a loaded model whose entry changes (reproduced in the final review: QUASAR
        loaded on fp8, restore of an int8 backup stopped it). So a loaded model (any model when
        the state is unknown) whose new entry differs from the one in the file gets a hold on
        the file's entry; the hold watcher writes the new entry once it unloads. This also
        mends a crash between store.save and the holds write in _save. A loaded model whose
        entry cannot be found (the file was not written by the panel) is refused instead."""
        known, down = state if state is not None else self._ask()
        holds = self._live_holds(known, down)
        old_text = self.writer.current_text()
        current = extract_model_blocks(old_text)
        fresh = extract_model_blocks(render_config(doc, {}))
        foreign = bool(old_text) and not old_text.startswith(HEADER + "\n")
        at_risk = list(fresh) if known is None else [] if down else known
        unprotected = []
        for model_id in at_risk:
            if model_id in holds or model_id not in fresh:
                continue
            running_block = current.get(model_id)
            if running_block is None:
                if foreign:
                    unprotected.append(model_id)
            elif running_block != fresh[model_id]:
                holds[model_id] = running_block
        if unprotected:
            if known is None:
                raise PanelError(503, "switcher_unknown", REWRITE_UNKNOWN_MESSAGE)
            names = ", ".join(row["name"] for row in self._named(doc, unprotected))
            raise PanelError(409, "unload_first", f"{names} is loaded and the switcher file was not written by the "
                                                  "control panel, so this would restart it. Unload it first.")
        return render_config(doc, holds), holds

    def _rewrite(self, doc: Mapping[str, Any], state: tuple[list[str] | None, bool] | None = None) -> dict[str, str]:
        text, holds = self._plan_rewrite(doc, state)
        if self.writer.write(text):
            self._mark_written()
        self._write_holds(holds)
        return holds

    def _restart_done(self, model_ids: list[str], ok: bool, message: str) -> None:
        self.last_restart = {"models": model_ids, "ok": ok, "at": _now_iso(), "message": message}
        self._last_restart_at = self._clock()

    def _reload_after_write(self, model_ids: list[str], text: str) -> None:
        # Bounded wait (Task 7 review): if the switcher's rebuild fails for a reason
        # --check-config misses, its hash never changes; loading then would start the model on
        # the old settings, so give up with a plain message instead.
        expected = config_sha256(text)
        deadline = self._clock() + self.restart_wait_s
        while self.switcher.config_hash() != expected:
            if self._clock() >= deadline:
                self._restart_done(model_ids, False, RESTART_STALE_MESSAGE)
                return
            self._sleep(0.5)
        for model_id in model_ids:
            try:
                self.switcher.load(model_id)
            except SwitcherError as exc:
                self._restart_done(model_ids, False, f"Restarting {model_id} failed: {exc.message}")
                return
            except OSError:
                self._restart_done(model_ids, False, "The switcher is not running, so the model was not loaded again.")
                return
        self._restart_done(model_ids, True, "Restarted with the new settings.")

    # ---- registry status, import, restore ----
    def registry_status(self) -> dict[str, Any]:
        try:
            _, revision = self.store.load()
        except RegistryMissing:
            return {"status": "missing", "configPath": str(self.writer.path), "backups": self.store.backups()}
        except RegistryCorrupt as exc:
            return {"status": "corrupt", "message": exc.message, "backups": exc.backups}
        return {"status": "ok", "revision": revision, "backups": self.store.backups()}

    def import_live(self, when_loaded: str | None = None) -> dict[str, Any]:
        with self._lock:
            if self.store.exists():
                raise ImportRefused("The control panel already has its model list. Restore a backup to go back.")
            text = self.writer.current_text()
            if text is None:
                raise ImportRefused(f"No switcher config was found at {self.writer.path}.")
            try:
                # Spec section 1 (ruling F12): the helper's current boot file, which a profile
                # activation may have switched away from the default one.
                boot_settings = self._boot_file().load()
            except (BootParseError, OSError) as exc:
                raise ImportRefused(f"The helper's start-up file could not be read: {exc}") from exc
            registry, warnings = import_live(text, boot_settings, env=os.environ, profiles=self.profiles.list())
            # The same limits every save checks (final review item 8): an imported model that
            # breaks one would make every later save fail, so say which and copy nothing.
            # Checked against config.example.yaml + the real boot file: no errors, so today's
            # box imports as before.
            limits = self._model_limit_errors(registry)
            if limits:
                names = {m["id"]: m["name"] for m in registry["models"]}
                reasons = " ".join(f"{names.get(row['where'], row['where'])}: {row['message']}" for row in limits)
                raise ImportRefused(f"Nothing was copied, because these settings can't be used: {reasons} "
                                    "Change them in the helper's start-up file and try again.")
            known = self._loaded_or_unknown()
            if known is None:
                # The import rewrites every entry; if /running only timed out while a model was
                # loaded, P5 would restart it unasked (controller ruling, fix round 1).
                raise PanelError(503, "switcher_unknown", IMPORT_UNKNOWN_MESSAGE)
            loaded = known
            if loaded and when_loaded != "restart":
                # Every entry is rewritten, so P5 restarts whatever is loaded: only "restart" is offered.
                raise ChooseRestart(self._named(registry, loaded), next_time_allowed=False)
            new_text = render_config(registry, {})
            staged = self.writer.check(new_text)
            try:
                self._unload_for_restart(registry, loaded)
                revision = self.store.save(registry, expected_revision=None)
            except BaseException:
                self.writer.discard(staged)
                raise
            self.writer.backup_before_registry()
            self.writer.commit(staged)
            self._mark_written()
            self._clear_restart()
            self._write_holds({})
            self._ensure_profiles(registry)
            if loaded:
                self._spawn(self._reload_after_write, loaded, new_text)
            return {"status": "imported", "revision": revision, "warnings": warnings, "restarting": loaded}

    def restore(self, name: str) -> dict[str, Any]:
        """Check everything first (the backup, the holds plan, --check-config), then restore
        the registry and swap the file in: a refusal now changes nothing, where before the
        registry was already restored when the switcher refused the file (final review)."""
        with self._lock:
            doc = self.store.read_backup(name)
            text, holds = self._plan_rewrite(doc)
            staged = self.writer.check(text) if text != self.writer.current_text() else None
            try:
                revision = self.store.restore(name)
            except BaseException:
                if staged is not None:
                    self.writer.discard(staged)
                raise
            if staged is not None:
                self.writer.commit(staged)
                self._mark_written()
            self._write_holds(holds)
            return {"status": "restored", "revision": revision, "held": sorted(holds)}

    # ---- views ----
    def system_view(self) -> dict[str, Any]:
        doc, revision = self.store.load()
        values = {dial.name: doc["system"][dial.name] for dial in SYSTEM_DIALS}
        return {"kind": "system", "title": "System", "revision": revision,
                "dials": [dial.as_dict(values[dial.name]) for dial in SYSTEM_DIALS],
                "groups": [{"name": SYSTEM_GROUP, "plain": "System", "info": "Settings for the model switcher itself."}],
                "settings": values, "base": values, "baseFrom": {}}

    def engine_view(self, engine: str) -> dict[str, Any]:
        if engine not in ENGINES:
            raise KeyError(engine)
        doc, revision = self.store.load()
        values = _display(engine, engine_defaults(doc, engine))
        dials, groups = _engine_dials(engine, values, None)
        return {"kind": "engine", "engine": engine, "engineLabel": ENGINE_LABELS[engine],
                "title": f"{ENGINE_LABELS[engine]} defaults", "revision": revision, "dials": dials, "groups": groups,
                "settings": values, "base": values, "baseFrom": {},
                "models": [m["name"] for m in doc["models"] if m["engine"] == engine]}

    def model_view(self, model_id: str, preset: Any = ACTIVE) -> dict[str, Any]:
        doc, revision = self.store.load()
        model = find_model(doc, model_id)
        presets = model.get("presets") or {}
        chosen = model.get("activePreset") if preset is ACTIVE else (preset or None)
        if chosen is not None and chosen not in presets:
            raise KeyError(chosen)
        engine = model["engine"]
        base = base_settings(doc, model, chosen)
        from_preset = preset_values(model, chosen)
        effective = {**base, **(model.get("overrides") or {})}
        info = read_model(expand(model["artifact"])) if engine == "freetoken" else None
        shown = _display(engine, effective)
        dials, groups = _engine_dials(engine, shown, model.get("runtime"), info)
        identity = {"model.name": model["name"], "model.aliases": ", ".join(model.get("aliases") or []),
                    "model.ramNeedGB": model["ramNeedGB"],
                    "model.idleMinutes": -1 if model.get("idleMinutes") is None else model["idleMinutes"]}
        running = self.switcher.running()
        up = running is not None and not is_down(running)
        return {
            "kind": "model", "id": model_id, "name": model["name"], "title": model["name"], "revision": revision,
            "engine": engine, "engineLabel": ENGINE_LABELS[engine], "runtime": model.get("runtime"),
            "runtimeLabel": ninfer_dials.RUNTIME_LABELS.get(model.get("runtime"), ""),
            "state": running.get(model_id, "stopped") if up else "unknown",
            "presets": sorted(presets), "activePreset": chosen, "savedPreset": model.get("activePreset"),
            "dials": [dial.as_dict(identity[dial.name]) for dial in IDENTITY_DIALS] + dials,
            "groups": [{"name": IDENTITY_GROUP, "plain": IDENTITY_GROUP, "info": "Its name, other names and memory."}] + groups,
            "settings": {**shown, **identity}, "base": _display(engine, base),
            "baseFrom": {name: ("preset" if name in from_preset else "default") for name in _display(engine, base)},
            "model": info.as_dict() if info is not None else None,
        }

    # ---- saves ----
    def save_system(self, system: Mapping[str, Any], revision: str | None, when_loaded: str | None) -> dict[str, Any]:
        def mutate(doc: dict) -> dict:
            current = doc["system"]
            try:
                current["floorGB"] = _number(system.get("floorGB", current["floorGB"]))
                current["waitSeconds"] = _whole(system.get("waitSeconds", current["waitSeconds"]))
                current["defaultIdleMinutes"] = _whole(system.get("defaultIdleMinutes", current["defaultIdleMinutes"]))
            except (TypeError, ValueError):
                raise RegistryValidationError([{"field": "system", "message": "Use numbers for the cushion; whole numbers for the wait and idle time."}])
            current["latestWins"] = system.get("latestWins", current["latestWins"])
            return doc
        return self._save(mutate, revision, when_loaded)

    def save_engine_defaults(self, engine: str, settings: Mapping[str, Any], revision: str | None,
                             when_loaded: str | None) -> dict[str, Any]:
        if engine not in ENGINES:
            raise KeyError(engine)

        def mutate(doc: dict) -> dict:
            values = canonical_engine_settings(engine, settings)
            if engine == "ninfer":
                doc["engines"]["ninfer"]["defaults"] = differences(values, ninfer_dials.BUILTINS)
            else:
                doc["engines"]["freetoken"]["defaults"] = {**doc["engines"]["freetoken"]["defaults"], **values}
            return doc

        def inherits(before: dict, after: dict) -> dict:
            old, new = engine_defaults(before, engine), engine_defaults(after, engine)
            changed = [name for name in set(old) | set(new) if not same(old.get(name), new.get(name))]
            names = [m["name"] for m in after["models"] if m["engine"] == engine and any(
                name not in preset_values(m) and name not in (m.get("overrides") or {}) for name in changed)]
            return {"inherits": names}

        return self._save(mutate, revision, when_loaded, inherits)

    def save_model(self, model_id: str, settings: Mapping[str, Any], identity: Mapping[str, Any],
                   active_preset: str | None, revision: str | None, when_loaded: str | None) -> dict[str, Any]:
        def mutate(doc: dict) -> dict:
            model = find_model(doc, model_id)
            chosen = active_preset or None
            if chosen is not None and chosen not in (model.get("presets") or {}):
                raise RegistryValidationError([{"field": "activePreset", "message": "That preset no longer exists."}])
            model["activePreset"] = chosen
            values = canonical_engine_settings(model["engine"], {k: v for k, v in settings.items() if k != "ModelPath"})
            model["overrides"] = differences(values, base_settings(doc, model))
            errors = []
            if "name" in identity:
                model["name"] = str(identity["name"]).strip()
            if "aliases" in identity:
                model["aliases"] = _split_aliases(identity["aliases"])
            if "ramNeedGB" in identity:
                try:
                    model["ramNeedGB"] = _number(identity["ramNeedGB"])
                except (TypeError, ValueError):
                    errors.append({"field": "model.ramNeedGB", "message": "The PC memory it needs must be a number of GB."})
            if "idleMinutes" in identity:
                raw = identity["idleMinutes"]
                if raw in (None, "", -1, "-1"):
                    model["idleMinutes"] = None
                else:
                    try:
                        model["idleMinutes"] = _whole(raw)
                    except (TypeError, ValueError):
                        errors.append({"field": "model.idleMinutes", "message": "Unload when idle must be whole minutes."})
            if errors:
                raise RegistryValidationError(errors)
            return doc
        return self._save(mutate, revision, when_loaded)

    def preset_action(self, model_id: str, action: str, name: str, new_name: str | None, revision: str | None,
                      when_loaded: str | None) -> dict[str, Any]:
        if action not in ("add", "rename", "delete"):
            raise KeyError(action)
        name = str(name or "").strip()

        def mutate(doc: dict) -> dict:
            model = find_model(doc, model_id)
            presets = model.setdefault("presets", {})
            if action == "add":
                if name in presets:
                    raise RegistryValidationError([{"field": "preset", "message": f"A preset called {name} already exists."}])
                effective = effective_settings(doc, model)
                effective.pop("ModelPath", None)
                presets[name] = differences(effective, engine_defaults(doc, model["engine"]))
            elif action == "rename":
                target = str(new_name or "").strip()
                if name not in presets:
                    raise KeyError(name)
                if target in presets and target != name:
                    raise RegistryValidationError([{"field": "preset", "message": f"A preset called {target} already exists."}])
                presets[target] = presets.pop(name)
                if model.get("activePreset") == name:
                    model["activePreset"] = target
            else:
                if name not in presets:
                    raise KeyError(name)
                presets.pop(name)
                if model.get("activePreset") == name:
                    model["activePreset"] = None
            return doc
        return self._save(mutate, revision, when_loaded)

    # ---- fit, now, models, load ----
    def fit(self, model_id: str, settings: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
        doc, _ = self.store.load()
        model = find_model(doc, model_id)
        engine = model["engine"]
        values = canonical_engine_settings(engine, {k: v for k, v in settings.items() if k != "ModelPath"})
        draft = {**effective_settings(doc, model), **values}
        card = self._card_probe()
        total = card["totalBytes"] if card else None
        out: dict[str, Any] = {"engine": engine, "verdict": "unknown", "needBytes": None, "cardTotalBytes": total,
                               "components": [], "notes": [], "suggestion": None, "message": ""}
        if engine == "ninfer":
            path = expand(model["artifact"])
            try:
                size = self._artifact_size(path)
            except OSError:
                out["message"] = f"The model file was not found at {path}."
            else:
                # Fix round 1 ruling: the startup check takes the live card reading as the desktop
                # only when the switcher answers and has nothing loaded (then the card holds only
                # the desktop and other programs); otherwise the fixed 2.6 GiB is used.
                desktop = None
                if card:
                    loaded, down = self._ask()
                    if loaded is not None and not down and not loaded:
                        desktop = card.get("usedBytes")
                estimate = ninfer_fit.estimate(draft, size, card_total_bytes=total, desktop_bytes=desktop)
                out.update(needBytes=estimate["needBytes"], components=estimate["components"], notes=estimate["notes"])
                # Two checks: the card memory in use once up (needBytes) and NInfer's own startup
                # refusal on its up-front runtime reservation (fit round 2, 2026-09-25).
                out.update(runtimeReservationBytes=estimate["runtimeReservationBytes"],
                           runtimeRoomBytes=estimate["runtimeRoomBytes"])
                out["verdict"] = ninfer_fit.worst_verdict(ninfer_fit.verdict(estimate["needBytes"], total),
                                                          estimate["startupVerdict"])
                if estimate["startupVerdict"] in ("wont_fit", "tight"):
                    out["message"] = estimate["startupMessage"]
                if not total:
                    out["message"] = "The graphics card could not be read."
        else:
            try:
                result = self.estimate_service.estimate_settings(draft, boot_file=self._boot_file())
            except EstimateUnavailable as exc:
                out["message"] = exc.message
            except SettingsValidationError as exc:
                raise RegistryValidationError(exc.errors) from exc
            else:
                vram = result["resources"]["vram"]
                need, card_total = int(vram["empty"]["need_bytes"]), int(vram["total_bytes"]) or total
                out.update(needBytes=need, cardTotalBytes=card_total)
                out["verdict"] = "wont_fit" if not result.get("fits_empty") else ninfer_fit.verdict(need, card_total)
                # Ruling F11: the verdict is for an empty card, so leave out the planner's "now"
                # rows (what happens to be on the card today) and keep "empty" and "both".
                out["components"] = [{"label": c.get("name"), "bytes": c.get("bytes")}
                                     for c in result.get("components") or []
                                     if c.get("resource") == "vram" and c.get("scenario") in ("empty", "both")]
                suggestion = result.get("suggestion")
                if isinstance(suggestion, dict) and isinstance(suggestion.get("settings"), dict) and suggestion.get("fits_empty"):
                    out["suggestion"] = {"settings": suggestion["settings"], "changes": suggestion.get("changes") or []}
        try:
            need_gb = _number(identity.get("ramNeedGB", model["ramNeedGB"]))
        except (TypeError, ValueError):
            need_gb = model["ramNeedGB"]
        free = self._windows_free()
        out["ram"] = {"needGB": need_gb, "cushionGB": doc["system"]["floorGB"],
                      "windowsFreeGB": None if free is None else round(free / GIB, 1),
                      "loadedNow": model_id in self._loaded()}
        return out

    def _switcher_stale(self) -> bool:
        """True when the switcher answers with a config hash that is not the file's (P5): it
        is still on older settings, for example after a rebuild it could not do. Not flagged
        within STALE_GRACE_S of the panel's own write (final review item 5)."""
        if self._last_write is not None and self._clock() - self._last_write < STALE_GRACE_S:
            return False
        live = self.switcher.config_hash()
        text = self.writer.current_text()
        return live is not None and text is not None and live != config_sha256(text)

    def now(self) -> dict[str, Any]:
        running = self.switcher.running()
        up = running is not None and not is_down(running)
        try:
            doc, _ = self.store.load()
            names, floor = {m["id"]: m["name"] for m in doc["models"]}, doc["system"]["floorGB"]
        except RegistryError:
            names, floor = {}, None
        rows = [{"id": m, "name": names.get(m, m), "state": s} for m, s in sorted((running or {}).items())]
        last = self.last_restart
        if last is not None and self._last_restart_at is not None and self._clock() - self._last_restart_at >= RESTART_SHOWN_S:
            last = None
        return {"switcher": {"up": up, "running": rows, "stale": up and self._switcher_stale()},
                "card": self._card_probe(), "windowsFreeBytes": self._windows_free(), "cushionGB": floor,
                "held": sorted(self._read_holds()), "lastRestart": last}

    def models(self) -> dict[str, Any]:
        doc, revision = self.store.load()
        running = self.switcher.running()
        up = running is not None and not is_down(running)
        holds = self._read_holds()
        rows = []
        for model in doc["models"]:
            rows.append({
                "id": model["id"], "name": model["name"], "engine": model["engine"],
                "engineLabel": ENGINE_LABELS[model["engine"]], "runtime": model.get("runtime"),
                "runtimeLabel": ninfer_dials.RUNTIME_LABELS.get(model.get("runtime"), ""),
                "activePreset": model.get("activePreset"), "presets": sorted(model.get("presets") or {}),
                "ramNeedGB": model["ramNeedGB"], "idleMinutes": idle_minutes(doc, model),
                "idleFromSystem": model.get("idleMinutes") is None,
                "state": running.get(model["id"], "stopped") if up else "unknown",
                "held": model["id"] in holds,
            })
        return {"revision": revision, "switcherUp": up, "models": rows}

    def load(self, model_id: str) -> dict[str, Any]:
        doc, _ = self.store.load()
        find_model(doc, model_id)
        try:
            self.switcher.load(model_id)
        except SwitcherError as exc:
            raise PanelError(exc.status, exc.code, exc.message) from exc
        except OSError as exc:
            raise PanelError(503, "switcher_down", "The model switcher is not running.") from exc
        self._clear_restart()
        return {"id": model_id, "state": "ready"}

    def unload(self, model_id: str) -> dict[str, Any]:
        doc, _ = self.store.load()
        find_model(doc, model_id)
        if not self.switcher.unload(model_id):
            raise PanelError(503, "switcher_down", "The model switcher is not running.")
        self._clear_restart()
        return {"id": model_id, "state": "stopped"}

    def effective(self, model_id: str) -> dict[str, Any]:
        doc, _ = self.store.load()
        model = find_model(doc, model_id)
        if model["engine"] != "freetoken":
            raise PanelError(409, "not_freetoken", f"{model_id} is not a FreeToken model.")
        return {"id": model_id, "name": model["name"], "settings": freetoken_profile_settings(doc, model)}

    # ---- holds and start-up ----
    def release_finished_holds(self) -> list[str]:
        with self._lock:
            holds = self._read_holds()
            if not holds:
                return []
            known, down = self._ask()
            if known is None or down:
                return []  # switcher down or slow: skip this tick, never release on a blip
            released = [model_id for model_id in holds if model_id not in known]
            if not released:
                return []
            try:
                doc, _ = self.store.load()
            except RegistryError:
                return []
            self._rewrite(doc, (known, down))
            return sorted(released)

    def sync_config(self) -> None:
        """At helper start: make the switcher file match the registry (and current holds)."""
        try:
            with self._lock:
                doc, _ = self.store.load()
                self._rewrite(doc)
        except (RegistryError, SwitcherRefused, OSError, PanelError):
            pass  # the page shows the registry problem; the old file stays live

    def start_hold_watcher(self, interval: float = 5.0) -> None:
        if self._watcher is not None:
            return

        def loop() -> None:
            while not self._stop.wait(interval):
                try:
                    self.release_finished_holds()
                except Exception:  # noqa: BLE001 - the next tick tries again
                    pass

        self._watcher = threading.Thread(target=loop, name="panel-hold-watcher", daemon=True)
        self._watcher.start()

    def stop_hold_watcher(self) -> None:
        self._stop.set()


# ---- routes ----
class _SaveBody(BaseModel):
    revision: str | None = None
    whenLoaded: str | None = None


class SystemBody(_SaveBody):
    system: dict[str, Any] = Field(default_factory=dict)


class EngineBody(_SaveBody):
    settings: dict[str, Any] = Field(default_factory=dict)


class ModelBody(_SaveBody):
    settings: dict[str, Any] = Field(default_factory=dict)
    identity: dict[str, Any] = Field(default_factory=dict)
    activePreset: str | None = None


class PresetBody(_SaveBody):
    name: str = ""
    newName: str | None = None


class FitBody(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)
    identity: dict[str, Any] = Field(default_factory=dict)


class ImportBody(BaseModel):
    whenLoaded: str | None = None


class RestoreBody(BaseModel):
    backup: str


def create_panel_router(service: PanelService) -> APIRouter:
    router = APIRouter(prefix="/api/panel")

    async def call(fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return await run_in_threadpool(fn, *args)
        except RegistryMissing:
            return JSONResponse(status_code=409, content={"code": "registry_missing",
                                "message": "The control panel has no model list yet. Copy today's settings first."})
        except RegistryCorrupt as exc:
            return JSONResponse(status_code=409, content={"code": "registry_corrupt", "message": exc.message, "backups": exc.backups})
        except StaleRevision:
            return JSONResponse(status_code=409, content={"code": "stale_revision",
                                "message": "These settings were changed somewhere else. Reload and make the change again."})
        except ChooseRestart as exc:
            return JSONResponse(status_code=409, content={"code": "choose_restart", "affected": exc.affected,
                                                          "nextTimeAllowed": exc.next_time_allowed})
        except RegistryValidationError as exc:
            return JSONResponse(status_code=422, content={"detail": exc.errors})
        except SwitcherRefused as exc:
            return JSONResponse(status_code=422, content={"code": "switcher_refused",
                                "message": f"The switcher refused these settings: {exc}"})
        except ImportRefused as exc:
            return JSONResponse(status_code=409, content={"code": "import_refused", "message": str(exc)})
        except PanelError as exc:
            return JSONResponse(status_code=exc.status, content=exc.payload)
        except KeyError as exc:
            return JSONResponse(status_code=404, content={"detail": f"not found: {exc.args[0] if exc.args else ''}"})

    @router.get("/registry")
    async def registry_status():
        return await call(service.registry_status)

    @router.post("/import")
    async def import_route(body: ImportBody | None = None):
        return await call(service.import_live, (body or ImportBody()).whenLoaded)

    @router.post("/registry/restore")
    async def restore(body: RestoreBody):
        return await call(service.restore, body.backup)

    @router.get("/now")
    async def now():
        return await call(service.now)

    @router.get("/models")
    async def models():
        return await call(service.models)

    @router.get("/views/system")
    async def system_view():
        return await call(service.system_view)

    @router.get("/views/engine/{engine}")
    async def engine_view(engine: str):
        return await call(service.engine_view, engine)

    @router.get("/views/model/{model_id}")
    async def model_view(model_id: str, preset: str | None = Query(default=None)):
        return await call(service.model_view, model_id, ACTIVE if preset is None else preset)

    @router.put("/system")
    async def save_system(body: SystemBody):
        return await call(service.save_system, body.system, body.revision, body.whenLoaded)

    @router.put("/engines/{engine}/defaults")
    async def save_defaults(engine: str, body: EngineBody):
        return await call(service.save_engine_defaults, engine, body.settings, body.revision, body.whenLoaded)

    @router.put("/models/{model_id}")
    async def save_model(model_id: str, body: ModelBody):
        return await call(service.save_model, model_id, body.settings, body.identity, body.activePreset,
                          body.revision, body.whenLoaded)

    @router.post("/models/{model_id}/presets/{action}")
    async def preset(model_id: str, action: str, body: PresetBody):
        return await call(service.preset_action, model_id, action, body.name, body.newName, body.revision, body.whenLoaded)

    @router.post("/models/{model_id}/fit")
    async def fit(model_id: str, body: FitBody):
        return await call(service.fit, model_id, body.settings, body.identity)

    @router.post("/models/{model_id}/load")
    async def load(model_id: str):
        return await call(service.load, model_id)

    @router.post("/models/{model_id}/unload")
    async def unload(model_id: str):
        return await call(service.unload, model_id)

    @router.get("/models/{model_id}/effective")
    async def effective(model_id: str):
        return await call(service.effective, model_id)

    return router


__all__ = ["ChooseRestart", "IDENTITY_DIALS", "PanelError", "PanelService", "SYSTEM_DIALS", "create_panel_router"]

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

Stage B adds models (a file or folder on the PC, or a Hugging Face download checked against
the repo's published checksums) and removes them (unload first, then save, then the model's
profile, then its files when asked, then Pi). Detection runs again on the server at Save.
Removing needs a known switcher state: P5 stops a removed model that is still loaded, so an
unknown state could otherwise pull a model out from under a chat.
"""

from __future__ import annotations

import collections
import concurrent.futures
import copy
import datetime as _dt
import errno
import json
import logging
import math
import os
import shutil
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
from .dials import (DIAL_BY_NAME, DIALS, GROUP_INFO, MODEL_AWARE_DIALS, Dial, adapt_dial, dial_value_for_display,
                    validate_settings)
from .download import AddUnsupported, DownloadConflict, InvalidRepository
from .memory_fit import EstimateUnavailable, SettingsValidationError
from .model_detect import PART_RE, OwnershipUnknown, detect, model_files, referenced_files
from .model_info import read_model
from .pi_sync import PiSync
from .profiles_manager import ProfileError, ProfileValidationError
from .registry import (
    ACTIVE, ENGINE_LABELS, ENGINES, ID_RULE, MODEL_ID_RE, NAME_MAX, RegistryCorrupt, RegistryError, RegistryMissing,
    RegistryValidationError, StaleRevision, base_settings, canonical_engine_settings, canonicalize, differences,
    effective_settings, engine_defaults, expand, find_model, freetoken_profile_settings, idle_minutes, preset_values,
    same, validate_registry,
)
from .registry_import import ImportRefused, import_live
from .swap_config import HEADER, SwitcherRefused, config_sha256, extract_model_blocks, profile_id, render_config
from .switcher import LOADED_STATES, SwitcherError, is_down

logger = logging.getLogger("freetoken.daemon.settings.panel")

GIB = 1024 ** 3
PROFILE_NOTE = "Control panel settings for this model"
# A download's staging folder (download.py places files as .incoming-<id> beside the target and
# renames at the end): a half-written file there must not be added.
STAGING_PREFIX = ".incoming-"
STAGING_MESSAGE = ("That is inside a download's staging folder, so it may be half-written. Wait for the "
                   "download to finish, then add the finished file or folder.")
NOT_FOUND_MESSAGE = "That model is no longer in the list. Reload the page."
# A 404 from any panel route (a model id, a preset, a settings page or a download that is gone)
# says so in plain words; it used to be {"detail": "not found: <key>"} (stage A deferred minor).
MISSING_MESSAGE = "That model or preset is no longer in the list. Reload the page."
MISSING_DOWNLOAD_MESSAGE = "The settings page no longer knows that download. It may have restarted; start it again."
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
# The switcher file was replaced by something other than the panel (an old copy put back, a hand
# edit) after the restart's write: the switcher is on that file, not the new settings (PR #20 review).
RESTART_SUPERSEDED_MESSAGE = ("The switcher file was changed outside the control panel before the restart finished, "
                              "so the new settings are not in use and the model was not loaded again. "
                              "Open its settings and save again.")
ARTIFACT_CHANGED_MESSAGE = ("That model's files changed since its settings were opened (it may have been removed and "
                            "added again), so nothing was removed. Reload the page and try again.")
TEST_RUNNING_MESSAGE = "A test is running on the Test tab. Wait for it to finish, or stop it there."
RESTART_PENDING_MESSAGE = ("The control panel is restarting a model with its new settings. "
                           "Try again when it has finished.")
PANEL_BUSY_MESSAGE = "The control panel is loading or unloading a model right now. Try again when it has finished."
TEST_MARKER = "playground-test.json"
# A model an earlier test left on test settings (its put-away was refused because an app was
# using it). Kept on disk until the model is seen unloaded, so a helper restart in between
# still knows it runs test settings (PR #18 review round 2).
TEST_LEFTOVER = "playground-leftover.json"
RECOVERING_MESSAGE = ("The settings page is finishing an earlier test after a restart. "
                      "Try again in a moment.")

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


def _and_list(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


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


def default_add_roots() -> dict[str, Path]:
    """Spec section 7: ~/models/<name> for FreeToken folders, ~/ninfer-work/models/ for NInfer."""
    home = Path.home()
    return {"folder": Path(os.environ.get("FREETOKEN_MODELS_DIR") or home / "models"),
            "ninfer": Path(os.environ.get("FREETOKEN_NINFER_MODELS_DIR") or home / "ninfer-work" / "models")}


def _in_staging(path: str) -> bool:
    text = str(path or "").strip().strip('"').strip("'")
    if not text:
        return False
    return any(part.startswith(STAGING_PREFIX) for part in Path(os.path.abspath(os.path.expanduser(text))).parts)


def _is_network_error(exc: BaseException) -> bool:
    """Hub, HTTP and socket errors, told apart from a local OSError without importing the
    Hub: ConnectionError and TimeoutError are OSError subclasses, so they go first."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    for klass in type(exc).__mro__:
        module = klass.__module__ or ""
        if module.split(".")[0] in ("huggingface_hub", "requests", "urllib3", "urllib", "http", "socket", "ssl"):
            return True
    return False


def _hub_reason(exc: BaseException) -> str:
    """Plain words for a Hub failure. The exception's own text carries the request URL, the
    Hub's request id and its HTML, none of which belong on the page (review item 11): the
    class name and HTTP status are enough to say what happened, and the detail goes to the log."""
    names = {klass.__name__ for klass in type(exc).__mro__}
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if "GatedRepoError" in names or status in (401, 403):
        return "That repo is gated or private on Hugging Face, so it cannot be read from here."
    if "RepositoryNotFoundError" in names or "RevisionNotFoundError" in names or status == 404:
        return "Hugging Face has no model repo at that link. Check the owner and name."
    if "EntryNotFoundError" in names:
        return "A file the repo lists is missing from Hugging Face."
    if status == 429:
        return "Hugging Face is asking this PC to slow down. Try again in a minute."
    if isinstance(status, int) and status >= 500:
        return "Hugging Face is having trouble right now. Try again later."
    if isinstance(exc, TimeoutError) or "Timeout" in "".join(names):
        return "Hugging Face did not answer in time. Try again."
    return "Couldn't reach Hugging Face. Check the internet connection and try again."


def _after_save(step: str, fn: Callable[[], dict[str, Any]], fallback: Mapping[str, Any],
                words: str) -> dict[str, Any]:
    """Run one step that follows a completed add/remove. The list change already happened,
    so a failure here is reported in the response, never raised into a 500 (review round)."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - reported, the add/remove itself succeeded
        logger.exception("%s failed after the list was saved", step)
        return {**fallback, "message": f"{words} ({exc.__class__.__name__}: {exc})."}


def _home_path(path: str) -> str:
    home = str(Path.home()).rstrip("/")
    return "~/" + path[len(home) + 1:] if path.startswith(home + "/") else path


def _fit_to_model(doc: Mapping[str, Any], entry: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """FreeToken defaults a new model cannot use become its own values at its limit.

    Without this the add fails, and a registry holding such a model would refuse every later
    save (_save checks every FreeToken model's limits). Checked 2026-09-25 against five() +
    a Llama config with max_position_embeddings 8192: only ContextTokens (262,144 by default)
    is over, and 8192 clears validate_settings; a 64-expert, 24-layer Qwen3-MoE also needs
    MoECacheSize 5332 -> 1536. Dials stored in another form (GpuOwnedLayers "auto:{n}") are
    left for Jay: the save then names them."""
    info = read_model(expand(entry["artifact"]))
    effective = effective_settings(doc, entry)
    effective.pop("ModelPath", None)
    overrides: dict[str, Any] = {}
    notes: list[str] = []
    for error in validate_settings(effective, info):
        dial = DIAL_BY_NAME.get(error["field"])
        if dial is None or dial.stored_as or dial.name not in MODEL_AWARE_DIALS or dial.name in overrides:
            continue
        value, bounds = effective.get(dial.name), adapt_dial(dial, info)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if bounds.get("max") is not None and value > bounds["max"]:
            overrides[dial.name], word = bounds["max"], "most"
        elif bounds.get("min") is not None and value < bounds["min"]:
            overrides[dial.name], word = bounds["min"], "least"
        else:
            continue
        notes.append(f"{dial.plain or dial.name} set to {overrides[dial.name]:,}, the {word} this model allows.")
    return overrides, notes


class PanelService:
    def __init__(self, *, store, writer, switcher, profiles, boot_file: Callable[[], BootFile],
                 estimate_service, card_probe=None, windows_free_probe=None,
                 holds_path=None, artifact_size=None, spawn=None, clock=time.monotonic, sleep=time.sleep,
                 restart_wait_s: float = 30.0, downloads=None, pi=None, add_roots=None, profile_deleted=None,
                 freetoken_state: Callable[[], Any] | None = None,
                 freetoken_control: Callable[[str], dict[str, Any]] | None = None) -> None:
        self.store, self.writer, self.switcher, self.profiles = store, writer, switcher, profiles
        self._boot_file, self.estimate_service = boot_file, estimate_service
        self._card_probe = card_probe or _default_card_probe
        self._windows_free = windows_free_probe or _default_windows_free
        self.holds_path = Path(holds_path) if holds_path else store.path.with_name("held-models.json")
        self._artifact_size = artifact_size or os.path.getsize
        self._spawn = spawn or _default_spawn
        self._clock, self._sleep, self.restart_wait_s = clock, sleep, restart_wait_s
        self.downloads = downloads
        self.pi = pi if pi is not None else PiSync(enabled=False)
        self.add_roots = {k: Path(v) for k, v in (add_roots or default_add_roots()).items()}
        # Called with profiles.delete()'s result so the app can move the helper back to its
        # default boot file when the removed model's profile was the active one.
        self.profile_deleted = profile_deleted
        # Sleep (docs/superpowers/specs/2026-09-25-freetoken-sleep-design.md section 2.5): the
        # FreeToken server's public state word ("sleeping" while the card is given back) and its
        # /v1/sleep | /v1/wake, both through the helper's ProcessManager. Either may be None:
        # the route tests hand in a manager that predates sleep, and then every row reads awake.
        self._freetoken_state = freetoken_state
        self._freetoken_control = freetoken_control
        self._lock = threading.RLock()
        # Pi changes run after the panel lock is released (slow /mnt/c), but in the order the
        # registry committed them: queued under the panel lock, run one at a time under
        # _pi_lock by whichever caller gets there first (review, PR #17).
        self._pi_queue: collections.deque[tuple[Callable[[], dict[str, Any]], concurrent.futures.Future]] = collections.deque()
        self._pi_lock = threading.Lock()
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None
        # The last failed-or-done restart and when it ended, as ONE value: now() reads it without
        # the panel lock (a save holds that lock through an unload of up to 240 s, and the
        # Right-now strip must not wait for it), and two separate fields could be read half
        # old, half new (stage A deferred minor).
        self._restart: tuple[dict[str, Any], float] | None = None
        self._last_write: float | None = None
        # Every switcher file the panel wrote, by hash, with a generation that counts up per
        # write: a restart accepts the switcher on a LATER panel write, never on an older text
        # copied back by hand (PR #20 review: both hashes old then read as "applied").
        self._write_gen = 0
        self._written: dict[str, int] = {}
        # Test tab (part 3). The overlay is what the switcher and FreeToken's adapter see during
        # a test: the test model on one preset alone. It lives in memory and is never written to
        # the registry; the marker lets a restarted helper find a model left on test settings.
        self.test_running = False
        # begin_test refuses while a save's restart (_reload_after_write) or a panel load/unload
        # is still running: the test would put that model away or load over it.
        self._restarts_pending = 0
        self._actions = 0
        self.test_settings: dict[str, Any] | None = None
        self.test_marker_path = self.holds_path.with_name(TEST_MARKER)
        self.test_leftover_path = self.holds_path.with_name(TEST_LEFTOVER)
        self._test_leftover: dict[str, Any] | None = self._read_test_leftover()
        # Helper start-up recovery (server.start_panel) holds the test guard: a test or a panel
        # action started mid-recovery would race its put-away and its switcher-file rewrite.
        self.recovering = False

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

    def _mark_written(self, text: str) -> None:
        self._last_write = self._clock()
        self._write_gen += 1
        self._written[config_sha256(text)] = self._write_gen

    def _clear_restart(self) -> None:
        self._restart = None

    @property
    def last_restart(self) -> dict[str, Any] | None:
        restart = self._restart
        return None if restart is None else restart[0]

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

    def _queue_pi(self, step: Callable[[], dict[str, Any]]) -> concurrent.futures.Future:
        """Call with the panel lock held, right after the registry save, so the queue holds
        Pi changes in commit order."""
        done: concurrent.futures.Future = concurrent.futures.Future()
        self._pi_queue.append((step, done))
        return done

    def _run_pi(self, done: concurrent.futures.Future) -> dict[str, Any]:
        """Call without the panel lock: runs every queued Pi change (this caller's and any
        committed before it) in order, then answers this caller's."""
        with self._pi_lock:
            while self._pi_queue:
                step, future = self._pi_queue.popleft()
                try:
                    future.set_result(step())
                except BaseException as exc:  # noqa: BLE001 - handed to its own caller
                    future.set_exception(exc)
        return done.result()

    # ---- the one save path ----
    def _save(self, mutate: Callable[[dict], dict], revision: str | None, when_loaded: str | None,
              extra: Callable[[dict, dict], dict] | None = None,
              before_commit: Callable[[list[str] | None], None] | None = None) -> dict[str, Any]:
        """``before_commit`` runs after every check (registry, model limits, --check-config)
        and right before the registry is written: remove_model puts a loaded model away there,
        so a remove that the checks refuse never unloads it first (review round). It is given
        this save's own switcher answer (loaded ids, or None when unknown), the last one taken
        before the write, so it decides on that and not on an earlier look (review, PR #17)."""
        if when_loaded not in (None, "restart", "next-time"):
            raise RegistryValidationError([{"field": "whenLoaded", "message": "Choose restart or next-time."}])
        with self._lock:
            self._guard_test()
            current, current_revision = self.store.load()
            if revision != current_revision:
                raise StaleRevision("The model list was changed somewhere else.")
            proposed = canonicalize(mutate(copy.deepcopy(current)))
            errors = validate_registry(proposed) or self._model_limit_errors(proposed)
            if errors:
                # "where" is a model id; the page shows it for a row it cannot place beside a
                # dial (another model's limit), so give the model's name (stage A deferred minor).
                names = {m.get("id"): m.get("name") for m in proposed.get("models") or [] if isinstance(m, dict)}
                raise RegistryValidationError([
                    {**error, "where": names.get(error["where"]) or error["where"]} if error.get("where") else error
                    for error in errors])
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
            # current as stored may not be canonical (a restored backup, a hand edit). The
            # entries compared are what the switcher actually gets: current rendered as stored
            # against proposed rendered. Canonicalising both sides hid a real change
            # ("max-concurrency": "04" renders --max-concurrency 04, canonical 4), so llama-swap
            # restarted the model with no question asked (PR #20 review). The profile comparison
            # in _affected stays canonical: freetoken.sh pushes canonical settings, so a spelling
            # like GpuOwnedLayers "auto" vs "auto:6" changes nothing that runs (stage A minor).
            stored = canonicalize(current)
            try:
                before = extract_model_blocks(render_config(current, {}))
            except Exception:  # noqa: BLE001 - a stored value the renderer cannot take as written
                before = extract_model_blocks(render_config(stored, {}))
            after = extract_model_blocks(render_config(proposed, {}))
            # A removed model's hold goes with it (Stage B): its entry is no longer in the list.
            holds = {model_id: block for model_id, block in holds.items() if model_id in after}
            if known is None:
                # A model new to the list has no entry in the switcher yet, so it cannot be
                # loaded: adding one while the state is unknown is safe (Stage B).
                moved = [m for m in after if m in before and m not in holds and before[m] != after[m]]
                if moved:
                    raise PanelError(503, "switcher_unknown", _unknown_save_message(
                        [row["name"] for row in self._named(proposed, moved)]))
            affected = self._affected(stored, proposed, loaded, before, after)
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
                if staged is not None:
                    # Before anything changes (PR #20 review): a copy that cannot be made used
                    # to fail inside commit(), after the registry was already saved.
                    self.writer.backup_hand_written()
                if before_commit is not None:
                    before_commit(known)
                self._unload_for_restart(proposed, restarting)
                new_revision = self.store.save(proposed, expected_revision=current_revision)
            except BaseException:
                if staged is not None:
                    self.writer.discard(staged)
                raise
            self._write_holds(holds)
            if staged is not None:
                self.writer.commit(staged)
                self._mark_written(new_text)
            self._clear_restart()
            self._ensure_profiles(proposed)
            if restarting:
                self._spawn_reload(restarting, new_text)
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
        model kept running on the old settings (final review item 4).

        An exception from the unload (a dropped connection, a half answer) is the same "not
        confirmed", in plain words, not a bare 500 (stage A deferred minor). With several
        models, the ones already put away are named: nothing was saved, but they are no
        longer loaded."""
        done: list[str] = []
        for row in self._named(doc, model_ids):
            try:
                ok = self.switcher.unload(row["id"])
            except Exception:  # noqa: BLE001 - reported below in plain words; detail to the log
                logger.exception("unloading %s before a save failed", row["id"])
                ok = False
            if not ok:
                message = f"Couldn't put {row['name']} away to restart it, so nothing was saved."
                if done:
                    names = _and_list(done)
                    message += (f" {names} {'was' if len(done) == 1 else 'were'} already put away and "
                                f"{'is' if len(done) == 1 else 'are'} not loaded now; load "
                                f"{'it' if len(done) == 1 else 'them'} again from the Models list.")
                raise PanelError(503, "restart_failed", message + " Try again in a moment.")
            done.append(row["name"])

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
        doc = self._for_switcher(doc)
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
            self._mark_written(text)
        self._write_holds(holds)
        return holds

    def _restart_done(self, model_ids: list[str], ok: bool, message: str) -> None:
        self._restart = ({"models": model_ids, "ok": ok, "at": _now_iso(), "message": message}, self._clock())

    def _spawn_reload(self, model_ids: list[str], text: str) -> None:
        """Start _reload_after_write, counted as a pending restart until it ends (every path)."""
        with self._lock:
            self._restarts_pending += 1
        try:
            self._spawn(self._reload_after_write, model_ids, text)
        except BaseException:
            with self._lock:
                self._restarts_pending -= 1
            raise

    def _reload_after_write(self, model_ids: list[str], text: str) -> None:
        try:
            self._reload_models(model_ids, text)
        finally:
            with self._lock:
                self._restarts_pending -= 1

    def _reload_models(self, model_ids: list[str], text: str) -> None:
        # Bounded wait (Task 7 review): if the switcher's rebuild fails for a reason
        # --check-config misses, its hash never changes; loading then would start the model on
        # the old settings, so give up with a plain message instead.
        if not self.wait_for_switcher(text):
            message = RESTART_SUPERSEDED_MESSAGE if self._superseded(text) else RESTART_STALE_MESSAGE
            self._restart_done(model_ids, False, message)
            return
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
            self._guard_test()
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
                self.writer.backup_hand_written()
                self._unload_for_restart(registry, loaded)
                revision = self.store.save(registry, expected_revision=None)
            except BaseException:
                self.writer.discard(staged)
                raise
            self.writer.backup_before_registry()
            self.writer.commit(staged)
            self._mark_written(new_text)
            self._clear_restart()
            self._write_holds({})
            self._ensure_profiles(registry)
            if loaded:
                self._spawn_reload(loaded, new_text)
            return {"status": "imported", "revision": revision, "warnings": warnings, "restarting": loaded}

    def restore(self, name: str) -> dict[str, Any]:
        """Check everything first (the backup, the holds plan, --check-config), then restore
        the registry and swap the file in: a refusal now changes nothing, where before the
        registry was already restored when the switcher refused the file (final review)."""
        with self._lock:
            self._guard_test()
            doc = self.store.read_backup(name)
            text, holds = self._plan_rewrite(doc)
            staged = self.writer.check(text) if text != self.writer.current_text() else None
            try:
                if staged is not None:
                    self.writer.backup_hand_written()
                revision = self.store.restore(name)
            except BaseException:
                if staged is not None:
                    self.writer.discard(staged)
                raise
            if staged is not None:
                self.writer.commit(staged)
                self._mark_written(text)
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
            "artifact": model["artifact"],
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
                estimate = ninfer_fit.estimate(draft, size, card_total_bytes=total, desktop_bytes=desktop,
                                             runtime=model.get("runtime"))
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

    # ---- sleep (the loaded FreeToken model gives the card back, stays in PC memory) ----
    def _sleep_word(self) -> str:
        """"asleep" or "awake" for the loaded FreeToken model (one server, one port)."""
        try:
            state = self._freetoken_state() if self._freetoken_state is not None else None
        except Exception:  # noqa: BLE001 - an unreachable helper probe reads as awake
            state = None
        return "asleep" if state == "sleeping" else "awake"

    def _mark_sleep(self, rows: list[dict[str, Any]], engines: Mapping[str, str]) -> None:
        """Add ``sleep`` to every row: the FreeToken row llama-swap reports ``ready`` gets
        "asleep" or "awake", every other row null. A sleeping FreeToken is still "ready" to
        the switcher (no llama-swap patch, spec section 2.5), so the word comes from the
        server's own state, asked once per listing."""
        loaded = [row for row in rows if engines.get(row["id"]) == "freetoken" and row.get("state") == "ready"]
        word = self._sleep_word() if loaded else None
        for row in rows:
            row["sleep"] = word if row in loaded else None

    def now(self) -> dict[str, Any]:
        running = self.switcher.running()
        up = running is not None and not is_down(running)
        try:
            doc, _ = self.store.load()
            names, floor = {m["id"]: m["name"] for m in doc["models"]}, doc["system"]["floorGB"]
            engines = {m["id"]: m["engine"] for m in doc["models"]}
        except RegistryError:
            names, floor, engines = {}, None, {}
        rows = [{"id": m, "name": names.get(m, m), "state": s} for m, s in sorted((running or {}).items())]
        self._mark_sleep(rows, engines)
        # One consistent look at the panel's own state (stage A deferred minor): the restart
        # pair is one value, and the Test-tab fields are copied together. Not under self._lock,
        # which a save holds through a 240 s unload: the strip must keep answering meanwhile.
        restart, test_running, overlay_now = self._restart, self.test_running, self.test_settings
        last = None
        if restart is not None and self._clock() - restart[1] < RESTART_SHOWN_S:
            last = restart[0]
        # Test tab: a leftover (model still on test settings) is shown until it unloads.
        loaded_now = {row["id"] for row in rows if row["state"] in LOADED_STATES}
        leftover = self.test_leftover
        if leftover is not None and up and leftover["model"] not in loaded_now:
            self.test_leftover = leftover = None
        test = None
        if test_running:
            overlay = overlay_now or {}
            test = {"running": True, "model": overlay.get("model"),
                    "name": names.get(overlay.get("model"), overlay.get("model")), "preset": overlay.get("preset")}
        return {"switcher": {"up": up, "running": rows, "stale": up and self._switcher_stale()},
                "card": self._card_probe(), "windowsFreeBytes": self._windows_free(), "cushionGB": floor,
                "held": sorted(self._read_holds()), "lastRestart": last, "test": test,
                "testLeftover": None if leftover is None else {**leftover, "name": names.get(leftover["model"], leftover["model"])}}

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
                "aliases": list(model.get("aliases") or []),
                "ramNeedGB": model["ramNeedGB"], "idleMinutes": idle_minutes(doc, model),
                "idleFromSystem": model.get("idleMinutes") is None,
                "state": running.get(model["id"], "stopped") if up else "unknown",
                "held": model["id"] in holds,
            })
        self._mark_sleep(rows, {m["id"]: m["engine"] for m in doc["models"]})
        return {"revision": revision, "switcherUp": up, "models": rows}

    def sleep_model(self, model_id: str, action: str) -> dict[str, Any]:
        """Sleep or wake the loaded FreeToken model. Sleep keeps it loaded (llama-swap still
        says "ready"), so this goes to the helper's server proxy, never to the switcher."""
        doc, _ = self.store.load()
        model = find_model(doc, model_id)
        if model["engine"] != "freetoken":
            raise PanelError(409, "not_freetoken", "Only FreeToken models can sleep.")
        running = self.switcher.running()
        if running is None or is_down(running) or running.get(model_id) != "ready":
            raise PanelError(409, "not_loaded", f"{model['name']} is not loaded, so there is nothing to {action}.")
        if self._freetoken_control is None:
            raise PanelError(503, "helper_missing", "The settings helper cannot reach FreeToken.")
        # Refused while a Test-tab run holds the card, like load and unload.
        self._begin_action()
        try:
            result = self._freetoken_control(action)
        finally:
            self._end_action()
        status = result.get("status")
        if status == "ok":
            return {"id": model_id, "sleep": "asleep" if action == "sleep" else "awake", "result": result}
        # The server's own words (api_server /v1/sleep|/v1/wake): "busy" while a chat runs,
        # "rejected" carries the reason (a game holding the card), "timeout" after the 330 s
        # helper wait (process_manager.sleep_server), "unreachable" when nothing answered.
        if status == "busy":
            message = "A chat is still running. Try again when it finishes."
        elif status == "unreachable":
            message = "FreeToken is not answering."
        elif status == "timeout":
            message = f"FreeToken took too long to {action}. Check the Server tab in a minute."
        else:
            message = str(result.get("error") or f"Could not {action} {model['name']}.")
        raise PanelError({"busy": 409, "timeout": 504}.get(status, 503), f"{action}_{status or 'failed'}", message)

    def _begin_action(self) -> None:
        """The test guard for a panel load/unload, checked under the lock begin_test takes, and
        the action counted so begin_test refuses until it ends."""
        with self._lock:
            self._guard_test()
            self._actions += 1

    def _end_action(self) -> None:
        with self._lock:
            self._actions -= 1

    def load(self, model_id: str) -> dict[str, Any]:
        self._begin_action()
        try:
            doc, _ = self.store.load()
            find_model(doc, model_id)
            try:
                self.switcher.load(model_id)
            except SwitcherError as exc:
                raise PanelError(exc.status, exc.code, exc.message) from exc
            except OSError as exc:
                raise PanelError(503, "switcher_down", "The model switcher is not running.") from exc
        finally:
            self._end_action()
        self._clear_restart()
        return {"id": model_id, "state": "ready"}

    def unload(self, model_id: str) -> dict[str, Any]:
        self._begin_action()
        try:
            doc, _ = self.store.load()
            find_model(doc, model_id)
            if not self.switcher.unload(model_id):
                raise PanelError(503, "switcher_down", "The model switcher is not running.")
        finally:
            self._end_action()
        self._clear_restart()
        return {"id": model_id, "state": "stopped"}

    def effective(self, model_id: str) -> dict[str, Any]:
        doc, _ = self.store.load()
        doc = self._for_switcher(doc)  # the FreeToken adapter reads the test settings during a test
        model = find_model(doc, model_id)
        if model["engine"] != "freetoken":
            raise PanelError(409, "not_freetoken", f"{model_id} is not a FreeToken model.")
        return {"id": model_id, "name": model["name"], "settings": freetoken_profile_settings(doc, model)}

    # ---- add and remove (Stage B) ----
    @staticmethod
    def _taken(doc: Mapping[str, Any]) -> list[str]:
        return [m["id"] for m in doc["models"]] + [a for m in doc["models"] for a in m.get("aliases") or []]

    @staticmethod
    def _owner(doc: Mapping[str, Any], name: str) -> str | None:
        wanted = name.lower()
        for model in doc["models"]:
            if model["id"].lower() == wanted or any(str(a).lower() == wanted for a in model.get("aliases") or []):
                return model["name"]
        return None

    @staticmethod
    def _already(doc: Mapping[str, Any], path: str) -> str | None:
        real = os.path.realpath(path)
        for model in doc["models"]:
            if os.path.realpath(expand(model["artifact"])) == real:
                return model["name"]
        return None

    def add_info(self) -> dict[str, Any]:
        return {"roots": {name: str(path) for name, path in self.add_roots.items()},
                "download": self.downloads.latest_add() if self.downloads is not None else None}

    def detect_path(self, path: str) -> dict[str, Any]:
        doc, _ = self.store.load()
        if _in_staging(path):
            found = detect("", taken=())  # the unsupported shape, with the reason swapped in
            found.update(path=str(path).strip(), reason=STAGING_MESSAGE, already=None)
            return found
        found = detect(path, taken=self._taken(doc))
        found["already"] = self._already(doc, found["path"]) if found["kind"] != "unsupported" else None
        return found

    def _hub(self, method: str, *args: Any) -> Any:
        if self.downloads is None:
            raise PanelError(503, "downloads_off", "Downloads are not available on this helper.")
        try:
            return getattr(self.downloads, method)(*args, folder_root=self.add_roots["folder"],
                                                   ninfer_root=self.add_roots["ninfer"])
        except AddUnsupported as exc:
            raise PanelError(422, "not_supported", str(exc)) from exc
        except InvalidRepository as exc:
            raise PanelError(422, "bad_link", str(exc)) from exc
        except DownloadConflict as exc:
            raise PanelError(409, "download_conflict", str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - told apart below; nothing here may escape as a bare 500
            if _is_network_error(exc):
                # The Hub's own errors (no such repo, gated, no network), in plain words; the
                # raw text (URL, request id) goes to the helper log only.
                logger.warning("Hugging Face %s failed: %s: %s", method, type(exc).__name__, exc)
                raise PanelError(502, "hub_error", _hub_reason(exc)) from exc
            if isinstance(exc, OSError):
                # A local file problem is not the Hub's fault: a full disk is 507, the rest 500.
                full = exc.errno in (errno.ENOSPC, errno.EDQUOT)
                where = self.add_roots["folder"]
                raise PanelError(507 if full else 500, "write_failed",
                                 f"Couldn't write to {where}: {exc.strerror or exc}.") from exc
            logger.exception("the download planner failed (%s)", method)
            raise PanelError(500, "add_failed", "Something went wrong preparing that download. "
                                                "Check the helper log for the details.") from exc

    def plan_download(self, link: str, entry: str | None = None) -> dict[str, Any]:
        return self._hub("plan_add", link, entry)

    def start_download(self, link: str, entry: str | None = None) -> dict[str, Any]:
        return self._hub("start_add", link, entry).as_dict()

    @staticmethod
    def _add_job(job: Any, job_id: str) -> dict[str, Any]:
        if job is None or job.kind != "add":
            raise KeyError(job_id)
        return job.as_dict()

    def download_status(self, job_id: str) -> dict[str, Any]:
        return self._add_job(self.downloads.get(job_id) if self.downloads is not None else None, job_id)

    def cancel_download(self, job_id: str) -> dict[str, Any]:
        return self._add_job(self.downloads.cancel(job_id) if self.downloads is not None else None, job_id)

    def add_model(self, path: str, model_id: Any, name: Any, ram_need: Any, revision: str | None) -> dict[str, Any]:
        with self._lock:
            doc, current_revision = self.store.load()
            if revision != current_revision:
                raise StaleRevision("The model list was changed somewhere else.")
            if _in_staging(path):
                raise PanelError(422, "not_supported", STAGING_MESSAGE)
            found = detect(path, taken=self._taken(doc))
            if found["kind"] == "unsupported":
                raise PanelError(422, "not_supported", found["reason"])
            already = self._already(doc, found["path"])
            if already:
                raise PanelError(409, "already_added", f"This is already in the list as {already}.")
            model_id, name = str(model_id or "").strip(), str(name or "").strip()
            errors: list[dict[str, str]] = []
            if not MODEL_ID_RE.fullmatch(model_id):
                errors.append({"field": "add.id", "message": f"The id: {ID_RULE}"})
            elif (owner := self._owner(doc, model_id)) is not None:
                errors.append({"field": "add.id", "message": f"The id {model_id} is already used by {owner}."})
            if not name or len(name) > NAME_MAX or any(ord(c) < 32 or ord(c) == 127 for c in name):
                errors.append({"field": "add.name", "message": f"The name must be 1 to {NAME_MAX} characters on one line."})
            ram: int | float | None = None
            try:
                ram = _number(ram_need)
                if not 0 <= ram <= 512:
                    raise ValueError("out of range")
            except (TypeError, ValueError):
                errors.append({"field": "add.ramNeedGB", "message": "The PC memory it needs must be a number from 0 to 512 GB."})
            if errors:
                raise RegistryValidationError(errors)
            entry = {"id": model_id, "name": name, "engine": found["engine"], "runtime": found["runtime"],
                     "artifact": _home_path(found["path"]), "ramNeedGB": ram, "idleMinutes": None,
                     "aliases": [], "overrides": {}, "presets": {}, "activePreset": None}
            adjusted: list[str] = []
            if entry["engine"] == "freetoken":
                entry["overrides"], adjusted = _fit_to_model(doc, entry)

            def mutate(proposed: dict) -> dict:
                proposed["models"].append(copy.deepcopy(entry))
                return proposed

            result = self._save(mutate, current_revision, None)
            engines = {m["id"]: m["engine"] for m in doc["models"]}
            result.update(status="added", id=model_id, name=name, adjusted=adjusted)
            pi = self._queue_pi(lambda: _after_save(
                "Pi add", lambda: self.pi.add(model_id, name, entry["engine"], engines),
                {"status": "not_updated", "notes": []}, "The model was added, but Pi's list could not be updated"))
        # Pi's files live on /mnt/c, which is slow: written after the lock is released, from
        # values captured inside it, so other panel requests are not held up (review round).
        result["pi"] = self._run_pi(pi)
        return result

    def _check_deletable(self, doc: Mapping[str, Any], model: Mapping[str, Any], files: list[str]) -> None:
        """Runs before anything changes. Only the model's own files: a NInfer entry or part, or
        a folder holding config.json; never HOME or above; never a path another model's files
        are, contain or sit inside (a v3 entry can read another model's parts)."""
        name = model["name"]
        if not files:
            raise PanelError(409, "files_missing", f"{name}'s files were not found, so there is nothing to delete. "
                                                   "Remove it without deleting files.")
        home = Path(os.path.realpath(Path.home()))
        # A symlink loses only the link, so the path checked is the link's own place, not
        # where it points (review round, PR #17).
        targets = [Path(os.path.realpath(Path(item).parent)) / Path(item).name if Path(item).is_symlink()
                   else Path(os.path.realpath(item)) for item in files]
        for original, real in zip(files, targets):
            if real == home or real in home.parents or real == Path("/"):
                raise PanelError(409, "files_unsafe", f"{original} is your home folder or above it, so nothing was deleted.")
            if model["engine"] == "freetoken":
                if not (Path(original) / "config.json").is_file():
                    raise PanelError(409, "files_unsafe", f"{original} does not look like a model folder, so nothing was deleted.")
            elif not (real.name.endswith(".ninfer") or PART_RE.match(real.name)):
                raise PanelError(409, "files_unsafe", f"{original} is not a NInfer file, so nothing was deleted.")
            elif Path(original).is_dir() and not Path(original).is_symlink():
                # NInfer files are single files; a folder with a part's name is not one to rmtree.
                raise PanelError(409, "files_unsafe", f"{original} is a folder, not a NInfer file, so nothing was deleted.")
        used: list[tuple[Path, str]] = []
        for other in doc["models"]:
            if other["id"] == model["id"]:
                continue
            # Every file the other model's header names, even when it cannot load right now
            # (a missing part): a shared part must never go with this model (review round, PR #17).
            try:
                paths = referenced_files(other["engine"], other["artifact"])
            except OwnershipUnknown:
                raise PanelError(409, "files_unknown", f"{other['name']}'s files could not be read, so it is not "
                                                       "clear which files it uses and nothing was removed. Remove "
                                                       f"{name} without deleting files, or fix {other['name']} first.") from None
            # Both where each path points and the path as written (its folder resolved, the
            # name kept): a symlinked part is checked above as the link's own place, so a
            # target-only list missed a link another model reads (review round 2, PR #17).
            for path in paths:
                used.append((Path(os.path.realpath(path)), other["name"]))
                used.append((Path(os.path.realpath(Path(path).parent)) / Path(path).name, other["name"]))
        for real in targets:
            for other_path, other_name in used:
                if other_path == real or real in other_path.parents or other_path in real.parents:
                    raise PanelError(409, "files_shared", f"{other_name} uses the same files, so nothing was removed. "
                                                          f"Remove it without deleting files, or remove {other_name} first.")

    @staticmethod
    def _delete_files(files: list[str], engine: str = "freetoken") -> dict[str, Any]:
        """A symlinked entry or folder loses only the link: the files it points to may be
        another copy's, or on a drive the panel was never asked to touch (review round). A
        NInfer model is files only: nothing of it is ever removed as a folder."""
        gone, links, failed = [], [], []
        for item in files:
            path = Path(item)
            try:
                if path.is_symlink():
                    path.unlink()
                    links.append(str(path))
                elif path.is_file():
                    path.unlink()
                elif path.is_dir():
                    if engine == "ninfer":
                        failed.append(f"{path} (a folder, not a NInfer file; it was kept)")
                        continue
                    shutil.rmtree(path)
                gone.append(str(path))
            except OSError as exc:
                failed.append(f"{path} ({exc.strerror or exc})")
        if failed:
            return {"deleted": False, "paths": gone, "links": links,
                    "message": "The model was removed, but some of its files could not be deleted: " + "; ".join(failed) + "."}
        if links:
            kept = ", ".join(links)
            if len(links) == len(gone):
                message = f"{kept}: the link was removed; the files it points to were kept."
            else:
                message = f"Its files were deleted. {kept}: the link was removed; the files it points to were kept."
            return {"deleted": True, "paths": gone, "links": links, "message": message}
        return {"deleted": True, "paths": gone, "links": links, "message": "Its files were deleted."}

    def _drop_profile(self, model: Mapping[str, Any]) -> dict[str, Any] | None:
        """Runs after the list is saved: nothing here may raise, so each step reports instead
        (a bad profiles file, or a boot file the helper cannot read when it moves back)."""
        if model["engine"] != "freetoken":
            return None
        try:
            result = self.profiles.delete(profile_id(model["id"]))
        except ProfileError as exc:
            return {"deleted": False, "message": f"Its FreeToken settings profile could not be deleted: {exc}"}
        except Exception as exc:  # noqa: BLE001 - the model is already off the list
            logger.exception("profile delete failed after the list was saved")
            return {"deleted": False, "message": "Its FreeToken settings profile could not be deleted "
                                                 f"({exc.__class__.__name__}: {exc})."}
        out = {"deleted": bool(result.get("deleted"))}
        if result.get("deleted") and result.get("activeProfileId") and result.get("bootFilePath") and self.profile_deleted:
            try:
                self.profile_deleted(result)
            except Exception as exc:  # noqa: BLE001 - the profile is gone; the helper stays on its current boot file
                logger.exception("moving the helper off the deleted profile failed")
                out["message"] = ("Its profile was deleted, but the helper could not move back to the default "
                                  f"boot file ({exc.__class__.__name__}: {exc}). Pick a profile on the Server tab.")
        return out

    def remove_model(self, model_id: str, delete_files: bool, revision: str | None,
                     artifact: str | None = None) -> dict[str, Any]:
        """``artifact`` is the model's files as the page's Remove question named them: a model
        removed and added again under the same id must not have the new one's files deleted
        after a question about the old one's (PR #20 review)."""
        with self._lock:
            doc, current_revision = self.store.load()
            try:
                model = find_model(doc, model_id)
            except KeyError:
                model = None
            if model is not None and artifact is not None and model["artifact"] != artifact:
                raise PanelError(409, "artifact_changed", ARTIFACT_CHANGED_MESSAGE)
            if revision != current_revision:
                raise StaleRevision("The model list was changed somewhere else.")
            if model is None:
                raise PanelError(404, "not_found", NOT_FOUND_MESSAGE)
            name = model["name"]
            files = model_files(model["engine"], model["artifact"]) if delete_files else []
            if delete_files:
                self._check_deletable(doc, model, files)
            known, _down = self._ask()
            if known is None:
                raise PanelError(503, "switcher_unknown", f"Can't tell whether {name} is loaded right now, "
                                                          "so it was not removed. Try again in a moment.")
            unloaded = False

            def put_away(final: list[str] | None) -> None:
                # Runs inside _save once every check has passed, so a remove the checks refuse
                # (a limit, --check-config) never pulls the model out first (review round). The
                # decision uses _save's own, final switcher answer: the model may have been
                # loaded since the look above, and an unknown state refuses (review, PR #17).
                nonlocal unloaded
                if final is None:
                    raise PanelError(503, "switcher_unknown", f"Can't tell whether {name} is loaded right now, "
                                                              "so it was not removed. Try again in a moment.")
                if model_id in final:
                    if not self.switcher.unload(model_id):
                        raise PanelError(503, "unload_failed", f"Couldn't put {name} away, so nothing was removed. "
                                                               "Try again in a moment.")
                    unloaded = True

            def mutate(proposed: dict) -> dict:
                proposed["models"] = [m for m in proposed["models"] if m["id"] != model_id]
                return proposed

            try:
                result = self._save(mutate, current_revision, None, before_commit=put_away)
            except PanelError:
                raise
            except Exception as exc:
                if not unloaded:
                    raise
                logger.exception("the list could not be saved after %s was put away", model_id)
                raise PanelError(500, "remove_failed", f"{name} was put away but not removed ({exc}). "
                                                       "It is still in the list; try again in a moment.") from exc
            result.update(status="removed", id=model_id, name=name)
            result["profile"] = _after_save("profile delete", lambda: self._drop_profile(model),
                                            {"deleted": False},
                                            "Its FreeToken settings profile could not be deleted")
            result["files"] = _after_save("file delete", lambda: self._delete_files(files, model["engine"]),
                                          {"deleted": False, "paths": [], "links": []},
                                          "The model was removed, but its files could not be deleted") if delete_files else None
            pi = self._queue_pi(lambda: _after_save(
                "Pi remove", lambda: self.pi.remove(model_id),
                {"status": "not_updated", "notes": []}, "The model was removed, but Pi's list could not be updated"))
        # Pi (slow /mnt/c) after the lock is released, like add_model.
        result["pi"] = self._run_pi(pi)
        return result

    # ---- test tab (part 3) ----
    def _guard_test(self) -> None:
        if self.test_running:
            raise PanelError(409, "test_running", TEST_RUNNING_MESSAGE)
        if self.recovering:
            raise PanelError(409, "busy", RECOVERING_MESSAGE)

    def begin_recovery(self) -> None:
        with self._lock:
            self.recovering = True

    def end_recovery(self) -> None:
        with self._lock:
            self.recovering = False

    def begin_test(self) -> None:
        with self._lock:
            self._guard_test()
            if self._restarts_pending:
                raise PanelError(409, "busy", RESTART_PENDING_MESSAGE)
            if self._actions:
                raise PanelError(409, "busy", PANEL_BUSY_MESSAGE)
            # test_leftover is not cleared here: the model may still run test settings. The
            # runner clears it once it puts that model away (playground.py _put_away).
            self.test_running = True

    def end_test(self) -> None:
        with self._lock:
            self.test_running = False

    def held_models(self) -> list[str]:
        return sorted(self._read_holds())

    def _for_switcher(self, doc: Mapping[str, Any]) -> Mapping[str, Any]:
        """doc as the switcher and the FreeToken adapter should see it: during a test the test
        model runs the chosen preset alone, without the model's own overrides (a preset is a
        full snapshot of the settings it was saved from, see preset_action "add")."""
        test = self.test_settings
        if not test:
            return doc
        out = copy.deepcopy(dict(doc))
        try:
            model = find_model(out, test["model"])
        except KeyError:
            return doc
        model["activePreset"], model["overrides"] = test["preset"], {}
        return out

    def test_preset_key(self, model_id: str, preset: str | None) -> str | None:
        """preset, or None when running it would give exactly the saved settings (same switcher
        entry and, for FreeToken, the same profile settings), so no restart is needed."""
        if not preset:
            return None
        doc, _ = self.store.load()
        model = find_model(doc, model_id)
        trial = copy.deepcopy(doc)
        tried = find_model(trial, model_id)
        tried["activePreset"], tried["overrides"] = preset, {}
        same_entry = (extract_model_blocks(render_config(doc, {})).get(model_id)
                      == extract_model_blocks(render_config(trial, {})).get(model_id))
        same_profile = model["engine"] != "freetoken" or \
            freetoken_profile_settings(doc, model) == freetoken_profile_settings(trial, tried)
        return None if same_entry and same_profile else preset

    def set_test_settings(self, model_id: str | None, preset: str | None) -> str:
        """Point the switcher at the test settings, or clear them (model_id or preset None), and
        return the text now in the file. Never writes the registry. The marker is written
        before the file when setting, and removed after the file when clearing, so a crash in
        between always leaves a marker for recover()."""
        with self._lock:
            doc, _ = self.store.load()
            if model_id is not None and preset:
                if preset not in (find_model(doc, model_id).get("presets") or {}):
                    raise KeyError(preset)
                self.test_settings = {"model": model_id, "preset": preset}
                self._write_test_marker()
            else:
                self.test_settings = None
            text, holds = self._plan_rewrite(doc)
            if self.writer.write(text):
                self._mark_written(text)
            self._write_holds(holds)
            if self.test_settings is None:
                self.clear_test_marker()
            return text

    def _write_test_marker(self) -> None:
        self.test_marker_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.test_marker_path.with_name(self.test_marker_path.name + ".tmp")
        temporary.write_text(json.dumps({**(self.test_settings or {}), "at": _now_iso()}) + "\n", encoding="utf-8")
        os.replace(temporary, self.test_marker_path)

    def read_test_marker(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.test_marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and data.get("model") else None

    def clear_test_marker(self) -> None:
        self.test_marker_path.unlink(missing_ok=True)

    def note_test_leftover(self, model_id: str, preset: str | None) -> None:
        self.test_leftover = {"model": model_id, "preset": preset}

    @property
    def test_leftover(self) -> dict[str, Any] | None:
        return self._test_leftover

    @test_leftover.setter
    def test_leftover(self, value: dict[str, Any] | None) -> None:
        """Kept in memory and on disk: cleared only once the model is seen unloaded (now(),
        playground._put_away), so a helper restart in between still knows about it."""
        self._test_leftover = value
        try:
            if value is None:
                self.test_leftover_path.unlink(missing_ok=True)
            else:
                self.test_leftover_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.test_leftover_path.with_name(self.test_leftover_path.name + ".tmp")
                temporary.write_text(json.dumps({**value, "at": _now_iso()}) + "\n", encoding="utf-8")
                os.replace(temporary, self.test_leftover_path)
        except OSError:
            logger.exception("The Test tab's leftover note could not be saved")

    def _read_test_leftover(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.test_leftover_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or not data.get("model"):
            return None
        return {"model": str(data["model"]), "preset": data.get("preset")}

    def _later_panel_write(self, expected: str, live: str) -> bool:
        """live is a file the panel itself wrote after the one hashed as expected."""
        gen = self._written.get(live)
        return gen is not None and gen > self._written.get(expected, 0)

    def _superseded(self, text: str) -> bool:
        """The file on disk is neither text nor a later panel write: something else replaced it."""
        expected = config_sha256(text)
        on_disk = self.writer.current_text()
        if on_disk is None:
            return False
        now = config_sha256(on_disk)
        return now != expected and not self._later_panel_write(expected, now)

    def wait_for_switcher(self, text: str, stop: threading.Event | None = None) -> bool:
        """True once the switcher reports text's hash (P5), or the hash of a file the panel
        wrote after text; False after restart_wait_s, as soon as stop is set (the Test tab's
        Stop), or as soon as the switcher is on a file the panel did not write after text.

        The file may be rewritten by the panel after this write and before the switcher
        catches up (the hold watcher, a second save): comparing only against text gave up
        although the switcher was on the newer file (stage A deferred minor). Any other file
        (an old copy put back, a hand edit) is not accepted: its hash once read as "applied"
        and a superseded restart loaded the model on the old settings (PR #20 review)."""
        expected = config_sha256(text)
        deadline = self._clock() + self.restart_wait_s
        while True:
            live = self.switcher.config_hash()
            if live is not None:
                if live == expected or self._later_panel_write(expected, live):
                    return True
                on_disk = self.writer.current_text()
                if on_disk is not None and live == config_sha256(on_disk) and self._superseded(text):
                    return False  # the switcher is on the file that replaced ours; waiting won't help
            if self._clock() >= deadline or (stop is not None and stop.is_set()):
                return False
            self._sleep(0.5)

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


class DetectBody(BaseModel):
    path: str = ""


class LinkBody(BaseModel):
    link: str = ""
    entry: str | None = None


class AddBody(_SaveBody):
    path: str = ""
    id: str = ""
    name: str = ""
    ramNeedGB: Any = None


class RemoveBody(BaseModel):
    revision: str | None = None
    deleteFiles: bool = False
    artifact: str | None = None


def create_panel_router(service: PanelService) -> APIRouter:
    router = APIRouter(prefix="/api/panel")

    async def call(fn: Callable[..., Any], *args: Any, missing: str = MISSING_MESSAGE) -> Any:
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
            logger.info("panel %s: not found: %r", getattr(fn, "__name__", "call"), exc.args[0] if exc.args else None)
            return JSONResponse(status_code=404, content={"code": "not_found", "message": missing})

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

    @router.post("/models/{model_id}/sleep")
    async def sleep_model(model_id: str):
        return await call(service.sleep_model, model_id, "sleep")

    @router.post("/models/{model_id}/wake")
    async def wake_model(model_id: str):
        return await call(service.sleep_model, model_id, "wake")

    @router.get("/models/{model_id}/effective")
    async def effective(model_id: str):
        return await call(service.effective, model_id)

    @router.get("/add/info")
    async def add_info():
        return await call(service.add_info)

    @router.post("/add/detect")
    async def add_detect(body: DetectBody):
        return await call(service.detect_path, body.path)

    @router.post("/add/plan")
    async def add_plan(body: LinkBody):
        return await call(service.plan_download, body.link, body.entry)

    @router.post("/add/downloads")
    async def add_download(body: LinkBody):
        return await call(service.start_download, body.link, body.entry)

    @router.get("/add/downloads/{job_id}")
    async def add_download_status(job_id: str):
        return await call(service.download_status, job_id, missing=MISSING_DOWNLOAD_MESSAGE)

    @router.post("/add/downloads/{job_id}/cancel")
    async def add_download_cancel(job_id: str):
        return await call(service.cancel_download, job_id, missing=MISSING_DOWNLOAD_MESSAGE)

    @router.post("/models")
    async def add_model(body: AddBody):
        return await call(service.add_model, body.path, body.id, body.name, body.ramNeedGB, body.revision)

    @router.post("/models/{model_id}/remove")
    async def remove_model(model_id: str, body: RemoveBody):
        return await call(service.remove_model, model_id, body.deleteFiles, body.revision, body.artifact)

    return router


__all__ = ["ChooseRestart", "IDENTITY_DIALS", "PanelError", "PanelService", "SYSTEM_DIALS", "create_panel_router",
           "default_add_roots"]

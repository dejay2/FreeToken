"""The control panel's registry: every local model, its engine and its settings.

One JSON file on the serving box (~/.config/freetoken/registry.json, never tracked) is the single
source of truth. The switcher config (swap_config.py) and each FreeToken model's helper profile
are generated from it. Settings layer, later wins: engine defaults -> the active preset -> the
model's own changes. Every save keeps the previous file as registry.json.bak-<time>; the newest
20 stay. A file that is not valid JSON, or valid JSON with bad content, is "corrupt": nothing is
generated from it and the page offers the backups (spec: Error handling).
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from . import ninfer_dials
from .dials import DIAL_BY_NAME as FREETOKEN_DIALS
from .dials import DIALS as FREETOKEN_DIAL_LIST
from .dials import canonical_value, validate_settings

REGISTRY_VERSION = 1
BACKUPS_KEPT = 20
ENGINES = ("ninfer", "freetoken")
ENGINE_LABELS = {"ninfer": "NInfer", "freetoken": "FreeToken"}
RUNTIMES = {"ninfer": ninfer_dials.RUNTIMES, "freetoken": ("freetoken",)}
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
ID_RULE = "use 1 to 63 small letters, numbers, dots, dashes or underscores, starting with a letter or number."
NAME_MAX = 120
PRESET_NAME_MAX = 60
SCHEMA_PATH = Path(__file__).with_name("registry.schema.json")
SYSTEM_DEFAULTS = {"floorGB": 6, "waitSeconds": 300, "latestWins": True, "defaultIdleMinutes": 0,
                   "helperURL": "http://127.0.0.1:2031"}
ACTIVE = object()  # "the model's active preset"


class RegistryError(RuntimeError):
    pass


class RegistryMissing(RegistryError):
    pass


class RegistryCorrupt(RegistryError):
    def __init__(self, message: str, backups: list[str]) -> None:
        super().__init__(message)
        self.message = message
        self.backups = backups


class StaleRevision(RegistryError):
    pass


class RegistryValidationError(ValueError):
    def __init__(self, errors: list[dict[str, str]]) -> None:
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))
        self.errors = errors


def default_path() -> Path:
    return Path(os.environ.get("FREETOKEN_REGISTRY") or Path.home() / ".config" / "freetoken" / "registry.json")


def revision_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dumps(doc: Mapping[str, Any]) -> bytes:
    return (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def same(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def differences(settings: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in settings.items() if name not in base or not same(value, base[name])}


def expand(path: str) -> str:
    return os.path.expanduser(path)


def find_model(doc: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    for model in doc.get("models") or []:
        if isinstance(model, dict) and model.get("id") == model_id:
            return model
    raise KeyError(model_id)


def catalogue_defaults(engine: str) -> dict[str, Any]:
    if engine == "ninfer":
        return dict(ninfer_dials.BUILTINS)
    # Canonical, like every stored value: GpuOwnedLayers' catalogue default "auto" is stored as
    # "auto:6", and an uncanonical base would make an untouched dial look "changed for this model".
    out: dict[str, Any] = {}
    for dial in FREETOKEN_DIAL_LIST:
        if dial.name == "ModelPath":
            continue
        try:
            out[dial.name] = canonical_value(dial, dial.default)
        except (TypeError, ValueError, OverflowError):
            out[dial.name] = dial.default
    return out


def engine_defaults(doc: Mapping[str, Any], engine: str) -> dict[str, Any]:
    return {**catalogue_defaults(engine), **(doc["engines"][engine].get("defaults") or {})}


def preset_values(model: Mapping[str, Any], preset: Any = ACTIVE) -> dict[str, Any]:
    name = model.get("activePreset") if preset is ACTIVE else preset
    if not name:
        return {}
    return dict((model.get("presets") or {}).get(name) or {})


def base_settings(doc: Mapping[str, Any], model: Mapping[str, Any], preset: Any = ACTIVE) -> dict[str, Any]:
    return {**engine_defaults(doc, model["engine"]), **preset_values(model, preset)}


def effective_settings(doc: Mapping[str, Any], model: Mapping[str, Any], preset: Any = ACTIVE) -> dict[str, Any]:
    effective = {**base_settings(doc, model, preset), **(model.get("overrides") or {})}
    if model["engine"] == "freetoken":
        effective["ModelPath"] = expand(model["artifact"])
    return effective


def freetoken_profile_settings(doc: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    """What freetoken.sh --profile writes into the helper profile model-<id>."""
    return {name: value for name, value in effective_settings(doc, model).items() if name in FREETOKEN_DIALS}


def idle_minutes(doc: Mapping[str, Any], model: Mapping[str, Any]) -> int:
    value = model.get("idleMinutes")
    return int(doc["system"]["defaultIdleMinutes"] if value is None else value)


def canonical_engine_settings(engine: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate_engine_settings(engine, settings, cross=False)
    if errors:
        raise RegistryValidationError(errors)
    if engine == "ninfer":
        return ninfer_dials.canonical_settings(settings)
    return {name: canonical_value(FREETOKEN_DIALS[name], value) for name, value in settings.items()}


def validate_engine_settings(engine: str, settings: Any, runtime: str | None = None, cross: bool = True) -> list[dict[str, str]]:
    if not isinstance(settings, Mapping):
        return [{"field": "settings", "message": "Settings must be a list of names and values."}]
    if engine == "ninfer":
        return ninfer_dials.validate(settings, runtime, cross=cross)
    errors: list[dict[str, str]] = []
    known: dict[str, Any] = {}
    for name, value in settings.items():
        if name == "ModelPath":
            errors.append({"field": name, "message": "The model folder is set for each model, not here."})
        elif name not in FREETOKEN_DIALS:
            errors.append({"field": name, "message": f"Unknown FreeToken setting {name}"})
        else:
            known[name] = value
    return errors + validate_settings(known, ceilings_only=True)


def _one_line(text: Any, limit: int) -> bool:
    return isinstance(text, str) and 0 < len(text.strip()) and len(text) <= limit and not any(
        ord(char) < 32 or ord(char) == 127 for char in text)


def _where(errors: list[dict[str, str]], where: str) -> list[dict[str, str]]:
    return [{**error, "where": where} for error in errors]


def summarize_errors(errors: list[dict[str, str]], limit: int = 5) -> str:
    """One plain-words line for the first few errors (Task 4's registry_import reuses this)."""
    return "; ".join(f"{e.get('where', '')} {e['field']}: {e['message']}".strip() for e in errors[:limit])


def validate_system(system: Any) -> list[dict[str, str]]:
    if not isinstance(system, Mapping):
        return [{"field": "system", "message": "The system settings are missing."}]
    errors = []

    def number(name: str, low: float, high: float, whole: bool, message: str) -> None:
        value = system.get(name)
        ok = not isinstance(value, bool) and isinstance(value, (int, float)) and low <= value <= high
        if ok and whole and not float(value).is_integer():
            ok = False
        if not ok:
            errors.append({"field": name, "message": message})

    number("floorGB", 0, 64, False, "The PC memory cushion must be 0 to 64 GB.")
    number("waitSeconds", 0, 3600, True, "The memory wait must be 0 to 3,600 whole seconds.")
    number("defaultIdleMinutes", 0, 1440, True, "Unload when idle must be 0 to 1,440 whole minutes.")
    if not isinstance(system.get("latestWins"), bool):
        errors.append({"field": "latestWins", "message": "Newest pick wins must be on or off."})
    if not isinstance(system.get("helperURL"), str) or not system["helperURL"].startswith("http://127.0.0.1:"):
        errors.append({"field": "helperURL", "message": "The settings page address must be http://127.0.0.1:<port>."})
    return errors


def _model_errors(model: Any, index: int, taken: dict[str, str]) -> list[dict[str, str]]:
    if not isinstance(model, dict):
        return [{"field": f"models[{index}]", "message": "Each model must be an object."}]
    errors: list[dict[str, str]] = []
    model_id = model.get("id") if isinstance(model.get("id"), str) and MODEL_ID_RE.fullmatch(model.get("id")) else None
    where = model_id or f"models[{index}]"

    def add(field: str, message: str) -> None:
        errors.append({"field": field, "message": message, "where": where})

    if model_id is None:
        errors.append({"field": f"models[{index}].id", "message": f"Model id {model.get('id')!r}: {ID_RULE}"})
    if not _one_line(model.get("name"), NAME_MAX):
        add("model.name", f"The name must be 1 to {NAME_MAX} characters on one line.")
    engine = model.get("engine")
    if engine not in ENGINES:
        add("engine", "The engine must be ninfer or freetoken.")
        return errors
    runtime = model.get("runtime")
    if runtime not in RUNTIMES[engine]:
        add("runtime", "The runtime must be one of " + ", ".join(RUNTIMES[engine]) + ".")
    artifact = model.get("artifact")
    if not _one_line(artifact, 4096) or not artifact.startswith(("~/", "/")):
        add("artifact", "The model file or folder must be a full path (starting with / or ~/).")
    ram = model.get("ramNeedGB")
    if isinstance(ram, bool) or not isinstance(ram, (int, float)) or not 0 <= ram <= 512:
        add("model.ramNeedGB", "The PC memory it needs must be 0 to 512 GB.")
    idle = model.get("idleMinutes")
    if idle is not None and (isinstance(idle, bool) or not isinstance(idle, int) or not 0 <= idle <= 1440):
        add("model.idleMinutes", "Unload when idle must be 0 to 1,440 whole minutes, or the System setting.")
    aliases = model.get("aliases", [])
    if not isinstance(aliases, list):
        add("model.aliases", "Other names must be a list.")
    else:
        for alias in aliases:
            if not isinstance(alias, str) or not ALIAS_RE.fullmatch(alias):
                add("model.aliases", f"Other name {alias!r}: use letters, numbers, dots, dashes, underscores, colons or slashes (up to 128).")
            elif alias in taken:
                add("model.aliases", f"Other name {alias} is already used by {taken[alias]}.")
            else:
                taken[alias] = where
    overrides = model.get("overrides", {})
    errors += _where(validate_engine_settings(engine, overrides, runtime if engine == "ninfer" else None, cross=False), where)
    presets = model.get("presets", {})
    if not isinstance(presets, dict):
        add("presets", "Presets must be a list of names and values.")
        presets = {}
    for name, values in presets.items():
        if not _one_line(name, PRESET_NAME_MAX):
            add("preset", f"A preset name must be 1 to {PRESET_NAME_MAX} characters on one line.")
        errors += _where(validate_engine_settings(engine, values, runtime if engine == "ninfer" else None, cross=False), where)
    active = model.get("activePreset")
    if active is not None and active not in presets:
        add("activePreset", "The chosen preset does not exist.")
    return errors


def validate_registry(doc: Any) -> list[dict[str, str]]:
    if not isinstance(doc, dict):
        return [{"field": "", "message": "The model list must be a JSON object."}]
    errors: list[dict[str, str]] = []
    if doc.get("version") != REGISTRY_VERSION:
        errors.append({"field": "version", "message": f"Unknown list version {doc.get('version')!r}; this page reads version {REGISTRY_VERSION}."})
    errors += validate_system(doc.get("system"))
    engines = doc.get("engines")
    if not isinstance(engines, dict) or set(engines) != set(ENGINES):
        errors.append({"field": "engines", "message": "The list needs exactly the ninfer and freetoken engines."})
    else:
        for engine in ENGINES:
            block = engines[engine]
            if not isinstance(block, dict) or not isinstance(block.get("defaults"), dict):
                errors.append({"field": f"engines.{engine}.defaults", "message": "The engine defaults are missing."})
            else:
                errors += _where(validate_engine_settings(engine, block["defaults"], cross=False), f"{ENGINE_LABELS[engine]} defaults")
    models = doc.get("models")
    if not isinstance(models, list):
        errors.append({"field": "models", "message": "The model list is missing."})
        return errors
    taken: dict[str, str] = {}
    for index, model in enumerate(models):
        model_id = model.get("id") if isinstance(model, dict) else None
        if isinstance(model_id, str) and MODEL_ID_RE.fullmatch(model_id):
            if model_id in taken:
                errors.append({"field": f"models[{index}].id", "message": f"Model id {model_id} is already used by {taken[model_id]}."})
            taken[model_id] = model_id
    for index, model in enumerate(models):
        errors += _model_errors(model, index, taken)
    if errors:
        return errors
    for model in models:
        effective = effective_settings(doc, model)
        if model["engine"] == "ninfer":
            errors += _where(ninfer_dials.validate(effective, model["runtime"]), model["id"])
        else:
            effective.pop("ModelPath", None)
            errors += _where(validate_settings(effective, ceilings_only=True), model["id"])
    return errors


def canonicalize(doc: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(doc))
    for engine in ENGINES:
        out["engines"][engine]["defaults"] = canonical_engine_settings(engine, out["engines"][engine]["defaults"])
    for model in out["models"]:
        model["overrides"] = canonical_engine_settings(model["engine"], model.get("overrides") or {})
        model["presets"] = {name: canonical_engine_settings(model["engine"], values)
                            for name, values in (model.get("presets") or {}).items()}
        model.setdefault("aliases", [])
        model.setdefault("idleMinutes", None)
        model.setdefault("activePreset", None)
    return out


class RegistryStore:
    def __init__(self, path: str | os.PathLike[str] | None = None, *, now: Callable[[], _dt.datetime] | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()
        self._now = now or _dt.datetime.now
        self._lock = threading.Lock()

    def exists(self) -> bool:
        return self.path.exists()

    def backups(self) -> list[str]:
        prefix = self.path.name + ".bak-"
        try:
            names = [entry.name for entry in self.path.parent.iterdir() if entry.name.startswith(prefix)]
        except FileNotFoundError:
            return []
        return sorted(names, reverse=True)

    def load(self) -> tuple[dict[str, Any], str]:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError as exc:
            raise RegistryMissing(f"No model list at {self.path}.") from exc
        doc = self._parse(data)
        return doc, revision_of(data)

    def _parse(self, data: bytes) -> dict[str, Any]:
        try:
            doc = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RegistryCorrupt(f"The model list is not valid JSON ({exc}).", self.backups()) from exc
        errors = validate_registry(doc)
        if errors:
            raise RegistryCorrupt(f"The model list has problems: {summarize_errors(errors)}", self.backups())
        return doc

    def save(self, doc: Mapping[str, Any], *, expected_revision: str | None) -> str:
        errors = validate_registry(doc)
        if errors:
            raise RegistryValidationError(errors)
        data = dumps(canonicalize(doc))
        with self._lock:
            try:
                current = self.path.read_bytes()
            except FileNotFoundError:
                current = None
            if (current is None) != (expected_revision is None) or (
                current is not None and revision_of(current) != expected_revision
            ):
                raise StaleRevision("The model list was changed somewhere else.")
            if current == data:
                return revision_of(data)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if current is not None:
                self._backup(current)
            self._atomic_write(data)
            self._prune()
        return revision_of(data)

    def read_backup(self, name: str) -> dict[str, Any]:
        """The registry a backup holds, checked like the current file (a bad backup is refused)."""
        if name not in self.backups():
            raise KeyError(name)
        return self._parse((self.path.parent / name).read_bytes())

    def restore(self, name: str) -> str:
        self.read_backup(name)
        data = (self.path.parent / name).read_bytes()
        with self._lock:
            try:
                current = self.path.read_bytes()
            except FileNotFoundError:
                current = None
            if current is not None:
                # A damaged current file is kept as registry.json.corrupt-<time>, outside the
                # backup list: kept as a .bak it became the newest backup, so "restore the
                # newest" brought the damage straight back (final review, open item).
                try:
                    self._parse(current)
                except RegistryCorrupt:
                    self._backup(current, kind="corrupt")
                else:
                    self._backup(current)
            self._atomic_write(data)
            self._prune()
        return revision_of(data)

    def _backup(self, data: bytes, kind: str = "bak") -> None:
        stamp = self._now().strftime("%Y%m%d-%H%M%S-%f")
        target = self.path.with_name(f"{self.path.name}.{kind}-{stamp}")
        target.write_bytes(data)
        os.chmod(target, 0o600)

    def _atomic_write(self, data: bytes) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def _prune(self) -> None:
        for name in self.backups()[BACKUPS_KEPT:]:
            (self.path.parent / name).unlink(missing_ok=True)


__all__ = [name for name in dir() if not name.startswith("_")]

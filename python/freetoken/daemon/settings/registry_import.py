"""First-run import: build the registry from today's switcher config and the helper's boot file.

Spec section 1: memoryGate becomes `system`; each model's cmd flags and filters become its
overrides against the new engine defaults; the helper's boot file becomes FreeToken defaults.
Values the generator fixes per engine (unloadTimeout, env, clampParams, proxy, checkEndpoint)
are not stored; when today's value differs, the import says so in a warning so nothing changes
silently. A command it cannot read (another adapter, an unknown NInfer option) refuses the
whole import with the reason.

``boot_settings`` is the already-loaded content of the helper's *current* boot file (Ruling
F12): this module never opens a boot file itself or hard-codes a default path. The caller (the
control panel, Task 10) resolves the current path the same way the helper does -- its
``self._boot_file()`` -- reads it, and passes the resulting settings mapping in here.
"""

from __future__ import annotations

import math
import os
import shlex
from typing import Any, Iterable, Mapping

from . import ninfer_dials
from .dials import DIAL_BY_NAME as FREETOKEN_DIALS
from .dials import canonical_value
from .registry import (
    ID_RULE, MODEL_ID_RE, PRESET_NAME_MAX, SYSTEM_DEFAULTS, differences, engine_defaults, same, summarize_errors,
    validate_registry,
)
from .swap_config import NINFER_ENV, PROXY, UNLOAD_TIMEOUT, describe_launch, expand_macros


class ImportRefused(RuntimeError):
    pass


def _tilde(path: str, home: str) -> str:
    if home and (path == home or path.startswith(home.rstrip("/") + "/")):
        return "~" + path[len(home.rstrip("/")):]
    return path


def _minutes(seconds: Any, what: str, warnings: list[str]) -> int:
    seconds = int(seconds or 0)
    minutes = math.ceil(seconds / 60)
    if seconds % 60:
        warnings.append(f"{what} {seconds} is not whole minutes; it becomes {minutes} minute(s).")
    return minutes


def _number(value: Any) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def _freetoken_values(settings: Mapping[str, Any], warnings: list[str], label: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in settings.items():
        dial = FREETOKEN_DIALS.get(name)
        if dial is None or name == "ModelPath":
            continue
        try:
            out[name] = canonical_value(dial, value)
        except (TypeError, ValueError, OverflowError):
            warnings.append(f"{label}: {name} = {value!r} could not be read and was left out.")
    return out


def _check_fixed(model_id: str, entry: Mapping[str, Any], engine: str, warnings: list[str]) -> None:
    timeout = entry.get("unloadTimeout")
    if timeout is not None and int(timeout) != UNLOAD_TIMEOUT[engine]:
        warnings.append(f"{model_id}: unloadTimeout {timeout} becomes {UNLOAD_TIMEOUT[engine]} (one value per engine).")
    if entry.get("proxy") not in (None, PROXY[engine]):
        warnings.append(f"{model_id}: proxy {entry.get('proxy')} becomes {PROXY[engine]}.")
    if engine == "ninfer":
        if list(entry.get("env") or []) != NINFER_ENV:
            warnings.append(f"{model_id}: env {entry.get('env')} becomes {NINFER_ENV}.")
        clamp = ((entry.get("filters") or {}).get("clampParams") or {})
        wanted = ninfer_dials.clamp_params()
        if clamp and {k: [float(x) for x in v] for k, v in clamp.items()} != {k: [float(x) for x in v] for k, v in wanted.items()}:
            warnings.append(f"{model_id}: clampParams become the catalogue ranges {wanted}.")


def _profiles_to_presets(registry: dict[str, Any], profiles: Iterable[Mapping[str, Any]], warnings: list[str]) -> None:
    freetoken_models = [m for m in registry["models"] if m["engine"] == "freetoken"]
    defaults = engine_defaults(registry, "freetoken")
    for profile in profiles:
        profile_id = str(profile.get("id") or "")
        if profile.get("kind") == "default" or profile_id in ("", "default") or profile_id.startswith("model-"):
            continue
        settings = profile.get("settings") or {}
        values = differences(_freetoken_values(settings, warnings, f"profile {profile.get('name')}"), defaults)
        if not values:
            continue
        folder = os.path.basename(str(settings.get("ModelPath") or "").rstrip("/\\"))
        targets = [m for m in freetoken_models if not folder or os.path.basename(m["artifact"].rstrip("/")) == folder]
        base_name = str(profile.get("name") or profile_id).strip()[:PRESET_NAME_MAX] or profile_id
        for model in targets:
            name, number = base_name, 2
            while name in model["presets"]:
                suffix = f" ({number})"
                name, number = base_name[: PRESET_NAME_MAX - len(suffix)] + suffix, number + 1
            model["presets"][name] = dict(values)


def import_live(config_text: str, boot_settings: Mapping[str, Any], *, env: Mapping[str, str],
                profiles: Iterable[Mapping[str, Any]] = ()) -> tuple[dict[str, Any], list[str]]:
    import yaml

    try:
        doc = yaml.safe_load(config_text)
    except yaml.YAMLError as exc:
        raise ImportRefused(f"today's switcher config is not readable YAML: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), dict) or not doc["models"]:
        raise ImportRefused("today's switcher config lists no models")
    warnings: list[str] = []
    home = env.get("HOME", "")
    gate = doc.get("memoryGate") or {}
    fifo = ((((doc.get("routing") or {}).get("scheduler") or {}).get("settings") or {}).get("fifo") or {})
    system = {
        "floorGB": _number(gate.get("floorGB", SYSTEM_DEFAULTS["floorGB"])),
        "waitSeconds": int(gate.get("waitSeconds", SYSTEM_DEFAULTS["waitSeconds"])),
        "latestWins": fifo.get("latestWins", True) is not False,
        "defaultIdleMinutes": _minutes(doc.get("globalTTL", 0), "globalTTL", warnings),
        "helperURL": str(gate.get("helperURL") or SYSTEM_DEFAULTS["helperURL"]),
    }
    macros = {str(key): str(value) for key, value in (doc.get("macros") or {}).items()}
    rows = []
    for raw_id, entry in doc["models"].items():
        model_id, entry = str(raw_id), entry or {}
        if not MODEL_ID_RE.fullmatch(model_id):
            raise ImportRefused(f"model {model_id!r}: {ID_RULE}")
        try:
            launch = describe_launch(shlex.split(expand_macros(str(entry.get("cmd") or ""), macros, env)))
        except ValueError as exc:
            raise ImportRefused(f"model {model_id}: {exc}") from exc
        rows.append((model_id, entry, launch))
    ninfer_rows = [row for row in rows if row[2]["engine"] == "ninfer"]
    ninfer_defaults: dict[str, Any] = {}
    if ninfer_rows:
        for name, value in ninfer_rows[0][2]["settings"].items():
            if same(value, ninfer_dials.BUILTINS[name]):
                continue
            if all(same(row[2]["settings"][name], value) for row in ninfer_rows):
                ninfer_defaults[name] = value
    ninfer_base = {**ninfer_dials.BUILTINS, **ninfer_defaults}
    models = []
    for model_id, entry, launch in rows:
        engine = launch["engine"]
        model: dict[str, Any] = {"id": model_id, "name": str(entry.get("name") or model_id), "engine": engine}
        if engine == "ninfer":
            if launch["modelId"] != model_id:
                warnings.append(f"{model_id}: the engine was told to call itself {launch['modelId']}; it becomes {model_id}.")
            if (launch["host"], launch["port"]) != ("127.0.0.1", "8090"):
                warnings.append(f"{model_id}: host/port {launch['host']}:{launch['port']} become 127.0.0.1:8090.")
            model.update(runtime=launch["runtime"], artifact=_tilde(launch["artifact"], home),
                         overrides=differences(launch["settings"], ninfer_base))
        else:
            model.update(runtime="freetoken", artifact=_tilde(launch["folder"], home), overrides={})
        _check_fixed(model_id, entry, engine, warnings)
        ttl = entry.get("ttl")
        model.update(
            ramNeedGB=_number(entry.get("ramNeedGB", 0) or 0),
            idleMinutes=None if ttl is None else _minutes(ttl, f"{model_id}: ttl", warnings),
            aliases=[str(alias) for alias in entry.get("aliases") or []],
            presets={},
            activePreset=None,
        )
        models.append(model)
    registry = {
        "version": 1,
        "system": system,
        "engines": {"ninfer": {"defaults": ninfer_defaults},
                    "freetoken": {"defaults": _freetoken_values(boot_settings, warnings, "boot file")}},
        "models": models,
    }
    _profiles_to_presets(registry, profiles, warnings)
    errors = validate_registry(registry)
    if errors:
        raise ImportRefused(f"the imported list has problems: {summarize_errors(errors)}")
    return registry, warnings


__all__ = ["ImportRefused", "import_live"]

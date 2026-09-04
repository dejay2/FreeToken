"""Small JSON store for the settings helper's named profiles."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .boot_parser import BootFile, BootValidationError
from .dials import validate_settings


PRESET_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "id": "profile-a",
        "name": "Profile A (BF16 Default)",
        "description": "262K context, 4,188 slots, BF16 KV cache, parking off",
        "isPreset": True,
        "settings": {"KVDtype": "bf16", "MoECacheSize": 4188, "KVPark": "off"},
    },
    {
        "id": "profile-b",
        "name": "Profile B (FP8)",
        "description": "262K context, 5,332 slots, FP8 KV cache, parking off",
        "isPreset": True,
        "settings": {"KVDtype": "fp8", "MoECacheSize": 5332, "KVPark": "off"},
    },
)


class ProfileError(RuntimeError):
    pass


class ProfileValidationError(ValueError):
    def __init__(self, errors: list[dict[str, str]]) -> None:
        self.errors = errors
        super().__init__("; ".join(item["message"] for item in errors))


class ProfilesManager:
    def __init__(self, path: str | os.PathLike[str], *, boot_file=None, clock=time.time) -> None:
        self.path = Path(path)
        self.boot_file = Path(boot_file) if boot_file is not None else None
        self._clock = clock

    def list(self) -> list[dict[str, Any]]:
        return [*self._presets(), *self._custom()]

    all = list

    def get(self, profile_id: str) -> dict[str, Any] | None:
        for profile in self.list():
            if profile["id"] == profile_id:
                return profile
        return None

    def create(self, name: str, description: str, settings: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise ProfileValidationError([{"field": "name", "message": "Profile name is required"}])
        if not isinstance(description, str):
            raise ProfileValidationError([{"field": "description", "message": "Description must be text"}])
        if not isinstance(settings, dict):
            raise ProfileValidationError([{"field": "settings", "message": "must be an object"}])
        errors = validate_settings(settings)
        if errors:
            raise ProfileValidationError(errors)
        profiles = self._custom()
        base = f"prof-{int(self._clock())}"
        profile_id = base
        while any(item["id"] == profile_id for item in profiles):
            profile_id = f"{base}-{uuid.uuid4().hex[:6]}"
        profile = {
            "id": profile_id,
            "name": name.strip(),
            "description": description,
            "isPreset": False,
            "settings": dict(settings),
        }
        profiles.append(profile)
        self._write_custom(profiles)
        return profile

    def delete(self, profile_id: str) -> dict[str, Any]:
        profiles = self._custom()
        kept = [item for item in profiles if item["id"] != profile_id]
        deleted = len(kept) != len(profiles)
        if deleted:
            self._write_custom(kept)
        return {"deleted": deleted, "id": profile_id}

    def apply(self, profile_id: str, boot_file=None) -> dict[str, Any]:
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        path = boot_file if boot_file is not None else self.boot_file
        if path is None:
            raise ProfileError("no boot file configured")
        try:
            saved = BootFile(path).save(dict(profile["settings"]))
        except BootValidationError:
            raise
        return {
            "applied": True,
            "profileId": profile_id,
            "backupPath": str(BootFile(path).backup_path),
            "settings": saved,
        }

    def _presets(self) -> list[dict[str, Any]]:
        # Return fresh objects so a route caller cannot mutate the process-wide constants.
        return json.loads(json.dumps(PRESET_PROFILES))

    def _custom(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except FileNotFoundError:
            return []
        except (OSError, ValueError, TypeError):
            return []
        if isinstance(doc, dict):
            items = doc.get("profiles", [])
        else:
            items = doc
        if not isinstance(items, list):
            return []
        out = []
        for item in items:
            if not isinstance(item, dict) or item.get("isPreset"):
                continue
            if not item.get("id") or not isinstance(item.get("settings"), dict):
                continue
            out.append(
                {
                    "id": str(item["id"]),
                    "name": str(item.get("name", "")),
                    "description": str(item.get("description", "")),
                    "isPreset": False,
                    "settings": dict(item["settings"]),
                }
            )
        return out

    def _write_custom(self, profiles: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump({"profiles": profiles}, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, self.path)


ProfileManager = ProfilesManager


__all__ = [
    "PRESET_PROFILES",
    "ProfileError",
    "ProfileManager",
    "ProfileValidationError",
    "ProfilesManager",
]

"""JSON metadata and PowerShell files for the settings helper's named profiles."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from .boot_parser import BootFile, BootParseError, BootValidationError
from .dials import validate_settings
from .profile_scripts import render_boot_script


PRESET_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "id": "profile-a",
        "name": "Profile A (BF16 Default)",
        "description": "262K context, 4,188 slots, BF16 KV cache, parking off",
        "isPreset": True,
        "label": "preset",
        "kind": "preset",
        "settings": {"KVDtype": "bf16", "MoECacheSize": 4188, "KVPark": "off"},
    },
    {
        "id": "profile-b",
        "name": "Profile B (FP8)",
        "description": "262K context, 5,332 slots, FP8 KV cache, parking off",
        "isPreset": True,
        "label": "preset",
        "kind": "preset",
        "settings": {"KVDtype": "fp8", "MoECacheSize": 5332, "KVPark": "off"},
    },
)


_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_UNSET = object()
_DEFAULT_LAUNCHER_RELATIVE = r"..\scripts\start-freetoken-windows.ps1"


class ProfileError(RuntimeError):
    pass


class ProfileValidationError(ValueError):
    def __init__(self, errors: list[dict[str, str]]) -> None:
        self.errors = errors
        super().__init__("; ".join(item["message"] for item in errors))


class ProfilesManager:
    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        boot_file=None,
        clock=time.time,
        profile_dir: str | os.PathLike[str] | None = None,
        launcher_relative: str = _DEFAULT_LAUNCHER_RELATIVE,
    ) -> None:
        self.path = Path(path)
        self.default_boot_file = Path(boot_file) if boot_file is not None else None
        self.profile_dir = Path(profile_dir) if profile_dir is not None else self.path.parent / "boot-profiles"
        self.launcher_relative = launcher_relative
        self._clock = clock
        self._active_id = "default" if self.default_boot_file is not None else ""
        self.boot_file = self.default_boot_file
        if self.default_boot_file is not None:
            self._restore_active()

    # ---- active file and listing -----------------------------------------

    @property
    def active_profile_id(self) -> str:
        return self._active_id

    @property
    def active_id(self) -> str:
        return self.active_profile_id

    def configure_boot_file(self, boot_file: str | os.PathLike[str]) -> None:
        """Attach the helper's original boot file and restore a saved active profile."""
        target = Path(boot_file)
        if self.default_boot_file is None:
            self.default_boot_file = target
            self.boot_file = target
            self._active_id = "default"
            self._restore_active()
            return
        if self._active_id in {"", "default"}:
            self.default_boot_file = target
            self.boot_file = target
            self._active_id = "default"
            self._restore_active()

    def list(self) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        default = self._default_profile()
        if default is not None:
            profiles.append(default)
        profiles.extend(self._presets())
        profiles.extend(self._custom())
        for profile in profiles:
            active = profile["id"] == self._active_id
            profile["active"] = active
            profile["isActive"] = active
        return profiles

    all = list

    def get(self, profile_id: str) -> dict[str, Any] | None:
        for profile in self.list():
            if profile["id"] == profile_id:
                return profile
        return None

    def profile_path(self, profile_id: str) -> Path:
        if not _PROFILE_ID_RE.fullmatch(str(profile_id)):
            raise ProfileError(f"invalid profile id: {profile_id}")
        return self.profile_dir / f"{profile_id}.ps1"

    # ---- create, update, delete ------------------------------------------

    def create(self, name: str, description: str, settings: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise ProfileValidationError([{"field": "name", "message": "Profile name is required"}])
        if not isinstance(description, str):
            raise ProfileValidationError([{"field": "description", "message": "Description must be text"}])
        if not isinstance(settings, dict):
            raise ProfileValidationError([{"field": "settings", "message": "must be an object"}])
        errors = validate_settings(settings, ceilings_only=True)  # a profile may belong to another model
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
            "label": "profile",
            "kind": "profile",
            "settings": dict(settings),
            "bootFile": self._relative_profile_file(profile_id),
        }
        self._write_profile_file(profile)
        profiles.append(profile)
        self._write_custom(profiles)
        return self._profile_payload(profile)

    def update(
        self,
        profile_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a file-backed profile and rewrite its runnable file."""
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        if profile.get("isDefault") or profile.get("isPreset"):
            raise ProfileError("only saved profiles can be changed")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ProfileValidationError([{"field": "name", "message": "Profile name is required"}])
        if description is not None and not isinstance(description, str):
            raise ProfileValidationError([{"field": "description", "message": "Description must be text"}])
        next_settings = dict(profile.get("settings") or {}) if settings is None else settings
        if not isinstance(next_settings, dict):
            raise ProfileValidationError([{"field": "settings", "message": "must be an object"}])
        errors = validate_settings(next_settings, ceilings_only=True)
        if errors:
            raise ProfileValidationError(errors)

        profiles = self._custom()
        for item in profiles:
            if item["id"] != profile_id:
                continue
            if name is not None:
                item["name"] = name.strip()
            if description is not None:
                item["description"] = description
            item["settings"] = dict(next_settings)
            item["bootFile"] = self._relative_profile_file(profile_id)
            self._write_profile_file(item)
            self._write_custom(profiles)
            return self._profile_payload(item)
        raise KeyError(profile_id)

    def sync_active(self, settings: dict[str, Any]) -> None:
        """Keep the JSON snapshot and generated file in step with an edited active profile."""
        if not self._active_id or self._active_id == "default":
            return
        profiles = self._custom()
        for item in profiles:
            if item["id"] == self._active_id:
                item["settings"] = dict(settings)
                item["bootFile"] = self._relative_profile_file(item["id"])
                self._write_profile_file(item)
                self._write_custom(profiles)
                return

    def delete(self, profile_id: str) -> dict[str, Any]:
        if profile_id == "default":
            return {"deleted": False, "id": profile_id}
        if any(item["id"] == profile_id for item in self._presets()):
            return {"deleted": False, "id": profile_id}

        profiles = self._custom()
        kept = [item for item in profiles if item["id"] != profile_id]
        deleted = len(kept) != len(profiles)
        if not deleted:
            return {"deleted": False, "id": profile_id}

        if self._active_id == profile_id:
            self._active_id = "default" if self.default_boot_file is not None else ""
            self.boot_file = self.default_boot_file
        self._write_custom(kept)
        try:
            self.profile_path(profile_id).unlink()
        except FileNotFoundError:
            pass
        result = {"deleted": True, "id": profile_id}
        if self.default_boot_file is not None:
            result.update(
                {
                    "activeProfileId": self._active_id,
                    "bootFilePath": str(self.boot_file) if self.boot_file is not None else "",
                }
            )
        return result

    # ---- activation and old preset behavior -------------------------------

    def activate(self, profile_id: str) -> dict[str, Any]:
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        if profile.get("isPreset"):
            raise ProfileError("preset profiles are applied to the active file; they are not file-backed")

        if profile.get("isDefault"):
            if self.default_boot_file is None:
                raise ProfileError("no default boot file configured")
            target = self.default_boot_file
        else:
            target = self.profile_path(profile_id)
            if not target.is_file():
                self._write_profile_file(profile)

        # Parse the target before changing the pointer, so an invalid profile cannot become active.
        try:
            settings = BootFile(target).load()
        except (BootParseError, OSError) as exc:
            raise ProfileError(f"profile boot file could not be read: {exc}") from exc
        self.boot_file = target
        self._active_id = profile_id
        self._write_custom(self._custom())
        return {
            "activated": True,
            "profileId": profile_id,
            "bootFilePath": str(target),
            "settings": settings,
        }

    def apply(self, profile_id: str, boot_file=None) -> dict[str, Any]:
        """Apply a preset to the active file, or activate a saved file-backed profile.

        ``boot_file`` is retained for callers of the original manager API: when supplied, even a
        custom profile is written to that explicit target instead of changing the active pointer.
        """
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        if boot_file is None and not profile.get("isPreset"):
            activated = self.activate(profile_id)
            return {
                "applied": True,
                "activated": True,
                "profileId": profile_id,
                "bootFilePath": activated["bootFilePath"],
                "settings": activated["settings"],
            }

        path = Path(boot_file) if boot_file is not None else self.boot_file
        if path is None:
            raise ProfileError("no boot file configured")
        try:
            saved = BootFile(path).save(dict(profile["settings"]))
        except BootValidationError:
            raise
        return {
            "applied": True,
            "activated": False,
            "profileId": profile_id,
            "backupPath": str(BootFile(path).backup_path),
            "bootFilePath": str(path),
            "settings": saved,
        }

    # ---- file and JSON helpers --------------------------------------------

    def _presets(self) -> list[dict[str, Any]]:
        # Return fresh objects so a route caller cannot mutate the process-wide constants.
        out = json.loads(json.dumps(PRESET_PROFILES))
        for item in out:
            item["bootFile"] = ""
            item["modelFolder"] = self._model_folder(item.get("settings", {}))
        return out

    def _default_profile(self) -> dict[str, Any] | None:
        if self.default_boot_file is None:
            return None
        try:
            settings = BootFile(self.default_boot_file).load()
        except (BootParseError, OSError):
            settings = {}
        return {
            "id": "default",
            "name": f"Default ({self.default_boot_file.name})",
            "description": "The helper's original start-up file",
            "isPreset": False,
            "isDefault": True,
            "label": "default",
            "kind": "default",
            "bootFile": str(self.default_boot_file),
            "settings": settings,
            "modelFolder": self._model_folder(settings),
        }

    def _custom(self) -> list[dict[str, Any]]:
        _, items = self._read_store()
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict) or item.get("isPreset"):
                continue
            profile_id = item.get("id")
            if not isinstance(profile_id, str) or not _PROFILE_ID_RE.fullmatch(profile_id):
                continue
            if not isinstance(item.get("settings"), dict):
                continue
            out.append(
                {
                    "id": profile_id,
                    "name": str(item.get("name", "")),
                    "description": str(item.get("description", "")),
                    "isPreset": False,
                    "label": "profile",
                    "kind": "profile",
                    "settings": dict(item["settings"]),
                    "bootFile": self._relative_profile_file(profile_id),
                    "modelFolder": self._model_folder(item["settings"]),
                }
            )
        return out

    def _profile_payload(self, profile: dict[str, Any]) -> dict[str, Any]:
        payload = dict(profile)
        payload["bootFile"] = self._relative_profile_file(profile["id"])
        payload["modelFolder"] = self._model_folder(payload.get("settings", {}))
        return payload

    def _write_profile_file(self, profile: dict[str, Any]) -> None:
        path = self.profile_path(str(profile["id"]))
        text = render_boot_script(
            dict(profile.get("settings") or {}),
            launcher_relative=self.launcher_relative,
            title=str(profile.get("name") or "FreeToken profile"),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)

    def _read_store(self) -> tuple[str | None, list[Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                document = json.load(fh)
        except (FileNotFoundError, OSError, ValueError, TypeError):
            return None, []
        if isinstance(document, dict):
            active = document.get("active")
            items = document.get("profiles", [])
            return (str(active) if isinstance(active, str) else None), items if isinstance(items, list) else []
        if isinstance(document, list):
            return None, document
        return None, []

    def _write_custom(self, profiles: list[dict[str, Any]], active: Any = _UNSET) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if active is _UNSET:
            active = self._active_id or None
        document: dict[str, Any] = {
            "profiles": [
                {
                    "id": str(item["id"]),
                    "name": str(item.get("name", "")),
                    "description": str(item.get("description", "")),
                    "isPreset": False,
                    "label": "profile",
                    "kind": "profile",
                    "settings": dict(item.get("settings") or {}),
                    "bootFile": self._relative_profile_file(str(item["id"])),
                }
                for item in profiles
                if isinstance(item, dict) and item.get("id")
            ]
        }
        if self.default_boot_file is not None:
            document["active"] = str(active or "default")
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(document, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, self.path)

    def _restore_active(self) -> None:
        if self.default_boot_file is None:
            return
        active, _ = self._read_store()
        if not active or active == "default":
            self._active_id = "default"
            self.boot_file = self.default_boot_file
            return
        if not _PROFILE_ID_RE.fullmatch(active):
            self._active_id = "default"
            self.boot_file = self.default_boot_file
            return
        profile = next((item for item in self._custom() if item["id"] == active), None)
        if profile is None:
            self._active_id = "default"
            self.boot_file = self.default_boot_file
            return
        target = self.profile_path(active)
        try:
            if not target.is_file():
                self._write_profile_file(profile)
            BootFile(target).load()
        except (OSError, BootParseError, ValueError):
            self._active_id = "default"
            self.boot_file = self.default_boot_file
            return
        self._active_id = active
        self.boot_file = target

    def _relative_profile_file(self, profile_id: str) -> str:
        relative = Path("boot-profiles") / f"{profile_id}.ps1"
        return str(relative)

    @staticmethod
    def _model_folder(settings: dict[str, Any]) -> str:
        value = settings.get("ModelPath", "") if isinstance(settings, dict) else ""
        text = str(value or "").rstrip("\\/")
        if not text:
            return ""
        return text.replace("\\", "/").rsplit("/", 1)[-1]


ProfileManager = ProfilesManager


__all__ = [
    "PRESET_PROFILES",
    "ProfileError",
    "ProfileManager",
    "ProfileValidationError",
    "ProfilesManager",
]

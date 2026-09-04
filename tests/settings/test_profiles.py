from __future__ import annotations

import json
from pathlib import Path

from freetoken.daemon.settings.profiles_manager import ProfilesManager


def test_profiles_include_presets_and_support_crud(tmp_path):
    manager = ProfilesManager(tmp_path / "boot-profiles.json")
    profiles = manager.list()
    assert {item["id"] for item in profiles} >= {"profile-a", "profile-b"}
    created = manager.create("Profile C", "SSD", {"KVPark": "ssd", "MoECacheSize": 4188})
    assert created["isPreset"] is False
    assert manager.get(created["id"])["name"] == "Profile C"
    assert manager.delete(created["id"]) == {"deleted": True, "id": created["id"]}
    assert manager.get(created["id"]) is None
    assert json.loads((tmp_path / "boot-profiles.json").read_text())


def test_profile_apply_updates_boot_file(tmp_path):
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -Port 2020\n", encoding="utf-8")
    manager = ProfilesManager(tmp_path / "boot-profiles.json")
    profile = manager.create("Test", "", {"Port": 2021})
    result = manager.apply(profile["id"], boot)
    assert result["applied"] is True
    assert result["profileId"] == profile["id"]
    assert "-Port 2021" in boot.read_text(encoding="utf-8")


def test_profile_store_recovers_from_missing_file(tmp_path):
    manager = ProfilesManager(tmp_path / "missing.json")
    assert manager.list()[0]["id"] == "profile-a"
    assert not (tmp_path / "missing.json").exists()


def test_profile_delete_does_not_delete_presets(tmp_path):
    manager = ProfilesManager(tmp_path / "boot-profiles.json")
    assert manager.delete("profile-a") == {"deleted": False, "id": "profile-a"}
    assert manager.get("profile-a") is not None

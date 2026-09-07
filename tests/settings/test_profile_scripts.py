from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.boot_parser import BootFile
from freetoken.daemon.settings.profile_scripts import render_boot_script
from freetoken.daemon.settings.process_manager import ProcessManager
from freetoken.daemon.settings.profiles_manager import ProfilesManager


def _qwen_settings() -> dict[str, object]:
    return {
        "ModelPath": r"D:\Models\Qwen3.8-Flash-Next-NVFP4",
        "Port": 2020,
        "ContextTokens": 262144,
        "KVCacheTokens": 262144,
        "MaxRunningRequests": 4,
        "MoECacheSize": 4188,
        "GpuOwnedLayers": "auto:3",
        "CudaGraphMaxBS": 4,
        "KVDtype": "bf16",
        "DesktopPython": r"(Join-Path $env:LOCALAPPDATA 'FreeToken\venv\Scripts\python.exe')",
        "DenseQuant": "int8",
        "EmbedHost": True,
        "EnableVision": True,
        "VisionPackagesPath": r"D:\FreeToken\vision-packages",
        "VisionExecution": "layer-stream",
        "VisionWeights": "mmap",
        "ExpertLoad": "parallel",
        "EnableCacheReport": True,
        "CollectRoutingStats": True,
        "KVPark": "ssd",
        "KVParkIdleMs": 5000,
        "KVParkMinTokens": 8192,
        "KVParkRAMGiB": 2.0,
        "KVParkSSDDir": r"D:\FreeToken\kv-park",
        "KVParkSSDGiB": 32.0,
        "KVParkWindowMiB": 256,
        "MoEVramReserveBytes": -1,
        "MoECacheHeadroomBytes": -1,
        "MemoryGovernor": True,
        "GovernorVRAMFreeGB": 1.5,
        "GovernorRAMFreeGB": 4.0,
        "FREETOKEN_MTP_SPECULATE": "0",
        "FREETOKEN_MTP_RESIDENT": "0",
        "FREETOKEN_MTP_SHADOW": "0",
        "FREETOKEN_MTP_SPEC_GRAPH": "0",
        "FREETOKEN_MTP_SPEC_DEPTH": 3,
        "FREETOKEN_MTP_SPEC_CONF_CUT": 0.65,
        "FREETOKEN_MTP_SPEC_MIN_EMITTED": 2.4,
        "FREETOKEN_MTP_SPEC_COST_AWARE": "1",
    }


def _dense_settings() -> dict[str, object]:
    return {
        "ModelPath": r"D:\Models\Llama-dense",
        "Port": 2022,
        "ContextTokens": 131072,
        "KVCacheTokens": 0,
        "MaxRunningRequests": 2,
        "MoECacheSize": 0,
        "GpuOwnedLayers": "",
        "CudaGraphMaxBS": -1,
        "KVDtype": "bf16",
        "DesktopPython": r"C:\Python\python.exe",
        "DenseQuant": "",
        "EmbedHost": False,
        "EnableVision": False,
        "VisionPackagesPath": r"D:\FreeToken\vision-packages",
        "VisionExecution": "gpu",
        "VisionWeights": "ram",
        "ExpertLoad": "serial",
        "EnableCacheReport": False,
        "CollectRoutingStats": False,
        "KVPark": "off",
        "KVParkIdleMs": 0,
        "KVParkMinTokens": 8192,
        "KVParkRAMGiB": 2.0,
        "KVParkSSDDir": r"D:\FreeToken\kv-park",
        "KVParkSSDGiB": 32.0,
        "KVParkWindowMiB": 256,
        "MoEVramReserveBytes": -1,
        "MoECacheHeadroomBytes": -1,
        "MemoryGovernor": False,
        "GovernorVRAMFreeGB": 1.5,
        "GovernorRAMFreeGB": 4.0,
        "FREETOKEN_MTP_SPECULATE": "0",
        "FREETOKEN_MTP_RESIDENT": "0",
        "FREETOKEN_MTP_SHADOW": "0",
        "FREETOKEN_MTP_SPEC_GRAPH": "0",
        "FREETOKEN_MTP_SPEC_DEPTH": 5,
        "FREETOKEN_MTP_SPEC_CONF_CUT": 0.8,
        "FREETOKEN_MTP_SPEC_MIN_EMITTED": 2.4,
        "FREETOKEN_MTP_SPEC_COST_AWARE": "1",
    }


def test_qwen_profile_script_round_trips_through_boot_parser(tmp_path: Path) -> None:
    path = tmp_path / "qwen.ps1"
    path.write_text(render_boot_script(_qwen_settings(), title="Qwen profile"), encoding="utf-8")

    assert BootFile(path).load() == _qwen_settings()
    assert "-EnableVision `" in path.read_text(encoding="utf-8")
    assert "$env:FREETOKEN_MTP_SPECULATE = '0'" in path.read_text(encoding="utf-8")
    assert "$env:FREETOKEN_MTP_SPEC_DEPTH = '3'" in path.read_text(encoding="utf-8")
    assert "$env:FREETOKEN_MTP_SPEC_CONF_CUT = '0.65'" in path.read_text(encoding="utf-8")


def test_dense_profile_script_round_trips_and_omits_false_switches(tmp_path: Path) -> None:
    path = tmp_path / "dense.ps1"
    path.write_text(render_boot_script(_dense_settings(), title="Dense profile"), encoding="utf-8")
    text = path.read_text(encoding="utf-8")

    assert BootFile(path).load() == _dense_settings()
    assert "-EmbedHost" not in text
    assert "-EnableVision" not in text
    assert "-EnableCacheReport" not in text
    assert "-MemoryGovernor" not in text
    assert "$launcher = Join-Path $PSScriptRoot '..\\scripts\\start-freetoken-windows.ps1'" in text


def test_default_profile_is_first_and_cannot_be_deleted(tmp_path: Path) -> None:
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text(render_boot_script(_qwen_settings(), title="Default"), encoding="utf-8")
    manager = ProfilesManager(tmp_path / "boot-profiles.json", boot_file=boot)

    profiles = manager.list()
    assert profiles[0]["id"] == "default"
    assert profiles[0]["isDefault"] is True
    assert profiles[0]["bootFile"] == str(boot)
    assert profiles[0]["settings"] == BootFile(boot).load()
    assert manager.delete("default") == {"deleted": False, "id": "default"}
    assert manager.list()[0]["id"] == "default"
    assert all(item.get("label") == "preset" for item in profiles[1:3])


def _process(tmp_path: Path) -> ProcessManager:
    return ProcessManager(
        boot_file=tmp_path / "boot-2020.ps1",
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *args, **kwargs: None,
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: None,
        poll_interval=0,
    )


def test_activate_switches_all_pointers_and_survives_new_app(tmp_path: Path) -> None:
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text(render_boot_script(_qwen_settings(), title="Default"), encoding="utf-8")
    store = tmp_path / "boot-profiles.json"
    manager = ProfilesManager(store, boot_file=boot)
    created = manager.create("Night mode", "saved", {"Port": 2023, "ModelPath": r"D:\Models\Night"})
    profile_path = tmp_path / "boot-profiles" / f"{created['id']}.ps1"
    assert profile_path.is_file()

    app = create_app(
        boot_file=boot,
        process_manager=_process(tmp_path),
        profiles=ProfilesManager(store, boot_file=boot),
        static_path=tmp_path / "missing.html",
    )
    with TestClient(app) as client:
        activated = client.post(f"/api/profiles/{created['id']}/activate")
        assert activated.status_code == 200, activated.text
        settings = client.get("/api/settings").json()
        assert settings["bootFilePath"] == str(profile_path)
        listed = client.get("/api/profiles").json()["profiles"]
        assert listed[0]["id"] == "default" and listed[0]["isDefault"] is True
        assert next(item for item in listed if item["id"] == created["id"])["active"] is True
        assert app.state.process_manager.boot_file == profile_path

    document = json.loads(store.read_text(encoding="utf-8"))
    assert document["active"] == created["id"]

    restored_manager = ProfilesManager(store, boot_file=boot)
    restored_app = create_app(
        boot_file=boot,
        process_manager=_process(tmp_path),
        profiles=restored_manager,
        static_path=tmp_path / "missing.html",
    )
    with TestClient(restored_app) as client:
        assert client.get("/api/settings").json()["bootFilePath"] == str(profile_path)
        assert restored_app.state.process_manager.boot_file == profile_path


def test_profile_update_rewrites_its_startup_file(tmp_path: Path) -> None:
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text(render_boot_script(_qwen_settings(), title="Default"), encoding="utf-8")
    manager = ProfilesManager(tmp_path / "boot-profiles.json", boot_file=boot)
    created = manager.create("Day mode", "saved", {"Port": 2023, "ModelPath": r"D:\Models\Day"})

    manager.update(created["id"], settings={"Port": 2024, "ModelPath": r"D:\Models\Updated"})

    profile_path = tmp_path / "boot-profiles" / f"{created['id']}.ps1"
    assert BootFile(profile_path).load()["Port"] == 2024
    assert BootFile(profile_path).load()["ModelPath"] == r"D:\Models\Updated"

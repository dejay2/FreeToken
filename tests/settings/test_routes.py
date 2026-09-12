from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.boot_parser import BootFile
from freetoken.daemon.settings.profiles_manager import ProfilesManager
from freetoken.daemon.settings.process_manager import ProcessManager


def make_client(tmp_path):
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -Port 2020\n", encoding="utf-8")
    proc = ProcessManager(
        boot_file=boot,
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *a, **k: None,
        readiness=lambda: {"state": "serving", "geometry": {"num_pages": 4096}},
        sleep=lambda _: None,
        poll_interval=0,
    )
    app = create_app(
        boot_file=boot,
        process_manager=proc,
        profiles=ProfilesManager(tmp_path / "boot-profiles.json"),
        log_path=tmp_path / "server.log",
        static_path=tmp_path / "missing-index.html",
    )
    return TestClient(app), boot


def test_lifecycle_route_freezes_full_snapshot_and_stop_rejects_settings(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"architectures":["Qwen4ExpForConditionalGeneration"],'
        '"model_type":"qwen4_exp","num_hidden_layers":48,"num_experts":512,'
        '"max_position_embeddings":262144,"hidden_size":16,"moe_intermediate_size":32,'
        '"quantization_config":{"quant_algo":"NVFP4"}}',
        encoding="utf-8",
    )
    boot = tmp_path / "boot-2020.ps1"
    # -GpuOwnedLayers 0: this test starts with a toy 120-slot cache, and the catalogue default
    # ("auto" = 6 layers) would charge 6 x 512 slots against it -- the pair the validator now
    # refuses (dials._expert_slot_charge_errors).
    boot.write_text(
        f"& $launcher `\n    -ModelPath '{model}' `\n    -GpuOwnedLayers 0 `\n    -Port 2020\n",
        encoding="utf-8",
    )

    class RecordingManager:
        def __init__(self):
            self.boot_file = boot
            self.log_path = tmp_path / "server.log"
            self.port = 2020
            self.calls = []

        def start(self, action="start", *, settings=None, force=False):
            self.calls.append((action, settings, force))
            return "job-recorded"

        def job(self, job_id):
            return {"jobId": job_id, "stage": "booting"}

    manager = RecordingManager()
    app = create_app(
        boot_file=boot,
        process_manager=manager,
        profiles=ProfilesManager(tmp_path / "boot-profiles.json"),
        static_path=tmp_path / "missing-index.html",
    )
    with TestClient(app) as client:
        started = client.post(
            "/api/server/start",
            json={"force": True, "settings": {"MoECacheSize": 120}},
        )
        assert started.status_code == 202, started.text
        assert manager.calls[0][0] == "start"
        assert manager.calls[0][1]["ModelPath"] == str(model)
        assert manager.calls[0][1]["MoECacheSize"] == 120
        assert manager.calls[0][2] is True
        rejected = client.post(
            "/api/server/stop",
            json={"settings": {"MoECacheSize": 120}},
        )
        assert rejected.status_code == 422


def test_lifecycle_override_does_not_gain_an_estimator_metadata_gate(tmp_path):
    missing_model = tmp_path / "model-does-not-exist"
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text(
        f"& $launcher `\n    -ModelPath '{missing_model}' `\n    -Port 2020\n",
        encoding="utf-8",
    )

    class RecordingManager:
        boot_file = boot
        log_path = tmp_path / "server.log"
        port = 2020

        def __init__(self):
            self.calls = []

        def start(self, action="start", *, settings=None, force=False):
            self.calls.append((action, settings, force))
            return "job-override"

        def job(self, job_id):
            return {"jobId": job_id, "stage": "booting"}

    manager = RecordingManager()
    app = create_app(
        boot_file=boot,
        process_manager=manager,
        profiles=ProfilesManager(tmp_path / "boot-profiles.json"),
        static_path=tmp_path / "missing-index.html",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/server/start",
            json={"force": True, "settings": {"MoECacheSize": 120}},
        )

    assert response.status_code == 202, response.text
    assert manager.calls[0][0] == "start"
    assert manager.calls[0][1]["MoECacheSize"] == 120
    assert manager.calls[0][2] is True


def test_settings_routes_return_metadata_and_save(tmp_path):
    client, boot = make_client(tmp_path)
    response = client.get("/api/settings")
    assert response.status_code == 200
    doc = response.json()
    assert doc["settings"]["Port"] == 2020
    assert any(dial["name"] == "KVDtype" for dial in doc["dials"])
    saved = client.put("/api/settings", json={"settings": {"Port": 2021}})
    assert saved.status_code == 200
    assert saved.json()["status"] == "saved"
    assert "-Port 2021" in boot.read_text(encoding="utf-8")


def test_profile_lifecycle_routes(tmp_path):
    client, _ = make_client(tmp_path)
    created = client.post("/api/profiles", json={"name": "Test", "description": "x", "settings": {"Port": 2021}})
    assert created.status_code == 201
    profile_id = created.json()["id"]
    assert any(item["id"] == profile_id for item in client.get("/api/profiles").json()["profiles"])
    applied = client.post(f"/api/profiles/{profile_id}/apply")
    assert applied.status_code == 200 and applied.json()["applied"] is True
    deleted = client.delete(f"/api/profiles/{profile_id}")
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True


def test_status_logs_and_lifecycle_job_routes(tmp_path):
    client, _ = make_client(tmp_path)
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["helper"]["version"] == "1.5.0"
    auto = status.json()["autoRestart"]
    assert auto["enabled"] is True and auto["gave_up"] is False and "restarts_last_hour" in auto
    saved = client.put("/api/settings", json={"settings": {"FREETOKEN_AUTO_RESTART": False}})
    assert saved.status_code == 200
    assert client.get("/api/status").json()["autoRestart"]["enabled"] is False, "a Save reaches the watchdog flag"
    client.put("/api/settings", json={"settings": {"FREETOKEN_AUTO_RESTART": True}})
    assert client.get("/api/logs?limit=10").json()["lines"] == []
    response = client.post("/api/server/start", json={})
    assert response.status_code == 202
    job_id = response.json()["jobId"]
    job = client.get(f"/api/server/jobs/{job_id}")
    assert job.status_code == 200
    assert job.json()["jobId"] == job_id


def test_validation_failure_has_spec_shape(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.put("/api/settings", json={"settings": {"ContextTokens": 500000}})
    assert response.status_code == 422
    assert response.json()["detail"][0]["field"] == "ContextTokens"


def test_unknown_job_and_action_are_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/api/server/jobs/nope").status_code == 404
    assert client.post("/api/server/reboot").status_code == 422


def test_root_is_plain_not_found_until_frontend_arrives(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/").status_code == 404


def test_model_catalog_follows_the_active_profile_model_path(tmp_path):
    current = tmp_path / "current-model"
    alternate_root = tmp_path / "alternate-library"
    alternate = alternate_root / "alternate-model"
    current.mkdir()
    alternate.mkdir(parents=True)
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text(
        f"& $launcher `\n    -ModelPath '{current}' `\n    -Port 2020\n",
        encoding="utf-8",
    )
    store = tmp_path / "boot-profiles.json"
    profiles = ProfilesManager(store, boot_file=boot)
    proc = ProcessManager(
        boot_file=boot,
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *a, **k: None,
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: None,
        poll_interval=0,
    )
    app = create_app(boot_file=boot, process_manager=proc, profiles=profiles, static_path=tmp_path / "missing.html")

    with TestClient(app) as client:
        created = client.post(
            "/api/profiles",
            json={"name": "Alternate", "description": "", "settings": {"ModelPath": str(alternate)}},
        )
        assert created.status_code == 201, created.text
        activated = client.post(f"/api/profiles/{created.json()['id']}/activate")
        assert activated.status_code == 200, activated.text
        assert app.state.models_dir == alternate_root
        assert app.state.download_manager.models_dir == alternate_root
        listed = client.get("/api/models").json()["models"]
        assert [item["path"] for item in listed] == [str(alternate)]


def test_saving_or_starting_an_under_floor_slot_pair_is_refused(tmp_path):
    """The 2026-09-07 11:22 BST live failure, as the page would have seen it: 4288 expert
    slots with 8 layers on the card leaves 192 for the streaming layers against a floor of
    1024, and the boot died immediately. Both the save and the start now refuse it first."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"architectures":["Qwen4ExpForConditionalGeneration"],'
        '"model_type":"qwen4_exp","num_hidden_layers":48,"num_experts":512,'
        '"max_position_embeddings":262144,"hidden_size":2560,"moe_intermediate_size":640,'
        '"quantization_config":{"quant_algo":"NVFP4"}}',
        encoding="utf-8",
    )
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text(
        f"& $launcher `\n    -ModelPath '{model}' `\n    -MoECacheSize 6750 `\n"
        f"    -GpuOwnedLayers auto:6 `\n    -Port 2020\n",
        encoding="utf-8",
    )
    proc = ProcessManager(
        boot_file=boot,
        stop_script=tmp_path / "stop.ps1",
        log_path=tmp_path / "server.log",
        lock_path=tmp_path / "gpu.lock",
        runner=lambda *a, **k: None,
        readiness=lambda: {"state": "serving"},
        sleep=lambda _: None,
        poll_interval=0,
    )
    app = create_app(
        boot_file=boot,
        process_manager=proc,
        profiles=ProfilesManager(tmp_path / "boot-profiles.json"),
        log_path=tmp_path / "server.log",
        static_path=tmp_path / "missing-index.html",
    )
    client = TestClient(app)

    refused = client.put(
        "/api/settings",
        json={"settings": {"MoECacheSize": 4288, "GpuOwnedLayers": "auto:8"}},
    )
    assert refused.status_code == 422
    detail = refused.json()["detail"][0]
    assert detail["field"] == "MoECacheSize" and detail["minimum"] == 5120
    assert "leaving 192" in detail["message"] and "5,120 or more" in detail["message"]
    assert "-MoECacheSize 6750" in boot.read_text(encoding="utf-8"), "nothing was written"

    # a patch that raises only the layer count is checked against the saved slot total
    layers_only = client.put("/api/settings", json={"settings": {"GpuOwnedLayers": "auto:12"}})
    assert layers_only.status_code == 422
    assert layers_only.json()["detail"][0]["minimum"] == 7168

    # and the same pair cannot be smuggled past the lifecycle route either
    started = client.post(
        "/api/server/start",
        json={"force": True, "settings": {"MoECacheSize": 4288, "GpuOwnedLayers": "auto:8"}},
    )
    assert started.status_code == 422
    assert started.json()["detail"][0]["minimum"] == 5120

    # the pair the engine accepts saves normally
    saved = client.put(
        "/api/settings",
        json={"settings": {"MoECacheSize": 5120, "GpuOwnedLayers": "auto:8"}},
    )
    assert saved.status_code == 200, saved.text
    assert "-MoECacheSize 5120" in boot.read_text(encoding="utf-8")

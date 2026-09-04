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
    assert client.get("/api/status").status_code == 200
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

from __future__ import annotations

from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.browse import is_model_folder, list_directory, resolve_start
from freetoken.daemon.settings.process_manager import ProcessManager
from freetoken.daemon.settings.profiles_manager import ProfilesManager


def _tree(tmp_path):
    model = tmp_path / "Some-Model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model-00001.safetensors").write_bytes(b"\0")
    plain = tmp_path / "Pictures"
    plain.mkdir()
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    return model, plain


def test_model_folder_needs_config_and_weights(tmp_path):
    model, plain = _tree(tmp_path)
    assert is_model_folder(model) is True
    assert is_model_folder(plain) is False
    (plain / "config.json").write_text("{}", encoding="utf-8")
    assert is_model_folder(plain) is False, "config.json alone is not a model"


def test_list_directory_marks_models_and_hides_files_for_folder_kinds(tmp_path):
    model, plain = _tree(tmp_path)
    listing = list_directory(str(tmp_path), "model")
    names = {entry["name"]: entry for entry in listing["entries"]}
    assert set(names) == {"Some-Model", "Pictures"}, "files and dot-folders stay out of a folder listing"
    assert names["Some-Model"]["isModel"] is True
    assert names["Pictures"]["isModel"] is False
    assert listing["isModel"] is False
    assert listing["parent"] == str(tmp_path.resolve().parent)

    files = list_directory(str(tmp_path), "file")
    assert any(entry["name"] == "notes.txt" and entry["kind"] == "file" for entry in files["entries"])

    inside = list_directory(str(model), "model")
    assert inside["isModel"] is True


def test_unresolvable_paths_fall_back_to_the_drive_list(tmp_path):
    assert resolve_start("") is None
    assert resolve_start("(Join-Path $env:LOCALAPPDATA 'x')") is None
    assert resolve_start("$visionPackages") is None
    listing = list_directory("Q:\\does\\not\\exist\\anywhere\\really", "folder")
    assert listing["path"] == ""
    assert listing["entries"] and all(entry["kind"] == "dir" for entry in listing["entries"])
    # A missing leaf inside an existing folder opens that folder, which is what a typed-in
    # but not yet created parking directory looks like.
    assert resolve_start(str(tmp_path / "kv-park-not-yet-made")) == tmp_path.resolve()


def test_browse_route_lists_and_rejects_bad_kind(tmp_path):
    _tree(tmp_path)
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -Port 2020\n", encoding="utf-8")
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
    response = client.get("/api/browse", params={"path": str(tmp_path), "kind": "model"})
    assert response.status_code == 200
    body = response.json()
    assert {entry["name"] for entry in body["entries"]} == {"Some-Model", "Pictures"}
    assert client.get("/api/browse", params={"kind": "everything"}).status_code == 422
    doc = client.get("/api/settings").json()
    assert {group["name"] for group in doc["groups"]} >= {dial["group"] for dial in doc["dials"]}

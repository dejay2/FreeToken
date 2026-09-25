"""Add and remove models on the control panel (Stage B). Review focus 2, 3 and 5 live here."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.boot_parser import BootFile
from freetoken.daemon.settings.download import DownloadManager
from freetoken.daemon.settings.panel import PanelService
from freetoken.daemon.settings.pi_sync import PROVIDER, PiSync
from freetoken.daemon.settings.process_manager import ProcessManager
from freetoken.daemon.settings.profiles_manager import ProfilesManager
from freetoken.daemon.settings.registry import RegistryStore, find_model
from freetoken.daemon.settings.swap_config import SwapConfigWriter, extract_model_blocks, render_config
from tests.settings.registry_fixtures import five
from tests.settings.test_download_add import Hub
from tests.settings.test_model_detect import write_folder, write_v2, write_v3
from tests.settings.test_panel_routes import FakeEstimates, FakeSwitcher, checker
from tests.settings.test_pi_sync import write_pi

GIB = 1024 ** 3
FIVE_IDS = [m["id"] for m in five()["models"]]


@pytest.fixture
def box(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    ninfer, models = home / "ninfer-work" / "models", home / "models"
    ninfer.mkdir(parents=True)
    models.mkdir(parents=True)
    write_v2(ninfer / "quasar_27b_nvfp4.ninfer")
    write_v3(ninfer / "fable_27b_nvfp4.ninfer")
    write_v3(ninfer / "twin_nvfp4.ninfer")
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -KVDtype 'fp8' `\n    -Port 2020\n", encoding="utf-8")
    cfg = tmp_path / "llama-swap" / "config.yaml"
    cfg.parent.mkdir()
    binary = tmp_path / "llama-swap-bin"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    writer = SwapConfigWriter(cfg, binary=binary, runner=checker())
    switcher = FakeSwitcher(cfg)
    profiles = ProfilesManager(tmp_path / "boot-profiles.json", boot_file=boot)
    store = RegistryStore(tmp_path / "freetoken" / "registry.json")
    pi_dir = write_pi(tmp_path / "pi")
    hub = Hub({})
    downloads = DownloadManager(models, api_factory=hub, config_fetcher=lambda _: hub.config or {},
                                snapshot_downloader=hub.snapshot, pc_memory=64 * GIB, card_memory=32 * GIB,
                                disk_free=lambda _: 10 ** 12)
    service = PanelService(
        store=store, writer=writer, switcher=switcher, profiles=profiles,
        boot_file=lambda: BootFile(boot), default_boot=lambda: boot, estimate_service=FakeEstimates(),
        card_probe=lambda: {"totalBytes": 32 * GIB, "usedBytes": 2 * GIB}, windows_free_probe=lambda: 40 * GIB,
        spawn=lambda fn, *args: None, sleep=lambda _: None,
        downloads=downloads, pi=PiSync(pi_dir), add_roots={"folder": models, "ninfer": ninfer},
    )
    proc = ProcessManager(boot_file=boot, stop_script=tmp_path / "stop.ps1", log_path=tmp_path / "server.log",
                          lock_path=tmp_path / "gpu.lock", runner=lambda *a, **k: None,
                          readiness=lambda: {"state": "unreachable"}, sleep=lambda _: None, poll_interval=0)
    app = create_app(boot_file=boot, process_manager=proc, profiles=profiles, log_path=tmp_path / "server.log",
                     static_path=tmp_path / "missing.html", panel=service)
    store.save(five(), expected_revision=None)
    writer.write(render_config(store.load()[0], {}))
    return SimpleNamespace(client=TestClient(app), app=app, service=service, store=store, cfg=cfg, switcher=switcher,
                           profiles=profiles, pi=pi_dir, ninfer=ninfer, models=models, hub=hub, boot=boot, home=home)


def revision(box):
    return box.client.get("/api/panel/registry").json()["revision"]


def pi_ids(box):
    return [m["id"] for m in json.loads((box.pi / "models.json").read_text())["providers"][PROVIDER]["models"]]


def enabled(box):
    return json.loads((box.pi / "settings.json").read_text())["enabledModels"]


def add(box, path, **identity):
    body = {"revision": revision(box), "path": str(path), "id": "small_9b", "name": "Small 9B (NInfer)",
            "ramNeedGB": 6, **identity}
    return box.client.post("/api/panel/models", json=body)


def remove(box, model_id, **body):
    return box.client.post(f"/api/panel/models/{model_id}/remove", json={"revision": revision(box), **body})


def ids(box):
    return [m["id"] for m in box.store.load()[0]["models"]]


# ---- detect and add ----
def test_detect_says_what_a_path_is_and_suggests_an_identity(box):
    write_v3(box.ninfer / "small_9b.ninfer", parts=1)
    found = box.client.post("/api/panel/add/detect", json={"path": "~/ninfer-work/models/small_9b.ninfer"}).json()
    assert (found["kind"], found["runtime"], found["already"]) == ("ninfer", "ninfer-upstream", None)
    assert found["suggested"]["id"] == "small_9b" and len(found["files"]) == 2
    write_v2(box.ninfer / "quasar-27b.ninfer")
    clash = box.client.post("/api/panel/add/detect", json={"path": str(box.ninfer / "quasar-27b.ninfer")}).json()
    assert clash["suggested"]["id"] == "quasar-27b-2"
    known = box.client.post("/api/panel/add/detect", json={"path": str(box.ninfer / "quasar_27b_nvfp4.ninfer")}).json()
    assert known["runtime"] == "ninfer" and known["already"] == "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)"
    info = box.client.get("/api/panel/add/info").json()
    assert info == {"roots": {"folder": str(box.models), "ninfer": str(box.ninfer)}, "download": None}


def test_adding_a_ninfer_model_updates_the_list_the_switcher_and_pi(box):
    write_v3(box.ninfer / "small_9b.ninfer")
    answer = add(box, "~/ninfer-work/models/small_9b.ninfer")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert (body["status"], body["name"], body["pi"]["status"]) == ("added", "Small 9B (NInfer)", "updated")
    model = find_model(box.store.load()[0], "small_9b")
    assert (model["engine"], model["runtime"], model["artifact"]) == ("ninfer", "ninfer-upstream", "~/ninfer-work/models/small_9b.ninfer")
    assert model["ramNeedGB"] == 6 and model["idleMinutes"] is None and model["overrides"] == {}
    assert "small_9b" in extract_model_blocks(box.cfg.read_text())
    assert pi_ids(box)[-1] == "small_9b" and enabled(box)[-1] == f"{PROVIDER}/small_9b"
    pi_entry = json.loads((box.pi / "models.json").read_text())["providers"][PROVIDER]["models"][-1]
    assert pi_entry["contextWindow"] == 150000  # copied from quasar-27b, the NInfer neighbour
    row = next(r for r in box.client.get("/api/panel/models").json()["models"] if r["id"] == "small_9b")
    assert row["aliases"] == [] and row["runtime"] == "ninfer-upstream"


def test_the_server_checks_everything_again_at_save(box):
    (box.ninfer / "notes.txt").write_text("hi")
    refused = add(box, box.ninfer / "notes.txt")
    assert refused.status_code == 422 and refused.json()["code"] == "not_supported"
    again = add(box, box.ninfer / "quasar_27b_nvfp4.ninfer")
    assert again.status_code == 409 and again.json()["code"] == "already_added"
    write_v2(box.ninfer / "small_9b.ninfer")
    for identity, field in (({"id": "Bad Id"}, "add.id"), ({"id": "qwen3.8-flash-next-nvfp4"}, "add.id"),
                            ({"id": "quasar-27b"}, "add.id"), ({"name": ""}, "add.name"),
                            ({"ramNeedGB": "lots"}, "add.ramNeedGB"), ({"ramNeedGB": 900}, "add.ramNeedGB")):
        answer = add(box, box.ninfer / "small_9b.ninfer", **identity)
        assert answer.status_code == 422 and answer.json()["detail"][0]["field"] == field, (identity, answer.text)
    stale = box.client.post("/api/panel/models", json={"revision": "old", "path": str(box.ninfer / "small_9b.ninfer"),
                                                       "id": "small_9b", "name": "S", "ramNeedGB": 6})
    assert stale.json()["code"] == "stale_revision"
    assert ids(box) == FIVE_IDS and "small_9b" not in pi_ids(box)


def test_adding_works_while_the_switcher_state_is_unknown(box):
    box.switcher.up = False  # unknown: a model new to the list has no entry yet, so it cannot be loaded
    write_v2(box.ninfer / "small_9b.ninfer")
    assert add(box, box.ninfer / "small_9b.ninfer").status_code == 200


def test_a_small_freetoken_model_gets_the_defaults_fitted_to_it(box):
    folder = write_folder(box.models / "Tiny-Llama", max_position_embeddings=8192)
    answer = add(box, folder, id="tiny-llama", name="Tiny Llama (FreeToken)", ramNeedGB=2)
    assert answer.status_code == 200, answer.text
    assert find_model(box.store.load()[0], "tiny-llama")["overrides"] == {"ContextTokens": 8192}
    assert "8,192" in answer.json()["adjusted"][0]
    assert box.profiles.get("model-tiny-llama")["settings"]["ModelPath"] == str(folder)


def test_pi_out_of_reach_still_adds_and_says_so(box):
    for path in box.pi.iterdir():
        path.unlink()
    box.pi.rmdir()
    write_v2(box.ninfer / "small_9b.ninfer")
    body = add(box, box.ninfer / "small_9b.ninfer").json()
    assert body["status"] == "added" and body["pi"]["status"] == "not_updated"
    assert "small_9b" in ids(box)


# ---- remove ----
def test_removing_a_loaded_model_unloads_it_first(box):
    box.switcher.states = {"quasar-27b": "ready"}
    answer = remove(box, "quasar-27b")
    assert answer.status_code == 200, answer.text
    assert box.switcher.calls == [("unload", "quasar-27b")]
    assert "quasar-27b" not in ids(box) and "quasar-27b" not in extract_model_blocks(box.cfg.read_text())
    assert "quasar-27b" not in pi_ids(box) and f"{PROVIDER}/quasar-27b" not in enabled(box)
    assert (box.ninfer / "quasar_27b_nvfp4.ninfer").is_file(), "files stay unless asked"
    assert answer.json()["files"] is None and answer.json()["pi"]["notes"]  # quasar was Pi's default


def test_remove_is_refused_while_the_switcher_state_is_unknown(box):
    box.switcher.up = False
    before = box.store.load()[1]
    answer = remove(box, "quasar-27b")
    assert answer.status_code == 503 and answer.json()["code"] == "switcher_unknown"
    assert box.store.load()[1] == before and "quasar-27b" in pi_ids(box)


def test_a_failed_unload_removes_nothing(box):
    box.switcher.states, box.switcher.unload_ok = {"quasar-27b": "ready"}, False
    before, text = box.store.load()[1], box.cfg.read_text()
    answer = remove(box, "quasar-27b", deleteFiles=True)
    assert answer.status_code == 503 and answer.json()["code"] == "unload_failed"
    assert box.store.load()[1] == before and box.cfg.read_text() == text
    assert (box.ninfer / "quasar_27b_nvfp4.ninfer").is_file() and "quasar-27b" in pi_ids(box)


def test_delete_files_takes_the_entry_and_its_parts_and_nothing_else(box):
    write_v3(box.ninfer / "small_9b.ninfer", parts=2)
    assert add(box, box.ninfer / "small_9b.ninfer").status_code == 200
    answer = remove(box, "small_9b", deleteFiles=True)
    assert answer.status_code == 200, answer.text
    assert answer.json()["files"]["deleted"] is True and len(answer.json()["files"]["paths"]) == 3
    assert sorted(p.name for p in box.ninfer.iterdir()) == [
        "fable_27b_nvfp4.ninfer", "quasar_27b_nvfp4.ninfer", "twin_nvfp4.ninfer"]


def test_files_another_model_uses_are_never_deleted(box):
    doc, rev = box.store.load()
    doc["models"].append({**find_model(doc, "quasar-27b"), "id": "quasar-copy", "name": "QUASAR copy"})
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "quasar-copy", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_shared"
    assert "quasar-copy" in ids(box) and (box.ninfer / "quasar_27b_nvfp4.ninfer").is_file()


def test_parts_another_model_reads_are_never_deleted(box):
    write_v3(box.ninfer / "twin_nvfp4.ninfer", parts=1)          # twin is now split
    write_v3(box.ninfer / "copy.ninfer", part_names=["twin_nvfp4.ninfer.part-0001"])  # an entry reading twin's part
    doc, rev = box.store.load()
    doc["models"].append({**find_model(doc, "twin-27b"), "id": "twin-copy", "name": "Twin copy",
                          "artifact": "~/ninfer-work/models/copy.ninfer"})
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "twin-copy", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_shared"
    assert (box.ninfer / "twin_nvfp4.ninfer.part-0001").is_file() and (box.ninfer / "copy.ninfer").is_file()


def test_delete_files_refuses_a_folder_that_is_not_a_model(box):
    odd = box.models / "odd"
    odd.mkdir()
    (odd / "keep.txt").write_text("x")
    doc, rev = box.store.load()
    find_model(doc, "qwen3.8-flash")["artifact"] = "~/models/odd"
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "qwen3.8-flash", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_unsafe"
    assert (odd / "keep.txt").is_file() and "qwen3.8-flash" in ids(box)


def test_the_home_folder_is_never_deleted(box):
    (box.home / "config.json").write_text("{}")
    doc, rev = box.store.load()
    find_model(doc, "qwen3.8-flash")["artifact"] = "~/"
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "qwen3.8-flash", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_unsafe"
    assert (box.home / "config.json").is_file()


def test_removing_a_freetoken_model_drops_its_profile_and_moves_the_helper_off_it(box):
    box.service._ensure_profiles(box.store.load()[0])
    activated = box.client.post("/api/profiles/model-qwen3.8-flash/activate")
    assert activated.status_code == 200, activated.text
    assert Path(box.app.state.boot_file.path) != box.boot
    answer = remove(box, "qwen3.8-flash")
    assert answer.status_code == 200, answer.text
    assert box.profiles.get("model-qwen3.8-flash") is None
    assert Path(box.app.state.boot_file.path) == box.boot


def test_a_removed_model_leaves_no_hold_behind(box):
    box.switcher.states = {"quasar-27b": "ready"}
    view = box.client.get("/api/panel/views/model/quasar-27b").json()
    assert view["artifact"] == "~/ninfer-work/models/quasar_27b_nvfp4.ninfer"
    settings = {k: v for k, v in view["settings"].items() if not k.startswith("model.")}
    settings["draft-tokens"] = 5
    held = box.client.put("/api/panel/models/quasar-27b", json={
        "revision": view["revision"], "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "next-time"})
    assert held.json()["held"] == ["quasar-27b"]
    box.switcher.up, box.switcher.refused = False, True   # down: nothing is loaded, holds are kept
    assert remove(box, "quasar-27b").status_code == 200
    assert "quasar-27b" not in json.loads(box.service.holds_path.read_text())


# ---- downloads through the panel ----
def wait_job(box, job_id):
    for _ in range(400):
        job = box.client.get(f"/api/panel/add/downloads/{job_id}").json()
        if job["stage"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.005)
    raise AssertionError(job)


def test_a_link_downloads_checks_and_adds(box, tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    write_v3(source / "small_9b.ninfer", parts=1)
    box.hub.files = {p.name: p.read_bytes() for p in sorted(source.iterdir())}
    plan = box.client.post("/api/panel/add/plan", json={"link": "https://huggingface.co/owner/small-9b"}).json()
    assert plan["engine"] == "ninfer" and plan["target"] == str(box.ninfer / "small_9b.ninfer")
    job = box.client.post("/api/panel/add/downloads", json={"link": "owner/small-9b"}).json()
    done = wait_job(box, job["id"])
    assert done["stage"] == "done" and len(done["verified"]) == 2, done
    assert box.client.get("/api/panel/add/info").json()["download"]["id"] == job["id"]
    found = box.client.post("/api/panel/add/detect", json={"path": done["resultPath"]}).json()
    assert found["runtime"] == "ninfer-upstream" and found["already"] is None
    assert add(box, done["resultPath"], id=found["suggested"]["id"]).status_code == 200


def test_a_bad_checksum_leaves_nothing_behind(box):
    box.hub.files = {"small_9b.ninfer": b"x" * 64, "SHA256SUMS": f"{'0' * 64}  small_9b.ninfer\n".encode()}
    job = box.client.post("/api/panel/add/downloads", json={"link": "owner/small-9b"}).json()
    done = wait_job(box, job["id"])
    assert done["stage"] == "failed" and "checksum" in done["error"]
    assert not (box.ninfer / "small_9b.ninfer").exists()
    assert not [p for p in box.ninfer.iterdir() if p.name.startswith(".incoming-")]
    assert ids(box) == FIVE_IDS


def test_link_errors_are_plain(box):
    assert box.client.post("/api/panel/add/plan", json={"link": "https://example.com/x/y"}).json()["code"] == "bad_link"
    box.hub.files = {"README.md": b"hi"}
    answer = box.client.post("/api/panel/add/plan", json={"link": "owner/nothing"})
    assert answer.status_code == 422 and "not supported by your engines" in answer.json()["message"]
    assert box.client.get("/api/panel/add/downloads/nope").status_code == 404

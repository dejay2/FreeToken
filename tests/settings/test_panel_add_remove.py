"""Add and remove models on the control panel (Stage B). Review focus 2, 3 and 5 live here."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.boot_parser import BootFile, BootParseError
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
                                file_fetcher=hub.fetch, pc_memory=64 * GIB, card_memory=32 * GIB,
                                disk_free=lambda _: 10 ** 12)
    service = PanelService(
        store=store, writer=writer, switcher=switcher, profiles=profiles,
        boot_file=lambda: BootFile(boot), estimate_service=FakeEstimates(),
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


# ---- review round: steps after the save, symlinks, staging folders, order of checks ----
def lock_free_probe(box, calls):
    """A Pi stand-in whose add/remove report whether the panel lock was free at the time.
    Only another thread can tell: an RLock re-enters for the thread that holds it."""
    def probe(*_args, **_kwargs):
        seen = []

        def look():
            got = box.service._lock.acquire(blocking=False)
            if got:
                box.service._lock.release()
            seen.append(got)
        worker = threading.Thread(target=look)
        worker.start()
        worker.join(2)
        calls.append(seen[0])
        return {"status": "updated", "message": "", "notes": []}
    return SimpleNamespace(add=probe, remove=probe)


def test_pi_add_raising_does_not_undo_the_add(box, monkeypatch):
    def boom(*_a, **_k):
        raise TypeError("Pi's models.json holds a list where a dict was expected")
    monkeypatch.setattr(box.service.pi, "add", boom)
    write_v2(box.ninfer / "small_9b.ninfer")
    answer = add(box, box.ninfer / "small_9b.ninfer")
    assert answer.status_code == 200, answer.text
    assert answer.json()["status"] == "added" and "small_9b" in ids(box)
    assert answer.json()["pi"]["status"] == "not_updated" and "Pi" in answer.json()["pi"]["message"]


def test_pi_remove_raising_does_not_undo_the_remove(box, monkeypatch):
    def boom(*_a, **_k):
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(box.service.pi, "remove", boom)
    answer = remove(box, "quasar-27b")
    assert answer.status_code == 200, answer.text
    assert "quasar-27b" not in ids(box)
    assert answer.json()["pi"]["status"] == "not_updated" and "Input/output error" in answer.json()["pi"]["message"]


def test_profile_delete_raising_does_not_undo_the_remove(box, monkeypatch):
    box.service._ensure_profiles(box.store.load()[0])

    def boom(*_a, **_k):
        raise OSError(13, "Permission denied")
    monkeypatch.setattr(box.profiles, "delete", boom)
    answer = remove(box, "qwen3.8-flash")
    assert answer.status_code == 200, answer.text
    assert "qwen3.8-flash" not in ids(box)
    profile = answer.json()["profile"]
    assert profile["deleted"] is False and "Permission denied" in profile["message"]


def test_a_failing_boot_switch_after_the_profile_delete_is_reported_not_raised(box):
    box.service._ensure_profiles(box.store.load()[0])
    assert box.client.post("/api/profiles/model-qwen3.8-flash/activate").status_code == 200

    def boom(_result):
        raise BootParseError("boot-2020.ps1: no launcher line")
    box.service.profile_deleted = boom
    answer = remove(box, "qwen3.8-flash")
    assert answer.status_code == 200, answer.text
    assert "qwen3.8-flash" not in ids(box) and box.profiles.get("model-qwen3.8-flash") is None
    profile = answer.json()["profile"]
    assert profile["deleted"] is True and "no launcher line" in profile["message"]


def test_deleting_a_symlinked_model_removes_only_the_link(box, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_v2(store / "small_9b.ninfer")
    link = box.ninfer / "small_9b.ninfer"
    link.symlink_to(store / "small_9b.ninfer")
    assert add(box, link).status_code == 200
    answer = remove(box, "small_9b", deleteFiles=True)
    assert answer.status_code == 200, answer.text
    assert not link.is_symlink() and (store / "small_9b.ninfer").is_file()
    files = answer.json()["files"]
    assert files["deleted"] is True and "the link was removed; the files it points to were kept" in files["message"]
    assert "were deleted" not in files["message"]


def test_a_symlinked_folder_model_keeps_its_target(box, tmp_path):
    target = write_folder(tmp_path / "store" / "Tiny-Llama", max_position_embeddings=8192)
    link = box.models / "Tiny-Llama"
    link.symlink_to(target)
    assert add(box, link, id="tiny-llama", name="Tiny Llama", ramNeedGB=2).status_code == 200
    answer = remove(box, "tiny-llama", deleteFiles=True)
    assert answer.status_code == 200, answer.text
    assert not link.exists() and not link.is_symlink() and (target / "config.json").is_file()
    assert "the link was removed" in answer.json()["files"]["message"]


def test_a_download_staging_folder_is_refused(box):
    staging = box.ninfer / ".incoming-abc123"
    staging.mkdir()
    write_v2(staging / "small_9b.ninfer")
    found = box.client.post("/api/panel/add/detect", json={"path": str(staging / "small_9b.ninfer")}).json()
    assert found["kind"] == "unsupported" and "download" in found["reason"] and found["already"] is None
    answer = add(box, staging / "small_9b.ninfer")
    assert answer.status_code == 422 and answer.json()["code"] == "not_supported"
    assert "download" in answer.json()["message"] and "small_9b" not in ids(box)


def test_a_remove_the_switcher_would_refuse_unloads_nothing(box):
    box.switcher.states = {"quasar-27b": "ready"}
    box.service.writer._runner = checker(ok=False)
    before, text = box.store.load()[1], box.cfg.read_text()
    answer = remove(box, "quasar-27b")
    assert answer.status_code == 422 and answer.json()["code"] == "switcher_refused"
    assert box.switcher.calls == [], "the model must not be put away before the checks pass"
    assert box.store.load()[1] == before and box.cfg.read_text() == text and "quasar-27b" in pi_ids(box)


def test_a_failure_after_the_unload_says_the_model_was_put_away(box, monkeypatch):
    box.switcher.states = {"quasar-27b": "ready"}

    def boom(*_a, **_k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(box.store, "save", boom)
    answer = remove(box, "quasar-27b")
    assert answer.status_code == 500, answer.text
    assert box.switcher.calls == [("unload", "quasar-27b")]
    message = answer.json()["message"]
    assert "put away" in message and "not removed" in message and "No space left" in message
    assert "quasar-27b" in ids(box)


def test_hub_errors_are_told_apart(box, monkeypatch):
    def raising(exc):
        def model_info(*_a, **_k):
            raise exc
        return model_info
    monkeypatch.setattr(box.hub, "model_info", raising(OSError(28, "No space left on device")))
    answer = box.client.post("/api/panel/add/plan", json={"link": "owner/small-9b"})
    assert answer.status_code == 507, answer.text
    assert answer.json()["message"].startswith(f"Couldn't write to {box.models}") and "Hugging Face" not in answer.json()["message"]
    monkeypatch.setattr(box.hub, "model_info", raising(OSError(13, "Permission denied")))
    answer = box.client.post("/api/panel/add/plan", json={"link": "owner/small-9b"})
    assert answer.status_code == 500 and "Permission denied" in answer.json()["message"]
    monkeypatch.setattr(box.hub, "model_info", raising(RuntimeError("a bug in the planner")))
    answer = box.client.post("/api/panel/add/plan", json={"link": "owner/small-9b"})
    assert answer.status_code == 500 and answer.json()["code"] == "add_failed"
    assert "Hugging Face" not in answer.json()["message"] and "a bug in the planner" not in answer.json()["message"]
    monkeypatch.setattr(box.hub, "model_info", raising(ConnectionError("Name or service not known")))
    answer = box.client.post("/api/panel/add/plan", json={"link": "owner/small-9b"})
    assert answer.status_code == 502 and answer.json()["code"] == "hub_error" and "Hugging Face" in answer.json()["message"]
    # Review item 11: the Hub's own text (URL, request id) is logged, never shown.
    assert "Name or service not known" not in answer.json()["message"]
    raw = "404 Client Error for url: https://huggingface.co/api/models/owner/small-9b (Request ID: Root=1-abc)"
    http_error = type("HfHubHTTPError", (OSError,), {"__module__": "huggingface_hub.utils._errors"})(raw)
    http_error.response = SimpleNamespace(status_code=404)
    monkeypatch.setattr(box.hub, "model_info", raising(http_error))
    answer = box.client.post("/api/panel/add/plan", json={"link": "owner/small-9b"})
    assert answer.status_code == 502 and answer.json()["code"] == "hub_error"
    assert answer.json()["message"] == "Hugging Face has no model repo at that link. Check the owner and name."
    gated = type("GatedRepoError", (OSError,), {"__module__": "huggingface_hub.utils._errors"})(raw)
    monkeypatch.setattr(box.hub, "model_info", raising(gated))
    assert "gated or private" in box.client.post("/api/panel/add/plan", json={"link": "owner/small-9b"}).json()["message"]


def test_a_download_that_loses_the_hub_fails_in_plain_words(box, tmp_path, monkeypatch, caplog):
    """Review item 11: a Hub HTTP error mid-download carries a URL and request id; the job's
    error line says the connection was lost and the raw text goes to the helper log."""
    source = tmp_path / "src"
    source.mkdir()
    write_v3(source / "small_9b.ninfer", parts=1)
    box.hub.files = {p.name: p.read_bytes() for p in sorted(source.iterdir())}
    raw = "500 Server Error for url: https://huggingface.co/owner/small-9b/resolve/main/x (Request ID: Root=1-abc)"
    http_error = type("HfHubHTTPError", (OSError,), {"__module__": "huggingface_hub.utils._errors"})(raw)

    def fetch(repo, name, destination, cancelled):
        if name.endswith(".part-0001"):
            raise http_error
        return box.hub.fetch(repo, name, destination, cancelled)
    monkeypatch.setattr(box.service.downloads, "_file_fetcher", fetch)
    job = box.client.post("/api/panel/add/downloads", json={"link": "owner/small-9b"}).json()
    done = wait_job(box, job["id"])
    assert done["stage"] == "failed"
    assert done["error"] == "The connection to Hugging Face was lost, so the download stopped."
    assert "Request ID" in caplog.text and "huggingface.co" in caplog.text
    assert not (box.ninfer / "small_9b.ninfer").exists()


def test_removing_an_unknown_model_says_reload(box):
    answer = remove(box, "gone")
    assert answer.status_code == 404 and answer.json()["code"] == "not_found"
    assert answer.json()["message"] == "That model is no longer in the list. Reload the page."


def test_pi_is_called_after_the_lock_is_released(box):
    calls = []
    box.service.pi = lock_free_probe(box, calls)
    write_v2(box.ninfer / "small_9b.ninfer")
    assert add(box, box.ninfer / "small_9b.ninfer").status_code == 200
    assert remove(box, "small_9b").status_code == 200
    assert calls == [True, True], "Pi (a slow /mnt/c write) must run outside the panel lock"


@pytest.mark.parametrize("artifact", ["/", "/home", "~/.."])
def test_paths_above_home_are_never_deleted(box, artifact):
    doc, rev = box.store.load()
    find_model(doc, "qwen3.8-flash")["artifact"] = artifact
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "qwen3.8-flash", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_unsafe", answer.text
    assert "qwen3.8-flash" in ids(box) and box.home.is_dir() and box.ninfer.is_dir()


def test_a_removed_model_leaves_no_hold_behind_while_the_switcher_is_up(box):
    box.switcher.states = {"quasar-27b": "ready"}
    view = box.client.get("/api/panel/views/model/quasar-27b").json()
    settings = {k: v for k, v in view["settings"].items() if not k.startswith("model.")}
    settings["draft-tokens"] = 5
    held = box.client.put("/api/panel/models/quasar-27b", json={
        "revision": view["revision"], "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "next-time"})
    assert held.json()["held"] == ["quasar-27b"] and "quasar-27b" in json.loads(box.service.holds_path.read_text())
    answer = remove(box, "quasar-27b")
    assert answer.status_code == 200, answer.text
    assert box.switcher.calls == [("unload", "quasar-27b")]
    assert "quasar-27b" not in json.loads(box.service.holds_path.read_text())
    assert "quasar-27b" not in extract_model_blocks(box.cfg.read_text()) and answer.json()["held"] == []


# ---- review, PR #17 ----
def test_pi_changes_run_in_the_order_the_list_was_changed(box):
    """An add whose Pi write is slow, then a remove of the same model: Pi must not see the
    remove first (it would end up listing a model the panel no longer has)."""
    order, entered, release = [], threading.Event(), threading.Event()

    def slow_add(*_a, **_k):
        entered.set()
        release.wait(5)
        order.append("add")
        return {"status": "updated", "message": "", "notes": []}

    def quick_remove(*_a, **_k):
        order.append("remove")
        return {"status": "updated", "message": "", "notes": []}
    box.service.pi = SimpleNamespace(add=slow_add, remove=quick_remove)
    write_v2(box.ninfer / "small_9b.ninfer")
    rev = box.store.load()[1]
    adder = threading.Thread(target=box.service.add_model,
                             args=(str(box.ninfer / "small_9b.ninfer"), "small_9b", "Small 9B", 6, rev))
    adder.start()
    assert entered.wait(5)
    answers = []
    remover = threading.Thread(target=lambda: answers.append(
        box.service.remove_model("small_9b", False, box.store.load()[1])))
    remover.start()
    time.sleep(0.3)
    assert order == [], "the remove's Pi change ran before the add's"
    release.set()
    adder.join(5)
    remover.join(5)
    assert order == ["add", "remove"] and answers[0]["pi"]["status"] == "updated"


def switcher_answers(box, *answers):
    """running() gives these answers in turn (the last one repeats)."""
    seq = list(answers)

    def running():
        value = seq.pop(0) if len(seq) > 1 else seq[0]
        return None if value is None else dict(value)
    box.switcher.running = running


def test_remove_puts_away_a_model_loaded_since_the_first_look(box):
    switcher_answers(box, {}, {"quasar-27b": "ready"})
    answer = remove(box, "quasar-27b")
    assert answer.status_code == 200, answer.text
    assert box.switcher.calls == [("unload", "quasar-27b")]


def test_remove_is_refused_when_the_final_look_is_unknown(box):
    rev = box.store.load()[1]
    switcher_answers(box, {}, None)
    answer = box.client.post("/api/panel/models/quasar-27b/remove", json={"revision": rev})
    assert answer.status_code == 503 and answer.json()["code"] == "switcher_unknown", answer.text
    assert "quasar-27b" in ids(box) and box.switcher.calls == []


def test_removing_a_symlinked_alias_keeps_the_originals_parts(box):
    original = write_v3(box.ninfer / "original.ninfer", parts=1)
    alias = box.ninfer / "alias.ninfer"
    alias.symlink_to(original.name)
    assert add(box, alias, id="alias", name="Alias").status_code == 200
    answer = remove(box, "alias", deleteFiles=True)
    assert answer.status_code == 200, answer.text
    assert not alias.is_symlink() and original.is_file() and (box.ninfer / "original.ninfer.part-0001").is_file()
    assert answer.json()["files"]["paths"] == [str(alias)]


def test_a_part_listed_by_a_model_that_cannot_load_is_still_protected(box):
    write_v3(box.ninfer / "small_9b.ninfer", parts=1)
    assert add(box, box.ninfer / "small_9b.ninfer").status_code == 200
    write_v3(box.ninfer / "other.ninfer", part_names=["small_9b.ninfer.part-0001", "other.ninfer.part-0001"])
    (box.ninfer / "other.ninfer.part-0001").unlink()  # other cannot load now, but still reads small_9b's part
    doc, rev = box.store.load()
    doc["models"].append({**find_model(doc, "twin-27b"), "id": "other", "name": "Other",
                          "artifact": "~/ninfer-work/models/other.ninfer"})
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "small_9b", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_shared", answer.text
    assert (box.ninfer / "small_9b.ninfer.part-0001").is_file() and "small_9b" in ids(box)


def test_delete_is_refused_when_another_models_files_cannot_be_read(box):
    write_v3(box.ninfer / "small_9b.ninfer", parts=1)
    assert add(box, box.ninfer / "small_9b.ninfer").status_code == 200
    (box.ninfer / "twin_nvfp4.ninfer").write_bytes(b"NINFER\x00\x03" + b"\xff" * 64)  # header unreadable
    answer = remove(box, "small_9b", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_unknown", answer.text
    assert "Twin" in answer.json()["message"] or "twin" in answer.json()["message"].lower()
    assert (box.ninfer / "small_9b.ninfer.part-0001").is_file() and "small_9b" in ids(box)


def test_a_folder_named_like_a_part_is_never_deleted(box):
    write_v2(box.ninfer / "small_9b.ninfer")
    assert add(box, box.ninfer / "small_9b.ninfer").status_code == 200
    (box.ninfer / "small_9b.ninfer").write_bytes(b"broken")  # damaged: parts found by name
    odd = box.ninfer / "small_9b.ninfer.part-0001"
    odd.mkdir()
    (odd / "keep.txt").write_text("x")
    answer = remove(box, "small_9b", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_unsafe", answer.text
    assert (odd / "keep.txt").is_file() and "small_9b" in ids(box)


def test_a_symlinked_part_another_model_reads_is_never_deleted(box, tmp_path):
    """The part is a link; the other model names the link. Checked by target only, the link
    was deleted and the other model broken (review round 2, PR #17)."""
    write_v3(box.ninfer / "small_9b.ninfer", parts=1)
    part = box.ninfer / "small_9b.ninfer.part-0001"
    store = tmp_path / "store"
    store.mkdir()
    part.rename(store / part.name)
    part.symlink_to(store / part.name)
    assert add(box, box.ninfer / "small_9b.ninfer").status_code == 200
    write_v3(box.ninfer / "other.ninfer", part_names=["small_9b.ninfer.part-0001"])
    doc, rev = box.store.load()
    doc["models"].append({**find_model(doc, "twin-27b"), "id": "other", "name": "Other",
                          "artifact": "~/ninfer-work/models/other.ninfer"})
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "small_9b", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_shared", answer.text
    assert part.is_symlink() and "small_9b" in ids(box)

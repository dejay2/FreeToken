"""Control panel stage A: the deferred reviewer minors (server side). Each test fails without its fix."""

from __future__ import annotations

import json
import os
import threading

import pytest

from freetoken.daemon.settings import registry as reg
from freetoken.daemon.settings.profiles_manager import ProfilesManager
from freetoken.daemon.settings.registry import RegistryStore, find_model
from freetoken.daemon.settings.swap_config import HEADER, SwapConfigWriter, config_sha256, render_config
from tests.settings.registry_fixtures import five
from tests.settings.test_panel_routes import checker, engine_settings, env, run_spawned, seed  # noqa: F401
from tests.settings.test_registry import make_store


# ---- 2: plain words, never Python's own text ----
def test_a_damaged_list_says_so_in_plain_words(tmp_path):
    store = make_store(tmp_path)
    store.save(five(), expected_revision=None)
    store.path.write_text('{"version": 1, "models": [', encoding="utf-8")
    with pytest.raises(reg.RegistryCorrupt) as caught:
        store.load()
    assert "not valid JSON" in caught.value.message
    for leak in ("Expecting", "line 1", "column", "char "):
        assert leak not in caught.value.message, caught.value.message


def test_a_value_that_slips_past_the_checks_is_a_plain_422(env, monkeypatch):
    revision = seed(env)

    def broken(settings):
        raise ValueError("invalid literal for int() with base 10: 'x'")

    monkeypatch.setattr(reg.ninfer_dials, "canonical_settings", broken)
    answer = env.client.put("/api/panel/engines/ninfer/defaults", json={"revision": revision, "settings": {}})
    assert answer.status_code == 422, answer.text
    assert "invalid literal" not in answer.text and "could not be stored" in answer.text


# ---- 3: staged and written files appear whole or not at all ----
def _failing_write(monkeypatch, after_bytes: int = 5):
    real = os.write

    def write(fd, data):
        real(fd, bytes(data[:after_bytes]))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "write", write)


def test_a_failed_registry_write_leaves_no_partial_file(tmp_path, monkeypatch):
    store = RegistryStore(tmp_path / "registry.json")
    first = store.save(five(), expected_revision=None)
    before = store.path.read_bytes()
    doc = five()
    doc["system"]["floorGB"] = 9
    _failing_write(monkeypatch)
    with pytest.raises(OSError):
        store.save(doc, expected_revision=first)
    monkeypatch.undo()
    assert store.path.read_bytes() == before
    assert not [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_a_failed_stage_leaves_no_partial_new_file(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    binary = tmp_path / "llama-swap"
    binary.write_text("#!/bin/sh\n")
    writer = SwapConfigWriter(cfg, binary=binary, runner=checker())
    _failing_write(monkeypatch)
    with pytest.raises(OSError):
        writer.check(render_config(five(), {}))
    monkeypatch.undo()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["llama-swap"]


def test_short_writes_are_finished(tmp_path, monkeypatch):
    real = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real(fd, bytes(data[:7])))  # 7 bytes a call
    reg.write_atomic(tmp_path / "x.json", b"0123456789" * 10)
    monkeypatch.undo()
    assert (tmp_path / "x.json").read_bytes() == b"0123456789" * 10


# ---- 4: an unchanged model is not "changed" because the stored list is not canonical ----
def test_a_hand_spelled_value_does_not_ask_to_restart_an_unchanged_model(env):
    doc = five()
    find_model(doc, "qwen3.8-flash")["overrides"]["GpuOwnedLayers"] = "auto"  # canonical: "auto:6"
    env.store.path.parent.mkdir(parents=True, exist_ok=True)
    env.store.path.write_bytes(reg.dumps(doc))  # as a restore or a hand edit leaves it
    env.writer.write(render_config(doc, {}))
    revision = env.store.load()[1]
    env.switcher.states = {"qwen3.8-flash": "ready"}
    answer = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7}})
    assert answer.status_code == 200, answer.text
    assert answer.json()["restarting"] == [] and ("unload", "qwen3.8-flash") not in env.switcher.calls


# ---- 9 and 10: profile pushes (freetoken.sh --profile) ----
def test_upsert_keeps_and_updates_the_description(tmp_path):
    manager = ProfilesManager(tmp_path / "boot-profiles.json", profile_dir=tmp_path / "boot-profiles")
    settings = {"ModelPath": "/m/B", "KVDtype": "fp8"}
    manager.upsert("model-flash", name="Flash", description="first words", settings=settings)
    again = manager.upsert("model-flash", name="Flash", description="new words", settings=settings)
    assert again["changed"] is False  # words only: a running server still matches
    assert manager.get("model-flash")["description"] == "new words"
    manager.upsert("model-flash", name="Flash", description="", settings={**settings, "KVDtype": "bf16"})
    assert manager.get("model-flash")["description"] == "new words"


def test_replace_push_with_bad_values_is_a_422(env):
    bad = env.client.put("/api/profiles/model-flash", json={"name": "Flash", "settings": {"KVDtype": "fp3"}, "replace": True})
    assert bad.status_code == 422, bad.text
    wrong_id = env.client.put("/api/profiles/prof-1", json={"name": "x", "settings": {}, "replace": True})
    assert wrong_id.status_code == 422, wrong_id.text
    no_name = env.client.put("/api/profiles/model-flash", json={"name": " ", "settings": {}, "replace": True})
    assert no_name.status_code == 422, no_name.text
    assert env.profiles.get("model-flash") is None


# ---- 11: the restart waits for the switcher, not for one exact text ----
def test_restart_loads_when_the_file_moved_on_after_the_write(env):
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    answer = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "whenLoaded": "restart"})
    assert answer.status_code == 200 and answer.json()["restarting"] == ["quasar-27b"], answer.text
    # Another panel write lands before the switcher catches up (the hold watcher, a second save).
    later = env.client.put("/api/panel/system", json={"revision": answer.json()["revision"], "system": {"floorGB": 7}})
    assert later.status_code == 200, later.text
    env.service.restart_wait_s = 0
    run_spawned(env)
    assert env.service.last_restart["ok"] is True, env.service.last_restart
    assert ("load", "quasar-27b") in env.switcher.calls


# ---- 12 and 27: a failed put-away before a save ----
def test_an_unload_error_before_a_save_is_plain_and_changes_nothing(env):
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "ready"}

    def unload(model_id):
        raise OSError("connection reset by peer")

    env.switcher.unload = unload
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    before_cfg, before_reg = env.cfg.read_text(), env.store.path.read_bytes()
    answer = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "whenLoaded": "restart"})
    assert answer.status_code == 503, answer.text
    assert answer.json()["message"].startswith("Couldn't put Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2) away")
    assert "connection reset" not in answer.text
    assert env.cfg.read_text() == before_cfg and env.store.path.read_bytes() == before_reg
    assert not (env.cfg.parent / "config.yaml.new").exists()


def test_a_partial_multi_model_unload_names_the_ones_put_away(env):
    revision = seed(env)
    env.switcher.states = {"fable-27b": "ready", "twin-27b": "ready"}
    real = env.switcher.unload
    env.switcher.unload = lambda model_id: real(model_id) if model_id == "fable-27b" else False
    view = env.client.get("/api/panel/views/engine/ninfer").json()
    settings = dict(view["settings"])
    settings["max-pending-requests"] = 12
    answer = env.client.put("/api/panel/engines/ninfer/defaults", json={
        "revision": revision, "settings": settings, "whenLoaded": "restart"})
    assert answer.status_code == 503, answer.text
    message = answer.json()["message"]
    assert "Couldn't put Twin 27B NVFP4 (NInfer) away" in message
    assert "Fable 27B NVFP4 (NInfer) was already put away" in message and "load it again" in message


# ---- 13: a missing id is a plain 404 ----
@pytest.mark.parametrize("method,url", [
    ("get", "/api/panel/views/model/nope"),
    ("get", "/api/panel/views/model/quasar-27b?preset=nope"),
    ("get", "/api/panel/views/engine/nope"),
    ("post", "/api/panel/models/nope/load"),
    ("post", "/api/panel/models/nope/unload"),
    ("post", "/api/panel/models/nope/sleep"),
    ("get", "/api/panel/models/nope/effective"),
    ("post", "/api/panel/models/nope/fit"),
    ("get", "/api/panel/add/downloads/nope"),
])
def test_a_missing_id_is_a_plain_404(env, method, url):
    seed(env)
    answer = getattr(env.client, method)(url, **({"json": {}} if method == "post" else {}))
    assert answer.status_code == 404, answer.text
    body = answer.json()
    assert body["code"] == "not_found" and "nope" not in body["message"] and body["message"].endswith(".")


def test_a_missing_model_on_save_is_a_plain_404(env):
    revision = seed(env)
    answer = env.client.put("/api/panel/models/nope", json={"revision": revision, "settings": {}, "identity": {}})
    assert answer.status_code == 404 and answer.json()["code"] == "not_found"
    preset = env.client.post("/api/panel/models/quasar-27b/presets/rename", json={"revision": revision, "name": "nope", "newName": "x"})
    assert preset.status_code == 404 and preset.json()["code"] == "not_found"


# ---- 17: another model's problem names that model ----
def test_another_models_limit_names_the_model(env, monkeypatch):
    revision = seed(env)
    from freetoken.daemon.settings import panel as panel_module
    monkeypatch.setattr(panel_module.PanelService, "_model_limit_errors",
                        lambda self, doc: [{"field": "ContextTokens", "message": "Too long for this model.",
                                            "where": "qwen3.8-flash-abliterated"}])
    answer = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7}})
    assert answer.status_code == 422
    assert answer.json()["detail"][0]["where"] == "Qwen3.8 Flash Next ABLITERATED NVFP4 (FreeToken, uncensored)"


# ---- 23: .corrupt-* copies are pruned ----
def test_only_the_newest_five_damaged_copies_stay(tmp_path):
    store = make_store(tmp_path)
    first = store.save(five(), expected_revision=None)
    doc = five()
    doc["system"]["floorGB"] = 7
    store.save(doc, expected_revision=first)
    for index in range(8):
        store.path.write_text("{broken %d" % index, encoding="utf-8")
        store.restore(store.backups()[0])  # the damaged file is set aside as .corrupt-<time>
    kept = sorted(p.name for p in tmp_path.iterdir() if ".corrupt-" in p.name)
    assert len(kept) == reg.CORRUPT_KEPT == 5
    assert [(tmp_path / name).read_text() for name in kept] == ["{broken %d" % i for i in range(3, 8)]


# ---- 24: a hand-written switcher file is copied before it is replaced ----
def test_restore_over_a_hand_written_config_keeps_a_copy(env):
    first = seed(env)
    env.client.put("/api/panel/system", json={"revision": first, "system": {"floorGB": 7}})
    hand = "# my own file\nmodels: {}\n"
    env.cfg.write_text(hand)
    backup = env.store.backups()[0]
    answer = env.client.post("/api/panel/registry/restore", json={"backup": backup})
    assert answer.status_code == 200, answer.text
    assert env.cfg.read_text().startswith(HEADER)
    copies = [p for p in env.cfg.parent.iterdir() if p.name.startswith("config.yaml.hand-")]
    assert len(copies) == 1 and copies[0].read_text() == hand
    # A file the panel wrote itself is not copied again.
    second = env.store.load()[1]
    env.client.put("/api/panel/system", json={"revision": second, "system": {"floorGB": 8}})
    assert len([p for p in env.cfg.parent.iterdir() if p.name.startswith("config.yaml.hand-")]) == 1


# ---- 25: the Right-now strip reads one consistent restart note and never waits on a save ----
def test_now_answers_while_a_save_holds_the_panel_lock(env):
    seed(env)
    env.service._restart_done(["quasar-27b"], False, "Restarting quasar-27b failed: boom")
    got = {}
    with env.service._lock:
        worker = threading.Thread(target=lambda: got.update(now=env.service.now()))
        worker.start()
        worker.join(5)
    assert "now" in got, "now() waited for the panel lock"
    assert got["now"]["lastRestart"]["message"] == "Restarting quasar-27b failed: boom"
    env.service._clear_restart()
    assert env.service.now()["lastRestart"] is None and env.service.last_restart is None

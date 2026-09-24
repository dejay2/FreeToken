"""The control panel's routes against a fake switcher. Review focus 1, 3 and 4 live here."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.boot_parser import BootFile
from freetoken.daemon.settings.panel import PanelService
from freetoken.daemon.settings.process_manager import ProcessManager
from freetoken.daemon.settings.profiles_manager import ProfilesManager
from freetoken.daemon.settings.registry import RegistryStore, find_model
from freetoken.daemon.settings.swap_config import SwapConfigWriter, config_sha256, extract_model_blocks, render_config
from freetoken.daemon.settings.switcher import SwitcherError
from tests.settings.registry_fixtures import five

GIB = 1024 ** 3
REPO = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO / "engines" / "config" / "config.example.yaml").read_text(encoding="utf-8")


class FakeSwitcher:
    def __init__(self, cfg: Path):
        self.cfg, self.states, self.up, self.calls, self.load_error = cfg, {}, True, [], None

    def running(self):
        return dict(self.states) if self.up else None

    def config_hash(self):
        return config_sha256(self.cfg.read_text()) if self.up and self.cfg.exists() else None

    def unload(self, model_id):
        self.calls.append(("unload", model_id))
        self.states.pop(model_id, None)
        return self.up

    def load(self, model_id, *, timeout=900.0):
        self.calls.append(("load", model_id))
        if not self.up:
            raise OSError("connection refused")
        if self.load_error:
            raise self.load_error
        self.states[model_id] = "ready"


class FakeEstimates:
    def __init__(self):
        self.calls = []

    def estimate_settings(self, settings, *, boot_file, environ=None, action=None):
        self.calls.append(dict(settings))
        return {"fits_empty": True, "components": [], "suggestion": None,
                "resources": {"vram": {"total_bytes": 32 * GIB, "empty": {"need_bytes": 26 * GIB}}}}


def checker(ok=True):
    def run(args, **kwargs):
        out = "config is valid\n" if ok else "config validation failed: boom\n"
        return subprocess.CompletedProcess(args, 0 if ok else 1, stdout=out, stderr="")
    return run


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/home/jay")
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
    spawned = []
    estimates = FakeEstimates()
    service = PanelService(
        store=store, writer=writer, switcher=switcher, profiles=profiles,
        boot_file=lambda: BootFile(boot), default_boot=lambda: boot, estimate_service=estimates,
        card_probe=lambda: {"totalBytes": 32 * GIB, "usedBytes": 2 * GIB},
        windows_free_probe=lambda: 40 * GIB, artifact_size=lambda path: 19_782_132_224,
        spawn=lambda fn, *args: spawned.append((fn, args)), sleep=lambda _: None,
    )
    proc = ProcessManager(boot_file=boot, stop_script=tmp_path / "stop.ps1", log_path=tmp_path / "server.log",
                          lock_path=tmp_path / "gpu.lock", runner=lambda *a, **k: None,
                          readiness=lambda: {"state": "unreachable"}, sleep=lambda _: None, poll_interval=0)
    app = create_app(boot_file=boot, process_manager=proc, profiles=profiles, log_path=tmp_path / "server.log",
                     static_path=tmp_path / "missing.html", panel=service)
    return SimpleNamespace(client=TestClient(app), service=service, store=store, writer=writer, switcher=switcher,
                           cfg=cfg, spawned=spawned, profiles=profiles, estimates=estimates)


def seed(env, doc=None):
    revision = env.store.save(doc or five(), expected_revision=None)
    env.writer.write(render_config(env.store.load()[0], {}))
    return revision


def engine_settings(view):
    return {k: v for k, v in view["settings"].items() if not k.startswith("model.")}


def run_spawned(env):
    for fn, args in env.spawned:
        fn(*args)
    env.spawned.clear()


def test_missing_registry_offers_the_import(env):
    assert env.client.get("/api/panel/registry").json()["status"] == "missing"
    assert env.client.get("/api/panel/models").json()["code"] == "registry_missing"


def test_import_copies_todays_config_backs_it_up_and_generates_ours(env):
    env.cfg.write_text(EXAMPLE)
    answer = env.client.post("/api/panel/import", json={})
    assert answer.status_code == 200, answer.text
    assert (env.cfg.parent / "config.yaml.bak-before-registry").read_text() == EXAMPLE
    assert env.cfg.read_text().startswith("# generated by the control panel")
    doc, _ = env.store.load()
    assert doc["engines"]["freetoken"]["defaults"]["KVDtype"] == "fp8"
    assert env.profiles.get("model-qwen3.8-flash") is not None
    assert env.client.post("/api/panel/import", json={}).json()["code"] == "import_refused"


def test_import_with_a_loaded_model_must_restart_it(env):
    env.cfg.write_text(EXAMPLE)
    env.switcher.states = {"quasar-27b": "ready"}
    first = env.client.post("/api/panel/import", json={})
    assert first.status_code == 409 and first.json()["nextTimeAllowed"] is False
    assert [row["id"] for row in first.json()["affected"]] == ["quasar-27b"]
    assert not env.store.exists() and env.cfg.read_text() == EXAMPLE
    second = env.client.post("/api/panel/import", json={"whenLoaded": "restart"})
    assert second.status_code == 200, second.text
    assert env.switcher.calls == [("unload", "quasar-27b")]
    run_spawned(env)
    assert env.switcher.calls[-1] == ("load", "quasar-27b")


def test_engine_default_change_on_a_loaded_model_asks_and_lists_it(env):
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    view = env.client.get("/api/panel/views/engine/ninfer").json()
    settings = dict(view["settings"])
    settings["max-concurrency"] = 5
    asked = env.client.put("/api/panel/engines/ninfer/defaults", json={"revision": revision, "settings": settings})
    assert asked.status_code == 409 and asked.json()["code"] == "choose_restart"
    assert [row["id"] for row in asked.json()["affected"]] == ["quasar-27b"]
    assert "--max-concurrency 5" not in env.cfg.read_text()
    saved = env.client.put("/api/panel/engines/ninfer/defaults",
                           json={"revision": revision, "settings": settings, "whenLoaded": "next-time"})
    assert saved.status_code == 200, saved.text
    assert saved.json()["held"] == ["quasar-27b"]
    assert sorted(saved.json()["inherits"]) == sorted([
        "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)", "Fable 27B NVFP4 (NInfer)", "Twin 27B NVFP4 (NInfer)"])
    blocks = extract_model_blocks(env.cfg.read_text())
    assert "--max-concurrency 4" in blocks["quasar-27b"] and "--max-concurrency 5" in blocks["fable-27b"]


def test_a_change_that_moves_no_loaded_model_asks_nothing(env):
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    answer = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 8}})
    assert answer.status_code == 200, answer.text
    assert "floorGB: 8" in env.cfg.read_text() and env.switcher.calls == []


def test_save_while_a_load_is_in_flight_holds_or_restarts(env):
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "starting"}
    view = env.client.get("/api/panel/views/model/quasar-27b").json()
    settings = engine_settings(view)
    settings["draft-tokens"] = 5
    body = {"revision": revision, "settings": settings, "identity": {}, "activePreset": None}
    asked = env.client.put("/api/panel/models/quasar-27b", json=body)
    assert asked.status_code == 409 and asked.json()["affected"][0]["id"] == "quasar-27b"
    held = env.client.put("/api/panel/models/quasar-27b", json={**body, "whenLoaded": "next-time"})
    assert held.status_code == 200, held.text
    assert "--draft-tokens 7" in extract_model_blocks(env.cfg.read_text())["quasar-27b"]
    env.switcher.states = {"quasar-27b": "ready"}
    assert env.service.release_finished_holds() == []
    env.switcher.states = {}
    assert env.service.release_finished_holds() == ["quasar-27b"]
    assert "--draft-tokens 5" in extract_model_blocks(env.cfg.read_text())["quasar-27b"]


def test_restart_now_unloads_writes_and_loads_again(env):
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    answer = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "restart"})
    assert answer.status_code == 200, answer.text
    assert answer.json()["restarting"] == ["quasar-27b"]
    assert env.switcher.calls == [("unload", "quasar-27b")]
    assert "--draft-tokens 5" in env.cfg.read_text()
    run_spawned(env)
    assert env.switcher.calls[-1] == ("load", "quasar-27b")
    assert env.service.last_restart["ok"] is True


def test_switcher_refusal_keeps_the_old_config_and_registry(env):
    revision = seed(env)
    before = env.cfg.read_text()
    env.writer._runner = checker(ok=False)
    answer = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 9}})
    assert answer.status_code == 422
    assert answer.json()["message"] == "The switcher refused these settings: config validation failed: boom"
    assert env.cfg.read_text() == before and env.store.load()[0]["system"]["floorGB"] == 6


def test_model_view_marks_sources_and_saving_stores_only_differences(env):
    doc = five()
    quasar = find_model(doc, "quasar-27b")
    quasar["presets"] = {"Fast agents": {"max-concurrency": 6}}
    quasar["activePreset"] = "Fast agents"
    revision = seed(env, doc)
    view = env.client.get("/api/panel/views/model/quasar-27b").json()
    assert view["baseFrom"]["max-concurrency"] == "preset" and view["baseFrom"]["max-context"] == "default"
    assert view["settings"]["kv-dtype"] == "int8" and view["base"]["kv-dtype"] == "bf16"
    assert view["settings"]["model.idleMinutes"] == 0 and view["savedPreset"] == "Fast agents"
    settings = engine_settings(view)
    settings["kv-dtype"] = view["base"]["kv-dtype"]  # what "reset" does on the page
    settings["max-context"] = 120000
    answer = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "activePreset": "Fast agents",
        "identity": {"name": "QUASAR", "aliases": "q27, quasar", "ramNeedGB": "19", "idleMinutes": -1}})
    assert answer.status_code == 200, answer.text
    saved = find_model(env.store.load()[0], "quasar-27b")
    assert saved["overrides"] == {"spec": "dflash2", "draft-tokens": 7, "max-context": 120000}
    assert (saved["name"], saved["aliases"], saved["ramNeedGB"], saved["idleMinutes"]) == ("QUASAR", ["q27", "quasar"], 19, None)
    none_view = env.client.get("/api/panel/views/model/quasar-27b?preset=").json()
    assert none_view["activePreset"] is None and none_view["settings"]["max-concurrency"] == 4


def test_presets_add_rename_delete(env):
    revision = seed(env)
    added = env.client.post("/api/panel/models/quasar-27b/presets/add", json={"revision": revision, "name": "Quiet"})
    assert added.status_code == 200, added.text
    assert find_model(env.store.load()[0], "quasar-27b")["presets"]["Quiet"] == {
        "kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 7}
    renamed = env.client.post("/api/panel/models/quasar-27b/presets/rename",
                              json={"revision": added.json()["revision"], "name": "Quiet", "newName": "Calm"})
    assert renamed.status_code == 200 and "Calm" in find_model(env.store.load()[0], "quasar-27b")["presets"]
    bad = env.client.post("/api/panel/models/quasar-27b/presets/add", json={"revision": renamed.json()["revision"], "name": ""})
    assert bad.status_code == 422
    gone = env.client.post("/api/panel/models/quasar-27b/presets/delete",
                           json={"revision": renamed.json()["revision"], "name": "Calm"})
    assert gone.status_code == 200 and find_model(env.store.load()[0], "quasar-27b")["presets"] == {}


def test_unchanged_save_writes_nothing(env):
    revision = seed(env)
    calls = []
    env.writer._runner = lambda args, **kw: calls.append(args) or subprocess.CompletedProcess(args, 0, "", "")
    view = env.client.get("/api/panel/views/system").json()
    answer = env.client.put("/api/panel/system", json={"revision": revision, "system": view["settings"]})
    assert answer.status_code == 200 and answer.json()["revision"] == revision
    assert calls == [] and env.store.backups() == []


def test_effective_settings_for_the_adapter(env):
    seed(env)
    body = env.client.get("/api/panel/models/qwen3.8-flash/effective").json()
    assert body["settings"]["ModelPath"] == "/home/jay/models/Qwen3.8-Flash-Next-NVFP4"
    assert body["settings"]["KVDtype"] == "fp8"
    assert env.client.get("/api/panel/models/quasar-27b/effective").json()["code"] == "not_freetoken"
    assert env.client.get("/api/panel/models/nope/effective").status_code == 404


def test_right_now_strip_when_the_switcher_is_down(env):
    seed(env)
    env.switcher.up = False
    body = env.client.get("/api/panel/now").json()
    assert body["switcher"] == {"up": False, "running": []}
    assert body["card"]["totalBytes"] == 32 * GIB and body["windowsFreeBytes"] == 40 * GIB and body["cushionGB"] == 6
    rows = env.client.get("/api/panel/models").json()
    assert rows["switcherUp"] is False and {row["state"] for row in rows["models"]} == {"unknown"}


def test_fit_for_ninfer_and_freetoken(env):
    seed(env)
    view = env.client.get("/api/panel/views/model/quasar-27b").json()
    fit = env.client.post("/api/panel/models/quasar-27b/fit",
                          json={"settings": engine_settings(view), "identity": {"ramNeedGB": 18}}).json()
    assert fit["verdict"] == "fits" and abs(fit["needBytes"] - 29.7 * GIB) / (29.7 * GIB) < 0.10
    assert fit["ram"] == {"needGB": 18, "cushionGB": 6, "windowsFreeGB": 40.0, "loadedNow": False}
    ft_view = env.client.get("/api/panel/views/model/qwen3.8-flash").json()
    ft = env.client.post("/api/panel/models/qwen3.8-flash/fit", json={"settings": engine_settings(ft_view), "identity": {}}).json()
    assert ft["verdict"] == "fits" and ft["needBytes"] == 26 * GIB
    assert env.estimates.calls[-1]["ModelPath"] == "/home/jay/models/Qwen3.8-Flash-Next-NVFP4"


def test_corrupt_registry_offers_backups_and_restore(env):
    first = seed(env)
    second = env.client.put("/api/panel/system", json={"revision": first, "system": {"floorGB": 7}}).json()["revision"]
    env.store.path.write_text("{oops", encoding="utf-8")
    status = env.client.get("/api/panel/registry").json()
    assert status["status"] == "corrupt" and status["backups"]
    before = env.cfg.read_text()
    blocked = env.client.put("/api/panel/system", json={"revision": second, "system": {"floorGB": 8}})
    assert blocked.json()["code"] == "registry_corrupt" and env.cfg.read_text() == before
    restored = env.client.post("/api/panel/registry/restore", json={"backup": status["backups"][0]})
    assert restored.status_code == 200, restored.text
    assert env.store.load()[0]["system"]["floorGB"] == 6


def test_load_passes_switcher_errors_through(env):
    seed(env)
    env.switcher.load_error = SwitcherError(409, "model_superseded", "load of quasar-27b cancelled: model fable-27b was requested instead")
    superseded = env.client.post("/api/panel/models/quasar-27b/load")
    assert superseded.status_code == 409 and superseded.json()["code"] == "model_superseded"
    env.switcher.load_error = None
    env.switcher.up = False
    down = env.client.post("/api/panel/models/quasar-27b/load")
    assert down.status_code == 503 and down.json()["code"] == "switcher_down"


def test_first_save_creates_the_freetoken_profiles(env):
    revision = seed(env)
    assert env.profiles.get("model-qwen3.8-flash") is None
    env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7}})
    assert env.profiles.get("model-qwen3.8-flash")["settings"]["ModelPath"] == "/home/jay/models/Qwen3.8-Flash-Next-NVFP4"


# ---- controller rulings F2, F3, F11, F12 and the Task 7 carry (bounded wait for the new hash) ----
def test_changing_a_loaded_freetoken_model_asks_to_restart(env):
    # F2: a FreeToken entry carries only the folder and the --profile id, so the switcher
    # block does not move; the profile settings do, and a loaded model runs the old ones.
    revision = seed(env)
    env.switcher.states = {"qwen3.8-flash": "ready"}
    before = env.cfg.read_text()
    settings = engine_settings(env.client.get("/api/panel/views/model/qwen3.8-flash").json())
    settings["KVDtype"] = "bf16"
    body = {"revision": revision, "settings": settings, "identity": {}, "activePreset": None}
    asked = env.client.put("/api/panel/models/qwen3.8-flash", json=body)
    assert asked.status_code == 409 and asked.json()["code"] == "choose_restart"
    assert [row["id"] for row in asked.json()["affected"]] == ["qwen3.8-flash"]
    assert asked.json()["nextTimeAllowed"] is True
    restarted = env.client.put("/api/panel/models/qwen3.8-flash", json={**body, "whenLoaded": "restart"})
    assert restarted.status_code == 200, restarted.text
    assert restarted.json()["restarting"] == ["qwen3.8-flash"] and restarted.json()["held"] == []
    assert env.cfg.read_text() == before and env.switcher.calls == [("unload", "qwen3.8-flash")]
    run_spawned(env)
    assert env.switcher.calls[-1] == ("load", "qwen3.8-flash") and env.service.last_restart["ok"] is True
    effective = env.client.get("/api/panel/models/qwen3.8-flash/effective").json()
    assert effective["settings"]["KVDtype"] == "bf16"


def test_next_time_on_a_loaded_freetoken_model_needs_no_hold(env):
    revision = seed(env)
    env.switcher.states = {"qwen3.8-flash": "ready"}
    settings = engine_settings(env.client.get("/api/panel/views/model/qwen3.8-flash").json())
    settings["KVDtype"] = "bf16"
    saved = env.client.put("/api/panel/models/qwen3.8-flash", json={
        "revision": revision, "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "next-time"})
    assert saved.status_code == 200, saved.text
    assert saved.json()["restarting"] == [] and saved.json()["held"] == [] and env.switcher.calls == []


def test_an_unrelated_save_does_not_ask_again_about_a_held_model(env):
    # F3: "affected" compares the registry before and after, not the file (which still
    # carries the held entry), and the hold carries over untouched.
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    held = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "next-time"})
    assert held.status_code == 200 and held.json()["held"] == ["quasar-27b"]
    unrelated = env.client.put("/api/panel/system", json={"revision": held.json()["revision"], "system": {"floorGB": 8}})
    assert unrelated.status_code == 200, unrelated.text
    assert unrelated.json()["held"] == ["quasar-27b"] and env.switcher.calls == []
    text = env.cfg.read_text()
    assert "floorGB: 8" in text and "--draft-tokens 7" in extract_model_blocks(text)["quasar-27b"]
    env.switcher.states = {}
    assert env.service.release_finished_holds() == ["quasar-27b"]
    assert "--draft-tokens 5" in extract_model_blocks(env.cfg.read_text())["quasar-27b"]


def test_a_second_change_to_a_held_model_keeps_the_running_entry(env):
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    body = {"revision": revision, "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "next-time"}
    first = env.client.put("/api/panel/models/quasar-27b", json=body).json()
    settings["draft-tokens"] = 6
    again = env.client.put("/api/panel/models/quasar-27b", json={**body, "revision": first["revision"], "whenLoaded": None})
    assert again.status_code == 409 and again.json()["code"] == "choose_restart"
    second = env.client.put("/api/panel/models/quasar-27b", json={**body, "revision": first["revision"]})
    assert second.status_code == 200 and second.json()["held"] == ["quasar-27b"]
    assert "--draft-tokens 7" in extract_model_blocks(env.cfg.read_text())["quasar-27b"]


def test_import_reads_the_current_boot_file(env, tmp_path):
    # F12: the helper's active boot file (a profile may have switched it), not the default one.
    current = tmp_path / "boot-current.ps1"
    current.write_text("& $launcher `\n    -KVDtype 'bf16' `\n    -Port 2020\n", encoding="utf-8")
    env.service._boot_file = lambda: BootFile(current)
    env.cfg.write_text(EXAMPLE)
    answer = env.client.post("/api/panel/import", json={})
    assert answer.status_code == 200, answer.text
    assert env.store.load()[0]["engines"]["freetoken"]["defaults"]["KVDtype"] == "bf16"


def test_restart_gives_up_when_the_switcher_keeps_the_old_file(env):
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    env.switcher.config_hash = lambda: "stale"
    ticks = iter(range(0, 1000))
    env.service._clock = lambda: next(ticks)
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    answer = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "restart"})
    assert answer.status_code == 200, answer.text
    run_spawned(env)
    assert env.switcher.calls == [("unload", "quasar-27b")]
    assert env.service.last_restart["ok"] is False
    assert env.service.last_restart["message"] == (
        "The switcher didn't pick up the new settings; the old ones are still in use. Check the switcher log.")


def test_freetoken_fit_lists_only_the_empty_card_components(env):
    # F11: the planner also reports the "now" scenario (what is on the card today); the
    # fit box is about an empty card, so only "empty" and "both" rows are shown.
    seed(env)
    components = [
        {"name": "Weights", "resource": "vram", "scenario": "both", "bytes": 10 * GIB},
        {"name": "KV now", "resource": "vram", "scenario": "now", "bytes": 1 * GIB},
        {"name": "KV empty", "resource": "vram", "scenario": "empty", "bytes": 2 * GIB},
        {"name": "Host", "resource": "ram", "scenario": "both", "bytes": 3 * GIB},
    ]
    original = env.estimates.estimate_settings

    def with_components(settings, **kwargs):
        return {**original(settings, **kwargs), "components": components}

    env.estimates.estimate_settings = with_components
    view = env.client.get("/api/panel/views/model/qwen3.8-flash").json()
    fit = env.client.post("/api/panel/models/qwen3.8-flash/fit", json={"settings": engine_settings(view), "identity": {}}).json()
    assert [c["label"] for c in fit["components"]] == ["Weights", "KV empty"]

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
from freetoken.daemon.settings.switcher import DOWN, SwitcherError
from tests.settings.registry_fixtures import five

GIB = 1024 ** 3
REPO = Path(__file__).resolve().parents[2]
EXAMPLE = (REPO / "engines" / "config" / "config.example.yaml").read_text(encoding="utf-8")


class FakeSwitcher:
    """up=False alone is "unknown" (a timeout or HTTP error); up=False with refused=True is
    "down" (connection refused)."""

    def __init__(self, cfg: Path):
        self.cfg, self.states, self.up, self.calls, self.load_error = cfg, {}, True, [], None
        self.refused, self.unload_ok = False, True

    def running(self):
        if not self.up:
            return DOWN if self.refused else None
        return dict(self.states)

    def config_hash(self):
        return config_sha256(self.cfg.read_text()) if self.up and self.cfg.exists() else None

    def unload(self, model_id):
        self.calls.append(("unload", model_id))
        if not self.unload_ok:
            return False
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
    assert body["switcher"] == {"up": False, "running": [], "stale": False}
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


# ---- fix round 1: an unanswered /running is "unknown", not "nothing loaded" ----
def hold_quasar(env):
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    held = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "next-time"})
    assert held.status_code == 200 and held.json()["held"] == ["quasar-27b"]
    return held.json()["revision"]


def quasar_block(env):
    return extract_model_blocks(env.cfg.read_text())["quasar-27b"]


def test_a_switcher_blip_does_not_release_a_hold(env):
    hold_quasar(env)
    env.switcher.up = False
    assert env.service.release_finished_holds() == []
    assert "--draft-tokens 7" in quasar_block(env) and env.client.get("/api/panel/now").json()["held"] == ["quasar-27b"]
    env.switcher.up = True
    assert env.service.release_finished_holds() == []  # still loaded
    env.switcher.states = {}
    assert env.service.release_finished_holds() == ["quasar-27b"]
    assert "--draft-tokens 5" in quasar_block(env)


def test_sync_config_with_the_switcher_down_keeps_holds(env):
    hold_quasar(env)
    env.switcher.up = False
    env.service.sync_config()
    assert "--draft-tokens 7" in quasar_block(env)
    assert env.client.get("/api/panel/now").json()["held"] == ["quasar-27b"]


def test_restore_with_the_switcher_down_keeps_holds(env):
    hold_quasar(env)
    env.switcher.up = False
    backup = env.store.backups()[0]
    assert env.client.post("/api/panel/registry/restore", json={"backup": backup}).status_code == 200
    assert "--draft-tokens 7" in quasar_block(env)
    assert env.client.get("/api/panel/now").json()["held"] == ["quasar-27b"]


def test_a_save_while_the_switcher_is_unknown_keeps_existing_holds(env):
    revision = hold_quasar(env)
    env.switcher.up = False
    other = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 8}})
    assert other.status_code == 200, other.text
    assert other.json()["held"] == ["quasar-27b"] and "--draft-tokens 7" in quasar_block(env)
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 6
    again = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": other.json()["revision"], "settings": settings, "identity": {}, "activePreset": None})
    assert again.status_code == 200, again.text
    assert again.json()["held"] == ["quasar-27b"] and "--draft-tokens 7" in quasar_block(env)
    env.switcher.up, env.switcher.states = True, {}
    assert env.service.release_finished_holds() == ["quasar-27b"]
    assert "--draft-tokens 6" in quasar_block(env)


def test_import_is_refused_while_the_switcher_state_is_unknown(env):
    env.cfg.write_text(EXAMPLE)
    env.switcher.up = False
    refused = env.client.post("/api/panel/import", json={"whenLoaded": "restart"})
    assert refused.status_code == 503 and refused.json()["code"] == "switcher_unknown"
    assert refused.json()["message"].startswith("Can't tell whether a model is loaded right now")
    assert not env.store.exists() and env.cfg.read_text() == EXAMPLE and env.switcher.calls == []
    env.switcher.up = True
    assert env.client.post("/api/panel/import", json={}).status_code == 200


# ---- fix round 2: while the state is unknown, a save that moves an unheld entry is refused ----
def test_a_save_moving_a_possibly_loaded_model_is_refused_while_unknown(env):
    revision = seed(env)
    env.switcher.states = {"fable-27b": "ready"}
    settings = engine_settings(env.client.get("/api/panel/views/model/fable-27b").json())
    settings["draft-tokens"] = 3
    before, backups = env.cfg.read_text(), env.store.backups()
    env.switcher.up = False
    refused = env.client.put("/api/panel/models/fable-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "activePreset": None})
    assert refused.status_code == 503 and refused.json()["code"] == "switcher_unknown"
    assert refused.json()["message"] == ("Can't tell whether Fable 27B NVFP4 (NInfer) is loaded right now, "
                                         "so saving could restart it. Try again in a moment.")
    assert env.cfg.read_text() == before and env.store.backups() == backups
    assert env.store.load()[1] == revision and env.switcher.calls == []


def test_an_engine_default_change_while_unknown_names_every_moved_model(env):
    revision = seed(env)
    env.switcher.up = False
    settings = dict(env.client.get("/api/panel/views/engine/ninfer").json()["settings"])
    settings["max-concurrency"] = 5
    refused = env.client.put("/api/panel/engines/ninfer/defaults", json={"revision": revision, "settings": settings})
    assert refused.status_code == 503 and refused.json()["code"] == "switcher_unknown"
    assert "QUASAR" in refused.json()["message"] and "Twin 27B" in refused.json()["message"]
    assert refused.json()["message"].endswith("so saving could restart one of them. Try again in a moment.")


def test_a_system_only_save_while_unknown_still_saves(env):
    revision = seed(env)
    env.switcher.up = False
    saved = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 9}})
    assert saved.status_code == 200, saved.text
    assert "floorGB: 9" in env.cfg.read_text() and env.store.load()[0]["system"]["floorGB"] == 9


# ---- final whole-branch review fix wave ----
def switcher_down(env):
    env.switcher.up, env.switcher.refused = False, True


def test_switcher_client_tells_down_from_unknown():
    # Item 1: a refused connection is "down" (nothing loaded); a timeout, an HTTP error or a
    # bad body stays "unknown" (None).
    import io
    import socket
    import urllib.error

    from freetoken.daemon.settings.switcher import SwitcherClient, is_down

    def raising(exc):
        def urlopen(request, timeout):
            raise exc
        return urlopen

    class Response:
        def __init__(self, body):
            self.status, self._body = 200, body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self._body

    def answering(body):
        return lambda request, timeout: Response(body)

    assert is_down(SwitcherClient(urlopen=raising(urllib.error.URLError(ConnectionRefusedError(111, "refused")))).running())
    assert is_down(SwitcherClient(urlopen=raising(ConnectionRefusedError(111, "refused"))).running())
    assert SwitcherClient(urlopen=raising(urllib.error.URLError(socket.timeout("timed out")))).running() is None
    assert SwitcherClient(urlopen=raising(TimeoutError("timed out"))).running() is None
    http_500 = urllib.error.HTTPError("http://x/running", 500, "boom", {}, io.BytesIO(b"{}"))
    assert SwitcherClient(urlopen=raising(http_500)).running() is None
    assert SwitcherClient(urlopen=answering(b"not json")).running() is None
    up = SwitcherClient(urlopen=answering(b'{"running": [{"model": "q", "state": "ready"}]}')).running()
    assert up == {"q": "ready"} and not is_down(up)
    empty = SwitcherClient(urlopen=answering(b'{"running": []}')).running()
    assert empty == {} and not is_down(empty)


def test_a_save_while_the_switcher_is_down_saves_and_writes(env):
    # Item 1 (critical): spec error table, "settings still save and apply on its next start".
    revision = seed(env)
    switcher_down(env)
    settings = engine_settings(env.client.get("/api/panel/views/model/fable-27b").json())
    settings["draft-tokens"] = 3
    saved = env.client.put("/api/panel/models/fable-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "activePreset": None})
    assert saved.status_code == 200, saved.text
    assert "--draft-tokens 3" in extract_model_blocks(env.cfg.read_text())["fable-27b"]
    assert env.switcher.calls == [] and saved.json()["restarting"] == []
    assert env.client.get("/api/panel/now").json()["switcher"]["up"] is False
    rows = env.client.get("/api/panel/models").json()
    assert rows["switcherUp"] is False and {row["state"] for row in rows["models"]} == {"unknown"}
    assert env.client.get("/api/panel/views/model/fable-27b").json()["state"] == "unknown"


def test_holds_stay_while_the_switcher_is_down(env):
    revision = hold_quasar(env)
    switcher_down(env)
    assert env.service.release_finished_holds() == []
    other = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 8}})
    assert other.status_code == 200 and other.json()["held"] == ["quasar-27b"]
    assert "--draft-tokens 7" in quasar_block(env)
    env.switcher.up, env.switcher.states = True, {}
    assert env.service.release_finished_holds() == ["quasar-27b"]
    assert "--draft-tokens 5" in quasar_block(env)


def test_import_while_the_switcher_is_down_goes_ahead(env):
    env.cfg.write_text(EXAMPLE)
    switcher_down(env)
    answer = env.client.post("/api/panel/import", json={})
    assert answer.status_code == 200, answer.text
    assert answer.json()["restarting"] == [] and env.switcher.calls == []
    assert env.cfg.read_text().startswith("# generated by the control panel")


def test_restore_while_a_model_is_loaded_keeps_its_running_entry(env):
    # Item 2 (critical): restore rewrote a loaded, unheld model's entry, so P5 stopped it.
    revision = seed(env)
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    saved = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "activePreset": None})
    assert saved.status_code == 200 and saved.json()["held"] == []
    env.switcher.states = {"quasar-27b": "ready"}
    restored = env.client.post("/api/panel/registry/restore", json={"backup": env.store.backups()[0]})
    assert restored.status_code == 200, restored.text
    assert restored.json()["held"] == ["quasar-27b"]
    assert "--draft-tokens 5" in quasar_block(env)  # what is running stays in the file
    assert find_model(env.store.load()[0], "quasar-27b")["overrides"]["draft-tokens"] == 7
    assert env.switcher.calls == [] and env.client.get("/api/panel/now").json()["held"] == ["quasar-27b"]
    env.switcher.states = {}
    assert env.service.release_finished_holds() == ["quasar-27b"]
    assert "--draft-tokens 7" in quasar_block(env)


def _move_entries_behind_the_files_back(env, revision):
    # What a catalogue/generator change or a crash between store.save and the holds write
    # leaves behind: the registry moved two entries, the file did not.
    doc, _ = env.store.load()
    find_model(doc, "quasar-27b").setdefault("overrides", {})["max-concurrency"] = 5
    find_model(doc, "fable-27b").setdefault("overrides", {})["max-concurrency"] = 5
    env.store.save(doc, expected_revision=revision)


def test_start_up_sync_holds_a_loaded_model_whose_entry_moved(env):
    revision = seed(env)
    _move_entries_behind_the_files_back(env, revision)
    env.switcher.states = {"quasar-27b": "ready"}
    env.service.sync_config()
    blocks = extract_model_blocks(env.cfg.read_text())
    assert "--max-concurrency 5" not in blocks["quasar-27b"] and "--max-concurrency 5" in blocks["fable-27b"]
    assert env.client.get("/api/panel/now").json()["held"] == ["quasar-27b"]
    env.switcher.states = {}
    assert env.service.release_finished_holds() == ["quasar-27b"]
    assert "--max-concurrency 5" in quasar_block(env)


def test_start_up_sync_while_unknown_holds_every_moved_entry(env):
    revision = seed(env)
    _move_entries_behind_the_files_back(env, revision)
    before = env.cfg.read_text()
    env.switcher.up = False
    env.service.sync_config()
    assert env.cfg.read_text() == before  # every moved entry is held on what the file had
    assert env.client.get("/api/panel/now").json()["held"] == ["fable-27b", "quasar-27b"]
    env.switcher.up, env.switcher.states = True, {"fable-27b": "ready"}
    assert env.service.release_finished_holds() == ["quasar-27b"]
    blocks = extract_model_blocks(env.cfg.read_text())
    assert "--max-concurrency 5" in blocks["quasar-27b"] and "--max-concurrency 5" not in blocks["fable-27b"]


def test_a_refused_restore_changes_nothing(env):
    # Item 2 fold-in: restore used to restore the registry, then answer 422 when the switcher
    # refused the file. Every check now runs first.
    first = seed(env)
    second = env.client.put("/api/panel/system", json={"revision": first, "system": {"floorGB": 7}}).json()["revision"]
    before = env.cfg.read_text()
    env.writer._runner = checker(ok=False)
    refused = env.client.post("/api/panel/registry/restore", json={"backup": env.store.backups()[0]})
    assert refused.status_code == 422 and refused.json()["code"] == "switcher_refused"
    assert env.store.load()[1] == second and env.cfg.read_text() == before


def test_restore_over_a_file_the_panel_did_not_write_asks_to_unload_first(env):
    first = seed(env)
    second = env.client.put("/api/panel/system", json={"revision": first, "system": {"floorGB": 7}}).json()["revision"]
    env.cfg.write_text(EXAMPLE)
    env.switcher.states = {"quasar-27b": "ready"}
    refused = env.client.post("/api/panel/registry/restore", json={"backup": env.store.backups()[0]})
    assert refused.status_code == 409 and refused.json()["code"] == "unload_first"
    assert refused.json()["message"].startswith("Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2) is loaded")
    assert env.store.load()[1] == second and env.cfg.read_text() == EXAMPLE


def test_restart_now_saves_nothing_when_the_model_does_not_unload(env):
    # Item 4: an unload the switcher does not confirm used to save and report "restarting".
    revision = seed(env)
    env.switcher.states = {"quasar-27b": "ready"}
    env.switcher.unload_ok = False
    before = env.cfg.read_text()
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    answer = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "restart"})
    assert answer.status_code == 503 and answer.json()["code"] == "restart_failed"
    assert answer.json()["message"] == ("Couldn't put Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2) away to restart it, "
                                        "so nothing was saved. Try again in a moment.")
    assert env.store.load()[1] == revision and env.cfg.read_text() == before and env.spawned == []


def test_right_now_says_when_the_switcher_is_on_older_settings(env):
    # Item 5: the switcher's config hash (P5) differs from the file's.
    revision = seed(env)
    clock = [1000.0]
    env.service._clock = lambda: clock[0]
    env.switcher.config_hash = lambda: "older"
    assert env.client.get("/api/panel/now").json()["switcher"]["stale"] is True
    assert env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 8}}).status_code == 200
    assert env.client.get("/api/panel/now").json()["switcher"]["stale"] is False  # llama-swap polls every 2 s
    clock[0] += 11
    assert env.client.get("/api/panel/now").json()["switcher"]["stale"] is True
    env.switcher.config_hash = lambda: config_sha256(env.cfg.read_text())
    assert env.client.get("/api/panel/now").json()["switcher"]["stale"] is False
    env.switcher.config_hash = lambda: None  # an older switcher without P5: say nothing
    assert env.client.get("/api/panel/now").json()["switcher"]["stale"] is False


def test_a_failed_restart_note_clears_on_the_next_save_or_after_ten_minutes(env):
    # Item 6.
    revision = seed(env)
    clock = [0.0]
    env.service._clock = lambda: clock[0]
    env.service._sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    env.switcher.states = {"quasar-27b": "ready"}
    env.switcher.config_hash = lambda: "stale"
    settings = engine_settings(env.client.get("/api/panel/views/model/quasar-27b").json())
    settings["draft-tokens"] = 5
    saved = env.client.put("/api/panel/models/quasar-27b", json={
        "revision": revision, "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "restart"})
    assert saved.status_code == 200
    run_spawned(env)
    assert env.client.get("/api/panel/now").json()["lastRestart"]["ok"] is False
    clock[0] += 599
    assert env.client.get("/api/panel/now").json()["lastRestart"] is not None
    clock[0] += 2
    assert env.client.get("/api/panel/now").json()["lastRestart"] is None
    env.service._restart_done(["quasar-27b"], False, "Restarting quasar-27b failed: boom")
    assert env.client.get("/api/panel/now").json()["lastRestart"] is not None
    assert env.client.put("/api/panel/system", json={"revision": saved.json()["revision"],
                                                     "system": {"floorGB": 8}}).status_code == 200
    assert env.client.get("/api/panel/now").json()["lastRestart"] is None
    env.service._restart_done(["quasar-27b"], False, "x")
    assert env.client.post("/api/panel/models/fable-27b/load").status_code == 200
    assert env.client.get("/api/panel/now").json()["lastRestart"] is None


def test_import_refuses_settings_a_model_could_never_save(env, monkeypatch):
    # Item 8: every save checks each FreeToken model's own limits; an import that breaks one
    # would make every later save fail, so it is refused with the reasons.
    env.cfg.write_text(EXAMPLE)
    monkeypatch.setattr(env.service, "_model_limit_errors", lambda doc: [
        {"field": "KVCacheTokens", "message": "More than this model's longest chat.", "where": "qwen3.8-flash"}])
    refused = env.client.post("/api/panel/import", json={})
    assert refused.status_code == 409 and refused.json()["code"] == "import_refused"
    message = refused.json()["message"]
    assert message.startswith("Nothing was copied, because these settings can't be used:")
    assert "More than this model's longest chat." in message and "qwen3.8-flash:" not in message  # named, not id'd
    assert message.endswith("Change them in the helper's start-up file and try again.")
    assert not env.store.exists() and env.cfg.read_text() == EXAMPLE and env.switcher.calls == []


def test_status_route_runs_in_the_threadpool(env):
    # Open item: /api/status probes the model server (~4 s when it is down); as an async
    # route it blocked the event loop and every panel request with it.
    import inspect

    route = next(r for r in env.client.app.routes if getattr(r, "path", "") == "/api/status")
    assert not inspect.iscoroutinefunction(route.endpoint)
    assert env.client.get("/api/status").json()["helper"]["status"] == "up"


def test_ninfer_fit_warns_when_ninfer_would_refuse_to_start(env):
    # Fit round 2 (2026-09-25): the used-memory estimate can look fine while NInfer's up-front
    # runtime reservation does not fit; the route must say "won't fit" with the plain reason.
    seed(env)
    view = env.client.get("/api/panel/views/model/quasar-27b").json()
    settings = {**engine_settings(view), "max-concurrency": 8}
    fit = env.client.post("/api/panel/models/quasar-27b/fit", json={"settings": settings, "identity": {}}).json()
    assert fit["verdict"] == "wont_fit"
    assert fit["runtimeReservationBytes"] > fit["runtimeRoomBytes"] > 0
    assert fit["message"].startswith("NInfer would refuse to start")


def _quasar_room(env, running, refused=False):
    seed(env)
    if running is None:
        env.switcher.up, env.switcher.refused = False, refused
    else:
        env.switcher.states = running
    env.service._card_probe = lambda: {"totalBytes": 32 * GIB, "usedBytes": 4 * GIB}
    view = env.client.get("/api/panel/views/model/quasar-27b").json()
    fit = env.client.post("/api/panel/models/quasar-27b/fit", json={"settings": engine_settings(view), "identity": {}}).json()
    return fit["runtimeRoomBytes"]


def test_startup_room_uses_the_live_desktop_only_when_nothing_is_loaded(env):
    from freetoken.daemon.settings.ninfer_fit import startup_room
    fixed = startup_room(19_782_132_224, 32 * GIB)
    assert _quasar_room(env, {}) == startup_room(19_782_132_224, 32 * GIB, 4 * GIB) < fixed


def test_startup_room_keeps_the_fixed_desktop_when_a_model_is_loaded(env):
    from freetoken.daemon.settings.ninfer_fit import startup_room
    assert _quasar_room(env, {"quasar-27b": "ready"}) == startup_room(19_782_132_224, 32 * GIB)


def test_startup_room_keeps_the_fixed_desktop_when_the_switcher_state_is_unknown(env):
    from freetoken.daemon.settings.ninfer_fit import startup_room
    assert _quasar_room(env, None) == startup_room(19_782_132_224, 32 * GIB)


def test_startup_room_keeps_the_fixed_desktop_when_the_switcher_is_down(env):
    from freetoken.daemon.settings.ninfer_fit import startup_room
    assert _quasar_room(env, None, refused=True) == startup_room(19_782_132_224, 32 * GIB)

"""The panel's test overlay, guards and marker (Test tab). Review focus 1 and 3."""

from __future__ import annotations

import pytest

from freetoken.daemon.settings.panel import PanelError
from freetoken.daemon.settings.registry import find_model
from freetoken.daemon.settings.swap_config import extract_model_blocks, render_config
from tests.settings.test_panel_routes import env, seed  # noqa: F401 - the shared fixture
from tests.settings.registry_fixtures import five


def with_presets():
    doc = five()
    find_model(doc, "quasar-27b")["presets"] = {
        "Fast": {"kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 3},
        "Same": {"kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 7},
    }
    find_model(doc, "qwen3.8-flash")["presets"] = {"Short": {"KVCacheTokens": 131072}}
    return doc


def test_overlay_renders_the_preset_alone_and_clearing_restores(env):
    seed(env, with_presets())
    registry = env.store.path.read_bytes()
    text = env.service.set_test_settings("quasar-27b", "Fast")
    assert env.cfg.read_text() == text
    assert "--draft-tokens 3" in extract_model_blocks(text)["quasar-27b"]
    assert env.service.read_test_marker()["model"] == "quasar-27b"
    assert env.store.path.read_bytes() == registry  # the registry is never written
    back = env.service.set_test_settings(None, None)
    assert back == render_config(env.store.load()[0], {}) == env.cfg.read_text()
    assert env.service.read_test_marker() is None and env.service.test_settings is None


def test_unknown_preset_is_refused(env):
    seed(env, with_presets())
    with pytest.raises(KeyError):
        env.service.set_test_settings("quasar-27b", "Nope")
    assert env.service.test_settings is None


def test_effective_follows_the_overlay(env):
    seed(env, with_presets())
    read = lambda: env.client.get("/api/panel/models/qwen3.8-flash/effective").json()["settings"]["KVCacheTokens"]
    assert read() == 262208
    env.service.set_test_settings("qwen3.8-flash", "Short")
    assert read() == 131072
    env.service.set_test_settings(None, None)
    assert read() == 262208


def test_a_preset_equal_to_saved_needs_no_restart(env):
    seed(env, with_presets())
    assert env.service.test_preset_key("quasar-27b", "Same") is None
    assert env.service.test_preset_key("quasar-27b", "Fast") == "Fast"
    assert env.service.test_preset_key("quasar-27b", None) is None
    assert env.service.test_preset_key("qwen3.8-flash", "Short") == "Short"


def test_every_panel_write_is_refused_while_testing(env):
    revision = seed(env, with_presets())
    env.service.begin_test()
    with pytest.raises(PanelError) as again:
        env.service.begin_test()
    assert again.value.payload["code"] == "test_running"
    save = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7}})
    assert save.status_code == 409 and save.json()["code"] == "test_running"
    assert "Test tab" in save.json()["message"]
    assert env.client.post("/api/panel/models/twin-27b/load").status_code == 409
    assert env.client.post("/api/panel/models/twin-27b/unload").status_code == 409
    assert env.client.post("/api/panel/registry/restore", json={"backup": "x"}).status_code == 409
    assert env.switcher.calls == []
    env.service.end_test()
    assert env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7}}).status_code == 200


def test_wait_for_switcher(env):
    seed(env, with_presets())
    assert env.service.wait_for_switcher(env.cfg.read_text()) is True
    env.service.restart_wait_s = 0
    assert env.service.wait_for_switcher("something else") is False


def test_now_shows_the_test_and_prunes_a_leftover_once_unloaded(env):
    seed(env, with_presets())
    env.switcher.states = {"quasar-27b": "ready"}
    env.service.begin_test()
    env.service.set_test_settings("quasar-27b", "Fast")
    now = env.client.get("/api/panel/now").json()
    assert now["test"] == {"running": True, "model": "quasar-27b",
                           "name": "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)", "preset": "Fast"}
    env.service.note_test_leftover("quasar-27b", "Fast")
    env.service.end_test()
    now = env.client.get("/api/panel/now").json()
    assert now["test"] is None and now["testLeftover"]["preset"] == "Fast"
    env.switcher.states = {}
    assert env.client.get("/api/panel/now").json()["testLeftover"] is None

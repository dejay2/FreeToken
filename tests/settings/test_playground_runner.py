"""The Test tab runner: plan (Task 4); run, put-back, stop and recover (Task 5)."""

from __future__ import annotations

import pytest

from freetoken.daemon.settings.playground import WARMUP_MESSAGES, ChatFailed, PlaygroundError
from freetoken.daemon.settings.swap_config import extract_model_blocks, render_config
from freetoken.daemon.settings.switcher import SwitcherError
from tests.settings.playground_fakes import make

QUASAR = "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)"
FABLE = "Fable 27B NVFP4 (NInfer)"
TWIN = "Twin 27B NVFP4 (NInfer)"


def kinds(plan):
    return [step["kind"] for step in plan["steps"]]


def test_plan_model_vs_model(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    plan = env.runner.plan({"prompt": "Hi", "sides": [{"model": "fable-27b"}, {"model": "twin-27b"}]})
    assert plan["before"] == "quasar-27b" and plan["held"] is False and plan["warnings"] == []
    assert kinds(plan) == ["unload", "settings", "load", "warmup", "answer",
                           "unload", "settings", "load", "warmup", "answer", "restore"]
    assert [step["label"] for step in plan["steps"][:3]] == [
        f"Put away {QUASAR}", f"Use saved settings for {FABLE}", f"Load {FABLE}"]
    assert plan["steps"][-1]["label"] == f"Put things back: {QUASAR} on its saved settings"
    # 5+0+20+5+30 twice, then put-back: put Twin away (5) and load QUASAR (20)
    assert plan["estimateS"] == 145
    assert plan["sides"][0]["sampling"] == {"max_tokens": 512}


def test_a_setup_already_on_the_card_is_not_reloaded(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    plan = env.runner.plan({"prompt": "Hi", "sides": [{"model": "quasar-27b", "preset": "Same"}], "warmup": False})
    assert kinds(plan) == ["answer", "restore"] and plan["steps"][-1]["guessS"] == 0
    side = plan["sides"][0]
    assert side["runPreset"] is None and side["settingsLabel"] == "preset “Same”"


def test_a_model_on_old_settings_is_reloaded_even_for_saved_settings(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.service._write_holds({"quasar-27b": "  # --- model quasar-27b ---\n"})
    plan = env.runner.plan({"prompt": "Hi", "sides": [{"model": "quasar-27b"}], "warmup": False})
    assert plan["held"] is True and kinds(plan) == ["unload", "settings", "load", "answer", "restore"]


def test_sampling_is_read_and_blank_means_the_models_own(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    plan = env.runner.plan({"prompt": "Hi", "sides": [
        {"model": "fable-27b", "temperature": "0.7", "top_p": "", "top_k": "20", "maxTokens": "256"}]})
    assert plan["sides"][0]["sampling"] == {"temperature": 0.7, "top_k": 20, "max_tokens": 256}
    assert plan["before"] is None and kinds(plan)[:2] == ["settings", "load"]


def test_plan_refuses_when_the_loaded_model_is_answering(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.probe.rows = [{"model": "quasar-27b", "req_headers": {"X-Session-ID": "claude-code"}}]
    with pytest.raises(PlaygroundError) as refused:
        env.runner.plan({"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert refused.value.status == 409
    assert refused.value.payload == {"code": "in_use", "message": f"{QUASAR} is answering something right now. Try again when it's done."}
    env.probe.unknown = True
    with pytest.raises(PlaygroundError) as unknown:
        env.runner.plan({"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert unknown.value.status == 503


def test_a_request_by_alias_counts_as_answering(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"qwen3.8-flash": "ready"})
    env.probe.rows = [{"model": "Qwen3.8-Flash-Next-NVFP4", "req_headers": {}}]
    with pytest.raises(PlaygroundError) as refused:
        env.runner.plan({"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert refused.value.payload["code"] == "in_use"


def test_plan_warns_when_the_loaded_model_was_used_recently(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.probe.used = {"quasar-27b": env.clock.wall() - 40}
    plan = env.runner.plan({"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert plan["warnings"] == [f"{QUASAR} was used 40 seconds ago. The test will put it away."]
    assert env.runner.plan({"prompt": "Hi", "sides": [{"model": "quasar-27b"}]})["warnings"] == []  # it stays loaded
    env.probe.used = {"quasar-27b": env.clock.wall() - 300}
    assert env.runner.plan({"prompt": "Hi", "sides": [{"model": "fable-27b"}]})["warnings"] == []


def test_plan_refuses_while_the_card_is_changing_or_unknown(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"fable-27b": "starting"})
    body = {"prompt": "Hi", "sides": [{"model": "twin-27b"}]}
    with pytest.raises(PlaygroundError) as loading:
        env.runner.plan(body)
    assert (loading.value.status, loading.value.payload["code"]) == (409, "loading")
    env.switcher.up = False
    with pytest.raises(PlaygroundError) as unknown:
        env.runner.plan(body)
    assert unknown.value.payload["code"] == "switcher_unknown"
    env.switcher.refused = True
    with pytest.raises(PlaygroundError) as down:
        env.runner.plan(body)
    assert down.value.payload == {"code": "switcher_down", "message": "The model switcher is not running."}


@pytest.mark.parametrize("body, code, words", [
    ({"prompt": "  ", "sides": [{"model": "fable-27b"}]}, "prompt", "Type a prompt first."),
    ({"prompt": "x" * 20_001, "sides": [{"model": "fable-27b"}]}, "prompt",
     "The prompt is too long: at most 20,000 characters."),
    ({"prompt": "Hi", "sides": []}, "sides", "Pick one or two setups."),
    ({"prompt": "Hi", "sides": [{"model": "fable-27b"}] * 3}, "sides", "Pick one or two setups."),
    ({"prompt": "Hi", "sides": [{"model": "nope"}]}, "model", "Setup A: pick a model."),
    ({"prompt": "Hi", "sides": [{"model": "fable-27b"}, {"model": "fable-27b", "preset": "Gone"}]}, "preset",
     "Setup B: the preset “Gone” no longer exists."),
    ({"prompt": "Hi", "sides": [{"model": "fable-27b", "temperature": 3}]}, "temperature",
     "Setup A: Creativity (temperature) must be between 0 and 2."),
    ({"prompt": "Hi", "sides": [{"model": "fable-27b", "top_k": "lots"}]}, "top_k", "Setup A: Top-k must be a number."),
    ({"prompt": "Hi", "sides": [{"model": "fable-27b", "maxTokens": 0}]}, "maxTokens",
     "Setup A: Longest answer must be between 1 and 4,096 tokens."),
])
def test_plan_refusals_in_plain_words(tmp_path, monkeypatch, body, code, words):
    env = make(tmp_path, monkeypatch)
    with pytest.raises(PlaygroundError) as refused:
        env.runner.plan(body)
    assert refused.value.status == 422 and refused.value.payload == {"code": code, "message": words}

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


def start(env, body):
    env.runner.start({"confirm": True, **body})
    return env.runner.snapshot()


def test_model_vs_model_runs_in_order_and_puts_back(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}, {"model": "twin-27b"}]})
    assert job["status"] == "done", job
    assert all(step["state"] == "done" for step in job["steps"])
    assert env.switcher.calls == [("unload", "quasar-27b"), ("load", "fable-27b"), ("unload", "fable-27b"),
                                  ("load", "twin-27b"), ("unload", "twin-27b"), ("load", "quasar-27b")]
    assert env.switcher.states == {"quasar-27b": "ready"}
    assert job["restore"] == f"{QUASAR} is loaded again on its saved settings (20.0 s)."
    assert [side["answer"] for side in job["sides"]] == ["Hello there friend"] * 2
    assert set(env.chat.sessions) == {"ft-test-" + job["id"]}
    body = env.chat.answers()[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True} and body["max_tokens"] == 512
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert env.service.test_running is False


def test_load_time_is_its_own_step(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.switcher.load_s = {"fable-27b": 20}
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    side = job["sides"][0]
    assert side["loadMs"] == 20000
    assert next(step for step in job["steps"] if step["kind"] == "load")["ms"] == 20000
    assert side["stats"]["firstWordMs"] == 200 and side["stats"]["totalMs"] == 300  # the warm-up is not counted
    assert job["restore"] == "Nothing was loaded before the test." and env.switcher.states == {}


def test_preset_compare_never_writes_registry_and_restores_config(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    registry = env.store.path.read_bytes()
    job = start(env, {"prompt": "Hi", "sides": [{"model": "quasar-27b", "preset": "Fast"}, {"model": "quasar-27b"}]})
    assert job["status"] == "done"
    first, second = (extract_model_blocks(text)["quasar-27b"] for text in env.switcher.files_at_load["quasar-27b"])
    assert "--draft-tokens 3" in first and "--draft-tokens 7" in second
    assert env.store.path.read_bytes() == registry
    assert env.cfg.read_text() == render_config(env.store.load()[0], {})
    assert env.service.test_settings is None and env.service.read_test_marker() is None
    assert env.switcher.calls == [("unload", "quasar-27b"), ("load", "quasar-27b"),
                                  ("unload", "quasar-27b"), ("load", "quasar-27b")]
    assert job["restore"] == f"{QUASAR} is loaded on its saved settings, as before."


def test_freetoken_preset_reaches_the_adapter_through_effective(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    read = lambda m: env.client.get(f"/api/panel/models/{m}/effective").json()["settings"]["KVCacheTokens"]
    seen = []
    env.switcher.on_load = lambda m: seen.append(read(m))
    job = start(env, {"prompt": "Hi", "sides": [{"model": "qwen3.8-flash", "preset": "Short"}]})
    assert job["status"] == "done" and seen == [131072]
    assert read("qwen3.8-flash") == 262208
    assert env.switcher.states == {}  # nothing was loaded before, so the test model is put away


def test_answer_error_still_puts_back(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.chat.errors["twin-27b"] = ChatFailed(500, "chat_failed", "engine crashed")
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}, {"model": "twin-27b", "preset": None}]})
    assert job["status"] == "failed" and job["message"] == f"{TWIN} could not answer: engine crashed"
    assert [step["state"] for step in job["steps"] if step["kind"] == "answer"] == ["done", "failed"]
    assert job["steps"][-1]["state"] == "done" and env.switcher.states == {"quasar-27b": "ready"}


def test_stop_during_load_cancels_it_and_puts_back(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.switcher.on_load = lambda m: env.runner.stop() if m == "fable-27b" else None
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b", "preset": "Three"}, {"model": "quasar-27b"}]})
    assert job["status"] == "stopped" and job["message"] == "Stopped."
    assert env.switcher.calls == [("unload", "quasar-27b"), ("load", "fable-27b"),
                                  ("unload", "fable-27b"), ("load", "quasar-27b")]
    states = [(step["kind"], step["state"]) for step in job["steps"]]
    assert states[:3] == [("unload", "done"), ("settings", "done"), ("load", "failed")]
    assert all(state == "skipped" for kind, state in states[3:-1]) and states[-1] == ("restore", "done")
    assert env.cfg.read_text() == render_config(env.store.load()[0], {}) and env.service.read_test_marker() is None


def test_stop_while_answering_keeps_the_partial_numbers(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.chat.on_stream = lambda body: env.runner.stop() if body["messages"] != WARMUP_MESSAGES else None
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "stopped" and job["sides"][0]["stats"]["finishReason"] == "cancelled"


def test_superseded_load_stops_and_does_not_load_over_the_other_app(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})

    def other_app(model_id):
        if model_id == "fable-27b":
            env.switcher.states = {"twin-27b": "ready"}  # another app asked for Twin; P1 cancels our load
            raise SwitcherError(409, "model_superseded", "superseded by twin-27b")

    env.switcher.on_load = other_app
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "yielded"
    assert job["message"] == "Another app asked for a different model, so the test stopped to let it through."
    assert env.switcher.calls == [("unload", "quasar-27b"), ("load", "fable-27b")]
    assert job["restore"] == f"{QUASAR} was not loaded again, because an app is using {TWIN}."
    assert env.switcher.states == {"twin-27b": "ready"}


def test_answer_refused_when_the_card_changed_after_loading(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.switcher.after_load = lambda m: setattr(env.switcher, "states", {"twin-27b": "ready"})
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "yielded" and env.chat.bodies == []
    assert job["restore"] == f"{QUASAR} was not loaded again, because an app is using {TWIN}."


def test_put_back_leaves_a_busy_test_model_and_holds_its_entry(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})

    def someone_starts(body):
        if body["messages"] != WARMUP_MESSAGES:
            env.probe.rows = [{"model": "quasar-27b", "req_headers": {"X-Session-ID": "someone"}}]

    env.chat.on_stream = someone_starts
    job = start(env, {"prompt": "Hi", "sides": [{"model": "quasar-27b", "preset": "Fast"}]})
    assert job["status"] == "done"
    assert env.switcher.calls == [("unload", "quasar-27b"), ("load", "quasar-27b")]
    assert job["restore"] == (f"{QUASAR} is still on test settings because an app is using it. "
                              "It goes back to its saved settings at its next load.")
    assert env.service.held_models() == ["quasar-27b"]
    assert "--draft-tokens 3" in extract_model_blocks(env.cfg.read_text())["quasar-27b"]
    assert env.client.get("/api/panel/now").json()["testLeftover"]["preset"] == "Fast"
    env.switcher.states = {}
    env.service.release_finished_holds()
    assert "--draft-tokens 7" in extract_model_blocks(env.cfg.read_text())["quasar-27b"]
    assert env.client.get("/api/panel/now").json()["testLeftover"] is None


def test_panel_writes_refused_while_testing(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    seen = {}

    def meddle(body):
        if body["messages"] != WARMUP_MESSAGES:
            revision = env.store.load()[1]
            seen["save"] = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7}})
            seen["load"] = env.client.post("/api/panel/models/twin-27b/load")
            try:
                env.runner.start({"prompt": "Again", "sides": [{"model": "twin-27b"}], "confirm": True})
            except PlaygroundError as exc:
                seen["start"] = exc.payload["code"]

    env.chat.on_stream = meddle
    start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert seen["save"].status_code == 409 and seen["save"].json()["code"] == "test_running"
    assert seen["load"].status_code == 409 and seen["start"] == "test_running"
    revision = env.store.load()[1]
    assert env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7}}).status_code == 200


def test_switcher_never_picks_up_settings_nothing_loads(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.switcher.stale = True
    job = start(env, {"prompt": "Hi", "sides": [{"model": "quasar-27b", "preset": "Fast"}]})
    assert job["status"] == "failed"
    assert job["message"] == "The switcher didn't pick up the test settings, so nothing was loaded."
    assert env.switcher.calls == [("unload", "quasar-27b")]  # neither the test load nor the put-back load
    assert job["steps"][-1]["state"] == "failed" and "nothing was loaded again" in job["restore"]
    assert env.service.test_settings is None and env.service.read_test_marker() is None


def test_helper_restart_puts_an_idle_test_model_away(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.service.set_test_settings("quasar-27b", "Fast")  # the helper stopped mid-test...
    env.switcher.states = {"quasar-27b": "ready"}         # ...with QUASAR on the test settings
    service, runner = env.build()
    assert runner.recover() == "unloaded"
    assert env.switcher.calls[-1] == ("unload", "quasar-27b") and service.read_test_marker() is None
    service.sync_config()
    assert env.cfg.read_text() == render_config(env.store.load()[0], {})


def test_helper_restart_leaves_a_busy_test_model_and_says_so(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.service.set_test_settings("quasar-27b", "Fast")
    env.switcher.states = {"quasar-27b": "ready"}
    env.probe.rows = [{"model": "quasar-27b", "req_headers": {}}]
    service, runner = env.build()
    assert runner.recover() == "left" and ("unload", "quasar-27b") not in env.switcher.calls
    service.sync_config()
    assert service.held_models() == ["quasar-27b"]
    assert "--draft-tokens 3" in extract_model_blocks(env.cfg.read_text())["quasar-27b"]
    assert service.now()["testLeftover"]["preset"] == "Fast"
    assert runner.recover() == "none"


def test_start_refused_as_test_running_when_the_panel_is_already_testing(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.service.begin_test()  # e.g. a second runner over the same panel
    with pytest.raises(PlaygroundError) as refused:
        env.runner.start({"prompt": "Hi", "sides": [{"model": "fable-27b"}], "confirm": True})
    assert refused.value.status == 409 and refused.value.payload["code"] == "test_running"
    assert env.runner.snapshot() == {"status": "idle"} and env.switcher.calls == []


def test_a_thread_that_cannot_start_ends_the_test(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)

    def no_threads(fn, *args):
        raise RuntimeError("can't start new thread")

    env.runner._spawn = no_threads
    with pytest.raises(RuntimeError):
        env.runner.start({"prompt": "Hi", "sides": [{"model": "fable-27b"}], "confirm": True})
    job = env.runner.snapshot()
    assert job["status"] == "failed" and env.service.test_running is False

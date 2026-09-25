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


# ---- review fixes ----

def test_a_model_starting_during_an_answer_makes_the_next_load_yield(tmp_path, monkeypatch):
    """Item 1: the card is read again right before each load."""
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})

    def other_app(body):
        if body["model"] == "fable-27b" and body["messages"] != WARMUP_MESSAGES:
            env.switcher.states["quasar-27b"] = "starting"  # another app asked for QUASAR

    env.chat.on_stream = other_app
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}, {"model": "twin-27b"}]})
    assert job["status"] == "yielded"
    assert job["message"] == f"Another app started using {QUASAR}, so the test stopped to let it through."
    assert ("load", "twin-27b") not in env.switcher.calls
    assert env.switcher.states == {"quasar-27b": "starting"}


def test_a_model_starting_during_the_hash_wait_makes_the_load_yield(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    real = env.switcher.config_hash

    def hash_then_other_app():
        env.switcher.states.setdefault("twin-27b", "starting")
        return real()

    env.switcher.config_hash = hash_then_other_app
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "yielded" and ("load", "fable-27b") not in env.switcher.calls
    assert job["restore"] == "Nothing was loaded before the test."


def test_put_back_does_not_load_over_a_model_started_during_its_hash_wait(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    real = env.switcher.config_hash

    def hash_then_other_app():
        if env.runner.job and env.runner.job["status"] == "restoring":
            env.switcher.states.setdefault("twin-27b", "starting")
        return real()

    env.switcher.config_hash = hash_then_other_app
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "done"
    assert job["restore"] == f"{QUASAR} was not loaded again, because an app is using {TWIN}."
    assert env.switcher.calls[-1] == ("unload", "fable-27b")  # no put-back load


def test_a_request_between_the_inflight_read_and_the_unload_is_not_killed(tmp_path, monkeypatch):
    """Item 3: the unload is P7's if-idle unload; llama-swap refusing it as busy yields."""
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.probe.busy = {"quasar-27b"}  # the in-flight read showed nothing; llama-swap holds a request
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "yielded"
    assert job["message"] == (f"{QUASAR} started answering something for another app, "
                              "so the test stopped to leave it alone.")
    assert env.switcher.calls == [("busy", "quasar-27b")] and env.switcher.states == {"quasar-27b": "ready"}
    assert job["restore"] == f"{QUASAR} is loaded on its saved settings, as before."


def test_put_back_keeps_a_test_model_llama_swap_says_is_busy(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})

    def someone_starts(body):
        if body["messages"] != WARMUP_MESSAGES:
            env.probe.busy = {"quasar-27b"}

    env.chat.on_stream = someone_starts
    job = start(env, {"prompt": "Hi", "sides": [{"model": "quasar-27b", "preset": "Fast"}]})
    assert env.switcher.calls[-1] == ("busy", "quasar-27b")
    assert job["restore"].startswith(f"{QUASAR} is still on test settings because an app is using it.")
    assert env.service.test_leftover == {"model": "quasar-27b", "preset": "Fast"}


def test_a_stop_between_steps_never_starts_the_answer(tmp_path, monkeypatch):
    """Item 4: Stop lands after the step check (here while /running is read for the answer)."""
    env = make(tmp_path, monkeypatch)
    real = env.switcher.running

    def running():
        if env.chat.bodies and env.runner.job["status"] == "running":
            env.runner.stop()
        return real()

    env.switcher.running = running
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "stopped" and env.chat.answers() == []


def test_a_new_test_clears_the_last_stop(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.chat.aborted = True  # left by the last test's Stop
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "done" and job["sides"][0]["answer"] == "Hello there friend"


def test_a_stop_before_the_load_registers_puts_the_model_away_again(tmp_path, monkeypatch):
    """Item 5: Stop's unload reached llama-swap before P6 registered the load, so the load
    finished; the model is put away again even when put-back would leave it."""
    env = make(tmp_path, monkeypatch)

    def stop_too_early(model_id):
        env.switcher.loading = None  # the swap is not registered yet: the unload cancels nothing
        env.runner.stop()

    env.switcher.on_load = stop_too_early
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}], "putBack": False})
    assert job["status"] == "stopped" and env.switcher.states == {}
    assert env.switcher.calls == [("load", "fable-27b"), ("unload", "fable-27b"), ("unload", "fable-27b")]


def test_stop_during_the_hash_wait_is_stopped_not_failed(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.switcher.stale = True
    real = env.switcher.config_hash

    def hash_and_stop():
        if env.runner.job["status"] == "running":
            env.runner.stop()
        return real()

    env.switcher.config_hash = hash_and_stop
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b", "preset": "Three"}]})
    assert job["status"] == "stopped" and job["message"] == "Stopped."
    settings = next(step for step in job["steps"] if step["kind"] == "settings")
    assert settings["ms"] < 30_000  # did not sit out the whole 30 s wait


def test_put_back_that_cannot_tell_notes_the_leftover_and_waits(tmp_path, monkeypatch):
    """Item 6."""
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    waits = []
    real_wait = env.service.wait_for_switcher
    env.service.wait_for_switcher = lambda text, stop=None: waits.append(text) or real_wait(text, stop=stop)

    def blind(body):
        if body["messages"] != WARMUP_MESSAGES:
            env.switcher.up = False  # /running times out from here on

    env.chat.on_stream = blind
    job = start(env, {"prompt": "Hi", "sides": [{"model": "quasar-27b", "preset": "Fast"}]})
    assert job["steps"][-1]["state"] == "failed"
    assert job["restore"].startswith("Can't tell what's loaded right now")
    assert env.service.test_leftover == {"model": "quasar-27b", "preset": "Fast"}
    assert len(waits) == 2  # the test settings, then the saved file at put-back
    assert env.service.test_settings is None


def test_put_back_exception_clears_the_overlay(tmp_path, monkeypatch):
    """Item 13: an error inside put-back still clears the test settings."""
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.switcher.after_load = lambda m: setattr(env.switcher, "load_error", RuntimeError("socket closed"))
    job = start(env, {"prompt": "Hi", "sides": [{"model": "quasar-27b", "preset": "Fast"}]})
    assert job["steps"][-1]["state"] == "failed"
    assert job["restore"] == "Putting things back didn't finish (socket closed). Check the Models tab."
    assert env.service.test_settings is None and env.service.read_test_marker() is None


def test_put_back_exception_with_a_failed_clear_notes_the_leftover(tmp_path, monkeypatch):
    """Item 7."""
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    real = env.service.set_test_settings

    def set_or_fail(model_id, preset):
        if model_id is None:
            raise OSError("disk full")
        return real(model_id, preset)

    env.service.set_test_settings = set_or_fail
    env.probe.busy = set()
    env.chat.on_stream = lambda body: env.probe.busy.add("quasar-27b") if body["messages"] != WARMUP_MESSAGES else None
    job = start(env, {"prompt": "Hi", "sides": [{"model": "quasar-27b", "preset": "Fast"}]})
    assert job["steps"][-1]["state"] == "failed"
    assert "The test settings could not be taken out of the switcher file." in job["restore"]
    assert env.service.test_leftover == {"model": "quasar-27b", "preset": "Fast"}
    assert env.service.read_test_marker()["model"] == "quasar-27b"  # recover() finds it later
    assert env.service.test_running is False


def test_an_answer_without_a_finish_signal_fails_the_test(tmp_path, monkeypatch):
    """Item 8."""
    from tests.settings.playground_fakes import sse
    env = make(tmp_path, monkeypatch)
    env.chat.scripts["fable-27b"] = [(0.2, sse({"choices": [{"index": 0, "delta": {"content": "Hel"}}]}))]
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "failed"
    assert job["message"] == f"{FABLE} could not answer: the answer was cut off before it finished."
    assert job["sides"][0]["answer"] == "Hel" and job["sides"][0]["error"] == "The answer ended without a finish signal."
    assert job["steps"][-1]["state"] == "done"


def test_start_plans_outside_the_lock_and_checks_again(tmp_path, monkeypatch):
    """Item 9."""
    import threading
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    free = []

    def read_from_another_thread():
        reader = threading.Thread(target=env.runner.snapshot)
        reader.start()
        reader.join(timeout=2.0)
        free.append(not reader.is_alive())

    env.probe.on_inflight = read_from_another_thread
    start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert free and all(free)

    env.probe.on_inflight = lambda: setattr(env.runner, "job", {"status": "running", "steps": []})
    with pytest.raises(PlaygroundError) as refused:
        env.runner.start({"prompt": "Hi", "sides": [{"model": "fable-27b"}], "confirm": True})
    assert refused.value.payload["code"] == "test_running" and env.service.test_running is False


def test_put_back_load_superseded_says_another_app_took_the_card(tmp_path, monkeypatch):
    """Item 11."""
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})

    def superseded(model_id):
        if model_id == "quasar-27b":
            env.switcher.states = {"twin-27b": "starting"}
            raise SwitcherError(409, "model_superseded", "superseded by twin-27b")

    env.switcher.on_load = superseded
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "done" and job["steps"][-1]["state"] == "done"
    assert job["restore"] == f"{QUASAR} was not loaded again, because another app took the graphics card."


def test_start_refused_while_a_panel_restart_is_pending(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.service._restarts_pending = 1
    with pytest.raises(PlaygroundError) as refused:
        env.runner.start({"prompt": "Hi", "sides": [{"model": "fable-27b"}], "confirm": True})
    assert (refused.value.status, refused.value.payload["code"]) == (409, "busy")
    assert env.runner.snapshot() == {"status": "idle"}


def test_helper_restart_leaves_a_starting_test_model_llama_swap_holds(tmp_path, monkeypatch):
    """Item 13: recover() with the model starting (a request waits on it)."""
    env = make(tmp_path, monkeypatch)
    env.service.set_test_settings("quasar-27b", "Fast")
    env.switcher.states = {"quasar-27b": "starting"}
    env.probe.busy = {"quasar-27b"}
    service, runner = env.build()
    assert runner.recover() == "left"
    assert env.switcher.states == {"quasar-27b": "starting"} and service.read_test_marker() is None
    assert service.test_leftover == {"model": "quasar-27b", "preset": "Fast"}


def test_helper_restart_with_the_state_unknown_notes_the_leftover(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.service.set_test_settings("quasar-27b", "Fast")
    env.switcher.up = False
    service, runner = env.build()
    assert runner.recover() == "unknown" and env.switcher.calls == []
    assert service.test_leftover == {"model": "quasar-27b", "preset": "Fast"} and service.read_test_marker() is None


# ---- live bug 2026-09-25: put-back after Stop read the test's own cancelled load as another app's ----

def _stop_while_loading(env, model, sticky_reads):
    """Stop lands during the load of model; llama-swap keeps listing that cancelled load as
    "starting" for the next sticky_reads reads of /running (None: for good)."""
    real = env.switcher.running
    left = {"n": 0, "on": False}

    def running():
        states = real()
        if left["on"] and (sticky_reads is None or left["n"] < sticky_reads):
            left["n"] += 1
            states = {**states, model: "starting"}
        return states

    def on_load(model_id):
        if model_id == model:
            env.switcher.states = {model: "starting"}
            env.runner.stop()
            left["on"] = True

    env.switcher.running = running
    env.switcher.on_load = on_load


@pytest.mark.parametrize("sticky", [3, None])
def test_put_back_after_stop_ignores_the_tests_own_cancelled_load(tmp_path, monkeypatch, sticky):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    _stop_while_loading(env, "twin-27b", sticky)
    job = start(env, {"prompt": "Hi", "sides": [{"model": "twin-27b"}]})
    assert job["status"] == "stopped" and job["message"] == "Stopped."
    assert env.switcher.calls[-1] == ("load", "quasar-27b")
    assert job["restore"].startswith(f"{QUASAR} is loaded again on its saved settings")
    assert env.switcher.states == {"quasar-27b": "ready"}


def test_put_back_after_stop_still_yields_to_another_apps_model(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    def on_load(model_id):
        if model_id == "twin-27b":
            env.runner.stop()
            env.switcher.states = {"fable-27b": "starting"}  # another app asked for Fable

    env.switcher.on_load = on_load
    job = start(env, {"prompt": "Hi", "sides": [{"model": "twin-27b"}]})
    assert job["status"] == "stopped"
    assert job["restore"] == f"{QUASAR} was not loaded again, because an app is using {FABLE}."
    assert ("load", "quasar-27b") not in env.switcher.calls and env.switcher.states == {"fable-27b": "starting"}


def test_put_back_after_stop_yields_when_another_app_waits_on_the_same_model(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    _stop_while_loading(env, "twin-27b", None)
    real = env.probe.inflight

    def inflight():
        if env.runner.job and env.runner.job["status"] in ("stopping", "restoring"):
            env.probe.rows = [{"model": "twin-27b", "req_headers": {"X-Session-ID": "claude-code"}}]
        return real()

    env.probe.inflight = inflight
    job = start(env, {"prompt": "Hi", "sides": [{"model": "twin-27b"}]})
    assert job["status"] == "stopped"
    assert job["restore"] == f"{QUASAR} was not loaded again, because an app is using {TWIN}."
    assert ("load", "quasar-27b") not in env.switcher.calls


def test_b_loads_although_as_put_away_model_is_still_listed(tmp_path, monkeypatch):
    """Same confusion between setups: A's model, put away by B's first step, still shows on
    /running; B's re-check before its load must not read it as another app's."""
    env = make(tmp_path, monkeypatch)
    real = env.switcher.running

    def running():
        states = real()
        if ("unload", "fable-27b") in env.switcher.calls and ("load", "twin-27b") not in env.switcher.calls:
            states = {**states, "fable-27b": "ready"}
        return states

    env.switcher.running = running
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}, {"model": "twin-27b"}]})
    assert job["status"] == "done" and ("load", "twin-27b") in env.switcher.calls


def test_b_still_yields_when_another_app_asks_for_as_model(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    real = env.switcher.running

    def running():
        states = real()
        if ("unload", "fable-27b") in env.switcher.calls and ("load", "twin-27b") not in env.switcher.calls:
            env.probe.rows = [{"model": "fable-27b", "req_headers": {"X-Session-ID": "claude-code"}}]
            states = {**states, "fable-27b": "starting"}
        return states

    env.switcher.running = running
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}, {"model": "twin-27b"}]})
    assert job["status"] == "yielded" and ("load", "twin-27b") not in env.switcher.calls
    assert job["message"] == f"Another app started using {FABLE}, so the test stopped to let it through."


def test_a_refused_load_is_marked_as_not_loaded(tmp_path, monkeypatch):
    """Live 2026-09-25: the memory gate refused FreeToken and the row read "already loaded"."""
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})

    def gate(model_id):
        if model_id == "fable-27b":
            raise SwitcherError(503, "not_enough_memory", "not enough free memory to load fable-27b")

    env.switcher.on_load = gate
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "failed" and job["sides"][0]["loadFailed"] is True

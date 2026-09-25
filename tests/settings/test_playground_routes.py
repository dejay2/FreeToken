"""The Test tab's routes and the helper's start-up order."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from freetoken.daemon.settings.server import start_panel
from tests.settings.playground_fakes import make

STATIC = Path(__file__).resolve().parents[2] / "python" / "freetoken" / "daemon" / "settings" / "static"


def routes(tmp_path, monkeypatch, **kwargs):
    return make(tmp_path, monkeypatch, with_routes=True, static_path=STATIC / "index.html", **kwargs)


def test_options_list_models_presets_and_guesses(tmp_path, monkeypatch):
    env = routes(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    body = env.client.get("/api/playground/options").json()
    quasar = next(m for m in body["models"] if m["id"] == "quasar-27b")
    assert quasar["presets"] == ["Fast", "Same"] and quasar["state"] == "ready" and quasar["engineLabel"] == "NInfer"
    assert set(quasar) == {"id", "name", "engine", "engineLabel", "presets", "activePreset", "state"}
    assert body["loadGuessS"] == {"freetoken": 150, "ninfer": 20}
    assert body["defaults"] == {"maxTokens": 512, "maxAnswerTokens": 4096} and body["switcherUp"] is True


def test_options_without_a_registry_is_a_plain_refusal(tmp_path, monkeypatch):
    env = routes(tmp_path, monkeypatch)
    env.store.path.unlink()
    answer = env.client.get("/api/playground/options")
    assert answer.status_code == 409 and answer.json()["code"] == "registry_missing"


def test_plan_and_refusals_are_plain_json(tmp_path, monkeypatch):
    env = routes(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    ok = env.client.post("/api/playground/plan", json={"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert ok.status_code == 200 and ok.json()["before"] == "quasar-27b"
    bad = env.client.post("/api/playground/plan", json={"prompt": "", "sides": [{"model": "fable-27b"}]})
    assert bad.status_code == 422 and bad.json() == {"code": "prompt", "message": "Type a prompt first."}
    empty = env.client.post("/api/playground/plan")
    assert empty.status_code == 422 and empty.json()["code"] == "prompt"


def test_start_runs_and_current_shows_the_finished_job(tmp_path, monkeypatch):
    env = routes(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    assert env.client.get("/api/playground/runs/current").json() == {"status": "idle"}
    started = env.client.post("/api/playground/runs", json={
        "prompt": "Hi", "sides": [{"model": "fable-27b"}], "expectBefore": "quasar-27b", "confirm": True})
    assert started.status_code == 200, started.text
    current = env.client.get("/api/playground/runs/current").json()
    assert current["status"] == "done" and current["sides"][0]["answer"] == "Hello there friend"
    assert all("_t0" not in step for step in current["steps"])
    assert env.client.post("/api/playground/runs/current/stop").json()["status"] == "done"


def test_start_needs_confirm_and_an_unchanged_card(tmp_path, monkeypatch):
    env = routes(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.probe.used = {"quasar-27b": env.clock.wall() - 40}
    body = {"prompt": "Hi", "sides": [{"model": "fable-27b"}], "expectBefore": "quasar-27b"}
    first = env.client.post("/api/playground/runs", json=body)
    assert first.status_code == 409 and first.json()["code"] == "confirm" and first.json()["plan"]["warnings"]
    moved = env.client.post("/api/playground/runs", json={**body, "expectBefore": None, "confirm": True})
    assert moved.status_code == 409 and moved.json()["code"] == "changed"
    assert moved.json()["plan"]["before"] == "quasar-27b"
    assert env.service.test_running is False and env.switcher.calls == []


def test_start_refused_while_the_panel_guard_is_held(tmp_path, monkeypatch):
    env = routes(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.service.begin_test()
    answer = env.client.post("/api/playground/runs", json={
        "prompt": "Hi", "sides": [{"model": "fable-27b"}], "expectBefore": "quasar-27b", "confirm": True})
    assert answer.status_code == 409 and answer.json()["code"] == "test_running"
    env.service.end_test()


def test_playground_js_is_served(tmp_path, monkeypatch):
    env = routes(tmp_path, monkeypatch)
    answer = env.client.get("/playground.js")
    assert answer.status_code == 200 and "javascript" in answer.headers["content-type"]


def test_helper_start_recovers_before_it_syncs():
    calls = []
    app = SimpleNamespace(state=SimpleNamespace(
        playground=SimpleNamespace(recover=lambda: calls.append("recover")),
        panel=SimpleNamespace(sync_config=lambda: calls.append("sync"),
                              start_hold_watcher=lambda: calls.append("watch"))))
    start_panel(app)
    assert calls == ["recover", "sync", "watch"]


def test_a_failing_recover_never_stops_the_helper(caplog):
    """Review item 4: a recover() that raises is logged, and the marker's model is noted as a
    leftover so the Right-now strip still says it sits on test settings."""
    calls = []

    def boom():
        raise OSError("switcher down")

    app = SimpleNamespace(state=SimpleNamespace(
        playground=SimpleNamespace(recover=boom),
        panel=SimpleNamespace(sync_config=lambda: calls.append("sync"),
                              start_hold_watcher=lambda: calls.append("watch"),
                              read_test_marker=lambda: {"model": "quasar-27b", "preset": "Fast"},
                              note_test_leftover=lambda model, preset: calls.append(("leftover", model, preset)))))
    with caplog.at_level("ERROR", logger="freetoken.daemon.settings.server"):
        start_panel(app)
    assert calls == [("leftover", "quasar-27b", "Fast"), "sync", "watch"]
    assert any("switcher down" in record.exc_text for record in caplog.records if record.exc_text)


def test_a_failing_recover_without_a_marker_notes_nothing():
    calls = []

    def boom():
        raise OSError("switcher down")

    app = SimpleNamespace(state=SimpleNamespace(
        playground=SimpleNamespace(recover=boom),
        panel=SimpleNamespace(sync_config=lambda: calls.append("sync"),
                              start_hold_watcher=lambda: calls.append("watch"),
                              read_test_marker=lambda: None,
                              note_test_leftover=lambda model, preset: calls.append(("leftover", model, preset)))))
    start_panel(app)
    assert calls == ["sync", "watch"]

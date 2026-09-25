"""PR #18 review (Codex) fixes for the Test tab: each test fails without its fix."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from freetoken.daemon.settings import server as helper_server
from freetoken.daemon.settings.playground import NOT_FREE, SwitcherProbe
from freetoken.daemon.settings.playground_speed import AnswerTracker
from freetoken.daemon.settings.switcher import SwitcherError
from tests.settings.playground_fakes import make, sse

QUASAR = "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)"
FABLE = "Fable 27B NVFP4 (NInfer)"
PG_JS = Path(__file__).resolve().parents[2] / "python" / "freetoken" / "daemon" / "settings" / "static" / "playground.js"


def start(env, body):
    env.runner.start({"confirm": True, **body})
    return env.runner.snapshot()


# ---- 1: Stop and the after-readiness fallback never cancel another app that joined the load ----

def test_stop_during_a_load_another_app_joined_leaves_it_loading(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})

    def joined(model_id):
        if model_id == "fable-27b":
            env.switcher.states = {"fable-27b": "starting"}
            env.probe.busy.add("fable-27b")  # another app's request now waits on the same load
            env.runner.stop()

    env.switcher.on_load = joined
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "stopped"
    assert ("unload", "fable-27b") not in env.switcher.calls and ("cancel", "fable-27b") in env.switcher.calls
    assert env.switcher.states == {"fable-27b": "starting"} and ("load", "quasar-27b") not in env.switcher.calls
    assert job["restore"] == f"{QUASAR} was not loaded again, because an app is using {FABLE}."


def test_stop_after_the_load_finished_puts_away_only_if_idle(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)

    def stop_too_early(model_id):
        env.switcher.loading = None  # the cancel reaches llama-swap after the load finished
        env.probe.busy.add("fable-27b")  # and another app is using the model by then
        env.runner.stop()

    env.switcher.on_load = stop_too_early
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "stopped" and env.switcher.states == {"fable-27b": "ready"}
    assert ("unload", "fable-27b") not in env.switcher.calls and ("busy", "fable-27b") in env.switcher.calls


def test_put_back_retries_a_cut_short_load_until_its_cancel_lands(tmp_path, monkeypatch):
    """llama-swap drops the cancelled request's claim a moment after the socket closes: the
    first P7 answer may still be busy."""
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    answers = iter(["busy", "busy"])
    real = env.probe.unload_if_idle
    env.probe.unload_if_idle = lambda m: next(answers, None) or real(m) if m == "twin-27b" else real(m)

    def on_load(model_id):
        if model_id == "twin-27b":
            env.switcher.states = {"twin-27b": "starting"}
            env.runner.stop()

    env.switcher.on_load = on_load
    job = start(env, {"prompt": "Hi", "sides": [{"model": "twin-27b"}]})
    assert job["status"] == "stopped" and ("unload", "twin-27b") in env.switcher.calls
    assert env.switcher.states == {"quasar-27b": "ready"}


# ---- 2: loads use P8 if-free and yield when the card is busy ----

def test_a_test_load_on_a_busy_card_yields_without_preempting(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.probe.not_free.add("fable-27b")  # e.g. another app's load waits in the memory gate
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "yielded" and job["message"] == NOT_FREE
    assert ("load", "fable-27b") not in env.switcher.calls and job["sides"][0]["loadFailed"] is True
    assert all(session.startswith("ft-test-") for session in env.probe.load_sessions)


def test_a_put_back_load_on_a_busy_card_says_so(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.probe.not_free.add("quasar-27b")
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "done" and ("load", "quasar-27b") not in env.switcher.calls
    assert job["restore"] == f"{QUASAR} was not loaded again, because another app is loading or using a model."


@pytest.fixture
def server():
    routes, seen = {}, []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen.append({"path": self.path, "headers": dict(self.headers)})
            try:
                routes[self.path.split("?")[0]](self)
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield routes, seen, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _reply(handler, status, body: bytes):
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


@pytest.mark.parametrize("status, body, code", [
    (200, b'{"model": "m/1", "state": "ready"}', None),
    (409, json.dumps({"src": "llama-swap", "error": {"message": "m/1 was not loaded because x is loading or in use",
                                                     "type": "invalid_request_error", "code": "busy"}}).encode(), "busy"),
    (503, b'{"error": {"code": "not_enough_memory", "message": "no room"}}', "not_enough_memory"),
    (501, b"this router cannot load only if free", "load_failed"),
])
def test_load_if_free_uses_p8(server, status, body, code):
    routes, seen, url = server
    routes["/api/models/load/m%2F1"] = lambda h: _reply(h, status, body)
    probe = SwitcherProbe(url)
    if code is None:
        probe.load_if_free("m/1", "ft-test-1")
    else:
        with pytest.raises(SwitcherError) as caught:
            probe.load_if_free("m/1", "ft-test-1")
        assert caught.value.code == code and caught.value.status == status
    assert seen[0]["path"] == "/api/models/load/m%2F1?ifFree=1" and seen[0]["headers"]["X-Session-ID"] == "ft-test-1"


def test_cancel_load_ends_a_waiting_load_quickly(server):
    routes, _, url = server
    release = threading.Event()
    routes["/api/models/load/m"] = lambda h: (release.wait(5), _reply(h, 200, b"{}"))
    probe = SwitcherProbe(url)
    threading.Timer(0.2, probe.cancel_load).start()
    began = time.monotonic()
    with pytest.raises(SwitcherError) as caught:
        probe.load_if_free("m")
    release.set()
    assert caught.value.code == "cancelled" and time.monotonic() - began < 2
    with pytest.raises(SwitcherError):  # sticky until reset
        probe.load_if_free("m")
    probe.reset_load()
    probe.load_if_free("m")


# ---- 3: recovery runs only once the helper answers HTTP ----

def test_helper_recovers_only_after_it_listens(tmp_path, monkeypatch):
    import uvicorn

    events = []
    recovered = threading.Event()
    panel = SimpleNamespace(sync_config=lambda: events.append("sync"),
                            start_hold_watcher=lambda: (events.append("watch"), recovered.set()),
                            stop_hold_watcher=lambda: None)
    app = SimpleNamespace(state=SimpleNamespace(panel=panel,
                                                playground=SimpleNamespace(recover=lambda: events.append("recover"))))

    class FakeProcessManager:
        def __init__(self, **kwargs):
            pass

        start_governor = start_watchdog = stop_watchdog = stop_governor = close = lambda self: None

    monkeypatch.setattr(helper_server, "ProcessManager", FakeProcessManager)
    monkeypatch.setattr(helper_server, "ProfilesManager", lambda *a, **k: None)
    monkeypatch.setattr(helper_server, "create_app", lambda **kwargs: app)
    with socket.socket() as spare:
        spare.bind(("127.0.0.1", 0))
        port = spare.getsockname()[1]

    def fake_run(app_, host, port, log_level):
        time.sleep(0.3)  # recovery must wait for this
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen()
            events.append("listening")
            recovered.wait(5)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    helper_server.main(["--port", str(port), "--boot-file", str(tmp_path / "boot.ps1")])
    assert events == ["listening", "recover", "sync", "watch"]


def test_start_panel_when_listening_gives_up_when_the_helper_exits():
    calls, stop = [], threading.Event()
    stop.set()
    app = SimpleNamespace(state=SimpleNamespace(playground=SimpleNamespace(recover=lambda: calls.append("recover")),
                                                panel=SimpleNamespace(sync_config=lambda: None,
                                                                      start_hold_watcher=lambda: None)))
    thread = helper_server.start_panel_when_listening(app, 1, accepts=lambda port: False, sleep=lambda s: None,
                                                      stop=stop)
    thread.join(2)
    assert not thread.is_alive() and calls == []


# ---- 4: the marker goes only after the saved file is written ----

def test_recover_keeps_the_marker_when_the_saved_file_cannot_be_written(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.service.set_test_settings("quasar-27b", "Fast")  # the helper stopped mid-test
    env.switcher.states = {"quasar-27b": "ready"}
    service, runner = env.build()

    def broken(text):
        raise OSError("disk full")

    monkeypatch.setattr(service.writer, "write", broken)
    with pytest.raises(OSError):
        runner.recover()
    assert service.read_test_marker()["model"] == "quasar-27b"
    assert service.test_leftover == {"model": "quasar-27b", "preset": "Fast"}


# ---- 5: a model left on test settings is reloaded for saved settings ----

def test_a_leftover_test_model_is_reloaded_for_saved_settings(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.service.note_test_leftover("quasar-27b", "Fast")  # an app blocked the last put-away
    plan = env.runner.plan({"prompt": "Hi", "sides": [{"model": "quasar-27b"}], "warmup": False})
    assert plan["held"] is True and [s["kind"] for s in plan["steps"]] == ["unload", "settings", "load", "answer", "restore"]
    job = start(env, {"prompt": "Hi", "sides": [{"model": "quasar-27b"}], "warmup": False})
    assert job["status"] == "done" and ("unload", "quasar-27b") in env.switcher.calls
    assert env.service.test_leftover is None  # put away, then loaded on saved settings


def test_the_leftover_note_stays_until_the_model_was_put_away(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.service.note_test_leftover("quasar-27b", "Fast")
    env.probe.busy.add("quasar-27b")  # the put-away is refused again
    job = start(env, {"prompt": "Hi", "sides": [{"model": "quasar-27b"}]})
    assert job["status"] == "yielded"
    assert env.service.test_leftover == {"model": "quasar-27b", "preset": "Fast"}


# ---- 6: an error envelope in the stream fails the answer ----

def test_tracker_reads_an_error_envelope():
    tracker = AnswerTracker(started=0.0)
    tracker.feed_line(sse({"error": {"message": "KV cache is full", "type": "server_error", "code": "kv_full"}}), 0.1)
    tracker.feed_line("data: [DONE]\n", 0.1)
    assert tracker.error == {"message": "KV cache is full", "code": "kv_full"} and tracker.done
    plain = AnswerTracker(started=0.0)
    plain.feed_line(sse({"error": "engine stopped"}), 0.1)
    assert plain.error == {"message": "engine stopped", "code": None}


def test_an_error_then_done_fails_the_step(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    env.chat.scripts["fable-27b"] = [(0.1, sse({"error": {"message": "KV cache is full", "code": "kv_full"}})),
                                     (0.0, "data: [DONE]\n")]
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "failed" and job["message"] == f"{FABLE} could not answer: KV cache is full (kv_full)"
    side = job["sides"][0]
    assert side["error"] == "The engine reported an error: KV cache is full" and side["completed"] is False
    assert [s["state"] for s in job["steps"] if s["kind"] == "answer"] == ["failed"]


def test_a_clean_answer_is_marked_completed(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "done" and job["sides"][0]["completed"] is True


# ---- 7 and 8: the page ----

def _node(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to run the page's JavaScript")
    prelude = f"const p = require({json.dumps(str(PG_JS))}); const assert = require('node:assert/strict');\n"
    run = subprocess.run([node, "-e", prelude + script], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr


def test_better_tag_only_between_two_completed_answers():
    _node(r"""
const ok = (key, totalMs) => ({key, completed: true, error: null, stats: {totalMs, finishReason: 'stop'}});
assert.equal(p.betterSide(ok('A', 1000), ok('B', 2000), 'totalMs'), 'A');
assert.equal(p.betterSide(ok('A', 1000), {...ok('B', 2000), error: 'The answer ended without a finish signal.'}, 'totalMs'), null);
assert.equal(p.betterSide({...ok('A', 1000), error: 'The engine reported an error: x'}, ok('B', 2000), 'totalMs'), null);
assert.equal(p.betterSide(ok('A', 1000), {...ok('B', 2000), completed: false}, 'totalMs'), null);
const noFinish = {key: 'B', stats: {totalMs: 2000, finishReason: null}};
assert.equal(p.betterSide(ok('A', 1000), noFinish, 'totalMs'), null);
assert.equal(p.speedRows(ok('A', 1000), {...ok('B', 2000), error: 'x'}).find((r) => r.key === 'totalMs').better, false);
""")


def test_history_keeps_load_failed_and_completion():
    _node(r"""
const job = {id: '1', startedAt: '2026-09-25T09:00:00Z', prompt: 'Hi', status: 'failed',
  sides: [{key: 'A', model: 'm', name: 'M', loadMs: 300000, loadFailed: true, completed: false, stats: null}]};
const record = p.historyRecord(job);
assert.equal(record.sides[0].loadFailed, true);
assert.equal(record.sides[0].completed, false);
assert.ok(p.historyMarkdown(record).includes('did not load'));
""")

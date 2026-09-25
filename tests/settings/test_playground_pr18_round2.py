"""PR #18 review (Codex) round 2 fixes for the Test tab: each test fails without its fix."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from freetoken.daemon.settings import server as helper_server
from freetoken.daemon.settings.panel import RECOVERING_MESSAGE, PanelError
from freetoken.daemon.settings.playground import IF_FREE_HEADER, NOT_FREE, ChatFailed, PlaygroundError, SwitcherChat
from freetoken.daemon.settings.process_manager import ProcessManager
from tests.settings.playground_fakes import make

QUASAR = "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)"


def start(env, body):
    env.runner.start({"confirm": True, **body})
    return env.runner.snapshot()


# ---- 1: put-back cancels a stopped load that has no process (it waited in the memory gate) ----

def test_put_back_cancels_a_stopped_load_that_waited_in_the_memory_gate(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})

    def gated_then_stopped(model_id):
        if model_id == "fable-27b":
            # fable's load waits in the memory gate: no process, so not in /running. Until it is
            # cancelled, llama-swap's P8 refuses the put-back load of QUASAR as busy.
            env.probe.not_free.add("quasar-27b")
            env.runner.stop()

    real = env.probe.unload_if_idle

    def unload_if_idle(model_id):
        if model_id == "fable-27b":
            env.probe.not_free.discard("quasar-27b")  # the orphaned swap is cancelled
        return real(model_id)

    env.switcher.on_load = gated_then_stopped
    env.probe.unload_if_idle = unload_if_idle
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "stopped" and ("unload", "fable-27b") in env.switcher.calls
    assert ("load", "quasar-27b") in env.switcher.calls and env.switcher.states == {"quasar-27b": "ready"}
    assert job["restore"].startswith(f"{QUASAR} is loaded again")


def test_put_back_leaves_a_stopped_load_another_app_joined(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})

    def joined(model_id):
        if model_id == "fable-27b":
            env.probe.busy.add("fable-27b")  # another app's request waits on the same gated load
            env.probe.not_free.add("quasar-27b")
            env.runner.stop()

    env.switcher.on_load = joined
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "stopped" and ("unload", "fable-27b") not in env.switcher.calls
    assert ("busy", "fable-27b") in env.switcher.calls and ("load", "quasar-27b") not in env.switcher.calls


# ---- 2: test chats ask for if-free admission and yield on busy ----

def _busy(body):
    raise ChatFailed(409, "busy", "model fable-27b was not loaded because qwen3.8-flash is loading or in use")


@pytest.mark.parametrize("warmup", [True, False])
def test_a_busy_chat_yields_in_plain_words(tmp_path, monkeypatch, warmup):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    if warmup:
        env.chat.on_stream = lambda body: _busy(body) if body["model"] == "fable-27b" else None
    else:
        env.chat.errors["fable-27b"] = ChatFailed(409, "busy", "model fable-27b was not loaded")
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}], "warmup": warmup})
    assert job["status"] == "yielded" and job["message"] == NOT_FREE


@pytest.fixture
def chat_server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen.append(dict(self.headers))
            body = b'{"src": "llama-swap", "error": {"message": "busy", "code": "busy"}}'
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield seen, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_chat_sends_the_if_free_header_and_reads_busy(chat_server):
    seen, url = chat_server
    with pytest.raises(ChatFailed) as caught:
        list(SwitcherChat(url).stream({"model": "m", "messages": []}, "ft-test-1"))
    assert caught.value.code == "busy" and caught.value.status == 409
    assert seen[0][IF_FREE_HEADER] == "1" and seen[0]["X-Session-ID"] == "ft-test-1"


# ---- 3: a leftover survives a helper restart until the model is seen unloaded ----

def test_the_leftover_note_survives_a_helper_restart(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.service.note_test_leftover("quasar-27b", "Fast")  # an app blocked the put-away
    env.probe.rows = [{"model": "quasar-27b", "req_headers": {}}]
    service, runner = env.build()  # the helper restarts; no test marker is left
    assert service.test_leftover == {"model": "quasar-27b", "preset": "Fast"}
    assert runner.recover() == "left" and ("unload", "quasar-27b") not in env.switcher.calls
    env.probe.rows = []
    service, runner = env.build()
    assert runner.recover() == "unloaded" and ("unload", "quasar-27b") in env.switcher.calls
    assert service.test_leftover is None and not service.test_leftover_path.exists()
    assert env.build()[0].test_leftover is None


def test_a_leftover_seen_unloaded_is_forgotten_on_disk(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    env.service.note_test_leftover("quasar-27b", "Fast")
    env.switcher.states = {}
    assert env.service.now()["testLeftover"] is None
    assert env.build()[0].test_leftover is None


# ---- 4: start-up recovery holds the test guard ----

def test_recovery_refuses_a_test_and_panel_actions_with_busy(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch, with_routes=True)
    env.service.begin_recovery()
    with pytest.raises(PlaygroundError) as refused:
        env.runner.start({"prompt": "Hi", "sides": [{"model": "fable-27b"}], "confirm": True})
    assert refused.value.status == 409 and refused.value.payload == {"code": "busy", "message": RECOVERING_MESSAGE}
    with pytest.raises(PanelError):
        env.service.load("fable-27b")
    # The adapter's lifecycle routes never take the guard (freetoken.sh's unload calls stop).
    stops = []
    monkeypatch.setattr(ProcessManager, "start", lambda self, action, **kw: stops.append(action) or "job-1")
    assert env.client.post("/api/server/stop").status_code == 202 and stops == ["stop"]
    env.service.end_recovery()
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert job["status"] == "done"


def test_the_guard_is_up_before_the_helper_listens_and_lifted_after_recovery(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    listening, seen = threading.Event(), []

    def recover():
        seen.append(env.service.recovering)
        with pytest.raises(PanelError) as refused:
            env.service.begin_test()
        seen.append(refused.value.payload["code"])

    app = SimpleNamespace(state=SimpleNamespace(panel=env.service, playground=SimpleNamespace(recover=recover)))
    monkeypatch.setattr(env.service, "start_hold_watcher", lambda: None)
    thread = helper_server.start_panel_when_listening(app, 1, accepts=lambda port: listening.is_set(),
                                                      sleep=lambda s: listening.wait(0.05))
    with pytest.raises(PanelError) as early:  # not listening yet, and already guarded
        env.service.begin_test()
    assert early.value.payload["code"] == "busy"
    listening.set()
    thread.join(5)
    assert not thread.is_alive() and seen == [True, "busy"]
    env.service.begin_test()  # lifted
    env.service.end_test()


def test_the_guard_is_lifted_when_the_helper_exits_without_listening(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    stop = threading.Event()
    stop.set()
    app = SimpleNamespace(state=SimpleNamespace(panel=env.service, playground=SimpleNamespace(recover=lambda: None)))
    thread = helper_server.start_panel_when_listening(app, 1, accepts=lambda port: False, sleep=lambda s: None,
                                                      stop=stop)
    thread.join(2)
    assert env.service.recovering is False


# ---- 5: preflight exits mark the load as failed ----

def test_card_taken_during_the_settings_step_marks_load_failed(tmp_path, monkeypatch):
    env = make(tmp_path, monkeypatch)
    real = env.service.wait_for_switcher

    def taken(text, stop=None):
        env.switcher.states = {"twin-27b": "ready"}  # another app loaded Twin meanwhile
        return real(text, stop=stop)

    monkeypatch.setattr(env.service, "wait_for_switcher", taken)
    job = start(env, {"prompt": "Hi", "sides": [{"model": "fable-27b", "preset": "Three"}]})
    assert job["status"] == "yielded" and ("load", "fable-27b") not in env.switcher.calls
    assert job["sides"][0]["loadFailed"] is True

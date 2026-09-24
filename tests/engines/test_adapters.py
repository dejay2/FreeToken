"""Adapter contract tests with a fake FreeToken helper/server and fake engines.

No GPU, no real engines: the helper is a tiny HTTP server in a thread whose state the
tests script, and the "engine" is a shell script that records its argv and sleeps.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ADAPTERS = REPO / "engines" / "adapters"


class FakeHelper:
    """Scriptable stand-in for the settings helper (:2031) and FreeToken server (:2020)."""

    def __init__(self):
        self.state = "unreachable"   # server.state in /api/status
        self.job = None              # currentJob id or None
        self.armed = False
        self.model_path = "/m/A"
        self.stop_result = "stopped"  # stage the stop job ends in
        self.start_result = "serving"
        self.calls: list[str] = []
        self.jobs: dict[str, str] = {}
        helper = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                # separators=(",", ":") matches Starlette's JSONResponse.render (the real
                # settings helper's serializer): no space after ":" or ",". freetoken.sh's
                # `case` match for the PUT /api/settings ack looks for the literal substring
                # '"status":"saved"' with no space, so the fake must emit the same compact
                # form the real helper does, or the adapter's (correct) parsing spuriously
                # "fails" against a fake that formats JSON differently than production.
                data = json.dumps(body, separators=(",", ":")).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                helper.calls.append("GET " + self.path)
                if self.path == "/api/status":
                    return self._send(200, {
                        "server": {"state": helper.state},
                        "currentJob": {"jobId": helper.job} if helper.job else None,
                        "autoRestart": {"armed": helper.armed, "enabled": True, "gave_up": False},
                    })
                if self.path == "/api/settings":
                    return self._send(200, {"settings": {"ModelPath": helper.model_path}})
                if self.path.startswith("/api/server/jobs/"):
                    jid = self.path.rsplit("/", 1)[1]
                    if jid not in helper.jobs:
                        return self._send(404, {"detail": "not found"})
                    return self._send(200, {"jobId": jid, "stage": helper.jobs[jid]})
                if self.path.startswith("/ready"):
                    want = self.path.split("model=", 1)[1] if "model=" in self.path else ""
                    if helper.state == "serving" and os.path.basename(helper.model_path) != want:
                        return self._send(503, {"status": "not_ready", "reason": "another model is loaded"})
                    return self._send(200 if helper.state == "serving" else 503, {})
                return self._send(404, {})

            def do_PUT(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                helper.calls.append("PUT " + self.path)
                helper.model_path = body["settings"]["ModelPath"]
                return self._send(200, {"status": "saved", "settings": body["settings"]})

            def do_POST(self):
                helper.calls.append("POST " + self.path)
                jid = f"job-{len(helper.jobs) + 1}"
                if self.path == "/api/server/stop":
                    helper.jobs[jid] = helper.stop_result
                    if helper.stop_result == "stopped":
                        helper.state, helper.job, helper.armed = "unreachable", None, False
                    return self._send(202, {"jobId": jid})
                if self.path == "/api/server/start":
                    helper.jobs[jid] = helper.start_result
                    if helper.start_result == "serving":
                        helper.state = "serving"
                    return self._send(202, {"jobId": jid})
                return self._send(404, {})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def helper():
    h = FakeHelper()
    yield h
    h.close()


def env_for(helper, tmp_path):
    env = dict(os.environ)
    env.update(FREETOKEN_HELPER=helper.url, FREETOKEN_SERVER=helper.url,
               NINFER_PROCESS_NAME=f"fake-ninfer-{os.getpid()}", NINFER_STOP_POLLS="3")
    return env


def fake_engine(tmp_path) -> Path:
    p = tmp_path / "engine.sh"
    p.write_text('#!/usr/bin/env bash\necho "$@" > "$(dirname "$0")/argv"\nexec sleep 30\n')
    p.chmod(0o755)
    return p


def test_ninfer_execs_the_runtime_when_the_card_is_free(helper, tmp_path):
    eng = fake_engine(tmp_path)
    proc = subprocess.Popen([ADAPTERS / "ninfer.sh", eng, "/a.ninfer", "quasar-27b", "--port", "8090"],
                            env=env_for(helper, tmp_path))
    for _ in range(50):
        if (tmp_path / "argv").exists():
            break
        time.sleep(0.1)
    proc.terminate()
    proc.wait(5)
    assert (tmp_path / "argv").read_text().split() == ["/a.ninfer", "--model-id", "quasar-27b", "--port", "8090"]
    assert "POST /api/server/stop" not in helper.calls


def test_ninfer_stops_freetoken_first(helper, tmp_path):
    helper.state = "serving"
    eng = fake_engine(tmp_path)
    proc = subprocess.Popen([ADAPTERS / "ninfer.sh", eng, "/a.ninfer", "fable-27b"], env=env_for(helper, tmp_path))
    for _ in range(50):
        if (tmp_path / "argv").exists():
            break
        time.sleep(0.1)
    proc.terminate()
    proc.wait(5)
    assert "POST /api/server/stop" in helper.calls
    assert (tmp_path / "argv").exists()


def test_ninfer_refuses_when_freetoken_will_not_stop(helper, tmp_path):
    helper.state, helper.stop_result = "serving", "failed"
    eng = fake_engine(tmp_path)
    r = subprocess.run([ADAPTERS / "ninfer.sh", eng, "/a.ninfer", "fable-27b"], env=env_for(helper, tmp_path),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert not (tmp_path / "argv").exists()
    assert "refusing" in r.stderr


def test_ninfer_stops_freetoken_when_its_watchdog_is_live(helper, tmp_path):
    helper.armed = True  # crashed server about to be restarted
    eng = fake_engine(tmp_path)
    proc = subprocess.Popen([ADAPTERS / "ninfer.sh", eng, "/a.ninfer", "fable-27b"], env=env_for(helper, tmp_path))
    time.sleep(1.5)
    proc.terminate()
    proc.wait(5)
    assert "POST /api/server/stop" in helper.calls


def test_freetoken_boots_then_stops_on_sigterm(helper, tmp_path):
    proc = subprocess.Popen([ADAPTERS / "freetoken.sh", "/m/B"], env=env_for(helper, tmp_path),
                            stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        if helper.state == "serving":
            break
        time.sleep(0.1)
    assert helper.model_path == "/m/B" and "POST /api/server/start" in helper.calls
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(20) == 0
    assert helper.calls.count("POST /api/server/stop") == 1


def test_freetoken_sigterm_exits_nonzero_when_stop_fails(helper, tmp_path):
    proc = subprocess.Popen([ADAPTERS / "freetoken.sh", "/m/B"], env=env_for(helper, tmp_path))
    for _ in range(100):
        if helper.state == "serving":
            break
        time.sleep(0.1)
    helper.stop_result = "failed"
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(20) != 0


def test_freetoken_adopts_the_same_model_without_rebooting(helper, tmp_path):
    helper.state, helper.model_path = "serving", "/m/B"
    proc = subprocess.Popen([ADAPTERS / "freetoken.sh", "/m/B"], env=env_for(helper, tmp_path))
    time.sleep(1.5)
    proc.terminate()
    proc.wait(20)
    assert "POST /api/server/start" not in helper.calls

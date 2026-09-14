"""The settings helper must preserve cache contracts without importing the engine."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.daemon.settings.prompt_cache import create_prompt_cache_router


@pytest.fixture
def bridge():
    calls = []
    reply = {"code": 200, "body": {"status": "ok", "result": {"prefixes": []}}}

    class Upstream(BaseHTTPRequestHandler):
        def handle_request(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append((self.command, self.path, json.loads(raw) if raw else None))
            self.send_response(reply["code"])
            self.send_header("Content-Type", "application/json")
            if "location" in reply:
                self.send_header("Location", reply["location"])
            self.end_headers()
            self.wfile.write(reply.get("raw", json.dumps(reply["body"]).encode()))

        do_GET = do_POST = do_PUT = do_DELETE = handle_request

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    app = FastAPI()
    app.include_router(create_prompt_cache_router(lambda: server.server_port))
    with TestClient(app) as client:
        yield client, calls, reply
    server.shutdown()
    server.server_close()
    thread.join()


def registration():
    return {"name": "agent:one", "format": "openai", "prefix_tokens": 512,
            "ttl_seconds": 300, "request": {"model": "local", "messages": [
                {"role": "system", "content": "<shared instructions>"},
                {"role": "user", "content": "Test"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "chat_template_kwargs": {"enable_thinking": False}}}


def test_registration_preserves_request_template_and_errors(bridge):
    client, calls, reply = bridge
    body = registration()
    reply.update(code=400, body={"status": "invalid", "result": {}, "error": "prefix exceeds pool"})
    response = client.post("/api/prompt-cache/prefixes", json=body)
    assert response.status_code == 400
    assert response.json() == reply["body"]
    assert calls == [("POST", "/v1/cache/prefixes", body)]


@pytest.mark.parametrize("method,path,upstream,body", [
    ("GET", "/prefixes", "/v1/cache/prefixes", None),
    ("GET", "/models", "/v1/models", None),
    ("GET", "/status", "/v1/cache/status", None),
    ("GET", "/recent", "/v1/cache/recent", None),
    ("GET", "/recent/0123456789abcdef0123456789abcdef", "/v1/cache/recent/0123456789abcdef0123456789abcdef", None),
    ("DELETE", "/recent", "/v1/cache/recent", None),
    ("PUT", "/recent/settings", "/v1/cache/recent/settings", {"enabled": False}),
    ("POST", "/prefixes/agent:one/warm", "/v1/cache/prefixes/agent:one/warm", None),
    ("DELETE", "/prefixes/agent:one", "/v1/cache/prefixes/agent:one", None),
    ("PUT", "/settings", "/v1/cache/prefixes/settings", {"max_retained_bytes": 0}),
])
def test_only_fixed_loopback_operations_are_forwarded(bridge, method, path, upstream, body):
    client, calls, reply = bridge
    reply["code"] = 202 if path.endswith("/warm") else 200
    if path.endswith("/warm"):
        reply["body"] = {"status": "warming", "result": {"state": "warming"}, "error": None}
    response = client.request(method, "/api/prompt-cache" + path, json=body)
    assert response.status_code == reply["code"]
    assert response.json() == reply["body"]
    assert calls == [(method, upstream, body)]


@pytest.mark.parametrize("change", [
    {"name": "../admin"}, {"format": "other"}, {"prefix_tokens": -1},
    {"ttl_seconds": 86401}, {"request": []},
])
def test_invalid_registration_never_reaches_server(bridge, change):
    client, calls, _ = bridge
    assert client.post("/api/prompt-cache/prefixes", json=registration() | change).status_code == 422
    assert not calls


@pytest.mark.parametrize("format,path", [("openai", "/v1/chat/completions"), ("anthropic", "/v1/messages")])
def test_generation_is_nonstreaming_and_bounded_without_changing_template(bridge, format, path):
    client, calls, _ = bridge
    body = registration()["request"] | {"stream": True, "max_tokens": 99999, "n": 8}
    original = json.loads(json.dumps(body))
    assert client.post("/api/prompt-cache/test", json={"format": format, "request": body}).status_code == 200
    method, actual_path, sent = calls[0]
    assert (method, actual_path) == ("POST", path)
    assert sent["stream"] is False and sent["max_tokens"] == 64 and sent["n"] == 1
    for key in ("messages", "model", "tools", "chat_template_kwargs"):
        assert sent[key] == original[key]


def test_malformed_or_redirecting_upstream_does_not_look_successful(bridge):
    client, calls, reply = bridge
    reply["raw"] = b"<html>Not the cache API</html>"
    assert client.get("/api/prompt-cache/prefixes").status_code == 502
    reply.update(code=302, location="http://localhost:1/forbidden")
    assert client.get("/api/prompt-cache/prefixes").status_code == 502
    assert len(calls) == 2  # No redirect is followed.


def test_offline_server_has_actionable_error():
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_port
    server.server_close()
    app = FastAPI()
    app.include_router(create_prompt_cache_router(lambda: port))
    with TestClient(app) as client:
        response = client.get("/api/prompt-cache/prefixes")
    assert response.status_code == 503
    assert "server" in response.json()["error"].lower()

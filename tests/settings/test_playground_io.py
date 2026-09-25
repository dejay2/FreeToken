"""SwitcherChat and SwitcherProbe against a real local HTTP server.

The fake server speaks the shapes of the frozen llama-swap (engines/llama-swap):
- /api/events writes ``event:message\\ndata:{"type","data"}`` envelopes and opens with the
  proxy log history, then an ``inflight`` event whose data is
  ``{"operation": "snapshot", "requests": [...]}`` (internal/server/apigroup.go, inflight.go);
- /api/metrics/activity answers ``{"data": [...], "page", "limit", "total", ...}`` newest
  first (internal/store/activity.go, ActivityPage);
- an error is ``{"src": "llama-swap", "error": {"message", "type", "code"}}``
  (internal/swaputil/httperror.go).
"""

from __future__ import annotations

import datetime as dt
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from freetoken.daemon.settings.playground import ChatFailed, SwitcherChat, SwitcherProbe, parse_go_time


@pytest.fixture
def server():
    routes, seen = {}, []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _go(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
            try:
                routes[(self.command, self.path.split("?")[0])](self)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = do_POST = _go

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield routes, seen, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def reply(handler, status, body: bytes, ctype="application/json"):
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.end_headers()
    handler.wfile.write(body)
    handler.wfile.flush()


def test_chat_streams_lines_and_sends_the_session(server):
    routes, seen, url = server
    routes[("POST", "/v1/chat/completions")] = lambda h: reply(
        h, 200, b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n', "text/event-stream")
    lines = list(SwitcherChat(url).stream({"model": "m", "stream": True}, "ft-test-abc"))
    assert [line.strip() for line in lines if line.strip()] == ['data: {"choices":[{"delta":{"content":"Hi"}}]}', "data: [DONE]"]
    assert seen[0]["headers"]["X-Session-ID"] == "ft-test-abc"
    assert json.loads(seen[0]["body"]) == {"model": "m", "stream": True}


def test_error_body_becomes_chat_failed(server):
    routes, _, url = server
    # The real envelope (swaputil.NewErrorEnvelope) also carries "src" and "error.type".
    routes[("POST", "/v1/chat/completions")] = lambda h: reply(
        h, 409, json.dumps({"src": "llama-swap", "error": {"message": "superseded by twin-27b",
                                                            "type": "invalid_request_error",
                                                            "code": "model_superseded"}}).encode())
    with pytest.raises(ChatFailed) as failed:
        list(SwitcherChat(url).stream({}, "s"))
    assert (failed.value.status, failed.value.code, failed.value.message) == (409, "model_superseded", "superseded by twin-27b")
    routes[("POST", "/v1/chat/completions")] = lambda h: reply(h, 500, b"engine crashed", "text/plain")
    with pytest.raises(ChatFailed) as plain:
        list(SwitcherChat(url).stream({}, "s"))
    assert (plain.value.code, plain.value.message) == ("chat_failed", "engine crashed")


def test_abort_ends_the_stream_quickly(server):
    routes, _, url = server

    def slow(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n')
        handler.wfile.flush()
        time.sleep(5)

    routes[("POST", "/v1/chat/completions")] = slow
    chat, got, started = SwitcherChat(url), [], time.monotonic()
    for line in chat.stream({}, "s"):
        got.append(line)
        chat.abort()
    assert got and time.monotonic() - started < 2.0


def test_abort_from_another_thread_ends_the_stream_quickly(server):
    """Stop is pressed from the route thread while the runner thread is blocked in readline."""
    routes, _, url = server

    def slow(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n')
        handler.wfile.flush()
        time.sleep(5)

    routes[("POST", "/v1/chat/completions")] = slow
    chat, got, started = SwitcherChat(url), [], time.monotonic()
    threading.Timer(0.3, chat.abort).start()
    got.extend(chat.stream({}, "s"))
    assert got and time.monotonic() - started < 2.0


def test_refused_connection_is_switcher_down():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(ChatFailed) as failed:
        list(SwitcherChat(f"http://127.0.0.1:{port}").stream({}, "s"))
    assert failed.value.code == "switcher_down"


@pytest.mark.parametrize("framing", ["chunked", "chunked-mid-chunk", "content-length"])
def test_dropped_connection_is_cut_off(server, framing):
    """llama-swap streams chunked (Go's net/http with no Content-Length), and the drop can land
    between chunks or inside one; an engine answering with a Content-Length that it never
    fills is the other way a drop can look. HTTPResponse.readline() hides the chunked cases
    on 3.13, which is why SwitcherChat reads with read1()."""
    routes, _, url = server
    chunk = b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'

    def drop(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        if framing == "chunked":
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
            handler.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")  # no terminating 0 chunk
        elif framing == "chunked-mid-chunk":
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
            handler.wfile.write(b"%x\r\n" % (len(chunk) + 4096) + chunk)  # chunk never completes
        else:
            handler.send_header("Content-Length", str(len(chunk) + 4096))  # promises more than it sends
            handler.end_headers()
            handler.wfile.write(chunk)
        handler.wfile.flush()
        handler.connection.shutdown(socket.SHUT_RDWR)

    routes[("POST", "/v1/chat/completions")] = drop
    got = []
    with pytest.raises(ChatFailed) as failed:
        got.extend(SwitcherChat(url).stream({}, "s"))
    assert got and failed.value.code == "cut_off"


def test_probe_reads_the_inflight_snapshot(server):
    routes, _, url = server
    rows = [{"model": "quasar-27b", "req_headers": {"X-Session-ID": "someone"}}]

    def events(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        for envelope in ({"type": "logData", "data": json.dumps({"source": "proxy", "data": "x" * 5000})},
                         {"type": "inflight", "data": json.dumps({"operation": "snapshot", "requests": rows})}):
            handler.wfile.write(b"event:message\ndata:" + json.dumps(envelope).encode() + b"\n\n")
        handler.wfile.flush()
        time.sleep(5)  # the real stream stays open

    routes[("GET", "/api/events")] = events
    started = time.monotonic()
    assert SwitcherProbe(url).inflight() == rows
    assert time.monotonic() - started < 2.0
    routes[("GET", "/api/events")] = lambda h: reply(h, 500, b"no")
    assert SwitcherProbe(url).inflight() is None


def test_probe_gives_up_when_no_snapshot_arrives(server):
    routes, _, url = server

    def only_logs(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(b'event:message\ndata:{"type":"logData","data":"{}"}\n\n')
        handler.wfile.flush()
        time.sleep(5)

    routes[("GET", "/api/events")] = only_logs
    started = time.monotonic()
    assert SwitcherProbe(url, timeout=0.5).inflight() is None
    assert time.monotonic() - started < 2.0


def test_last_used_reads_the_newest_activity_row(server):
    routes, seen, url = server
    routes[("GET", "/api/metrics/activity")] = lambda h: reply(
        h, 200, json.dumps({"data": [{"timestamp": "2026-09-25T10:00:00.123456789Z", "model": "quasar-27b"}],
                            "page": 1, "limit": 1, "total": 3, "total_pages": 3}).encode())
    expected = dt.datetime(2026, 9, 25, 10, 0, 0, 123456, tzinfo=dt.timezone.utc).timestamp()
    assert SwitcherProbe(url).last_used("quasar-27b") == pytest.approx(expected)
    assert "model=quasar-27b" in seen[-1]["path"] and "limit=1" in seen[-1]["path"]
    routes[("GET", "/api/metrics/activity")] = lambda h: reply(h, 200, b'{"data": []}')
    assert SwitcherProbe(url).last_used("quasar-27b") is None
    routes[("GET", "/api/metrics/activity")] = lambda h: reply(h, 500, b"failed to get activity", "text/plain")
    assert SwitcherProbe(url).last_used("quasar-27b") is None


def test_parse_go_time():
    assert parse_go_time("2026-09-25T11:00:00+01:00") == parse_go_time("2026-09-25T10:00:00Z")
    assert parse_go_time("2026-09-25T10:00:00Z") == dt.datetime(2026, 9, 25, 10, tzinfo=dt.timezone.utc).timestamp()
    assert parse_go_time("nonsense") is None and parse_go_time(None) is None and parse_go_time("") is None


def test_an_abort_before_the_stream_holds_until_reset(server):
    """Review item 4: a Stop that lands between two steps must end the next stream too, so
    stream() no longer clears the abort; reset() (a new test) does."""
    routes, seen, url = server
    routes[("POST", "/v1/chat/completions")] = lambda h: reply(h, 200, b"data: [DONE]\n\n", "text/event-stream")
    chat = SwitcherChat(url)
    chat.abort()
    assert list(chat.stream({"model": "m"}, "s")) == [] and seen == []  # never connected
    chat.reset()
    assert [line.strip() for line in chat.stream({"model": "m"}, "s") if line.strip()] == ["data: [DONE]"]


@pytest.mark.parametrize("status, body, result", [
    (200, b"OK", "unloaded"),
    (409, json.dumps({"src": "llama-swap", "error": {"message": "model m is answering a request, so it was not unloaded",
                                                     "type": "invalid_request_error", "code": "busy"}}).encode(), "busy"),
    (409, b'{"error": {"code": "conflict"}}', "failed"),
    (501, b'{"error": {"code": "not_implemented"}}', "failed"),
])
def test_unload_if_idle_uses_p7(server, status, body, result):
    routes, seen, url = server
    routes[("POST", "/api/models/unload/Qwen3.8%2FFlash")] = lambda h: reply(h, status, body)
    assert SwitcherProbe(url).unload_if_idle("Qwen3.8/Flash") == result
    assert seen[0]["path"] == "/api/models/unload/Qwen3.8%2FFlash?ifIdle=1"


def test_unload_if_idle_fails_when_the_switcher_is_down():
    with socket.socket() as spare:
        spare.bind(("127.0.0.1", 0))
        port = spare.getsockname()[1]
    assert SwitcherProbe(f"http://127.0.0.1:{port}").unload_if_idle("m") == "failed"

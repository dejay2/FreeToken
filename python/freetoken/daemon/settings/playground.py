"""The Test tab's server side (own model system part 3).

Runs one prompt on one or two setups (A, then B) on the one graphics card, times each answer,
then puts back what was loaded before. Spec: docs/superpowers/specs/2026-09-25-playground-design.md.

A setup is a model, its saved settings or one of its presets, and answer settings sent with
the request. A preset reaches the switcher through the panel's test overlay
(PanelService.set_test_settings); the registry file is never written. Answers are timed here,
next to the switcher, never in the browser, so tailnet latency never enters the numbers. A
model is loaded through P6 before its answer starts, so load time is its own step.

Rules that keep Jay's models safe (plan Review Focus 1-3):
- put-back (_restore) runs after every ending: done, failed, stopped, yielded;
- a model another app is using (llama-swap's in-flight list) is never put away or loaded over;
- nothing answers unless /running says the setup's model is ready;
- nothing loads until the switcher reports the hash of the file we wrote (P5).

This part of the file is the I/O: the streamed chat (SwitcherChat) and the read-only probe
of llama-swap's in-flight list and activity log (SwitcherProbe). Both talk to the frozen
llama-swap in engines/llama-swap; the shapes they read are pinned there:
- /v1/chat/completions errors: ``{"src": "llama-swap", "error": {"message", "type", "code"}}``
  (internal/swaputil/httperror.go), code ``model_superseded`` for P1 (409) and
  ``not_enough_memory`` for P2 (503); an engine's own error may be plain text;
- /api/events: ``event:message\\ndata:{"type": ..., "data": "<json string>"}`` envelopes; the
  connection opens with the two log histories, model status, UI config, the profile, then one
  ``inflight`` event ``{"operation": "snapshot", "requests": [{"model", "req_headers", ...}]}``
  (internal/server/apigroup.go handleAPIEvents, inflight.go Current);
- /api/metrics/activity?model=<id>&limit=1: ``{"data": [{"timestamp", "model", ...}], "page",
  "limit", "total", "total_pages"}`` sorted by id descending, so data[0] is the newest
  (internal/store/activity.go ActivityPage, sqlite/activity.go activityOrderBy).
"""

from __future__ import annotations

import datetime as _dt
import http.client
import json
import os
import re
import socket
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterator

from .switcher import DEFAULT_URL

# Plan guesses, measured on the RTX 5090 box: a FreeToken boot of Qwen3.8 Flash takes about
# 2.5 min, an NInfer 27B about 20 s (own switcher part 1 acceptance, 2026-09-24).
LOAD_GUESS_S = {"freetoken": 150, "ninfer": 20}
UNLOAD_GUESS_S = {"freetoken": 15, "ninfer": 5}
RECENT_USE_S = 120
MAX_PROMPT_CHARS = 20_000
MAX_SYSTEM_CHARS = 8_000
DEFAULT_ANSWER_TOKENS = 512
MAX_ANSWER_TOKENS = 4_096
MAX_TEXT_CHARS = 60_000
SESSION_HEADER = "X-Session-ID"
SESSION_PREFIX = "ft-test-"
WARMUP_MESSAGES = [{"role": "user", "content": "Reply with the single word OK."}]
WARMUP_TOKENS = 8


class ChatFailed(RuntimeError):
    """A chat that did not stream to its end. ``status`` is 0 when no HTTP answer came back;
    ``code`` is llama-swap's error code, ``switcher_down``, ``cut_off`` or ``chat_failed``."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _base_url(base_url: str | None) -> str:
    return (base_url or os.environ.get("FREETOKEN_SWITCHER_URL") or DEFAULT_URL).rstrip("/")


def _chat_error(status: int, data: bytes) -> ChatFailed:
    """llama-swap answers {"error": {"code", "message"}} for P1 (409 model_superseded) and P2
    (503 not_enough_memory); an engine error may be plain text."""
    text = data.decode("utf-8", "replace")
    try:
        body = json.loads(text)
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return ChatFailed(status, str(error.get("code") or "chat_failed"), str(error.get("message") or text[:300]))
    return ChatFailed(status, "chat_failed", text.strip()[:300] or f"HTTP {status}")


_FRACTION = re.compile(r"\.(\d{6})\d+")


def parse_go_time(value: Any) -> float | None:
    """Go's RFC 3339 time (up to nanoseconds, Z or an offset) as epoch seconds.

    Go marshals time.Time as RFC3339Nano; Python's fromisoformat takes at most six fraction
    digits, so the rest is cut off (a loss under a microsecond)."""
    if not isinstance(value, str) or not value:
        return None
    text = _FRACTION.sub(r".\1", value.strip()).replace("Z", "+00:00")
    try:
        moment = _dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_dt.timezone.utc)
    return moment.timestamp()


class SwitcherChat:
    """One streamed chat at a time through the switcher. abort() may come from any thread: it
    shuts the socket down, and the stream then simply ends."""

    def __init__(self, base_url: str | None = None, *, timeout: float = 900.0) -> None:
        self.base_url, self.timeout = _base_url(base_url), timeout
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._aborted = False

    def stream(self, body: dict[str, Any], session: str) -> Iterator[str]:
        """Yield the raw SSE lines of one chat. Raises ChatFailed on a non-200 answer, on a
        refused connection (switcher_down) or when the connection drops (cut_off)."""
        parts = urllib.parse.urlsplit(self.base_url)
        conn = http.client.HTTPConnection(parts.hostname or "127.0.0.1", parts.port or 80, timeout=self.timeout)
        with self._lock:
            self._aborted = False
        try:
            try:
                conn.connect()
                with self._lock:
                    self._sock = conn.sock
                    aborted = self._aborted
                if aborted:  # Stop landed between connect() and the socket being published
                    return
                conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode("utf-8"),
                             headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                                      SESSION_HEADER: session})
                response = conn.getresponse()
            except OSError as exc:
                if self._aborted:
                    return
                raise ChatFailed(0, "switcher_down", f"the model switcher did not answer ({exc})") from exc
            if response.status != 200:
                raise _chat_error(response.status, response.read(8192))
            # Lines are split here from read1() rather than taken from response.readline():
            # llama-swap streams chunked, and on Python 3.13 HTTPResponse.readline() swallows
            # the IncompleteRead of a dropped chunked stream (its peek() path) and returns a
            # plain EOF, so a crashed engine would look like a clean end. read1() lets the
            # IncompleteRead through (measured with a local drop-after-one-chunk server,
            # tests/settings/test_playground_io.py::test_dropped_connection_is_cut_off).
            pending = b""
            while True:
                try:
                    block = response.read1(65536)
                except (OSError, ValueError, http.client.HTTPException):
                    # a dropped stream (IncompleteRead) or a socket shut down by abort()
                    if self._aborted:
                        return
                    raise ChatFailed(0, "cut_off", "the connection closed before the answer finished") from None
                if not block:
                    if pending:
                        yield pending.decode("utf-8", "replace")
                    # Under a Content-Length a drop is a plain EOF; the bytes still owed tell it
                    # apart from a clean end.
                    if response.length and not self._aborted:
                        raise ChatFailed(0, "cut_off", "the connection closed before the answer finished")
                    return
                pending += block
                while (cut := pending.find(b"\n")) >= 0:
                    line, pending = pending[:cut + 1], pending[cut + 1:]
                    yield line.decode("utf-8", "replace")
        finally:
            with self._lock:
                self._sock = None
            conn.close()

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class SwitcherProbe:
    """What llama-swap knows about other apps' requests (no patch needed: /api/events opens with
    an in-flight snapshot, and /api/metrics/activity lists requests newest first)."""

    def __init__(self, base_url: str | None = None, *, timeout: float = 5.0,
                 urlopen: Callable[..., Any] = urllib.request.urlopen) -> None:
        self.base_url, self.timeout, self._urlopen = _base_url(base_url), timeout, urlopen

    def inflight(self) -> list[dict[str, Any]] | None:
        """The rows of the first in-flight snapshot on /api/events, or None when unknown
        (switcher down, an error answer, or no snapshot within the timeout)."""
        request = urllib.request.Request(self.base_url + "/api/events", headers={"Accept": "text/event-stream"})
        deadline = time.monotonic() + self.timeout
        try:
            with self._urlopen(request, timeout=self.timeout) as response:
                while time.monotonic() < deadline:
                    raw = response.readline()
                    if not raw:
                        return None
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        envelope = json.loads(line[5:])
                    except ValueError:
                        continue
                    if not isinstance(envelope, dict) or envelope.get("type") != "inflight":
                        continue
                    try:
                        event = json.loads(envelope.get("data") or "{}")
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("operation") == "snapshot":
                        return [row for row in event.get("requests") or [] if isinstance(row, dict)]
        except (OSError, ValueError):
            return None
        return None

    def last_used(self, model_id: str) -> float | None:
        """Epoch seconds of the newest activity row for the model, or None when there is none
        or the switcher did not answer."""
        query = urllib.parse.urlencode({"model": model_id, "limit": 1})
        try:
            with self._urlopen(f"{self.base_url}/api/metrics/activity?{query}", timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8") or "{}")
        except (OSError, ValueError):
            return None
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return None
        return parse_go_time(rows[0].get("timestamp"))

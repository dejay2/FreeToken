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

import copy
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
import uuid
from typing import Any, Callable, Iterator

from .panel import PanelError
from .playground_speed import AnswerTracker, answer_stats
from .registry import RegistryCorrupt, RegistryError, RegistryMissing, find_model
from .swap_config import SwitcherRefused
from .switcher import DEFAULT_URL, LOADED_STATES, SwitcherError, is_down

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


# ---- the runner: plan part (Task 4). The run, put-back, stop and recover part follows. ----

ACTIVE = ("running", "stopping", "restoring")
SAMPLING = (("temperature", "Creativity (temperature)", 0.0, 2.0, False),
            ("top_p", "Top-p", 0.0, 1.0, False),
            ("top_k", "Top-k", 0, 200, True))
STOPPED = "Stopped."
YIELDED = "Another app asked for a different model, so the test stopped to let it through."


class PlaygroundError(RuntimeError):
    def __init__(self, status: int, code: str, message: str, extra: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status, self.payload = status, {"code": code, "message": message, **(extra or {})}


class _Halt(Exception):
    """Ends a test early; kind becomes the job's status: failed, stopped or yielded."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind, self.message = kind, message


def _thread(fn: Callable[..., Any], *args: Any) -> None:
    threading.Thread(target=fn, args=args, name="playground-test", daemon=True).start()


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _header(row: dict[str, Any], name: str) -> str:
    headers = row.get("req_headers") if isinstance(row.get("req_headers"), dict) else {}
    for key, value in headers.items():
        if str(key).lower() == name.lower():
            return str(value)
    return ""


def _seconds(ms: int) -> str:
    if ms < 60_000:
        return f"{ms / 1000:.1f} s"
    total = round(ms / 1000)
    return f"{total // 60} min {total % 60} s"


class PlaygroundRunner:
    """One test at a time; the job dict is what GET /api/playground/runs/current returns."""

    def __init__(self, panel: Any, *, chat: Any = None, probe: Any = None,
                 spawn: Callable[..., None] | None = None, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time) -> None:
        self.panel, self.switcher = panel, panel.switcher
        self.chat = chat if chat is not None else SwitcherChat()
        self.probe = probe if probe is not None else SwitcherProbe()
        self._spawn = spawn or _thread
        self._clock, self._wall = clock, wall
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self.job: dict[str, Any] | None = None
        self._session = ""
        self._loading: str | None = None
        self._ours: str | None = None  # the model this test loaded last
        self._own_last: dict[str, float] = {}
        self._aliases: dict[str, set[str]] = {}
        self._labels: dict[str, str] = {}

    # ---- reading ----
    def _doc(self) -> dict[str, Any]:
        try:
            doc, _ = self.panel.store.load()
        except RegistryMissing:
            raise PlaygroundError(409, "registry_missing",
                                  "The control panel has no model list yet. Copy today's settings first.") from None
        except RegistryCorrupt as exc:
            raise PlaygroundError(409, "registry_corrupt", exc.message) from None
        self._aliases = {m["id"]: {m["id"], *(m.get("aliases") or [])} for m in doc["models"]}
        self._labels = {m["id"]: m["name"] for m in doc["models"]}
        return doc

    def _name(self, model_id: str | None) -> str:
        return self._labels.get(model_id or "", model_id or "")

    def _others_using(self, model_id: str) -> bool | None:
        """True when another app has a request in flight on model_id (by id or alias); None
        when llama-swap cannot tell. The test's own requests carry its session header."""
        rows = self.probe.inflight()
        if rows is None:
            return None
        names = self._aliases.get(model_id, {model_id})
        return any(str(row.get("model")) in names
                   and not (self._session and _header(row, SESSION_HEADER) == self._session) for row in rows)

    def _card(self) -> str | None:
        running = self.switcher.running()
        if running is None:
            raise PlaygroundError(503, "switcher_unknown", "Can't tell which model is loaded right now. Try again in a moment.")
        if is_down(running):
            raise PlaygroundError(503, "switcher_down", "The model switcher is not running.")
        if any(state not in ("ready", "stopped", "shutdown") for state in running.values()):
            raise PlaygroundError(409, "loading", "A model is loading or unloading right now. Try again when it has finished.")
        ready = sorted(model for model, state in running.items() if state == "ready")
        return ready[0] if ready else None

    # ---- the request ----
    def _request(self, body: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise PlaygroundError(422, "prompt", "Type a prompt first.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise PlaygroundError(422, "prompt", f"The prompt is too long: at most {MAX_PROMPT_CHARS:,} characters.")
        system = str(body.get("system") or "").strip()
        if len(system) > MAX_SYSTEM_CHARS:
            raise PlaygroundError(422, "system", f"The system message is too long: at most {MAX_SYSTEM_CHARS:,} characters.")
        raw = body.get("sides")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 2:
            raise PlaygroundError(422, "sides", "Pick one or two setups.")
        sides = [self._side(key, item if isinstance(item, dict) else {}, doc) for key, item in zip("AB", raw)]
        return {"prompt": prompt, "system": system, "sides": sides,
                "warmup": bool(body.get("warmup", True)), "putBack": bool(body.get("putBack", True))}

    def _side(self, key: str, item: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
        model_id = str(item.get("model") or "")
        try:
            model = find_model(doc, model_id)
        except KeyError:
            raise PlaygroundError(422, "model", f"Setup {key}: pick a model.") from None
        preset = item.get("preset") or None
        if preset is not None and preset not in (model.get("presets") or {}):
            raise PlaygroundError(422, "preset", f"Setup {key}: the preset “{preset}” no longer exists.")
        sampling: dict[str, Any] = {}
        for name, label, low, high, whole in SAMPLING:
            value = item.get(name)
            if value is None or value == "":
                continue
            try:
                number = int(value) if whole else float(value)
            except (TypeError, ValueError):
                raise PlaygroundError(422, name, f"Setup {key}: {label} must be a number.") from None
            if isinstance(value, bool) or not low <= number <= high:
                raise PlaygroundError(422, name, f"Setup {key}: {label} must be between {low:g} and {high:g}.")
            sampling[name] = number
        try:
            tokens = int(item.get("maxTokens", DEFAULT_ANSWER_TOKENS))
        except (TypeError, ValueError):
            tokens = 0
        if not 1 <= tokens <= MAX_ANSWER_TOKENS:
            raise PlaygroundError(422, "maxTokens",
                                  f"Setup {key}: Longest answer must be between 1 and {MAX_ANSWER_TOKENS:,} tokens.")
        sampling["max_tokens"] = tokens
        return {"key": key, "model": model_id, "name": model["name"], "engine": model["engine"], "preset": preset,
                "runPreset": self.panel.test_preset_key(model_id, preset),
                "settingsLabel": f"preset “{preset}”" if preset else "Saved settings",
                "sampling": sampling, "answer": "", "reasoning": "", "stats": None, "loadMs": None, "error": None}

    # ---- the plan ----
    def _steps(self, sides: list[dict[str, Any]], before: str | None, held: bool, warmup: bool, put_back: bool,
               engines: dict[str, str]) -> list[dict[str, Any]]:
        """The steps the job walks. A setup already on the card on the right settings is not
        reloaded; a model on "old settings until the next load" (held) always is."""
        steps: list[dict[str, Any]] = []
        current: tuple[str | None, str | None] = (before, "(held)" if held else None)

        def add(kind: str, side: str | None, model: str | None, preset: str | None, label: str, guess: int) -> None:
            steps.append({"kind": kind, "side": side, "model": model, "preset": preset, "label": label,
                          "guessS": guess, "state": "waiting", "ms": None, "detail": ""})

        for side in sides:
            key, model, preset = side["key"], side["model"], side["runPreset"]
            if current != (model, preset):
                if current[0] is not None:
                    add("unload", key, current[0], None, f"Put away {self._name(current[0])}",
                        UNLOAD_GUESS_S[engines[current[0]]])
                words = f"preset “{preset}”" if preset else "saved settings"
                add("settings", key, model, preset, f"Use {words} for {side['name']}", 0)
                add("load", key, model, preset, f"Load {side['name']}", LOAD_GUESS_S[side["engine"]])
                current = (model, preset)
            if warmup:
                add("warmup", key, model, preset, f"Warm up {side['name']} (not counted)", 5)
            add("answer", key, model, preset, f"Answer with setup {key}", 30)
        guess = 0
        if current != (before, None):
            if current[0] is not None and (put_back or current[1] is not None or before is None):
                guess += UNLOAD_GUESS_S[engines[current[0]]]
            if put_back and before is not None:
                guess += LOAD_GUESS_S[engines[before]]
        label = f"Put things back: {self._name(before)} on its saved settings" if before and put_back else "Put things back"
        add("restore", None, before, None, label, guess)
        return steps

    def plan(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.job is not None and self.job["status"] in ACTIVE:
                raise PlaygroundError(409, "test_running", "A test is already running.")
        doc = self._doc()
        request = self._request(body, doc)
        before = self._card()
        held = before is not None and before in self.panel.held_models()
        engines = {m["id"]: m["engine"] for m in doc["models"]}
        if before is not None:
            busy = self._others_using(before)
            if busy is None:
                raise PlaygroundError(503, "switcher_unknown",
                                      "Can't tell whether the loaded model is busy. Try again in a moment.")
            if busy:
                raise PlaygroundError(409, "in_use",
                                      f"{self._name(before)} is answering something right now. Try again when it's done.")
        steps = self._steps(request["sides"], before, held, request["warmup"], request["putBack"], engines)
        warnings: list[str] = []
        if before is not None and any(step["kind"] == "unload" and step["model"] == before for step in steps):
            used, own = self.probe.last_used(before), self._own_last.get(before)
            # The test's own requests show up in the activity list too; they end before own.
            if used is not None and (own is None or used > own + 2) and self._wall() - used < RECENT_USE_S:
                seconds = max(1, round(self._wall() - used))
                warnings.append(f"{self._name(before)} was used {seconds} seconds ago. The test will put it away.")
        return {**request, "before": before, "held": held, "steps": steps,
                "estimateS": sum(step["guessS"] for step in steps), "warnings": warnings}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self.job is None:
                return {"status": "idle"}
            out = copy.deepcopy(self.job)
            now = self._clock()
        for step in out["steps"]:
            started = step.pop("_t0", None)
            if step["state"] == "running" and started is not None:
                step["elapsedMs"] = max(0, round((now - started) * 1000))
        return out

# Test Playground (own model system, part 3): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A **Test** tab on the control panel. It runs one prompt on one or two setups (a
model, its saved settings or a preset, and answer settings), one after the other on the one
graphics card. Each answer gets speed numbers: first word after, writing speed, whole answer,
sizes, guesses kept. Loading is timed as its own step. Afterwards the tab always puts back what
was loaded before, and never takes the card from another app. It keeps the last 20 tests in
the browser, with Copy and Export.

**Architecture:**
- `daemon/settings/playground_speed.py` (pure): folds SSE lines into an `AnswerTracker` and
  turns it into the stats dict.
- `daemon/settings/playground.py`:
  - `SwitcherChat` streams one chat through llama-swap and can be aborted;
  - `SwitcherProbe` reads llama-swap's in-flight snapshot and last activity;
  - `PlaygroundRunner` does plan, start, the step walk, put-back, stop and recover;
  - `create_playground_router` holds the `/api/playground/*` routes.
- `daemon/settings/panel.py` gains:
  - an in-memory **test overlay**: the switcher file and FreeToken's effective settings show
    the test model on one preset. The registry is never written;
  - a marker file for crash recovery;
  - `begin_test`/`end_test` guards on every panel write;
  - `wait_for_switcher`;
  - `test`/`testLeftover` in the Right-now data.
- The page adds `static/playground.js` plus a tab and section in `index.html`. `panel.js` gains
  the tab switch and the strip lines.
- llama-swap's frozen copy is **not** changed.

**Tech Stack:** Python 3.13 (FastAPI, pydantic, pytest, `http.client`, `urllib`), plain
browser JavaScript (node 24 for tests), llama-swap v257 frozen (P1, P2, P4, P5, P6 used as
they are).

**Spec:** `docs/superpowers/specs/2026-09-25-playground-design.md`.

**Dry run (2026-09-25):** every code block in Tasks 1-8 was applied to a scratch copy of
`fedf802`. All 67 new tests passed, and the full `tests/settings tests/daemon` run showed no
new failures besides the two pinned lists in `test_static_page.py` that Task 8 updates.
Implementers still run each red/green step; the dry run only means the code as written is
known to work.

## Global Constraints

- **Branch and worktree:** `feat/playground` in
  `/home/jay/projects/FreeToken/.worktrees/feat-playground`, from `mtp-upstream-merge` at
  `fedf802`. The PR targets `mtp-upstream-merge`.
- **Process rules:**
  - Implementers (subagents) run **no git write commands** (no add, commit, checkout, switch,
    reset, clean, stash, rm, mv, push). The controller commits after each task's review.
  - **Live tests only by the controller**, on the box, after asking Jay. Check
    `curl -s 127.0.0.1:2040/running` first. **Never evict a model Jay is using.** At most
    **one FreeToken boot per live session**.
  - No remote machines for implementers.
  - Two review rounds at most. Each reviewer lists every finding at once.
  - Never force-push. The push target is `origin` (`dejay2/FreeToken`); `upstream` is
    fetch-only.
- **Commits:** `type: subject` (`feat|fix|perf|refactor|build|ci|docs|test|chore`). Every
  message ends with the line `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- **llama-swap:** not touched by this plan. If a fix ever needs it, every changed spot carries
  `// FreeToken patch Pn:`, the patch is listed in `engines/llama-swap/FROZEN.md`, and the UI
  or binary is built only with `scripts/engines/build.sh` (Go 1.27.1, Node 24.9.0 pinned).
- **Fixed numbers:**
  - load guesses `LOAD_GUESS_S = {"freetoken": 150, "ninfer": 20}`;
  - unload guesses `{"freetoken": 15, "ninfer": 5}`;
  - recent use `RECENT_USE_S = 120`;
  - prompt at most 20,000 characters, system message at most 8,000;
  - longest answer 1-4,096 tokens, default 512;
  - warm-up: `"Reply with the single word OK."`, 8 tokens;
  - job text kept per answer: 60,000 characters;
  - page poll 500 ms;
  - history: 20 tests, answers cut at 20,000 characters, key `ft-test-history-v1`;
  - the "better" tag needs a difference of at least 3%.
- **Session header:** every request the test sends carries `X-Session-ID: ft-test-<10 hex>`.
- **Registry is never written by the Test tab.** Presets reach the switcher only through the
  overlay. The marker is `~/.config/freetoken/playground-test.json`, next to
  `held-models.json`.
- **Plain words on the page** for Jay (non-technical). The labels are fixed in the spec's
  decision 13.
- **Commands:**
  - Python tests from the worktree root:
    `PYTHONPATH=python .venv/bin/python -m pytest <paths> -q` (the worktree has its own
    `.venv`).
  - Node tests run inside pytest through `node -e`.

## Review Focus

1. **Settings not put back.** The failure: after a preset test that ends in success, an
   answer error, Stop during a load, a switcher that never picks up the file, or a helper
   restart mid-test, the registry has changed, the switcher file does not match the registry,
   the marker is left, or a model runs test settings with no message. Pinned by:
   - `test_preset_compare_never_writes_registry_and_restores_config`,
     `test_answer_error_still_puts_back`, `test_stop_during_load_cancels_it_and_puts_back`,
     `test_switcher_never_picks_up_settings_nothing_loads` (Task 5);
   - `test_helper_restart_puts_an_idle_test_model_away`,
     `test_helper_restart_leaves_a_busy_test_model_and_says_so` (Task 5);
   - `test_overlay_renders_the_preset_alone_and_clearing_restores` (Task 2).
2. **Taking the card from another app.** The failure: the test puts away or loads over a model
   another app is using. That covers a busy model at plan time, a request arriving mid-test,
   P1 "latest wins" superseding our load, and put-back loading over the other app's model.
   Pinned by:
   - `test_plan_refuses_when_the_loaded_model_is_answering`,
     `test_plan_warns_when_the_loaded_model_was_used_recently` (Task 4);
   - `test_superseded_load_stops_and_does_not_load_over_the_other_app`,
     `test_put_back_leaves_a_busy_test_model_and_holds_its_entry` (Task 5).
3. **Wrong model or wrong settings answering without a word.** The failure: an answer is
   taken from a model that is not the setup's, the load runs before the switcher has the
   test file, or put-back loads before the switcher has the saved file. Pinned by:
   - `test_answer_refused_when_the_card_changed_after_loading`,
     `test_switcher_never_picks_up_settings_nothing_loads` (Task 5);
   - `test_freetoken_preset_reaches_the_adapter_through_effective` (Task 5);
   - `test_panel_writes_refused_while_testing` (Task 5).
4. **Load time counted as answer time.** The failure: "first word after" or "whole answer"
   includes a model load or the warm-up. Pinned by `test_load_time_is_its_own_step` (Task 5)
   and `test_first_word_is_measured_from_the_request` (Task 1).
5. **Speed numbers that claim more than the engine said.** The failures:
   - a writing speed from one token, or a divide by zero;
   - guess-ahead numbers shown for FreeToken;
   - "engine" shown for a speed we measured;
   - token counts from chunks not marked `~`.
   Pinned by `test_ninfer_timings_give_engine_speed_and_guesses`,
   `test_freetoken_usage_gives_measured_speed_and_no_guesses`,
   `test_one_token_answer_has_no_speed`, `test_chunks_stand_in_for_tokens_when_usage_is_missing`
   (Task 1) and `test_times_and_speeds_in_plain_words` (Task 7).

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `python/freetoken/daemon/settings/playground_speed.py` | create | `AnswerTracker`, `answer_stats` (pure) |
| `python/freetoken/daemon/settings/playground.py` | create | `SwitcherChat`, `SwitcherProbe`, `parse_go_time`, `PlaygroundRunner`, `create_playground_router` |
| `python/freetoken/daemon/settings/panel.py` | modify | test overlay, marker, guards, `wait_for_switcher`, `held_models`, `test_preset_key`, `now()` fields |
| `python/freetoken/daemon/settings/app.py` | modify | build the runner, include the router, serve `/playground.js` |
| `python/freetoken/daemon/settings/server.py` | modify | `start_panel(app)`: recover, then sync, then the hold watcher |
| `python/freetoken/daemon/settings/static/playground.js` | create | Test tab (pure helpers + browser part) |
| `python/freetoken/daemon/settings/static/index.html` | modify | tab button, `#test-view` section, CSS, script tag |
| `python/freetoken/daemon/settings/static/panel.js` | modify | `showMain('test')`, Right-now strip test lines |
| `tests/settings/playground_fakes.py` | create | fake card switcher, chat, probe, clock, `make()` |
| `tests/settings/test_playground_speed.py` | create | Task 1 |
| `tests/settings/test_playground_panel.py` | create | Task 2 |
| `tests/settings/test_playground_io.py` | create | Task 3 |
| `tests/settings/test_playground_runner.py` | create | Tasks 4 and 5 |
| `tests/settings/test_playground_routes.py` | create | Task 6 |
| `tests/settings/test_playground_page.py` | create | Tasks 7 and 8 |
| `tests/settings/test_static_page.py` | modify (Task 8) | two pinned lists gain `/playground.js` and the `test` tab |
| `README.md` | modify | one paragraph under "Windows settings helper" |
| `docs/research/playground-acceptance-<date>.md`, `docs/research/img/playground-*.png` | create (Task 10) | live acceptance |

---

### Task 1: Speed numbers (`playground_speed.py`)

**Files:**
- Create: `python/freetoken/daemon/settings/playground_speed.py`
- Test: `tests/settings/test_playground_speed.py`

**Interfaces:**
- `AnswerTracker(started: float)`:
  - `.feed_line(line: str, now: float) -> None`;
  - `.answer_text`, `.reasoning_text`;
  - fields `first_token`, `answer_started`, `last_token`, `ended`, `chunks`, `usage`,
    `timings`, `finish_reason`, `served_model`, `done`.
- `answer_stats(tracker, *, cancelled=False) -> dict`. The keys: `firstWordMs`,
  `answerStartMs`, `thinkingMs`, `totalMs`, `promptTokens`, `cachedTokens`,
  `completionTokens`, `approxTokens`, `writeTps`, `writeSource` (`"engine"`, `"measured"`
  or `None`), `guesses` (`{"proposed", "kept", "keptPct"}` or `None`) and `finishReason`.

- [ ] **Step 1: Write the failing tests**

```python
"""Speed numbers for one streamed answer. Review focus 4 and 5."""

from __future__ import annotations

import json

from freetoken.daemon.settings.playground_speed import AnswerTracker, answer_stats


def sse(obj) -> str:
    return "data: " + json.dumps(obj) + "\n"


def feed(tracker, timed_lines):
    for at, line in timed_lines:
        tracker.feed_line(line, at)


def test_freetoken_usage_gives_measured_speed_and_no_guesses():
    # FreeToken (server/openai_api.py): a role chunk, content chunks, a finish chunk, then a
    # usage-only chunk with cached tokens under prompt_tokens_details; no timings block.
    t = AnswerTracker(started=10.0)
    feed(t, [
        (10.1, sse({"model": "qwen3.8-flash", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})),
        (10.5, sse({"choices": [{"index": 0, "delta": {"content": "Hello"}}]})),
        (11.0, sse({"choices": [{"index": 0, "delta": {"content": " there"}}]})),
        (11.5, sse({"choices": [{"index": 0, "delta": {"content": " friend"}}]})),
        (11.5, sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})),
        (11.5, sse({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 11, "total_tokens": 111,
                                              "prompt_tokens_details": {"cached_tokens": 64}}})),
        (11.5, "data: [DONE]\n"),
    ])
    t.ended = 11.6
    stats = answer_stats(t)
    assert t.answer_text == "Hello there friend" and t.done and t.served_model == "qwen3.8-flash"
    assert stats["firstWordMs"] == 500 and stats["totalMs"] == 1600
    assert stats["promptTokens"] == 100 and stats["cachedTokens"] == 64 and stats["completionTokens"] == 11
    assert stats["approxTokens"] is False
    assert stats["writeTps"] == 10.0 and stats["writeSource"] == "measured"  # (11 - 1) / (11.5 - 10.5)
    assert stats["guesses"] is None and stats["finishReason"] == "stop"


def test_ninfer_timings_give_engine_speed_and_guesses():
    # NInfer (src/serve/openai_chat_response.cpp): usage plus llama.cpp-style timings with
    # draft_n / draft_n_accepted when speculation ran.
    t = AnswerTracker(started=0.0)
    feed(t, [
        (0.3, sse({"model": "quasar-27b", "choices": [{"index": 0, "delta": {"content": "Hi"}}]})),
        (1.3, sse({"choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": "length"}]})),
        (1.3, sse({"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 42},
                   "timings": {"cache_n": 12, "prompt_n": 18, "prompt_ms": 80.0, "predicted_n": 42,
                               "predicted_ms": 1000.0, "predicted_per_second": 41.26,
                               "draft_n": 514, "draft_n_accepted": 365}})),
        (1.3, "data: [DONE]\n"),
    ])
    stats = answer_stats(t)
    assert stats["writeTps"] == 41.3 and stats["writeSource"] == "engine"
    assert stats["guesses"] == {"proposed": 514, "kept": 365, "keptPct": 71}
    assert stats["cachedTokens"] == 12 and stats["promptTokens"] == 30
    assert stats["finishReason"] == "length"


def test_first_word_is_measured_from_the_request():
    # The runner starts the tracker when it sends the request, after the model is loaded; a
    # load can therefore never show up in firstWordMs.
    t = AnswerTracker(started=500.0)
    t.feed_line(sse({"choices": [{"delta": {"content": "A"}}]}), 500.25)
    assert answer_stats(t)["firstWordMs"] == 250


def test_thinking_then_answer():
    t = AnswerTracker(started=0.0)
    feed(t, [
        (0.2, sse({"choices": [{"delta": {"reasoning_content": "Let me think"}}]})),
        (1.2, sse({"choices": [{"delta": {"reasoning_content": " more"}}]})),
        (1.7, sse({"choices": [{"delta": {"content": "Answer"}}]})),
    ])
    stats = answer_stats(t)
    assert t.reasoning_text == "Let me think more" and t.answer_text == "Answer"
    assert stats["firstWordMs"] == 200 and stats["answerStartMs"] == 1700 and stats["thinkingMs"] == 1500


def test_one_token_answer_has_no_speed():
    t = AnswerTracker(started=0.0)
    feed(t, [(0.1, sse({"choices": [{"delta": {"content": "OK"}}]})),
             (0.1, sse({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}))])
    stats = answer_stats(t)
    assert stats["writeTps"] is None and stats["writeSource"] is None and stats["completionTokens"] == 1


def test_chunks_stand_in_for_tokens_when_usage_is_missing():
    t = AnswerTracker(started=0.0)
    feed(t, [(0.1, sse({"choices": [{"delta": {"content": "a"}}]})),
             (0.2, sse({"choices": [{"delta": {"content": "b"}}]})),
             (0.3, sse({"choices": [{"delta": {"content": "c"}}]}))])
    stats = answer_stats(t, cancelled=True)
    assert stats["completionTokens"] == 3 and stats["approxTokens"] is True
    assert stats["writeTps"] == 10.0 and stats["writeSource"] == "measured"
    assert stats["finishReason"] == "cancelled" and stats["promptTokens"] is None


def test_junk_lines_and_bad_numbers_are_ignored():
    t = AnswerTracker(started=0.0)
    for line in ("", ": keep-alive\n", "event: message\n", "data: {not json\n", "data: [1, 2]\n",
                 sse({"choices": ["x", {"delta": "y"}]}),
                 sse({"choices": [], "usage": {"prompt_tokens": True, "completion_tokens": float("nan")}})):
        t.feed_line(line, 1.0)
    stats = answer_stats(t)
    assert t.chunks == 0 and not t.done
    assert stats["promptTokens"] is None and stats["completionTokens"] is None and stats["firstWordMs"] is None
```

Note: `json.dumps(float("nan"))` writes `NaN`, which `json.loads` reads back as a float NaN.
The `_number` guard must drop it.

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_speed.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'freetoken.daemon.settings.playground_speed'`.

- [ ] **Step 3: Write the module**

```python
"""Speed numbers for one streamed chat answer (the Test tab, own model system part 3). Pure.

What each engine sends on a streamed /v1/chat/completions with stream_options.include_usage
(read from the code, 2026-09-25):
- FreeToken (python/freetoken/server/openai_api.py): a final chunk with `usage`
  {prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details.cached_tokens only
  when nonzero}; no `timings`, and no guess-ahead (MTP) counts per request.
- NInfer, both copies (engines/ninfer*/src/serve/openai_chat_response.cpp): the usage chunk
  also carries llama.cpp-style `timings` {cache_n, prompt_n, prompt_ms, predicted_n,
  predicted_ms, predicted_per_second, ...} and draft_n / draft_n_accepted when speculation ran.
So first word, thinking time and whole answer come from our own clock; writing speed is the
engine's when it reports one ("engine") and ours otherwise ("measured": n tokens have n - 1
gaps, and the first token's own wait is "first word after"); guesses kept only when the
engine reports them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _ms(start: float | None, end: float | None) -> int | None:
    if start is None or end is None:
        return None
    return max(0, round((end - start) * 1000))


@dataclass
class AnswerTracker:
    started: float
    first_token: float | None = None
    answer_started: float | None = None
    last_token: float | None = None
    ended: float | None = None
    chunks: int = 0
    answer: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    served_model: str | None = None
    done: bool = False

    @property
    def answer_text(self) -> str:
        return "".join(self.answer)

    @property
    def reasoning_text(self) -> str:
        return "".join(self.reasoning)

    def feed_line(self, line: str, now: float) -> None:
        """Fold one SSE line in. Anything that is not a JSON data line is ignored."""
        text = line.strip()
        if not text.startswith("data:"):
            return
        data = text[5:].strip()
        if data == "[DONE]":
            self.done = True
            return
        try:
            payload = json.loads(data)
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        if isinstance(payload.get("model"), str):
            self.served_model = payload["model"]
        if isinstance(payload.get("usage"), dict):
            self.usage = payload["usage"]
        if isinstance(payload.get("timings"), dict):
            self.timings = payload["timings"]
        for choice in payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            content = delta.get("content") if isinstance(delta.get("content"), str) else ""
            thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
            thought = thought if isinstance(thought, str) else ""
            if content or thought or delta.get("tool_calls"):
                self.chunks += 1
                if self.first_token is None:
                    self.first_token = now
                self.last_token = now
            if content:
                self.answer.append(content)
                if self.answer_started is None:
                    self.answer_started = now
            if thought:
                self.reasoning.append(thought)
            reason = choice.get("finish_reason")
            if isinstance(reason, str) and reason:
                self.finish_reason = reason


def answer_stats(tracker: AnswerTracker, *, cancelled: bool = False) -> dict[str, Any]:
    usage, timings = tracker.usage, tracker.timings
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    prompt = _number(usage.get("prompt_tokens"))
    cached = _number(details.get("cached_tokens"))
    completion = _number(usage.get("completion_tokens"))
    if prompt is None and _number(timings.get("prompt_n")) is not None:
        prompt = _number(timings.get("prompt_n")) + (_number(timings.get("cache_n")) or 0.0)
    if cached is None:
        cached = _number(timings.get("cache_n"))
    if completion is None:
        completion = _number(timings.get("predicted_n"))
    approx = completion is None and tracker.chunks > 0
    if approx:
        completion = float(tracker.chunks)
    end = tracker.ended if tracker.ended is not None else tracker.last_token
    stats: dict[str, Any] = {
        "firstWordMs": _ms(tracker.started, tracker.first_token),
        "answerStartMs": _ms(tracker.started, tracker.answer_started),
        "thinkingMs": _ms(tracker.first_token, tracker.answer_started) if tracker.reasoning else None,
        "totalMs": _ms(tracker.started, end),
        "promptTokens": None if prompt is None else int(prompt),
        "cachedTokens": None if cached is None else int(cached),
        "completionTokens": None if completion is None else int(completion),
        "approxTokens": approx,
        "writeTps": None,
        "writeSource": None,
        "guesses": None,
        "finishReason": tracker.finish_reason or ("cancelled" if cancelled else None),
    }
    engine_rate = _number(timings.get("predicted_per_second"))
    first, last = tracker.first_token, tracker.last_token
    if engine_rate is not None and engine_rate > 0:
        stats["writeTps"], stats["writeSource"] = round(engine_rate, 1), "engine"
    elif completion is not None and completion >= 2 and first is not None and last is not None and last > first:
        stats["writeTps"], stats["writeSource"] = round((completion - 1) / (last - first), 1), "measured"
    proposed, kept = _number(timings.get("draft_n")), _number(timings.get("draft_n_accepted"))
    if proposed is not None and proposed > 0 and kept is not None:
        stats["guesses"] = {"proposed": int(proposed), "kept": int(kept), "keptPct": round(100 * kept / proposed)}
    return stats
```

- [ ] **Step 4: Run them and see them pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_speed.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Controller commits**

```bash
git add python/freetoken/daemon/settings/playground_speed.py tests/settings/test_playground_speed.py
git commit -m "feat(settings): speed numbers for one streamed answer

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Test overlay and guards in the panel (`panel.py`)

**Files:**
- Modify: `python/freetoken/daemon/settings/panel.py`
- Test: `tests/settings/test_playground_panel.py`

**Interfaces (new on `PanelService`):**
- fields `test_running: bool`, `test_settings: dict | None` (`{"model", "preset"}`),
  `test_leftover: dict | None`, `test_marker_path: Path`;
- `begin_test()` / `end_test()`: `begin_test` raises `PanelError(409, "test_running")` when
  a test is already running;
- `held_models() -> list[str]`;
- `test_preset_key(model_id, preset) -> str | None`: `None` when running that preset gives
  exactly the saved settings;
- `set_test_settings(model_id | None, preset | None) -> str`: sets or clears the overlay,
  rewrites the switcher file (holds as usual) and returns the text;
- `read_test_marker() -> dict | None`, `clear_test_marker()`,
  `note_test_leftover(model_id, preset)`;
- `wait_for_switcher(text) -> bool`: bounded by `restart_wait_s`;
- `now()` gains `"test"` (`{"running", "model", "name", "preset"}` or `None`) and
  `"testLeftover"` (`{"model", "name", "preset"}` or `None`).

- [ ] **Step 1: Write the failing tests**

```python
"""The panel's test overlay, guards and marker (Test tab). Review focus 1 and 3."""

from __future__ import annotations

import pytest

from freetoken.daemon.settings.panel import PanelError
from freetoken.daemon.settings.registry import find_model
from freetoken.daemon.settings.swap_config import extract_model_blocks, render_config
from tests.settings.test_panel_routes import env, seed  # noqa: F401 - the shared fixture
from tests.settings.registry_fixtures import five


def with_presets():
    doc = five()
    find_model(doc, "quasar-27b")["presets"] = {
        "Fast": {"kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 3},
        "Same": {"kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 7},
    }
    find_model(doc, "qwen3.8-flash")["presets"] = {"Short": {"KVCacheTokens": 131072}}
    return doc


def test_overlay_renders_the_preset_alone_and_clearing_restores(env):
    seed(env, with_presets())
    registry = env.store.path.read_bytes()
    text = env.service.set_test_settings("quasar-27b", "Fast")
    assert env.cfg.read_text() == text
    assert "--draft-tokens 3" in extract_model_blocks(text)["quasar-27b"]
    assert env.service.read_test_marker()["model"] == "quasar-27b"
    assert env.store.path.read_bytes() == registry  # the registry is never written
    back = env.service.set_test_settings(None, None)
    assert back == render_config(env.store.load()[0], {}) == env.cfg.read_text()
    assert env.service.read_test_marker() is None and env.service.test_settings is None


def test_unknown_preset_is_refused(env):
    seed(env, with_presets())
    with pytest.raises(KeyError):
        env.service.set_test_settings("quasar-27b", "Nope")
    assert env.service.test_settings is None


def test_effective_follows_the_overlay(env):
    seed(env, with_presets())
    read = lambda: env.client.get("/api/panel/models/qwen3.8-flash/effective").json()["settings"]["KVCacheTokens"]
    assert read() == 262208
    env.service.set_test_settings("qwen3.8-flash", "Short")
    assert read() == 131072
    env.service.set_test_settings(None, None)
    assert read() == 262208


def test_a_preset_equal_to_saved_needs_no_restart(env):
    seed(env, with_presets())
    assert env.service.test_preset_key("quasar-27b", "Same") is None
    assert env.service.test_preset_key("quasar-27b", "Fast") == "Fast"
    assert env.service.test_preset_key("quasar-27b", None) is None
    assert env.service.test_preset_key("qwen3.8-flash", "Short") == "Short"


def test_every_panel_write_is_refused_while_testing(env):
    revision = seed(env, with_presets())
    env.service.begin_test()
    with pytest.raises(PanelError) as again:
        env.service.begin_test()
    assert again.value.payload["code"] == "test_running"
    save = env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7}})
    assert save.status_code == 409 and save.json()["code"] == "test_running"
    assert "Test tab" in save.json()["message"]
    assert env.client.post("/api/panel/models/twin-27b/load").status_code == 409
    assert env.client.post("/api/panel/models/twin-27b/unload").status_code == 409
    assert env.client.post("/api/panel/registry/restore", json={"backup": "x"}).status_code == 409
    assert env.switcher.calls == []
    env.service.end_test()
    assert env.client.put("/api/panel/system", json={"revision": revision, "system": {"floorGB": 7}}).status_code == 200


def test_wait_for_switcher(env):
    seed(env, with_presets())
    assert env.service.wait_for_switcher(env.cfg.read_text()) is True
    env.service.restart_wait_s = 0
    assert env.service.wait_for_switcher("something else") is False


def test_now_shows_the_test_and_prunes_a_leftover_once_unloaded(env):
    seed(env, with_presets())
    env.switcher.states = {"quasar-27b": "ready"}
    env.service.begin_test()
    env.service.set_test_settings("quasar-27b", "Fast")
    now = env.client.get("/api/panel/now").json()
    assert now["test"] == {"running": True, "model": "quasar-27b",
                           "name": "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)", "preset": "Fast"}
    env.service.note_test_leftover("quasar-27b", "Fast")
    env.service.end_test()
    now = env.client.get("/api/panel/now").json()
    assert now["test"] is None and now["testLeftover"]["preset"] == "Fast"
    env.switcher.states = {}
    assert env.client.get("/api/panel/now").json()["testLeftover"] is None
```

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_panel.py -q`
Expected: failures with `AttributeError: 'PanelService' object has no attribute 'set_test_settings'`,
and the same for `test_preset_key`, `begin_test`, `wait_for_switcher` and `note_test_leftover`.

- [ ] **Step 3: Implement in `panel.py`**

1. Under `RESTART_STALE_MESSAGE` add:

```python
TEST_RUNNING_MESSAGE = "A test is running on the Test tab. Wait for it to finish, or stop it there."
TEST_MARKER = "playground-test.json"
```

2. At the end of `PanelService.__init__` add:

```python
        # Test tab (part 3). The overlay is what the switcher and FreeToken's adapter see during
        # a test: the test model on one preset alone. It lives in memory and is never written to
        # the registry; the marker lets a restarted helper find a model left on test settings.
        self.test_running = False
        self.test_settings: dict[str, Any] | None = None
        self.test_leftover: dict[str, Any] | None = None
        self.test_marker_path = self.holds_path.with_name(TEST_MARKER)
```

3. Guards. Add `self._guard_test()` as the **first line inside `with self._lock:`** in
   `_save`, `import_live` and `restore`. In `load` and `unload` add it as the first line of the
   method body.

4. `_plan_rewrite`: the first statement after its docstring becomes
   `doc = self._for_switcher(doc)`.

5. `effective`: replace `doc, _ = self.store.load()` with

```python
        doc, _ = self.store.load()
        doc = self._for_switcher(doc)  # the FreeToken adapter reads the test settings during a test
```

6. `_reload_after_write`: replace its wait loop with the shared method.

```python
    def _reload_after_write(self, model_ids: list[str], text: str) -> None:
        # Bounded wait (Task 7 review): if the switcher's rebuild fails for a reason
        # --check-config misses, its hash never changes; loading then would start the model on
        # the old settings, so give up with a plain message instead.
        if not self.wait_for_switcher(text):
            self._restart_done(model_ids, False, RESTART_STALE_MESSAGE)
            return
        for model_id in model_ids:
            ...  # unchanged from here
```

7. New section before `# ---- holds and start-up ----`:

```python
    # ---- test tab (part 3) ----
    def _guard_test(self) -> None:
        if self.test_running:
            raise PanelError(409, "test_running", TEST_RUNNING_MESSAGE)

    def begin_test(self) -> None:
        with self._lock:
            self._guard_test()
            self.test_running = True
            self.test_leftover = None

    def end_test(self) -> None:
        with self._lock:
            self.test_running = False

    def held_models(self) -> list[str]:
        return sorted(self._read_holds())

    def _for_switcher(self, doc: Mapping[str, Any]) -> Mapping[str, Any]:
        """doc as the switcher and the FreeToken adapter should see it: during a test the test
        model runs the chosen preset alone, without the model's own overrides (a preset is a
        full snapshot of the settings it was saved from, see preset_action "add")."""
        test = self.test_settings
        if not test:
            return doc
        out = copy.deepcopy(dict(doc))
        try:
            model = find_model(out, test["model"])
        except KeyError:
            return doc
        model["activePreset"], model["overrides"] = test["preset"], {}
        return out

    def test_preset_key(self, model_id: str, preset: str | None) -> str | None:
        """preset, or None when running it would give exactly the saved settings (same switcher
        entry and, for FreeToken, the same profile settings), so no restart is needed."""
        if not preset:
            return None
        doc, _ = self.store.load()
        model = find_model(doc, model_id)
        trial = copy.deepcopy(doc)
        tried = find_model(trial, model_id)
        tried["activePreset"], tried["overrides"] = preset, {}
        same_entry = (extract_model_blocks(render_config(doc, {})).get(model_id)
                      == extract_model_blocks(render_config(trial, {})).get(model_id))
        same_profile = model["engine"] != "freetoken" or \
            freetoken_profile_settings(doc, model) == freetoken_profile_settings(trial, tried)
        return None if same_entry and same_profile else preset

    def set_test_settings(self, model_id: str | None, preset: str | None) -> str:
        """Point the switcher at the test settings, or clear them (model_id or preset None), and
        return the text now in the file. Never writes the registry. The marker is written
        before the file when setting, and removed after the file when clearing, so a crash in
        between always leaves a marker for recover()."""
        with self._lock:
            doc, _ = self.store.load()
            if model_id is not None and preset:
                if preset not in (find_model(doc, model_id).get("presets") or {}):
                    raise KeyError(preset)
                self.test_settings = {"model": model_id, "preset": preset}
                self._write_test_marker()
            else:
                self.test_settings = None
            text, holds = self._plan_rewrite(doc)
            if self.writer.write(text):
                self._mark_written()
            self._write_holds(holds)
            if self.test_settings is None:
                self.clear_test_marker()
            return text

    def _write_test_marker(self) -> None:
        self.test_marker_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.test_marker_path.with_name(self.test_marker_path.name + ".tmp")
        temporary.write_text(json.dumps({**(self.test_settings or {}), "at": _now_iso()}) + "\n", encoding="utf-8")
        os.replace(temporary, self.test_marker_path)

    def read_test_marker(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.test_marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and data.get("model") else None

    def clear_test_marker(self) -> None:
        self.test_marker_path.unlink(missing_ok=True)

    def note_test_leftover(self, model_id: str, preset: str | None) -> None:
        self.test_leftover = {"model": model_id, "preset": preset}

    def wait_for_switcher(self, text: str) -> bool:
        """True once the switcher reports text's hash (P5); False after restart_wait_s."""
        expected = config_sha256(text)
        deadline = self._clock() + self.restart_wait_s
        while self.switcher.config_hash() != expected:
            if self._clock() >= deadline:
                return False
            self._sleep(0.5)
        return True
```

8. `now()`: before `return`, add the lines below and the two keys.

```python
        loaded_now = {row["id"] for row in rows if row["state"] in LOADED_STATES}
        leftover = self.test_leftover
        if leftover is not None and up and leftover["model"] not in loaded_now:
            self.test_leftover = leftover = None
        test = None
        if self.test_running:
            overlay = self.test_settings or {}
            test = {"running": True, "model": overlay.get("model"),
                    "name": names.get(overlay.get("model"), overlay.get("model")), "preset": overlay.get("preset")}
        return {"switcher": {...unchanged...}, ..., "lastRestart": last,
                "test": test,
                "testLeftover": None if leftover is None else {**leftover, "name": names.get(leftover["model"], leftover["model"])}}
```

   (`LOADED_STATES` comes from `.switcher`; add it to that import.)

- [ ] **Step 4: Run the new tests and every existing panel test**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_panel.py tests/settings/test_panel_routes.py tests/settings/test_panel_page.py -q`
Expected: every test passes (7 new). `test_restart_now_unloads_writes_and_loads_again` still
passes through `wait_for_switcher`.

- [ ] **Step 5: Controller commits**

```bash
git add python/freetoken/daemon/settings/panel.py tests/settings/test_playground_panel.py
git commit -m "feat(settings): test overlay, marker and write guards in the control panel

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Streaming chat and switcher probe (`playground.py`, I/O part)

**Files:**
- Create: `python/freetoken/daemon/settings/playground.py` (this part: constants, errors,
  `parse_go_time`, `SwitcherChat`, `SwitcherProbe`)
- Test: `tests/settings/test_playground_io.py`

**Interfaces:**
- `SwitcherChat(base_url=None, *, timeout=900.0)`:
  - `.stream(body: dict, session: str) -> Iterator[str]` yields raw SSE lines;
  - it raises `ChatFailed(status, code, message)` on a non-200 answer, on a refused
    connection (`code="switcher_down"`), or when the connection drops (`code="cut_off"`);
  - `.abort()` works from any thread; the stream then just ends.
- `SwitcherProbe(base_url=None, *, timeout=5.0)`:
  - `.inflight() -> list[dict] | None` gives the rows of the first `inflight` snapshot on
    `/api/events`, or `None` when unknown;
  - `.last_used(model_id) -> float | None` gives the epoch seconds of the newest
    `/api/metrics/activity?model=<id>&limit=1` row.
- `parse_go_time(value) -> float | None`.

- [ ] **Step 1: Write the failing tests**

```python
"""SwitcherChat and SwitcherProbe against a real local HTTP server."""

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
    routes[("POST", "/v1/chat/completions")] = lambda h: reply(
        h, 409, json.dumps({"error": {"code": "model_superseded", "message": "superseded by twin-27b"}}).encode())
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


def test_refused_connection_is_switcher_down():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(ChatFailed) as failed:
        list(SwitcherChat(f"http://127.0.0.1:{port}").stream({}, "s"))
    assert failed.value.code == "switcher_down"


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


def test_last_used_reads_the_newest_activity_row(server):
    routes, seen, url = server
    routes[("GET", "/api/metrics/activity")] = lambda h: reply(
        h, 200, json.dumps({"data": [{"timestamp": "2026-09-25T10:00:00.123456789Z", "model": "quasar-27b"}]}).encode())
    expected = dt.datetime(2026, 9, 25, 10, 0, 0, 123456, tzinfo=dt.timezone.utc).timestamp()
    assert SwitcherProbe(url).last_used("quasar-27b") == pytest.approx(expected)
    assert "model=quasar-27b" in seen[-1]["path"] and "limit=1" in seen[-1]["path"]
    routes[("GET", "/api/metrics/activity")] = lambda h: reply(h, 200, b'{"data": []}')
    assert SwitcherProbe(url).last_used("quasar-27b") is None


def test_parse_go_time():
    assert parse_go_time("2026-09-25T11:00:00+01:00") == parse_go_time("2026-09-25T10:00:00Z")
    assert parse_go_time("nonsense") is None and parse_go_time(None) is None
```

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_io.py -q`
Expected: collection error `No module named 'freetoken.daemon.settings.playground'`.

- [ ] **Step 3: Write the I/O part of `playground.py`**

```python
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
    """Go's RFC 3339 time (up to nanoseconds, Z or an offset) as epoch seconds."""
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
        parts = urllib.parse.urlsplit(self.base_url)
        conn = http.client.HTTPConnection(parts.hostname or "127.0.0.1", parts.port or 80, timeout=self.timeout)
        with self._lock:
            self._aborted = False
        try:
            try:
                conn.connect()
                with self._lock:
                    self._sock = conn.sock
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
            while True:
                try:
                    raw = response.readline()
                except (OSError, ValueError, http.client.HTTPException):
                    if self._aborted:
                        return
                    raise ChatFailed(0, "cut_off", "the connection closed before the answer finished") from None
                if not raw:
                    return
                yield raw.decode("utf-8", "replace")
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
                    envelope = json.loads(line[5:])
                    if not isinstance(envelope, dict) or envelope.get("type") != "inflight":
                        continue
                    event = json.loads(envelope.get("data") or "{}")
                    if isinstance(event, dict) and event.get("operation") == "snapshot":
                        return [row for row in event.get("requests") or [] if isinstance(row, dict)]
        except (OSError, ValueError):
            return None
        return None

    def last_used(self, model_id: str) -> float | None:
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
```

- [ ] **Step 4: Run them and see them pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_io.py -q`
Expected: `7 passed` in under 10 s.

- [ ] **Step 5: Controller commits**

```bash
git add python/freetoken/daemon/settings/playground.py tests/settings/test_playground_io.py
git commit -m "feat(settings): stream a test chat and read the switcher's in-flight list

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---
### Task 4: The runner's plan (`playground.py`, plan part) and the shared fakes

**Files:**
- Modify: `python/freetoken/daemon/settings/playground.py` (append: errors, helpers and the
  first half of `PlaygroundRunner`)
- Create: `tests/settings/playground_fakes.py`
- Test: `tests/settings/test_playground_runner.py` (plan tests)

**Interfaces:**
- `PlaygroundError(status, code, message, extra=None)` with `.status` and `.payload`
  (`{"code", "message", **extra}`).
- `PlaygroundRunner(panel, *, chat=None, probe=None, spawn=None, clock=time.monotonic, wall=time.time)`.
- `.plan(body) -> dict` with `{prompt, system, sides, warmup, putBack, before, held, steps,
  estimateS, warnings}`.
  - Each side: `{key, model, name, engine, preset, runPreset, settingsLabel, sampling,
    answer, reasoning, stats, loadMs, error}`.
  - Each step: `{kind, side, model, preset, label, guessS, state, ms, detail}`, where kind
    is one of `unload | settings | load | warmup | answer | restore`.
- `.snapshot() -> dict` gives the job, or `{"status": "idle"}`.

- [ ] **Step 1: Write the fakes**

`tests/settings/playground_fakes.py`:

```python
"""Fakes for the Test tab: a one-card switcher, a scripted chat, a probe and a clock."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.boot_parser import BootFile
from freetoken.daemon.settings.panel import PanelService
from freetoken.daemon.settings.playground import WARMUP_MESSAGES, PlaygroundRunner
from freetoken.daemon.settings.process_manager import ProcessManager
from freetoken.daemon.settings.profiles_manager import ProfilesManager
from freetoken.daemon.settings.registry import RegistryStore, find_model
from freetoken.daemon.settings.swap_config import SwapConfigWriter, render_config
from freetoken.daemon.settings.switcher import SwitcherError
from tests.settings.registry_fixtures import five
from tests.settings.test_panel_routes import FakeEstimates, FakeSwitcher, checker

GIB = 1024 ** 3


def presets_doc() -> dict:
    doc = five()
    find_model(doc, "quasar-27b")["presets"] = {
        "Fast": {"kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 3},
        "Same": {"kv-dtype": "int8", "spec": "dflash2", "draft-tokens": 7},  # equals the saved settings
    }
    find_model(doc, "fable-27b")["presets"] = {"Three": {"kv-dtype": "fp8", "spec": "mtp", "draft-tokens": 3}}
    find_model(doc, "qwen3.8-flash")["presets"] = {"Short": {"KVCacheTokens": 131072}}
    return doc


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def wall(self) -> float:
        return 1_800_000_000.0 + self.now


class CardSwitcher(FakeSwitcher):
    """One graphics card: a load puts every other model away. on_load runs when a load starts
    (it may change states, call runner.stop() or raise); after_load runs once it finished."""

    def __init__(self, cfg: Path, clock: Clock) -> None:
        super().__init__(cfg)
        self.clock, self.load_s, self.stale = clock, {}, False
        self.on_load = self.after_load = None
        self.loading, self.cancelled = None, set()
        self.files_at_load: dict[str, list[str]] = {}

    def config_hash(self):
        return "stale" if self.stale else super().config_hash()

    def unload(self, model_id):
        self.calls.append(("unload", model_id))
        if not self.unload_ok:
            return False
        if self.loading == model_id:
            self.cancelled.add(model_id)  # P1: an unload cancels a half-finished swap
        self.states.pop(model_id, None)
        return True

    def load(self, model_id, *, timeout=900.0):
        self.calls.append(("load", model_id))
        self.loading = model_id
        try:
            self.files_at_load.setdefault(model_id, []).append(self.cfg.read_text())
            if self.on_load is not None:
                self.on_load(model_id)
            if model_id in self.cancelled:
                self.cancelled.discard(model_id)
                raise SwitcherError(502, "load_failed", "the load was cancelled")
            if self.load_error:
                raise self.load_error
            self.clock.advance(self.load_s.get(model_id, 20))
            self.states = {model_id: "ready"}
        finally:
            self.loading = None
        if self.after_load is not None:
            self.after_load(model_id)


def sse(obj) -> str:
    return "data: " + json.dumps(obj) + "\n"


def answer_script(words=("Hello", " there", " friend"), prompt=40, timings=None):
    """(seconds before the line, line): the first word after 0.2 s, then one word each 0.05 s."""
    script = [(0.2, sse({"model": None, "choices": [{"index": 0, "delta": {"content": words[0]}}]}))]
    script += [(0.05, sse({"choices": [{"index": 0, "delta": {"content": word}}]})) for word in words[1:]]
    last = {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": len(words)}}
    if timings:
        last["timings"] = timings
    script += [(0.0, sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})),
               (0.0, sse(last)), (0.0, "data: [DONE]\n")]
    return script


class FakeChat:
    def __init__(self, clock: Clock) -> None:
        self.clock, self.scripts, self.errors = clock, {}, {}
        self.bodies, self.sessions = [], []
        self.on_stream, self.aborted = None, False

    def stream(self, body, session):
        self.bodies.append(body)
        self.sessions.append(session)
        self.aborted = False
        warmup = body["messages"] == WARMUP_MESSAGES
        if self.on_stream is not None:
            self.on_stream(body)
        if not warmup and body["model"] in self.errors:
            raise self.errors[body["model"]]
        for delay, line in answer_script(("OK",)) if warmup else self.scripts.get(body["model"], answer_script()):
            if self.aborted:
                return
            self.clock.advance(delay)
            yield line

    def abort(self):
        self.aborted = True

    def answers(self):
        return [body for body in self.bodies if body["messages"] != WARMUP_MESSAGES]


class FakeProbe:
    def __init__(self) -> None:
        self.rows, self.used, self.unknown = [], {}, False

    def inflight(self):
        return None if self.unknown else list(self.rows)

    def last_used(self, model_id):
        return self.used.get(model_id)


def make(tmp_path, monkeypatch, loaded=None, doc=None, with_routes=False, static_path=None):
    """Registry with presets, a generated switcher file, one card, and inline threads."""
    monkeypatch.setenv("HOME", "/home/jay")
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -KVDtype 'fp8' `\n    -Port 2020\n", encoding="utf-8")
    cfg = tmp_path / "llama-swap" / "config.yaml"
    cfg.parent.mkdir()
    binary = tmp_path / "llama-swap-bin"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    clock = Clock()
    writer = SwapConfigWriter(cfg, binary=binary, runner=checker())
    switcher = CardSwitcher(cfg, clock)
    profiles = ProfilesManager(tmp_path / "boot-profiles.json", boot_file=boot)
    store = RegistryStore(tmp_path / "freetoken" / "registry.json")
    store.save(doc or presets_doc(), expected_revision=None)
    writer.write(render_config(store.load()[0], {}))
    switcher.states = dict(loaded or {})
    chat, probe = FakeChat(clock), FakeProbe()

    def build():
        """A fresh PanelService and runner over the same files: what a helper restart gives."""
        service = PanelService(
            store=store, writer=writer, switcher=switcher, profiles=profiles,
            boot_file=lambda: BootFile(boot), default_boot=lambda: boot, estimate_service=FakeEstimates(),
            card_probe=lambda: {"totalBytes": 32 * GIB, "usedBytes": 2 * GIB}, windows_free_probe=lambda: 40 * GIB,
            artifact_size=lambda path: 19_782_132_224, spawn=lambda fn, *args: fn(*args),
            clock=clock, sleep=clock.advance)
        runner = PlaygroundRunner(service, chat=chat, probe=probe, spawn=lambda fn, *args: fn(*args),
                                  clock=clock, wall=clock.wall)
        return service, runner

    service, runner = build()
    proc = ProcessManager(boot_file=boot, stop_script=tmp_path / "stop.ps1", log_path=tmp_path / "server.log",
                          lock_path=tmp_path / "gpu.lock", runner=lambda *a, **k: None,
                          readiness=lambda: {"state": "unreachable"}, sleep=lambda _: None, poll_interval=0)
    extra = {"playground": runner} if with_routes else {}
    app = create_app(boot_file=boot, process_manager=proc, profiles=profiles, log_path=tmp_path / "server.log",
                     static_path=static_path or tmp_path / "missing.html", panel=service, **extra)
    return SimpleNamespace(service=service, runner=runner, store=store, writer=writer, switcher=switcher,
                           chat=chat, probe=probe, clock=clock, cfg=cfg, client=TestClient(app), build=build)
```

- [ ] **Step 2: Write the failing plan tests**

`tests/settings/test_playground_runner.py`:

```python
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
```

- [ ] **Step 3: Run them and see them fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_runner.py -q`
Expected: `ImportError: cannot import name 'PlaygroundError'`. `make()` also needs
`create_app(playground=...)` only when `with_routes=True`, which no Task 4 test uses.

- [ ] **Step 4: Append the plan part of the runner to `playground.py`**

Extend the imports at the top:

```python
import copy
import uuid

from .panel import PanelError
from .playground_speed import AnswerTracker, answer_stats
from .registry import RegistryCorrupt, RegistryError, RegistryMissing, find_model
from .swap_config import SwitcherRefused
from .switcher import DEFAULT_URL, LOADED_STATES, SwitcherError, is_down
```

Then append:

```python
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
```

- [ ] **Step 5: Run the plan tests and see them pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_runner.py -q`
Expected: `17 passed` (8 tests, plus 9 parametrised refusals).

- [ ] **Step 6: Controller commits**

```bash
git add python/freetoken/daemon/settings/playground.py tests/settings/playground_fakes.py tests/settings/test_playground_runner.py
git commit -m "feat(settings): plan a test: steps, time guesses, busy and recent-use checks

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Run, put back, stop and recover (`playground.py`, run part)

**Files:**
- Modify: `python/freetoken/daemon/settings/playground.py` (append methods to `PlaygroundRunner`)
- Test: `tests/settings/test_playground_runner.py` (append)

**Interfaces:**
- `.start(body) -> dict` plans again. It raises 409 `changed` (with `plan`) when
  `expectBefore` differs, and 409 `confirm` (with `plan`) when there are warnings and no
  `confirm`. Otherwise it calls `panel.begin_test()`, spawns `_run` and returns the snapshot.
- `.stop() -> dict` works only while the status is `running`. It closes the stream and
  unloads a model mid-load (on a spawned thread).
- `.recover() -> str` returns one of `none | gone | unloaded | left | unknown`.
- The job's `status` is one of `running | stopping | restoring | done | failed | stopped |
  yielded`. The job also has `message` and `restore` (the put-back words).

- [ ] **Step 1: Append the failing tests**

```python
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
```

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_runner.py -q`
Expected: the Task 5 tests fail with `AttributeError: 'PlaygroundRunner' object has no attribute 'start'`
(and `recover`). The Task 4 tests still pass.

- [ ] **Step 3: Append the run part to `PlaygroundRunner`**

```python
    # ---- start and stop ----
    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            plan = self.plan(body)
            if "expectBefore" in body and body.get("expectBefore") != plan["before"]:
                raise PlaygroundError(409, "changed", "Something changed since the plan was shown. "
                                      "Check the new plan and press Start again.", {"plan": plan})
            if plan["warnings"] and not body.get("confirm"):
                raise PlaygroundError(409, "confirm", "Check the plan and press Start.", {"plan": plan})
            self.panel.begin_test()
            self._stop.clear()
            self._session = SESSION_PREFIX + uuid.uuid4().hex[:10]
            self._ours = None
            job = {**plan, "id": self._session[len(SESSION_PREFIX):], "status": "running",
                   "startedAt": _now_iso(), "finishedAt": None, "message": "", "restore": ""}
            self.job = job
        self._spawn(self._run, job)
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        """Ends the current step: closes the answer stream, or unloads the model being loaded
        (P1 cancels a half-finished swap on unload), so a 2.5 min FreeToken load never makes
        Stop wait. Put-back still runs."""
        with self._lock:
            if self.job is None or self.job["status"] != "running":
                return self.snapshot()
            self._stop.set()
            self.job["status"] = "stopping"
            loading = self._loading
        self.chat.abort()
        if loading is not None:
            self._spawn(self.switcher.unload, loading)
        return self.snapshot()

    # ---- the walk ----
    def _mark(self, step: dict[str, Any], state: str, detail: str | None = None) -> None:
        with self._lock:
            now = self._clock()
            if state == "running":
                step["_t0"] = now
            elif step.get("_t0") is not None:
                step["ms"] = max(0, round((now - step["_t0"]) * 1000))
            step["state"] = state
            if detail is not None:
                step["detail"] = detail

    def _run(self, job: dict[str, Any]) -> None:
        halt: _Halt | None = None
        work = [step for step in job["steps"] if step["kind"] != "restore"]
        try:
            try:
                for step in work:
                    if self._stop.is_set():
                        raise _Halt("stopped", STOPPED)
                    self._mark(step, "running")
                    try:
                        self._do(job, step)
                    except _Halt as exc:
                        self._mark(step, "failed", exc.message)
                        raise
                    except Exception as exc:  # noqa: BLE001 - put-back must still run
                        self._mark(step, "failed", str(exc))
                        raise _Halt("failed", f"Something went wrong: {exc}") from exc
                    self._mark(step, "done")
            except _Halt as exc:
                halt = exc
            for step in work:
                if step["state"] == "waiting":
                    self._mark(step, "skipped")
            restore = job["steps"][-1]
            with self._lock:
                job["status"] = "restoring"
            self._mark(restore, "running")
            ok, words = self._restore(job)
            self._mark(restore, "done" if ok else "failed", words)
            with self._lock:
                job["restore"] = words
                job["status"] = "done" if halt is None else halt.kind
                job["message"] = "" if halt is None else halt.message
        finally:
            with self._lock:
                if job["status"] in ACTIVE:
                    job["status"], job["message"] = "failed", job["message"] or "The test ended unexpectedly."
                job["finishedAt"] = _now_iso()
            self.panel.end_test()

    def _do(self, job: dict[str, Any], step: dict[str, Any]) -> None:
        kind, model = step["kind"], step["model"]
        side = next(s for s in job["sides"] if s["key"] == step["side"])
        if kind == "unload":
            busy = self._others_using(model)
            if busy is None:
                raise _Halt("failed", f"Couldn't check whether {self._name(model)} is busy, so the test stopped.")
            if busy:
                raise _Halt("yielded", f"{self._name(model)} started answering something for another app, "
                                       "so the test stopped to leave it alone.")
            if not self.switcher.unload(model):
                raise _Halt("failed", f"Couldn't put {self._name(model)} away.")
            if self._ours == model:
                self._ours = None
        elif kind == "settings":
            try:
                text = self.panel.set_test_settings(model if step["preset"] else None, step["preset"])
            except (PanelError, SwitcherRefused, RegistryError, KeyError, OSError) as exc:
                raise _Halt("failed", f"Couldn't write the test settings: {exc}") from None
            if not self.panel.wait_for_switcher(text):
                raise _Halt("failed", "The switcher didn't pick up the test settings, so nothing was loaded.")
        elif kind == "load":
            self._load(side, model)
        else:
            self._answer(job, side, warmup=kind == "warmup")

    def _load(self, side: dict[str, Any], model: str) -> None:
        with self._lock:
            self._loading = model
        started = self._clock()
        try:
            self.switcher.load(model)
        except SwitcherError as exc:
            if self._stop.is_set():
                raise _Halt("stopped", STOPPED) from None
            if exc.code == "model_superseded":
                raise _Halt("yielded", YIELDED) from None
            raise _Halt("failed", f"Loading {side['name']} failed: {exc.message or exc.code}") from None
        except OSError:
            raise _Halt("failed", "The model switcher is not running.") from None
        finally:
            with self._lock:
                self._loading = None
        self._ours = model
        with self._lock:
            side["loadMs"] = (side["loadMs"] or 0) + max(0, round((self._clock() - started) * 1000))
        if self._stop.is_set():
            raise _Halt("stopped", STOPPED)

    def _answer(self, job: dict[str, Any], side: dict[str, Any], *, warmup: bool) -> None:
        running = self.switcher.running() or {}
        if running.get(side["model"]) != "ready":
            raise _Halt("yielded", f"{side['name']} is no longer loaded; another app may have asked for a "
                                   "different model. The test stopped.")
        if warmup:
            body: dict[str, Any] = {"model": side["model"], "messages": WARMUP_MESSAGES, "max_tokens": WARMUP_TOKENS}
        else:
            messages = [{"role": "system", "content": job["system"]}] if job["system"] else []
            messages.append({"role": "user", "content": job["prompt"]})
            body = {"model": side["model"], "messages": messages, **side["sampling"]}
        body.update({"stream": True, "stream_options": {"include_usage": True}})
        tracker = AnswerTracker(started=self._clock())
        try:
            for line in self.chat.stream(body, self._session):
                tracker.feed_line(line, self._clock())
                if not warmup:
                    self._publish(side, tracker)
                if self._stop.is_set():
                    break
        except ChatFailed as exc:
            if self._stop.is_set():
                raise _Halt("stopped", STOPPED) from None
            if exc.code == "model_superseded":
                raise _Halt("yielded", YIELDED) from None
            raise _Halt("failed", f"{side['name']} could not answer: {exc.message}") from None
        finally:
            tracker.ended = self._clock()
            self._own_last[side["model"]] = self._wall()
        stopped = self._stop.is_set()
        if not warmup:
            self._publish(side, tracker, final=True, cancelled=stopped)
            with self._lock:
                names = self._aliases.get(side["model"], {side["model"]})
                if tracker.served_model and tracker.served_model not in names:
                    side["error"] = f"The answer came from {tracker.served_model}, not {side['name']}."
                elif not stopped and not tracker.done and tracker.finish_reason is None:
                    side["error"] = "The answer ended without a finish signal."
        if stopped:
            raise _Halt("stopped", STOPPED)

    def _publish(self, side: dict[str, Any], tracker: AnswerTracker, *, final: bool = False,
                 cancelled: bool = False) -> None:
        stats = answer_stats(tracker, cancelled=cancelled and final)
        with self._lock:
            side["answer"] = tracker.answer_text[:MAX_TEXT_CHARS]
            side["reasoning"] = tracker.reasoning_text[:MAX_TEXT_CHARS]
            side["stats"] = stats

    # ---- put back ----
    def _restore(self, job: dict[str, Any]) -> tuple[bool, str]:
        """Always runs last. Puts an idle test model away, clears the overlay, and loads the
        model from before on its saved settings, unless another app's model is on the card,
        Jay said not to, or the switcher has not picked up the saved file."""
        before, put_back = job["before"], job["putBack"]
        words: list[str] = []
        try:
            test = self.panel.test_settings
            running = self.switcher.running()
            if running is None or is_down(running):
                self.panel.set_test_settings(None, None)
                return False, "Can't tell what's loaded right now, so nothing was loaded again. Check the Models tab."
            loaded = sorted(model for model, state in running.items() if state in LOADED_STATES)
            left = None
            for model in list(loaded):
                on_test = bool(test) and model == test["model"]
                if not (on_test or model == self._ours):
                    continue  # another app's model
                if not on_test and (model == before or not put_back):
                    continue  # our setup on saved settings, and it may stay
                if self._others_using(model) is False and self.switcher.unload(model):
                    loaded.remove(model)
                    self._ours = None if self._ours == model else self._ours
                elif on_test:
                    left = model
                    self.panel.note_test_leftover(model, test["preset"])
                    words.append(f"{self._name(model)} is still on test settings because an app is using it. "
                                 "It goes back to its saved settings at its next load.")
            text = self.panel.set_test_settings(None, None)
            if not self.panel.wait_for_switcher(text):
                words.append("The switcher hasn't picked up the saved settings yet, so nothing was loaded again. "
                             "Check the Models tab.")
                return False, " ".join(words)
            if before is None:
                words.append("Nothing was loaded before the test.")
            elif before == left:
                pass
            elif before in loaded:
                words.append(f"{self._name(before)} is loaded on its saved settings, as before.")
            elif not put_back:
                words.append(f"{self._name(before)} was not loaded again, as asked.")
            elif loaded:
                words.append(f"{self._name(before)} was not loaded again, because an app is using {self._name(loaded[0])}.")
            else:
                started = self._clock()
                self.switcher.load(before)
                took = max(0, round((self._clock() - started) * 1000))
                words.append(f"{self._name(before)} is loaded again on its saved settings ({_seconds(took)}).")
            return True, " ".join(words)
        except Exception as exc:  # noqa: BLE001 - never leave the overlay set
            try:
                self.panel.set_test_settings(None, None)
            except Exception:  # noqa: BLE001
                pass
            words.append(f"Putting things back didn't finish ({exc}). Check the Models tab.")
            return False, " ".join(words)

    # ---- helper start ----
    def recover(self) -> str:
        """At helper start: a test running when the helper stopped may have left its model on
        test settings. Put it away when nobody uses it; otherwise leave it and say so on the
        Right-now strip. Nothing on disk needs restoring: the registry was never written, and
        sync_config() rewrites the switcher file right after this (holding a busy NInfer
        model's running entry, as for any loaded model)."""
        marker = self.panel.read_test_marker()
        if marker is None:
            return "none"
        try:
            self._doc()
        except PlaygroundError:
            pass
        model, preset = str(marker.get("model") or ""), marker.get("preset")
        running = self.switcher.running()
        result = "gone"
        if running is None:
            self.panel.note_test_leftover(model, preset)
            result = "unknown"
        elif not is_down(running) and running.get(model) in LOADED_STATES:
            if self._others_using(model) is False and self.switcher.unload(model):
                result = "unloaded"
            else:
                self.panel.note_test_leftover(model, preset)
                result = "left"
        self.panel.clear_test_marker()
        return result
```

- [ ] **Step 4: Run the runner tests and see them pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_runner.py -q`
Expected: `31 passed`.

- [ ] **Step 5: Run the whole settings suite (nothing else broke)**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings tests/daemon -q`
Expected: all pass, and the only skips are the ones that skip on the devbox today.

- [ ] **Step 6: Controller commits**

```bash
git add python/freetoken/daemon/settings/playground.py tests/settings/test_playground_runner.py
git commit -m "feat(settings): run a test, always put things back, stop and recover

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---
### Task 6: Routes, app wiring and start-up order

**Files:**
- Modify: `python/freetoken/daemon/settings/playground.py` (append `options()` and
  `create_playground_router`)
- Modify: `python/freetoken/daemon/settings/app.py`, `python/freetoken/daemon/settings/server.py`
- Test: `tests/settings/test_playground_routes.py`

**Interfaces:**
- `GET /api/playground/options` returns `{models:[{id, name, engine, engineLabel, presets,
  activePreset, state}], switcherUp, loadGuessS, defaults:{maxTokens, maxAnswerTokens}}`.
- `POST /api/playground/plan` returns the plan, or an error payload `{code, message}`.
- `POST /api/playground/runs` returns the job snapshot, or 409 `changed`/`confirm` (with
  `plan`) or `test_running`.
- `GET /api/playground/runs/current` returns the job, or `{"status": "idle"}`.
- `POST /api/playground/runs/current/stop` returns the job.
- `GET /playground.js` serves the static script.
- `create_app(..., playground: PlaygroundRunner | None = None)`.
- `server.start_panel(app)`: recover, then sync, then the hold watcher.

- [ ] **Step 1: Write the failing tests**

```python
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
    assert body["loadGuessS"] == {"freetoken": 150, "ninfer": 20}
    assert body["defaults"] == {"maxTokens": 512, "maxAnswerTokens": 4096} and body["switcherUp"] is True


def test_plan_and_refusals_are_plain_json(tmp_path, monkeypatch):
    env = routes(tmp_path, monkeypatch, loaded={"quasar-27b": "ready"})
    ok = env.client.post("/api/playground/plan", json={"prompt": "Hi", "sides": [{"model": "fable-27b"}]})
    assert ok.status_code == 200 and ok.json()["before"] == "quasar-27b"
    bad = env.client.post("/api/playground/plan", json={"prompt": "", "sides": [{"model": "fable-27b"}]})
    assert bad.status_code == 422 and bad.json() == {"code": "prompt", "message": "Type a prompt first."}


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


def test_a_failing_recover_never_stops_the_helper():
    calls = []

    def boom():
        raise OSError("switcher down")

    app = SimpleNamespace(state=SimpleNamespace(
        playground=SimpleNamespace(recover=boom),
        panel=SimpleNamespace(sync_config=lambda: calls.append("sync"),
                              start_hold_watcher=lambda: calls.append("watch"))))
    start_panel(app)
    assert calls == ["sync", "watch"]
```

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_routes.py -q`
Expected: `ImportError: cannot import name 'start_panel'`.

- [ ] **Step 3: Implement**

Append to `playground.py`. Add the imports `from fastapi import APIRouter, Body`,
`from fastapi.responses import JSONResponse` and
`from starlette.concurrency import run_in_threadpool`.

```python
    # (method of PlaygroundRunner)
    def options(self) -> dict[str, Any]:
        try:
            listing = self.panel.models()
        except RegistryMissing:
            raise PlaygroundError(409, "registry_missing",
                                  "The control panel has no model list yet. Copy today's settings first.") from None
        except RegistryCorrupt as exc:
            raise PlaygroundError(409, "registry_corrupt", exc.message) from None
        keys = ("id", "name", "engine", "engineLabel", "presets", "activePreset", "state")
        return {"models": [{key: row[key] for key in keys} for row in listing["models"]],
                "switcherUp": listing["switcherUp"], "loadGuessS": dict(LOAD_GUESS_S),
                "defaults": {"maxTokens": DEFAULT_ANSWER_TOKENS, "maxAnswerTokens": MAX_ANSWER_TOKENS}}


def create_playground_router(runner: PlaygroundRunner) -> APIRouter:
    router = APIRouter(prefix="/api/playground")

    async def call(fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return await run_in_threadpool(fn, *args)
        except (PlaygroundError, PanelError) as exc:
            return JSONResponse(status_code=exc.status, content=exc.payload)

    @router.get("/options")
    async def options():
        return await call(runner.options)

    @router.post("/plan")
    async def plan(body: dict[str, Any] = Body(default_factory=dict)):
        return await call(runner.plan, body)

    @router.post("/runs")
    async def start(body: dict[str, Any] = Body(default_factory=dict)):
        return await call(runner.start, body)

    @router.get("/runs/current")
    async def current():
        return await call(runner.snapshot)

    @router.post("/runs/current/stop")
    async def stop():
        return await call(runner.stop)

    return router
```

`app.py`:
- import `from .playground import PlaygroundRunner, create_playground_router`;
- add the parameter `playground: PlaygroundRunner | None = None` after `panel`;
- after `app.include_router(create_panel_router(panel))` add:

```python
    if playground is None:
        playground = PlaygroundRunner(panel)
    app.state.playground = playground
    app.include_router(create_playground_router(playground))
```

- next to the `/panel.js` route add:

```python
    playground_script = static.with_name("playground.js")

    @app.get("/playground.js")
    async def playground_js():
        if playground_script.is_file():
            return FileResponse(playground_script, media_type="text/javascript")
        return PlainTextResponse("", status_code=404)
```

`server.py`: add the function below and replace the three panel lines in `main()`
(`panel = app.state.panel` / `panel.sync_config()` / `panel.start_hold_watcher()`) with
`start_panel(app)`.

```python
def start_panel(app) -> None:
    """Control panel start-up. First put away a model an interrupted Test-tab test left on test
    settings (playground.recover), then make the switcher file match the registry, then keep
    releasing "next time" holds once their model unloads (panel.py). A failing recover must
    never stop the helper: the Right-now strip still shows what is loaded."""
    try:
        app.state.playground.recover()
    except Exception:  # noqa: BLE001
        pass
    app.state.panel.sync_config()
    app.state.panel.start_hold_watcher()
```

- [ ] **Step 4: Run them and the import-safety tests**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_routes.py tests/settings/test_settings_import_safety.py tests/daemon -q`
Expected: all pass. `playground.py` imports no torch.

- [ ] **Step 5: Controller commits**

```bash
git add python/freetoken/daemon/settings/playground.py python/freetoken/daemon/settings/app.py python/freetoken/daemon/settings/server.py tests/settings/test_playground_routes.py
git commit -m "feat(settings): Test tab routes; recover an interrupted test at helper start

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: The page's helpers (`static/playground.js`)

**Files:**
- Create: `python/freetoken/daemon/settings/static/playground.js`
- Test: `tests/settings/test_playground_page.py` (node part)

**Interfaces (pure, exported for node):**
- `fmtMs`, `fmtGuess`, `fmtRate`, `guessWords`, `stopWords`, `tokensWords`;
- `metricText(side, key)`, `betterSide(a, b, key)`, `speedRows(side, other)`;
- `stepLine`, `planLines`, `statusWords`;
- `historyRecord`, `historyAdd`, `historyMarkdown`, `sideTitle`;
- `loadHistory(storage)`, `saveHistory(storage, list)`;
- `PG_HISTORY_KEY`, `PG_HISTORY_MAX`.

Browser functions (not unit-tested; Task 10 checks them live): `pgOpen`, `pgRun`, `pgStart`,
`pgStop`, `pgPoll`, `pgRenderJob`, `pgRenderHistory`, `pgCopy`, `pgExport`, `pgClear`.

- [ ] **Step 1: Write the failing node tests**

```python
"""The Test tab page: plain-words helpers (node) and the page contract."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
STATIC = REPO / "python" / "freetoken" / "daemon" / "settings" / "static"
PG_JS = STATIC / "playground.js"
PANEL_JS = STATIC / "panel.js"
PAGE = STATIC / "index.html"


def _node(script: str, module: Path = PG_JS) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to run the page's JavaScript")
    prelude = f"const p = require({json.dumps(str(module))}); const assert = require('node:assert/strict');\n"
    run = subprocess.run([node, "-e", prelude + script], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr


def test_times_and_speeds_in_plain_words():
    _node(r"""
assert.equal(p.fmtMs(420), '420 ms');
assert.equal(p.fmtMs(12840), '12.8 s');
assert.equal(p.fmtMs(119999), '2 min 0 s');
assert.equal(p.fmtMs(151000), '2 min 31 s');
assert.equal(p.fmtMs(null), '—');
assert.equal(p.fmtGuess(20), '20 s');
assert.equal(p.fmtGuess(150), '2.5 min');
assert.equal(p.fmtRate({writeTps: 41.26, writeSource: 'engine'}), '41.3 tokens a second');
assert.equal(p.fmtRate({writeTps: 9.94, writeSource: 'measured'}), '~9.9 tokens a second');
assert.equal(p.fmtRate({writeTps: null}), '—');
assert.equal(p.guessWords({guesses: {proposed: 514, kept: 365, keptPct: 71}}), '71% (365 of 514 guesses)');
assert.equal(p.guessWords({guesses: null}), 'not reported by this engine');
assert.equal(p.stopWords('length'), 'hit the longest-answer limit');
assert.equal(p.stopWords('cancelled'), 'stopped');
assert.equal(p.tokensWords(812, false, 640), '812 tokens (640 reused)');
assert.equal(p.tokensWords(12, true), '~12 tokens');
""")


def test_better_tag_needs_a_real_difference():
    _node(r"""
const a = {key: 'A', loadMs: 20000, stats: {firstWordMs: 400, writeTps: 40, writeSource: 'engine', totalMs: 12000,
  guesses: {keptPct: 71, kept: 1, proposed: 1}}};
const b = {key: 'B', loadMs: null, stats: {firstWordMs: 410, writeTps: 30, writeSource: 'engine', totalMs: 16000, guesses: null}};
assert.equal(p.betterSide(a, b, 'writeTps'), 'A');
assert.equal(p.betterSide(a, b, 'totalMs'), 'A');
assert.equal(p.betterSide(a, b, 'firstWordMs'), null);   // 2.4% apart: no tag
assert.equal(p.betterSide(a, b, 'guessPct'), null);      // B did not report
assert.equal(p.betterSide(a, b, 'loadMs'), null);        // loading is never judged
const rowsB = p.speedRows(b, a);
assert.equal(rowsB.find((r) => r.key === 'loadMs').text, 'already loaded');
assert.equal(rowsB.find((r) => r.key === 'writeTps').better, false);
assert.equal(p.speedRows(a, b).find((r) => r.key === 'writeTps').better, true);
assert.deepEqual(p.speedRows(a, b).map((r) => r.label), ['Loading (not counted)', 'First word after', 'Writing speed',
  'Whole answer', 'Prompt size', 'Answer size', 'Guesses kept', 'Stopped because']);
""")


def test_steps_plan_and_status_lines():
    _node(r"""
assert.equal(p.stepLine({state: 'running', label: 'Load Fable', elapsedMs: 4200}), '◐ Load Fable · 4.2 s');
assert.equal(p.stepLine({state: 'done', label: 'Put away QUASAR', ms: 5100}), '✓ Put away QUASAR · 5.1 s');
assert.equal(p.stepLine({state: 'waiting', label: 'Load Fable', guessS: 20}), '○ Load Fable · about 20 s');
assert.equal(p.stepLine({state: 'skipped', label: 'Answer with setup B', guessS: 30}), '– Answer with setup B');
assert.equal(p.stepLine({state: 'failed', label: 'Load Fable', ms: 900, detail: 'Loading Fable failed: no memory'}),
  '✗ Load Fable · 900 ms — Loading Fable failed: no memory');
const lines = p.planLines({steps: [{label: 'Put away QUASAR', guessS: 5}, {label: 'Load Qwen', guessS: 150}],
  estimateS: 155, warnings: ['QUASAR was used 40 seconds ago. The test will put it away.']});
assert.deepEqual(lines.steps, ['Put away QUASAR · about 5 s', 'Load Qwen · about 2.5 min']);
assert.equal(lines.total, 'About 2.5 min in all.');
assert.equal(lines.warnings.length, 1);
assert.equal(p.statusWords({status: 'yielded', message: 'Another app asked for a different model, so the test stopped to let it through.'}),
  'Stopped to let another app through. Another app asked for a different model, so the test stopped to let it through.');
assert.equal(p.statusWords({status: 'stopped', message: 'Stopped.'}), 'Stopped.');
assert.equal(p.statusWords({status: 'done', message: ''}), 'Done.');
assert.equal(p.statusWords(null), '');
""")


def test_history_keeps_twenty_newest_and_survives_a_full_store():
    _node(r"""
let list = [];
for (let i = 0; i < 25; i++) list = p.historyAdd(list, {id: 'j' + i});
assert.equal(list.length, 20); assert.equal(list[0].id, 'j24'); assert.equal(list[19].id, 'j5');
assert.equal(p.historyAdd(list, {id: 'j24', x: 1}).filter((r) => r.id === 'j24').length, 1);
const store = { data: {}, setItem(k, v) { if (JSON.parse(v).length > 2) throw new Error('QuotaExceededError'); this.data[k] = v; },
  getItem(k) { return this.data[k] ?? null; }, removeItem(k) { delete this.data[k]; } };
assert.equal(p.saveHistory(store, list.slice(0, 5)).length, 2);
assert.deepEqual(p.loadHistory(store).map((r) => r.id), ['j24', 'j23']);
assert.deepEqual(p.saveHistory(store, []), []);
assert.equal(store.data[p.PG_HISTORY_KEY], undefined);
assert.deepEqual(p.loadHistory({ getItem() { return '{broken'; } }), []);
assert.deepEqual(p.loadHistory({ getItem() { throw new Error('SecurityError'); } }), []);
const rec = p.historyRecord({id: 'x', startedAt: '2026-09-25T10:00:00Z', prompt: 'Hi', status: 'done', steps: [{}],
  sides: [{key: 'A', model: 'm', name: 'M', answer: 'a'.repeat(30000), stats: null}]});
assert.equal(rec.sides[0].answer.length, 20000); assert.equal(rec.steps, undefined); assert.equal(p.PG_HISTORY_MAX, 20);
""")


def test_markdown_copy_has_prompt_table_and_answers():
    _node(r"""
const md = p.historyMarkdown({at: '2026-09-25T10:00:00Z', prompt: 'Line one\nLine two',
  restore: 'QUASAR is loaded again on its saved settings (21.0 s).', sides: [
  {key: 'A', name: 'Fable', preset: 'Fast', settingsLabel: 'preset “Fast”', loadMs: 20000, answer: 'Answer A',
   stats: {firstWordMs: 400, writeTps: 41.3, writeSource: 'engine', totalMs: 12000, promptTokens: 30, completionTokens: 500,
           guesses: {proposed: 514, kept: 365, keptPct: 71}, finishReason: 'stop'}},
  {key: 'B', name: 'Fable', preset: null, settingsLabel: 'Saved settings', loadMs: null, stats: null, answer: ''}]});
assert.match(md, /^## Test 2026-09-25T10:00:00Z\n\n> Line one\n> Line two\n/);
assert.match(md, /\| \| A · Fable · preset “Fast” \| B · Fable · Saved settings \|/);
assert.match(md, /\| Writing speed \| 41\.3 tokens a second \| — \|/);
assert.match(md, /\| Guesses kept \| 71% \(365 of 514 guesses\) \| not reported by this engine \|/);
assert.match(md, /### A · Fable · preset “Fast”\n\nAnswer A\n/);
assert.match(md, /### B · Fable · Saved settings\n\n\(no answer\)\n/);
assert.match(md, /_QUASAR is loaded again on its saved settings \(21\.0 s\)\._\n$/);
assert.doesNotMatch(md, /\n\n\n/);
""")
```

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_page.py -q`
Expected: 5 failures with `Cannot find module '.../static/playground.js'`.

- [ ] **Step 3: Write `static/playground.js`**

```javascript
'use strict';
/* The Test tab (docs/superpowers/specs/2026-09-25-playground-design.md). The helper runs the
   test (switch, answer, time, put back); this page plans it, starts it, polls it every 0.5 s
   and keeps the last 20 tests in this browser. The helpers above the browser section are pure
   and exported for node. $, json, setNotice and askConfirm come from index.html and panel.js. */

const PG_POLL_MS = 500;
const PG_HISTORY_KEY = 'ft-test-history-v1';
const PG_HISTORY_MAX = 20;
const PG_ANSWER_KEEP = 20000;
const PG_ACTIVE = ['running', 'stopping', 'restoring'];
const pg = { options: null, job: null, timer: null, plan: null, history: [], shown: null, wired: false };

function pgEsc(value) { return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function fmtMs(ms) {
  const n = Number(ms);
  if (ms == null || ms === '' || !Number.isFinite(n)) return '—';
  if (n < 1000) return `${Math.round(n)} ms`;
  if (n < 60000) return `${(n / 1000).toFixed(1)} s`;
  const total = Math.round(n / 1000);
  return `${Math.floor(total / 60)} min ${total % 60} s`;
}
function fmtGuess(seconds) {
  const s = Number(seconds) || 0;
  return s < 60 ? `${Math.round(s)} s` : `${Math.round((s / 60) * 2) / 2} min`;
}
function fmtRate(stats) {
  if (!stats || stats.writeTps == null) return '—';
  const value = Number(stats.writeTps).toFixed(1);
  return stats.writeSource === 'engine' ? `${value} tokens a second` : `~${value} tokens a second`;
}
function guessWords(stats) {
  const g = stats && stats.guesses;
  return g ? `${g.keptPct}% (${g.kept} of ${g.proposed} guesses)` : 'not reported by this engine';
}
const PG_STOP_WORDS = { stop: 'finished', length: 'hit the longest-answer limit', tool_calls: 'wanted to use a tool', cancelled: 'stopped' };
function stopWords(reason) { return reason ? (PG_STOP_WORDS[reason] || String(reason)) : '—'; }
function tokensWords(n, approx, reused) {
  if (n == null) return '—';
  return `${approx ? '~' : ''}${Number(n).toLocaleString('en-GB')} tokens${reused ? ` (${Number(reused).toLocaleString('en-GB')} reused)` : ''}`;
}

// [key, label, lower is better (true), higher is better (false), never judged (null)]
const PG_METRICS = [
  ['loadMs', 'Loading (not counted)', null],
  ['firstWordMs', 'First word after', true],
  ['writeTps', 'Writing speed', false],
  ['totalMs', 'Whole answer', true],
  ['promptTokens', 'Prompt size', null],
  ['completionTokens', 'Answer size', null],
  ['guessPct', 'Guesses kept', false],
  ['finishReason', 'Stopped because', null],
];
function metricValue(side, key) {
  const s = (side && side.stats) || {};
  if (key === 'loadMs') return side ? side.loadMs : null;
  if (key === 'guessPct') return s.guesses ? s.guesses.keptPct : null;
  return s[key];
}
function metricText(side, key) {
  const s = (side && side.stats) || {};
  switch (key) {
    case 'loadMs': return side && side.loadMs != null ? fmtMs(side.loadMs) : 'already loaded';
    case 'firstWordMs': case 'totalMs': return fmtMs(s[key]);
    case 'writeTps': return fmtRate(s);
    case 'promptTokens': return tokensWords(s.promptTokens, false, s.cachedTokens);
    case 'completionTokens': return tokensWords(s.completionTokens, s.approxTokens);
    case 'guessPct': return guessWords(s);
    default: return stopWords(s.finishReason);
  }
}
function betterSide(a, b, key) {
  const metric = PG_METRICS.find((row) => row[0] === key);
  if (!metric || metric[2] == null || !a || !b) return null;
  const rawA = metricValue(a, key); const rawB = metricValue(b, key);
  if (rawA == null || rawB == null) return null;
  const x = Number(rawA); const y = Number(rawB);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  const top = Math.max(Math.abs(x), Math.abs(y));
  if (top === 0 || Math.abs(x - y) / top < 0.03) return null;
  return (metric[2] ? x < y : x > y) ? a.key : b.key;
}
function speedRows(side, other) {
  return PG_METRICS.map(([key, label]) => ({ key, label, text: metricText(side, key), better: !!other && betterSide(side, other, key) === side.key }));
}

const PG_STEP_ICON = { waiting: '○', running: '◐', done: '✓', failed: '✗', skipped: '–' };
function stepLine(step) {
  let time = '';
  if (step.state === 'running') time = fmtMs(step.elapsedMs || 0);
  else if (step.ms != null) time = fmtMs(step.ms);
  else if (step.state === 'waiting' && step.guessS) time = `about ${fmtGuess(step.guessS)}`;
  return `${PG_STEP_ICON[step.state] || '○'} ${step.label}${time ? ` · ${time}` : ''}${step.detail ? ` — ${step.detail}` : ''}`;
}
function planLines(plan) {
  return {
    steps: (plan.steps || []).map((step) => `${step.label}${step.guessS ? ` · about ${fmtGuess(step.guessS)}` : ''}`),
    total: `About ${fmtGuess(plan.estimateS || 0)} in all.`,
    warnings: plan.warnings || [],
  };
}
const PG_STATUS_WORDS = { running: 'Running…', stopping: 'Stopping…', restoring: 'Putting things back…', done: 'Done.',
  failed: "The test didn't finish.", stopped: 'Stopped.', yielded: 'Stopped to let another app through.' };
function statusWords(job) {
  if (!job) return '';
  const base = PG_STATUS_WORDS[job.status] || String(job.status);
  return job.message && job.message !== base ? `${base} ${job.message}` : base;
}

function historyRecord(job) {
  const cut = (text) => String(text || '').slice(0, PG_ANSWER_KEEP);
  return { id: job.id, at: job.startedAt, prompt: job.prompt, system: job.system || '', status: job.status,
    message: job.message || '', restore: job.restore || '',
    sides: (job.sides || []).map((s) => ({ key: s.key, model: s.model, name: s.name, preset: s.preset || null,
      settingsLabel: s.settingsLabel, sampling: s.sampling, loadMs: s.loadMs, stats: s.stats, error: s.error || null,
      answer: cut(s.answer), reasoning: cut(s.reasoning) })) };
}
function historyAdd(list, record, max = PG_HISTORY_MAX) { return [record, ...(list || []).filter((row) => row.id !== record.id)].slice(0, max); }
function sideTitle(side) { return `${side.key} · ${side.name} · ${side.settingsLabel || (side.preset ? `preset “${side.preset}”` : 'Saved settings')}`; }
function historyMarkdown(record) {
  const sides = record.sides || [];
  const lines = [`## Test ${record.at || ''}`.trim(), '', ...String(record.prompt || '').split('\n').map((line) => `> ${line}`), ''];
  lines.push(`| | ${sides.map(sideTitle).join(' | ')} |`, `|---|${sides.map(() => '---').join('|')}|`);
  for (const [key, label] of PG_METRICS) lines.push(`| ${label} | ${sides.map((s) => metricText(s, key)).join(' | ')} |`);
  for (const s of sides) lines.push('', `### ${sideTitle(s)}`, '', s.error ? `(${s.error})` : '', String(s.answer || '').trim() || '(no answer)');
  if (record.restore) lines.push('', `_${record.restore}_`);
  return `${lines.filter((line, i, all) => !(line === '' && all[i - 1] === '')).join('\n')}\n`;
}
function loadHistory(storage) {
  try { const value = JSON.parse(storage.getItem(PG_HISTORY_KEY) || '[]'); return Array.isArray(value) ? value : []; } catch (_) { return []; }
}
function saveHistory(storage, list) {
  let items = (list || []).slice();
  while (items.length) {
    try { storage.setItem(PG_HISTORY_KEY, JSON.stringify(items)); return items; } catch (_) { items = items.slice(0, -1); }
  }
  try { storage.removeItem(PG_HISTORY_KEY); } catch (_) {}
  return [];
}

/* ---------- browser ---------- */
function pgStore() { try { return window.localStorage; } catch (_) { return null; } }
function pgShowError(text) { const box = $('pg-error'); box.textContent = text || ''; box.hidden = !text; }
function pgPost(url, body) { return json(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body || {}) }); }
function pgFillPresets(side) {
  const model = ((pg.options && pg.options.models) || []).find((m) => m.id === $(`pg-${side}-model`).value);
  const select = $(`pg-${side}-preset`);
  const keep = select.value;
  const presets = (model && model.presets) || [];
  select.innerHTML = `<option value="">Saved settings</option>${presets.map((name) => `<option value="${pgEsc(name)}">Preset “${pgEsc(name)}”</option>`).join('')}`;
  select.value = presets.includes(keep) ? keep : '';
}
function pgFillModels() {
  const models = (pg.options && pg.options.models) || [];
  const loaded = models.find((m) => m.state === 'ready');
  for (const side of ['a', 'b']) {
    const select = $(`pg-${side}-model`);
    const keep = select.value;
    select.innerHTML = models.map((m) => `<option value="${pgEsc(m.id)}">${pgEsc(m.name)}${m.state === 'ready' ? ' (loaded)' : ''}</option>`).join('');
    select.value = models.some((m) => m.id === keep) ? keep : ((loaded || models[0] || {}).id || '');
    pgFillPresets(side);
  }
}
function pgSide(side) {
  const value = (field) => $(`pg-${side}-${field}`).value.trim();
  return { model: value('model'), preset: value('preset') || null, temperature: value('temperature'), top_p: value('top_p'),
    top_k: value('top_k'), maxTokens: value('maxTokens') || 512 };
}
function pgRequest() {
  const sides = [pgSide('a')];
  if ($('pg-b-on').checked) sides.push(pgSide('b'));
  return { prompt: $('pg-prompt').value, system: $('pg-system').value, sides, warmup: $('pg-warmup').checked, putBack: $('pg-putback').checked };
}
async function pgLoadOptions() {
  const { response, body } = await json('/api/playground/options');
  if (!response.ok) { pgShowError(panelErrorText(body, "Couldn't read the model list.")); return; }
  pg.options = body;
  pgFillModels();
}
function pgRenderPlan() {
  const box = $('pg-plan');
  if (!pg.plan) { box.hidden = true; return; }
  const lines = planLines(pg.plan);
  $('pg-plan-steps').innerHTML = lines.steps.map((line) => `<li>${pgEsc(line)}</li>`).join('');
  $('pg-plan-total').textContent = lines.total;
  $('pg-plan-warnings').innerHTML = lines.warnings.map((w) => `<p class="hint warn">${pgEsc(w)}</p>`).join('');
  box.hidden = false;
}
async function pgRun() {
  pgShowError('');
  const { response, body } = await pgPost('/api/playground/plan', pgRequest());
  if (!response.ok) { pg.plan = null; pgRenderPlan(); pgShowError(panelErrorText(body, "Couldn't plan the test.")); return; }
  pg.plan = body;
  pgRenderPlan();
}
async function pgStart() {
  if (!pg.plan) return;
  const { response, body } = await pgPost('/api/playground/runs', { ...pgRequest(), expectBefore: pg.plan.before, confirm: true });
  if (!response.ok) {
    if (body && body.plan) { pg.plan = body.plan; pgRenderPlan(); }
    pgShowError(panelErrorText(body, "Couldn't start the test."));
    return;
  }
  pg.plan = null; pgRenderPlan(); pg.shown = null; pg.job = body; pgRenderJob(); pgSchedule();
}
async function pgStop() { const { body } = await pgPost('/api/playground/runs/current/stop'); if (body && body.status) { pg.job = body; pgRenderJob(); } }
function pgSchedule() { clearTimeout(pg.timer); pg.timer = setTimeout(pgPoll, PG_POLL_MS); }
async function pgPoll() {
  clearTimeout(pg.timer);
  let result;
  try { result = await json('/api/playground/runs/current'); } catch (_) { pgSchedule(); return; }
  if (!result.response.ok) { pgSchedule(); return; }
  pg.job = result.body.status === 'idle' ? null : result.body;
  pgRenderJob();
  if (pg.job && PG_ACTIVE.includes(pg.job.status)) pgSchedule();
  else if (pg.job) { pgRemember(pg.job); if (pg.options) pgLoadOptions(); }
}
function pgRemember(job) {
  if (!job || !job.id || PG_ACTIVE.includes(job.status) || pg.history.some((row) => row.id === job.id)) return;
  const next = historyAdd(pg.history, historyRecord(job));
  const store = pgStore();
  pg.history = store ? saveHistory(store, next) : next;
  pgRenderHistory();
}
function pgResultsHtml(record) {
  const sides = record.sides || [];
  return sides.map((side) => {
    const other = sides.find((s) => s.key !== side.key);
    const rows = speedRows(side, other).map((row) => `<tr><th scope="row">${pgEsc(row.label)}</th><td>${pgEsc(row.text)}${row.better ? ' <span class="pg-better">better</span>' : ''}</td></tr>`).join('');
    const thinking = side.reasoning ? `<details class="pg-thinking"><summary>Thinking</summary><div class="pg-answer">${pgEsc(side.reasoning)}</div></details>` : '';
    const error = side.error ? `<p class="error">${pgEsc(side.error)}</p>` : '';
    const answer = side.answer ? pgEsc(side.answer) : '<span class="muted">(no answer yet)</span>';
    return `<article class="pg-result" data-side="${pgEsc(side.key)}"><h3>${pgEsc(sideTitle(side))}</h3><table class="pg-speed">${rows}</table>${error}${thinking}<div class="pg-answer">${answer}</div></article>`;
  }).join('');
}
function pgRenderJob() {
  const record = pg.shown || pg.job;
  const active = !!(pg.job && PG_ACTIVE.includes(pg.job.status));
  $('pg-run').disabled = active;
  $('pg-stop').hidden = !(active && pg.job.status === 'running');
  $('pg-steps').innerHTML = record && record.steps ? record.steps.map((step) => `<li class="pg-step ${pgEsc(step.state)}">${pgEsc(stepLine(step))}</li>`).join('') : '';
  $('pg-status').textContent = pg.shown ? `Earlier test from ${pg.shown.at || ''}` : statusWords(pg.job);
  $('pg-results').innerHTML = record ? pgResultsHtml(record) : '';
  $('pg-restore').textContent = record ? (record.restore || '') : '';
}
function pgRenderHistory() {
  const list = pg.history;
  $('pg-history').innerHTML = list.length ? list.map((row, i) => `<div class="pg-history-row"><div><strong>${pgEsc(row.at || '')}</strong> <span class="small">${pgEsc(String(row.prompt || '').slice(0, 80))}</span><div class="small">${(row.sides || []).map((s) => `${pgEsc(s.key)}: ${pgEsc(s.name)}, ${pgEsc(fmtRate(s.stats))}`).join(' · ')}</div></div><div class="actions"><button class="button small" type="button" data-pg-show="${i}">Show</button><button class="button small" type="button" data-pg-copy="${i}">Copy</button></div></div>`).join('') : '<p class="empty">No tests yet.</p>';
}
async function pgCopy(i) {
  const record = pg.history[i];
  if (!record) return;
  try { await navigator.clipboard.writeText(historyMarkdown(record)); setNotice('Copied.', 'good'); } catch (_) { setNotice("Couldn't copy: the browser blocked it.", 'warn'); }
}
function pgExport() {
  const blob = new Blob([JSON.stringify(pg.history, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `freetoken-tests-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
async function pgClear() {
  if (!(await askConfirm('Clear every earlier test from this browser?', 'Clear'))) return;
  const store = pgStore();
  pg.history = store ? saveHistory(store, []) : [];
  pg.shown = null;
  pgRenderHistory(); pgRenderJob();
}
function pgWire() {
  if (pg.wired) return;
  pg.wired = true;
  $('pg-a-model').addEventListener('change', () => pgFillPresets('a'));
  $('pg-b-model').addEventListener('change', () => pgFillPresets('b'));
  $('pg-b-on').addEventListener('change', () => { $('pg-b-fields').disabled = !$('pg-b-on').checked; });
  $('pg-run').addEventListener('click', pgRun);
  $('pg-start').addEventListener('click', pgStart);
  $('pg-plan-cancel').addEventListener('click', () => { pg.plan = null; pgRenderPlan(); });
  $('pg-stop').addEventListener('click', pgStop);
  $('pg-export').addEventListener('click', pgExport);
  $('pg-clear').addEventListener('click', pgClear);
  $('pg-history').addEventListener('click', (event) => {
    const show = event.target.closest('[data-pg-show]');
    const copy = event.target.closest('[data-pg-copy]');
    if (show) { pg.shown = pg.history[Number(show.dataset.pgShow)] || null; pgRenderJob(); }
    if (copy) pgCopy(Number(copy.dataset.pgCopy));
  });
}
async function pgOpen() {
  pgWire();
  const store = pgStore();
  pg.history = store ? loadHistory(store) : [];
  pgRenderHistory();
  await pgLoadOptions();
  await pgPoll();
}

if (typeof module !== 'undefined') module.exports = { fmtMs, fmtGuess, fmtRate, guessWords, stopWords, tokensWords, metricText, betterSide,
  speedRows, stepLine, planLines, statusWords, historyRecord, historyAdd, historyMarkdown, sideTitle, loadHistory, saveHistory,
  PG_HISTORY_KEY, PG_HISTORY_MAX, pg };
```

- [ ] **Step 4: Run them and see them pass**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_page.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Controller commits**

```bash
git add python/freetoken/daemon/settings/static/playground.js tests/settings/test_playground_page.py
git commit -m "feat(settings): Test tab page script: speed rows, steps, history, copy and export

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: The tab, its section, and the Right-now strip (`index.html`, `panel.js`)

**Files:**
- Modify: `python/freetoken/daemon/settings/static/index.html`, `python/freetoken/daemon/settings/static/panel.js`
- Test: `tests/settings/test_playground_page.py` (append)

- [ ] **Step 1: Append the failing tests**

```python
def test_page_has_the_test_tab_and_every_id_the_script_uses():
    page = PAGE.read_text(encoding="utf-8")
    for needle in ('data-main="test"', 'id="test-view"', '<script src="/playground.js"></script>',
                   'id="pg-prompt"', 'id="pg-system"', 'id="pg-a-model"', 'id="pg-a-preset"', 'id="pg-b-on"',
                   'id="pg-b-fields"', 'id="pg-warmup"', 'id="pg-putback"', 'id="pg-run"', 'id="pg-start"',
                   'id="pg-plan-cancel"', 'id="pg-stop"', 'id="pg-steps"', 'id="pg-results"', 'id="pg-restore"',
                   'id="pg-history"', 'id="pg-export"', 'id="pg-clear"'):
        assert needle in page, needle
    assert page.index('<script src="/panel.js">') < page.index('<script src="/playground.js">')
    script = PG_JS.read_text(encoding="utf-8")
    for element in set(re.findall(r"\$\('(pg-[\w-]+)'\)", script)):
        assert f'id="{element}"' in page, element
    for side in ("a", "b"):
        for field in ("model", "preset", "temperature", "top_p", "top_k", "maxTokens"):
            assert f'id="pg-{side}-{field}"' in page, (side, field)


def test_show_main_knows_the_test_tab():
    text = PANEL_JS.read_text(encoding="utf-8")
    assert "'freetoken', 'test']" in text and "pgOpen()" in text


def test_right_now_strip_shows_a_running_test_and_a_leftover():
    _node(r"""
const base = {switcher: {up: true, running: [{id: 'quasar-27b', name: 'QUASAR', state: 'ready'}]}, card: null, held: []};
const running = p.nowStripHtml({...base, test: {running: true, model: 'fable-27b', name: 'Fable', preset: 'Three'}});
assert.match(running, /Test running: Fable on preset “Three”/);
assert.match(p.nowStripHtml({...base, test: {running: true, model: null}}), /A test is running on the Test tab\./);
const left = p.nowStripHtml({...base, testLeftover: {model: 'quasar-27b', name: 'QUASAR', preset: 'Fast'}});
assert.match(left, /QUASAR is still on test settings \(preset “Fast”\) because an app was using it\. It goes back to its saved settings at its next load\./);
assert.doesNotMatch(p.nowStripHtml(base), /test/i);
""", module=PANEL_JS)
```

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_page.py -q`
Expected: 3 new failures (`data-main="test"` missing, `'test'` not in panel.js, no strip lines).

- [ ] **Step 3: `panel.js`**

`showMain`:

```javascript
function showMain(name) {
  panel.main = ['models', 'system', 'ninfer', 'freetoken', 'test'].includes(name) ? name : 'models';
  document.querySelectorAll('[data-main]').forEach((button) => button.setAttribute('aria-selected', String(button.dataset.main === panel.main)));
  try { localStorage.setItem('ft-main', panel.main); } catch (_) {}
  const testView = $('test-view');
  if (testView) testView.hidden = panel.main !== 'test';
  if (panel.main === 'test') { clearView(); showEditor(false); $('models-view').hidden = true; if (typeof pgOpen === 'function') pgOpen(); return; }
  if (panel.main === 'models') { clearView(); showEditor(false); loadModels(); return; }
  openView(panel.main === 'system' ? '/api/panel/views/system' : `/api/panel/views/engine/${panel.main}`);
}
```

`nowStripHtml`: add these after the `stale` line, and put `${test}${leftover}` after
`${held}` in the "Loaded" stat.

```javascript
  const test = now.test && now.test.running
    ? `<div class="sub">${now.test.model ? `Test running: ${panelEsc(now.test.name || now.test.model)} on preset “${panelEsc(now.test.preset)}”` : 'A test is running on the Test tab.'}</div>` : '';
  const leftover = now.testLeftover
    ? `<div class="error">${panelEsc(now.testLeftover.name || now.testLeftover.model)} is still on test settings${now.testLeftover.preset ? ` (preset “${panelEsc(now.testLeftover.preset)}”)` : ''} because an app was using it. It goes back to its saved settings at its next load.</div>` : '';
```

- [ ] **Step 4: `index.html`**

Tab button after "FreeToken defaults":

```html
      <button class="tab-btn" role="tab" type="button" data-main="test" aria-selected="false">Test</button>
```

Section after `<section class="panel" id="models-view">…</section>`:

```html
    <section class="panel" id="test-view" hidden>
      <div class="panel-head"><div><h2>Test</h2><p class="small">Ask one or two setups the same thing and compare speed and answers. Loading a model is timed on its own and never counted in the answer. Afterwards the model that was loaded goes back on its saved settings, and a model another app is using is never put away.</p></div></div>
      <p class="error" id="pg-error" hidden></p>
      <label class="pg-label" for="pg-prompt">What should the model answer?</label>
      <textarea id="pg-prompt" rows="5" maxlength="20000" placeholder="For example: explain how a heat pump works in three short paragraphs."></textarea>
      <details class="pg-box"><summary>System message (optional)</summary><textarea id="pg-system" rows="3" maxlength="8000"></textarea></details>
      <div class="pg-grid">
        <div class="pg-setup">
          <h3>Setup A</h3>
          <label>Model <select id="pg-a-model"></select></label>
          <label>Settings <select id="pg-a-preset"></select></label>
          <details class="pg-box"><summary>Answer settings</summary>
            <label>Creativity (temperature) <input id="pg-a-temperature" type="number" min="0" max="2" step="0.1" placeholder="the model's own"></label>
            <label>Top-p <input id="pg-a-top_p" type="number" min="0" max="1" step="0.05" placeholder="the model's own"></label>
            <label>Top-k <input id="pg-a-top_k" type="number" min="0" max="200" step="1" placeholder="the model's own"></label>
            <label>Longest answer (tokens) <input id="pg-a-maxTokens" type="number" min="1" max="4096" step="1" value="512"></label>
          </details>
        </div>
        <div class="pg-setup">
          <h3>Setup B</h3>
          <label class="pg-toggle"><input type="checkbox" id="pg-b-on"> Compare with a second setup</label>
          <fieldset id="pg-b-fields" disabled>
            <label>Model <select id="pg-b-model"></select></label>
            <label>Settings <select id="pg-b-preset"></select></label>
            <details class="pg-box"><summary>Answer settings</summary>
              <label>Creativity (temperature) <input id="pg-b-temperature" type="number" min="0" max="2" step="0.1" placeholder="the model's own"></label>
              <label>Top-p <input id="pg-b-top_p" type="number" min="0" max="1" step="0.05" placeholder="the model's own"></label>
              <label>Top-k <input id="pg-b-top_k" type="number" min="0" max="200" step="1" placeholder="the model's own"></label>
              <label>Longest answer (tokens) <input id="pg-b-maxTokens" type="number" min="1" max="4096" step="1" value="512"></label>
            </details>
          </fieldset>
        </div>
      </div>
      <div class="pg-options">
        <label><input type="checkbox" id="pg-warmup" checked> Warm up after loading (not counted)</label>
        <label><input type="checkbox" id="pg-putback" checked> Load what's loaded now again afterwards</label>
      </div>
      <div class="actions"><button class="button primary" id="pg-run" type="button">Run</button><button class="button danger" id="pg-stop" type="button" hidden>Stop</button></div>
      <div class="pg-plan" id="pg-plan" hidden>
        <h3>The plan</h3>
        <ol id="pg-plan-steps"></ol>
        <p class="small" id="pg-plan-total"></p>
        <div id="pg-plan-warnings"></div>
        <div class="actions"><button class="button primary" id="pg-start" type="button">Start</button><button class="button" id="pg-plan-cancel" type="button">Cancel</button></div>
      </div>
      <p class="small" id="pg-status" aria-live="polite"></p>
      <ul class="pg-steps" id="pg-steps"></ul>
      <div class="pg-grid" id="pg-results"></div>
      <p class="small" id="pg-restore"></p>
      <div class="pg-history">
        <div class="panel-head"><h3>Earlier tests (this browser)</h3><div class="actions"><button class="button small" id="pg-export" type="button">Export all</button><button class="button small ghost" id="pg-clear" type="button">Clear history</button></div></div>
        <div id="pg-history"></div>
      </div>
    </section>
```

CSS, next to the Stage A panel rules:

```css
    /* Test tab (part 3) */
    #test-view textarea { width: 100%; }
    .pg-label { display: block; font-weight: 600; margin: 4px 0 6px; }
    .pg-box { margin: 10px 0; }
    .pg-box > summary { cursor: pointer; color: var(--muted); font-weight: 600; }
    .pg-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 14px; margin: 14px 0; }
    .pg-setup { border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; background: var(--panel-2); min-width: 0; }
    .pg-setup label { display: block; margin: 8px 0; font-size: .9rem; }
    .pg-setup fieldset { border: 0; padding: 0; margin: 0; min-width: 0; }
    .pg-setup fieldset:disabled { opacity: .55; }
    .pg-options { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 6px 0 12px; font-size: .9rem; }
    .pg-plan { border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 12px; padding: 12px 16px; margin: 12px 0; background: var(--panel-2); }
    .pg-steps { list-style: none; padding: 0; margin: 8px 0; font-size: .9rem; font-variant-numeric: tabular-nums; }
    .pg-steps li { padding: 3px 0; overflow-wrap: anywhere; }
    .pg-steps li.running { font-weight: 600; }
    .pg-steps li.failed { color: var(--bad); }
    .pg-steps li.skipped { color: var(--muted); }
    .pg-result { border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; background: var(--panel); min-width: 0; }
    .pg-result h3 { font-size: 1rem; margin: 0 0 8px; overflow-wrap: anywhere; }
    .pg-speed { width: 100%; border-collapse: collapse; font-size: .88rem; font-variant-numeric: tabular-nums; margin-bottom: 10px; }
    .pg-speed th { text-align: left; font-weight: 500; color: var(--muted); padding: 3px 8px 3px 0; white-space: nowrap; }
    .pg-speed td { padding: 3px 0; overflow-wrap: anywhere; }
    .pg-better { font-size: .72rem; background: var(--good-soft); color: var(--good); border-radius: 999px; padding: 1px 7px; margin-left: 4px; }
    .pg-answer { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 480px; overflow: auto; border-top: 1px solid var(--line); padding-top: 8px; font-size: .92rem; }
    .pg-thinking > summary { cursor: pointer; color: var(--muted); font-size: .85rem; }
    .pg-history { margin-top: 18px; border-top: 1px solid var(--line); padding-top: 12px; }
    .pg-history-row { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; padding: 8px 0; border-bottom: 1px solid var(--line); }
    .pg-history-row > div:first-child { min-width: 0; flex: 1 1 240px; overflow-wrap: anywhere; }
```

Script tag right after `<script src="/panel.js"></script>`:

```html
  <script src="/playground.js"></script>
```

- [ ] **Step 5: Run the page tests and the existing page tests**

`tests/settings/test_static_page.py` pins two things this task changes (verified in a dry
run). Update exactly these two assertions:
- line 72: `assert parser.external_assets == ["/panel.js", "/playground.js"], "the page loads only its own panel.js and playground.js"`;
- line 130: the `data-main` list becomes `["models", "system", "ninfer", "freetoken", "test"]`.

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings/test_playground_page.py tests/settings/test_panel_page.py tests/settings/test_static_page.py -q`
Expected: all pass.

- [ ] **Step 6: Controller commits**

```bash
git add python/freetoken/daemon/settings/static/index.html python/freetoken/daemon/settings/static/panel.js tests/settings/test_playground_page.py tests/settings/test_static_page.py
git commit -m "feat(settings): Test tab on the control panel page

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Docs and the full suite

**Files:**
- Modify: `README.md` (under "Windows settings helper", after the control panel paragraph)

- [ ] **Step 1: README paragraph**

```markdown
The **Test** tab asks one or two setups the same prompt and shows, for every answer: how long
until the first word, the writing speed (the engine's own figure when it reports one, else
measured, marked `~`), the whole answer's time, prompt and answer sizes, and for NInfer how many
guessed-ahead words were kept (FreeToken does not report that per request yet). A setup is a
model plus its saved settings or one of its presets, plus answer settings. There is one
graphics card, so setups run one after the other. Each switch is its own timed step and is
never counted in an answer. A preset is tried through an in-memory overlay on the switcher
file; the registry is never written. Afterwards the model that was loaded before goes back
on its saved settings. The tab never puts away or loads over a model another app is using,
and says so when it has to leave something. The last 20 tests stay in the browser with Copy
(Markdown) and Export (JSON). Design: `docs/superpowers/specs/2026-09-25-playground-design.md`.
```

- [ ] **Step 2: Full settings and daemon suites**

Run: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings tests/daemon -q`
Expected: the same failures as on `fedf802` and no new ones. On the devbox, 2026-09-25,
`fedf802` already fails 13 tests: `test_boot_parser` (4), `test_browse` (1),
`test_crash_watchdog` (1), `test_governor_dials` (3), `test_governor_loop` (1),
`test_memory_plan` (2) and `test_model_info` (1). They depend on the box's boot file, the
Windows paths or the model folders. Diff the `FAILED` lines against that list (no checkout
needed). Record the counts in the task report.

- [ ] **Step 3: Controller commits**

```bash
git add README.md
git commit -m "docs: Test tab in the settings helper section

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push -u origin feat/playground
```

---

### Task 10: Live acceptance with screenshots (controller only, ask Jay first)

**Files:**
- Create: `docs/research/playground-acceptance-<YYYY-MM-DD>.md`
- Create: `docs/research/img/playground-<step>.png` (desktop 1440x900; phone 390x844 where noted)

The box is reached by the route in memory: `ssh 5090`, then a stdin script into
`wsl -d vllm -e bash -l`. The screenshots come from the devbox with
`~/.npm-global/bin/chrome-devtools-axi` against the tailnet page
`https://5090.tail45ff04.ts.net`.

- [ ] **Step 1: Ask Jay for the window.** Say:
  - about 40 minutes;
  - QUASAR, Fable and Twin are loaded and unloaded several times;
  - **one** FreeToken boot (about 2.5 min, plus 2.5 min to put it back if it was loaded);
  - the helper restarts twice.

  Ask which models he is using right now. Do not start without a yes. Then run
  `curl -s 127.0.0.1:2040/running`, and confirm that whatever is listed may be put away.

- [ ] **Step 2: Deploy and record the starting state**

```bash
cd ~/FreeToken && git fetch -q && git checkout -q feat/playground && git pull -q
sha256sum ~/.config/freetoken/registry.json ~/llama-swap/config.yaml | tee /tmp/pg-before.sha
systemctl --user restart freetoken-settings && sleep 5
curl -s 127.0.0.1:2031/api/playground/options | python3 -c "import json,sys; b=json.load(sys.stdin); print(b['switcherUp'], [m['id'] for m in b['models']])"
curl -s 127.0.0.1:2031/api/playground/runs/current
```
Expected: `True` and the five model ids; `{"status":"idle"}`. The helper restart keeps a
serving FreeToken (helper 1.4.0). If FreeToken was loaded and Jay did not OK a restart, stop
here.

- [ ] **Step 3: Screens: the empty tab**

```bash
AXI=~/.npm-global/bin/chrome-devtools-axi
$AXI open https://5090.tail45ff04.ts.net && $AXI resize 1440 900 && $AXI wait 1500
$AXI snapshot | grep -n 'Test' | head      # find the Test tab uid
$AXI click @<uid-of-Test-tab> && $AXI wait 800
$AXI screenshot docs/research/img/playground-empty.png
$AXI resize 390 844 && $AXI screenshot docs/research/img/playground-empty-phone.png && $AXI resize 1440 900
```
Expected:
- the tab shows the prompt box, Setup A (the loaded model is preselected) and Setup B
  (switched off);
- the phone shot has no sideways scrolling: `$AXI eval "document.documentElement.scrollWidth <= innerWidth"`
  prints `true`.

- [ ] **Step 4: One setup on QUASAR, saved settings** (QUASAR loaded first, or it is loaded
  as step 1)
  - Prompt: "Explain how a heat pump works in three short paragraphs." Longest answer 512.
  - Press Run, check the plan, then Start.
  - Expected:
    - the steps tick through;
    - the result shows First word after, Writing speed **without** `~` (NInfer reports it)
      and "Guesses kept" with a percentage (DFlash2);
    - the Right-now strip shows no test line afterwards.
  - Cross-check the writing speed against llama-swap's Activity row for the same request
    (`curl -s '127.0.0.1:2040/api/metrics/activity?model=quasar-27b&limit=1'`, `tokens_per_second`).
    They should agree within 5%.
  - Screenshot `playground-quasar-one.png`.

- [ ] **Step 5: QUASAR vs Fable** (model vs model)
  - Setup A QUASAR, setup B Fable, same prompt. Start.
  - Expected:
    - the steps are: put away QUASAR (only if B runs first), load Fable, … put things back;
    - "Loading (not counted)" shows about 20 s for Fable;
    - put-back ends with "QUASAR is loaded again on its saved settings (…)";
    - `curl -s 127.0.0.1:2040/running` shows quasar-27b `ready`.
  - Screenshots:
    - `playground-compare-running.png`, taken during B's load;
    - `playground-compare-done.png`, taken at the end, with "better" tags visible.

- [ ] **Step 6: Fable saved vs a temporary preset** (Review focus 1)
  - On Fable's settings, set "Words guessed ahead" to 3, press "Save as new preset" with the
    name "PG test 3", reset the dial and Save. (This is the only registry write in the
    session and is undone in the last bullet.)
  - Record `sha256sum ~/.config/freetoken/registry.json ~/llama-swap/config.yaml`.
  - Test: A = Fable, Saved settings; B = Fable, Preset "PG test 3".
  - During B's answer, run `ps -o args= -p $(pgrep -x ninfer-serve) | grep -o -- '--draft-tokens [0-9]*'`.
    It prints `--draft-tokens 3`.
  - The Right-now strip shows "Test running: Fable… on preset “PG test 3”".
  - Also during the test: saving any dial on the Models tab shows "A test is running on the
    Test tab…".
  - Afterwards:
    - both sha256 sums equal the recorded ones;
    - `ls ~/.config/freetoken/playground-test.json` fails (no marker);
    - QUASAR (the model before) is loaded again.
  - Delete the "PG test 3" preset.
  - Screenshot `playground-preset-compare.png`.

- [ ] **Step 7: Stop during a load**
  - Test: A = Twin. Press Stop while "Load Twin" runs.
  - Expected:
    - the load step shows ✗ with "Stopped.", and the rest shows – skipped;
    - put-back loads QUASAR again;
    - `/running` shows quasar-27b only;
    - the config sha256 equals the one from before the test.
  - Screenshot `playground-stopped.png`.

- [ ] **Step 8: Helper restart mid-test** (recover)
  - Test: A = Fable, Preset "PG test 3". Re-create it for this step and delete it afterwards.
  - During A's answer: `systemctl --user restart freetoken-settings`.
  - After about 10 s:
    - `ls ~/.config/freetoken/playground-test.json` fails;
    - `/running` does not list fable-27b (put away because idle);
    - `sha256sum ~/llama-swap/config.yaml` equals the registry render;
    - the page shows no test.

- [ ] **Step 9: In-use refusal** (Review focus 2)
  - With QUASAR loaded, start a long stream in another shell:
    `curl -sN 127.0.0.1:2040/v1/chat/completions -H 'content-type: application/json' -d '{"model":"quasar-27b","stream":true,"max_tokens":2000,"messages":[{"role":"user","content":"Count to 500."}]}' > /dev/null &`
  - Press Run with A = Fable.
  - Expected: "…is answering something right now. Try again when it's done." Nothing
    unloads.
  - Wait for the curl to end, then press Run again. The plan warns "…was used N seconds
    ago…".
  - Screenshot `playground-in-use.png`.

- [ ] **Step 10: One FreeToken setup** (the session's only FreeToken boot; skip if Jay said no)
  - Test: A = qwen3.8-flash, saved settings, 256 tokens.
  - Expected:
    - the load step takes about 150 s and is shown under "Loading (not counted)";
    - First word after is in seconds, not minutes;
    - Writing speed is shown with `~` (measured);
    - Guesses kept says "not reported by this engine";
    - put-back puts FreeToken away and loads QUASAR.
  - Screenshot `playground-freetoken.png`.

- [ ] **Step 11: History, Copy, Export**
  - Reload the page and open the Test tab. "Earlier tests" lists every run above, newest
    first.
  - Press Show on the QUASAR vs Fable run. The results come back.
  - Press Copy, then check the Markdown with
    `$AXI eval "navigator.clipboard.readText()"` (or paste it into the doc).
  - Press Export all. A JSON file downloads.
  - Screenshot `playground-history.png`.

- [ ] **Step 12: Put the box back**
  - Leave the box on the branch until the merge; it is the same helper plus the tab.
  - If any step failed and cannot be fixed in the session:
    `git checkout -q mtp-upstream-merge && git pull -q && systemctl --user restart freetoken-settings`.
  - Confirm with Jay what should be loaded, and load it from the Models tab.

- [ ] **Step 13: Write and commit the acceptance doc.** It holds a table of every step with
  pass or fail and the numbers:
  - load times, first word, writing speed (page vs Activity), guesses kept;
  - sha256 sums before and after;
  - the `ps` flags seen;
  - the recover outcome;
  - any leftovers.

  It links every screenshot.

```bash
git add docs/research/playground-acceptance-*.md docs/research/img/playground-*.png
git commit -m "docs(research): Test tab live acceptance

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push
```

---

### Task 11: Codex review (Astra, xhigh) and fixes, two rounds at most

- [ ] **Step 1: Resolve the model** (never from memory):
  `bash ~/.claude/skills/codex-models/scripts/codex-models.sh resolve astra`. It prints
  something like `gpt-6-astra	low medium high xhigh max ultra`, and `xhigh` must be listed.

- [ ] **Step 2: Round 1**

```bash
CR=~/.claude/skills/codex-review/scripts/codex-review.sh
bash $CR --model astra --level xhigh -C /home/jay/projects/FreeToken/.worktrees/feat-playground \
  --base origin/mtp-upstream-merge --focus "Read docs/superpowers/specs/2026-09-25-playground-design.md and \
docs/superpowers/plans/2026-09-25-playground.md first. Check, in this order: (1) the five Review Focus items; \
(2) thread safety of PlaygroundRunner (job dict, _stop, _loading, stop() racing _run) and PanelService \
(test_running vs _save/load/unload, the overlay vs the hold watcher); (3) that _restore can never load over \
another app's model, never loads before the switcher has the saved file, and always clears the overlay and \
marker; (4) that no answer number can include a load or the warm-up; (5) plain words on the page. \
List every finding at once."
```

  The last line is `VERDICT: FINDINGS <n>` or `VERDICT: NO-MARKERS`. Read the whole text
  either way.

- [ ] **Step 3: Fix every P0/P1** (and any P2 that touches a Review Focus item) through an
  implementer, with no git writes. Each fix gets a test that fails first. Run the full
  suite. The controller commits `fix(settings): …` per theme.

- [ ] **Step 4: Round 2 on the fixes only**: `bash $CR --model astra --level xhigh -C <worktree> --commit <sha>`
  for each fix commit, or `--base <round-1 head>`. Stop after this round. Anything left
  goes in the PR body as a follow-up.

---

### Task 12: PR, merge, memory

- [ ] **Step 1: Open the PR**

```bash
gh pr create --base mtp-upstream-merge --head feat/playground \
  --title "feat: Test tab: speed numbers and side-by-side compare with safe put-back" --body-file - <<'BODY'
## What changed for Jay
- A **Test** tab on the control panel. Ask one or two setups the same prompt. A setup is a model, its saved settings or a preset, and answer settings.
- Every answer shows first word after, writing speed, whole answer, sizes, and guesses kept (NInfer).
- Loading is its own timed step and is never counted in an answer.
- Afterwards the model that was loaded goes back on its saved settings. A model another app is using is never put away.
- The last 20 tests stay in the browser, with Copy and Export.

## How
- The helper runs the test (daemon/settings/playground.py). Presets go through an in-memory overlay on the switcher file; the registry is never written. A marker file lets a restarted helper put a leftover test model away.
- No llama-swap change (FROZEN.md untouched). It uses P1, P2, P5 and P6 as they are, plus the upstream /api/events in-flight snapshot and /api/metrics/activity.

## Tests
- `PYTHONPATH=python .venv/bin/python -m pytest tests/settings tests/daemon -q`: <counts>
- Live acceptance: docs/research/playground-acceptance-<date>.md (with screenshots)
- Codex review (Astra xhigh): <round 1 findings / fixes>, <round 2 result>

## Follow-ups
- FreeToken per-request guess-ahead counts (timings.draft_n / draft_n_accepted in the usage chunk)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
```

- [ ] **Step 2: Ask Jay to merge.** Squash-merge only on his yes, and only when Task 10 passed
  and the review rounds are done: `gh pr merge <n> --squash`. Then put the box on
  `mtp-upstream-merge` (`git checkout -q mtp-upstream-merge && git pull -q && systemctl --user restart freetoken-settings`),
  but only with no FreeToken loaded, or with Jay's OK.

- [ ] **Step 3: Memory.** Add `project-test-playground.md` with:
  - the merge commit;
  - the design choice (panel tab, helper-run job, overlay, no llama-swap patch);
  - the marker path;
  - the acceptance numbers;
  - the follow-ups.

  Add one line to `MEMORY.md`.

---

## Self-review against the spec

| Spec item | Where |
|---|---|
| Speed numbers per answer, engine vs measured, guesses when reported | Task 1; page words in Task 7 |
| What each engine sends (FreeToken usage only; NInfer timings + draft counts) | Task 1 module docstring; spec table |
| Side-by-side A/B: model vs model, preset vs preset, answer settings | Tasks 4-5 (setup = model + preset + sampling) |
| Sequential with the switch shown and timed on its own | Task 4 steps; Task 5 `_load`; Task 7 `stepLine` |
| Presets without writing the registry; always put back | Task 2 overlay; Task 5 `_restore`, `recover` |
| Never evict a model someone is using | Task 4 `plan` checks; Task 5 `_do` unload check, `_restore`, `_load` P1 |
| Never leave the wrong settings silently | Task 2 `testLeftover`; Task 5 leftover path; Task 8 strip |
| Lives on the control panel; llama-swap untouched | Tasks 6-8; Global Constraints |
| History (20, localStorage, Copy, Export) | Task 7 |
| Live acceptance with screenshots | Task 10 |
| Codex review (Astra xhigh) and merge | Tasks 11-12 |

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
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
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

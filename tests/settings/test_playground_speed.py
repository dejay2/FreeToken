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

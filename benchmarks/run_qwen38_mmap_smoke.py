"""End-to-end smoke checks for Qwen3.8 Flash Next with mmap-backed PLE."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, payload: dict[str, Any], timeout: int = 900) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def post_stream(
    url: str, payload: dict[str, Any], timeout: int = 900
) -> tuple[str, str, dict[str, Any] | None]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    usage: dict[str, Any] | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])
                if delta.get("content"):
                    content_parts.append(delta["content"])
    return "".join(reasoning_parts), "".join(content_parts), usage


def assert_model(base_url: str, model: str, expected_context: int | None) -> None:
    body = get_json(f"{base_url}/models")
    matches = [item for item in body.get("data", []) if item.get("id") == model]
    assert len(matches) == 1, f"expected one {model!r} model, got {matches!r}"
    card = matches[0]
    assert card.get("context_length") == 262144, card
    assert set(card.get("supported_reasoning_efforts") or []) == {
        "low",
        "medium",
        "xhigh",
    }, card

    if expected_context is not None:
        status = get_json(f"{base_url}/cache/status")
        geometry = status["geometry"]
        usable = (geometry["num_pages"] - 1) * geometry["page_size"]
        assert usable == expected_context, (usable, geometry)
    print("PASS model and live context information")


def assert_text(chat_url: str, model: str) -> None:
    body = post_json(
        chat_url,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "Reply with exactly: first text works"},
                {"role": "user", "content": "Test the first request."},
            ],
            "max_tokens": 96,
            "reasoning_effort": "low",
        },
    )
    content = body["choices"][0]["message"].get("content") or ""
    assert content.strip() == "first text works", content
    print("PASS first text request")


def assert_stream(chat_url: str, model: str) -> None:
    reasoning, content, usage = post_stream(
        chat_url,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "Reply with exactly: stream works"},
                {"role": "user", "content": "Test the streamed request."},
            ],
            "max_tokens": 96,
            "reasoning_effort": "low",
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert reasoning.strip(), "stream contained no reasoning_content"
    assert content.strip() == "stream works", content
    assert usage and usage.get("total_tokens", 0) > 0, usage
    print("PASS streamed reasoning, content, and usage")


def assert_tool(chat_url: str, model: str) -> None:
    body = post_json(
        chat_url,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Use get_weather for Paris. Do not answer without the tool.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
            "max_tokens": 512,
            "reasoning_effort": "low",
        },
    )
    calls = body["choices"][0]["message"].get("tool_calls") or []
    assert len(calls) == 1, body
    function = calls[0].get("function") or {}
    assert function.get("name") == "get_weather", function
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    assert arguments.get("city", "").lower() == "paris", arguments
    print("PASS parsed tool call")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:2020/v1")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--expected-context", type=int)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"

    try:
        assert_model(base_url, args.model, args.expected_context)
        assert_text(chat_url, args.model)
        assert_stream(chat_url, args.model)
        assert_tool(chat_url, args.model)
        assert_model(base_url, args.model, args.expected_context)
    except (AssertionError, KeyError, json.JSONDecodeError, urllib.error.URLError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    print("PASS server remained healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

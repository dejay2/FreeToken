import json
import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from freetoken.server.recent_prompts import RecentPrompts, register_recent_prompt_routes


def payload(text="shared instructions", **extra):
    return {"model": "local", "messages": [{"role": "system", "content": text},
            {"role": "user", "content": "task"}], "chat_template_kwargs": {"enable_thinking": False},
            "tools": [{"type": "function", "function": {"name": "read", "parameters": {
                "type": "object", "properties": {"image": {"type": "string"}}}}}], **extra}


def record(store, body=None, format="openai"):
    return store.add(format, json.dumps(body or payload()).encode(), epoch=store.epoch)


def test_exact_payload_is_kept_immutable_but_not_exposed_in_list():
    store = RecentPrompts()
    body = payload("<script>alert(1)</script>")
    id = record(store, body)
    body["messages"][0]["content"] = "changed later"
    listing = store.list()
    row = listing["prompts"][0]
    assert row["id"] == id and row["format"] == "openai"
    assert row["message_count"] == 2 and row["preview"] == "task"
    assert "request" not in row
    saved = store.get(id)
    assert saved["request"]["messages"][0]["content"] == "<script>alert(1)</script>"
    saved["request"]["messages"].clear()
    assert len(store.get(id)["request"]["messages"]) == 2


def test_count_bytes_and_age_bound_retention():
    now = [0.0]
    size = len(json.dumps(payload()).encode())
    store = RecentPrompts(capacity=2, max_bytes=size * 2, max_request_bytes=size, ttl_seconds=10, clock=lambda: now[0])
    first = record(store)
    second = record(store)
    third = record(store)
    assert [p["id"] for p in store.list()["prompts"]] == [third, second]
    assert store.get(first) is None
    assert store.list()["stored_bytes"] == size * 2
    assert record(store, payload("too big" * 100)) is None
    assert store.list()["skipped_count"] == 1
    assert store.get(second) is not None  # Oversize input does not evict valid entries.
    now[0] = 10
    assert store.list()["prompts"] == [] and store.list()["stored_bytes"] == 0


def test_byte_budget_evicts_before_count_limit():
    size = len(json.dumps(payload()).encode())
    store = RecentPrompts(capacity=50, max_bytes=size, max_request_bytes=size)
    first = record(store)
    second = record(store)
    assert store.get(first) is None
    assert [p["id"] for p in store.list()["prompts"]] == [second]


@pytest.mark.parametrize("body", [
    payload(cache_private=True), payload(private=True),
    payload(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": "data:..."}]}]),
    payload(messages=[{"role": "user", "content": [{"type": "tool_result", "content": [{"type": "image", "source": {}}]}]}]),
])
def test_private_and_image_requests_are_not_retained(body):
    store = RecentPrompts()
    assert record(store, body) is None
    assert not store.list()["prompts"]


def test_clear_and_pause_invalidate_captures_already_in_flight():
    store = RecentPrompts()
    record(store)
    epoch = store.epoch
    store.clear()
    assert store.add("openai", json.dumps(payload()).encode(), epoch=epoch) is None
    assert not store.list()["prompts"]
    epoch = store.epoch
    store.set_enabled(False)
    assert record(store) is None
    store.set_enabled(True)
    assert store.add("openai", json.dumps(payload()).encode(), epoch=epoch) is None
    assert record(store)


def app_client():
    app = FastAPI()
    store = register_recent_prompt_routes(app)

    class Chat(BaseModel):
        model: str
        messages: list[dict]
        stream: bool = False

    @app.post("/v1/chat/completions")
    @app.post("/v1/messages")
    async def generate(body: Chat):
        if body.model == "rejected":
            return JSONResponse({"error": "rejected"}, status_code=400)
        if body.stream:
            async def chunks():
                yield b"data: first\n\n"
                yield b"data: [DONE]\n\n"
            return StreamingResponse(chunks(), media_type="text/event-stream")
        return {"choices": [{"message": {"content": "ok"}}]}

    @app.post("/v1/messages/count_tokens")
    async def count_tokens():
        return {"input_tokens": 100}

    return TestClient(app), store


@pytest.mark.parametrize("path,format", [("/v1/chat/completions", "openai"), ("/v1/messages", "anthropic")])
def test_live_routes_capture_unchanged_input_and_support_select_clear_pause(path, format):
    client, store = app_client()
    body = payload(unknown_field={"keep": "exactly"})
    assert client.post(path, json=body).status_code == 200
    result = client.get("/v1/cache/recent").json()["result"]
    id = result["prompts"][0]["id"]
    selected = client.get(f"/v1/cache/recent/{id}").json()["result"]
    assert selected["format"] == format and selected["request"] == body
    assert client.put("/v1/cache/recent/settings", json={"enabled": False}).status_code == 200
    assert client.post(path, json=body).status_code == 200
    assert len(store.list()["prompts"]) == 1
    assert client.delete("/v1/cache/recent").status_code == 200
    assert client.get(f"/v1/cache/recent/{id}").status_code == 404
    assert client.get("/v1/cache/recent").json()["result"]["enabled"] is False


def test_capture_leaves_streaming_body_untouched_and_skips_rejected_or_count_requests():
    client, store = app_client()
    with client.stream("POST", "/v1/chat/completions", json=payload(stream=True)) as response:
        assert response.read() == b"data: first\n\ndata: [DONE]\n\n"
    assert len(store.list()["prompts"]) == 1
    assert client.post("/v1/chat/completions", json=payload(model="rejected")).status_code == 400
    assert client.post("/v1/chat/completions", json={"model": "invalid"}).status_code == 422
    assert client.post("/v1/messages/count_tokens", json=payload()).status_code == 200
    assert len(store.list()["prompts"]) == 1


def test_recorder_failure_does_not_fail_generation(monkeypatch):
    client, store = app_client()
    def fail(*args, **kwargs):
        raise RuntimeError("history failed")
    monkeypatch.setattr(store, "add", fail)
    assert client.post("/v1/chat/completions", json=payload()).status_code == 200


def test_expiration_task_preserves_server_lifespan_and_clears_snapshots_on_shutdown():
    events = []
    @asynccontextmanager
    async def original(app):
        events.append("model startup")
        yield {"model": "ready"}
        events.append("model shutdown")
    app = FastAPI(lifespan=original)
    store = register_recent_prompt_routes(app)
    async def run():
        async with app.router.lifespan_context(app) as state:
            assert state == {"model": "ready"}
            assert events == ["model startup"]
            record(store)
            assert store.list()["prompts"]
        assert store.list()["prompts"] == []
        assert events == ["model startup", "model shutdown"]
    asyncio.run(run())

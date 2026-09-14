from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import freetoken.message as message
from freetoken.server import api_server


class _State:
    def __init__(self, response: dict | None = None) -> None:
        self.maintenance_state = "serving"
        self.config = SimpleNamespace(reasoning_parser=None, served_model_name="local-model")
        self.response = response or {"status": "ok", "result": {}}
        self.sent: list[object] = []

    async def dispatch_prefix(self, msg, timeout: float = 30.0):
        self.sent.append(msg)
        return dict(self.response)


def _client(state: _State) -> TestClient:
    from freetoken.server.prefix_api import register_prefix_routes

    app = FastAPI()
    register_prefix_routes(app, lambda: state, lambda: {})
    return TestClient(app)


def _register(client: TestClient, *, format: str = "openai", request: dict | None = None):
    request = request or {
        "model": "local-model",
        "messages": [{"role": "system", "content": "shared rules"}],
        "max_tokens": 8,
    }
    return client.post(
        "/v1/cache/prefixes",
        json={
            "name": "agent-base",
            "format": format,
            "request": request,
            "prefix_tokens": 64,
            "ttl_seconds": 120,
        },
    )


def test_api_server_registers_prefix_routes():
    paths = {(route.path, tuple(sorted(route.methods or ()))) for route in api_server.app.routes}
    assert any(path == "/v1/cache/prefixes" and "POST" in methods for path, methods in paths)
    assert any(path == "/v1/cache/prefixes" and "GET" in methods for path, methods in paths)
    assert any(path == "/v1/cache/prefixes/{name}" and "GET" in methods for path, methods in paths)
    assert any(path == "/v1/cache/prefixes/{name}/warm" and "POST" in methods for path, methods in paths)
    assert any(path == "/v1/cache/prefixes/{name}" and "DELETE" in methods for path, methods in paths)
    assert any(path == "/v1/cache/prefixes/settings" and "PUT" in methods for path, methods in paths)


def test_openai_registration_preserves_later_system_message_and_tool_schema():
    state = _State({"status": "ok", "result": {"name": "agent-base", "aligned_tokens": 64}})
    client = _client(state)
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    response = _register(
        client,
        request={
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "initial rules"},
                {"role": "user", "content": "first task"},
                {"role": "system", "content": "later budget"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "read", "description": "Read a file", "parameters": schema},
                }
            ],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "result": {"name": "agent-base", "aligned_tokens": 64},
        "error": None,
    }
    sent = state.sent[0]
    assert isinstance(sent, getattr(message, "PrefixCacheMsg"))
    assert sent.action == "register"
    assert sent.name == "agent-base"
    assert sent.prefix_tokens == 64
    assert sent.ttl_seconds == 120
    assert [item["role"] for item in sent.text] == ["system", "user", "system"]
    assert sent.text[2]["content"] == "later budget"
    assert sent.tools[0]["function"]["parameters"] == schema
    assert sent.preserve_system_order is False


def test_anthropic_registration_uses_chronological_system_and_native_tools():
    state = _State()
    client = _client(state)
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    response = _register(
        client,
        format="anthropic",
        request={
            "model": "local-model",
            "system": "initial rules",
            "messages": [
                {"role": "user", "content": "first task"},
                {"role": "system", "content": "later budget"},
            ],
            "tools": [{"name": "read", "description": "Read a file", "input_schema": schema}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    sent = state.sent[0]
    assert [item["role"] for item in sent.text] == ["system", "user", "system"]
    assert sent.text[2]["content"] == "later budget"
    assert sent.tools == [
        {
            "type": "function",
            "function": {"name": "read", "description": "Read a file", "parameters": schema},
        }
    ]
    assert sent.preserve_system_order is True


@pytest.mark.parametrize(
    "format,request_body",
    [
        (
            "openai",
            {
                "model": "local-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
                    }
                ],
                "max_tokens": 8,
            },
        ),
        (
            "anthropic",
            {
                "model": "local-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {"type": "base64", "data": "AA=="}}],
                    }
                ],
                "max_tokens": 8,
            },
        ),
    ],
)
def test_registration_rejects_images_before_dispatch(format: str, request_body: dict):
    state = _State()
    response = _register(_client(state), format=format, request=request_body)
    assert response.status_code == 400
    assert response.json()["status"] == "invalid"
    assert "image" in response.json()["error"].lower()
    assert state.sent == []


def test_registration_rejects_image_nested_inside_anthropic_tool_result():
    state = _State()
    response = _register(_client(state), format="anthropic", request={
        "model": "local-model", "max_tokens": 8,
        "messages": [{"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "call_1",
            "content": [{"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "AA=="}}],
        }]}],
    })
    assert response.status_code == 400
    assert "image" in response.json()["error"].lower()
    assert state.sent == []


@pytest.mark.parametrize("flag", ["cache_private", "private"])
def test_registration_rejects_private_request_flags(flag: str):
    state = _State()
    request = {
        "model": "local-model",
        "messages": [{"role": "user", "content": "secret tail"}],
        "max_tokens": 8,
        flag: True,
    }
    response = _register(_client(state), request=request)
    assert response.status_code == 400
    assert response.json()["status"] == "invalid"
    assert "private" in response.json()["error"].lower()
    assert state.sent == []


def test_registration_preserves_empty_text_turn_for_canonical_template_rendering():
    state = _State()
    response = _register(
        _client(state),
        request={
            "model": "local-model",
            "messages": [{"role": "user", "content": ""}],
            "max_tokens": 8,
        },
    )
    assert response.status_code == 200
    assert state.sent[0].text == [{"role": "user", "content": ""}]


def test_malformed_messages_are_reported_as_invalid_instead_of_crashing_precheck():
    state = _State()
    response = _register(
        _client(state),
        request={"model": "local-model", "messages": None, "max_tokens": 8},
    )
    assert response.status_code == 400
    assert response.json()["status"] == "invalid"
    assert state.sent == []


def test_management_actions_dispatch_correlated_messages_and_warm_returns_202():
    state = _State()
    client = _client(state)

    assert client.get("/v1/cache/prefixes").status_code == 200
    assert client.get("/v1/cache/prefixes/agent-base").status_code == 200
    warm = client.post("/v1/cache/prefixes/agent-base/warm")
    assert warm.status_code == 202
    assert client.delete("/v1/cache/prefixes/agent-base").status_code == 200
    assert client.put(
        "/v1/cache/prefixes/settings", json={"max_retained_bytes": 16 * 1024 * 1024}
    ).status_code == 200

    assert [(msg.action, msg.name) for msg in state.sent] == [
        ("list", ""),
        ("get", "agent-base"),
        ("warm", "agent-base"),
        ("delete", "agent-base"),
        ("configure", ""),
    ]
    assert state.sent[-1].max_retained_bytes == 16 * 1024 * 1024


def test_configure_rejects_negative_retention_without_dispatch():
    state = _State()
    response = _client(state).put(
        "/v1/cache/prefixes/settings", json={"max_retained_bytes": -1}
    )
    assert response.status_code == 422
    assert state.sent == []


def test_unsupported_backend_status_is_preserved_as_a_conflict():
    state = _State(
        {
            "status": "unsupported",
            "result": {"supported": False},
            "error": "model cache family is unsupported",
        }
    )
    response = _register(_client(state))
    assert response.status_code == 409
    assert response.json() == {
        "status": "unsupported",
        "result": {"supported": False},
        "error": "model cache family is unsupported",
    }


def test_http_timeout_is_reported_as_gateway_timeout():
    state = _State()

    async def timeout(msg, timeout=30.0):
        raise asyncio.TimeoutError

    state.dispatch_prefix = timeout
    response = _client(state).get("/v1/cache/prefixes")
    assert response.status_code == 504
    assert response.json() == {
        "status": "failed",
        "result": {},
        "error": "prefix control request timed out",
    }


def _manager():
    class _Queue:
        async def put(self, msg):
            return None

        async def get(self):
            await asyncio.Future()

        def stop(self):
            return None

    return api_server.FrontendManager(
        config=SimpleNamespace(kv_park="off", kv_dtype="bf16"),
        send_tokenizer=_Queue(),
        recv_tokenizer=_Queue(),
        maintenance_state="serving",
    )


@pytest.mark.anyio
async def test_frontend_prefix_future_cleans_up_after_correlated_response():
    manager = _manager()
    PrefixCacheMsg = getattr(message, "PrefixCacheMsg")
    PrefixCacheReply = getattr(message, "PrefixCacheReply")

    async def send(msg):
        asyncio.get_running_loop().call_soon(
            manager._resolve_prefix,
            PrefixCacheReply(msg.request_id, "ok", {"name": msg.name}),
        )

    manager.send_one = send
    result = await manager.dispatch_prefix(
        PrefixCacheMsg(request_id="done-1", action="get", name="agent-base"), timeout=1
    )
    assert result == {"status": "ok", "result": {"name": "agent-base"}, "error": None}
    assert manager.prefix_futures == {}


@pytest.mark.anyio
async def test_frontend_prefix_future_cleans_up_on_dispatch_failure_and_timeout():
    manager = _manager()
    PrefixCacheMsg = getattr(message, "PrefixCacheMsg")

    async def fail_send(msg):
        raise RuntimeError("queue closed")

    manager.send_one = fail_send
    with pytest.raises(RuntimeError, match="queue closed"):
        await manager.dispatch_prefix(PrefixCacheMsg("dispatch-1", "list"), timeout=1)
    assert manager.prefix_futures == {}

    async def silent_send(msg):
        return None

    manager.send_one = silent_send
    with pytest.raises(asyncio.TimeoutError):
        await manager.dispatch_prefix(PrefixCacheMsg("timeout-1", "list"), timeout=0.001)
    assert manager.prefix_futures == {}


@pytest.mark.anyio
async def test_frontend_prefix_future_cleans_up_on_cancellation():
    manager = _manager()
    PrefixCacheMsg = getattr(message, "PrefixCacheMsg")
    manager.send_one = lambda msg: asyncio.sleep(0)
    task = asyncio.create_task(
        manager.dispatch_prefix(PrefixCacheMsg("cancel-1", "list"), timeout=30)
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.prefix_futures == {}


@pytest.mark.anyio
async def test_backend_death_resolves_prefix_waiters_as_failed():
    manager = _manager()
    PrefixCacheMsg = getattr(message, "PrefixCacheMsg")
    manager.send_one = lambda msg: asyncio.sleep(0)
    task = asyncio.create_task(
        manager.dispatch_prefix(PrefixCacheMsg("dead-1", "get", "agent-base"), timeout=30)
    )
    await asyncio.sleep(0)

    api_server._fail_open_waiters(manager, "scheduler exited")

    result = await task
    assert result == {"status": "failed", "result": {}, "error": "scheduler exited"}
    assert manager.prefix_futures == {}

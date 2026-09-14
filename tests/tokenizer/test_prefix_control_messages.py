from __future__ import annotations

import asyncio

import pytest
import torch
from jinja2 import TemplateError

import freetoken.message as message
from freetoken.message import BaseBackendMsg, BaseFrontendMsg, BaseTokenizerMsg
import freetoken.tokenizer.server as tokenizer_server


class _Queue:
    def __init__(self) -> None:
        self.items: list[object] = []

    def put(self, item: object) -> None:
        self.items.append(item)


class _TokenizeManager:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.messages: list[object] = []

    def tokenize(self, messages):
        self.messages.extend(messages)
        if self.error is not None:
            raise self.error
        return [torch.tensor([11, 22, 33, 44], dtype=torch.int32)]


def _types():
    expected = (
        "PrefixCacheMsg",
        "PrefixCacheBackendMsg",
        "PrefixCacheResultMsg",
        "PrefixCacheReply",
    )
    for name in expected:
        assert hasattr(message, name), f"missing wire type {name}"
    return tuple(getattr(message, name) for name in expected)


def test_prefix_control_wire_types_round_trip_free_form_results_and_tensors():
    PrefixCacheMsg, PrefixCacheBackendMsg, PrefixCacheResultMsg, PrefixCacheReply = _types()

    request = PrefixCacheMsg(
        request_id="front-1",
        action="register",
        name="agent-base",
        text=[{"role": "system", "content": "base"}],
        tools=[{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        chat_template_kwargs={"enable_thinking": False},
        preserve_system_order=True,
        prefix_tokens=128,
        ttl_seconds=45.0,
    )
    decoded_request = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(request))
    assert decoded_request == request

    backend = PrefixCacheBackendMsg(
        request_id="front-1",
        action="register",
        name="agent-base",
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        prefix_tokens=128,
        ttl_seconds=45.0,
    )
    decoded_backend = BaseBackendMsg.decoder(backend.encoder())
    assert decoded_backend.request_id == "front-1"
    assert decoded_backend.input_ids.dtype == torch.int32
    assert torch.equal(decoded_backend.input_ids, torch.tensor([1, 2, 3], dtype=torch.int32))

    result = PrefixCacheResultMsg(
        request_id="front-1",
        status="ok",
        result={"prefix": {"name": "agent-base"}, "nested": [1, {"ready": True}]},
    )
    decoded_result = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(result))
    assert decoded_result == result

    reply = PrefixCacheReply(
        request_id="front-1", status="failed", result={}, error="worker failed"
    )
    decoded_reply = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply))
    assert decoded_reply == reply


def test_registration_uses_ordinary_tokenizer_and_forwards_exact_int32_ids():
    PrefixCacheMsg, PrefixCacheBackendMsg, _, _ = _types()
    backend, frontend = _Queue(), _Queue()
    manager = _TokenizeManager()
    msg = PrefixCacheMsg(
        request_id="register-1",
        action="register",
        name="agent-base",
        text=[
            {"role": "system", "content": "base"},
            {"role": "user", "content": "task"},
        ],
        tools=[{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        chat_template_kwargs={"enable_thinking": False},
        preserve_system_order=True,
        prefix_tokens=3,
        ttl_seconds=90.0,
    )

    assert tokenizer_server._forward_prefix_msg(msg, manager, backend, frontend)

    assert frontend.items == []
    assert len(manager.messages) == 1
    rendered = manager.messages[0]
    assert rendered.text == msg.text
    assert rendered.tools == msg.tools
    assert rendered.chat_template_kwargs == msg.chat_template_kwargs
    assert rendered.preserve_system_order is True
    assert len(backend.items) == 1
    forwarded = backend.items[0]
    assert isinstance(forwarded, PrefixCacheBackendMsg)
    assert forwarded.request_id == "register-1"
    assert forwarded.action == "register"
    assert forwarded.name == "agent-base"
    assert forwarded.prefix_tokens == 3
    assert forwarded.ttl_seconds == 90.0
    assert forwarded.input_ids.dtype == torch.int32
    assert forwarded.input_ids.tolist() == [11, 22, 33, 44]


def test_system_scope_uses_computed_boundary_with_original_full_ids():
    PrefixCacheMsg, _, _, _ = _types()
    backend, frontend = _Queue(), _Queue()
    manager = _TokenizeManager()
    seen = []
    def boundary(msg, ids):
        seen.append((msg, ids.tolist()))
        return 2
    manager.system_prefix_tokens = boundary
    msg = PrefixCacheMsg(request_id='system-1', action='register', name='base',
                         text=[{'role':'system','content':'rules'}, {'role':'user','content':'hi'}],
                         prefix_scope='system')
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert decoded.prefix_scope == 'system'
    assert tokenizer_server._forward_prefix_msg(decoded, manager, backend, frontend)
    assert not frontend.items
    assert seen[0][0].text == msg.text
    assert backend.items[0].input_ids.tolist() == [11,22,33,44]
    assert backend.items[0].prefix_tokens == 2


@pytest.mark.parametrize("action", ["list", "get", "warm", "delete", "configure"])
def test_non_registration_actions_pass_through_without_tokenization(action: str):
    PrefixCacheMsg, PrefixCacheBackendMsg, _, _ = _types()
    backend, frontend = _Queue(), _Queue()
    manager = _TokenizeManager(error=AssertionError("tokenizer must not run"))
    msg = PrefixCacheMsg(
        request_id=f"{action}-1",
        action=action,
        name="agent-base" if action in {"get", "warm", "delete"} else "",
        max_retained_bytes=4096 if action == "configure" else None,
    )

    assert tokenizer_server._forward_prefix_msg(msg, manager, backend, frontend)

    assert manager.messages == []
    assert frontend.items == []
    assert len(backend.items) == 1
    forwarded = backend.items[0]
    assert isinstance(forwarded, PrefixCacheBackendMsg)
    assert forwarded.request_id == msg.request_id
    assert forwarded.action == action
    assert forwarded.name == msg.name
    assert forwarded.max_retained_bytes == msg.max_retained_bytes
    assert forwarded.input_ids is None


@pytest.mark.parametrize("error, status", [
    (ValueError("template rejected roles"), "invalid"),
    (TemplateError("template rejected roles"), "invalid"),
    (RuntimeError("encoder worker broke"), "failed"),
])
def test_tokenizer_failure_returns_a_correlated_classified_reply(error, status):
    PrefixCacheMsg, _, _, PrefixCacheReply = _types()
    backend, frontend = _Queue(), _Queue()
    msg = PrefixCacheMsg(
        request_id="broken-1",
        action="register",
        name="broken",
        text=[{"role": "user", "content": "bad layout"}],
    )

    assert tokenizer_server._forward_prefix_msg(
        msg, _TokenizeManager(error=error), backend, frontend
    )

    assert backend.items == []
    assert len(frontend.items) == 1
    reply = frontend.items[0]
    assert isinstance(reply, PrefixCacheReply)
    assert reply.request_id == "broken-1"
    assert reply.status == status
    assert reply.result == {}
    assert str(error) in reply.error


def test_empty_registration_returns_correlated_invalid_reply():
    PrefixCacheMsg, _, _, PrefixCacheReply = _types()
    backend, frontend = _Queue(), _Queue()
    manager = _TokenizeManager()
    manager.tokenize = lambda messages: [torch.empty(0, dtype=torch.int32)]

    assert tokenizer_server._forward_prefix_msg(
        PrefixCacheMsg(request_id="empty-1", action="register", name="empty", text=""),
        manager,
        backend,
        frontend,
    )

    assert backend.items == []
    reply = frontend.items[0]
    assert isinstance(reply, PrefixCacheReply)
    assert reply.request_id == "empty-1"
    assert reply.status == "invalid"
    assert "at least one token" in reply.error


def test_scheduler_result_forwards_to_frontend_with_correlation():
    _, _, PrefixCacheResultMsg, PrefixCacheReply = _types()
    backend, frontend = _Queue(), _Queue()
    result = PrefixCacheResultMsg(
        request_id="result-1",
        status="unsupported",
        result={"supported": False},
        error="model cache family is unsupported",
    )

    assert tokenizer_server._forward_prefix_msg(result, _TokenizeManager(), backend, frontend)

    assert backend.items == []
    assert frontend.items == [
        PrefixCacheReply(
            request_id="result-1",
            status="unsupported",
            result={"supported": False},
            error="model cache family is unsupported",
        )
    ]

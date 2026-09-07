from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import freetoken.server.api_server as api
from freetoken.message import CacheResidencyReply, CacheStepMsg, CacheStepReply
from freetoken.server.api_server import (
    AdmissionClosedError,
    CacheStepRequest,
    FrontendManager,
    cache_residency,
    cache_step,
    dispatch_step,
)


def _make_manager(maintenance_state: str = "serving") -> FrontendManager:
    config = SimpleNamespace(
        served_model_name="test-model",
        kv_park="off",
        kv_dtype="bf16",
    )
    return FrontendManager(
        config=config,
        send_tokenizer=None,
        recv_tokenizer=None,
        maintenance_state=maintenance_state,
    )


def test_new_user_sync_returns_int_when_serving():
    manager = _make_manager("serving")
    uid = manager.new_user()
    assert isinstance(uid, int)
    assert uid == 0
    assert manager.stats.active == 1

    uid2 = manager.new_user()
    assert isinstance(uid2, int)
    assert uid2 == 1
    assert manager.stats.active == 2


def test_new_user_immediate_error_on_loading_failed_stopping():
    for state in ("loading", "failed", "stopping"):
        manager = _make_manager(state)
        with pytest.raises(AdmissionClosedError, match=f"server unavailable: engine is {state}"):
            manager.new_user()
        assert manager.stats.active == 0


@pytest.mark.anyio
async def test_new_user_waits_while_rebuilding_and_resumes():
    manager = _make_manager("rebuilding")
    assert not manager.rebuild_done.is_set()

    # Launch new_user in the background
    task = asyncio.create_task(manager.new_user())
    await asyncio.sleep(0.01)
    assert not task.done()
    assert manager.stats.active == 0

    # Simulate rebuild completing
    manager.maintenance_state = "serving"
    manager.rebuild_done.set()

    uid = await task
    assert isinstance(uid, int)
    assert uid == 0
    assert manager.stats.active == 1


@pytest.mark.anyio
async def test_new_user_timeout_raises_admission_closed():
    manager = _make_manager("rebuilding")
    assert not manager.rebuild_done.is_set()

    with pytest.raises(AdmissionClosedError, match="cache rebuild timed out"):
        await manager.new_user(timeout=0.02)
    assert manager.stats.active == 0


@pytest.mark.anyio
async def test_new_user_fails_if_engine_latches_failed_during_wait():
    manager = _make_manager("rebuilding")
    task = asyncio.create_task(manager.new_user())
    await asyncio.sleep(0.01)
    assert not task.done()

    # Engine crashes mid-rebuild
    manager.maintenance_state = "failed"
    manager.rebuild_done.set()

    with pytest.raises(AdmissionClosedError, match="server unavailable: engine is failed"):
        await task
    assert manager.stats.active == 0


@pytest.mark.anyio
async def test_new_user_async_wrapper():
    manager = _make_manager("serving")
    uid = await manager.new_user_async()
    assert isinstance(uid, int)
    assert uid == 0

    manager2 = _make_manager("rebuilding")
    task = asyncio.create_task(manager2.new_user_async())
    await asyncio.sleep(0.01)
    assert not task.done()

    manager2.maintenance_state = "serving"
    manager2.rebuild_done.set()
    uid2 = await task
    assert isinstance(uid2, int)
    assert uid2 == 0


@pytest.mark.anyio
async def test_generate_endpoint_waits_when_rebuilding():
    prev = api._GLOBAL_STATE
    manager = _make_manager("rebuilding")
    sent = []

    async def mock_send(msg):
        sent.append(msg)

    async def mock_stream(uid):
        yield f"data: {uid}\n\n"

    manager.send_one = mock_send
    manager.stream_generate = mock_stream
    manager.stream_with_cancellation = lambda gen, *_: gen
    api._GLOBAL_STATE = manager

    try:
        from fastapi import Request
        from freetoken.server.api_server import GenerateRequest, generate

        req = GenerateRequest(prompt="hello", max_tokens=10)
        http_req = Request(scope={"type": "http", "method": "POST", "path": "/generate", "headers": []})

        # When rebuilding, generate should wait
        task = asyncio.create_task(generate(req, http_req))
        await asyncio.sleep(0.01)
        assert not task.done()

        # Resolve rebuild
        manager.maintenance_state = "serving"
        manager.rebuild_done.set()

        resp = await task
        assert resp.status_code == 200
        assert len(sent) == 1
    finally:
        api._GLOBAL_STATE = prev


def test_cache_step_preflight_checks():
    client = TestClient(api.app)
    prev = api._GLOBAL_STATE
    try:
        api._GLOBAL_STATE = SimpleNamespace(maintenance_state="loading")
        r = client.post("/v1/cache/step", json={"axis": "vram", "direction": "down"})
        assert r.status_code == 503
        assert "loading" in r.json()["error"]

        api._GLOBAL_STATE = SimpleNamespace(maintenance_state="failed")
        r = client.post("/v1/cache/step", json={"axis": "vram", "direction": "down"})
        assert r.status_code == 503
        assert "latched" in r.json()["error"]

        api._GLOBAL_STATE = SimpleNamespace(maintenance_state="stopping")
        r = client.post("/v1/cache/step", json={"axis": "vram", "direction": "down"})
        assert r.status_code == 409
    finally:
        api._GLOBAL_STATE = prev


class _FakeBackend:
    """Stands in for the tokenizer/scheduler side of the wire: records what the API server sent
    and answers through the same reply handlers listen() runs (_resolve_step / _resolve_residency),
    so the test covers the real message path rather than a shortcut."""

    def __init__(self, manager: FrontendManager, *, answer: bool = True):
        self.manager = manager
        self.sent: list = []
        self.answer = answer
        self.owned = 4

    async def send_one(self, msg):
        self.sent.append(msg)
        if not self.answer:
            return
        if isinstance(msg, CacheStepMsg):
            self.owned -= 1
            reply = CacheStepReply(
                request_id=msg.request_id, status="ok", applied="gpu_owned->pinned", layer=3,
                at_floor=False, moe_cache_size=6144,
                layers={"owned": self.owned, "pinned": 48 - self.owned, "disk": 0},
                vram_free_bytes=1024**3,
            )
            asyncio.get_running_loop().call_soon(self.manager._resolve_step, reply)
        else:  # CacheResidencyMsg
            reply = CacheResidencyReply(
                request_id=msg.request_id, status="ok",
                residency={"layers": {3: "pinned"}, "moe_cache_size": 6144,
                           "owned": self.owned, "pinned": 48 - self.owned, "disk": 0},
            )
            asyncio.get_running_loop().call_soon(self.manager._resolve_residency, reply)


@pytest.mark.anyio
async def test_cache_step_and_residency_round_trip():
    prev = api._GLOBAL_STATE
    manager = _make_manager("serving")
    backend = _FakeBackend(manager)
    manager.send_one = backend.send_one
    api._GLOBAL_STATE = manager
    try:
        resp = await cache_step(CacheStepRequest(axis="vram", direction="down", ram_tight=True, timeout=2.0))
        assert resp.status_code == 200
        data = json.loads(resp.body)
        assert data["status"] == "ok"
        assert data["applied"] == "gpu_owned->pinned"
        assert data["layer"] == 3
        assert data["layers"] == {"owned": 3, "pinned": 45, "disk": 0}
        sent = backend.sent[0]
        assert isinstance(sent, CacheStepMsg) and (sent.axis, sent.direction, sent.ram_tight) == ("vram", "down", True)
        # the step reopened the maintenance gate
        assert manager.maintenance_state == "serving" and manager.rebuild_done.is_set()

        res = await cache_residency()
        assert res["owned"] == 3 and res["pinned"] == 45
        assert res["layers"][3] == "pinned"  # int key here; "3" once JSON-encoded over HTTP
        assert len(backend.sent) == 2
        # <5 s old: served from the cache, no second wire round trip
        backend.owned = 99
        res_cached = await cache_residency()
        assert res_cached["owned"] == 3 and len(backend.sent) == 2
    finally:
        api._GLOBAL_STATE = prev


@pytest.mark.anyio
async def test_cache_step_waits_during_rebuild_and_proceeds():
    prev = api._GLOBAL_STATE
    manager = _make_manager("rebuilding")
    backend = _FakeBackend(manager)
    manager.send_one = backend.send_one
    api._GLOBAL_STATE = manager
    try:
        task = asyncio.create_task(cache_step(CacheStepRequest(axis="vram", direction="down", timeout=2.0)))
        await asyncio.sleep(0.01)
        assert not task.done() and backend.sent == []
        # the in-flight rebuild finishes
        manager.maintenance_state = "serving"
        manager.rebuild_done.set()
        resp = await task
        assert resp.status_code == 200
        assert [type(m) for m in backend.sent] == [CacheStepMsg]
    finally:
        api._GLOBAL_STATE = prev


@pytest.mark.anyio
async def test_cache_step_times_out_when_rebuild_never_finishes():
    prev = api._GLOBAL_STATE
    manager = _make_manager("rebuilding")
    api._GLOBAL_STATE = manager
    try:
        resp = await cache_step(CacheStepRequest(axis="vram", direction="down", timeout=0.02))
        assert resp.status_code == 504
        assert "timed out" in resp.body.decode()
    finally:
        api._GLOBAL_STATE = prev


@pytest.mark.anyio
async def test_cache_step_reply_timeout_leaves_gate_closed_until_the_late_reply():
    """A step whose reply outlives the HTTP timeout keeps the gate shut (the engine is still
    rebuilding); the late reply reopens it, exactly like a rebuild."""
    prev = api._GLOBAL_STATE
    manager = _make_manager("serving")
    backend = _FakeBackend(manager, answer=False)
    manager.send_one = backend.send_one
    api._GLOBAL_STATE = manager
    try:
        resp = await cache_step(CacheStepRequest(axis="vram", direction="up", timeout=0.02))
        assert resp.status_code == 504
        assert manager.maintenance_state == "rebuilding" and not manager.rebuild_done.is_set()
        late = CacheStepReply(request_id=backend.sent[0].request_id, status="ok", applied=None)
        manager._resolve_step(late)
        assert manager.maintenance_state == "serving" and manager.rebuild_done.is_set()
    finally:
        api._GLOBAL_STATE = prev

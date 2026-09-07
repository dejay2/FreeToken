from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import freetoken.server.api_server as api
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


class _FakeEngine:
    def __init__(self):
        self.step_calls = []
        self.owned = 4
        self.slots = 6144

    def step_memory(self, axis: str, direction: str, ram_tight: bool = False):
        self.step_calls.append((axis, direction, ram_tight))
        if axis == "vram" and direction == "down":
            self.owned = 3
            return {
                "applied": "gpu_owned->pinned",
                "layer": 3,
                "moe_cache_size": self.slots,
                "at_floor": False,
                "vram_free_bytes": 1024**3,
            }
        return {"applied": None, "reason": "noop"}

    def residency_report(self):
        return {
            "layers": {0: "gpu_owned", 1: "gpu_owned", 2: "gpu_owned", 3: "pinned"},
            "moe_cache_size": self.slots,
            "owned": self.owned,
            "pinned": 45,
            "disk": 0,
        }


def test_cache_step_and_residency_with_engine():
    client = TestClient(api.app)
    prev = api._GLOBAL_STATE
    fake_engine = _FakeEngine()
    manager = _make_manager("serving")
    manager.engine = fake_engine
    api._GLOBAL_STATE = manager

    try:
        # Step VRAM down
        r = client.post("/v1/cache/step", json={"axis": "vram", "direction": "down", "ram_tight": True})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["applied"] == "gpu_owned->pinned"
        assert data["layer"] == 3
        assert fake_engine.step_calls == [("vram", "down", True)]

        # GET /v1/cache/residency
        r_res = client.get("/v1/cache/residency")
        assert r_res.status_code == 200
        res_data = r_res.json()
        assert res_data["owned"] == 3
        assert res_data["pinned"] == 45
        assert res_data["layers"]["3"] == "pinned"

        # Check caching (<5s returns cached)
        fake_engine.owned = 99
        r_cached = client.get("/v1/cache/residency")
        assert r_cached.json()["owned"] == 3  # still cached value
    finally:
        api._GLOBAL_STATE = prev


@pytest.mark.anyio
async def test_cache_step_waits_during_rebuild_and_proceeds():
    prev = api._GLOBAL_STATE
    fake_engine = _FakeEngine()
    manager = _make_manager("rebuilding")
    manager.engine = fake_engine
    api._GLOBAL_STATE = manager

    try:
        from fastapi import Request
        req = CacheStepRequest(axis="vram", direction="down", timeout=2.0)
        task = asyncio.create_task(cache_step(req))
        await asyncio.sleep(0.01)
        assert not task.done()

        # Rebuild finishes
        manager.maintenance_state = "serving"
        manager.rebuild_done.set()

        resp = await task
        assert resp.status_code == 200
        assert fake_engine.step_calls == [("vram", "down", False)]
    finally:
        api._GLOBAL_STATE = prev


@pytest.mark.anyio
async def test_cache_step_times_out_when_rebuild_never_finishes():
    prev = api._GLOBAL_STATE
    fake_engine = _FakeEngine()
    manager = _make_manager("rebuilding")
    manager.engine = fake_engine
    api._GLOBAL_STATE = manager

    try:
        req = CacheStepRequest(axis="vram", direction="down", timeout=0.02)
        resp = await cache_step(req)
        assert resp.status_code == 504
        assert "timed out" in resp.body.decode()
    finally:
        api._GLOBAL_STATE = prev

"""The API side of sleep: state words, the chat gate's auto-wake, and the two routes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import freetoken.server.api_server as api
from freetoken.message import CacheSleepMsg, CacheSleepReply
from freetoken.server.accounting import AdmissionClosedError
from freetoken.server.api_server import FrontendManager
from freetoken.server.control_api import build_health, is_ready, public_state


def manager(asleep=False) -> FrontendManager:
    config = SimpleNamespace(served_model_name="qwen", kv_park="ram", kv_dtype="fp8")
    m = FrontendManager(config=config, send_tokenizer=None, recv_tokenizer=None, maintenance_state="serving")
    m.asleep = asleep
    return m


def wire_scheduler(m: FrontendManager, *, wake_status="ok", error=None):
    """send_one answers each CacheSleepMsg the way the scheduler would, on the next loop turn."""
    sent = []

    async def send_one(msg):
        sent.append(msg)
        assert isinstance(msg, CacheSleepMsg)
        ok = wake_status == "ok"
        asleep = msg.action == "sleep" if ok else True
        reply = CacheSleepReply(request_id=msg.request_id, action=msg.action,
                                status="ok" if msg.action == "sleep" else wake_status,
                                asleep=asleep, released_bytes=24 << 30, elapsed_s=12.0, error=error)
        asyncio.get_running_loop().call_soon(m._resolve_sleep, reply)

    m.send_one = send_one
    return sent


def test_public_state_health_and_ready_say_sleeping():
    m = manager(asleep=True)
    m.ready_at = 0.0
    assert public_state(m) == "sleeping"
    doc = build_health(m, "1.0")
    assert doc["status"] == "ok" and doc["maintenance"] == "sleeping" and is_ready(doc)
    m.maintenance_state = "rebuilding"  # a wake in flight reads as the rebuild it is
    assert public_state(m) == "rebuilding"


@pytest.mark.anyio
async def test_a_chat_wakes_a_sleeping_model_once_and_is_admitted():
    m = manager(asleep=True)
    sent = wire_scheduler(m)
    uids = await asyncio.gather(m.new_user_async(), m.new_user_async(), m.new_user_async())
    assert sorted(uids) == [0, 1, 2]
    assert [msg.action for msg in sent] == ["wake"]  # three chats, one wake
    assert m.asleep is False and m.maintenance_state == "serving"
    assert m.sleep_info["last_wake_s"] == 12.0


@pytest.mark.anyio
async def test_a_chat_while_a_game_holds_the_card_gets_a_plain_refusal():
    m = manager(asleep=True)
    wire_scheduler(m, wake_status="rejected",
                   error="the graphics card has 2.0 GB free and waking needs 24.5 GB; close the game")
    with pytest.raises(AdmissionClosedError, match="close the game"):
        await m.new_user_async()
    assert m.asleep is True and m.maintenance_state == "serving" and m.stats.active == 0
    reason = await m.wait_until_serving()
    assert "could not wake" in reason


@pytest.mark.anyio
async def test_a_chat_that_waited_out_the_sleep_itself_then_wakes_the_model():
    m = manager()
    sent = wire_scheduler(m)
    sleep = asyncio.ensure_future(api.dispatch_sleep(m, action="sleep"))
    await asyncio.sleep(0)  # the sleep is in flight: the gate is shut
    assert m.maintenance_state == "rebuilding"
    reason = await m.wait_until_serving()
    assert reason is None and (await sleep)["status"] == "ok"
    assert [msg.action for msg in sent] == ["sleep", "wake"] and m.asleep is False


@pytest.mark.anyio
async def test_the_scheduler_auto_wake_reply_clears_the_flag_without_a_waiter():
    m = manager(asleep=True)
    api._open_maintenance(m, "auto-wake:1", "wake")
    m._resolve_sleep(CacheSleepReply(request_id="auto-wake:1", action="wake", status="ok", asleep=False))
    assert m.asleep is False and m.maintenance_state == "serving" and m.rebuild_done.is_set()


def test_the_routes(monkeypatch):
    m = manager()
    monkeypatch.setattr(api, "get_global_state", lambda: m)
    client = TestClient(api.app)

    async def fake_dispatch(state, *, action, timeout=api.WAKE_WAIT_S):
        state.asleep = action == "sleep"
        return {"status": "ok", "action": action, "asleep": state.asleep, "released_bytes": 24 << 30}

    monkeypatch.setattr(api, "dispatch_sleep", fake_dispatch)
    r = client.post("/v1/sleep")
    assert r.status_code == 200 and r.json()["asleep"] is True
    assert client.post("/v1/sleep").json()["note"] == "already asleep"
    assert client.get("/v1/cache/status").json()["state"] == "sleeping"
    r = client.post("/v1/wake")
    assert r.status_code == 200 and r.json()["woke"] is True and m.asleep is False
    m.maintenance_state = "loading"
    assert client.post("/v1/sleep").status_code == 503

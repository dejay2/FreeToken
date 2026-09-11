"""An active but silent scheduler must fail; completed chunks keep it healthy."""
import asyncio
from types import SimpleNamespace

import pytest

from freetoken.server.api_server import FrontendManager
from freetoken.server.control_api import build_health


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def manager(clock):
    return FrontendManager(
        config=SimpleNamespace(served_model_name="test", kv_park="off", kv_dtype="fp8"),
        send_tokenizer=None, recv_tokenizer=None, maintenance_state="serving", monotonic=clock,
    )


def test_silent_active_inference_fails_even_when_the_http_api_is_alive():
    clock = Clock()
    state = manager(clock)
    state.new_user()
    clock.now += 601
    health = build_health(state, "test")
    assert health["status"] == "error"
    assert state.maintenance_state == "failed"
    assert "inference" in health["message"]


def test_idle_time_does_not_count_against_a_new_request():
    clock = Clock()
    state = manager(clock)
    clock.now += 86400
    assert build_health(state, "test")["status"] == "ok"
    state.new_user()
    clock.now += 599
    assert build_health(state, "test")["status"] == "ok"
    clock.now += 2
    assert build_health(state, "test")["status"] == "error"


def test_control_activity_cannot_hide_stalled_inference():
    clock = Clock()
    state = manager(clock)
    state.new_user()
    for _ in range(7):
        clock.now += 100
        state.backend_last_seen = clock()
    assert build_health(state, "test")["status"] == "error"


def test_completed_prefill_chunks_keep_a_long_prompt_healthy():
    from freetoken.message import PrefillProgressReply

    clock = Clock()
    state = manager(clock)
    state.new_user()

    class Receiver:
        async def get(self):
            clock.now += 100
            return PrefillProgressReply(processed_tokens=8192, batch_size=1)

    state.recv_tokenizer = Receiver()

    async def run():
        task = asyncio.create_task(state.listen())
        # One event-loop yield per get makes the real listener observable and cancellable.
        original = state.recv_tokenizer.get

        async def get():
            await asyncio.sleep(0)
            return await original()

        state.recv_tokenizer.get = get
        for _ in range(20):
            await asyncio.sleep(0)
            assert build_health(state, "test")["status"] == "ok"
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert clock.now > 2200
        assert state.check_inference()["prefill_tokens"] >= 8192 * 12

    asyncio.run(run())


@pytest.mark.parametrize("error", [OSError("queue unavailable"), asyncio.CancelledError()])
def test_failed_initial_dispatch_does_not_leave_a_phantom_active_request(error):
    from freetoken.core import SamplingParams
    from freetoken.message import TokenizeMsg

    clock = Clock()
    state = manager(clock)
    uid = state.new_user()

    class Sender:
        async def put(self, msg):
            raise error

    state.send_tokenizer = Sender()
    state.recv_tokenizer = asyncio.Queue()
    with pytest.raises(type(error)):
        asyncio.run(state.send_one(TokenizeMsg(uid=uid, text="test", sampling_params=SamplingParams())))
    clock.now += 601
    assert state.stats.active == 0
    assert uid not in state.ack_map
    assert build_health(state, "test")["status"] == "ok"


def test_decode_and_abort_terminal_are_observed_after_client_disconnect():
    from freetoken.message import UserReply

    async def run():
        clock = Clock()
        state = manager(clock)
        uid = state.new_user()
        incoming = asyncio.Queue()
        outgoing = asyncio.Queue()
        state.recv_tokenizer = incoming
        state.send_tokenizer = outgoing
        listener = asyncio.create_task(state.listen())
        await state.abort_user(uid)
        assert state.stats.active == 1
        for _ in range(4):
            clock.now += 400
            await incoming.put(UserReply(uid=uid, incremental_output="x", finished=False, completion_tokens_delta=1))
            await asyncio.sleep(0)
            assert build_health(state, "test")["status"] == "ok"
        await incoming.put(UserReply(uid=uid, incremental_output="", finished=True, error="request aborted"))
        await asyncio.sleep(0)
        clock.now += 86400
        assert state.stats.active == 0
        assert build_health(state, "test")["status"] == "ok"
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)

    asyncio.run(run())


def test_successful_maintenance_gives_inference_a_fresh_deadline():
    from freetoken.message import CacheRebuildReply

    clock = Clock()
    state = manager(clock)
    state.new_user()
    state.maintenance_state = "rebuilding"
    clock.now += 1200
    assert not state.check_inference()["stuck"]
    state._resolve_rebuild(CacheRebuildReply(request_id="rebuild", status="ok"))
    clock.now += 599
    assert build_health(state, "test")["status"] == "ok"
    clock.now += 2
    assert build_health(state, "test")["status"] == "error"

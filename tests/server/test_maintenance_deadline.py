"""The maintenance gate's finite life (api_server.check_maintenance, 2026-09-09).

A /v1/cache/step or /v1/cache/rebuild closes the API gate; the reply reopens it. The HTTP
timeout only stops waiting, so a scheduler that never replies used to leave the gate closed
for ever with /health saying ok (the 5090 sat like that for 11 min 44 s on 2026-09-09). Now the
one operation that closed the gate is tracked, and its clock restarts on every unit of work
the scheduler reports for it (CacheProgressReply: queued, each drained batch or prompt chunk
ahead of it, executing, each engine phase) and on any other backend message. Only
MAINTENANCE_STUCK_S with NO such signal latches "failed" (never "serving": the engine may be
half torn down) so the helper's watchdog restarts it. Silent-but-working operations (review
F1) must survive; a silent stopped worker must fail finitely. Fake clock throughout.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import freetoken.server.api_server as api
from freetoken.message import CacheProgressReply, CacheStepReply
from freetoken.server.api_server import FrontendManager, dispatch_step
from freetoken.server.control_api import build_health


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _manager(clock: _Clock, *, answer_ids: list | None = None) -> FrontendManager:
    manager = FrontendManager(
        config=SimpleNamespace(served_model_name="m", kv_park="off", kv_dtype="bf16"),
        send_tokenizer=None,
        recv_tokenizer=None,
        maintenance_state="serving",
        monotonic=clock,
    )
    manager.sent = []

    async def send_one(msg):
        manager.sent.append(msg)

    manager.send_one = send_one
    return manager


def _reply(request_id: str, status: str = "ok") -> CacheStepReply:
    return CacheStepReply(request_id=request_id, status=status, applied="disk->pinned")


async def _timed_out_step(manager: FrontendManager) -> str:
    result = await dispatch_step(manager, axis="ram", direction="up", timeout=0.01)
    assert result["status"] == "timeout"
    assert manager.maintenance_state == "rebuilding"
    return manager.sent[-1].request_id


def test_a_never_replying_step_latches_failed_after_the_silence_limit():
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        request_id = await _timed_out_step(manager)
        snap = manager.check_maintenance()
        assert snap["operation"] == {"kind": "step", "request_id": request_id}
        assert snap["stuck"] is False and snap["state"] == "rebuilding"
        clock.now += api.MAINTENANCE_STUCK_S - 1
        assert manager.check_maintenance()["stuck"] is False
        assert build_health(manager, "t")["status"] == "ok"
        clock.now += 2
        health = build_health(manager, "t")  # the /health poller alone reaches the verdict
        assert health["status"] == "error" and "made no progress" in health["message"]
        assert manager.maintenance_state == "failed" and manager.rebuild_done.is_set()
        assert manager.last_rebuild["status"] == "stuck"
        snap = manager.check_maintenance()
        assert snap["stuck"] is True and snap["state"] == "failed"
        # A late "ok" for the expired operation must not reopen a possibly torn-down engine.
        manager._resolve_step(_reply(request_id))
        assert manager.maintenance_state == "failed"
        assert manager.new_user is not None
        with pytest.raises(api.AdmissionClosedError):
            manager.new_user()

    asyncio.run(run())


def _progress(request_id: str, phase: str, detail: str | None = None) -> CacheProgressReply:
    return CacheProgressReply(request_id=request_id, phase=phase, detail=detail)


def test_a_chunked_prompt_ahead_of_a_queued_step_is_never_declared_stuck():
    """Review F1: a real step queued behind one long prompt processed in chunks, parking off,
    no other request, so no token reply and no other backend message for the whole prompt.
    Each drained chunk is reported as progress; thirty minutes of that is healthy."""
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        request_id = await _timed_out_step(manager)
        manager._note_progress(_progress(request_id, "queued"))
        for _ in range(18):  # 30 minutes: one report every 100 s, nothing else from the backend
            clock.now += 100.0
            manager._note_progress(_progress(request_id, "waiting", "drained 1 prefill"))
            snap = manager.check_maintenance()
            assert snap["stuck"] is False and manager.maintenance_state == "rebuilding"
            assert snap["phase"] == "waiting" and snap["progress_idle_s"] == 0.0
        assert build_health(manager, "t")["status"] == "ok"
        assert manager.check_maintenance()["progress_count"] == 19
        # The prompt finishes, the step runs at the safe point and completes: serving again.
        manager._note_progress(_progress(request_id, "executing"))
        manager._resolve_step(_reply(request_id))
        assert manager.maintenance_state == "serving" and manager.fatal_error is None
        assert manager.maintenance_op is None

    asyncio.run(run())


def test_a_slow_rebuild_that_reports_its_phases_survives_well_past_the_deadline():
    """A genuine large rebuild: each engine phase (teardown, layer moves, pools, capture) takes
    minutes here, far longer than measured, and the operation as a whole runs for an hour.
    The clock measures one silent phase at a time, so it never trips."""
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        task = asyncio.ensure_future(
            api.dispatch_rebuild(manager, moe_cache_size=3000, num_pages=2048, timeout=0.01)
        )
        result = await task
        assert result["status"] == "timeout" and manager.maintenance_state == "rebuilding"
        request_id = manager.sent[-1].request_id
        phases = ["queued", "executing", "rebuild:prepared", "rebuild:validated", "rebuild:teardown",
                  "rebuild:layer_move", "rebuild:layer_move", "rebuild:pools", "rebuild:capture",
                  "rebuild:captured", "rebuild:rethreaded"]
        for phase in phases:
            clock.now += api.MAINTENANCE_STUCK_S - 60.0  # each phase just inside the limit
            manager._note_progress(_progress(request_id, phase))
            assert manager.check_maintenance()["stuck"] is False
        assert manager.check_maintenance()["age_s"] > 3600.0
        from freetoken.message import CacheRebuildReply

        manager._resolve_rebuild(CacheRebuildReply(request_id=request_id, status="ok"))
        assert manager.maintenance_state == "serving" and manager.fatal_error is None

    asyncio.run(run())


def test_silence_after_progress_still_fails_finitely_and_names_the_phase():
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        request_id = await _timed_out_step(manager)
        clock.now += 100.0
        manager._note_progress(_progress(request_id, "executing"))
        clock.now += 100.0
        manager._note_progress(_progress(request_id, "rebuild:teardown"))
        clock.now += api.MAINTENANCE_STUCK_S - 1
        assert manager.check_maintenance()["stuck"] is False, "the clock restarted on the last phase"
        clock.now += 2
        snap = manager.check_maintenance()
        assert snap["stuck"] is True and manager.maintenance_state == "failed"
        assert "phase 'rebuild:teardown'" in manager.fatal_error
        assert "after 2 progress reports" in manager.fatal_error

    asyncio.run(run())


def test_progress_for_another_operation_does_not_extend_the_open_one():
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        request_id = await _timed_out_step(manager)
        for _ in range(7):
            clock.now += 100.0
            manager._note_progress(_progress("some-older-op", "waiting"))
        snap = manager.check_maintenance()
        assert snap["stuck"] is True and snap["progress_count"] == 0
        assert manager.maintenance_state == "failed"
        assert request_id in manager.fatal_error

    asyncio.run(run())


def test_the_listener_routes_progress_replies_to_the_open_operation():
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        manager.recv_tokenizer = asyncio.Queue()
        listener = asyncio.ensure_future(manager.listen())
        try:
            request_id = await _timed_out_step(manager)
            clock.now += 50.0
            await manager.recv_tokenizer.put(_progress(request_id, "waiting", "drained 2 decode"))
            for _ in range(5):
                await asyncio.sleep(0)
            snap = manager.check_maintenance()
            assert snap["phase"] == "waiting" and snap["detail"] == "drained 2 decode"
            assert snap["progress_idle_s"] == 0.0 and snap["backend_idle_s"] == 0.0
        finally:
            listener.cancel()

    asyncio.run(run())


def test_backend_activity_keeps_a_slow_operation_alive():
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        request_id = await _timed_out_step(manager)
        for _ in range(5):  # tokens of running requests keep arriving: queued, not stuck
            clock.now += api.MAINTENANCE_STUCK_S - 10
            manager.backend_last_seen = clock.now
            assert manager.check_maintenance()["stuck"] is False
            assert manager.maintenance_state == "rebuilding"
        # The late but valid completion clears its own operation.
        manager._resolve_step(_reply(request_id))
        assert manager.maintenance_state == "serving" and manager.maintenance_op is None
        assert manager.check_maintenance()["operation"] is None

    asyncio.run(run())


def test_a_short_operation_never_trips_the_limit():
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        task = asyncio.ensure_future(dispatch_step(manager, axis="ram", direction="up", timeout=5.0))
        await asyncio.sleep(0)
        clock.now += 20.0
        manager._resolve_step(_reply(manager.sent[-1].request_id))
        result = await task
        assert result["status"] == "ok" and manager.maintenance_state == "serving"
        assert manager.fatal_error is None and manager.check_maintenance()["stuck"] is False

    asyncio.run(run())


def test_a_stale_reply_cannot_reopen_the_gate_under_a_newer_operation():
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        first = await _timed_out_step(manager)
        manager.maintenance_state = "serving"  # the operator's next request follows a recovery
        manager.maintenance_op = None
        second = await _timed_out_step(manager)
        manager._resolve_step(_reply(first))  # the old reply lands while the new one executes
        assert manager.maintenance_state == "rebuilding"
        assert manager.maintenance_op["request_id"] == second
        manager._resolve_step(_reply(second))
        assert manager.maintenance_state == "serving" and manager.maintenance_op is None

    asyncio.run(run())


def test_a_failed_reply_still_latches_and_a_dispatch_error_leaves_no_operation():
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        request_id = await _timed_out_step(manager)
        manager._resolve_step(_reply(request_id, status="failed"))
        assert manager.maintenance_state == "failed"

        broken = _manager(clock)

        async def boom(_msg):
            raise RuntimeError("zmq push failed")

        broken.send_one = boom
        result = await dispatch_step(broken, axis="ram", direction="up", timeout=1.0)
        assert result["status"] == "failed" and broken.maintenance_state == "serving"
        assert broken.maintenance_op is None and broken.check_maintenance()["operation"] is None

    asyncio.run(run())


def test_cache_status_exposes_the_operation_and_its_age():
    clock = _Clock()

    async def run():
        manager = _manager(clock)
        prev = api._GLOBAL_STATE
        api._GLOBAL_STATE = manager
        try:
            manager._create_listener_once = lambda: None
            request_id = await _timed_out_step(manager)
            clock.now += 42.0
            doc = await api.cache_status()
            assert doc["state"] == "rebuilding"
            assert doc["maintenance"]["operation"]["request_id"] == request_id
            assert doc["maintenance"]["age_s"] == 42.0 and doc["maintenance"]["backend_idle_s"] == 42.0
            assert doc["maintenance"]["deadline_s"] == api.MAINTENANCE_STUCK_S
            assert doc["maintenance"]["phase"] == "dispatched"
            assert doc["maintenance"]["progress_idle_s"] == 42.0 and doc["maintenance"]["progress_count"] == 0
            health = build_health(manager, "t")
            assert health["maintenance_age_s"] == 42.0 and health["maintenance_phase"] == "dispatched"
            assert health["maintenance_progress_idle_s"] == 42.0
        finally:
            api._GLOBAL_STATE = prev

    asyncio.run(run())

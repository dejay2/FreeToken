"""Rule 9 of the dynamic KV pool spec: one maintenance record per operation id. The race from
the 2026-09-12 review: a manual rebuild is dispatched (record opened at dispatch), the
scheduler starts an automatic operation before receiving it, and the automatic completion
must not reopen the gate while the manual one is outstanding."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import freetoken.server.api_server as api
from freetoken.message import (
    CacheProgressReply, CacheRebuildReply, KVDynamicStatusReply, MaintenanceBeginReply,
)
from freetoken.server.api_server import FrontendManager, dispatch_rebuild


class _Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _manager(clock):
    m = FrontendManager(
        config=SimpleNamespace(served_model_name="m", kv_park="ram", kv_dtype="fp8"),
        send_tokenizer=None, recv_tokenizer=None, maintenance_state="serving", monotonic=clock,
    )
    m.sent = []

    async def send_one(msg):
        m.sent.append(msg)

    m.send_one = send_one
    return m


def _rebuild_reply(request_id, status="ok"):
    return CacheRebuildReply(request_id=request_id, status=status, moe_cache_size=7000,
                             num_pages=1537, mamba_slots=12, num_swa_pages=0, error=None)


def test_automatic_completion_does_not_reopen_the_gate_under_a_dispatched_manual_operation():
    clock = _Clock()
    m = _manager(clock)

    async def run():
        task = asyncio.ensure_future(dispatch_rebuild(m, moe_cache_size=7000, num_pages=1537, timeout=0.05))
        await asyncio.sleep(0)                       # dispatch has opened M and sent it
        manual_id = m.sent[-1].request_id
        m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:1", kind="auto-kv", detail="grow"))
        assert set(m.maintenance_ops) == {manual_id, "auto-kv:i:1"}
        m._note_progress(CacheProgressReply(request_id="auto-kv:i:1", phase="rebuild:pools"))
        assert m.maintenance_ops["auto-kv:i:1"]["phase"] == "rebuild:pools"
        m._resolve_rebuild(_rebuild_reply("auto-kv:i:1"))
        assert m.maintenance_state == "rebuilding"        # M still outstanding
        assert not m.rebuild_done.is_set()
        assert set(m.maintenance_ops) == {manual_id}
        m._note_progress(CacheProgressReply(request_id=manual_id, phase="executing"))
        assert m.maintenance_ops[manual_id]["phase"] == "executing"   # not ignored
        m._resolve_rebuild(_rebuild_reply(manual_id))
        assert m.maintenance_state == "serving" and m.rebuild_done.is_set()
        assert m.maintenance_ops == {}
        result = await task
        assert result["status"] == "ok"

    asyncio.run(run())


def test_failed_automatic_operation_latches_failed_even_with_others_open():
    m = _manager(_Clock())
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:2", kind="auto-kv"))
    m._resolve_rebuild(_rebuild_reply("auto-kv:i:2", status="failed"))
    assert m.maintenance_state == "failed" and m.rebuild_done.is_set()
    # A failed latch clears every record, not just the one that failed: an orphaned record
    # from some other (never-closed) automatic operation must not survive to pin the gate
    # closed after a recovery/restart (2026-09-12 review round 1).
    assert m.maintenance_ops == {}


def test_begin_maintenance_does_not_resurrect_a_fatal_latch():
    """A buffered MaintenanceBeginReply that raced a crash must not flip a fatally-latched
    engine back to "rebuilding" and clear rebuild_done -- the same reason _resolve_rebuild
    leaves a fatal latch alone (2026-09-12 review round 1, reproduced by the reviewer)."""
    m = _manager(_Clock())
    m.fatal_error = "dead"
    m.maintenance_state = "failed"
    m.rebuild_done.set()
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:6", kind="auto-kv"))
    assert m.maintenance_state == "failed"
    assert m.rebuild_done.is_set()
    assert m.maintenance_ops == {}


def test_begin_maintenance_is_idempotent_for_an_id_already_open():
    m = _manager(_Clock())
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:7", kind="auto-kv", detail="grow"))
    m._note_progress(CacheProgressReply(request_id="auto-kv:i:7", phase="rebuild:pools"))
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:7", kind="auto-kv", detail="grow more"))
    # A duplicate/retried begin only refreshes detail -- it must not reset the clock or phase.
    assert m.maintenance_ops["auto-kv:i:7"]["phase"] == "rebuild:pools"
    assert m.maintenance_ops["auto-kv:i:7"]["detail"] == "grow more"
    assert len(m.maintenance_ops) == 1


def test_stale_reply_for_an_unknown_operation_changes_nothing():
    m = _manager(_Clock())
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:3", kind="auto-kv"))
    m._resolve_rebuild(_rebuild_reply("gone"))
    assert m.maintenance_state == "rebuilding" and "auto-kv:i:3" in m.maintenance_ops


def test_watchdog_times_the_oldest_silent_operation():
    clock = _Clock()
    m = _manager(clock)
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:4", kind="auto-kv"))
    clock.now += 5
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:5", kind="auto-kv"))
    snap = m.check_maintenance()
    assert snap["operation"]["request_id"] == "auto-kv:i:4" and snap["age_s"] == 5.0


def test_kv_dynamic_status_snapshot_is_kept_for_cache_status():
    m = _manager(_Clock())
    m._note_kv_dynamic(KVDynamicStatusReply(status={"enabled": True, "held": 1}))
    assert m.kv_dynamic_status == {"enabled": True, "held": 1}


def test_cache_status_exposes_kv_dynamic_and_open_operations():
    m = _manager(_Clock())
    m._begin_maintenance(MaintenanceBeginReply(request_id="auto-kv:i:8", kind="auto-kv"))
    m._note_kv_dynamic(KVDynamicStatusReply(status={"enabled": True, "held": 3}))

    async def run():
        prev = api._GLOBAL_STATE
        api._GLOBAL_STATE = m
        try:
            m._create_listener_once = lambda: None
            return await api.cache_status()
        finally:
            api._GLOBAL_STATE = prev

    body = asyncio.run(run())
    assert body["kv_dynamic"] == {"enabled": True, "held": 3}
    assert body["maintenance"]["open_operations"] == ["auto-kv:i:8"]

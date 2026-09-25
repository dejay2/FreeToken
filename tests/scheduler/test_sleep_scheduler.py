"""The scheduler's side of sleep: the safe point, held chats and auto-wake (review focus 3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine.sleep import SleepRefused, WakeFailed
from freetoken.message import (
    AbortBackendMsg,
    CacheProgressMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    CacheSleepBackendMsg,
    CacheSleepResultMsg,
    CacheStepBackendMsg,
    CacheStepResultMsg,
    ErrorReplyMsg,
    MaintenanceBeginMsg,
    UserMsg,
)
from freetoken.scheduler.scheduler import Scheduler

OK = {"released_bytes": 24 << 30, "vram_free_bytes": 23 << 30, "elapsed_s": 6.5, "note": None}


class FakeEngine:
    def __init__(self, asleep=False):
        self.sleep_snapshot = object() if asleep else None
        self.calls: list[str] = []
        self.num_pages, self.page_table, self.max_seq_len = 100, "table", 6400
        self.rebuild_teardown_started = False
        self.encoder_cache = None
        self.refuse = None
        self.explode = None
        self.wake_error = None  # raised AFTER the wake's point of no return

    def sleep_preflight(self):
        self.calls.append("preflight")
        if self.refuse:
            raise SleepRefused(self.refuse)

    def sleep(self):
        self.calls.append("sleep")
        self.rebuild_teardown_started = True
        if self.explode:
            raise RuntimeError(self.explode)
        self.sleep_snapshot, self.num_pages = object(), 1
        return {"asleep": True, **OK}

    def wake(self):
        self.calls.append("wake")
        if self.refuse:
            raise SleepRefused(self.refuse)
        self.rebuild_teardown_started = True
        if self.wake_error is not None:
            # wake_engine went back to sleep (SleepRefused) or could not (WakeFailed); either
            # way the pools were re-made at the sleep size and the snapshot is kept.
            self.num_pages = 1
            raise self.wake_error
        self.sleep_snapshot, self.num_pages = None, 100
        return {"asleep": False, **OK}

    def step_memory(self, **kwargs):
        self.calls.append(("step", kwargs["rebuild"]))
        return {"applied": "pinned->disk", "layer": 3, "moe_cache_size": 0}

    def asleep_rebuild(self, **kwargs):
        pass

    def residency_report(self):
        return {}


class FakeCacheManager:
    supports_runtime_rebuild = True
    prefill_chunk_budget = None

    def __init__(self, parks=False):
        self.park_store = object() if parks else None
        self.calls: list = []

    def prepare_rebuild(self):
        self.calls.append("park")

    def rebuild(self, num_pages, page_table):
        self.calls.append(("rebuild", num_pages, page_table))

    def check_integrity(self):
        self.calls.append("check")


def shell(*, asleep=False, prefill=False, decode=False, parks=False):
    s = Scheduler.__new__(Scheduler)
    s.sent = []
    s.send_result = s.sent.extend
    s.engine = FakeEngine(asleep)
    s.cache_manager = FakeCacheManager(parks)
    s.table_manager = SimpleNamespace(rebuild=lambda table: None, token_pool="pool")
    s.prefill_manager = SimpleNamespace(runnable=prefill, pending_list=[], abort_req=lambda uid: None)
    s.decode_manager = SimpleNamespace(runnable=decode, abort_req=lambda uid: None)
    s.config = SimpleNamespace(max_extend_tokens=8192, tp_info=SimpleNamespace(size=1))
    s.device = torch.device("cpu")
    s._pending_rebuild = None
    s._engine_failed = None
    s._kv_dynamic = None
    s._maintenance_request_id = None
    s._maintenance_progress_at = -float("inf")
    s._abort_tombstones = {}
    s._pending_abort_acks = set()
    s._last_data = None
    s._idle_wait_logged = False
    s._sleep_held = []
    s._auto_wake_seq = 0
    s.admitted = []
    s._admit_user_msg = s.admitted.append
    s._queue_for_kv_dynamic = lambda msg: False
    s._log_cache_geometry = lambda event: None
    s._drop_raw_picture = lambda msg: None
    s._release_request_tensors = lambda msg: None
    s._send_kv_dynamic_status = lambda: None
    return s


def sleep_results(s):
    return [m for m in s.sent if isinstance(m, CacheSleepResultMsg)]


def chat(uid=7):
    return UserMsg(uid=uid, input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), sampling_params=SamplingParams())


def run(s, msg):
    s._process_one_msg(msg)
    if s._pending_rebuild is not None:
        s._execute_pending_rebuild()


def test_sleep_is_refused_while_a_chat_runs():
    s = shell(decode=True)
    run(s, CacheSleepBackendMsg(request_id="r", action="sleep"))
    assert [(m.status, m.asleep) for m in sleep_results(s)] == [("busy", False)]
    assert s.engine.calls == []


def test_sleep_parks_conversations_then_sleeps_then_rethreads():
    s = shell(parks=True)
    run(s, CacheSleepBackendMsg(request_id="r", action="sleep"))
    assert s.engine.calls == ["preflight", "sleep"]
    assert s.cache_manager.calls == ["park", ("rebuild", 1, "table"), "check"]
    (reply,) = sleep_results(s)
    assert (reply.status, reply.asleep, reply.released_bytes) == ("ok", True, 24 << 30)


def test_a_refused_sleep_parks_nothing():
    s = shell(parks=True)
    s.engine.refuse = "layers [2] have no finished copy on the SSD yet"
    run(s, CacheSleepBackendMsg(request_id="r", action="sleep"))
    assert s.cache_manager.calls == [] and s.engine.calls == ["preflight"]
    assert [(m.status, m.asleep) for m in sleep_results(s)] == [("rejected", False)]


def test_a_sleep_that_fails_after_teardown_latches_failed():
    s = shell()
    s.engine.explode = "CUDA error: an illegal memory access was encountered"
    run(s, CacheSleepBackendMsg(request_id="r", action="sleep"))
    assert [m.status for m in sleep_results(s)] == ["failed"]
    assert s._engine_failed is not None


def test_a_chat_that_reaches_a_sleeping_scheduler_is_held_and_wakes_it():
    s = shell(asleep=True)
    s._process_one_msg(chat(7))
    assert [m.uid for m in s._sleep_held] == [7] and s.admitted == []
    begin = [m for m in s.sent if isinstance(m, MaintenanceBeginMsg)]
    assert [(m.request_id, m.kind) for m in begin] == [("auto-wake:1", "wake")]
    s._process_one_msg(chat(8))  # a second chat joins the same wake
    assert isinstance(s._pending_rebuild, CacheSleepBackendMsg) and s._auto_wake_seq == 1
    s._execute_pending_rebuild()
    assert [m.uid for m in s.admitted] == [7, 8] and s._sleep_held == []
    assert [(m.request_id, m.status, m.asleep) for m in sleep_results(s)] == [("auto-wake:1", "ok", False)]


def test_a_refused_auto_wake_answers_the_held_chats_in_plain_words():
    s = shell(asleep=True)
    s.engine.refuse = "the graphics card has 2.0 GB free and waking needs 24.5 GB; close the game"
    s._process_one_msg(chat(7))
    s._execute_pending_rebuild()
    errors = [m for m in s.sent if isinstance(m, ErrorReplyMsg)]
    assert [m.uid for m in errors] == [7] and "close the game" in errors[0].error
    assert [(m.status, m.asleep) for m in sleep_results(s)] == [("rejected", True)]
    assert s.admitted == [] and s._sleep_held == []


def test_an_abort_removes_a_held_chat():
    s = shell(asleep=True)
    s._process_one_msg(chat(7))
    s._process_one_msg(AbortBackendMsg(uid=7))
    s._execute_pending_rebuild()
    assert s.admitted == []


def test_a_manual_rebuild_is_refused_while_asleep():
    s = shell(asleep=True)
    s._current_cache_geometry = lambda: {"moe_cache_size": 0, "num_pages": 1, "num_mamba_slots": 0,
                                         "num_swa_pages": 0}
    s._process_one_msg(CacheRebuildBackendMsg(request_id="r", num_pages=10))
    (reply,) = [m for m in s.sent if isinstance(m, CacheRebuildResultMsg)]
    assert reply.status == "rejected" and "asleep" in reply.error


def test_only_a_ram_down_step_runs_while_asleep_and_it_uses_the_ssd_only_rebuild():
    s = shell(asleep=True)
    s._reply_step = lambda request_id, status, result=None, error=None: s.sent.append((status, error))
    assert s._execute_pending_step(CacheStepBackendMsg(request_id="a", axis="vram", direction="down")) == "rejected"
    assert s._execute_pending_step(CacheStepBackendMsg(request_id="b", axis="ram", direction="up")) == "rejected"
    assert s.engine.calls == []
    assert s._execute_pending_step(CacheStepBackendMsg(request_id="c", axis="ram", direction="down")) == "ok"
    assert s.engine.calls == [("step", s.engine.asleep_rebuild)]


def test_a_second_sleep_request_while_one_is_queued_is_busy():
    s = shell()
    s._process_one_msg(CacheSleepBackendMsg(request_id="r1", action="sleep"))
    s._process_one_msg(CacheSleepBackendMsg(request_id="r2", action="sleep"))
    assert [(m.request_id, m.status) for m in sleep_results(s)] == [("r2", "busy")]


def test_sleep_is_refused_at_once_while_a_chat_runs():
    """Design D11: a sleep sent mid-chat answers "busy" on arrival; it is never queued to wait
    for the decode to end (the loops hold a non-MoE-only operation until decode is idle)."""
    s = shell(decode=True)
    s._process_one_msg(CacheSleepBackendMsg(request_id="r", action="sleep"))
    assert s._pending_rebuild is None
    assert [(m.status, m.asleep) for m in sleep_results(s)] == [("busy", False)]


def test_a_chat_held_by_the_dynamic_pool_makes_sleep_busy():
    s = shell()
    s._kv_dynamic = SimpleNamespace(has_held=lambda: True, enabled=True)
    s._process_one_msg(CacheSleepBackendMsg(request_id="r", action="sleep"))
    assert [m.status for m in sleep_results(s)] == ["busy"] and s._pending_rebuild is None


def test_a_wake_that_went_back_to_sleep_rethreads_the_one_page_pool():
    """wake_engine's way back (release_to_sleep force_pools) re-makes the KV pool: the page
    managers must point at the new one-page table, or the next wake's pages are stale."""
    s = shell(asleep=True)
    s.engine.wake_error = SleepRefused("wake failed and the model went back to sleep: OOM")
    s._process_one_msg(chat(7))
    s._execute_pending_rebuild()
    assert ("rebuild", 1, "table") in s.cache_manager.calls
    assert [(m.status, m.asleep) for m in sleep_results(s)] == [("rejected", True)]
    assert [m.uid for m in s.sent if isinstance(m, ErrorReplyMsg)] == [7]
    assert s._engine_failed is None and s._sleep_held == []


def test_a_wake_that_cannot_go_back_to_sleep_latches_failed_and_answers_held_chats():
    s = shell(asleep=True)
    s.engine.wake_error = WakeFailed("wake failed (OOM) and going back to sleep failed too (OOM)")
    s._process_one_msg(chat(7))
    s._execute_pending_rebuild()
    assert [m.status for m in sleep_results(s)] == ["failed"]
    assert s._engine_failed is not None
    assert [m.uid for m in s.sent if isinstance(m, ErrorReplyMsg)] == [7]
    assert s.admitted == [] and s._sleep_held == []

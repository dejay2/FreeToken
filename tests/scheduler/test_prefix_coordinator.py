"""Real CPU page/state ownership for named, coalesced prefix preparation."""
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.prefixes import PrefixCoordinator
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq


def command(action, name="", **kwargs):
    return SimpleNamespace(action=action, name=name, input_ids=None, prefix_tokens=None,
                           ttl_seconds=300.0, max_retained_bytes=None, **kwargs)


def register(co, name="base", ids=None, length=None):
    msg = command("register", name)
    msg.input_ids = torch.arange(18, dtype=torch.int32) if ids is None else ids
    msg.prefix_tokens = length
    return co.command(msg)


@pytest.fixture
def setup():
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=24, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)
    table = torch.zeros(6, 128, dtype=torch.int32)
    cm = CacheManager(32, 4, table, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(6, table)
    pm = PrefillManager(cm, tm, DecodeManager(4))
    co = PrefixCoordinator(cm, pm.pending_list.append, max_seq_len=128,
                           kv_bytes_per_token=8, state_bytes=pool.bytes_per_slot())
    pm.prefix_coordinator = co
    # pending_list is replaced after each scheduling pass; bind through the manager.
    co.enqueue = lambda pending: pm.pending_list.append(pending)
    return co, pm, cm, tm, pool


def pending(uid, suffix=99, private=False):
    return PendingReq(uid, torch.cat((torch.arange(16, dtype=torch.int32),
                                    torch.tensor([suffix], dtype=torch.int32))),
                      SamplingParams(max_tokens=2), cache_private=private)


def finish_prefix(setup, budget=128):
    co, pm, cm, tm, pool = setup
    count = 0
    while True:
        batch = pm.schedule_next_batch(budget)
        assert batch is not None and all(r.prefill_only for r in batch.reqs)
        assert batch.prompt_admissions == []
        assert batch.size == 1
        req = batch.reqs[0]
        cm.allocate_paged(batch.reqs)
        req.complete_one()
        co.drained_chunk(req)
        count += 1
        assert not req.can_decode
        assert req.cached_len == req.device_len
        if isinstance(req, ChunkedReq):
            assert cm.prefix_cache.match_prefix(req.input_ids).cached_len == 0
            continue
        req.mamba_last_track_seqlen = None
        with cm.lazy_free_region():
            cm.cache_req(req, finished=True)
            tm.free(req.table_idx)
            req.table_idx = -1
            co.completed(req)
        cm.check_integrity()
        return count


def test_four_cold_agents_prepare_once_then_share_pages_and_private_state(setup):
    co, pm, cm, tm, pool = setup
    assert register(co)["status"] == "ok"
    assert all(co.route(pending(i, 90 + i)) for i in range(4))
    assert len(pm.pending_list) == 1
    assert finish_prefix(setup, budget=8) == 2
    batch = pm.schedule_next_batch(128)
    assert [r.uid for r in batch.reqs] == list(range(4))
    assert all(r.cached_len == 16 for r in batch.reqs)
    rows = [tm.page_table[r.table_idx, :16] for r in batch.reqs]
    assert all(torch.equal(rows[0], row) for row in rows[1:])
    assert len({r.linear_slot_idx for r in batch.reqs}) == 4
    assert len({r.mamba_restore_src for r in batch.reqs}) == 1
    cm.check_integrity()
    assert co.command(command("get", "base"))["result"]["preparations"] == 1
    assert co.command(command("get", "base"))["result"]["forwarded_tokens"] == 16


def test_aliases_deduplicate_source_and_cancel_only_one_follower(setup):
    co, pm, cm, tm, pool = setup
    assert register(co)["status"] == "ok"
    assert register(co, "other")["status"] == "ok"
    assert co.command(command("list"))["result"]["source_bytes"] == 16 * 4
    assert co.route(pending(1)) and co.route(pending(2))
    co.cancel_waiter(1)
    co.command(command("delete", "base"))
    finish_prefix(setup)
    assert [p.uid for p in pm.pending_list] == [2]
    assert co.command(command("get", "other"))["result"]["state"] == "ready"


def test_private_and_different_prefix_bypass_coalescing(setup):
    co, pm, *_ = setup
    register(co)
    assert not co.route(pending(1, private=True))
    other = pending(2)
    other.input_ids[0] = 777
    assert not co.route(other)
    assert not pm.pending_list


def test_failure_releases_followers_without_requeue_loop(setup):
    co, pm, *_ = setup
    register(co)
    co.route(pending(1))
    job = pm.pending_list.pop()
    assert co.failed(job.uid, "pool too small")
    assert [p.uid for p in pm.pending_list] == [1]
    assert co.command(command("get", "base"))["result"]["state"] == "error"


def test_rebuild_releases_leases_and_rewarms_from_sources(setup):
    co, pm, cm, tm, pool = setup
    register(co)
    assert co.command(command("warm", "base"))["status"] == "warming"
    finish_prefix(setup)
    assert co.command(command("list"))["result"]["retained_bytes"] > 0
    co.before_rebuild()
    cm.rebuild(32, tm.page_table)
    assert co.command(command("get", "base"))["result"]["state"] == "evicted"
    assert pool.num_free_slots == pool.num_slots - 1
    assert co.route(pending(8))
    finish_prefix(setup)
    cm.check_integrity()


def test_warm_resident_endpoint_does_not_create_zero_length_forward(setup):
    co, pm, *_ = setup
    register(co)
    co.command(command("warm", "base"))
    finish_prefix(setup)
    assert co.command(command("warm", "base"))["status"] == "ok"
    assert not pm.pending_list


def test_registration_is_bounded_and_rejects_invalid_boundaries(setup):
    co, *_ = setup
    assert register(co, length=3)["status"] == "invalid"
    assert register(co, length=19)["status"] == "invalid"
    assert register(co, length=17)["result"]["prefix_tokens"] == 16
    assert register(co, ids=torch.arange(18, dtype=torch.int32) + 10)["status"] == "invalid"
    msg = command("register", "nan")
    msg.input_ids = torch.arange(16, dtype=torch.int32)
    msg.ttl_seconds = float("nan")
    assert co.command(msg)["status"] == "invalid"


def test_scheduler_drains_prefix_without_user_reply_and_frees_request(setup):
    from freetoken.scheduler.scheduler import Scheduler

    co, pm, cm, tm, pool = setup
    register(co)
    co.route(pending(10))
    batch = pm.schedule_next_batch(128)
    cm.allocate_paged(batch.reqs)
    req = batch.reqs[0]
    req.complete_one()
    events = []
    stub = SimpleNamespace(
        cache_manager=cm, table_manager=tm, prefix_coordinator=co,
        decode_manager=pm.decode_manager, finished_reqs=set(),
        _spec_record_plain=lambda batch: None,
        _ship_replies=lambda batch, reply, **kw: events.append((reply, kw)),
        _emit_step_tokens=lambda *a: pytest.fail("prefix preparation emitted a token"),
    )
    stub._free_req_resources = lambda r: Scheduler._free_req_resources(stub, r)
    data = (SimpleNamespace(batch=batch), (torch.empty(0), torch.empty(0),
                                          SimpleNamespace(synchronize=lambda: events.append("drained"))))
    Scheduler._process_last_data(stub, data)
    assert events == ["drained", ([], {"generated_tokens": 0})]
    assert req.table_idx == -1
    assert [p.uid for p in pm.pending_list] == [10]
    assert tm.available_size == 6
    assert pool.num_free_slots == pool.num_slots - 2  # padding plus one tree snapshot
    cm.check_integrity()


def test_engine_prefill_only_skips_sampling_and_speculation(monkeypatch):
    from contextlib import nullcontext
    from freetoken.core import Batch, Req
    from freetoken.engine.engine import Engine

    req = Req(torch.arange(4, dtype=torch.int32), 0, 0, 0, -1,
              SamplingParams(max_tokens=0), None, prefill_only=True)
    batch = Batch([req], "prefill")
    stream = object()
    events = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda s: events.append(s)))
    engine = SimpleNamespace(
        stream=stream, device=torch.device("cpu"), mtp_shadow_observer=object(), spec_draft=object(),
        graph_runner=SimpleNamespace(can_use_cuda_graph=lambda b: False),
        ctx=SimpleNamespace(forward_batch=lambda b: nullcontext()),
        model=SimpleNamespace(forward_host_ctx=lambda *a: nullcontext(), forward=lambda: torch.zeros(1, 32)),
        cpu_moe_executor=None,
        sampler=SimpleNamespace(sample=lambda *a: pytest.fail("preparation must not sample")),
    )
    out = Engine.forward_batch(engine, batch, None)
    assert out.next_tokens_cpu.numel() == out.next_tokens_gpu.numel() == 0
    assert req.device_len == req.cached_len == 4
    assert req.input_ids.tolist() == [0, 1, 2, 3]
    assert events == [stream]


def test_parent_child_presets_count_shared_pages_once(setup):
    co, pm, cm, *_ = setup
    register(co, "base", length=8)
    co.command(command("warm", "base"))
    finish_prefix(setup)
    register(co, "skill", length=16)
    co.command(command("warm", "skill"))
    finish_prefix(setup)
    summary = co.command(command("list"))["result"]
    assert summary["gpu_unique_pages"] == summary["retained_pages"] == 4
    assert summary["gpu_unique_snapshots"] == 2
    assert co.command(command("get", "skill"))["result"]["forwarded_tokens"] == 8
    cm.check_integrity()


def test_new_parent_snapshot_cannot_push_existing_child_preference_over_budget(setup):
    co, pm, cm, *_ = setup
    register(co, "child", length=16)
    co.command(command("warm", "child"))
    finish_prefix(setup)
    co.max_retained_bytes = 16 * co.kv_bytes_per_token + co.state_bytes
    register(co, "parent", length=8)
    co.command(command("warm", "parent"))
    finish_prefix(setup)
    assert co.command(command("list"))["result"]["retained_bytes"] <= co.max_retained_bytes
    cm.check_integrity()


def test_ttl_expiry_and_pressure_do_not_unlock_active_requests(setup):
    co, pm, cm, *_ = setup
    now = [100.0]
    co.clock = lambda: now[0]
    register(co)
    co.route(pending(1))
    finish_prefix(setup)
    batch = pm.schedule_next_batch(128)
    live = batch.reqs[0]
    now[0] += 301
    co.expire()
    assert co.command(command("list"))["result"]["retained_bytes"] == 0
    assert live.cache_handle.node.ref_count == 1
    assert not co.release_preferences()
    assert live.cache_handle.node.ref_count == 1
    cm.check_integrity()


def test_deleted_inflight_preset_finishes_and_releases_its_source(setup):
    co, pm, cm, *_ = setup
    register(co)
    co.route(pending(1))
    co.command(command("delete", "base"))
    co.cancel_waiter(1)
    finish_prefix(setup)
    assert not pm.pending_list
    assert co.command(command("list"))["result"]["source_bytes"] == 0
    assert cm.prefix_cache.full_evictable_size == 16
    cm.check_integrity()


def test_unrelated_request_can_run_while_preparation_is_queued(setup):
    co, pm, *_ = setup
    unrelated = pending(100)
    unrelated.input_ids[0] = 77
    register(co)
    assert not co.route(unrelated)
    pm.pending_list.append(unrelated)
    co.route(pending(1))
    first = pm.schedule_next_batch(128)
    assert [r.uid for r in first.reqs] == [100]
    assert len(pm.pending_list) == 1 and pm.pending_list[0].prefill_only


def test_unrelated_arrival_after_preparation_runs_between_chunks(setup):
    co, pm, cm, *_ = setup
    register(co)
    co.route(pending(1))
    unrelated = pending(100)
    unrelated.input_ids[0] = 77
    pm.pending_list.append(unrelated)
    first = pm.schedule_next_batch(8)
    cm.allocate_paged(first.reqs)
    first.reqs[0].complete_one()
    co.drained_chunk(first.reqs[0])
    assert isinstance(first.reqs[0], ChunkedReq) and first.prefill_only
    second = pm.schedule_next_batch(128)
    assert [r.uid for r in second.reqs] == [100]
    assert pm.pending_list[0].prefill_only


def test_table_blocked_arrival_cannot_block_preparation_continuation(setup):
    co, pm, cm, tm, _ = setup
    register(co)
    co.route(pending(1))
    first = pm.schedule_next_batch(8)
    cm.allocate_paged(first.reqs)
    first.reqs[0].complete_one()
    co.drained_chunk(first.reqs[0])
    while tm.available_size:
        tm.allocate()
    # Reproduce the queue order after yielding a quantum under complete table pressure.
    pm.pending_list.insert(0, pending(100))
    second = pm.schedule_next_batch(8)
    assert second is not None and second.prefill_only
    assert second.reqs[0].extend_len == 8
    assert [r.uid for r in pm.pending_list] == [100]


def test_yielding_preparation_reserves_capacity_for_its_unfinished_tokens(setup):
    co, pm, cm, *_ = setup
    register(co, ids=torch.arange(112, dtype=torch.int32))
    co.command(command("warm", "base"))
    first = pm.schedule_next_batch(8)
    cm.allocate_paged(first.reqs)
    first.reqs[0].complete_one()
    co.drained_chunk(first.reqs[0])
    unrelated = PendingReq(100, torch.arange(64, dtype=torch.int32) + 1000, SamplingParams(max_tokens=2))
    pm.pending_list.insert(0, unrelated)
    # 104 future preparation tokens plus this request's 66 tokens cannot fit the
    # 120 currently available tokens, even though either alone passes the size gate.
    second = pm.schedule_next_batch(8)
    assert second is not None and second.prefill_only
    assert pm.pending_list[0].uid == 100


@pytest.mark.parametrize("blocker", ["table", "token_budget"])
def test_non_cache_admission_failure_preserves_retention(setup, blocker):
    co, pm, cm, tm, _ = setup
    register(co)
    co.command(command("warm", "base"))
    finish_prefix(setup)
    if blocker == "table":
        while tm.available_size:
            tm.allocate()
    pm.pending_list.append(pending(5))
    assert pm.schedule_next_batch(0 if blocker == "token_budget" else 128) is None
    assert co.command(command("get", "base"))["result"]["retained"] is True


@pytest.mark.parametrize("currency", ["kv", "state"])
def test_already_admitted_work_can_reclaim_preferences_at_allocation(setup, currency):
    co, pm, cm, tm, pool = setup
    register(co)
    co.command(command("warm", "base"))
    finish_prefix(setup)
    # A control warm can arrive after admission reserved a request's future capacity.
    # The eventual allocation must still be able to revoke that optional preference.
    if currency == "kv":
        pages = cm._allocate(29)
        assert len(pages) == 29
        cm._free(cm._page_to_token(pages))
    else:
        occupied = pool.alloc(pool.num_free_slots)
        cm.ensure_mamba_slots(1)
        assert pool.num_free_slots >= 1
        pool.free(occupied)
    assert not co.command(command("get", "base"))["result"]["retained"]
    cm.check_integrity()


def test_status_poll_does_not_change_gpu_eviction_recency(setup):
    co, pm, cm, *_ = setup
    register(co)
    co.command(command("warm", "base"))
    finish_prefix(setup)
    node = next(iter(cm.prefix_cache.root.children.values()))
    node.timestamp = 1234
    co.command(command("get", "base"))
    co.command(command("list"))
    assert node.timestamp == 1234


def test_scheduler_gives_decode_a_turn_after_a_preparation_quantum():
    from freetoken.core import Batch, Req
    from freetoken.scheduler.scheduler import Scheduler

    normal = Req(torch.arange(2, dtype=torch.int32), 0, 1, 5, 1, SamplingParams(), None)
    decode = Batch([normal], "decode")
    prepared = Req(torch.arange(4, dtype=torch.int32), 1, 0, 0, -1,
                   SamplingParams(max_tokens=0), None, prefill_only=True)
    prefill = Batch([prepared], "prefill")
    calls = []
    stub = SimpleNamespace(
        _last_batch_was_prefix_preparation=True, prefill_budget=32,
        prefill_manager=SimpleNamespace(schedule_next_batch=lambda budget: calls.append("prefill") or prefill),
        decode_manager=SimpleNamespace(schedule_next_batch=lambda: calls.append("decode") or decode),
        _prepare_batch=lambda batch: batch, _report_prompt_admissions=lambda batch: None,
    )
    assert Scheduler._schedule_next_batch(stub) is decode
    assert calls == ["decode"]
    assert Scheduler._schedule_next_batch(stub) is prefill
    assert calls == ["decode", "prefill"]


def test_supported_registry_is_explicit_and_source_budget_is_enforced(setup):
    co, pm, *_ = setup
    co.MAX_SOURCE_BYTES = 60
    assert register(co)["status"] == "busy"
    co.supported = False
    assert register(co)["status"] == "unsupported"
    assert not co.route(pending(1))
    assert not pm.pending_list

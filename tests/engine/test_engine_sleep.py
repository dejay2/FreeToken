"""Engine sleep and wake on the CPU harness (review focus 1, 2 and 4).

A real OffloadMoeCache, a real ExpertDiskCopy and the real Engine._move_layer: the only fakes
are the KV / GDN pools, the graph runner and the memory probe, which reports "free VRAM" from
what the harness holds on the "card" so released/needed byte counts are exact.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import Engine
from freetoken.engine.sleep import MAX_FAILED_WAKES, WAKE_MARGIN_BYTES, SleepRefused, WakeFailed
from tests.moe.test_disk_banks import FakeDiskEngine

EXPERT = (32 * 16 + 16 * 32) * 2  # bf16 bytes of one expert row across both banks
LAYER = 64 * EXPERT
PAGE, SLOT = 100, 1000  # "bytes" the harness charges per KV page and per GDN slot


class FakeKV:
    needs_rebind_on_rebuild = False

    def __init__(self):
        self.rebuilds: list[int] = []

    def rebuild_from_config(self, config, num_pages, *, num_swa_pages=None):
        self.rebuilds.append(num_pages)

    def attach_page_table(self, table):
        self.table = table


class FakeLinear:
    def __init__(self, n):
        self._n, self.rebuilds = n, []

    @property
    def num_slots(self):
        return self._n

    def rebuild(self, n):
        self.rebuilds.append(n)
        self._n = n


class FakeGraphs:
    def __init__(self, bs=(1,)):
        self.graph_bs_list, self.destroyed = list(bs), 0

    def destroy_cuda_graphs(self):
        self.destroyed += 1


class SleepEngine(FakeDiskEngine):
    TOTAL = 1 << 30

    def __init__(self, tmp_path, *, owned=(0, 2), cache_size=256, overlap=False):
        super().__init__(num_layers=4, num_experts=64, cache_size=cache_size, owned_layers=owned,
                         model_dir=tmp_path / "model", disk_dir=tmp_path / "disk", overlap=overlap)
        for lid in range(4):
            self.expert_disk_copy.write_layer(
                lid, {n: self.moe_offload_cache.bank_sources[n][lid] for n in self.bank_schema})
        self.original = {lid: {n: self.moe_offload_cache.bank_sources[n][lid].clone()
                               for n in self.bank_schema} for lid in range(4)}
        self.config.spec_decode = SimpleNamespace(enabled=False, graph_widths=())
        self.config.attention_backend = "triton"
        self.kv_cache = FakeKV()
        self._pool_cls = SimpleNamespace(min_kv_tokens=lambda config: config.page_size)
        self.linear_state_pool = FakeLinear(9)
        self.spec_state_ladder = None
        self.graph_runner = FakeGraphs()
        self.attn_backend = SimpleNamespace(reset_capture=lambda: None)
        self.spec_graph_runner = self.spec_draft = self.mtp_shadow_observer = None
        self._spec_graph_widths = ()
        self._graphs_deferred = None
        self._ram_parked_layers = []
        self.model = SimpleNamespace(mark_for_rebind=lambda: None)
        self.max_seq_len = 1024
        self.page_table = torch.zeros((5, 1024), dtype=torch.int32)
        self.ctx = SimpleNamespace(page_table=self.page_table)
        self.dummy_req = SimpleNamespace(table_idx=4)
        self.game_bytes = 0
        self.recaptures: list = []
        self.warmups = 0
        self.warmup_lengths: list = []
        self.spec_captures = 0
        self.budget_snapshots = 0
        self.sleep_snapshot = None

    def _sync_get_memory(self):
        cache = self.moe_offload_cache
        used = (cache.cache_size * EXPERT + len(cache.gpu_owned_layer_ids) * LAYER
                + self.num_pages * PAGE + self.linear_state_pool.num_slots * SLOT)
        free = self.TOTAL - used - self.game_bytes
        return free, free

    def _recapture_graphs(self, config, graph_bs, free_min):
        disk = self.moe_offload_cache.has_disk_layers
        self.recaptures.append((list(graph_bs), disk))
        self.graph_runner = FakeGraphs(() if disk else graph_bs)
        self._graphs_deferred = "deferred until no disk layers" if disk else None

    def _warmup_prefill(self, **kwargs):
        self.warmups += 1
        self.warmup_lengths.append(tuple(kwargs.get("lengths", ())))

    def _capture_spec_graphs_at_boot(self):
        self.spec_captures += 1

    def snapshot_pool_budget(self):
        self.budget_snapshots += 1

    sleep_preflight = Engine.sleep_preflight
    sleep = Engine.sleep
    wake = Engine.wake
    asleep_rebuild = Engine.asleep_rebuild
    _resize_kv_pool = Engine._resize_kv_pool
    _refresh_seq_state = Engine._refresh_seq_state
    _graph_bs_for_recapture = Engine._graph_bs_for_recapture
    _rearm_spec_graphs = Engine._rearm_spec_graphs
    _boot_warmup_lengths = Engine._boot_warmup_lengths


def residency(eng):
    return list(eng.moe_offload_cache.layer_residency)


def test_sleep_releases_the_card_and_keeps_the_host_banks(tmp_path):
    eng = SleepEngine(tmp_path)
    pinned = {lid: eng.moe_offload_cache.bank_sources["gate_up"][lid] for lid in (1, 3)}
    rep = eng.sleep()
    cache = eng.moe_offload_cache
    assert rep["asleep"] is True and rep["note"] is None
    assert rep["released_bytes"] == 256 * EXPERT + 2 * LAYER + 99 * PAGE + 8 * SLOT
    assert cache.cache_size == 0 and cache.bank_caches == {}
    assert residency(eng) == ["disk", "pinned", "disk", "pinned"]
    assert eng.num_pages == 1 and eng.kv_cache.rebuilds == [1]
    assert eng.linear_state_pool.num_slots == 1
    assert eng.graph_runner.destroyed == 1
    assert eng.max_seq_len == 16  # one 16-token page
    for lid, bank in pinned.items():
        assert eng.moe_offload_cache.bank_sources["gate_up"][lid] is bank


def test_wake_restores_the_geometry_and_the_owned_banks_byte_for_byte(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    rep = eng.wake()
    cache = eng.moe_offload_cache
    assert rep["asleep"] is False and eng.sleep_snapshot is None
    assert (cache.cache_size, eng.num_pages, eng.linear_state_pool.num_slots) == (256, 100, 9)
    assert residency(eng) == ["gpu_owned", "pinned", "gpu_owned", "pinned"]
    for lid in (0, 2):
        for name in eng.bank_schema:
            assert torch.equal(cache.bank_sources[name][lid], eng.original[lid][name])
    assert eng.recaptures == [([1], False)]
    assert eng.warmups == 1 and eng.budget_snapshots == 1
    assert eng._ram_spilled_layers == [] and eng.max_seq_len == 1024


def test_ram_parked_layers_are_still_parked_after_a_wake(tmp_path):
    eng = SleepEngine(tmp_path)
    eng._ram_parked_layers = [2]
    eng.sleep()
    assert eng._ram_parked_layers == []  # the move to the SSD un-parks it...
    eng.wake()
    assert eng._ram_parked_layers == [2]  # ...and the wake puts it back


def test_prefill_overlap_is_off_asleep_and_back_awake(tmp_path):
    eng = SleepEngine(tmp_path, overlap=True)
    eng.sleep()
    assert not eng.moe_offload_cache.prefill_overlap
    eng.wake()
    assert eng.moe_offload_cache.prefill_overlap
    assert len(eng.moe_offload_cache.prefill_bank_buffers) == 2


def test_sleep_is_refused_before_anything_is_freed_without_an_ssd_copy(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.expert_disk_copy.manifest["complete"].pop("2")
    with pytest.raises(SleepRefused, match="no finished copy on the SSD"):
        eng.sleep()
    assert eng.moe_offload_cache.cache_size == 256 and eng.graph_runner.destroyed == 0
    assert eng.sleep_snapshot is None and residency(eng)[2] == "gpu_owned"


def test_sleep_is_refused_while_the_shadow_observer_runs(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.mtp_shadow_observer = object()
    with pytest.raises(SleepRefused, match="shadow"):
        eng.sleep()
    assert eng.graph_runner.destroyed == 0


def test_wake_is_refused_before_touching_anything_while_a_game_holds_the_card(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    eng.game_bytes = eng.sleep_snapshot.free_after - WAKE_MARGIN_BYTES  # a game takes the card
    with pytest.raises(SleepRefused, match="close the game"):
        eng.wake()
    assert eng.sleep_snapshot is not None and eng.moe_offload_cache.cache_size == 0
    assert residency(eng) == ["disk", "pinned", "disk", "pinned"] and eng.recaptures == []
    eng.game_bytes = 0  # the game quits: the next wake works
    assert eng.wake()["asleep"] is False


def test_a_wake_that_fails_midway_goes_back_to_sleep_and_is_refused(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    eng._recapture_graphs = oom
    with pytest.raises(SleepRefused, match="went back to sleep"):
        eng.wake()
    assert eng.sleep_snapshot is not None
    assert eng.moe_offload_cache.cache_size == 0 and eng.num_pages == 1
    assert residency(eng) == ["disk", "pinned", "disk", "pinned"]
    del eng._recapture_graphs  # back to the class method
    assert eng.wake()["asleep"] is False
    assert residency(eng) == ["gpu_owned", "pinned", "gpu_owned", "pinned"]


def test_a_wake_that_cannot_go_back_to_sleep_raises_wake_failed(tmp_path, monkeypatch):
    eng = SleepEngine(tmp_path)
    eng.sleep()

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    eng._recapture_graphs = oom
    monkeypatch.setattr(eng.moe_offload_cache, "release_slots", boom)
    with pytest.raises(WakeFailed):
        eng.wake()


def test_a_second_sleep_or_wake_changes_nothing(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    assert eng.sleep()["note"] == "already asleep" and eng.graph_runner.destroyed == 1
    eng.wake()
    assert eng.wake()["note"] == "already awake" and len(eng.recaptures) == 1


def test_a_ram_squeeze_while_asleep_spills_to_the_ssd_and_the_layer_stays_there(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    rep = eng.step_memory(axis="ram", direction="down", rebuild=eng.asleep_rebuild)
    assert rep["applied"] == "pinned->disk" and rep["layer"] == 3
    assert eng.sleep_snapshot.spilled_while_asleep == [3]
    eng.wake()
    assert residency(eng) == ["gpu_owned", "pinned", "gpu_owned", "disk"]
    assert eng.recaptures == [([1], True)]  # graphs deferred while a layer is on the SSD
    assert eng._ram_spilled_layers == [3]  # the governor recalls it later, as today


def test_asleep_rebuild_refuses_anything_but_ssd_spills(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    with pytest.raises(SleepRefused):
        eng.asleep_rebuild(moe_cache_size=128)
    with pytest.raises(SleepRefused):
        eng.asleep_rebuild(layer_moves=[(1, "gpu_owned")])
    assert residency(eng)[1] == "pinned"


def test_the_mtp_draft_head_and_ladder_go_to_sleep_and_come_back(tmp_path, monkeypatch):
    built = []

    class Head:
        def __init__(self, engine, spec, **kwargs):
            built.append((engine, spec))

    class Ladder:
        rebinds = 0

        def rebind(self):
            Ladder.rebinds += 1

    monkeypatch.setattr("freetoken.engine.spec_draft.SpecDraftHead", Head)
    eng = SleepEngine(tmp_path)
    closed = []
    eng.spec_draft = SimpleNamespace(close=lambda: closed.append(True))
    eng.spec_state_ladder = Ladder()
    eng.config.spec_decode = SimpleNamespace(enabled=True, graph_widths=())
    eng.sleep()
    assert closed == [True] and eng.spec_draft is None
    assert eng.linear_state_pool.num_slots == 2 and Ladder.rebinds == 1  # padding + ladder slot
    eng.wake()
    assert built == [(eng, eng.config.spec_decode)] and isinstance(eng.spec_draft, Head)
    assert eng.spec_captures == 1 and Ladder.rebinds == 2


def test_a_wake_that_fails_inside_a_pool_rebuild_forces_the_pools_back_to_sleep_size(tmp_path):
    # A pool rebuild that dies mid-allocation leaves its tensors freed but its page / slot
    # count unchanged (both are set only after the allocation), so the way back to sleep must
    # rebuild the pools even when their counts already read "asleep".
    eng = SleepEngine(tmp_path)
    eng.sleep()
    real = eng.kv_cache.rebuild_from_config

    def oom_once(config, num_pages, *, num_swa_pages=None):
        eng.kv_cache.rebuild_from_config = real
        raise RuntimeError("CUDA out of memory")

    eng.kv_cache.rebuild_from_config = oom_once
    with pytest.raises(SleepRefused, match="went back to sleep"):
        eng.wake()
    # [1] from the sleep, then the forced re-make of the pool the OOM left without tensors
    assert eng.num_pages == 1 and eng.kv_cache.rebuilds == [1, 1]
    assert eng.linear_state_pool.num_slots == 1 and eng.linear_state_pool.rebuilds == [1, 9, 1]
    assert eng.wake()["asleep"] is False and eng.num_pages == 100


def test_wake_straight_after_sleep_is_not_refused_on_a_card_the_boot_filled(tmp_path):
    # The boot's auto-sized KV pool can leave well under WAKE_MARGIN_BYTES free; the wake only
    # asks for the margin the card really had before sleep, or it could never wake at all.
    eng = SleepEngine(tmp_path)
    free_before = eng._sync_get_memory()[0]
    eng.TOTAL -= free_before - (100 << 20)  # 100 MiB free while awake
    eng.sleep()
    assert eng.wake()["asleep"] is False


def test_asleep_rebuild_refuses_while_awake(tmp_path):
    eng = SleepEngine(tmp_path)
    with pytest.raises(SleepRefused, match="not asleep"):
        eng.asleep_rebuild(layer_moves=[(3, "disk")])
    assert residency(eng)[3] == "pinned"


def test_mtp_graphs_are_not_captured_while_a_spilled_layer_defers_the_graphs(tmp_path, monkeypatch):
    # A layer the governor spilled while asleep keeps decode eager (graphs deferred), exactly as
    # a live spill does; capturing the MTP widths against a disk layer would bake the disk
    # gather's host sync, so they stay armed and capture lazily after the recall.
    monkeypatch.setattr("freetoken.engine.spec_draft.SpecDraftHead", lambda engine, spec, **kw: object())
    eng = SleepEngine(tmp_path)
    eng.spec_draft = SimpleNamespace(close=lambda: None)
    eng.config.spec_decode = SimpleNamespace(enabled=True, graph_widths=())
    eng.sleep()
    eng.step_memory(axis="ram", direction="down", rebuild=eng.asleep_rebuild)
    eng.wake()
    assert eng.spec_draft is not None and eng.spec_captures == 0
    assert eng._graphs_deferred == "deferred until no disk layers"


def test_prefill_overlap_is_off_asleep_even_with_nothing_owned(tmp_path):
    # No owned layer means no _move_layer on the way down, so sleep turns overlap off itself;
    # release_slots alone would leave the flag on and the asleep spill's rebind would refuse
    # the DISK layer ("prefill overlap DMAs from registered banks").
    eng = SleepEngine(tmp_path, owned=(), overlap=True)
    eng.sleep()
    assert residency(eng) == ["pinned"] * 4 and eng.moe_offload_cache.cache_size == 0
    assert not eng.moe_offload_cache.prefill_overlap
    eng.wake()
    assert eng.moe_offload_cache.prefill_overlap
    assert len(eng.moe_offload_cache.prefill_bank_buffers) == 2


def test_overlap_stays_off_after_a_wake_until_the_asleep_spill_is_recalled(tmp_path):
    eng = SleepEngine(tmp_path, owned=(), overlap=True)
    eng.sleep()
    rep = eng.step_memory(axis="ram", direction="down", rebuild=eng.asleep_rebuild)
    assert rep["applied"] == "pinned->disk"
    eng.wake()
    assert not eng.moe_offload_cache.prefill_overlap  # a DISK layer is still there
    eng._move_layer(rep["layer"], "pinned")  # the governor's recall, as today
    assert eng.moe_offload_cache.prefill_overlap


def oom(*args, **kwargs):
    raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


def test_a_wake_that_fails_with_something_other_than_oom_is_wake_failed(tmp_path):
    # A CUDA fault is not a shortage a game can fix by quitting: no way back is tried, the
    # scheduler latches failed and the watchdog restarts the server.
    eng = SleepEngine(tmp_path)
    eng.sleep()

    def fault(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    eng._recapture_graphs = fault
    with pytest.raises(WakeFailed, match="illegal memory access"):
        eng.wake()


def test_repeated_oom_wakes_end_in_wake_failed(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    eng._recapture_graphs = oom
    for _ in range(MAX_FAILED_WAKES - 1):
        with pytest.raises(SleepRefused, match="went back to sleep"):
            eng.wake()
    with pytest.raises(WakeFailed, match="in a row"):
        eng.wake()


def test_a_successful_wake_starts_the_failed_wake_count_again(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    eng._recapture_graphs = oom
    for _ in range(MAX_FAILED_WAKES - 1):
        with pytest.raises(SleepRefused):
            eng.wake()
    del eng._recapture_graphs
    eng.wake()
    eng.sleep()
    eng._recapture_graphs = oom
    with pytest.raises(SleepRefused, match="went back to sleep"):
        eng.wake()


def test_sleep_drops_the_exl3_pointer_tables_before_the_slots_go(tmp_path, monkeypatch):
    # The packed tables hold references to the slot tensors; left in place, release_slots
    # frees nothing and sleep reports VRAM it never handed back.
    eng = SleepEngine(tmp_path)
    cache = eng.moe_offload_cache
    cache.exl3_scratch = SimpleNamespace(mgemm_tables={"gate_up": object()})
    seen = []
    real = cache.release_slots

    def release():
        seen.append(dict(cache.exl3_scratch.mgemm_tables))
        real()

    monkeypatch.setattr(cache, "release_slots", release)
    eng.sleep()
    assert seen == [{}]


def test_the_wake_self_check_runs_the_boot_warmup_lengths(tmp_path, monkeypatch):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    eng.wake()
    assert eng.warmup_lengths == [(80, 128)]  # triton backend, bf16 experts: the boot's set
    monkeypatch.setattr("freetoken.layers.moe._SMALL_PREFILL_ROWS", 16)
    eng.moe_offload_cache.quant_format = "nvfp4"
    eng.sleep()
    eng.wake()
    assert eng.warmup_lengths[-1] == (80, 128, 16)
    eng.config.attention_backend = "fa,triton"
    eng.moe_offload_cache.quant_format = "bf16"
    eng.sleep()
    eng.wake()
    assert eng.warmups == 2  # boot warmed nothing on this backend, so the wake runs nothing


def test_already_awake_reports_the_real_free_bytes(tmp_path):
    eng = SleepEngine(tmp_path)
    rep = eng.wake()
    assert rep["note"] == "already awake"
    assert rep["vram_free_bytes"] == eng._sync_get_memory()[0] > 0


def test_sleep_without_any_ssd_copy_says_so_plainly(tmp_path):
    eng = SleepEngine(tmp_path)
    eng.expert_disk_copy = None
    eng.moe_offload_cache.expert_disk_copy = None
    with pytest.raises(SleepRefused) as err:
        eng.sleep()
    assert "no SSD expert copy is configured" in str(err.value)
    assert "four minutes" not in str(err.value)
    assert eng.graph_runner.destroyed == 0


def test_the_draft_head_wakes_with_its_pre_sleep_kv_width_and_seed(tmp_path, monkeypatch):
    # After an idle sleep engine.max_seq_len reads the one-page pool (and on a dynamic pool the
    # pre-sleep geometry can be below boot): the draft KV must come back at the width it had.
    built = []

    class Head:
        def __init__(self, engine, spec, *, seed=None, num_pages=None):
            built.append((seed, num_pages, engine.max_seq_len))

        def close(self):
            pass

    monkeypatch.setattr("freetoken.engine.spec_draft.SpecDraftHead", Head)
    eng = SleepEngine(tmp_path)
    eng.spec_draft = SimpleNamespace(close=lambda: None, num_pages=33, seed=1234)
    eng.config.spec_decode = SimpleNamespace(enabled=True, graph_widths=())
    eng.sleep()
    eng.wake()
    assert built == [(1234, 33, 1024)]


def test_sleep_while_an_awake_spill_already_defers_the_graphs(tmp_path):
    # A governor spill while awake already built an eager runner and parked the boot sizes; the
    # snapshot must carry those, not the eager runner's empty list, and the wake keeps them
    # deferred while the layer is still on the SSD.
    eng = SleepEngine(tmp_path)
    eng._move_layer(3, "disk")
    eng._deferred_graph_bs = list(eng._graph_bs_for_recapture())
    eng._recapture_graphs(eng.config, eng._deferred_graph_bs, 0)
    assert eng._graphs_deferred and eng.graph_runner.graph_bs_list == []
    eng.recaptures.clear()
    eng.sleep()
    assert eng.sleep_snapshot.graph_bs == [1]
    eng.wake()
    assert eng.recaptures == [([1], True)]
    assert eng._graphs_deferred == "deferred until no disk layers"
    assert residency(eng) == ["gpu_owned", "pinned", "gpu_owned", "disk"]


def test_a_wake_that_fails_part_way_through_the_layers_goes_back_to_sleep(tmp_path, monkeypatch):
    eng = SleepEngine(tmp_path)
    eng.sleep()
    real = eng._move_layer
    moves = []

    def move(layer_id, target):
        moves.append((layer_id, target))
        if (layer_id, target) == (2, "gpu_owned"):
            oom()
        real(layer_id, target)

    monkeypatch.setattr(eng, "_move_layer", move)
    with pytest.raises(SleepRefused, match="went back to sleep"):
        eng.wake()
    # layer 0 had come home: the way back sends it to the SSD again
    assert moves == [(0, "gpu_owned"), (2, "gpu_owned"), (0, "disk")]
    assert residency(eng) == ["disk", "pinned", "disk", "pinned"]
    assert eng.moe_offload_cache.cache_size == 0 and eng.num_pages == 1
    monkeypatch.setattr(eng, "_move_layer", real)
    assert eng.wake()["asleep"] is False
    assert residency(eng) == ["gpu_owned", "pinned", "gpu_owned", "pinned"]


def test_a_wake_that_fails_after_the_draft_head_is_built_closes_it(tmp_path, monkeypatch):
    closed = []

    class Head:
        def __init__(self, engine, spec, **kwargs):
            pass

        def close(self):
            closed.append(self)

    monkeypatch.setattr("freetoken.engine.spec_draft.SpecDraftHead", Head)
    eng = SleepEngine(tmp_path)
    eng.spec_draft = SimpleNamespace(close=lambda: None)
    eng.config.spec_decode = SimpleNamespace(enabled=True, graph_widths=())
    eng.sleep()
    eng._capture_spec_graphs_at_boot = oom
    with pytest.raises(SleepRefused, match="went back to sleep"):
        eng.wake()
    assert len(closed) == 1 and eng.spec_draft is None
    del eng._capture_spec_graphs_at_boot
    eng.wake()
    assert isinstance(eng.spec_draft, Head) and len(closed) == 1


def test_a_wake_oom_inside_a_real_pool_rebuild_recovers_to_sleep(tmp_path, monkeypatch):
    # The real MHA and GDN pools, not the fakes: an OOM inside LinearStatePool.rebuild leaves
    # its tensors None, and the forced way back to sleep must still re-make both pools.
    from freetoken.distributed.info import DistributedInfo
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    monkeypatch.setattr("freetoken.kvcache.mha_pool.get_tp_info", lambda: DistributedInfo(rank=0, size=1))
    eng = SleepEngine(tmp_path)
    eng.kv_cache = MHAKVCache(num_kv_heads=2, num_layers=4, head_dim=8, num_pages=101, page_size=16,
                              dtype=torch.bfloat16, device=torch.device("cpu"), layer_ids=(1, 3))
    eng.kv_cache.attach_page_table = lambda table: None
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 2), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    eng.linear_state_pool = LinearStatePool(group=group, num_slots=9, dtype=torch.bfloat16,
                                            device=torch.device("cpu"), tp_size=1)
    eng.sleep()
    assert eng.kv_cache._kv_buffer.shape[2] == 2 and eng.linear_state_pool.num_slots == 1
    real = torch.zeros
    armed = [True]

    def zeros(*args, **kwargs):
        if armed[0]:
            armed[0] = False
            oom()
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)
    with pytest.raises(SleepRefused, match="went back to sleep"):
        eng.wake()
    assert eng.linear_state_pool.conv_states.shape[1] == 1
    assert eng.kv_cache._kv_buffer.shape[2] == 2
    assert eng.wake()["asleep"] is False
    assert eng.linear_state_pool.conv_states.shape[1] == 9 and eng.kv_cache._kv_buffer.shape[2] == 101


def test_nothing_keeps_the_draft_head_alive_through_the_sleep_flush(tmp_path, monkeypatch):
    # PR #19 review: a local `draft` in sleep_engine / release_to_sleep kept the head (and its
    # weights + private KV) alive through gc.collect and the final _sync_get_memory, so the
    # allocator flush could not free it and released_bytes undercounted.
    import weakref

    import freetoken.engine.sleep as sleep_mod

    class Draft:
        num_pages, seed = 7, 1

        def close(self):
            pass

    eng = SleepEngine(tmp_path)
    eng.spec_draft = Draft()
    ref = weakref.ref(eng.spec_draft)
    seen = []
    real_sync, real_collect = eng._sync_get_memory, sleep_mod.gc.collect
    eng._sync_get_memory = lambda: (seen.append(("sync", ref() is None)), real_sync())[1]
    monkeypatch.setattr(sleep_mod.gc, "collect", lambda *a: (seen.append(("gc", ref() is None)), real_collect(*a))[1])
    eng.sleep()
    assert seen[0] == ("sync", False)  # free_before, head still loaded
    assert seen[1:] and all(dead for _, dead in seen[1:]), seen


def test_draft_head_close_drops_its_device_state():
    from freetoken.engine.spec_draft import SpecDraftHead

    head = SpecDraftHead.__new__(SpecDraftHead)
    head.engine, head.num_pages, head.seed = object(), 33, 5
    head.kv_cache, head.model = SimpleNamespace(), torch.nn.Linear(2, 2)
    head.page_table, head._graph_in_sample = torch.zeros(2), torch.zeros(3)
    head.expert_runner = SimpleNamespace(close=lambda: None)
    head._graph_runner = SimpleNamespace(destroy=lambda: None)
    head.close()
    assert head.kv_cache is None and head.model is None and head.expert_runner is None
    assert head.page_table is None and head._graph_in_sample is None
    assert head.engine is not None and (head.num_pages, head.seed) == (33, 5)
    head.close()  # idempotent (shutdown after a sleep)


def test_a_failed_wake_frees_what_the_failing_constructor_held(tmp_path, monkeypatch):
    # PR #19 review: an OOM inside the draft head's constructor leaves its half-built tensors
    # reachable from exc.__traceback__ frames; the rollback flush must not see them alive.
    import weakref

    import freetoken.engine.sleep as sleep_mod

    held = []

    class Buffer:
        pass

    class Head:
        def __init__(self, engine, spec, **kwargs):
            scratch = Buffer()  # stands in for a tensor the constructor allocated
            held.append(weakref.ref(scratch))
            oom()

    monkeypatch.setattr("freetoken.engine.spec_draft.SpecDraftHead", Head)
    eng = SleepEngine(tmp_path)
    eng.spec_draft = SimpleNamespace(close=lambda: None)
    eng.sleep()
    real_release = sleep_mod.release_to_sleep
    alive_at_flush = []

    def release(engine, snap, **kw):
        alive_at_flush.append(held[0]() is not None)
        return real_release(engine, snap, **kw)

    monkeypatch.setattr(sleep_mod, "release_to_sleep", release)
    with pytest.raises(SleepRefused, match="went back to sleep"):
        eng.wake()
    assert alive_at_flush == [False]


def test_the_wake_refusal_shows_the_card_wide_free_memory_not_cudas_view(tmp_path, monkeypatch):
    # Live 2026-09-25 (WSL): cudaMemGetInfo said 17.7 GB free while another process's 25 GB
    # held the card (nvidia-smi: 30714 of 32607 MiB used, about 1.9 GB free).
    from freetoken.engine import sleep as sleep_mod

    GB = 1 << 30
    msg = sleep_mod._wake_refusal(int(17.7 * GB), 24 * GB, int(1.9 * GB))
    assert "has 1.9 GB free and waking needs 24.0 GB" in msg and "17.7" not in msg
    # NVML unreadable: no number rather than a wrong one.
    msg = sleep_mod._wake_refusal(int(17.7 * GB), 24 * GB, None)
    assert "GB free" not in msg and "17.7" not in msg and "needs 24.0 GB" in msg
    # The gate uses the card-wide figure too: cuda's view alone would wave this wake through.
    eng = SleepEngine(tmp_path)
    eng.sleep()
    monkeypatch.setattr(sleep_mod, "_card_free_bytes", lambda engine: 0)
    with pytest.raises(SleepRefused, match="has 0.0 GB free"):
        eng.wake()
    assert eng.sleep_snapshot is not None and eng.recaptures == []
    monkeypatch.setattr(sleep_mod, "_card_free_bytes", lambda engine: None)
    assert eng.wake()["asleep"] is False


def test_card_free_bytes_is_none_off_cuda(tmp_path):
    from freetoken.engine import sleep as sleep_mod

    assert sleep_mod._card_free_bytes(SleepEngine(tmp_path)) is None
    assert sleep_mod._nvml_free_bytes("GPU-00000000-0000-0000-0000-000000000000") is None

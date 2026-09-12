"""Tests for step_memory and residency_report in Engine (J1).

Verifies VRAM rung order:
  1. gpu_owned -> pinned (highest layer id)
  2. slots -512 down to floor
  3. KV -25% (idle-only; skipped while active)
  4. at_floor
Verifies VRAM up (reverse order).
Verifies RAM axis returns {"applied": None, "reason": "disk rung not built"}.
Verifies residency_report() output format.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from freetoken.engine.engine import Engine
from freetoken.moe.host_banks import HostResidency
from freetoken.moe.offload_cache import OffloadMoeCache


def _init_tp() -> None:
    if not torch.distributed.is_initialized():
        import tempfile

        f = tempfile.NamedTemporaryFile(delete=False)
        torch.distributed.init_process_group(
            backend="gloo", init_method=f"file://{f.name}", rank=0, world_size=1,
        )


class FakeEngine:
    """Lightweight test harness exposing Engine's step_memory, residency_report,
    _move_layer and rebuild_runtime_cache with mocked graph/model internals.
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_experts: int = 4,
        cache_size: int = 16,
        owned_layers: tuple[int, ...] = (1,),
        num_pages: int = 100,
        overlap: bool = False,
    ):
        _init_tp()
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            model_config=SimpleNamespace(
                num_moe_layers=num_layers,
                num_experts=num_experts,
                vocab_size=100,
            ),
            moe_backend="offload",
            moe_prefill_overlap=overlap,
            moe_cache_size=cache_size,
            cuda_graph_max_bs=1,
            max_running_req=4,
            page_size=16,
            max_seq_len=1024,
            memory_ratio=0.9,
        )
        self.num_pages = num_pages
        self._initial_num_pages = num_pages
        self._initial_moe_cache_size = cache_size
        self._gpu_owned_layer_ids = frozenset(owned_layers)
        self._host_banks = {}
        self._vram_ledger_inputs = None

        # Build OffloadMoeCache
        self.moe_offload_cache = OffloadMoeCache(
            num_layers=num_layers,
            num_experts=num_experts,
            cache_size=cache_size,
            device=self.device,
        )
        sources = {
            "gate_up": [torch.randn(num_experts, 8, 8) for _ in range(num_layers)],
            "down": [torch.randn(num_experts, 8, 8) for _ in range(num_layers)],
        }
        residency = [
            HostResidency.GPU_OWNED.value if i in self._gpu_owned_layer_ids else HostResidency.PINNED.value
            for i in range(num_layers)
        ]
        self.moe_offload_cache.set_bank_sources(
            sources, layer_residency=residency, gpu_owned_layers=self._gpu_owned_layer_ids
        )

        self.rebuild_runtime_cache = MagicMock(side_effect=self._mock_rebuild)

    def _mock_rebuild(self, *, moe_cache_size=None, num_pages=None, layer_moves=None, **kwargs):
        if layer_moves:
            for lid, tgt in layer_moves:
                Engine._move_layer(self, lid, tgt)
        if moe_cache_size is not None:
            self.moe_offload_cache.rebuild(moe_cache_size)
            self.config.moe_cache_size = moe_cache_size
        if num_pages is not None:
            self.num_pages = num_pages

    # Bind the methods from Engine
    step_memory = Engine.step_memory
    step_memory_noop = Engine.step_memory_noop
    _report_maintenance_progress = Engine._report_maintenance_progress
    residency_report = Engine.residency_report
    _move_layer = Engine._move_layer
    _stash_vram_ledger_inputs = Engine._stash_vram_ledger_inputs
    _kv_dynamic_active = Engine._kv_dynamic_active
    _kv_rung_floor_pages = Engine._kv_rung_floor_pages


def test_step_memory_ram_axis_returns_not_built_once_the_card_is_full():
    eng = FakeEngine(cache_size=4)  # at the slot floor: no room to park, and no disk copy
    rep = eng.step_memory(axis="ram", direction="down")
    assert rep["applied"] is None and rep["reason"] == "disk rung not built" and rep["at_floor"] is True
    eng = FakeEngine()  # room to park: the SSD is not needed for the first rung
    rep = eng.step_memory(axis="ram", direction="down")
    assert rep["applied"] == "pinned->gpu_owned"


def test_step_memory_invalid_arguments():
    eng = FakeEngine()
    with pytest.raises(ValueError, match="unknown axis"):
        eng.step_memory(axis="disk", direction="down")
    with pytest.raises(ValueError, match="unknown direction"):
        eng.step_memory(axis="vram", direction="sideways")


def test_step_memory_vram_down_ladder_and_floor():
    eng = FakeEngine(
        num_layers=4,
        num_experts=4,
        cache_size=1024,
        owned_layers=(2,),
        num_pages=100,
        overlap=False,
    )

    # 1. Step down: owned layer moves to pinned
    r1 = eng.step_memory("vram", "down", is_idle=True)
    assert r1["applied"] == "gpu_owned->pinned"
    assert r1["layer"] == 2
    assert not r1["at_floor"]
    assert len(eng._gpu_owned_layer_ids) == 0
    assert not eng.moe_offload_cache.is_gpu_owned_layer(2)

    # 2. Step down: slots shrink 1024 -> 512
    r2 = eng.step_memory("vram", "down", is_idle=True)
    assert r2["applied"] == "slots"
    assert r2["moe_cache_size"] == 512
    assert not r2["at_floor"]

    # 3. Step down: slots shrink 512 -> floor (floor = 4 for num_experts=4, max(4, 512-512) = 4)
    r3 = eng.step_memory("vram", "down", is_idle=True)
    assert r3["applied"] == "slots"
    assert r3["moe_cache_size"] == 4

    # 4. Step down: KV pool shrinks 100 -> 75
    r4 = eng.step_memory("vram", "down", is_idle=True)
    assert r4["applied"] == "kv"
    assert eng.num_pages == 75
    assert r4["at_floor"]

    # 5. Step down: at floor
    r5 = eng.step_memory("vram", "down", is_idle=True)
    assert r5["applied"] is None
    assert r5["at_floor"]


def test_step_memory_kv_rung_skipped_when_active():
    eng = FakeEngine(
        num_layers=4,
        num_experts=4,
        cache_size=4,  # already at floor
        owned_layers=(),  # no owned layers
        num_pages=100,
        overlap=False,
    )

    # Request active (is_idle=False): KV rung is skipped!
    r = eng.step_memory("vram", "down", is_idle=False)
    assert r["applied"] is None
    assert r["at_floor"]
    assert eng.num_pages == 100  # KV untouched


def test_step_memory_vram_up():
    eng = FakeEngine(
        num_layers=4,
        num_experts=4,
        cache_size=512,
        owned_layers=(),
        num_pages=75,
    )
    eng._initial_num_pages = 100
    eng._initial_moe_cache_size = 1024

    # 1. Step up: restores KV first
    r1 = eng.step_memory("vram", "up", is_idle=True)
    assert r1["applied"] == "kv"
    assert eng.num_pages == 100

    # 2. Step up: grows slots (512 -> 1024)
    r2 = eng.step_memory("vram", "up", is_idle=True)
    assert r2["applied"] == "slots"
    assert r2["moe_cache_size"] == 1024

    # 3. Step up: promotes a pinned layer to gpu_owned
    r3 = eng.step_memory("vram", "up", is_idle=True)
    assert r3["applied"] == "pinned->gpu_owned"
    assert len(eng._gpu_owned_layer_ids) == 1


def test_residency_report():
    eng = FakeEngine(
        num_layers=3,
        num_experts=4,
        cache_size=8,
        owned_layers=(0,),
    )
    rep = eng.residency_report()
    assert rep["owned"] == 1
    assert rep["pinned"] == 2
    assert rep["disk"] == 0
    assert rep["moe_cache_size"] == 8
    assert rep["layers"] == {"0": "gpu_owned", "1": "pinned", "2": "pinned"}
    # the report crosses the ZMQ/msgpack hop with strict map keys; int keys crash the worker
    import msgpack
    assert msgpack.unpackb(msgpack.packb(rep), raw=False, strict_map_key=True) == rep


def test_step_memory_uses_the_rebuild_callable_for_every_rung():
    """The scheduler passes its own rebuild_cache so a step never bypasses the safe point."""
    eng = FakeEngine(num_layers=4, num_experts=4, cache_size=1024, owned_layers=(2,), num_pages=100)
    seen: list[dict] = []

    def via_scheduler(**kwargs):
        seen.append(kwargs)
        eng._mock_rebuild(**kwargs)

    for _ in range(4):
        eng.step_memory("vram", "down", is_idle=True, rebuild=via_scheduler)
    assert [sorted(k for k, v in kw.items() if v is not None) for kw in seen] == [
        ["layer_moves"], ["moe_cache_size"], ["moe_cache_size"], ["num_pages"],
    ]
    assert eng.rebuild_runtime_cache.call_count == 0


def test_step_up_promotes_in_reverse_demotion_order():
    eng = FakeEngine(num_layers=5, num_experts=4, cache_size=4, owned_layers=(1, 3), num_pages=100)
    assert eng.step_memory("vram", "down", is_idle=True)["layer"] == 3
    assert eng.step_memory("vram", "down", is_idle=True)["layer"] == 1
    assert eng._demoted_layers == [3, 1]
    # up: the layer demoted last comes back first, then the one before it
    assert eng.step_memory("vram", "up", is_idle=True)["layer"] == 1
    assert eng.step_memory("vram", "up", is_idle=True)["layer"] == 3
    assert eng._demoted_layers == []
    assert eng._gpu_owned_layer_ids == frozenset({1, 3})


def test_step_up_promotion_failure_propagates():
    """A rebuild failure after teardown must reach the scheduler, not become 'nothing applied'."""
    eng = FakeEngine(num_layers=4, num_experts=4, cache_size=4, owned_layers=(), num_pages=100)

    def boom(**kwargs):
        raise RuntimeError("cuda oom during promotion")

    with pytest.raises(RuntimeError, match="promotion"):
        eng.step_memory("vram", "up", is_idle=True, rebuild=boom)


# ---------------------------------------------------------------------------------------
# RAM ladder: park on the card before the SSD (2026-09-08). A parked layer frees one layer
# of host RAM at the price of one layer of shared slots; the SSD rung costs eager decode.
# ---------------------------------------------------------------------------------------


class _DiskCopyStub:
    def layer_complete(self, layer_id: int) -> bool:
        return True

    def bank_bytes(self, name: str) -> int:
        return 0


def _ram_engine(cache_size: int = 16, overlap: bool = False):
    fe = FakeEngine(num_layers=4, num_experts=4, cache_size=cache_size, owned_layers=(), overlap=overlap)
    fe.expert_disk_copy = _DiskCopyStub()
    calls: list[dict] = []

    def rebuild(**kwargs):
        calls.append(kwargs)
        moves = kwargs.get("layer_moves") or []
        if any(tgt == "disk" or fe.moe_offload_cache.layer_residency[lid] == HostResidency.DISK.value
               for lid, tgt in moves):
            # A real spill or recall needs the disk reader; the test only checks the rung chosen.
            return
        fe._mock_rebuild(**kwargs)

    return fe, calls, rebuild


def test_ram_down_parks_on_the_card_before_spilling_and_stops_at_the_slot_floor():
    fe, calls, rebuild = _ram_engine(cache_size=16)  # floor without overlap: num_experts = 4
    parked = []
    for expected_slots in (12, 8, 4):
        res = fe.step_memory("ram", "down", rebuild=rebuild)
        assert res["applied"] == "pinned->gpu_owned", res
        assert res["moe_cache_size"] == expected_slots
        assert fe.moe_offload_cache.cache_size == expected_slots
        parked.append(res["layer"])
        assert fe.moe_offload_cache.layer_residency[res["layer"]] == HostResidency.GPU_OWNED.value
    assert parked == [0, 1, 2], "lowest ids first without routing stats"
    assert fe._ram_parked_layers == parked
    assert fe.residency_report()["ram_parked"] == parked
    # shrink-then-promote, never the other way round
    assert [list(c) for c in calls[:2]] == [["moe_cache_size"], ["layer_moves"]]
    res = fe.step_memory("ram", "down", rebuild=rebuild)
    assert res["applied"] == "pinned->disk" and res["layer"] == 3, "at the slot floor the SSD is next"
    assert calls[-1] == {"layer_moves": [(3, "disk")]}
    assert res["moe_cache_size"] == 4


def test_ram_up_unparks_after_recalls_and_grows_the_slots_back():
    fe, calls, rebuild = _ram_engine(cache_size=16)
    fe.step_memory("ram", "down", rebuild=rebuild)
    fe.step_memory("ram", "down", rebuild=rebuild)
    assert fe.moe_offload_cache.cache_size == 8 and fe._ram_parked_layers == [0, 1]
    calls.clear()
    res = fe.step_memory("ram", "up", rebuild=rebuild)
    assert res["applied"] == "gpu_owned->pinned" and res["layer"] == 1, "last parked, first unparked"
    assert res["moe_cache_size"] == 12 and fe.moe_offload_cache.cache_size == 12
    assert [list(c) for c in calls] == [["layer_moves"], ["moe_cache_size"]], "free the card first, then grow"
    assert fe.moe_offload_cache.layer_residency[1] == HostResidency.PINNED.value
    res = fe.step_memory("ram", "up", rebuild=rebuild)
    assert res["applied"] == "gpu_owned->pinned" and res["layer"] == 0 and res["moe_cache_size"] == 16
    assert fe._ram_parked_layers == []
    res = fe.step_memory("ram", "up", rebuild=rebuild)
    assert res["applied"] is None, "nothing left to bring home"


def test_ram_up_recalls_a_disk_layer_before_unparking():
    fe, calls, rebuild = _ram_engine(cache_size=16)
    fe.step_memory("ram", "down", rebuild=rebuild)
    fe._ram_spilled_layers = [3]
    fe.moe_offload_cache.layer_residency[3] = HostResidency.DISK.value  # as a real spill would leave it
    calls.clear()
    res = fe.step_memory("ram", "up", rebuild=rebuild)
    assert res["applied"] == "disk->pinned" and res["layer"] == 3
    assert fe._ram_parked_layers == [0], "the parked layer waits for the eager-decode penalty to go first"


def test_ram_up_never_grows_the_slots_past_the_boot_size():
    fe, calls, rebuild = _ram_engine(cache_size=16)
    fe.step_memory("ram", "down", rebuild=rebuild)  # 12 slots, layer 0 parked
    fe._mock_rebuild(moe_cache_size=16)  # the VRAM axis regrew the slots meanwhile
    calls.clear()
    res = fe.step_memory("ram", "up", rebuild=rebuild)
    assert res["applied"] == "gpu_owned->pinned" and res["moe_cache_size"] == 16
    assert calls == [{"layer_moves": [(0, "pinned")]}], "no growth call when already at the boot size"


def test_the_vram_axis_leaves_a_ram_parked_layer_alone_and_takes_slots_instead():
    fe, calls, rebuild = _ram_engine(cache_size=16)
    fe.step_memory("ram", "down", rebuild=rebuild)  # 12 slots, layer 0 parked
    assert fe._ram_parked_layers == [0]
    res = fe.step_memory("vram", "down", ram_tight=True, rebuild=rebuild)
    assert res["applied"] == "slots" and res["moe_cache_size"] == 4, "not gpu_owned->disk: the park stands"
    assert calls[-1] == {"moe_cache_size": 4}
    assert fe.moe_offload_cache.layer_residency[0] == HostResidency.GPU_OWNED.value
    assert fe._ram_parked_layers == [0]
    # a boot-owned layer is still fair game for the VRAM axis
    fe2 = FakeEngine(num_layers=4, num_experts=4, cache_size=16, owned_layers=(3,))
    fe2.expert_disk_copy = _DiskCopyStub()
    fe2.step_memory("ram", "down", rebuild=lambda **kw: fe2._mock_rebuild(**kw))
    assert fe2._ram_parked_layers == [0] and fe2._gpu_owned_layer_ids == {0, 3}
    res = fe2.step_memory("vram", "down", ram_tight=False, rebuild=lambda **kw: fe2._mock_rebuild(**kw))
    assert res["applied"] == "gpu_owned->pinned" and res["layer"] == 3


def test_a_parked_layer_moved_by_hand_is_forgotten():
    fe, calls, rebuild = _ram_engine(cache_size=16)
    fe.step_memory("ram", "down", rebuild=rebuild)
    fe._move_layer(0, "pinned")
    assert fe._ram_parked_layers == [], "back in RAM by another route: not parked any more"
    assert fe.step_memory("ram", "up", rebuild=rebuild)["applied"] is None


def test_a_rejected_promote_gives_the_slots_back():
    from freetoken.engine.engine import CacheRebuildRejected

    fe, calls, _ = _ram_engine(cache_size=16)

    def rebuild(**kwargs):
        calls.append(kwargs)
        if kwargs.get("layer_moves"):
            raise CacheRebuildRejected("promoting needs more than the free MiB")
        fe._mock_rebuild(**kwargs)

    res = fe.step_memory("ram", "down", rebuild=rebuild)
    assert res["applied"] is None and "park rejected" in res["reason"]
    assert res["moe_cache_size"] == 16 and fe.moe_offload_cache.cache_size == 16
    assert [list(c) for c in calls] == [["moe_cache_size"], ["layer_moves"], ["moe_cache_size"]]
    assert fe._ram_parked_layers == [] and fe.rebuild_teardown_started is False


def test_no_park_without_vram_margin_on_a_cuda_device():
    fe, calls, rebuild = _ram_engine(cache_size=16)
    fe.device = torch.device("cuda")
    fe._sync_get_memory = lambda: (100 << 20, 32 << 30)  # 100 MiB free: below the 512 MiB margin
    res = fe.step_memory("ram", "down", rebuild=rebuild)
    assert res["applied"] == "pinned->disk", "no margin on the card: the SSD rung, not a half-done park"
    fe._sync_get_memory = lambda: (1 << 30, 32 << 30)
    fe.device = torch.device("cpu")  # promotions in this harness allocate on cpu
    res = fe.step_memory("ram", "down", rebuild=rebuild)
    assert res["applied"] == "pinned->gpu_owned"


def test_ram_down_parks_the_busiest_layer_when_routing_stats_exist(monkeypatch):
    fe, calls, rebuild = _ram_engine(cache_size=16)
    fe.config.model_path = "/models/demo"
    import freetoken.moe.learned_routing as lr

    stats = SimpleNamespace(freq=[[1, 1, 1, 1], [9, 9, 9, 9], [2, 2, 2, 2], [0, 0, 0, 0]])
    monkeypatch.setattr(lr, "load_routing_stats", lambda *a, **k: stats)
    res = fe.step_memory("ram", "down", rebuild=rebuild)
    assert res["applied"] == "pinned->gpu_owned" and res["layer"] == 1, "the layer that routes most goes to the card"


def test_ram_down_respects_the_overlap_floor():
    fe, calls, rebuild = _ram_engine(cache_size=12, overlap=True)  # floor with overlap: 2 * 4 = 8
    res = fe.step_memory("ram", "down", rebuild=rebuild)
    assert res["applied"] == "pinned->gpu_owned" and res["moe_cache_size"] == 8
    res = fe.step_memory("ram", "down", rebuild=rebuild)
    assert res["applied"] == "pinned->disk", "8 - 4 would leave fewer than the 8 the overlap path needs"


# ---- residency-only no-op preflight (2026-09-09: 847 of 861 live governor steps were no-ops) --


def _cuda_looking(eng: FakeEngine) -> MagicMock:
    """Make the memory probe observable: on a CUDA device step_memory calls _sync_get_memory."""
    eng.device = SimpleNamespace(type="cuda")
    eng._sync_get_memory = MagicMock(return_value=(2 << 30, 2 << 30))
    return eng._sync_get_memory


def test_ram_up_with_everything_pinned_is_exhausted_without_a_memory_sync():
    eng = FakeEngine(owned_layers=())  # every layer pinned in host RAM, nothing spilled or parked
    sync = _cuda_looking(eng)
    rep = eng.step_memory(axis="ram", direction="up")
    assert rep["applied"] is None and rep["at_floor"] is False
    assert rep["exhausted"] is True and rep["reason"].startswith("nothing to recall")
    assert sync.call_count == 0, "a proven no-op must not synchronize the card"
    assert eng.rebuild_runtime_cache.call_count == 0
    assert eng.step_memory_noop("ram", "up") == rep


def test_ram_up_with_a_parked_layer_is_not_a_noop_and_unparks():
    eng = FakeEngine(owned_layers=(1,))
    eng._ram_parked_layers = [1]
    assert eng.step_memory_noop("ram", "up") is None
    rep = eng.step_memory(axis="ram", direction="up")
    assert rep["applied"] == "gpu_owned->pinned" and rep.get("exhausted", False) is False


def test_ram_down_with_no_pinned_layer_is_a_floor_no_op_without_a_sync():
    eng = FakeEngine(num_layers=4, owned_layers=(0, 1, 2))
    eng.moe_offload_cache.layer_residency[3] = HostResidency.DISK.value  # the last one spilled
    sync = _cuda_looking(eng)
    rep = eng.step_memory(axis="ram", direction="down")
    assert rep["applied"] is None and rep["at_floor"] is True and rep["exhausted"] is True
    assert sync.call_count == 0


def test_vram_up_noop_is_not_exhausted_while_a_kv_restore_waits_for_idle():
    # Everything on the card already except a shrunk KV pool: "up" can still apply once idle,
    # so the governor must keep asking; only a pool already at its boot size is exhausted.
    eng = FakeEngine(num_layers=4, owned_layers=(0, 1, 2), cache_size=16, num_pages=100)
    eng.num_pages = 75
    assert eng.step_memory_noop("vram", "up") is None
    rep = eng.step_memory(axis="vram", direction="up", is_idle=False)
    assert rep["applied"] is None and rep["exhausted"] is False
    eng.num_pages = 100
    rep = eng.step_memory(axis="vram", direction="up", is_idle=False)
    assert rep["applied"] is None and rep["exhausted"] is True
    assert eng.step_memory_noop("vram", "up") == rep


# ---- maintenance progress hook (2026-09-09 review F1: silence is not proof of no work) ------


def test_step_memory_reports_each_unit_through_the_progress_hook():
    eng = FakeEngine(num_layers=4, owned_layers=(0, 1, 2), cache_size=16, num_pages=100)
    seen: list = []
    eng.maintenance_progress = lambda phase, detail=None: seen.append((phase, detail))
    rep = eng.step_memory(axis="vram", direction="down")
    assert rep["applied"] is not None
    assert seen[0] == ("step:probed", "vram down"), seen
    # The hook is optional: a direct engine caller (no scheduler) has none.
    eng.maintenance_progress = None
    eng.step_memory(axis="vram", direction="down")
    # A hook that raises must not fail the step it reports on.
    eng.maintenance_progress = lambda *a: (_ for _ in ()).throw(RuntimeError("socket gone"))
    rep = eng.step_memory(axis="vram", direction="down")
    assert rep is not None


def test_a_proven_noop_step_reports_nothing_and_touches_nothing():
    eng = FakeEngine(owned_layers=())
    seen: list = []
    eng.maintenance_progress = lambda phase, detail=None: seen.append(phase)
    rep = eng.step_memory(axis="ram", direction="up")
    assert rep["exhausted"] is True and seen == []


def test_residency_report_carries_the_kv_pool_size():
    """The governor invalidates a remembered "nothing to promote" when KV pages change
    (review F4), so the residency report must say how many there are."""
    eng = FakeEngine(num_pages=100)
    assert eng.residency_report()["num_pages"] == 100
    eng.num_pages = 75
    assert eng.residency_report()["num_pages"] == 75


def test_a_pure_moe_shrink_is_the_only_rebuild_that_may_skip_the_budget():
    """The verdict validate_rebuild trusts (PR #4 review): slots at or below the current
    count and nothing else named. A sibling-pool target, a layer move or a grow all keep the
    budget check, whatever the resident total says."""
    from freetoken.engine.engine import Engine

    base = dict(current_moe=6262, moe_cache_size=5750, num_pages=None,
                num_mamba_slots=None, num_swa_pages=None, layer_moves=None)
    assert Engine._is_pure_moe_shrink(**base)
    assert Engine._is_pure_moe_shrink(**{**base, "moe_cache_size": 6262})
    for change in (
        {"moe_cache_size": 6263},        # a grow
        {"moe_cache_size": None},        # nothing to shrink
        {"num_pages": 8000},             # KV rung, even a smaller one
        {"num_mamba_slots": 13},         # GDN state pool
        {"num_swa_pages": 10},           # pinned window
        {"layer_moves": [(3, "gpu_owned")]},  # a full layer of VRAM
    ):
        assert not Engine._is_pure_moe_shrink(**{**base, **change}), change


# ----- dynamic KV pool integration (Task 7: 2026-09-12) -----


def test_vram_down_kv_rung_shrinks_to_the_dynamic_floor_when_the_dynamic_pool_is_on():
    eng = FakeEngine(num_layers=2, num_experts=4, cache_size=8, owned_layers=(), num_pages=400)
    eng.config.kv_dynamic = True
    eng.kv_dynamic_floor_pages = 100
    # slots already at the floor (num_experts=4, overlap False) so rung 3 is reached
    eng.moe_offload_cache.rebuild(4)
    eng.config.moe_cache_size = 4
    res = eng.step_memory("vram", "down", is_idle=True)
    assert res["applied"] == "kv" and eng.num_pages == 100  # not the old 300 (-25 %)


def test_vram_up_kv_rung_is_a_noop_under_the_dynamic_pool():
    eng = FakeEngine(num_layers=3, num_experts=4, cache_size=4, owned_layers=(0, 1), num_pages=50)
    eng.config.kv_dynamic = True
    eng.kv_dynamic_floor_pages = 50
    eng._initial_num_pages = 100
    assert eng.step_memory_noop("vram", "up")["exhausted"] is True
    # With dynamic off, the same state should return None (legacy restore pending)
    eng.config.kv_dynamic = False
    assert eng.step_memory_noop("vram", "up") is None


def test_vram_up_kv_rung_restores_when_dynamic_is_on_but_the_floor_is_unset():
    eng = FakeEngine(num_layers=3, num_experts=4, cache_size=4, owned_layers=(0, 1), num_pages=50)
    eng.config.kv_dynamic = True
    eng.kv_dynamic_floor_pages = None  # dynamic mode on but floor unset -> legacy behavior
    eng._initial_num_pages = 100
    assert eng.step_memory_noop("vram", "up") is None  # legacy restore pending


def test_vram_down_slot_rung_reports_at_floor_when_kv_is_already_at_the_dynamic_floor():
    eng = FakeEngine(num_layers=2, num_experts=4, cache_size=8, owned_layers=(), num_pages=100)
    eng.config.kv_dynamic = True
    eng.kv_dynamic_floor_pages = 100
    # slots at 8, one rung above floor (floor = 4). After shrink to floor (4),
    # KV can't shrink further (already at dynamic floor), so at_floor=True.
    res = eng.step_memory("vram", "down", is_idle=True)
    assert res["applied"] == "slots" and res["moe_cache_size"] == 4 and res["at_floor"] is True

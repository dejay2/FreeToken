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
    residency_report = Engine.residency_report
    _move_layer = Engine._move_layer
    _stash_vram_ledger_inputs = Engine._stash_vram_ledger_inputs


def test_step_memory_ram_axis_returns_not_built():
    eng = FakeEngine()
    rep = eng.step_memory(axis="ram", direction="down")
    assert rep == {"applied": None, "reason": "disk rung not built"}


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
    assert rep["layers"] == {0: "gpu_owned", 1: "pinned", 2: "pinned"}

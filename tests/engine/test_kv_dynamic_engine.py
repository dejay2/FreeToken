"""Engine side of the dynamic KV pool: one budget for planner and validator, shrinks resized
before grows, and the over-budget card case from the 2026-09-12 review."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from freetoken.engine.engine import Engine

PAGE, KV_PAGE, SLOT = 64, 13_248 * 64, 2_772_480


def test_snapshot_pool_budget_is_slots_plus_pages():
    eng = SimpleNamespace(
        num_pages=1025, moe_offload_cache=SimpleNamespace(cache_size=7200, bank_sources={}),
        _gpu_owned_layer_ids=frozenset(),
        _target_moe_and_expert_bytes=lambda mcs: (7200, SLOT),
        _kv_bytes_per_page=lambda: KV_PAGE,
    )
    assert Engine.snapshot_pool_budget(eng) == 7200 * SLOT + 1025 * KV_PAGE
    assert eng.pool_budget_bytes == 7200 * SLOT + 1025 * KV_PAGE


def test_rebuild_order_puts_a_kv_shrink_before_the_moe_grow():
    calls = []
    eng = _recording_engine(calls, num_pages=4097, slots=6260)
    Engine._resize_pools(eng, eng.config, moe_cache_size=7200, num_pages=1025, num_swa_pages=None,
                         num_mamba_slots=None)
    assert calls == [("kv", 1025), ("moe", 7200)]


def test_rebuild_order_keeps_moe_first_when_kv_grows():
    calls = []
    eng = _recording_engine(calls, num_pages=1025, slots=7200)
    Engine._resize_pools(eng, eng.config, moe_cache_size=7043, num_pages=1537, num_swa_pages=None,
                         num_mamba_slots=None)
    assert calls == [("moe", 7043), ("kv", 1537)]


def _recording_engine(calls, *, num_pages, slots):
    cache = SimpleNamespace(cache_size=slots, quant_format="nvfp4",
                            rebuild=lambda n: calls.append(("moe", n)))
    eng = SimpleNamespace(
        num_pages=num_pages, moe_offload_cache=cache, linear_state_pool=None,
        spec_state_ladder=None, _gpu_owned_layer_ids=frozenset(),
        config=SimpleNamespace(moe_cache_size=slots, model_config=SimpleNamespace(num_experts=512)),
        _resize_kv_pool=lambda config, n, swa: calls.append(("kv", n)),
        _report_maintenance_progress=lambda *a, **k: None,
    )
    return eng

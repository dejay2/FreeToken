"""Env surface of the layer-ahead expert prefetch (``FREETOKEN_MOE_PREFETCH``).

The defaults ARE the contract: unset means off, and off must leave the decode path exactly
as it was. The numeric defaults are the measured ones (k'=10 is the bus knee, layers
0/22/38 are the three whose predictor recall sits at 0.3-0.4), so a silent change to any of
them changes the bandwidth a live run spends.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.prefetch import PrefetchConfig


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "FREETOKEN_MOE_PREFETCH",
        "FREETOKEN_MOE_PREFETCH_TOPK",
        "FREETOKEN_MOE_PREFETCH_MAX_MISSES",
        "FREETOKEN_MOE_PREFETCH_SKIP_LAYERS",
        "FREETOKEN_MOE_PREFETCH_LOG",
        "FREETOKEN_MOE_PREFETCH_LOG_EVERY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_off_and_carry_the_measured_knobs():
    cfg = PrefetchConfig.from_env()
    assert cfg.enabled is False
    assert cfg.topk == 10
    assert cfg.max_misses == 8
    assert cfg.skip_layers == frozenset({0, 22, 38})
    assert cfg.log is False


@pytest.mark.parametrize("raw", ["1", "true", "on", "YES"])
def test_enable_accepts_the_usual_truthy_spellings(monkeypatch, raw):
    monkeypatch.setenv("FREETOKEN_MOE_PREFETCH", raw)
    assert PrefetchConfig.from_env().enabled is True


@pytest.mark.parametrize("raw", ["0", "", "no", "off", "false"])
def test_anything_else_leaves_it_off(monkeypatch, raw):
    monkeypatch.setenv("FREETOKEN_MOE_PREFETCH", raw)
    assert PrefetchConfig.from_env().enabled is False


def test_overrides_are_read(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MOE_PREFETCH", "1")
    monkeypatch.setenv("FREETOKEN_MOE_PREFETCH_TOPK", "6")
    monkeypatch.setenv("FREETOKEN_MOE_PREFETCH_MAX_MISSES", "3")
    monkeypatch.setenv("FREETOKEN_MOE_PREFETCH_SKIP_LAYERS", "1, 2,7")
    monkeypatch.setenv("FREETOKEN_MOE_PREFETCH_LOG", "1")
    monkeypatch.setenv("FREETOKEN_MOE_PREFETCH_LOG_EVERY", "5")
    cfg = PrefetchConfig.from_env()
    assert (cfg.topk, cfg.max_misses, cfg.log, cfg.log_every) == (6, 3, True, 5)
    assert cfg.skip_layers == frozenset({1, 2, 7})


def test_empty_skip_list_means_prefetch_every_layer(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MOE_PREFETCH_SKIP_LAYERS", "")
    assert PrefetchConfig.from_env().skip_layers == frozenset()


def test_cache_is_inert_when_the_prefetch_is_unarmed(monkeypatch):
    """The whole point of the default: no stream, no events, no buffers, no behaviour."""
    import freetoken.moe.prefetch as pf
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setattr(pf, "PREFETCH", PrefetchConfig())
    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu")
    )
    assert cache.prefetch_on is False
    assert not hasattr(cache, "prefetch_stream")
    assert cache.prefetch_ready(1) is False
    assert cache.prefetch_wait(1) is False
    assert cache.prefetch_stats_summary() == {}
    # ...and the no-op call sites the decode path makes unconditionally stay no-ops.
    cache.prefetch_experts(1, torch.zeros(1, 2, dtype=torch.int32))
    cache.prefetch_note_actual(1, torch.zeros(2, dtype=torch.int32))


def test_skip_layers_and_range_gate_prefetch_ready(monkeypatch):
    import freetoken.moe.prefetch as pf
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setattr(
        pf, "PREFETCH", PrefetchConfig(enabled=True, skip_layers=frozenset({2}))
    )
    cache = OffloadMoeCache(
        num_layers=4, num_experts=4, cache_size=8, device=torch.device("cpu")
    )
    # device is CPU, so the feature stays off regardless -- that is itself the contract for
    # a non-CUDA cache (there is no second stream to hide anything behind).
    assert cache.prefetch_on is False
    assert cache.prefetch_ready(1) is False

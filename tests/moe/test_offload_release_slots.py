"""OffloadMoeCache.release_slots: sleep frees the whole slot cache; rebuild brings it back cold."""

from __future__ import annotations

import torch

from freetoken.moe.host_banks import HostResidency
from freetoken.moe.offload_cache import OffloadMoeCache

E = 4  # experts per layer
ROW = 8 * 8 * 4  # one float32 [8, 8] row per bank


def _cache(cache_size: int = 16, owned=(1,), overlap: bool = False) -> OffloadMoeCache:
    cache = OffloadMoeCache(
        num_layers=4, num_experts=E, cache_size=cache_size, device=torch.device("cpu"),
        prefill_overlap=overlap,
    )
    sources = {
        "gate_up": [torch.randn(E, 8, 8) for _ in range(4)],
        "down": [torch.randn(E, 8, 8) for _ in range(4)],
    }
    residency = [
        HostResidency.GPU_OWNED.value if i in owned else HostResidency.PINNED.value for i in range(4)
    ]
    cache.set_bank_sources(sources, layer_residency=residency, gpu_owned_layers=frozenset(owned))
    return cache


def test_release_frees_every_slot_row_and_keeps_the_banks():
    cache = _cache()
    sources = {n: list(v) for n, v in cache.bank_sources.items()}
    owned = cache.resident_banks[1]
    freed = cache.release_slots()
    assert freed == 16 * ROW * 2  # two banks
    assert cache.cache_size == 0 and cache.bank_caches == {} and cache.banks == []
    assert bool((cache.slot_for_id == -1).all())
    assert cache.id_of_slot.numel() == 0 and cache.usage.numel() == 0
    assert not cache._copy_fused_ok
    for name, per_layer in sources.items():
        assert all(a is b for a, b in zip(per_layer, cache.bank_sources[name]))
    assert cache.resident_banks[1] is owned


def test_rebuild_after_release_is_a_cold_cache_of_the_old_size():
    cache = _cache()
    cache.release_slots()
    cache.rebuild(16)
    assert cache.cache_size == 16
    assert cache.bank_caches["gate_up"].shape == (16, 8, 8)
    assert bool((cache.id_of_slot == -1).all()) and int(cache.usage.sum()) == 0


def test_release_tears_down_prefill_overlap_and_rebuild_restores_it():
    cache = _cache(cache_size=2 * E, owned=(), overlap=True)
    assert cache.prefill_bank_buffers
    cache.release_slots()
    assert cache.prefill_bank_buffers == []
    cache.rebuild(2 * E)
    assert cache.prefill_overlap and len(cache.prefill_bank_buffers) == 2


def test_a_second_release_frees_nothing():
    cache = _cache()
    cache.release_slots()
    assert cache.release_slots() == 0


def test_release_drops_the_disk_staging_rows_and_the_fallback_latch():
    """M3: the disk backend's device scratch is GPU memory too, and the d2d-fallback log latch
    resets as it does on rebuild (the geometry changed)."""
    cache = _cache()
    cache._disk_device_scratch = {"gate_up": torch.empty(10, 8, 8), "down": torch.empty(10, 8, 8)}
    cache._hit_d2d_fallback_logged = True
    cache.release_slots()
    assert cache._disk_device_scratch is None
    assert cache._hit_d2d_fallback_logged is False


def test_row_bytes_survive_a_sleep():
    """M5: the settings page's per-slot cost must not read 0 while the slot cache is gone."""
    cache = _cache()
    before = cache.bytes_per_expert_row()
    assert before == ROW * 2
    cache.release_slots()
    assert cache.bytes_per_expert_row() == before
    cache.rebuild(16)
    assert cache.bytes_per_expert_row() == before

"""Tests for runtime layer rebind in OffloadMoeCache (J1).

Verifies that rebind_layer preserves bank_views / resident_views consistency,
keeps pointer tables sized num_layers, refuses leaving zero streaming layers,
and updates owned_layer_count and has_disk_layers properties.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.host_banks import HostResidency
from freetoken.moe.offload_cache import OffloadMoeCache


def _init_tp() -> None:
    if not torch.distributed.is_initialized():
        import tempfile

        f = tempfile.NamedTemporaryFile(delete=False)
        torch.distributed.init_process_group(
            backend="gloo", init_method=f"file://{f.name}", rank=0, world_size=1,
        )


def _make_dummy_cache(num_layers: int = 3, num_experts: int = 4, cache_size: int = 8, device="cpu"):
    _init_tp()
    dev = torch.device(device)
    cache = OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=dev,
    )
    sources = {
        "gate_up": [torch.randn(num_experts, 32, 8, device=dev) for _ in range(num_layers)],
        "down": [torch.randn(num_experts, 8, 16, device=dev) for _ in range(num_layers)],
    }
    residency = [HostResidency.PINNED.value] * num_layers
    cache.set_bank_sources(sources, layer_residency=residency)
    return cache, sources


def test_rebind_pinned_to_owned_to_pinned():
    cache, sources = _make_dummy_cache(num_layers=3, num_experts=4, cache_size=8)

    assert cache.owned_layer_count == 0
    assert not cache.is_gpu_owned_layer(1)
    assert not cache.has_disk_layers

    # Initial bank_views: returns the two slot caches
    bviews = cache.bank_views()
    assert len(bviews) == 2
    assert bviews[0].shape == (8, 32, 8)
    assert bviews[1].shape == (8, 8, 16)

    # Calling resident_views on a pinned layer raises AssertionError
    with pytest.raises(AssertionError, match="not GPU-owned"):
        cache.resident_views(1)

    # Rebind layer 1 from pinned -> gpu_owned
    dev_banks = {
        "gate_up": torch.randn(4, 32, 8),
        "down": torch.randn(4, 8, 16),
    }
    cache.rebind_layer(1, HostResidency.GPU_OWNED.value, dev_banks)

    assert cache.owned_layer_count == 1
    assert cache.is_gpu_owned_layer(1)
    assert not cache.is_gpu_owned_layer(0)
    assert not cache.is_gpu_owned_layer(2)

    # resident_views(1) now returns the newly bound device tensors
    rviews = cache.resident_views(1)
    assert rviews[0] is dev_banks["gate_up"]
    assert rviews[1] is dev_banks["down"]

    # bank_views() is unaffected and continues to return slot cache views
    bviews_after = cache.bank_views()
    assert len(bviews_after) == 2
    assert bviews_after[0].shape == (8, 32, 8)

    # Rebind layer 1 back to pinned
    host_banks = {
        "gate_up": torch.randn(4, 32, 8),
        "down": torch.randn(4, 8, 16),
    }
    cache.rebind_layer(1, HostResidency.PINNED.value, host_banks)

    assert cache.owned_layer_count == 0
    assert not cache.is_gpu_owned_layer(1)
    assert cache.bank_sources["gate_up"][1] is host_banks["gate_up"]
    assert cache.bank_sources["down"][1] is host_banks["down"]

    # resident_views(1) raises again
    with pytest.raises(AssertionError, match="not GPU-owned"):
        cache.resident_views(1)


def test_rebind_refuses_zero_streaming_layers():
    cache, _ = _make_dummy_cache(num_layers=2, num_experts=4, cache_size=8)

    # Rebind layer 0 to owned
    dev_banks_0 = {"gate_up": torch.randn(4, 32, 8), "down": torch.randn(4, 8, 16)}
    cache.rebind_layer(0, HostResidency.GPU_OWNED.value, dev_banks_0)
    assert cache.owned_layer_count == 1

    # Attempting to rebind layer 1 to owned would leave 0 streaming layers -> refuse
    dev_banks_1 = {"gate_up": torch.randn(4, 32, 8), "down": torch.randn(4, 8, 16)}
    with pytest.raises(ValueError, match="every MoE layer is GPU-owned|zero streaming"):
        cache.rebind_layer(1, HostResidency.GPU_OWNED.value, dev_banks_1)


def test_rebind_disk_layer_properties():
    cache, _ = _make_dummy_cache(num_layers=3, num_experts=4, cache_size=8)

    assert not cache.has_disk_layers
    assert not cache.is_disk_layer(2)

    # Rebind layer 2 to disk (banks=None)
    cache.rebind_layer(2, HostResidency.DISK.value, None)

    assert cache.has_disk_layers
    assert cache.is_disk_layer(2)
    assert not cache.is_disk_layer(0)
    assert not cache.is_disk_layer(1)

    # Rebind back to pinned
    host_banks = {"gate_up": torch.randn(4, 32, 8), "down": torch.randn(4, 8, 16)}
    cache.rebind_layer(2, HostResidency.PINNED.value, host_banks)

    assert not cache.has_disk_layers
    assert not cache.is_disk_layer(2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rebind_pointer_tables_cuda():
    cache, sources = _make_dummy_cache(num_layers=3, num_experts=4, cache_size=8, device="cuda")

    assert cache._copy_fused_ok
    assert len(cache._copy_src_ptrs) == 3
    assert (cache._copy_src_ptrs[1] != 0).all()

    # Rebind layer 1 to GPU owned
    dev_banks = {
        "gate_up": torch.randn(4, 32, 8, device="cuda"),
        "down": torch.randn(4, 8, 16, device="cuda"),
    }
    cache.rebind_layer(1, HostResidency.GPU_OWNED.value, dev_banks)

    assert cache._copy_fused_ok
    assert len(cache._copy_src_ptrs) == 3
    assert (cache._copy_src_ptrs[1] == 0).all(), "owned layer pointer entry must stay 0"
    assert (cache._copy_src_ptrs[0] != 0).all(), "pinned layer 0 pointer must be non-zero"
    assert (cache._copy_src_ptrs[2] != 0).all(), "pinned layer 2 pointer must be non-zero"

    # Rebind back to pinned
    from freetoken.kernel.pinned import alloc_pinned_tensor
    host_banks = {
        "gate_up": alloc_pinned_tensor(4, 32, 8, dtype=torch.float32),
        "down": alloc_pinned_tensor(4, 8, 16, dtype=torch.float32),
    }
    cache.rebind_layer(1, HostResidency.PINNED.value, host_banks)

    assert cache._copy_fused_ok
    assert len(cache._copy_src_ptrs) == 3
    assert (cache._copy_src_ptrs[1] != 0).all(), "pinned layer 1 pointer must resolve"

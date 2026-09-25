"""Sleep on a real card (the serving box; skips without CUDA). Run by the controller (plan Task 12).

What the CPU harness (test_engine_sleep.py, test_offload_release_slots.py) cannot show: that
``release_slots`` hands the slot cache's bytes back to the CUDA driver (WSL/WDDM included), and
that a ``rebuild`` after it lands on the same size and the same fused copy plan. Needs about
0.6 GB free on the card: 512 MiB of slot cache plus 64 MiB of pinned host banks.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (serving box)")

E, SLOTS = 64, 2048
ROW = 256 * 256 * 2  # one bf16 [256, 256] row


def _cuda_cache() -> OffloadMoeCache:
    cache = OffloadMoeCache(num_layers=4, num_experts=E, cache_size=SLOTS, device=torch.device("cuda"))
    sources = {
        name: [torch.randn(E, 256, 256, dtype=torch.bfloat16).pin_memory() for _ in range(4)]
        for name in ("gate_up", "down")
    }
    cache.set_bank_sources(sources, layer_residency=["pinned"] * 4, gpu_owned_layers=frozenset())
    return cache


def _free() -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


def test_release_slots_hands_the_bytes_back_to_the_card():
    cache = _cuda_cache()
    before = _free()
    freed = cache.release_slots()
    assert freed == SLOTS * ROW * 2  # two banks
    # release_slots ends with empty_cache, so the driver's free count must grow by what the
    # slot cache held; 5% is headroom for allocator rounding. A miss here means the platform
    # is not returning freed VRAM (a design risk-table finding, not something to loosen).
    assert _free() - before >= 0.95 * freed


def test_a_released_cache_rebuilds_to_the_same_size_and_copy_plan():
    cache = _cuda_cache()
    fused = cache._copy_fused_ok
    before = _free()
    cache.release_slots()
    cache.rebuild(SLOTS)
    assert cache.cache_size == SLOTS and cache.bank_caches["gate_up"].shape == (SLOTS, 256, 256)
    assert cache._copy_fused_ok == fused
    # Back to the boot footprint: the rebuilt cache is the released one's size, and the
    # bookkeeping (id_of_slot, usage, the copy-plan descriptors) is a few KB. 64 MiB covers
    # allocator segment rounding on either side.
    assert abs(_free() - before) <= 64 << 20

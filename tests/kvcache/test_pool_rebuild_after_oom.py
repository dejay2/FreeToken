"""A pool rebuild that dies mid-allocation must leave a pool the NEXT rebuild can still re-make.

The failed-wake path (engine/sleep.py wake_engine -> release_to_sleep force_pools) rebuilds the
KV and GDN pools straight after an OOM inside one of those same rebuilds. The rebuilds used to
read their geometry off the old tensors, which the OOM had already set to None, so the way back
to sleep died with AttributeError and the scheduler latched failed. The layout is now recorded
at construction; these tests fail one allocation and check the retry rebuilds the pool whole.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.distributed.info import DistributedInfo
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.mha_pool import MHAKVCache
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import KVCacheGroupSpec, LinearGatedDeltaGroupConfig

DEV = torch.device("cpu")


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    tp = lambda: DistributedInfo(rank=0, size=1)  # noqa: E731
    monkeypatch.setattr("freetoken.kvcache.mha_pool.get_tp_info", tp)
    monkeypatch.setattr("freetoken.kvcache.hybrid_swa_pool.get_tp_info", tp)


def fail_once(monkeypatch, name: str) -> None:
    """The next torch.<name> call raises a CUDA-style OOM; later calls go through."""
    real = getattr(torch, name)
    state = {"armed": True}

    def alloc(*args, **kwargs):
        if state["armed"]:
            state["armed"] = False
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB")
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, name, alloc)


def test_mha_pool_rebuilds_after_an_oom_left_its_buffer_none(monkeypatch):
    pool = MHAKVCache(num_kv_heads=2, num_layers=4, head_dim=8, num_pages=10, page_size=16,
                      dtype=torch.bfloat16, device=DEV, layer_ids=(1, 3))
    fail_once(monkeypatch, "empty")
    with pytest.raises(torch.OutOfMemoryError):
        pool.rebuild(200)
    assert pool._kv_buffer is None
    pool.rebuild(2)
    assert pool._kv_buffer.shape == (2, 2, 2, 16, 2, 8)
    assert pool._kv_buffer.dtype == torch.bfloat16
    assert pool.k_cache(3).shape == (2, 16, 2, 8)


@pytest.mark.parametrize("which", ["empty", "zeros"])  # the K/V slab, or the index tiers after it
def test_qsa_pool_rebuilds_after_an_oom(monkeypatch, which):
    pool = QSAKVCache(num_kv_heads=2, num_layers=8, head_dim=64, num_pages=4, page_size=64,
                      dtype=torch.bfloat16, device=DEV, index_head_dim=32, num_index_layers=4,
                      index_ratio=4, num_req_slots=4, layer_ids=(1, 3, 5, 7),
                      kv_dtype=torch.float8_e4m3fn)
    fail_once(monkeypatch, which)
    with pytest.raises(torch.OutOfMemoryError):
        pool.rebuild(40)
    assert pool._kv_buffer is None
    pool.rebuild(2)
    assert pool._kv_buffer.shape == (2, 4, 2, 64, 2, 64)
    assert pool._kv_buffer.dtype == torch.float8_e4m3fn
    assert pool._kv_scale_buffer.shape == (2, 4, 2, 64, 2)
    assert pool.cmp_scratch_base == 2 * 64 // 4


def test_linear_state_pool_rebuilds_after_an_oom(monkeypatch):
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=8, dtype=torch.bfloat16, device=DEV, tp_size=1)
    conv_shape, rec_shape = pool.conv_states.shape[2:], pool.recurrent_states.shape[2:]
    rec_dtype = pool.recurrent_states.dtype
    fail_once(monkeypatch, "zeros")
    with pytest.raises(torch.OutOfMemoryError):
        pool.rebuild(64)
    assert pool.conv_states is None and pool.num_slots == 8  # the count the bug report names
    pool.rebuild(1)
    assert pool.num_slots == 1
    assert pool.conv_states.shape == (2, 1, *conv_shape) and pool.conv_states.dtype == torch.bfloat16
    assert pool.recurrent_states.shape == (2, 1, *rec_shape)
    assert pool.recurrent_states.dtype == rec_dtype


def test_hybrid_swa_pool_rebuilds_after_an_oom(monkeypatch):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    groups = (
        KVCacheGroupSpec(name="full", layer_ids=(2, 5), num_kv_heads=2, head_dim=64,
                         sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0, 1, 3, 4), num_kv_heads=4, head_dim=32,
                         sliding_window=64),
    )
    pool = HybridSWAKVCache(groups=groups, num_layers=6, num_full_pages=4, page_size=16,
                            num_swa_tokens=32, dtype=torch.bfloat16, device=DEV)
    fail_once(monkeypatch, "empty")
    with pytest.raises(torch.OutOfMemoryError):
        pool.rebuild(400, 800)
    assert pool.full_kv_pool is None
    pool.rebuild(2, 16)
    assert pool.k_cache(2).shape == (2, 16, 2, 64)
    assert pool.k_cache(0).shape == (16, 1, 4, 32)
    assert pool.full_num_tokens == 32

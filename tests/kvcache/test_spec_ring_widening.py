"""The QSA pending ring must widen with the speculative depth, on the LIVE pool.

Design risk #1, the one that fails silently: the ring row is ``position % ring_capacity``
with no head pointer and no epoch tag (``attention/qsa_sparse.py`` ``_plan_index_writes``),
so at the default ``ring_capacity == index_ratio == 4`` a 4-row speculative step writing
positions P..P+3 aliases onto P-4..P-1 -- exactly the still-needed members of the open
compression group. Wrong compressed keys, wrong block selection, plausible-but-wrong text,
no crash.

The plumbing already existed (``create_kvcache_pool(num_speculative_tokens=...)``); the engine
boot never passed it, and ``kv_cost`` never priced it. Both are driven here off the SAME
``num_speculative_tokens`` on the engine config, so the pool and the boot budget cannot
disagree. With speculation off, every number below is today's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache import create_kv_pool
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import KVCacheGroupSpec

DEV = torch.device("cpu")


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _spec(index_ratio=4):
    return KVCacheGroupSpec(
        name="full",
        layer_ids=(1, 3, 5, 7),
        num_kv_heads=2,
        head_dim=64,
        sliding_window=None,
        index_head_dim=32,
        num_index_layers=4,
        index_ratio=index_ratio,
        attn_type=AttnType.QSA,
    )


def _config(spec, *, num_speculative_tokens=0, max_running_req=3, page_size=64):
    mc = SimpleNamespace(
        num_layers=8,
        has_swa_attention=False,
        has_linear_attention=True,
        num_kv_heads=spec.num_kv_heads,
        head_dim=spec.head_dim,
    )
    mc.kv_cache_group_specs = lambda: (spec,)
    return SimpleNamespace(
        model_config=mc,
        page_size=page_size,
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(size=1),
        max_running_req=max_running_req,
        cache_type="radix",
        num_speculative_tokens=num_speculative_tokens,
    )


# ------------------------------------------------------------------ the capacity arithmetic


def test_the_widened_capacity_is_eight_for_ratio_four_at_depth_three():
    # ring_capacity_for = ratio * ceil((ratio + depth) / ratio) = 4 * ceil(7/4) = 8
    assert QSAKVCache.ring_capacity_for(4, 3) == 8
    assert QSAKVCache.ring_capacity_for(4, 0) == 4


@pytest.mark.parametrize("ratio", [2, 4, 8, 16])
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_the_capacity_always_covers_a_whole_group_plus_the_drafts(ratio, depth):
    # The invariant the compressor needs: a closing group reads up to ratio - 1 past members,
    # and a w = 1 + depth row step writes depth more rows on top of them.
    capacity = QSAKVCache.ring_capacity_for(ratio, depth)
    assert capacity >= ratio + depth
    assert capacity % ratio == 0


# --------------------------------------------------------------------- the live pool + budget


def test_the_engine_pool_keeps_todays_ring_when_speculation_is_off():
    config = _config(_spec())
    pool = create_kv_pool(config, num_pages=4, device=DEV, dtype=torch.bfloat16)
    assert pool.ring_capacity == 4
    assert pool.pending_ring(0).shape == (4, 4, 32)
    assert pool.pending_position_ring(0).shape == (4, 4, 3)


def test_the_engine_pool_widens_its_ring_when_speculation_is_on():
    config = _config(_spec(), num_speculative_tokens=3)
    pool = create_kv_pool(config, num_pages=4, device=DEV, dtype=torch.bfloat16)
    assert pool.ring_capacity == 8
    assert pool.pending_ring(0).shape == (4, 8, 32)
    assert pool.pending_position_ring(0).shape == (4, 8, 3)
    # the per-token slider is untouched: only the fixed term grew
    off = create_kv_pool(_config(_spec()), num_pages=4, device=DEV, dtype=torch.bfloat16)
    assert pool.unit_bytes() == off.unit_bytes()


def test_the_kv_budget_counts_the_widened_ring():
    spec = _spec()
    row = 32 * 4 * 2  # index_head_dim * num_index_layers * 2 bytes
    slots = 4  # max_running_req + 1

    def fixed(depth):
        return QSAKVCache.kv_cost(_config(spec, num_speculative_tokens=depth))[1]

    def predicted(capacity):
        return slots * row * (capacity + 1) + slots * 4 * capacity * 3 * torch.int64.itemsize

    assert fixed(0) == predicted(4)
    assert fixed(3) == predicted(8)
    assert fixed(3) > fixed(0)
    # the per-page slider must not move with the depth
    assert QSAKVCache.kv_cost(_config(spec, num_speculative_tokens=3))[0] == (
        QSAKVCache.kv_cost(_config(spec))[0]
    )


def test_the_budget_and_the_pool_agree_on_the_same_config():
    config = _config(_spec(), num_speculative_tokens=3)
    pool = create_kv_pool(config, num_pages=4, device=DEV, dtype=torch.bfloat16)
    _per_page, fixed, _tokens, _reserve = QSAKVCache.kv_cost(config)
    ring_bytes = pool._pending_ring.numel() * pool._pending_ring.element_size()
    position_bytes = (
        pool._pending_position_ring.numel() * pool._pending_position_ring.element_size()
    )
    scratch_bytes = 4 * 4 * 32 * 2  # one scratch slab row per request slot, all index layers
    assert fixed == ring_bytes + position_bytes + scratch_bytes


def test_the_shipping_geometry_widens_from_four_to_eight():
    from freetoken.models.qwen4_exp.config import parse_config
    from tests.models.qwen4_exp.common import hf_config

    model_config = parse_config(hf_config())
    ratio = model_config.kv_cache_group_specs()[0].index_ratio
    assert ratio == 4
    assert QSAKVCache.ring_capacity_for(ratio) == 4
    assert QSAKVCache.ring_capacity_for(ratio, 3) == 8

"""FP8 main QSA K/V storage with an unchanged BF16 compressed index."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache.base import spec_kv_bytes_per_token
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import KVCacheGroupSpec

DEV = torch.device("cpu")
FULL_LAYER_IDS = tuple(range(3, 48, 4))
BF16_BYTES_PER_TOKEN = 25_344
FP8_BYTES_PER_TOKEN = 13_248


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _spec(
    *,
    attn_type=AttnType.QSA,
    num_kv_heads=2,
    head_dim=256,
    index_head_dim=128,
    num_index_layers=12,
    layer_ids=FULL_LAYER_IDS,
):
    return KVCacheGroupSpec(
        name="full",
        layer_ids=layer_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=None,
        index_head_dim=index_head_dim,
        num_index_layers=num_index_layers,
        index_ratio=4,
        attn_type=attn_type,
    )


def _config(kv_dtype: str, spec=None):
    spec = spec or _spec()
    model_config = SimpleNamespace(
        num_layers=48,
        has_swa_attention=False,
        has_linear_attention=True,
        num_kv_heads=2,
        head_dim=256,
        dsv4_args=None,
    )
    model_config.kv_cache_group_specs = lambda: (spec,)
    return SimpleNamespace(
        model_config=model_config,
        page_size=64,
        dtype=torch.bfloat16,
        kv_dtype=kv_dtype,
        tp_info=SimpleNamespace(size=1),
        max_running_req=3,
        num_speculative_tokens=0,
        cache_type="radix",
    )


def _pool(kv_dtype: torch.dtype | None = None, num_pages: int = 4):
    return QSAKVCache(
        num_kv_heads=2,
        num_layers=8,
        head_dim=64,
        num_pages=num_pages,
        page_size=64,
        dtype=torch.bfloat16,
        kv_dtype=kv_dtype,
        device=DEV,
        index_head_dim=32,
        num_index_layers=4,
        index_ratio=4,
        num_req_slots=4,
        layer_ids=(1, 3, 5, 7),
    )


def test_bf16_default_keeps_the_existing_pool_layout():
    pool = _pool()

    assert pool.dtype is torch.bfloat16
    assert pool.kv_dtype is torch.bfloat16
    assert pool.k_cache(1).dtype is torch.bfloat16
    assert pool.cmp_k_cache(0).dtype is torch.bfloat16
    assert pool.k_scale(1) is None
    assert pool.v_scale(1) is None


def test_bf16_default_keeps_pre_fp8_allocations_and_launch_signature(monkeypatch):
    import freetoken.kernel.triton.qsa.attend as attend

    allocations = []
    original_empty = torch.empty
    original_zeros = torch.zeros

    def record(kind, factory, *size, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
        allocations.append((kind, shape, kwargs.get("dtype"), torch.device(kwargs["device"])))
        return factory(*size, **kwargs)

    monkeypatch.setattr(torch, "empty", lambda *size, **kwargs: record(
        "empty", original_empty, *size, **kwargs
    ))
    monkeypatch.setattr(torch, "zeros", lambda *size, **kwargs: record(
        "zeros", original_zeros, *size, **kwargs
    ))

    pool = _pool()

    assert allocations == [
        ("empty", (2, 4, 4, 64, 2, 64), torch.bfloat16, DEV),
        ("zeros", (4, 68, 32), torch.bfloat16, DEV),
        ("zeros", (4, 4, 4, 32), torch.bfloat16, DEV),
        ("zeros", (4, 4, 4, 3), torch.int64, DEV),
    ]
    assert pool.k_scale(1) is None
    assert pool.v_scale(1) is None
    monkeypatch.setattr(torch, "empty", original_empty)
    monkeypatch.setattr(torch, "zeros", original_zeros)

    class LaunchRecorder:
        def __getitem__(self, grid):
            self.grid = grid
            return self

        def __call__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    launch = LaunchRecorder()
    monkeypatch.setattr(attend, "_qsa_sparse_paged_gqa_splitk_kernel", launch)
    q = torch.empty((1, 4, 16), dtype=torch.bfloat16)
    k_cache = torch.empty((1, 64, 2, 16), dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    logical_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    token_to_req = torch.tensor([0], dtype=torch.int32)
    out = torch.empty_like(q)

    assert attend.qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
    ) is out
    assert launch.grid == (1, 2, 1)
    assert tuple(map(id, launch.args[:9])) == tuple(map(id, (
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out,
        out,
        out,
    )))
    assert launch.args[9:] == (
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
    )
    assert launch.kwargs == {
        "TOPK": 4,
        "PAGE_SIZE": 64,
        "PAGE_TABLE_WIDTH": 1,
        "GROUP_SIZE": 2,
        "HEAD_DIM": 16,
        "NUM_QUERY_HEADS": 4,
        "NUM_SPLITS": 1,
        "NUM_TILES": 1,
        "BLOCK_M": 2,
        "BLOCK_N": 16,
        "num_warps": 4,
        "num_stages": 2,
    }


def test_fp8_changes_only_main_kv_and_allocates_per_token_head_scales():
    pool = _pool(torch.float8_e4m3fn)

    assert pool.dtype is torch.bfloat16
    assert pool.kv_dtype is torch.float8_e4m3fn
    assert pool.k_cache(1).dtype is torch.float8_e4m3fn
    assert pool.v_cache(1).dtype is torch.float8_e4m3fn
    assert pool.cmp_k_cache(0).dtype is torch.bfloat16
    assert pool.pending_ring(0).dtype is torch.bfloat16
    assert pool.k_scale(1).shape == (4, 64, 2)
    assert pool.v_scale(1).shape == (4, 64, 2)
    assert pool.k_scale(1).dtype is torch.float32


def test_fp8_rebuild_resizes_scale_rows_with_kv():
    pool = _pool(torch.float8_e4m3fn, num_pages=4)
    ident = id(pool)

    pool.rebuild(9)

    assert id(pool) == ident
    assert pool.k_cache(7).shape == (9, 64, 2, 64)
    assert pool.k_scale(7).shape == (9, 64, 2)
    assert pool.v_scale(7).shape == (9, 64, 2)


@pytest.mark.parametrize(
    "kv_dtype,expected",
    [("bf16", BF16_BYTES_PER_TOKEN), ("fp8", FP8_BYTES_PER_TOKEN)],
)
def test_qsa_budget_prices_selected_main_kv_and_scale_metadata(kv_dtype, expected):
    config = _config(kv_dtype)

    assert spec_kv_bytes_per_token(_spec(), config) == expected
    per_page, _fixed, page_tokens, _floor = QSAKVCache.kv_cost(config)
    assert per_page == expected * page_tokens


def test_fp8_unit_bytes_include_four_scale_bytes_per_kv_head_and_slab():
    spec = _spec(
        num_kv_heads=2,
        head_dim=64,
        index_head_dim=32,
        num_index_layers=4,
        layer_ids=(1, 3, 5, 7),
    )
    config = _config("fp8", spec)
    config.model_config.num_layers = 8
    pool = _pool(torch.float8_e4m3fn)

    assert pool.unit_bytes() == (spec_kv_bytes_per_token(spec, config), 0)
    assert pool.unit_bytes()[0] == 2 * 2 * 64 * 4 + 2 * 2 * 4 * 4 + 32 * 4 * 2 // 4


def test_factory_maps_fp8_only_for_qsa():
    from freetoken.kvcache import create_kv_pool

    pool = create_kv_pool(
        _config("fp8"), num_pages=3, device=DEV, dtype=torch.bfloat16
    )
    assert pool.kv_dtype is torch.float8_e4m3fn
    assert pool.dtype is torch.bfloat16


def test_factory_rejects_fp8_for_non_qsa_pool():
    from freetoken.kvcache import create_kv_pool

    config = _config("fp8", _spec(attn_type=AttnType.FULL, index_head_dim=0,
                                  num_index_layers=0, layer_ids=tuple(range(4))))
    config.model_config.has_linear_attention = False
    config.model_config.num_layers = 4

    with pytest.raises(ValueError, match="QSA"):
        create_kv_pool(config, num_pages=3, device=DEV, dtype=torch.bfloat16)

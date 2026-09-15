"""Short tails must fetch selected experts but keep prefill arithmetic."""
from types import SimpleNamespace

import pytest
import torch

from freetoken.layers import moe


def test_small_nvfp4_tail_uses_slot_banks_with_prefill_kernel(monkeypatch):
    # Full-layer materialization or switching to decode arithmetic breaks this contract.
    from freetoken.distributed import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    layer = moe.OffloadMoELayer(0, 8, 2, 16, 32)
    ids = torch.tensor([[1, 3], [3, 1]], dtype=torch.int32)
    weights = torch.ones(2, 2)
    hidden = torch.ones(2, 16)
    events = []
    banks = (torch.arange(12),)
    def ensure(layer_id, routes, *, prefill=False):
        assert prefill is True
        events.append('ensure')
        routes.add_(4)  # slot IDs deliberately differ from expert IDs
    cache = SimpleNamespace(
        quant_format='nvfp4', decode_target='gpu', cache_size=12, num_experts=8,
        is_cpu_layer=lambda _: False, is_disk_layer=lambda _: False,
        is_gpu_owned_layer=lambda _: False, prefill_overlap=False,
        ensure_experts=ensure, copy_missing=lambda: events.append('copy'),
        bank_views=lambda: banks,
    )
    layer.offload_cache = cache
    def gemm(c, h, w, routes, **kwargs):
        assert c is cache and h is hidden and w is weights
        assert routes.tolist() == [[5, 7], [7, 5]]
        assert kwargs['views'] is banks
        assert kwargs['n'] == 12 and kwargs['is_prefill'] is True
        assert kwargs['slot_prefill'] is True
        events.append('prefill-gemm')
        return h + 1
    monkeypatch.setattr(layer, '_expert_gemm', gemm)
    monkeypatch.setattr(moe, '_SMALL_PREFILL_ROWS', 64)
    result = layer._prefill_routed(hidden, weights, ids)
    assert events == ['ensure', 'copy', 'prefill-gemm']
    torch.testing.assert_close(result, hidden + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('rows', [1, 8, 59, 64])
def test_sparse_nvfp4_prefill_matches_full_layer_after_slot_reuse(monkeypatch, rows):
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from tests.moe.test_nvfp4_backends import _make_native_sources, H, I, E, L, TOPK
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=E*2,
                           device=torch.device('cuda'), quant_format='nvfp4', prefill_overlap=True)
    cache.set_bank_sources(_make_native_sources(torch.device('cuda')))
    layer = moe.OffloadMoELayer(0, E, TOPK, H, I)
    layer.offload_cache = cache
    torch.manual_seed(9)
    h = torch.randn(rows, H, device='cuda', dtype=torch.bfloat16) / 4
    w = torch.rand(rows, TOPK, device='cuda')
    ids = torch.randint(0, E, (rows, TOPK), device='cuda', dtype=torch.int32)
    # A full-layer prefill interleaved with sparse prefills must not leave stale slots.
    for layer_id in [0, 1, 0]:
        layer.layer_id = layer_id
        monkeypatch.setattr(moe, '_SMALL_PREFILL_ROWS', 0)
        expected = layer._prefill_routed(h.clone(), w, ids.clone())
        monkeypatch.setattr(moe, '_SMALL_PREFILL_ROWS', 64)
        actual = layer._prefill_routed(h.clone(), w, ids.clone())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        repeat = layer._prefill_routed(h.clone(), w, ids.clone())
        torch.testing.assert_close(repeat, expected, rtol=0, atol=0)


@pytest.mark.parametrize('condition', ['disabled','large','owned-neighbour','prefetch','cpu','disk','capacity'])
def test_sparse_prefill_falls_back_before_touching_slots(monkeypatch, condition):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    layer=moe.OffloadMoELayer(0, 8, 2, 16, 32)
    events=[]
    cache=SimpleNamespace(quant_format='nvfp4', decode_target='gpu', cache_size=16,
        gpu_owned_layer_ids=set(), prefill_overlap=False,
        is_cpu_layer=lambda _:condition=='cpu', is_disk_layer=lambda _:condition=='disk',
        is_gpu_owned_layer=lambda _:False, materialize_layer=lambda _:events.append('full'),
        copy_missing=lambda:events.append('copy'), bank_views=lambda n:(), alphas_for_layer=lambda _:None)
    if condition=='owned-neighbour':cache.gpu_owned_layer_ids={1}
    if condition=='capacity':cache.cache_size=1
    if condition=='prefetch':monkeypatch.setattr(moe._prefetch,'PREFETCH',SimpleNamespace(enabled=True))
    layer.offload_cache=cache
    monkeypatch.setattr(moe, '_SMALL_PREFILL_ROWS', 0 if condition=='disabled' else 64)
    monkeypatch.setattr(layer,'_expert_gemm',lambda c,h,w,i,**kw:h)
    rows=65 if condition=='large' else 2
    layer._prefill_routed(torch.ones(rows,16),torch.ones(rows,2),torch.zeros(rows,2,dtype=torch.int32))
    assert events==['full','copy']


def test_nvfp4_stays_on_prefill_kernel_without_false_refusal_warning(monkeypatch, caplog):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    layer=moe.OffloadMoELayer(0, 8, 2, 16, 32)
    layer.offload_cache=SimpleNamespace(quant_format='nvfp4',decode_target='gpu')
    monkeypatch.setattr(moe,'_SMALL_PREFILL_ROWS',64)
    monkeypatch.setattr(moe,'_small_prefill_refused',set())
    assert layer._small_prefill_moves_like_decode(59) is False
    assert not caplog.records


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_prefill_lookup_handles_production_size_query_without_query_squared_dedup():
    from freetoken.moe.offload_cache import OffloadMoeCache
    cache=OffloadMoeCache(num_layers=2,num_experts=512,cache_size=7435,
                         device=torch.device('cuda'),quant_format='nvfp4')
    raw=(torch.arange(590,device='cuda',dtype=torch.int32)%512).reshape(59,10)
    ids=raw.clone()
    cache.ensure_experts(0,ids,prefill=True)
    assert cache.num_indices.item()==512
    torch.testing.assert_close(cache.id_of_slot[ids.long()],raw,rtol=0,atol=0)
    ids2=raw.clone()
    cache.ensure_experts(0,ids2,prefill=True)
    assert cache.num_indices.item()==0
    torch.testing.assert_close(ids2,ids,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize("hidden,intermediate", [(256, 128), (2560, 640)])
def test_sparse_prefill_gemm_with_production_slot_extent(monkeypatch, hidden, intermediate):
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from tests.moe import test_nvfp4_backends as fixture
    if try_get_tp_info() is None:set_tp_info(rank=0,size=1)
    monkeypatch.setattr(fixture,'E',512)
    monkeypatch.setattr(fixture,'H',hidden)
    monkeypatch.setattr(fixture,'I',intermediate)
    source=fixture._make_native_sources(torch.device('cuda'))
    cache=OffloadMoeCache(num_layers=2,num_experts=512,cache_size=7279,
                         device=torch.device('cuda'),quant_format='nvfp4')
    cache.set_bank_sources(source)
    layer=moe.OffloadMoELayer(0,512,10,fixture.H,fixture.I);layer.offload_cache=cache
    h=torch.randn(59,fixture.H,device='cuda',dtype=torch.bfloat16)/4
    ids=torch.randint(0,512,(59,10),device='cuda',dtype=torch.int32)
    w=torch.ones(59,10,device='cuda')/10
    # Fill high slot IDs so grouping must handle the real cache extent.
    cache.usage[:6000]=100;cache.step.fill_(100)
    monkeypatch.setattr(moe,'_SMALL_PREFILL_ROWS',64)
    actual=layer._prefill_routed(h.clone(),w,ids.clone())
    torch.cuda.synchronize()
    monkeypatch.setattr(moe,'_SMALL_PREFILL_ROWS',0)
    expected=layer._prefill_routed(h.clone(),w,ids.clone())
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    monkeypatch.setattr(moe,'_SMALL_PREFILL_ROWS',64)
    repeat=layer._prefill_routed(h.clone(),w,ids.clone())
    torch.testing.assert_close(repeat,expected,rtol=0,atol=0)

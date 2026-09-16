"""Lossy image pad IDs must never authorize cross-request KV reuse."""
import torch

from freetoken.core import SamplingParams
from freetoken.message import MMItem, UserMsg
from freetoken.mm import mm_pad_value
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.prefill import PrefillManager


def test_distinct_images_with_equal_pad_ids_do_not_match_radix(monkeypatch):
    import freetoken.kernel

    def compare_cpu(a, b):
        size = min(a.numel(), b.numel())
        different = (a[:size] != b[:size]).nonzero().flatten()
        return int(different[0]) if different.numel() else size

    monkeypatch.setattr(freetoken.kernel, "fast_compare_key", compare_cpu)
    hashes = (7, 7 + (1 << 30))
    assert mm_pad_value(hashes[0]) == mm_pad_value(hashes[1])
    cache = CacheManager(16, 1, torch.zeros((4, 16), dtype=torch.int32), 'radix')
    manager = PrefillManager(cache, None, None)
    ids = torch.tensor([10, mm_pad_value(hashes[0]), 20, 30], dtype=torch.int32)
    # Populate the colliding token prefix to prove the private path ignores a real hit.
    cache.prefix_cache.insert_prefix(ids[:3], torch.arange(3, dtype=torch.int32))
    for uid, image_hash in enumerate(hashes):
        item = MMItem('image', image_hash, mm_pad_value(image_hash), [[1, 2]], feature=torch.tensor([float(uid)]))
        manager.add_one_req(UserMsg(uid, ids.clone(), SamplingParams(max_tokens=1), mm_items=[item]))
        pending = manager.pending_list[-1]
        assert pending.cache_private
        assert cache.match_req(pending).cuda_handle.cached_len == 0
    # Text-only matching still uses the shared radix normally.
    manager.add_one_req(UserMsg(2, ids.clone(), SamplingParams(max_tokens=1)))
    assert cache.match_req(manager.pending_list[-1]).cuda_handle.cached_len == 3

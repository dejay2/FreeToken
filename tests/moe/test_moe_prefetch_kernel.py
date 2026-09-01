"""The prefetch ensure kernel against a CPU-simulated LRU with step protection.

The kernel is the only piece of the prefetch that can corrupt the cache: it evicts slots
while the layer that owns the current step is still reading them. So it is checked the way
the hybrid ensure kernel is -- decision for decision against a python reference, over
random traces driven by the REAL ``ensure_experts`` so the ``step``/``usage`` state it
protects is the state a live decode actually produces.

Two invariants, on every step of every trace:

* the whole index (``slot_for_id``, ``id_of_slot``, ``usage``) and the emitted copy plan
  match :func:`prefetch_ensure_reference` exactly, and
* no slot stamped with the current step is ever chosen as a victim -- that is the
  protection the ordering argument in :mod:`freetoken.moe.prefetch` rests on.
"""

from __future__ import annotations

import pytest
import torch

import freetoken.moe.prefetch as pf

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E, C = 4, 32, 48
TOP_K, KPRIME, MAX_MISSES = 4, 6, 3


def _armed_cache(monkeypatch, *, max_misses=MAX_MISSES, cache_size=C, collect=True):
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setattr(
        pf,
        "PREFETCH",
        pf.PrefetchConfig(
            enabled=True, topk=KPRIME, max_misses=max_misses, skip_layers=frozenset()
        ),
    )
    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=cache_size, device=torch.device("cuda")
    )
    cache.collect_stats = collect
    return cache


def _snapshot(cache):
    return (
        cache.slot_for_id.cpu().clone(),
        cache.id_of_slot.cpu().clone(),
        cache.usage.cpu().clone(),
    )


def _run_pair(cache, layer_id, pred, max_misses):
    """Reference on a CPU snapshot, kernel on the device; return both plans."""
    slot, ids, usage = _snapshot(cache)
    step = int(cache.step.item())
    protected = {s for s in range(cache.cache_size) if int(usage[s]) == step}
    ref_dst, ref_src = pf.prefetch_ensure_reference(
        pred_ids=pred.cpu(),
        slot_for_id=slot,
        id_of_slot=ids,
        usage=usage,
        step=step,
        layer_id=layer_id,
        num_experts=cache.num_experts,
        cache_size=cache.cache_size,
        max_misses=max_misses,
    )
    pf.prefetch_ensure(cache, layer_id, pred, max_misses)
    torch.cuda.synchronize()
    n = int(cache.prefetch_num_indices.item())
    got_dst = cache.prefetch_evict_slots[:n].cpu().tolist()
    got_src = cache.prefetch_src_indices[:n].cpu().tolist()
    assert not (set(got_dst) & protected), (
        f"prefetch evicted a slot stamped with the live step {step}: "
        f"{sorted(set(got_dst) & protected)}"
    )
    assert (got_dst, got_src) == (ref_dst, ref_src)
    got = _snapshot(cache)
    assert torch.equal(got[0], slot), "slot_for_id diverged from the reference"
    assert torch.equal(got[1], ids), "id_of_slot diverged from the reference"
    assert torch.equal(got[2], usage), "usage diverged from the reference"
    return ref_dst


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_prefetch_ensure_matches_the_cpu_reference_over_random_traces(monkeypatch, seed):
    cache = _armed_cache(monkeypatch)
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(seed)
    rows = 1 + seed % 3  # 1 (plain decode) .. 3 (an MTP verify block)

    fetched_total = 0
    for _ in range(24):
        for layer in range(L - 1):
            ids = torch.randint(0, E, (rows, TOP_K), generator=gen, dtype=torch.int32)
            cache.ensure_experts(layer, ids.to(dev))  # advances step, stamps usage
            pred = torch.randint(0, E, (rows, KPRIME), generator=gen, dtype=torch.int32)
            fetched_total += len(_run_pair(cache, layer + 1, pred.to(dev), MAX_MISSES))
    assert fetched_total > 20, "the trace never exercised an eviction; nothing was proven"


def test_prefetch_admits_the_highest_scored_predictions_first(monkeypatch):
    """The bandwidth cap has to spend its budget on the top of the topk, not on ids."""
    cache = _armed_cache(monkeypatch, max_misses=2)
    dev = torch.device("cuda")
    cache.ensure_experts(0, torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=dev))
    # Descending predicted score, deliberately ASCENDING nowhere near expert-id order.
    pred = torch.tensor([[31, 5, 17, 9, 20, 11]], dtype=torch.int32, device=dev)
    pf.prefetch_ensure(cache, 1, pred, 2)
    torch.cuda.synchronize()
    n = int(cache.prefetch_num_indices.item())
    assert n == 2
    assert cache.prefetch_src_indices[:n].cpu().tolist() == [31, 5]


def test_multi_row_prediction_is_ranked_rank_major(monkeypatch):
    """With an MTP verify block every row's top-1 outranks any row's top-2."""
    cache = _armed_cache(monkeypatch, max_misses=3)
    dev = torch.device("cuda")
    cache.ensure_experts(0, torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=dev))
    pred = torch.tensor(
        [[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device=dev
    )  # rows x k'
    pf.prefetch_ensure(cache, 1, pred, 3)
    torch.cuda.synchronize()
    assert cache.prefetch_src_indices[:3].cpu().tolist() == [10, 20, 11]


def test_predicted_hits_are_touched_and_never_evicted_by_the_same_call(monkeypatch):
    """A prefetch must not evict the very experts it just confirmed resident."""
    cache = _armed_cache(monkeypatch, max_misses=4)
    dev = torch.device("cuda")
    # Layer 2's experts 0..3 go resident first, and then age (three more ensure calls).
    cache.ensure_experts(2, torch.arange(0, 4, dtype=torch.int32, device=dev).view(1, 4))
    hit_slots = set(cache.slot_for_id[2, 0:4].cpu().tolist())
    for layer in (0, 1, 0):
        cache.ensure_experts(
            layer, torch.arange(0, 4, dtype=torch.int32, device=dev).view(1, 4)
        )
    step = int(cache.step.item())
    live = {s for s in range(C) if int(cache.usage[s]) == step}
    assert not (hit_slots & live), "the aged hits should not already be step-protected"

    # Predict layer 2's four resident experts (hits) plus four it has never seen (misses).
    pred = torch.tensor([[0, 1, 2, 3, 20, 21, 22, 23]], dtype=torch.int32, device=dev)
    pf.prefetch_ensure(cache, 2, pred, 4)
    torch.cuda.synchronize()
    taken = set(cache.prefetch_evict_slots[:4].cpu().tolist())
    assert not (taken & hit_slots), "a prefetch evicted its own predicted hits"
    assert not (taken & live), "a prefetch evicted a slot the running layer is reading"
    # The hits were touched, so they now rank as freshly used.
    assert all(int(cache.usage[s]) == step for s in hit_slots)


def test_stats_count_predictions_hits_fetches_and_usefulness(monkeypatch):
    cache = _armed_cache(monkeypatch, max_misses=8)
    dev = torch.device("cuda")
    cache.ensure_experts(0, torch.tensor([[0, 1]], dtype=torch.int32, device=dev))
    pred = torch.tensor([[3, 4, 5, 6]], dtype=torch.int32, device=dev)
    pf.prefetch_ensure(cache, 1, pred, 8)
    # Layer 1 then really routes to 3, 4 and 9: two of the four predictions were useful.
    actual = torch.tensor([[3, 4, 9]], dtype=torch.int32, device=dev)
    cache.prefetch_note_actual(1, actual)
    torch.cuda.synchronize()
    s = cache.prefetch_stats_summary()
    assert s["calls"] == 1
    assert s["predicted_per_call"] == 4.0
    assert s["fetched_per_call"] == 4.0  # cold cache: every prediction was a miss
    assert s["useful_per_call"] == 2.0
    assert s["useful_fetched_per_call"] == 2.0
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(2 / 3)
    # ...and the routing report carries it, prefixed, without needing the histogram.
    assert cache.decode_routing_stats()["prefetch_precision"] == pytest.approx(0.5)


def test_max_misses_caps_the_bus_bytes(monkeypatch):
    cache = _armed_cache(monkeypatch, max_misses=2)
    dev = torch.device("cuda")
    cache.ensure_experts(0, torch.tensor([[0, 1]], dtype=torch.int32, device=dev))
    pred = torch.arange(10, 16, dtype=torch.int32, device=dev).view(1, 6)
    pf.prefetch_ensure(cache, 1, pred, 2)
    torch.cuda.synchronize()
    assert int(cache.prefetch_num_indices.item()) == 2
    s = cache.prefetch_stats_summary()
    assert s["missing_per_call"] == 6.0 and s["capped_per_call"] == 4.0
    # The four capped predictions stay non-resident: nothing half-inserted.
    resident = (cache.slot_for_id[1] >= 0).sum().item()
    assert resident == 2


def test_reset_clears_the_prefetch_marks_and_stats(monkeypatch):
    cache = _armed_cache(monkeypatch)
    dev = torch.device("cuda")
    cache.ensure_experts(0, torch.tensor([[0, 1]], dtype=torch.int32, device=dev))
    pf.prefetch_ensure(cache, 1, torch.tensor([[2, 3]], dtype=torch.int32, device=dev), 2)
    torch.cuda.synchronize()
    assert int(cache.prefetch_stats[pf.STAT_CALLS].item()) == 1
    cache.reset()
    torch.cuda.synchronize()
    assert int(cache.prefetch_stats.sum().item()) == 0
    assert int(cache.prefetch_mark.max().item()) == -1

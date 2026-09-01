"""Hybrid MoE decode (``decode_target == "hybrid"``) under CUDA-graph capture.

The hybrid decode path (``OffloadMoELayer._decode_hybrid``) is the only decode mode that
mixes GPU expert compute (cache hits + a capped PCIe fetch) with CPU expert compute (the
overflow misses, computed from the host banks by the ``_cpu_moe`` worker pool). Both halves
claim to be capture-safe -- the routing split is device-side elementwise, the CPU submit/sync
are host nodes -- so this asserts the claim end to end at MTP-verify width (4 rows):

* capture ``_decode_routed`` for a hybrid cache, then
* replay it with NEW routing and NEW activations, and check the result still equals the dense
  reference. That is the only interesting invariant: the routing split is recomputed on the
  GPU every replay, so a baked-in split (or a CPU partial reading capture-time pinned data)
  would double-count or drop routes and miss the reference by whole units.

Deliberately tiny (a few MiB of GPU memory): this is a capture-legality proof, not a bench.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as Fn

from freetoken.distributed import set_tp_info, try_get_tp_info

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E, H, I, TOP_K = 2, 8, 256, 128, 2
LAYER = 0
BS = 4  # the w=4 MTP verify batch


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _make_hybrid_cache(device: torch.device, *, max_fetch: int):
    """A real ``OffloadMoeCache`` in hybrid mode over pinned bf16 banks."""
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.offload_cache import OffloadMoeCache

    gate_up, down = [], []
    for _ in range(L):
        gu = alloc_pinned_tensor(E, 2 * I, H, dtype=torch.bfloat16)
        dn = alloc_pinned_tensor(E, H, I, dtype=torch.bfloat16)
        gu.copy_(torch.randn(E, 2 * I, H) * 0.1)
        dn.copy_(torch.randn(E, H, I) * 0.1)
        gate_up.append(gu)
        down.append(dn)

    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        # cache_size == num_experts: every step past the first evicts, so replays keep
        # producing a live hit/fetch/CPU split instead of settling into an all-hit cache.
        cache_size=E,
        device=device,
        decode_target="hybrid",
        hybrid_max_fetch=max_fetch,
    )
    cache.set_bank_sources({"gate_up": gate_up, "down": down})
    cache.collect_stats = True  # the production hybrid config; its accumulators are captured
    return cache


def _make_layer(cache):
    from freetoken.layers.moe import OffloadMoELayer

    _init_tp()
    layer = OffloadMoELayer(
        layer_id=LAYER,
        num_experts=E,
        top_k=TOP_K,
        hidden_size=H,
        intermediate_size=I,
    )
    layer.offload_cache = cache
    return layer


def _attach_executor(cache, device: torch.device):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    executor = CpuMoeExecutor(
        cache,
        top_k=TOP_K,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=4,
        max_tokens=BS,
        device=device,
    )
    cache.set_cpu_executor(executor)
    return executor


def _dense_reference(cache, layer_id, hidden, ids, w):
    """Full routed-expert output, ignoring the hybrid split entirely.

    ``gpu_routed + cpu_routed`` must equal this: each route is computed exactly once,
    whichever side computed it.
    """
    gu = cache.bank_sources["gate_up"][layer_id].float()
    dn = cache.bank_sources["down"][layer_id].float()
    xf = hidden.float().cpu()
    out = torch.zeros(ids.shape[0], H, dtype=torch.float32)
    for t in range(ids.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            gate = gu[e, :I, :] @ xf[t]
            up = gu[e, I : 2 * I, :] @ xf[t]
            g = (Fn.silu(gate) * up).bfloat16().float()
            out[t] += float(w[t, k]) * (dn[e] @ g)
    return out


def _random_routing(device):
    ids = torch.stack([torch.randperm(E)[:TOP_K] for _ in range(BS)]).to(torch.int32)
    w = torch.rand(BS, TOP_K, dtype=torch.float32)
    hidden = torch.randn(BS, H, dtype=torch.bfloat16)
    return hidden, ids, w


def _rel_error(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got.float().cpu() - ref).abs().max() / ref.abs().max())


def test_hybrid_decode_captures_and_replays_with_fresh_routing():
    device = torch.device("cuda")
    torch.manual_seed(0)

    stream = torch.cuda.Stream()
    entry_stream = torch.cuda.current_stream()
    cache = _make_hybrid_cache(device, max_fetch=1)
    layer = _make_layer(cache)
    torch.cuda.set_stream(stream)
    try:
        _attach_executor(cache, device)

        hidden, ids, w = _random_routing(device)
        d_hidden = hidden.to(device)
        d_ids = ids.to(device)
        d_w = w.to(device)

        # Eager pass: materializes the pinned CPU-executor IO buffers + host-func task,
        # the triton ensure kernel, and the JIT'd fused index copy -- everything the
        # capture must find already built. NB: _decode_routed rewrites d_ids in place.
        raw = d_ids.clone()
        eager = layer._decode_routed(d_hidden, d_w, d_ids)
        torch.cuda.synchronize()
        ref = _dense_reference(cache, LAYER, hidden, ids, w)
        assert _rel_error(eager, ref) < 2e-2, "eager hybrid decode does not match the dense reference"
        assert not torch.equal(d_ids, raw), "ensure_experts_hybrid did not rewrite the ids"

        # Capture. A capture-illegal op in the hybrid path surfaces here.
        graph = torch.cuda.CUDAGraph()
        d_ids.copy_(ids.to(device))
        with torch.cuda.graph(graph, stream=stream):
            captured = layer._decode_routed(d_hidden, d_w, d_ids)
        torch.cuda.synchronize()

        # Replay with brand-new routing/activations. The split between the GPU partial and
        # the CPU partial is recomputed device-side each replay; the sum must still be the
        # dense answer.
        overflowed = 0
        for it in range(4):
            torch.manual_seed(100 + it)
            new_hidden, new_ids, new_w = _random_routing(device)
            d_hidden.copy_(new_hidden)
            d_ids.copy_(new_ids)  # raw expert ids again; the graph rewrites them to slots
            d_w.copy_(new_w)
            graph.replay()
            torch.cuda.synchronize()
            new_ref = _dense_reference(cache, LAYER, new_hidden, new_ids, new_w)
            err = _rel_error(captured, new_ref)
            assert err < 2e-2, f"hybrid replay {it} rel err {err}"
            if int(cache.num_missing_full.item()) > int(cache.num_indices.item()):
                overflowed += 1
        assert overflowed, "no replay overflowed to the CPU; the CPU partial went untested"

        # The split must have been live, not degenerate: with max_fetch=1 and a cache the
        # size of one expert layer, most of each step's misses go to the CPU.
        missing = int(cache.stat_missing.item())
        fetched = int(cache.stat_fetched.item())
        assert missing > fetched > 0, (
            f"expected a live hybrid split, got missing={missing} fetched={fetched}"
        )
    finally:
        torch.cuda.set_stream(entry_stream)


def test_hybrid_decode_capture_matches_eager_split_accounting():
    """The captured graph's own ensure kernel keeps the movement counters reconciling.

    ``missing_full`` and ``num_indices`` are written by the captured kernel on every replay;
    if either were baked at capture time the per-replay accounting would go flat.
    """
    device = torch.device("cuda")
    torch.manual_seed(7)

    stream = torch.cuda.Stream()
    entry_stream = torch.cuda.current_stream()
    cache = _make_hybrid_cache(device, max_fetch=2)
    layer = _make_layer(cache)
    torch.cuda.set_stream(stream)
    try:
        _attach_executor(cache, device)
        hidden, ids, w = _random_routing(device)
        d_hidden, d_ids, d_w = hidden.to(device), ids.to(device), w.to(device)
        layer._decode_routed(d_hidden, d_w, d_ids)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        d_ids.copy_(ids.to(device))
        with torch.cuda.graph(graph, stream=stream):
            layer._decode_routed(d_hidden, d_w, d_ids)
        torch.cuda.synchronize()

        seen = []
        for it in range(4):
            torch.manual_seed(500 + it)
            new_hidden, new_ids, new_w = _random_routing(device)
            d_hidden.copy_(new_hidden)
            d_ids.copy_(new_ids)
            d_w.copy_(new_w)
            graph.replay()
            torch.cuda.synchronize()
            full = int(cache.num_missing_full.item())
            fetched = int(cache.num_indices.item())
            assert 0 <= fetched <= min(full, 2), (full, fetched)
            seen.append((full, fetched))
        assert any(full > fetched for full, fetched in seen), (
            f"the capped fetch never overflowed to the CPU: {seen}"
        )
    finally:
        torch.cuda.set_stream(entry_stream)

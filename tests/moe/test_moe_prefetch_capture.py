"""The layer-ahead expert prefetch end to end: numerics, capture, and the join.

Three claims, on a deliberately tiny (a few MiB) three-layer offload MoE:

1. **It changes nothing.** The prefetch only moves bytes into slots nobody is reading and
   writes index entries for a layer that has not run. Prefetch on and prefetch off must
   produce bit-identical activations, layer for layer, step for step.
2. **It captures.** The fork/join onto a second stream happens INSIDE the region a decode
   CUDA graph captures, so it has to survive capture and replay with fresh routing.
3. **The join is real.** With the prefetch copy made deliberately slow, the next layer's
   GEMV must still read landed bytes -- and with the join removed it must not, or the test
   would be proving nothing.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as Fn

import freetoken.layers.moe as moe_layers
import freetoken.moe.prefetch as pf
from freetoken.distributed import set_tp_info, try_get_tp_info

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E, H, I, TOP_K = 3, 8, 128, 64, 2
CACHE = 12
KPRIME, MAX_MISSES = 3, 2


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


class _Gate:
    """A router with the shape the real one has: ``forward(x) -> [rows, E]``."""

    def __init__(self, seed: int, device) -> None:
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(E, H, generator=g).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T.to(x.dtype)


@pytest.fixture
def banks():
    """One set of pinned host expert banks, shared by every cache the test builds."""
    from freetoken.kernel.pinned import alloc_pinned_tensor

    torch.manual_seed(11)
    gate_up, down = [], []
    for _ in range(L):
        gu = alloc_pinned_tensor(E, 2 * I, H, dtype=torch.bfloat16)
        dn = alloc_pinned_tensor(E, H, I, dtype=torch.bfloat16)
        gu.copy_(torch.randn(E, 2 * I, H) * 0.1)
        dn.copy_(torch.randn(E, H, I) * 0.1)
        gate_up.append(gu)
        down.append(dn)
    return {"gate_up": gate_up, "down": down}


def _make_cache(banks, *, prefetch: bool, monkeypatch, skip=frozenset()):
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setattr(
        pf,
        "PREFETCH",
        pf.PrefetchConfig(
            enabled=prefetch, topk=KPRIME, max_misses=MAX_MISSES, skip_layers=skip
        ),
    )
    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=CACHE, device=torch.device("cuda")
    )
    cache.set_bank_sources(banks)
    cache.collect_stats = True
    # Slot bytes start defined, so a layer reading a slot whose fill has not landed reads
    # zeros (a visibly wrong answer) rather than whatever the allocator left behind.
    for c in cache.bank_caches.values():
        c.zero_()
    return cache


def _make_layers(cache):
    from freetoken.layers.moe import OffloadMoELayer

    _init_tp()
    layers = []
    for lid in range(L):
        layer = OffloadMoELayer(
            layer_id=lid, num_experts=E, top_k=TOP_K, hidden_size=H, intermediate_size=I
        )
        layer.offload_cache = cache
        layers.append(layer)
    return layers


def _dense_reference(banks, layer_id, hidden, ids, w):
    gu = banks["gate_up"][layer_id].float()
    dn = banks["down"][layer_id].float()
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


def _rel_error(got, ref) -> float:
    return float((got.float().cpu() - ref).abs().max() / ref.abs().max())


def _routing(rows, gen):
    ids = torch.stack(
        [torch.randperm(E, generator=gen)[:TOP_K] for _ in range(rows * L)]
    ).to(torch.int32)
    return (
        torch.randn(rows, H, generator=gen, dtype=torch.float32).bfloat16(),
        ids.view(L, rows, TOP_K),
        torch.rand(L, rows, TOP_K, generator=gen, dtype=torch.float32),
    )


@pytest.fixture
def gates(monkeypatch):
    """Register a router per layer, the way the model does at construction."""
    routers = {lid: _Gate(lid, torch.device("cuda")) for lid in range(L)}
    monkeypatch.setattr(moe_layers, "_predict_routers", routers)
    return routers


def _run_step(layers, hidden, ids, w, device):
    outs = []
    for lid, layer in enumerate(layers):
        outs.append(
            layer._decode_routed(
                hidden, w[lid].to(device), ids[lid].to(device).clone()
            ).clone()
        )
    return outs


@pytest.mark.parametrize("rows", [1, 4])
def test_prefetch_does_not_change_a_single_activation(banks, gates, monkeypatch, rows):
    """Identical outputs with the prefetch on and off -- it is a cache policy, not math."""
    device = torch.device("cuda")
    off = _make_cache(banks, prefetch=False, monkeypatch=monkeypatch)
    on = _make_cache(banks, prefetch=True, monkeypatch=monkeypatch)
    assert off.prefetch_on is False and on.prefetch_on is True
    layers_off, layers_on = _make_layers(off), _make_layers(on)

    gen = torch.Generator().manual_seed(5)
    for _ in range(6):
        hidden, ids, w = _routing(rows, gen)
        d_hidden = hidden.to(device)
        a = _run_step(layers_off, d_hidden, ids, w, device)
        b = _run_step(layers_on, d_hidden, ids, w, device)
        torch.cuda.synchronize()
        for lid, (x, y) in enumerate(zip(a, b)):
            assert torch.equal(x, y), f"prefetch changed layer {lid}'s output"

    # ...and it was actually doing something: predictions were made and some were right.
    s = on.prefetch_stats_summary()
    assert s["calls"] >= 6 * (L - 1)
    assert s["useful_per_call"] > 0, "the predictor never named an expert the layer used"


def test_prefetch_captures_and_replays_with_fresh_routing(banks, gates, monkeypatch):
    """The fork/join lives inside the captured decode region and survives replay."""
    device = torch.device("cuda")
    cache = _make_cache(banks, prefetch=True, monkeypatch=monkeypatch)
    layers = _make_layers(cache)
    stream = torch.cuda.Stream()
    entry = torch.cuda.current_stream()
    torch.cuda.set_stream(stream)
    try:
        gen = torch.Generator().manual_seed(21)
        hidden, ids, w = _routing(1, gen)
        d_hidden = hidden.to(device)
        d_ids = [ids[lid].to(device) for lid in range(L)]
        d_w = [w[lid].to(device) for lid in range(L)]

        # Eager warm-up: JITs the triton kernels and the fused index copy, so the capture
        # finds everything built (a JIT compile inside capture is illegal).
        for lid, layer in enumerate(layers):
            layer._decode_routed(d_hidden, d_w[lid], d_ids[lid])
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        for lid in range(L):
            d_ids[lid].copy_(ids[lid])
        with torch.cuda.graph(graph, stream=stream):
            captured = [
                layers[lid]._decode_routed(d_hidden, d_w[lid], d_ids[lid])
                for lid in range(L)
            ]
        torch.cuda.synchronize()
        # Every fork was joined; an unbalanced one would have failed the capture above.
        assert not any(cache._prefetch_issued)

        for it in range(4):
            new_hidden, new_ids, new_w = _routing(1, gen)
            d_hidden.copy_(new_hidden.to(device))
            for lid in range(L):
                d_ids[lid].copy_(new_ids[lid])  # raw ids again; the graph rewrites them
                d_w[lid].copy_(new_w[lid])
            graph.replay()
            torch.cuda.synchronize()
            for lid in range(L):
                ref = _dense_reference(banks, lid, new_hidden, new_ids[lid], new_w[lid])
                err = _rel_error(captured[lid], ref)
                assert err < 2e-2, f"replay {it} layer {lid} rel err {err}"
    finally:
        torch.cuda.set_stream(entry)


def test_a_slow_prefetch_copy_is_still_waited_on(banks, gates, monkeypatch):
    """The join is a real dependency, not an accident of the copy being fast.

    ``torch.cuda._sleep`` stalls the side stream for ~20 ms between the fork and the copy.
    Layer 1 then runs; with the join it must read landed bytes. The control removes the
    join and shows the same run reading zeros -- so a regression that dropped the event
    would be caught rather than hidden by the copy finishing first anyway.
    """
    device = torch.device("cuda")

    def _run(*, join: bool) -> float:
        cache = _make_cache(banks, prefetch=True, monkeypatch=monkeypatch)
        layers = _make_layers(cache)
        original = cache._prefetch_copy

        def slow(target_layer):
            torch.cuda._sleep(40_000_000)  # ~20 ms on any modern GPU clock
            original(target_layer)

        cache._prefetch_copy = slow
        if not join:
            cache.prefetch_wait = lambda layer_id: False

        gen = torch.Generator().manual_seed(3)
        hidden, ids, w = _routing(1, gen)
        d_hidden = hidden.to(device)
        # Force the prediction for layer 1 to be exactly what layer 1 will route to, so
        # every one of layer 1's experts arrives through the (slow) prefetch copy.
        monkeypatch.setattr(
            moe_layers,
            "_predict_routers",
            {1: _ExactGate(ids[1].to(device))},
        )
        layers[0]._decode_routed(d_hidden, w[0].to(device), ids[0].to(device).clone())
        got = layers[1]._decode_routed(
            d_hidden, w[1].to(device), ids[1].to(device).clone()
        )
        torch.cuda.synchronize()
        ref = _dense_reference(banks, 1, hidden, ids[1], w[1])
        return _rel_error(got, ref)

    assert _run(join=True) < 2e-2, "layer 1 read bytes the prefetch had not landed"
    assert _run(join=False) > 0.5, (
        "removing the join did not corrupt the read; the test has no teeth"
    )


class _ExactGate:
    """A 'predictor' that always names a fixed expert set -- oracle-accurate on purpose."""

    def __init__(self, ids: torch.Tensor) -> None:
        self._ids = ids  # [rows, TOP_K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.shape[0], E), -1e4, device=x.device, dtype=torch.float32)
        for t in range(self._ids.shape[0]):
            for k in range(self._ids.shape[1]):
                logits[t, int(self._ids[t, k])] = 10.0 - k
        return logits


def test_skip_layers_leaves_the_fork_and_join_balanced(banks, gates, monkeypatch):
    """A skipped target means no fork AND no join -- capture would fail if they diverged."""
    device = torch.device("cuda")
    cache = _make_cache(banks, prefetch=True, monkeypatch=monkeypatch, skip=frozenset({1}))
    layers = _make_layers(cache)
    gen = torch.Generator().manual_seed(9)
    hidden, ids, w = _routing(1, gen)
    _run_step(layers, hidden.to(device), ids, w, device)
    torch.cuda.synchronize()
    assert not any(cache._prefetch_issued)
    # Layer 2 was still prefetched for (only target 1 was skipped).
    assert cache.prefetch_stats_summary()["calls"] == 1

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import freetoken.core as core
from freetoken.core import Batch, Context
from freetoken.distributed import set_tp_info, try_get_tp_info


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _batch(*, rows: int, marked: bool, phase: str = "prefill", requests: int = 1) -> Batch:
    batch = Batch(
        reqs=[SimpleNamespace(uid=index) for index in range(requests)],
        phase=phase,
    )
    if marked:
        batch.mtp_verify = True
    batch.input_ids = torch.zeros(rows, dtype=torch.int32)
    return batch


@contextmanager
def _active_batch(monkeypatch, batch: Batch):
    ctx = Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    with ctx.forward_batch(batch):
        yield


def _layer(*, num_experts: int = 16, top_k: int = 10, hidden_size: int = 32,
           intermediate_size: int = 24):
    from freetoken.layers.moe import OffloadMoELayer

    _init_tp()
    return OffloadMoELayer(
        layer_id=0,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


def test_private_verify_uses_decode_cache_but_remains_prefill(monkeypatch):
    layer = _layer()
    rows = 3
    hidden = torch.randn(rows, 32)
    weights = torch.full((rows, 10), 0.1)
    ids = torch.arange(10, dtype=torch.int32).repeat(rows, 1)
    calls: list[str] = []

    def decode(got_hidden, got_weights, got_ids):
        calls.append("decode")
        assert got_hidden is hidden
        assert got_weights is weights
        return got_hidden

    def prefill(*args):
        calls.append("prefill")
        return args[0]

    monkeypatch.setattr(layer, "_decode_routed", decode)
    monkeypatch.setattr(layer, "_prefill_routed", prefill)
    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda **kwargs: (weights, ids.clone()),
    )

    batch = _batch(rows=rows, marked=True)
    assert batch.is_prefill is True
    assert batch.is_decode is False
    with _active_batch(monkeypatch, batch):
        assert layer.routed_forward(hidden, weights, ids.clone()) is hidden
        assert layer.forward(hidden, torch.randn(rows, 16)) is hidden

    assert calls == ["decode", "decode"]
    assert batch.phase == "prefill"


def test_private_verify_marker_is_disabled_by_default(monkeypatch):
    layer = _layer()
    hidden = torch.randn(2, 32)
    weights = torch.full((2, 10), 0.1)
    ids = torch.arange(10, dtype=torch.int32).repeat(2, 1)
    calls: list[str] = []

    monkeypatch.setattr(
        layer,
        "_prefill_routed",
        lambda h, w, i: calls.append("prefill") or h,
    )
    monkeypatch.setattr(
        layer,
        "_decode_routed",
        lambda h, w, i: calls.append("decode") or h,
    )

    prefill = _batch(rows=2, marked=False)
    assert prefill.mtp_verify is False
    with _active_batch(monkeypatch, prefill):
        layer.routed_forward(hidden, weights, ids.clone())

    decode = _batch(rows=1, marked=False, phase="decode")
    with _active_batch(monkeypatch, decode):
        layer.routed_forward(hidden[:1], weights[:1], ids[:1].clone())

    # Direct-op tests and a few private model runners use batch-shaped namespaces.
    # Their missing marker must retain the old movement choice.
    legacy_prefill = SimpleNamespace(
        is_prefill=True,
        is_decode=False,
        reqs=[SimpleNamespace(uid=0)],
    )
    with _active_batch(monkeypatch, legacy_prefill):
        layer.routed_forward(hidden, weights, ids.clone())

    assert calls == ["prefill", "decode", "prefill"]


@pytest.mark.parametrize(
    ("rows", "phase", "requests"),
    [(1, "prefill", 1), (5, "prefill", 1), (2, "prefill", 2), (2, "decode", 1)],
)
def test_private_verify_rejects_invalid_public_batch_before_movement(
    monkeypatch, rows, phase, requests
):
    layer = _layer()
    hidden = torch.randn(rows, 32)
    weights = torch.full((rows, 10), 0.1)
    ids = torch.arange(10, dtype=torch.int32).repeat(rows, 1)

    def unexpected(*args):
        raise AssertionError("specialist movement ran before private batch validation")

    monkeypatch.setattr(layer, "_prefill_routed", unexpected)
    monkeypatch.setattr(layer, "_decode_routed", unexpected)
    batch = _batch(rows=rows, marked=True, phase=phase, requests=requests)

    with _active_batch(monkeypatch, batch):
        with pytest.raises(
            ValueError,
            match="private MTP verification requires one prefill request and 2 to 4 rows",
        ):
            layer.routed_forward(hidden, weights, ids)


def _movement_cache(*, decode_target="gpu"):
    from flashlib.kernels.slot_cache import Stat

    cache = SimpleNamespace(
        collect_stats=True,
        decode_target=decode_target,
        lru_stats=torch.zeros((2, 3), dtype=torch.int64),
        stat_calls=torch.zeros((), dtype=torch.int64),
        stat_active=torch.zeros((), dtype=torch.int64),
        stat_missing=torch.zeros((), dtype=torch.int64),
        stat_fetched=torch.zeros((), dtype=torch.int64),
        bank_caches={
            "gate_up": torch.zeros((8, 3), dtype=torch.float32),
            "down": torch.zeros((8, 2), dtype=torch.float16),
        },
    )
    cache._stat = Stat
    return cache


def test_pure_movement_uses_actual_fetched_delta_instead_of_missing_count():
    from freetoken.engine.mtp_fast_verify import MTPFastVerifier

    cache = _movement_cache()
    before = MTPFastVerifier._stats_snapshot(cache)
    cache.lru_stats[0, cache._stat.CALLS] += 2
    cache.lru_stats[0, cache._stat.ACTIVE] += 10
    cache.lru_stats[0, cache._stat.MISS] += 5
    cache.stat_fetched += 3

    counters = MTPFastVerifier._movement_counters(cache, before)
    movement = MTPFastVerifier._movement_result(cache, counters)

    assert movement["layer_calls"] == 2
    assert movement["active_experts"] == 10
    assert movement["hit_experts"] == 5
    assert movement["missing_experts"] == 5
    assert movement["fetched_experts"] == 3
    assert movement["cpu_experts"] == 0
    assert movement["bytes_per_expert"] == 16
    assert movement["h2d_bytes"] == movement["transfer_bytes"] == 48


def test_hybrid_movement_reconciles_actual_fetched_cpu_and_transfer_bytes():
    from freetoken.engine.mtp_fast_verify import MTPFastVerifier

    cache = _movement_cache(decode_target="hybrid")
    before = MTPFastVerifier._stats_snapshot(cache)
    cache.stat_calls += 2
    cache.stat_active += 11
    cache.stat_missing += 7
    cache.stat_fetched += 3

    counters = MTPFastVerifier._movement_counters(cache, before)
    movement = MTPFastVerifier._movement_result(cache, counters)

    assert movement["active_experts"] == 11
    assert movement["hit_experts"] == 4
    assert movement["missing_experts"] == 7
    assert movement["fetched_experts"] == 3
    assert movement["cpu_experts"] == 4
    assert movement["h2d_bytes"] == 48
    assert movement["d2d_rows"] == movement["d2d_bytes"] == 0


def test_hit_only_movement_records_zero_fetch_and_transfer_bytes():
    from freetoken.engine.mtp_fast_verify import MTPFastVerifier

    cache = _movement_cache()
    before = MTPFastVerifier._stats_snapshot(cache)
    cache.lru_stats[0, cache._stat.CALLS] += 1
    cache.lru_stats[0, cache._stat.ACTIVE] += 4
    movement = MTPFastVerifier._movement_result(
        cache,
        MTPFastVerifier._movement_counters(cache, before),
    )

    assert movement["missing_experts"] == 0
    assert movement["fetched_experts"] == 0
    assert movement["h2d_bytes"] == movement["transfer_bytes"] == 0


class _IdentitySlotCache:
    """A cache-shaped view whose slots intentionally equal source expert IDs."""

    quant_format = "bf16"
    decode_target = "gpu"
    prefill_overlap = False

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor):
        self._views = (gate_up, down)
        self.decode_calls = 0
        self.prefill_calls = 0

    def is_cpu_layer(self, layer_id: int) -> bool:
        return False

    def ensure_experts(self, layer_id: int, ids: torch.Tensor) -> None:
        self.decode_calls += 1

    def materialize_layer(self, layer_id: int) -> None:
        self.prefill_calls += 1

    def copy_missing(self) -> None:
        pass

    def bank_views(self, n=None):
        return self._views if n is None else tuple(view[:n] for view in self._views)

    def alphas_for_slots(self, layer_id: int):
        return None

    def alphas_for_layer(self, layer_id: int):
        return None


def _reference(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(hidden, dtype=torch.float32)
    for row in range(hidden.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[row, route])
            projected = gate_up[expert].float() @ hidden[row].float()
            gate, up = projected.chunk(2)
            activated = torch.nn.functional.silu(gate) * up
            result[row] += (down[expert].float() @ activated) * weights[row, route]
    return result.to(hidden.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_private_verify_multirow_matches_reference(monkeypatch, rows):
    torch.manual_seed(3100 + rows)
    device = torch.device("cuda")
    num_experts, top_k, hidden_size, intermediate = 16, 10, 32, 24
    layer = _layer(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate,
    )
    gate_up = (
        torch.randn(
            num_experts,
            2 * intermediate,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.25
    )
    down = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.25
    )
    cache = _IdentitySlotCache(gate_up, down)
    layer.offload_cache = cache
    hidden = 0.25 * torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
    raw_weights = torch.rand(rows, top_k, device=device)
    weights = (raw_weights / raw_weights.sum(dim=-1, keepdim=True)).contiguous()
    ids = torch.stack(
        [
            (torch.arange(top_k, device=device, dtype=torch.int32) + row) % num_experts
            for row in range(rows)
        ]
    ).contiguous()
    expected = _reference(hidden, gate_up, down, weights, ids)

    fast_batch = _batch(rows=rows, marked=True)
    with _active_batch(monkeypatch, fast_batch):
        fast = layer.routed_forward(hidden, weights, ids.clone())

    full_batch = _batch(rows=rows, marked=False)
    with _active_batch(monkeypatch, full_batch):
        full = layer.routed_forward(hidden, weights, ids.clone())

    torch.cuda.synchronize(device)
    assert cache.decode_calls == 1
    assert cache.prefill_calls == 1
    torch.testing.assert_close(fast, expected, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(full, expected, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(fast, full, rtol=5e-2, atol=5e-2)

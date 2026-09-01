"""FREETOKEN_MOE_SMALL_PREFILL_ROWS: a narrow prefill fetches experts instead of the bank.

The offload prefill path streams every expert of every layer (48 x 1.32 GiB over PCIe)
regardless of prompt size, so a 26-token chat turn and a prefix-cache hit with three new rows
each pay the full 63 GiB. The knob routes such batches through the decode LRU instead. These
tests pin the exact admission conditions, that N=0 leaves every choice byte-identical, that
the MTP verify guard is untouched, and that the two movements compute the same thing.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import freetoken.core as core
import freetoken.layers.moe as moe_module
from freetoken.core import Batch, Context
from freetoken.distributed import set_tp_info, try_get_tp_info


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _batch(*, rows: int, phase: str = "prefill", requests: int = 1) -> Batch:
    batch = Batch(
        reqs=[SimpleNamespace(uid=index) for index in range(requests)], phase=phase
    )
    batch.input_ids = torch.zeros(rows, dtype=torch.int32)
    return batch


@contextmanager
def _active_batch(monkeypatch, batch):
    ctx = Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    with ctx.forward_batch(batch):
        yield


def _layer(*, num_experts=16, top_k=10, hidden_size=32, intermediate_size=24, cache=object()):
    _init_tp()
    layer = moe_module.OffloadMoELayer(
        layer_id=0,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    layer.offload_cache = cache
    return layer


def _routed(monkeypatch, layer, rows, calls):
    """Stub both movements and return the arguments ``routed_forward`` should be given."""
    monkeypatch.setattr(layer, "_decode_routed", lambda h, w, i: calls.append("decode") or h)
    monkeypatch.setattr(layer, "_prefill_routed", lambda h, w, i: calls.append("prefill") or h)
    hidden = torch.randn(rows, layer.hidden_size)
    weights = torch.full((rows, layer.top_k), 0.1)
    ids = torch.arange(layer.top_k, dtype=torch.int32).repeat(rows, 1)
    return hidden, weights, ids


def _movement(monkeypatch, *, rows, limit, phase="prefill", requests=1, cache=object()):
    monkeypatch.setattr(moe_module, "_SMALL_PREFILL_ROWS", limit)
    layer = _layer(cache=cache)
    calls: list[str] = []
    hidden, weights, ids = _routed(monkeypatch, layer, rows, calls)
    with _active_batch(monkeypatch, _batch(rows=rows, phase=phase, requests=requests)):
        layer.routed_forward(hidden, weights, ids.clone())
        layer.forward(hidden, torch.randn(rows, layer.num_experts))
    return calls


def test_the_knob_is_off_by_default():
    """An unset env is 0, and 0 is the value the old behaviour is defined by."""
    assert moe_module._read_small_prefill_rows() == 0
    assert moe_module._SMALL_PREFILL_ROWS == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", 0), ("0", 0), ("4", 4), ("64", 64), ("999", 64), ("-3", 0), ("nonsense", 0)],
)
def test_the_env_is_clamped_to_the_row_ceiling(monkeypatch, raw, expected):
    monkeypatch.setenv(moe_module._SMALL_PREFILL_ROWS_ENV, raw)
    assert moe_module._read_small_prefill_rows() == expected


@pytest.mark.parametrize("rows", [1, 2, 6, 26, 64])
def test_disabled_leaves_every_prefill_on_the_streaming_path(monkeypatch, rows):
    """N=0: the movement choice is exactly ``batch.is_decode``, as it was."""
    assert _movement(monkeypatch, rows=rows, limit=0) == ["prefill", "prefill"]


def test_a_batch_at_the_limit_takes_decode_movement(monkeypatch):
    assert _movement(monkeypatch, rows=4, limit=4) == ["decode", "decode"]


def test_a_batch_one_row_past_the_limit_does_not(monkeypatch):
    assert _movement(monkeypatch, rows=5, limit=4) == ["prefill", "prefill"]


def test_the_limit_counts_rows_across_every_request_in_the_batch(monkeypatch):
    """Two requests of two rows each is a four-row batch, not two two-row ones."""
    assert _movement(monkeypatch, rows=4, limit=4, requests=2) == ["decode", "decode"]
    assert _movement(monkeypatch, rows=5, limit=4, requests=2) == ["prefill", "prefill"]


def test_a_decode_batch_is_unaffected(monkeypatch):
    assert _movement(monkeypatch, rows=1, limit=4, phase="decode") == ["decode", "decode"]
    assert _movement(monkeypatch, rows=1, limit=0, phase="decode") == ["decode", "decode"]


def test_a_layer_without_an_offload_cache_stays_on_the_streaming_path(monkeypatch):
    """The decode path dereferences the cache unconditionally; never send it there blind."""
    assert _movement(monkeypatch, rows=2, limit=8, cache=None) == ["prefill", "prefill"]


@pytest.mark.parametrize(
    ("max_tokens", "rows", "expected"),
    [(8, 4, "decode"), (8, 8, "decode"), (2, 4, "prefill")],
)
def test_a_cpu_decode_target_caps_the_width_at_its_pinned_io_size(
    monkeypatch, max_tokens, rows, expected
):
    """cpu/hybrid size their C++ scratch and pinned IO once; a wider submit runs past them."""
    cache = SimpleNamespace(cpu_executor=SimpleNamespace(max_tokens=max_tokens))
    assert _movement(monkeypatch, rows=rows, limit=16, cache=cache) == [expected] * 2


def test_a_zero_row_batch_is_never_admitted(monkeypatch):
    """``1 <= rows``: an empty extend has no routing to fetch and no reason to change paths."""
    layer = _layer()
    monkeypatch.setattr(moe_module, "_SMALL_PREFILL_ROWS", 8)
    assert layer._small_prefill_moves_like_decode(0) is False


def test_the_mtp_verify_guard_is_untouched(monkeypatch):
    """A marked batch still validates its shape, and still raises, whatever the knob says."""
    monkeypatch.setattr(moe_module, "_SMALL_PREFILL_ROWS", 64)
    layer = _layer()
    calls: list[str] = []
    hidden, weights, ids = _routed(monkeypatch, layer, 7, calls)
    batch = _batch(rows=7)
    batch.mtp_verify = True
    with _active_batch(monkeypatch, batch):
        with pytest.raises(ValueError, match="one prefill request and 2 to 6 rows"):
            layer.routed_forward(hidden, weights, ids.clone())
    assert calls == []


def test_a_batch_shaped_namespace_without_the_marker_still_works(monkeypatch):
    """Direct-op tests and private runners pass batch-shaped namespaces; keep tolerating them."""
    monkeypatch.setattr(moe_module, "_SMALL_PREFILL_ROWS", 8)
    layer = _layer()
    calls: list[str] = []
    hidden, weights, ids = _routed(monkeypatch, layer, 2, calls)
    legacy = SimpleNamespace(is_prefill=True, is_decode=False, reqs=[SimpleNamespace(uid=0)])
    with _active_batch(monkeypatch, legacy):
        layer.routed_forward(hidden, weights, ids.clone())
    assert calls == ["decode"]


# --------------------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------------------


class _IdentitySlotCache:
    """A cache-shaped view whose slots intentionally equal source expert IDs."""

    quant_format = "bf16"
    decode_target = "gpu"
    prefill_overlap = False
    cpu_executor = None

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor):
        self._views = (gate_up, down)
        self.decode_calls = 0
        self.prefill_calls = 0

    def is_cpu_layer(self, layer_id: int) -> bool:
        return False

    def prefetch_wait(self, layer_id: int) -> bool:
        return False  # FREETOKEN_MOE_PREFETCH is unarmed here

    def prefetch_ready(self, layer_id: int) -> bool:
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


def _reference(hidden, gate_up, down, weights, ids):
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
@pytest.mark.parametrize("rows", [3, 8])
def test_small_prefill_decode_movement_matches_the_streaming_path(monkeypatch, rows):
    """Same batch, same routing, both movements -- and both match the python reference."""
    torch.manual_seed(4100 + rows)
    device = torch.device("cuda")
    num_experts, top_k, hidden_size, intermediate = 16, 8, 32, 24
    gate_up = 0.25 * torch.randn(
        num_experts, 2 * intermediate, hidden_size, device=device, dtype=torch.bfloat16
    )
    down = 0.25 * torch.randn(
        num_experts, hidden_size, intermediate, device=device, dtype=torch.bfloat16
    )
    cache = _IdentitySlotCache(gate_up, down)
    layer = _layer(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate,
        cache=cache,
    )
    hidden = 0.25 * torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
    raw = torch.rand(rows, top_k, device=device)
    weights = (raw / raw.sum(dim=-1, keepdim=True)).contiguous()
    ids = torch.stack(
        [
            (torch.arange(top_k, device=device, dtype=torch.int32) + row) % num_experts
            for row in range(rows)
        ]
    ).contiguous()
    expected = _reference(hidden, gate_up, down, weights, ids)

    monkeypatch.setattr(moe_module, "_SMALL_PREFILL_ROWS", rows)
    with _active_batch(monkeypatch, _batch(rows=rows)):
        small = layer.routed_forward(hidden, weights, ids.clone())
    monkeypatch.setattr(moe_module, "_SMALL_PREFILL_ROWS", 0)
    with _active_batch(monkeypatch, _batch(rows=rows)):
        streamed = layer.routed_forward(hidden, weights, ids.clone())

    torch.cuda.synchronize(device)
    assert (cache.decode_calls, cache.prefill_calls) == (1, 1)
    torch.testing.assert_close(small, expected, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(streamed, expected, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(small, streamed, rtol=5e-2, atol=5e-2)

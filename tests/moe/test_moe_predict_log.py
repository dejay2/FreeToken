"""The routing-predictor hook (FREETOKEN_MOE_PREDICT_LOG) on a tiny CPU MoE layer.

Mirrors the routing logger it sits beside: a module-level directory gates it, the records
are appended JSONL, and a forward never breaks because a diagnostic did.
"""

import json

import pytest
import torch

import freetoken.layers.moe as moe
from freetoken.distributed import set_tp_info, try_get_tp_info

HIDDEN = 6
EXPERTS = 8
TOP_K = 3
TOKENS = 8


def _init_tp():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


class _Gate:
    """A router with the shape the real one has: ``forward(x) -> [T, num_experts]``."""

    def __init__(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(EXPERTS, HIDDEN, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T.to(x.dtype)


@pytest.fixture
def ctx_batch():
    """A global context holding one prefill batch, the way a forward sees it."""
    import freetoken.core as core
    from freetoken.core import Batch, Context, set_global_ctx

    core._GLOBAL_CTX = None
    ctx = Context(page_size=1)
    set_global_ctx(ctx)
    batch = Batch(reqs=[], phase="prefill")
    batch.input_ids = torch.arange(TOKENS, dtype=torch.int32)
    batch.positions = torch.arange(TOKENS, dtype=torch.int32)
    batch.out_loc = None
    with ctx.forward_batch(batch):
        yield batch
    core._GLOBAL_CTX = None


@pytest.fixture
def armed(monkeypatch, tmp_path):
    """Arm the hook against ``tmp_path`` with routers for layers 0..3 registered."""
    _init_tp()
    monkeypatch.setattr(moe, "_PREDICT_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(moe, "_PREDICT_DUMP", False)
    monkeypatch.setattr(moe, "_predict_records", [])
    monkeypatch.setattr(moe, "_predict_path", None)
    monkeypatch.setattr(moe, "_predict_routers", {})
    monkeypatch.setattr(moe, "_predict_dump", {})
    monkeypatch.setattr(moe, "_predict_dump_last_layer", -1)
    monkeypatch.setattr(moe, "_predict_dump_chunk", 0)
    for layer_id in range(4):
        moe.register_predict_router(layer_id, _Gate(layer_id))
    return tmp_path


def _layer(monkeypatch, layer_id: int):
    layer = moe.OffloadMoELayer(
        layer_id=layer_id,
        num_experts=EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size=4,
    )
    # The expert compute is not what this tests; the hook runs before it either way.
    monkeypatch.setattr(layer, "_prefill_routed", lambda h, w, i: h)
    return layer


def _records(directory):
    moe._predict_flush()
    files = list(directory.glob("predict-log-*.jsonl"))
    assert len(files) == 1, files
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def test_registry_is_a_noop_while_unarmed(monkeypatch):
    monkeypatch.setattr(moe, "_PREDICT_LOG_DIR", None)
    monkeypatch.setattr(moe, "_predict_routers", {})
    moe.register_predict_router(7, _Gate(0))
    moe.register_predict_router(None, _Gate(0))
    assert moe._predict_routers == {}


def test_record_shape(monkeypatch, armed, ctx_batch):
    layer = _layer(monkeypatch, 0)
    x = torch.randn(TOKENS, HIDDEN)
    layer.prefill_forward(x, x @ _Gate(0).weight.T)

    records = _records(armed)
    assert len(records) == 1
    record = records[0]
    assert set(record) == {
        "layer", "phase", "positions", "tokens",
        "actual_top10", "pred_next_top20", "pred_next2_top20",
    }
    assert record["layer"] == 0
    assert record["phase"] == "prefill"
    assert record["positions"] == list(range(TOKENS))
    assert record["tokens"] == list(range(TOKENS))
    assert len(record["actual_top10"]) == TOKENS
    assert all(len(row) == TOP_K for row in record["actual_top10"])
    # top-20 is clamped to the expert count on this toy layer
    for key in ("pred_next_top20", "pred_next2_top20"):
        assert len(record[key]) == TOKENS
        assert all(len(row) == EXPERTS for row in record[key])


def test_prediction_uses_the_next_layers_router_and_the_real_scorer(monkeypatch, armed, ctx_batch):
    layer = _layer(monkeypatch, 1)
    x = torch.randn(TOKENS, HIDDEN)
    layer.prefill_forward(x, x @ _Gate(1).weight.T)

    record = _records(armed)[0]
    for key, gate_id in (("pred_next_top20", 2), ("pred_next2_top20", 3)):
        _, expected = moe.fused_topk(
            hidden_states=x,
            gating_output=moe._predict_routers[gate_id].forward(x),
            topk=EXPERTS,
            renormalize=layer.renormalize,
        )
        assert record[key] == expected.tolist()


def test_missing_upper_layers_leave_empty_predictions(monkeypatch, armed, ctx_batch):
    layer = _layer(monkeypatch, 3)  # layers 4 and 5 are not registered
    x = torch.randn(TOKENS, HIDDEN)
    layer.prefill_forward(x, x @ _Gate(3).weight.T)

    record = _records(armed)[0]
    assert record["pred_next_top20"] == [] and record["pred_next2_top20"] == []
    assert len(record["actual_top10"]) == TOKENS


def test_a_broken_router_never_breaks_the_forward(monkeypatch, armed, ctx_batch):
    class _Broken:
        def forward(self, x):
            raise RuntimeError("router is on fire")

    moe._predict_routers[1] = _Broken()
    layer = _layer(monkeypatch, 0)
    x = torch.randn(TOKENS, HIDDEN)
    assert layer.prefill_forward(x, x @ _Gate(0).weight.T) is x
    assert moe._predict_records == []


def test_dump_stitches_three_layers_per_chunk(monkeypatch, armed, ctx_batch):
    monkeypatch.setattr(moe, "_PREDICT_DUMP", True)
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16)
    for layer_id in range(4):
        layer = _layer(monkeypatch, layer_id)
        layer.prefill_forward(x, (x.float() @ _Gate(layer_id).weight.T).to(x.dtype))
    moe._predict_dump_flush()

    written = sorted(path.name for path in armed.glob("x_layer*.pt"))
    # layers 0 and 1 have both an L+1 and an L+2 to predict; 2 and 3 do not
    assert written == ["x_layer00_0.pt", "x_layer01_0.pt"]

    blob = torch.load(armed / "x_layer00_0.pt", weights_only=True)
    rows = -(-TOKENS // moe._PREDICT_DUMP_STRIDE)  # every 4th token
    assert blob["layer"] == 0 and blob["num_experts"] == EXPERTS
    assert blob["x"].shape == (rows, HIDDEN) and blob["x"].dtype == torch.float16
    assert blob["positions"] == list(range(0, TOKENS, moe._PREDICT_DUMP_STRIDE))
    for key in ("top10_l", "top10_l1", "top10_l2"):
        assert blob[key].shape == (rows, TOP_K)
    # the stitched rows are the OTHER layers' real decisions, not layer 0's
    assert not torch.equal(blob["top10_l"], blob["top10_l1"])


def test_dump_flushes_when_a_new_forward_starts(monkeypatch, armed, ctx_batch):
    monkeypatch.setattr(moe, "_PREDICT_DUMP", True)
    x = torch.randn(TOKENS, HIDDEN)
    for pass_id in range(2):
        for layer_id in range(3):
            layer = _layer(monkeypatch, layer_id)
            layer.prefill_forward(x, x @ _Gate(layer_id).weight.T)
        assert len(moe._predict_dump) == 3
        if pass_id == 0:
            assert not list(armed.glob("x_layer*.pt")), "nothing written mid-forward"
    # layer 0 of the second forward flushed the first one and started over
    assert [path.name for path in armed.glob("x_layer*.pt")] == ["x_layer00_0.pt"]

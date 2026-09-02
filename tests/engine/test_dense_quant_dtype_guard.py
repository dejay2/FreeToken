"""``FREETOKEN_DENSE_QUANT=int8`` is a bf16 contract.

The int8 W8A16 GEMM feeds ``tl.dot`` bf16 operands regardless of the model dtype, so under
``--dtype float16`` an M>1 prefill rounds its activations to bf16 while the M=1 GEMV stays
exact in fp32 -- prefill and decode disagree -- and the fp32-``tiny`` scale floor underflows
to 0.0 in fp16, turning a zero weight row into a divide-by-zero. The repo already refuses the
same dtype for MXFP8 for the same reason; int8 has to be refused at the same gate, at config
time, not discovered in the first prefill.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _engine_config(dtype, **quant):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=dtype,
        attention_backend="fi",
    )
    quant = {"attn_quant": "none", "dense_quant": "none", "lm_head_quant": "none", **quant}
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=False,
            num_layers=10,
            expert_quant="none",
            **quant,
        ),
    )
    return config


@pytest.fixture(autouse=True)
def _arch(monkeypatch):
    from freetoken.engine import engine

    monkeypatch.setattr(engine, "is_sm100_family", lambda: False)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: False)
    monkeypatch.setattr(engine, "_flashinfer_available", lambda: True)
    monkeypatch.setattr(engine, "_sgl_flash_attn_available", lambda: True)


@pytest.mark.parametrize("component", ["attn_quant", "dense_quant", "lm_head_quant"])
def test_int8_dense_weights_refuse_float16_at_config_time(component):
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="float16") as info:
        _adjust_config(_engine_config(torch.float16, **{component: "int8"}))
    assert "int8" in str(info.value) and "bfloat16" in str(info.value)


@pytest.mark.parametrize("component", ["attn_quant", "dense_quant", "lm_head_quant"])
def test_int8_dense_weights_are_fine_in_bfloat16(component):
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_engine_config(torch.bfloat16, **{component: "int8"}))


def test_float16_without_int8_is_untouched_by_this_gate():
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_engine_config(torch.float16))

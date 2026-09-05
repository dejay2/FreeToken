"""CPU-only EXL3 backend policy and dispatch tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _config(**overrides):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/models/glm53-exl3",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        attention_backend="triton",
        **overrides,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            model_type="glm5_next",
            single_stream_only=False,
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=45,
            num_moe_layers=42,
            num_experts=288,
            num_experts_per_tok=8,
            expert_quant="exl3",
            hidden_act="silu",
            moe_weight_format=None,
            moe_backend="auto",
        ),
    )
    return config


@pytest.mark.parametrize(
    ("cuda_graph_bs", "cuda_graph_max_bs"),
    [([1, 2], 4), (None, None), ([1], 0)],
)
def test_exl3_rejects_cuda_graph_settings_with_exact_off_flag(
    cuda_graph_bs, cuda_graph_max_bs
):
    from freetoken.engine.engine import _adjust_config

    config = _config(
        moe_backend="offload",
        cuda_graph_bs=cuda_graph_bs,
        cuda_graph_max_bs=cuda_graph_max_bs,
    )

    with pytest.raises(ValueError, match=r"pass --cuda-graph-max-bs 0"):
        _adjust_config(config)

    assert config.cuda_graph_bs == cuda_graph_bs
    assert config.cuda_graph_max_bs == cuda_graph_max_bs


def test_exl3_graphs_off_pass_through_untouched_and_auto_select_offload(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    import freetoken.moe.bench_profile as bench_profile

    def unexpected_bench_call(*args, **kwargs):
        pytest.fail("EXL3 auto selection must not consult the hybrid benchmark")

    monkeypatch.setattr(bench_profile, "load_backend_recommendation", unexpected_bench_call)
    config = _config(moe_backend="auto", cuda_graph_bs=None, cuda_graph_max_bs=0)

    _adjust_config(config)

    assert config.moe_backend == "offload"
    assert config.moe_cache_auto is True
    assert config.cuda_graph_bs is None
    assert config.cuda_graph_max_bs == 0
    assert config.model_config.moe_backend == "offload"


@pytest.mark.parametrize("backend", ["cpu", "hybrid", "fused"])
def test_exl3_rejects_non_offload_backend_before_weight_loading(backend):
    from freetoken.engine.engine import _adjust_config

    config = _config(moe_backend=backend)
    with pytest.raises(ValueError, match="EXL3"):
        _adjust_config(config)


def test_exl3_rejects_cpu_layer_split_mode():
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="moe-cpu-layers"):
        _adjust_config(_config(moe_backend="offload", moe_cpu_layers="0"))


def test_exl3_allows_reviewed_gpu_owned_layers():
    from freetoken.engine.engine import _adjust_config

    config = _config(
        moe_backend="offload",
        moe_gpu_owned_layers="auto:1",
        cuda_graph_max_bs=0,
        cuda_graph_bs=None,
    )
    _adjust_config(config)

    assert config.moe_gpu_owned_layers == "auto:1"
    assert config.moe_backend == "offload"


def test_exl3_rejects_a_gpu_owned_cpu_layer_overlap():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config = _config(
        moe_backend="offload", moe_gpu_owned_layers="0,1", moe_cpu_layers="0,1"
    )
    with pytest.raises(ValueError, match="both GPU-owned and CPU layers"):
        _validate_gpu_owned_layers(config, 42)


def test_exl3_provider_is_registered_and_layer_dispatches_to_b2_operation(monkeypatch):
    from freetoken.moe.expert_banks import _PROVIDERS, _exl3_banks
    from freetoken.moe.fused_exl3 import fused_experts_exl3, require_exl3_gpu_only
    from freetoken.layers.moe import OffloadMoELayer

    assert _PROVIDERS["exl3"] is _exl3_banks

    import freetoken.moe.fused_exl3 as fused_exl3

    called = {}
    scratch = object()

    def fake_require(**kwargs):
        called["require"] = kwargs

    def fake_operation(hidden_states, banks, topk_weights, topk_ids, **kwargs):
        called["operation"] = (hidden_states, banks, topk_weights, topk_ids, kwargs)
        return torch.ones_like(hidden_states)

    monkeypatch.setattr(fused_exl3, "require_exl3_gpu_only", fake_require)
    monkeypatch.setattr(fused_exl3, "fused_experts_exl3", fake_operation)

    layer = OffloadMoELayer.__new__(OffloadMoELayer)
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer.exl3_scratch = scratch
    hidden = torch.zeros((1, 128), dtype=torch.bfloat16)
    weights = torch.ones((1, 8), dtype=torch.float32)
    ids = torch.zeros((1, 8), dtype=torch.int32)
    cache = SimpleNamespace(quant_format="exl3", decode_target="gpu")
    views = tuple(torch.zeros(1) for _ in range(9))

    result = layer._expert_gemm(
        cache,
        hidden,
        weights,
        ids,
        views=views,
        n=None,
        alphas=None,
        is_prefill=False,
    )

    assert torch.equal(result, torch.ones_like(hidden))
    assert called["require"] == {"device": hidden.device, "decode_target": "gpu"}
    assert called["operation"][1] is views
    assert called["operation"][4]["scratch"] is scratch


def test_exl3_allows_the_single_safe_cuda_graph(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    config = _config(
        moe_backend="offload",
        max_running_req=1,
        cuda_graph_bs=[1],
        cuda_graph_max_bs=1,
    )
    _adjust_config(config)

    assert config.max_running_req == 1
    assert config.cuda_graph_bs == [1]
    assert config.cuda_graph_max_bs == 1


def test_exl3_safe_decode_defaults_to_a_one_row_graph():
    from freetoken.engine.engine import _adjust_config

    config = _config(
        moe_backend="offload",
        max_running_req=1,
        cuda_graph_bs=None,
        cuda_graph_max_bs=None,
    )
    _adjust_config(config)

    assert config.cuda_graph_bs is None
    assert config.cuda_graph_max_bs == 1


def test_exl3_rejects_a_graph_wider_than_the_fixed_decode_arena():
    from freetoken.engine.engine import _adjust_config

    config = _config(
        moe_backend="offload",
        max_running_req=1,
        cuda_graph_bs=[1, 2],
        cuda_graph_max_bs=2,
    )
    with pytest.raises(ValueError, match=r"pass --cuda-graph-max-bs 0"):
        _adjust_config(config)

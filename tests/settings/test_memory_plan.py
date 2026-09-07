from __future__ import annotations

import copy
import json
import os
import struct
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from freetoken.engine import memory_plan  # noqa: E402


def test_mxfp4_tp_slot_padding_and_fit_boundary(startup_budget_case):
    from freetoken.models.gpt_oss.weight import local_mxfp4_intermediate_range
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

    config, kwargs = startup_budget_case
    config.model_config.expert_quant = "mxfp4"
    config.model_config.hidden_size = config.model_config.moe_intermediate_size = 2880
    config.tp_info = SimpleNamespace(size=4, rank=0)
    _, _, local_width = local_mxfp4_intermediate_range(2880, rank=0, world_size=4)
    expected = _BANK_BYTES_PER_EXPERT["mxfp4"](2880, local_width)

    per_slot, source, runtime, _ = memory_plan._expert_slot_bytes(config)

    assert (per_slot, source, runtime) == (expected, "mxfp4", "mxfp4_triton")
    assert local_width == 736 and per_slot == 3_386_944
    config.memory_ratio = 1.0
    config.num_page_override = 17
    kwargs.update(per_expert=per_slot, runtime_format=runtime)
    reference = memory_plan._scenario_geometry(config, **kwargs)
    kwargs["capacity"] = reference["resources"]["vram"]["need"]
    assert memory_plan._scenario_geometry(config, **kwargs)["issues"] == []
    kwargs["capacity"] -= 1
    result = memory_plan._scenario_geometry(config, **kwargs)
    assert result["issues"] or result["resources"]["vram"]["need"] > kwargs["capacity"]


def test_pinned_bank_need_excludes_locked_cpu_layers():
    assert (
        memory_plan._pinned_bank_need(
            100,
            num_moe_layers=5,
            owned_layers=frozenset({0}),
            cpu_layers=frozenset({1, 2}),
            split_residency=True,
        )
        == 50
    )


@pytest.mark.parametrize("num_experts,owned_count", [(512, 24), (0, 0)])
def test_owned_layer_resolution_uses_engine_fraction_grammar(num_experts, owned_count):
    config = SimpleNamespace(
        model_config=SimpleNamespace(num_moe_layers=48, num_experts=num_experts, expert_quant="nvfp4"),
        moe_gpu_owned_layers="0.5",
        moe_backend="offload",
        moe_cpu_layers=None,
        moe_learn_routing=False,
        model_path=None,
        moe_cache_size=0,
    )

    assert len(memory_plan._owned_layers(config)) == owned_count


@pytest.mark.parametrize("embed_host", [False, True])
@pytest.mark.parametrize("cpu_viable", [False, True])
def test_host_embedding_pin_quota_and_auto_placement(monkeypatch, startup_budget_case, embed_host, cpu_viable):
    from freetoken.daemon.settings import model_info
    from freetoken.engine import engine
    from freetoken.kvcache import linear_state_pool
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    from freetoken.moe.expert_banks import bank_bytes_estimate

    config, _ = startup_budget_case
    config.model_config.hidden_size = 1024
    config.model_config.moe_intermediate_size = 128
    config.model_config.expert_quant = "bf16"
    config.moe_cpu_layers = None
    config.moe_gpu_owned_layers = ""
    config.num_page_override = 17
    model = object.__new__(Qwen4ExpForCausalLM)
    model._embed_host, model._vision_execution = embed_host, "gpu"
    model.model = SimpleNamespace(embed_tokens=SimpleNamespace(
        weight=torch.empty((1024, 1024), device="meta", dtype=torch.bfloat16),
    ))
    embedding_bytes = 2 << 20
    bank_total = bank_bytes_estimate(config.model_config)
    cap = bank_total + embedding_bytes // 2
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(cap / (1 << 30)))
    monkeypatch.setattr(engine, "_cpu_moe_executor_viable", lambda _config: cpu_viable)
    monkeypatch.setattr(model_info, "read_model", lambda _path: SimpleNamespace(ple_bytes=0))
    monkeypatch.setattr(linear_state_pool, "state_pool_bytes", lambda _config: 0)
    monkeypatch.setattr(memory_plan, "_loader_buffer_bytes", lambda *_args, **_kwargs: (0, "none"))

    model_bytes = memory_plan._model_weight_bytes(model)
    host_tables, workspace, components = memory_plan._ple_and_vision_bytes(config, model_bytes, model=model)
    result = memory_plan._evaluate_scenarios(
        (config, config), machine={"vram_free_bytes": 1 << 40, "vram_total_bytes": 1 << 40},
        model_bytes=model_bytes, host_tables=host_tables, vision_workspace=workspace,
        table_components=components, per_expert=1, source_format="bf16", runtime_format="bf16",
    )["results"]["now"]

    assert host_tables == (embedding_bytes if embed_host else 0)
    assert bool(result["pinning"]["cpu_layers"]) is (embed_host and cpu_viable)
    assert result["pinning"]["cap_bytes"] == cap
    assert result["pinning"]["need_bytes"] == bank_total + (embedding_bytes if embed_host else 0) - (
        bank_total // config.model_config.num_moe_layers if embed_host and cpu_viable else 0
    )
    assert result["pinning"]["shortfall_bytes"] == (embedding_bytes // 2 if embed_host and not cpu_viable else 0)
    assert result["ram"]["resident"] == bank_total + (embedding_bytes if embed_host else 0)


def test_pin_cap_preserves_live_environment(monkeypatch):
    from freetoken.engine import engine

    observed = []
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "73")
    monkeypatch.setattr(
        engine,
        "_pin_budget_bytes",
        lambda reserved: observed.append(os.environ.get("FREETOKEN_PIN_BUDGET_GB")) or 123,
    )

    assert memory_plan._pin_cap(os.environ, 0) == 123
    assert observed == ["73"]


@pytest.mark.parametrize("ftw", [False, True])
def test_checkpoint_metadata_scanned_once_across_candidates(monkeypatch, tmp_path, startup_budget_case, ftw):
    import builtins
    from freetoken.daemon.settings import model_info
    from freetoken.engine import engine
    from freetoken.kvcache import linear_state_pool
    from freetoken.models import weight
    from freetoken.moe import expert_banks

    config, _ = startup_budget_case
    config.model_path = str(tmp_path)
    config.model_config.num_moe_layers, config.model_config.num_experts = 4, 8
    config.model_config.hidden_size = config.model_config.moe_intermediate_size = 32
    config.model_config.expert_quant = "bf16"
    config.tp_info = SimpleNamespace(size=1)
    config.moe_cpu_layers, config.moe_gpu_owned_layers = None, ""
    config.expert_load = "auto"
    config.num_page_override = 2
    config.kv_reserve_tokens = 64
    config.moe_cache_size = 16
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3MoeForCausalLM"], "num_hidden_layers": 4, "num_experts": 8,
        "hidden_size": 32, "moe_intermediate_size": 32, "max_position_embeddings": 1024,
    }))
    if ftw:
        (tmp_path / "freetoken_weight.json").write_text(json.dumps({"tensors": [{"kind": "experts_bank", "nbytes": 4096}]}))
    else:
        for i in range(2):
            header = json.dumps({f"model.layers.{i}.experts.0.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
            (tmp_path / f"shard-{i}.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    calls = {"model": 0, "ftw": 0, "scattered": 0, "header_open": 0, "candidates": 0}

    def counted(name, function):
        def call(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return call

    real_open = builtins.open
    def opened(path, *args, **kwargs):
        if str(path).endswith(".safetensors"):
            calls["header_open"] += 1
        return real_open(path, *args, **kwargs)

    def candidate(_base, settings, **_kwargs):
        calls["candidates"] += 1
        result = copy.copy(config)
        result.moe_cache_size = settings["MoECacheSize"]
        result.moe_gpu_owned_layers = settings["GpuOwnedLayers"]
        return result

    monkeypatch.setattr(builtins, "open", opened)
    monkeypatch.setattr(model_info, "read_model", counted("model", model_info.read_model))
    monkeypatch.setattr(expert_banks, "ftw_bank_bytes", counted("ftw", expert_banks.ftw_bank_bytes))
    monkeypatch.setattr(weight, "experts_scattered", counted("scattered", weight.experts_scattered))
    monkeypatch.setattr(engine, "_cpu_moe_executor_viable", lambda _config: True)
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", str(2048 / (1 << 30)))
    monkeypatch.setattr(memory_plan, "_parse_config", lambda *_args: config)
    monkeypatch.setattr(memory_plan, "_candidate_config", candidate)
    monkeypatch.setattr(memory_plan, "_meta_model", lambda _config: SimpleNamespace())
    monkeypatch.setattr(linear_state_pool, "state_pool_bytes", lambda _config: 0)
    request = {
        "version": 1, "argv": [],
        "settings": {"ModelPath": str(tmp_path), "MoECacheSize": 16, "GpuOwnedLayers": "", "ContextTokens": 64, "KVCacheTokens": 128},
        "machine": {"ram_free_bytes": 0, "ram_total_bytes": 32 << 30,
                    "vram_free_bytes": 32 << 30, "vram_total_bytes": 32 << 30},
    }
    for request_count in (1, 2):
        result = memory_plan.estimate_request(request)
        assert result["status"] == "ok", result
        assert calls["candidates"] > 2
        assert calls["model"] == calls["ftw"] == request_count
        assert calls["scattered"] == (0 if ftw else request_count)
        assert calls["header_open"] == (0 if ftw else 2 * request_count)


def test_unchanged_empty_fit_has_no_apply_suggestion(monkeypatch, startup_budget_case):
    from freetoken.kvcache import linear_state_pool

    config, _ = startup_budget_case
    config.model_config.is_moe = False
    config.model_config.num_experts = config.model_config.num_moe_layers = 0
    config.num_page_override = 17
    monkeypatch.setattr(memory_plan, "_parse_config", lambda *_args: config)
    monkeypatch.setattr(memory_plan, "_candidate_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(memory_plan, "_meta_model", lambda _config: SimpleNamespace())
    monkeypatch.setattr(linear_state_pool, "state_pool_bytes", lambda _config: 0)
    result = memory_plan.estimate_request({
        "version": 1, "argv": [],
        "settings": {"MoECacheSize": 0, "GpuOwnedLayers": "", "ContextTokens": 64, "KVCacheTokens": 1088},
        "machine": {"ram_free_bytes": 0, "ram_total_bytes": 32 << 30,
                    "vram_free_bytes": 32 << 30, "vram_total_bytes": 32 << 30},
    })

    assert result["status"] == "ok" and result["fits_empty"] and not result["fits_now"]
    assert result["suggestion"] is None
    assert any(issue["code"] == "unchanged_empty_fit" for issue in result["issues"])


def test_candidate_config_uses_fresh_launch_normalization():
    base = SimpleNamespace(
        moe_cache_size=0,
        moe_cache_auto=True,
        moe_gpu_owned_layers="auto",
        kv_reserve_tokens=262144,
        num_page_override=None,
        num_token_override=None,
        page_size=64,
    )
    settings = {
        "MoECacheSize": 12000,
        "GpuOwnedLayers": "auto:2",
        "ContextTokens": 8192,
        "KVCacheTokens": 16384,
    }

    def builder(candidate, *, base_env):
        assert candidate == settings
        return SimpleNamespace(
            argv=["python", "-m", "freetoken.cli", "serve", "--candidate"],
            env=dict(base_env),
        )

    def parser(argv, environment):
        assert argv == ["--candidate"]
        assert environment == {"FREETOKEN_MOE_GPU_OWNED_LAYERS": "auto:2"}
        return SimpleNamespace(
            moe_cache_size=12000,
            moe_cache_auto=False,
            moe_gpu_owned_layers="auto:2",
            kv_reserve_tokens=8192,
            num_page_override=256,
            num_token_override=None,
            page_size=64,
        )

    candidate = memory_plan._candidate_config(
        base,
        settings,
        environment={"FREETOKEN_MOE_GPU_OWNED_LAYERS": "auto:2"},
        launch_builder=builder,
        config_parser=parser,
    )

    assert candidate is not base
    assert candidate.moe_cache_size == 12000
    assert candidate.moe_gpu_owned_layers == "auto:2"
    assert candidate.kv_reserve_tokens == 8192
    assert candidate.num_page_override == 256
    assert base.moe_cache_size == 0
    assert base.moe_gpu_owned_layers == "auto"
    assert base.kv_reserve_tokens == 262144


def test_explicit_kv_override_is_normalized_to_pool_pages():
    base = SimpleNamespace(
        moe_cache_size=12000,
        moe_cache_auto=False,
        moe_gpu_owned_layers="auto:2",
        kv_reserve_tokens=23552,
        num_page_override=None,
        num_token_override=None,
        page_size=64,
    )
    settings = {"MoECacheSize": 12000, "GpuOwnedLayers": "auto:2", "KVCacheTokens": 24576}

    def builder(candidate, *, base_env):
        assert candidate == settings
        return SimpleNamespace(
            argv=["python", "-m", "freetoken.cli", "serve", "--candidate"],
            env=dict(base_env),
        )

    def parser(argv, _environment):
        assert argv == ["--candidate"]
        return SimpleNamespace(
            moe_cache_size=12000,
            moe_cache_auto=False,
            moe_gpu_owned_layers="auto:2",
            kv_reserve_tokens=23552,
            num_page_override=384,
            num_token_override=None,
            page_size=64,
        )

    candidate = memory_plan._candidate_config(
        base,
        settings,
        environment={},
        launch_builder=builder,
        config_parser=parser,
    )

    assert candidate.num_page_override == 384
    assert candidate.num_token_override is None


def test_meta_model_runs_dense_load_conversion_before_byte_accounting(monkeypatch):
    pytest.importorskip("triton")
    from freetoken.kernel.triton.int8_linear import Int8DenseLinear
    import freetoken.models as models
    from freetoken.utils import torch_dtype

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        holder = Int8DenseLinear(4, 8)
    monkeypatch.setattr(models, "create_model", lambda _config: holder)

    result = memory_plan._meta_model(SimpleNamespace(dtype=torch.bfloat16, model_config=object()))

    assert result is holder
    assert holder.weight.dtype == torch.int8
    assert holder.weight.numel() * holder.weight.element_size() == 8 * 4
    assert holder.weight_scale.numel() * holder.weight_scale.element_size() == 8 * 2


def test_meta_storage_walk_counts_converted_private_scale_once():
    torch = pytest.importorskip("torch")
    class Holder:
        def __init__(self):
            self.weight = torch.empty((8, 4), dtype=torch.int8, device="meta")
            self._scale = torch.empty((8,), dtype=torch.bfloat16, device="meta")
            self.alias = self.weight

    bytes_by_device = memory_plan.walk_tensor_storage(Holder(), device_for_key=lambda key: "cuda")

    assert bytes_by_device["cuda"] == 8 * 4 + 8 * 2


def test_cache_geometry_delegates_to_engine_budget_policy():
    result = memory_plan.solve_cache_geometry(
        baseline_free=1_900,
        weights_bytes=0,
        memory_ratio=1.0,
        cache_per_page=100,
        fixed_cache_size=0,
        per_expert_bytes=10,
        num_experts=10,
        total_experts=100,
        prefill_overlap=True,
        kv_reserve_tokens=100,
        page_size=10,
        quant_format="bf16",
    )

    assert result == (80, 11, True)


def test_moe_geometry_uses_pool_solver_without_context_clamping(monkeypatch):
    from freetoken.engine import cache_budget
    import freetoken.kvcache as kvcache

    seen: list[int] = []

    class Pool:
        @classmethod
        def kv_cost(cls, _config):
            return 64, 0, 64, 0

        @classmethod
        def solve_num_pages(cls, _config, available):
            seen.append(int(available))
            return 12_569

    monkeypatch.setattr(kvcache, "resolve_pool_class", lambda _model: Pool)
    monkeypatch.setattr(memory_plan, "_has_routed_moe", lambda _config: True)
    monkeypatch.setattr(memory_plan, "_owned_layers", lambda _config: frozenset())
    monkeypatch.setattr(cache_budget, "gpu_owned_reservation_bytes", lambda *_args: 0)
    monkeypatch.setattr(cache_budget, "lru_slots_after_owned_charge", lambda **kwargs: kwargs["moe_cache_size"])
    monkeypatch.setattr(cache_budget, "net_cache_budget_bytes", lambda *_args: 1 << 40)
    monkeypatch.setattr(cache_budget, "check_explicit_moe_cache_fits", lambda **_kwargs: None)
    monkeypatch.setattr(cache_budget, "resolve_vram_reserve_bytes", lambda *_args, **_kwargs: 0)

    config = SimpleNamespace(
        model_config=SimpleNamespace(num_experts=8, num_moe_layers=4, is_moe=True),
        spec_decode=SimpleNamespace(enabled=False),
        moe_cache_size=128,
        moe_cache_auto=False,
        kv_reserve_tokens=1024,
        num_page_override=None,
        max_seq_len=32_768,
        memory_ratio=0.9,
        moe_vram_reserve_bytes=0,
        moe_cache_headroom_bytes=0,
        attention_backend="qsa_sparse",
        cache_type="radix",
        moe_backend="offload",
        ple_backend="disk",
    )

    result = memory_plan._scenario_geometry(
        config,
        capacity=16 * (1 << 30),
        weights_gpu=2 * (1 << 30),
        fixed_pool=0,
        state_bytes=0,
        post_reserve=0,
        headroom=0,
        per_expert=1,
        runtime_format="bf16",
        owned=frozenset(),
        prefill_overlap=True,
        scenario_name="now",
    )

    assert result["geometry"]["num_pages"] == 12_569
    assert seen


@pytest.mark.parametrize("kv_dtype,bytes_per_page", [("bf16", 4224), ("fp8", 3200)])
def test_qsa_geometry_includes_scales_index_and_fixed_pending_buffers(kv_dtype, bytes_per_page):
    from freetoken.attention import AttnType
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.kvcache import resolve_pool_class

    spec = KVCacheGroupSpec(
        name="full", layer_ids=(0,), num_kv_heads=2, head_dim=8, sliding_window=None,
        index_head_dim=4, num_index_layers=1, index_ratio=4, attn_type=AttnType.QSA,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_moe=True, num_moe_layers=4, num_experts=8, kv_cache_group_specs=lambda: (spec,),
        ),
        dtype=torch.bfloat16, kv_dtype=kv_dtype, tp_info=SimpleNamespace(size=1),
        page_size=64, max_running_req=1, num_speculative_tokens=0,
        spec_decode=SimpleNamespace(enabled=False), moe_cache_size=16, moe_cache_auto=False,
        kv_reserve_tokens=128, num_page_override=8, memory_ratio=1.0,
        moe_vram_reserve_bytes=0, attention_backend="qsa_sparse", cache_type="radix",
        moe_backend="offload", ple_backend="mmap",
    )
    pool = resolve_pool_class(config.model_config)
    # FP8: 32 K/V + 16 FP32 scale + 2 BF16 index bytes/token, not half of BF16's 66.
    # Two request slots: 64 pending index + 16 scratch + 192 position-ring bytes.
    assert pool.kv_cost(config) == (bytes_per_page, 272, 64, 0)

    result = memory_plan._scenario_geometry(
        config, capacity=1 << 30, weights_gpu=1024, fixed_pool=0, state_bytes=0,
        post_reserve=0, headroom=0, per_expert=64, runtime_format="bf16", owned=frozenset(),
        prefill_overlap=True, scenario_name="now",
    )

    assert result["issues"] == []
    assert result["geometry"]["num_pages"] == 8  # Override has no extra sentinel allocation.
    assert result["geometry"]["usable_kv_tokens"] == 448
    kv = next(c for c in result["components"] if c["name"] == "KV cache")
    assert kv["bytes"] == 8 * bytes_per_page + 272
    assert result["resources"]["vram"]["resident"] == 2048 + kv["bytes"]


@pytest.fixture
def startup_budget_case(monkeypatch):
    import freetoken.kvcache as kvcache
    from freetoken.kvcache.base import BaseKVCachePool

    class Pool(BaseKVCachePool):
        @classmethod
        def kv_cost(cls, _config):
            return 1 << 20, 0, 64, 0

    monkeypatch.setattr(kvcache, "resolve_pool_class", lambda _model: Pool)
    gib = 1 << 30
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_moe=True, num_moe_layers=48, num_experts=512),
        spec_decode=SimpleNamespace(enabled=False),
        model_path="", moe_cache_size=4096, moe_cache_auto=False,
        moe_vram_reserve_bytes=3 * gib // 4, moe_cache_headroom_bytes=3 * gib // 2,
        memory_ratio=0.9, kv_reserve_tokens=1024, num_page_override=None, page_size=64,
        attention_backend="synthetic", cache_type="radix", moe_backend="offload", ple_backend="mmap",
    )
    kwargs = dict(
        capacity=32 * gib, weights_gpu=4 * gib, fixed_pool=0, state_bytes=gib,
        post_reserve=3 * gib // 4, headroom=3 * gib // 2, per_expert=2 << 20,
        runtime_format="bf16", owned=frozenset(), prefill_overlap=True, scenario_name="now",
    )
    return config, kwargs


@pytest.mark.parametrize("override,pages,fits", [(None, 16179, True), (4096, 4096, True), (32768, 32768, False)])
def test_explicit_policy_uses_reserve_floor_not_actual_kv(startup_budget_case, override, pages, fits):
    from freetoken.engine.cache_budget import check_explicit_moe_cache_fits, net_cache_budget_bytes

    config, kwargs = startup_budget_case
    config.num_page_override = override

    def precheck(capacity):
        net = net_cache_budget_bytes(config.memory_ratio, capacity, kwargs["weights_gpu"], kwargs["state_bytes"])
        check_explicit_moe_cache_fits(
            moe_cache_size=4096, per_expert_bytes=2 << 20,
            budget_bytes=net - 17 * (1 << 20),  # 1024/64 reserve pages + the sentinel
            owned_layers=0, num_experts=512,
            reserved_bytes=kwargs["post_reserve"] + kwargs["headroom"], requested_total=4096,
        )

    precheck(kwargs["capacity"])
    result = memory_plan._scenario_geometry(config, **kwargs)

    assert result["issues"] == []
    assert result["geometry"]["num_pages"] == pages
    vram = result["resources"]["vram"]
    assert (vram["need"] <= kwargs["capacity"]) is fits
    precheck(vram["policy_need"])
    with pytest.raises(ValueError):
        precheck(vram["policy_need"] - 1)


def test_auto_policy_uses_solved_pages_not_explicit_kv_override(startup_budget_case):
    config, kwargs = startup_budget_case
    config.moe_cache_auto = True
    config.max_seq_len = 1024
    baseline = memory_plan._scenario_geometry(config, **kwargs)
    config.num_page_override = 1024

    result = memory_plan._scenario_geometry(config, **kwargs)

    assert baseline["geometry"]["num_pages"] == 17
    assert result["geometry"]["num_pages"] == 1024
    assert result["geometry"]["lru_slots"] == baseline["geometry"]["lru_slots"]
    assert result["resources"]["vram"]["policy_need"] == baseline["resources"]["vram"]["policy_need"]
    assert result["resources"]["vram"]["need"] <= kwargs["capacity"]


@pytest.mark.parametrize("automatic_slots", [False, True])
def test_late_mtp_ladder_preserves_startup_pages_and_adds_physical_bytes(monkeypatch, startup_budget_case, automatic_slots):
    from freetoken.kvcache import linear_state_pool
    from freetoken.moe import expert_banks

    config, kwargs = startup_budget_case
    config.spec_decode.enabled = True
    config.moe_cache_auto = automatic_slots
    late_bytes = 2 << 20
    monkeypatch.setattr(linear_state_pool, "state_pool_bytes", lambda _config: kwargs["state_bytes"])
    monkeypatch.setattr(memory_plan, "_linear_ladder_bytes", lambda cfg: late_bytes if cfg.spec_decode.enabled else 0)
    monkeypatch.setattr(memory_plan, "_physical_post_cache_reserve", lambda _config: kwargs["post_reserve"])
    monkeypatch.setattr(memory_plan, "_owned_layers", lambda _config: frozenset())
    monkeypatch.setattr(memory_plan, "_placement_plan", lambda *_args, **_kw: (frozenset(), None, 0, False, True))
    monkeypatch.setattr(memory_plan, "_loader_buffer_bytes", lambda *_args, **_kw: (0, "none"))
    monkeypatch.setattr(expert_banks, "ftw_bank_bytes", lambda _path: None)
    monkeypatch.setattr(expert_banks, "bank_bytes_estimate", lambda *_args, **_kw: 0)
    startup = memory_plan._scenario_geometry(config, **kwargs)

    result = memory_plan._evaluate_scenarios(
        (config, config),
        machine={"vram_free_bytes": kwargs["capacity"], "vram_total_bytes": kwargs["capacity"]},
        model_bytes={"cuda": kwargs["weights_gpu"]}, host_tables=0, vision_workspace=0,
        table_components=[], per_expert=kwargs["per_expert"], source_format="bf16", runtime_format="bf16",
    )

    for row in result["results"].values():
        assert row["geometry"] == startup["geometry"]
        if not automatic_slots:
            assert row["geometry"]["num_pages"] == 16179
        vram = row["vram"]
        assert vram["resident"] == startup["resources"]["vram"]["resident"] + late_bytes
        assert vram["peak"] == startup["resources"]["vram"]["peak"] + late_bytes
        assert vram["policy_need"] == startup["resources"]["vram"]["policy_need"]
        assert vram["need"] == max(vram["peak"], vram["policy_need"])
        ladder = [c for c in row["components"] if c["source"] == "spec_state_ladder"]
        assert len(ladder) == 1 and ladder[0]["bytes"] == late_bytes
        assert ladder[0]["phase"] == "graphs" and ladder[0]["kind"] == "allocation"


@pytest.mark.parametrize("routed_moe", [False, True])
def test_state_ledger_matches_resident_once(monkeypatch, startup_budget_case, routed_moe):
    from freetoken.kvcache import linear_state_pool
    from freetoken.moe import expert_banks

    config, kwargs = startup_budget_case
    config.model_config.is_moe = routed_moe
    config.num_page_override = 17
    monkeypatch.setattr(linear_state_pool, "state_pool_bytes", lambda _config: 1024)
    monkeypatch.setattr(memory_plan, "_linear_ladder_bytes", lambda _config: 2048)
    monkeypatch.setattr(memory_plan, "_owned_layers", lambda _config: frozenset())
    monkeypatch.setattr(memory_plan, "_placement_plan", lambda *_args, **_kwargs: (frozenset(), None, 0, False, True))
    monkeypatch.setattr(memory_plan, "_loader_buffer_bytes", lambda *_args, **_kwargs: (0, "none"))
    monkeypatch.setattr(expert_banks, "ftw_bank_bytes", lambda _path: None)
    monkeypatch.setattr(expert_banks, "bank_bytes_estimate", lambda *_args, **_kwargs: 0)

    result = memory_plan._evaluate_scenarios(
        (config, config), machine={"vram_free_bytes": kwargs["capacity"], "vram_total_bytes": kwargs["capacity"]},
        model_bytes={"cuda": kwargs["weights_gpu"]}, host_tables=0, vision_workspace=0,
        table_components=[], per_expert=kwargs["per_expert"], source_format="bf16", runtime_format="bf16",
    )

    for row in result["results"].values():
        allocations = [c for c in row["components"] if c["resource"] == "vram" and c["kind"] in {"allocation", "allowance"}]
        assert sum(c["bytes"] for c in allocations) == row["vram"]["resident"]
        state = [c for c in allocations if "linear_state_pool" in c["source"]]
        ladder = [c for c in allocations if "spec_state_ladder" in c["source"]]
        assert len(state) == len(ladder) == 1
        assert (state[0]["bytes"], state[0]["phase"]) == (1024, "kv")
        assert (ladder[0]["bytes"], ladder[0]["phase"]) == (2048, "graphs")


@pytest.mark.parametrize("shard_dir", ["", "weights"], ids=["root", "nested"])
def test_parallel_loader_excludes_nonexpert_shards(tmp_path, startup_budget_case, shard_dir):
    config, _ = startup_budget_case
    config.model_path, config.expert_load = str(tmp_path), "parallel"
    config.model_config.expert_quant = "nvfp4"
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen4ExpForConditionalGeneration"], "num_hidden_layers": 48,
        "num_experts": 512, "hidden_size": 32, "moe_intermediate_size": 32,
    }))
    weight_map = {}
    shard_path = tmp_path / shard_dir
    shard_path.mkdir(exist_ok=True)

    def shard(filename, names, size):
        filename = (shard_path / filename).relative_to(tmp_path).as_posix()
        header = json.dumps({name: {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
                             for i, name in enumerate(names)}).encode()
        (tmp_path / filename).write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * (size - 8 - len(header)))
        weight_map.update((name, filename) for name in names)

    # Scaled R1d shape: a larger bf16-only file and ten PLE-only files must not
    # displace expert files in the previous + active + queued + producer buffer bound.
    shard("model-bf16-00011.safetensors", ["model.language_model.norm.weight"], 107_000)
    for i in range(10):
        shard(f"model-plefp8-{i:05}.safetensors",
              [f"model.language_model.ple.ple_embedding.ngram_embedding.shard_{i}.weight"], 50_000)
    shard("global-scales.safetensors", ["model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2"], 70_000)
    shard("mtp.safetensors", ["mtp.layers.0.mlp.experts.0.gate_proj.weight"], 80_000)
    for i, size in enumerate((1025, 12_289, 8193, 4097, 513)):
        names = [f"model.language_model.layers.{i}.mlp.experts.0.gate_proj.weight"]
        if i == 1:
            # Charge the entire mixed shard once, not just the selected tensor ranges.
            names += [f"model.language_model.layers.{i}.mlp.experts.1.gate_proj.weight_scale", "model.language_model.embed_tokens.weight"]
        shard(f"model-nvfp4-{i:05}.safetensors", names, size)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (shard_path / "unindexed.safetensors").write_bytes(b"\0" * 200_000)

    metadata = memory_plan._checkpoint_metadata(config)
    buffers, _ = memory_plan._loader_buffer_bytes(config, machine={}, metadata=metadata)

    assert buffers == 45_056  # Largest five whole files, rounded to 4096-byte direct buffers.


@pytest.mark.parametrize("prefetch, expected_peak", [(None, 5), (0, 4), (3, 6)])
def test_parallel_loader_prices_shard_transition_peak(tmp_path, monkeypatch, prefetch, expected_peak):
    import mmap
    import weakref
    from pathlib import Path
    from threading import Event

    from freetoken.models import weight
    from freetoken.utils import hf, progress

    reader = weight.iter_expert_tensors_parallel
    if prefetch is not None:
        monkeypatch.setitem(reader.__kwdefaults__, "prefetch", prefetch)
    live = set()
    peak = 0
    reads = 0
    yielded = 0
    transition_ready = False
    producer_ready = Event()

    def read_shard(path, workers, chunk):
        nonlocal peak, reads
        buf = mmap.mmap(-1, 4096)
        buf[:] = Path(path).read_bytes()
        live.add(path)
        weakref.finalize(buf, live.discard, path)
        peak = max(peak, len(live))
        reads += 1
        if reads == expected_peak:
            producer_ready.set()
        return buf

    def frombuffer(view, dtype):
        nonlocal yielded, transition_ready
        yielded += 1
        if yielded == 2:
            # Hold B active before yielding it: caller tensor still owns A, the queue
            # fills with C/D, and the producer reads E. No scheduler-dependent sleeps.
            transition_ready = producer_ready.wait(5)
        return torch.frombuffer(view, dtype=dtype)

    weight_map = {}
    for i in range(7):
        name, shard = f"expert-{i}.weight", f"shard-{i}.safetensors"
        header = json.dumps({name: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
        (tmp_path / shard).write_bytes((struct.pack("<Q", len(header)) + header).ljust(4096, b"\0"))
        weight_map[name] = shard
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    monkeypatch.setattr(weight, "read_shard_direct", read_shard)
    monkeypatch.setattr(weight, "drop_page_cache", lambda _path: None)
    monkeypatch.setattr(weight, "torch", SimpleNamespace(frombuffer=frombuffer))
    monkeypatch.setattr(hf, "download_hf_weight", lambda path: path)
    monkeypatch.setattr(progress, "byte_bar", lambda *_args, **_kwargs: SimpleNamespace(update=lambda _n: None, close=lambda: None))

    for _name, tensor in reader(str(tmp_path), lambda _name: True):
        assert tensor.numel() == 1
    del tensor
    assert transition_ready, "producer did not reach the transition peak"
    assert peak == expected_peak
    assert not live

    metadata = memory_plan._CheckpointMetadata(None, None, (), (4096,) * 7, False, (4096,) * 7)
    buffers, _ = memory_plan._loader_buffer_bytes(SimpleNamespace(expert_load="parallel"), machine={}, metadata=metadata)
    assert buffers == peak * 4096


def test_layer_stream_workspace_is_not_added_to_boot_reserve(monkeypatch):
    from freetoken.moe import expert_banks

    captured: list[int] = []
    monkeypatch.setattr(expert_banks, "ftw_bank_bytes", lambda _path: None)
    monkeypatch.setattr(expert_banks, "bank_bytes_estimate", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(memory_plan, "_has_routed_moe", lambda _config: False)
    monkeypatch.setattr(memory_plan, "_owned_layers", lambda _config: frozenset())
    monkeypatch.setattr(
        memory_plan,
        "_placement_plan",
        lambda *_args, **_kwargs: (frozenset(), None, 0, False, True),
    )
    monkeypatch.setattr(memory_plan, "_loader_buffer_bytes", lambda *_args, **_kwargs: (0, "none"))
    monkeypatch.setattr(memory_plan, "_linear_ladder_bytes", lambda _config: 0)
    monkeypatch.setattr(memory_plan, "_physical_post_cache_reserve", lambda _config: 123)

    def scenario(_config, **kwargs):
        captured.append(kwargs["post_reserve"])
        return {
            "geometry": {"num_pages": 2},
            "resources": {"vram": {"need": 1, "resident": 1, "peak": 1, "policy_need": 0}},
            "components": [],
            "issues": [],
            "effective": {},
        }

    monkeypatch.setattr(memory_plan, "_scenario_geometry", scenario)
    model_config = SimpleNamespace(
        num_moe_layers=0,
        linear_attention_group=lambda: None,
        slot_states=(),
    )
    config = SimpleNamespace(
        model_path="",
        model_config=model_config,
        spec_decode=SimpleNamespace(enabled=False),
        kv_park="off",
        kv_park_window_mib=0,
        moe_cache_headroom_bytes=0,
        moe_backend="offload",
    )

    memory_plan._evaluate_scenarios(
        (config, config),
        machine={
            "ram_free_bytes": 1000,
            "ram_total_bytes": 2000,
            "vram_free_bytes": 1000,
            "vram_total_bytes": 2000,
        },
        model_bytes={"cuda": 100, "cpu": 200},
        host_tables=0,
        vision_workspace=999,
        table_components=[],
        per_expert=0,
        source_format="none",
        runtime_format="none",
    )

    assert captured == [123, 123]

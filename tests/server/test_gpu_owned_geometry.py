"""GPU-owned MoE layers in the reported geometry: /v1/cache/status, the residency rate the
cache report prints, and the --moe-cache-rate denominator. CPU-only, no engine, no CUDA."""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.cache_report import cache_rate
from freetoken.server.api_server import cache_geometry
from freetoken.server.model_meta import moe_total_experts


def _state(gpu_owned_layers):
    return SimpleNamespace(
        stats=SimpleNamespace(kv_total_pages=0, mamba_total_slots=0),
        config=SimpleNamespace(
            page_size=64,
            moe_cache_policy="lru",
            moe_cache_size=4400,
            moe_cache_rate=None,
            moe_gpu_owned_layers="auto",
            model_config=SimpleNamespace(num_experts=512, num_moe_layers=48, dsv4_args=None),
        ),
        last_rebuild=None,
        cache_pools={
            "num_pages": 100, "page_size": 64, "moe_cache_size": 4400,
            "num_mamba_slots": 0, "swa_page_size": 0, "num_swa_pages": 0,
            "gpu_owned_layers": gpu_owned_layers,
        },
        unit_bytes={"kv_bytes_per_token": 1, "moe_bytes_per_expert": 2_772_480},
        swa_full_tokens_ratio=0.0,
        cache_budget_bytes=0,
        free_vram_bytes=0,
        cache_floors={},
    )


def test_the_geometry_carries_the_owned_layer_ids():
    geo = cache_geometry(_state([0, 1, 2, 6, 7, 22]))

    assert geo["gpu_owned_layers"] == [0, 1, 2, 6, 7, 22]
    assert geo["num_moe_layers"] == 48  # the MODEL is unchanged; only the rate denominator moves


def test_the_geometry_defaults_to_no_owned_layers():
    state = _state([])
    state.cache_pools.pop("gpu_owned_layers")

    assert cache_geometry(state)["gpu_owned_layers"] == []


def test_the_residency_rate_is_denominated_over_streaming_layers():
    # 4400 slots serve the 42 streaming layers, not all 48: reporting 4400 / (48 * 512)
    # would understate residency by an eighth
    geometry = {"num_experts": 512, "num_moe_layers": 48, "gpu_owned_layers": [0, 1, 2, 6, 7, 22]}

    assert cache_rate(4400, geometry) == 4400 / (42 * 512)
    assert cache_rate(4400, {"num_experts": 512, "num_moe_layers": 48}) == 4400 / (48 * 512)


def test_moe_total_experts_subtracts_the_resolved_owned_layers():
    config = SimpleNamespace(
        moe_backend="offload",
        moe_cpu_layers=None,
        moe_gpu_owned_layers="auto",
        model_config=SimpleNamespace(num_experts=512, num_moe_layers=48),
    )

    assert moe_total_experts(config) == 42 * 512
    config.moe_gpu_owned_layers = None
    assert moe_total_experts(config) == 48 * 512

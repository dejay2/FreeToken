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


def test_the_geometry_exposes_the_owned_reservation_in_bytes():
    """cache_budget_bytes was byte-identical with and without six owned layers, so nothing in
    /v1/cache/status showed that 7.93 GiB of the card is spoken for."""
    geo = cache_geometry(_state([0, 1, 2, 6, 7, 22]))

    assert geo["gpu_owned_reserved_bytes"] == 6 * 512 * 2_772_480
    assert cache_geometry(_state([]))["gpu_owned_reserved_bytes"] == 0


def test_the_owned_reservation_comes_from_the_engine_ack_when_it_has_one():
    """The engine measures the resident banks; the derived product is only the fallback."""
    state = _state([0, 1, 2, 6, 7, 22])
    state.cache_pools["gpu_owned_reserved_bytes"] = 123_456_789

    assert cache_geometry(state)["gpu_owned_reserved_bytes"] == 123_456_789


def test_the_moe_slot_ceiling_counts_streaming_layers_only():
    """A slider must not offer slots for layers that are permanently resident, and must not
    offer the bytes the owned banks already hold."""
    from freetoken.server.api_server import _cache_limits

    unit_bytes = {"kv_per_token": 1, "moe_per_expert": 2_772_480, "mamba_per_slot": 0,
                  "swa_per_token": 0}
    owned_bytes = 6 * 512 * 2_772_480
    geo = {
        "page_size": 64, "num_experts": 512, "num_moe_layers": 48,
        "gpu_owned_layers": [0, 1, 2, 6, 7, 22], "gpu_owned_reserved_bytes": owned_bytes,
    }
    budget = 30_000 * 2_772_480  # far more than the model has experts for
    limits = _cache_limits(geo, unit_bytes, budget, {})

    assert limits["moe_experts"]["max"] == 42 * 512  # streaming experts, not 48 * 512

    # ...and where the budget binds, the owned banks' bytes are already gone
    tight = 5_000 * 2_772_480
    assert _cache_limits(geo, unit_bytes, tight, {})["moe_experts"]["max"] == 5_000 - 3_072


def test_compute_cache_pools_measures_the_resident_owned_banks():
    import torch

    from freetoken.kvcache.cache_status import compute_cache_pools

    row = torch.zeros(4, 8, dtype=torch.float16)  # 64 B per bank tensor
    moe = SimpleNamespace(
        cache_size=4400,
        gpu_owned_layer_ids=frozenset({0, 1}),
        resident_banks={0: (row, row), 1: (row, row)},
    )
    engine = SimpleNamespace(
        num_pages=100,
        config=SimpleNamespace(
            page_size=64,
            model_config=SimpleNamespace(dsv4_args=None, has_swa_attention=False),
            cache_type="naive",
        ),
        moe_offload_cache=moe,
        linear_state_pool=None,
    )
    pools = compute_cache_pools(engine)

    assert pools["gpu_owned_layers"] == [0, 1]
    assert pools["gpu_owned_reserved_bytes"] == 4 * row.numel() * row.element_size()


def test_compute_cache_pools_reports_zero_without_owned_layers():
    from freetoken.kvcache.cache_status import compute_cache_pools

    engine = SimpleNamespace(
        num_pages=0, config=None, moe_offload_cache=None, linear_state_pool=None,
    )

    assert compute_cache_pools(engine)["gpu_owned_reserved_bytes"] == 0


def test_governor_step_refreshes_geometry_through_real_reply_transport():
    import asyncio
    import torch
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.server.api_server import FrontendManager, dispatch_step
    from freetoken.tokenizer.server import _forward_control_msg

    async def run():
        seed = _state(list(range(8)))
        seed.config.served_model_name = "m"
        seed.config.kv_park = "off"
        seed.config.kv_dtype = "fp8"
        seed.config.model_config.has_swa_attention = False
        manager = FrontendManager(config=seed.config, send_tokenizer=None,
                                  recv_tokenizer=None, maintenance_state="serving")
        manager.cache_pools = seed.cache_pools
        manager.unit_bytes = seed.unit_bytes
        manager.last_rebuild = {"status": "ok", "moe_cache_size": 3450,
                                "num_pages": 100, "mamba_slots": 12}
        manager._residency_cache = {"owned": 8, "moe_cache_size": 3450}
        row = torch.zeros(4, 8, dtype=torch.float16)
        engine = SimpleNamespace(
            config=seed.config, num_pages=250, linear_state_pool=SimpleNamespace(num_slots=17),
            moe_offload_cache=SimpleNamespace(cache_size=1402, gpu_owned_layer_ids=set(range(13)),
                                             resident_banks={i: (row,) for i in range(13)}),
            residency_report=lambda: {"owned": 13, "pinned": 30, "disk": 5},
        )
        scheduler = SimpleNamespace(engine=engine)
        frontend = SimpleNamespace(put=manager._resolve_step)
        scheduler.send_result = lambda messages: [
            _forward_control_msg(message, None, frontend) for message in messages
        ]

        async def send_one(message):
            Scheduler._reply_step(scheduler, message.request_id, "ok",
                                  {"applied": "pinned->gpu_owned", "layer": 12,
                                   "moe_cache_size": 1402})

        manager.send_one = send_one
        reply = await dispatch_step(manager, axis="ram", direction="down")
        assert reply["status"] == "ok"
        geometry = cache_geometry(manager)
        assert geometry["moe_cache_size"] == 1402
        assert geometry["num_pages"] == 250
        assert geometry["num_mamba_slots"] == 16
        assert geometry["gpu_owned_layers"] == list(range(13))
        assert geometry["gpu_owned_reserved_bytes"] == 13 * row.numel() * row.element_size()
        assert manager._residency_cache is None

    asyncio.run(run())


def test_stale_step_does_not_replace_newer_rebuild_geometry():
    from freetoken.message import CacheStepReply
    from freetoken.server.api_server import FrontendManager, _open_maintenance

    seed = _state([1, 2])
    seed.config.served_model_name = "m"
    seed.config.kv_park = "off"
    seed.config.kv_dtype = "fp8"
    manager = FrontendManager(config=seed.config, send_tokenizer=None,
                              recv_tokenizer=None, maintenance_state="serving")
    manager.cache_pools = seed.cache_pools
    manager.last_rebuild = {"moe_cache_size": 4000}
    _open_maintenance(manager, "new-rebuild", "rebuild")
    manager._resolve_step(CacheStepReply(
        request_id="old-step", status="ok", moe_cache_size=1000,
        cache_pools={"moe_cache_size": 1000, "gpu_owned_layers": [3]},
    ))
    assert cache_geometry(manager)["moe_cache_size"] == 4000
    assert manager.cache_pools["gpu_owned_layers"] == [1, 2]


def test_completed_step_updates_without_http_waiter_and_ignores_late_duplicates():
    from freetoken.message import CacheRebuildReply, CacheResidencyReply, CacheStepReply
    from freetoken.server.api_server import FrontendManager, _open_maintenance

    seed = _state([0])
    seed.config.served_model_name = "m"
    seed.config.kv_park = "off"
    seed.config.kv_dtype = "fp8"
    manager = FrontendManager(config=seed.config, send_tokenizer=None,
                              recv_tokenizer=None, maintenance_state="serving")
    manager.cache_pools = seed.cache_pools
    _open_maintenance(manager, "timed-out-http", "step")
    step = CacheStepReply(request_id="timed-out-http", status="ok", moe_cache_size=2000,
                          cache_pools={"moe_cache_size": 2000, "gpu_owned_layers": [0, 1],
                                       "gpu_owned_reserved_bytes": 123})
    manager._resolve_step(step)
    assert cache_geometry(manager)["moe_cache_size"] == 2000
    _open_maintenance(manager, "new-rebuild", "rebuild")
    manager._resolve_rebuild(CacheRebuildReply(request_id="new-rebuild", status="ok",
                                              moe_cache_size=3000, num_pages=200))
    manager._resolve_residency(CacheResidencyReply(
        request_id="old-poll", status="ok", residency={"moe_cache_size": 1000, "owned": 1},
    ))
    manager._resolve_step(step)
    assert cache_geometry(manager)["moe_cache_size"] == 3000
    assert cache_geometry(manager)["num_pages"] == 200

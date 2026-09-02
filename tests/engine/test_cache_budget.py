from __future__ import annotations

from types import SimpleNamespace

import pytest
import re
import torch

import os

from freetoken.engine.cache_budget import expert_bytes_per_slot, plan_cache_budget, resolve_moe_cache_auto
from freetoken.engine.engine import _pin_budget_bytes


def test_moe_priority_fills_experts_up_to_total():
    # budget large enough to cache every expert; KV gets the remainder.
    # per_expert=100, cache_per_page=10, total=8 experts (L*E), E=4.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_pages=5, max_slots=8,
    )
    assert size == 8  # capped at full residency
    assert pages == (2000 - 8 * 100) // 10  # == 120, remainder to KV
    assert overlap is True


def test_offload_case_experts_take_most_kv_gets_reserve_floor():
    # budget too small for full residency: experts take what they can, KV keeps its floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=1000, per_expert_bytes=100, cache_per_page=10,
        num_experts=2, total_experts=50, prefill_overlap=True,
        kv_reserve_pages=10, max_slots=50,
    )
    # raw = (1000 - 10*10) // 100 = 9 ; clamped to [4, 50] -> 9
    assert size == 9
    assert pages == max((1000 - 9 * 100) // 10, 10)  # remainder 10 pages, == floor
    assert overlap is True


def test_marlin_cap_clamps_count_and_rolls_bytes_to_kv():
    # budget would fund 1500 experts, but marlin caps at 992; freed bytes become KV pages.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=200_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=128, total_experts=4000, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=992,
    )
    assert size == 992
    assert pages == (200_000 - 992 * 100) // 10


def test_small_cache_disables_prefill_overlap():
    # cap below 2*num_experts -> overlap impossible, falls back to num_experts floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=10_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=8, total_experts=12, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=12,
    )
    assert overlap is False
    # raw = 10000//100 = 100, clamped to hi = min(12, 12) = 12.
    assert size == 12


def test_insufficient_kv_memory_raises():
    with pytest.raises(AssertionError, match="not enough memory"):
        plan_cache_budget(
            budget_bytes=410, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=0, max_slots=4,
        )  # experts eat 400, KV gets 1 page -> not > 1


def test_budget_too_small_for_min_moe_plus_reserve_raises():
    # Budget cannot fund even the minimum MoE slots + the KV reserve, so the floored plan
    # would exceed budget_bytes. Reject in arithmetic rather than OOM in a later CUDA alloc.
    with pytest.raises(AssertionError, match="budget too small"):
        plan_cache_budget(
            budget_bytes=300, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=10, max_slots=4,
        )  # min moe = 4 slots (400 B) + reserve (10 pages = 100 B) = 500 B > 300 B budget


def test_prefill_overlap_false_is_honored():
    # Even when the cache could fit 2*num_experts, an explicit False stays False.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=False,
        kv_reserve_pages=0, max_slots=8,
    )
    assert size == 8
    assert overlap is False
    assert pages == (2000 - 8 * 100) // 10


def test_expert_bytes_per_slot_sums_row_bytes_over_banks():
    sources = {
        "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)],  # row = 32*8*2 = 512
        "down": [torch.zeros(4, 8, 16, dtype=torch.float16)],     # row = 8*16*2 = 256
    }
    assert expert_bytes_per_slot(sources) == 512 + 256


def test_resolve_auto_applies_ratio_once_and_marlin_cap():
    # baseline 1000, weights 100, ratio 0.9 -> budget = 900 - 100 - 0(fixed) = 800
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=1000, weights_bytes=100, memory_ratio=0.9,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=50,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=1, quant_format="bf16",
    )
    # budget 800: experts cap at 8 -> 400 bytes; KV = 400//10 = 40 pages
    assert size == 8 and pages == 40 and overlap is True


def test_resolve_auto_reserves_usable_tokens_beyond_the_dummy_page():
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=940,
        weights_bytes=0,
        memory_ratio=1.0,
        cache_per_page=10,
        fixed_cache_size=0,
        per_expert_bytes=100,
        num_experts=2,
        total_experts=50,
        prefill_overlap=False,
        kv_reserve_tokens=256,
        page_size=64,
        quant_format="bf16",
    )
    assert overlap is False
    assert (pages - 1) * 64 >= 256
    assert size == 8 and pages == 14


def test_resolve_auto_marlin_caps_slots():
    size, _, _ = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=0, memory_ratio=1.0,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=100,
        num_experts=128, total_experts=4000, prefill_overlap=False,
        kv_reserve_tokens=0, page_size=1, quant_format="nvfp4_marlin",
    )
    assert size == 992


def _dsv4_adjust_cfg(**over):
    # A DSV4 _adjust_config stub mirroring the real checkpoint (ds_fp4 experts, dsv4_sparse
    # attention, offload MoE backend).
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, dsv4_args=SimpleNamespace(window_size=128), is_moe=True,
        expert_quant="ds_fp4", has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = True
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "offload"
        max_running_req = 1
        cuda_graph_max_bs = 1
        cuda_graph_bs = [1]
        max_seq_len = 1024
        max_extend_tokens = 4096
        page_size = 1
        attention_backend = "dsv4_sparse"
        moe_cpu_layers = None
        nvfp4_backend = "auto"
        num_page_override = None
        num_token_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    for k, v in over.items():
        object.__setattr__(cfg, k, v)
    return cfg


def test_adjust_config_allows_auto_for_dsv4():
    # DSV4 now supports --moe-cache-auto via the affine KV cost bridge (dsv4_auto_cost_model);
    # _adjust_config must NOT reject it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg()
    _adjust_config(cfg)  # must not raise
    assert cfg.moe_backend == "offload"
    assert cfg.moe_cache_auto is True  # resolved later at engine init, not here
    assert cfg.page_size == 128  # DSV4's KV page is the P-token window page


def test_adjust_config_resolves_num_tokens_for_dsv4():
    # --num-tokens resolves AFTER every page_size override, so DSV4's P=128 page divides it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072)
    _adjust_config(cfg)
    assert cfg.page_size == 128
    assert cfg.num_page_override == 1024


def test_adjust_config_rejects_num_tokens_not_multiple_of_page():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131000)  # not a multiple of 128
    with pytest.raises(ValueError, match="not a multiple"):
        _adjust_config(cfg)


def test_adjust_config_rejects_num_tokens_with_num_pages():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072, num_page_override=1024)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _adjust_config(cfg)


def test_adjust_config_resolves_num_tokens_generic():
    # Generic model keeps its page_size (1 here): tokens map 1:1 onto pages.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "fi"
        nvfp4_backend = "auto"
        num_page_override = None
        num_token_override = 5000

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    _adjust_config(cfg)
    assert cfg.num_page_override == 5000


def test_mha_kv_cost_simple_full_attention():
    import torch

    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.utils import div_even

    class StubModelConfig:
        has_swa_attention = False

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=tuple(range(3)),
                num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.bfloat16
        page_size = 16
        max_running_req = 4
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    per_token = 2 * 64 * div_even(8, 1, allow_replicate=True) * 2 * 3
    assert cache_per_page == per_token * 16
    assert fixed == 0


def test_engine_resolve_auto_moe_cache_size_maps_kwargs():
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        num_experts = 4
        num_moe_layers = 2  # total_experts = 8

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_prefill_overlap = True
        kv_reserve_tokens = 0
        max_seq_len = 64
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()
        # Exercised by test_engine_auto_sizing_charges_the_declared_post_cache_reserve; 0
        # here so this test stays about the kwarg mapping.
        moe_vram_reserve_bytes = 0
        moe_cache_headroom_bytes = 0

        class tp_info:
            size = 1

    class StubBanks:
        quant_format = "bf16"
        # 2 layers (num_moe_layers above) x 4 experts each -- per-layer host bank contract.
        sources = {
            "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)] * 2,  # row = 32*8*2 = 512
            "down": [torch.zeros(4, 8, 16, dtype=torch.float16)] * 2,     # row = 8*16*2 = 256
        }

    from freetoken.kvcache.mha_pool import MHAKVCache

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 10_000_000
    engine._weights_bytes = 1_000_000
    engine._pool_cls = MHAKVCache  # __init__ skipped -> install the generic pool family

    size, pages, overlap = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks())

    # cross-check against the same pure functions, proving the kwarg mapping is faithful
    from freetoken.engine.cache_budget import expert_bytes_per_slot, resolve_moe_cache_auto
    from freetoken.kvcache.mha_pool import MHAKVCache

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    expected = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=1_000_000, memory_ratio=0.9,
        cache_per_page=cache_per_page, fixed_cache_size=fixed,
        per_expert_bytes=expert_bytes_per_slot(StubBanks.sources),
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=16, quant_format="bf16",
    )
    expected_size, expected_pages, expected_overlap = expected
    # Auto sizing cannot allocate usable KV beyond the model context; one additional page is
    # the pool's internal dummy/sentinel.
    expected_pages = min(expected_pages, StubConfig.max_seq_len // StubConfig.page_size + 1)
    assert (size, pages, overlap) == (expected_size, expected_pages, expected_overlap)


# ---------------------------------------------------------------------------
# offload-cache sizing guard + auto-resolution (_require_offload_cache_size / _adjust_config),
# the floor rule compute_cache_floors documents above.
# ---------------------------------------------------------------------------


def _offload_engine_config(**overrides):
    """A frozen EngineConfig for a quantized-experts MoE checkpoint in the bare-invocation state
    (moe_backend="auto") unless overridden — the shared fixture for the _adjust_config tests."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        attention_backend="fi",
        **overrides,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="nvfp4",  # quantized experts -> must resolve to an offload backend
            moe_backend="auto",
        ),
    )
    return config


def test_guard_passes_when_size_covers_one_expert_per_layer():
    from freetoken.engine.engine import _require_offload_cache_size

    _require_offload_cache_size(cache_size=128, num_experts=128)  # no raise


def test_guard_raises_actionable_error_when_too_small():
    from freetoken.engine.engine import _require_offload_cache_size

    with pytest.raises(ValueError) as exc:
        _require_offload_cache_size(cache_size=0, num_experts=128)
    msg = str(exc.value)
    assert "128" in msg and "moe-cache" in msg


def test_adjust_config_defaults_moe_cache_auto_for_auto_resolved_offload_backend():
    """Bare `ft serve <FTW MoE checkpoint>`: no --moe-backend, no --moe-cache-* flags at all.

    args.py's parse-time default only fires when the backend is *already*
    offload-family at parse time -- but a bare invocation leaves moe_backend="auto" at parse
    time, and the "auto" -> offload/cpu/hybrid resolution only happens later, in _adjust_config,
    once the model_config (and its expert_quant) is known. This proves the engine-level
    resolution: a quantized-experts model auto-resolving to an offload-family backend also gets
    moe_cache_auto=True, so _init_offload_moe_cache's _require_offload_cache_size guard is never
    reached with moe_cache_size still 0.
    """
    from freetoken.engine.engine import _adjust_config
    from freetoken.moe import is_offload_moe_backend

    config = _offload_engine_config()
    _adjust_config(config)

    # Which member of the family gets picked is not this test's claim, and is not ours to
    # decide: a bare "auto" consults ~/.cache/freetoken/benchbw.json, so a box that has run
    # `ft bench bw` resolves nvfp4 experts to hybrid instead. Assert the family, not the member.
    assert is_offload_moe_backend(config.moe_backend)
    assert config.moe_cache_auto is True
    assert config.moe_cache_size == 0  # still unresolved -- the scheduler sizes it from VRAM


def test_page_table_width_covers_whole_trailing_pages():
    # _write_page_table writes WHOLE trailing pages, so the width must reach the last
    # page's end, not just the next multiple of 32 (DSV4's P=128 exposed the gap).
    from freetoken.engine.engine import _page_table_width

    assert _page_table_width(4001, 128) == 4096   # align32 alone gave 4032 -> OOB
    assert _page_table_width(4096, 128) == 4096   # page-aligned length unchanged
    assert _page_table_width(100, 1) == 128       # page_size 1 degenerates to align32
    assert _page_table_width(33, 128) == 128
    for max_seq_len in (1, 31, 33, 4001, 4095, 4096):
        for page_size in (1, 32, 64, 128):
            w = _page_table_width(max_seq_len, page_size)
            last_col = -(-max_seq_len // page_size) * page_size - 1
            assert w > last_col and w % 32 == 0


def _generic_rotary_cfg(max_position, override):
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
        rotary_config=SimpleNamespace(max_position=max_position),
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton"
        nvfp4_backend = "auto"
        num_page_override = None
        num_token_override = None
        max_seq_len_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    object.__setattr__(cfg, "max_seq_len_override", override)
    return cfg


def test_adjust_config_rejects_override_past_rope_table():
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="rope table"):
        _adjust_config(_generic_rotary_cfg(max_position=1024, override=2048))


def test_adjust_config_allows_override_at_rope_table_boundary():
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_generic_rotary_cfg(max_position=1024, override=1024))  # must not raise


def test_adjust_config_rope_gate_exempts_dsv4():
    # DSV4 sizes its own rope table from the resolved max_seq_len (_adjust_dsv4_config),
    # so the generic gate must not fire even when the override dwarfs max_position.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(max_seq_len_override=10_000_000)
    cfg.model_config.rotary_config = SimpleNamespace(max_position=1024)
    _adjust_config(cfg)  # must not raise


# ---- _pin_budget_bytes: host bytes already pinned outside the expert banks ----


def test_reserved_subtracts_from_the_cap(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "2")
    assert _pin_budget_bytes() == 2 * 2**30
    assert _pin_budget_bytes(reserved=2**30) == 2**30
    assert _pin_budget_bytes(reserved=4 * 2**30) == 0


def test_uncapped_platform_stays_uncapped(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    if hasattr(os, "uname") and "microsoft" in os.uname().release.lower():
        pytest.skip("WSL caps pinning")
    assert _pin_budget_bytes(reserved=2**30) is None


# --------------------------------------------------- GPU-owned MoE layers (--moe-gpu-owned-layers)

_PER_EXPERT = 2_772_480  # one Qwen3.8-Flash-Next NVFP4 expert row across the 6 banks
_E = 512                 # num_experts
_L = 48                  # num_moe_layers


def _auto_with_owned(owned: int):
    """resolve_moe_cache_auto exactly as Engine._resolve_auto_moe_cache_size calls it with
    ``owned`` GPU-owned layers: their bytes join fixed_cache_size, and they leave
    total_experts."""
    from freetoken.engine.cache_budget import gpu_owned_reservation_bytes

    return resolve_moe_cache_auto(
        baseline_free=40 << 30,
        weights_bytes=8 << 30,
        memory_ratio=0.9,
        cache_per_page=1 << 20,
        fixed_cache_size=gpu_owned_reservation_bytes(owned, _E, _PER_EXPERT),
        per_expert_bytes=_PER_EXPERT,
        num_experts=_E,
        total_experts=(_L - owned) * _E,
        prefill_overlap=True,
        kv_reserve_tokens=0,
        page_size=1,
        quant_format="nvfp4",
    )


def test_gpu_owned_reservation_is_a_whole_layer_of_slots():
    from freetoken.engine.cache_budget import gpu_owned_reservation_bytes

    assert gpu_owned_reservation_bytes(0, _E, _PER_EXPERT) == 0
    assert gpu_owned_reservation_bytes(6, _E, _PER_EXPERT) == 6 * _E * _PER_EXPERT
    assert gpu_owned_reservation_bytes(1, _E, _PER_EXPERT) == 1_419_509_760  # 1.322 GiB


def test_gpu_owned_reservation_shrinks_the_auto_slot_count_by_exactly_one_layer_each():
    base, _, base_overlap = _auto_with_owned(0)
    owned6, _, owned_overlap = _auto_with_owned(6)

    assert base - owned6 == 6 * _E  # 3072 slots, one full expert layer per owned layer
    assert base_overlap is True and owned_overlap is True


def test_expert_bytes_per_slot_reads_the_first_streaming_layer():
    from freetoken.engine.cache_budget import expert_bytes_per_slot

    # layer 0 is GPU-owned and (deliberately) a different row shape; the slot cost must come
    # from layer 1, the first STREAMING layer -- layer 0 is in the default owned set.
    sources = {
        "gate_up": [torch.zeros(4, 99, 8, dtype=torch.float16), torch.zeros(4, 32, 8, dtype=torch.float16)],
        "down": [torch.zeros(4, 99, 16, dtype=torch.float16), torch.zeros(4, 8, 16, dtype=torch.float16)],
    }

    assert expert_bytes_per_slot(sources, frozenset({0})) == 32 * 8 * 2 + 8 * 16 * 2
    assert expert_bytes_per_slot(sources) == 99 * 8 * 2 + 99 * 16 * 2  # no owned set: layer 0


def test_explicit_cache_size_that_overflows_the_budget_names_what_would_fit():
    from freetoken.engine.cache_budget import check_explicit_moe_cache_fits

    # budget funds 5000 slots total; 6 owned layers already take 3072 of them
    budget = 5000 * _PER_EXPERT
    check_explicit_moe_cache_fits(
        moe_cache_size=1928, per_expert_bytes=_PER_EXPERT, budget_bytes=budget,
        owned_layers=6, num_experts=_E,
    )  # exactly fits
    with pytest.raises(ValueError) as excinfo:
        check_explicit_moe_cache_fits(
            moe_cache_size=4400, per_expert_bytes=_PER_EXPERT, budget_bytes=budget,
            owned_layers=6, num_experts=_E,
        )
    message = str(excinfo.value)
    # 4400 is the LRU left after the charge, so the operator typed 4400 + 3072; and the size
    # the message names is likewise a TOTAL, 1928 + 3072 (see the L4 tests below).
    assert "--moe-cache-size 7472" in message
    assert "6 GPU-owned MoE layers" in message
    assert "lower --moe-cache-size (launcher: -MoECacheSize) to 5000 slots" in message
    assert "own at most 1 layer" in message


# ------------------------------- --moe-cache-size is the TOTAL slot budget (VRAM neutrality)


def test_owned_layers_are_charged_to_the_explicit_cache_size_not_added_to_it():
    """The measured regression: 6 owned layers ADDED to 6750 slots is +7.93 GiB of VRAM.

    Charging them keeps the flag VRAM-neutral -- the same 6750-slot budget, 3072 of it held
    by the owned layers, 3678 left for the LRU. See
    docs/research/gpu-owned-layers-speed-diagnosis-2026-09-02.md.
    """
    from freetoken.engine.cache_budget import lru_slots_after_owned_charge

    assert lru_slots_after_owned_charge(
        moe_cache_size=6750, owned_layers=6, num_experts=_E, floor=2 * _E
    ) == 3678
    # the charge is exactly one full expert layer per owned layer
    assert 6750 - 3678 == 6 * _E


def test_no_owned_layers_leaves_the_explicit_cache_size_untouched():
    from freetoken.engine.cache_budget import lru_slots_after_owned_charge

    assert lru_slots_after_owned_charge(
        moe_cache_size=6750, owned_layers=0, num_experts=_E, floor=2 * _E
    ) == 6750


def test_the_charged_total_never_exceeds_the_uncharged_one_slot_for_slot():
    """VRAM neutrality, stated as the property that motivated the change."""
    from freetoken.engine.cache_budget import (
        gpu_owned_reservation_bytes,
        lru_slots_after_owned_charge,
    )

    for owned in range(0, 8):
        lru = lru_slots_after_owned_charge(
            moe_cache_size=6750, owned_layers=owned, num_experts=_E, floor=_E
        )
        resident = lru * _PER_EXPERT + gpu_owned_reservation_bytes(owned, _E, _PER_EXPERT)
        assert resident == 6750 * _PER_EXPERT


def test_a_budget_too_small_for_the_owned_set_plus_the_lru_floor_names_the_size_that_works():
    from freetoken.engine.cache_budget import lru_slots_after_owned_charge

    # 6 owned layers charge 3072 slots; with prefill overlap the LRU floor is 1024
    assert lru_slots_after_owned_charge(
        moe_cache_size=4096, owned_layers=6, num_experts=_E, floor=2 * _E
    ) == 1024
    with pytest.raises(ValueError) as excinfo:
        lru_slots_after_owned_charge(
            moe_cache_size=4095, owned_layers=6, num_experts=_E, floor=2 * _E
        )
    message = str(excinfo.value)
    assert "--moe-cache-size 4095" in message
    assert "3072 slots" in message
    assert "at least 1024" in message
    assert "4096" in message


def test_the_charge_is_reported_against_the_owned_reservation_bytes():
    """The slot charge and the byte reservation must describe the same memory."""
    from freetoken.engine.cache_budget import (
        gpu_owned_reservation_bytes,
        lru_slots_after_owned_charge,
    )

    lru = lru_slots_after_owned_charge(
        moe_cache_size=6750, owned_layers=6, num_experts=_E, floor=2 * _E
    )
    assert (6750 - lru) * _PER_EXPERT == gpu_owned_reservation_bytes(6, _E, _PER_EXPERT)


# ------------------------------------- post-cache VRAM reservations (--moe-vram-reserve-bytes)


def test_post_cache_reserved_bytes_sums_the_reserve_and_the_headroom():
    from freetoken.engine.cache_budget import (
        DEFAULT_MOE_CACHE_HEADROOM_BYTES,
        DEFAULT_MOE_VRAM_RESERVE_BYTES,
        post_cache_reserved_bytes,
    )

    # The default reserve covers the resident MTP draft head (2.17 GiB measured on the
    # RTX 5090) plus the CUDA-graph pool and the vision layer-stream workspace.
    assert DEFAULT_MOE_VRAM_RESERVE_BYTES == 3 << 30
    assert DEFAULT_MOE_CACHE_HEADROOM_BYTES == 3 << 29  # 1.5 GiB
    assert post_cache_reserved_bytes(vram_reserve_bytes=0, headroom_bytes=0) == 0
    assert post_cache_reserved_bytes(
        vram_reserve_bytes=3 << 30, headroom_bytes=3 << 29
    ) == (3 << 30) + (3 << 29)
    with pytest.raises(ValueError):
        post_cache_reserved_bytes(vram_reserve_bytes=-1, headroom_bytes=0)
    with pytest.raises(ValueError):
        post_cache_reserved_bytes(vram_reserve_bytes=0, headroom_bytes=-1)


def test_the_reserve_removes_exactly_its_bytes_worth_of_auto_slots():
    """--moe-cache-auto must not spend the bytes the MTP draft head takes after sizing."""
    from freetoken.engine.cache_budget import post_cache_reserved_bytes

    def auto(reserve: int, headroom: int):
        return resolve_moe_cache_auto(
            baseline_free=40 << 30,
            weights_bytes=8 << 30,
            memory_ratio=0.9,
            cache_per_page=1 << 20,
            fixed_cache_size=post_cache_reserved_bytes(
                vram_reserve_bytes=reserve, headroom_bytes=headroom
            ),
            per_expert_bytes=_PER_EXPERT,
            num_experts=_E,
            total_experts=_L * _E,
            prefill_overlap=True,
            kv_reserve_tokens=0,
            page_size=1,
            quant_format="nvfp4",
        )

    bare, _, _ = auto(0, 0)
    reserved, _, _ = auto(3 << 30, 3 << 29)
    # The reservation buys no expert slots, and costs at most one slot of rounding.
    dropped = ((3 << 30) + (3 << 29)) // _PER_EXPERT
    assert dropped <= bare - reserved <= dropped + 1


def test_an_explicit_cache_size_that_eats_the_reserve_is_refused_naming_the_largest_fit():
    """Operator decision: fail loudly, never silently shrink. The message must name the
    largest slot count that fits after every known reservation."""
    from freetoken.engine.cache_budget import check_explicit_moe_cache_fits

    budget = 8000 * _PER_EXPERT
    reserved = 1000 * _PER_EXPERT  # reserve + headroom, in slot-equivalents
    # 6 owned layers charge 3072 slots; 8000 - 1000 - 3072 = 3928 remain for the LRU.
    check_explicit_moe_cache_fits(
        moe_cache_size=3928, per_expert_bytes=_PER_EXPERT, budget_bytes=budget,
        owned_layers=6, num_experts=_E, reserved_bytes=reserved,
    )  # exactly fits
    with pytest.raises(ValueError) as excinfo:
        check_explicit_moe_cache_fits(
            moe_cache_size=3929, per_expert_bytes=_PER_EXPERT, budget_bytes=budget,
            owned_layers=6, num_experts=_E, reserved_bytes=reserved,
        )
    message = str(excinfo.value)
    assert "to 7000 slots" in message  # the largest TOTAL that fits: 3928 LRU + 3072 owned
    assert str(reserved) in message  # the reservation is named, not hidden


def test_the_reserve_is_refused_for_a_no_owned_configuration_too():
    """The post-cache reservation bug is general: it bites --moe-cache-size with or without
    --moe-gpu-owned-layers."""
    from freetoken.engine.cache_budget import check_explicit_moe_cache_fits

    budget = 8000 * _PER_EXPERT
    reserved = 1000 * _PER_EXPERT
    check_explicit_moe_cache_fits(
        moe_cache_size=7000, per_expert_bytes=_PER_EXPERT, budget_bytes=budget,
        owned_layers=0, num_experts=_E, reserved_bytes=reserved,
    )
    with pytest.raises(ValueError) as excinfo:
        check_explicit_moe_cache_fits(
            moe_cache_size=7001, per_expert_bytes=_PER_EXPERT, budget_bytes=budget,
            owned_layers=0, num_experts=_E, reserved_bytes=reserved,
        )
    assert "7000" in str(excinfo.value)


# ------------------ L4: the refusal must quote the TYPED total and name a TOTAL that boots


#: Boot A of the 2026-09-02 live check, verbatim: RTX 5090 32 GB, Qwen3.8-Flash-Next-NVFP4,
#: ``-GpuOwnedLayers auto -MoECacheSize 6750``, default reserve + headroom.
#: See docs/research/measurements-gpu-owned-followups-live-2026-09-02.md.
_LIVE_BUDGET = 20_861_318_737           # net MoE budget after weights, KV reserve, GDN pool
_LIVE_RESERVED = (3 << 30) + (3 << 29)  # 3 GiB post-cache reserve + 1.5 GiB headroom
_LIVE_TOTAL = 6750                      # what the operator typed


def test_the_refusal_quotes_the_operators_total_and_names_a_total_that_boots():
    """Live boot A quoted a size nobody typed and named one that cannot be typed.

    The engine charges the owned layers to the explicit total BEFORE the fit check runs, so
    the check used to read the post-charge LRU count (3678) out of ``config.moe_cache_size``
    and print it as ``--moe-cache-size 3678`` -- the operator typed 6750. Worse, it named
    ``fits_slots`` (2709, also an LRU count) as the size to lower ``--moe-cache-size`` to,
    and ``--moe-cache-size`` is the TOTAL budget: boot B typed 2709 and was refused by the
    LRU-floor check (2709 - 3072 < 1024). The total that boots is 2709 + 3072 = 5781.
    """
    from freetoken.engine.cache_budget import (
        check_explicit_moe_cache_fits,
        lru_slots_after_owned_charge,
    )

    lru = lru_slots_after_owned_charge(
        moe_cache_size=_LIVE_TOTAL, owned_layers=6, num_experts=_E, floor=2 * _E
    )
    assert lru == 3678  # what the engine writes back over the operator's 6750

    with pytest.raises(ValueError) as excinfo:
        check_explicit_moe_cache_fits(
            moe_cache_size=lru, per_expert_bytes=_PER_EXPERT, budget_bytes=_LIVE_BUDGET,
            owned_layers=6, num_experts=_E, reserved_bytes=_LIVE_RESERVED,
            requested_total=_LIVE_TOTAL,
        )
    message = str(excinfo.value)

    # (a) the flag is quoted with the number the operator actually typed
    assert "--moe-cache-size 6750" in message
    assert "--moe-cache-size 3678" not in message
    # the operator boots through the launcher, so name that spelling too (issue 7)
    assert "-MoECacheSize" in message
    # the diagnosis is still there: what 6750 bought once the owned layers took their cut
    assert "3678" in message and "3072" in message

    # (b) the size it names is a TOTAL, and it is the largest total that fits
    named = int(re.search(r"to (\d+) slots", message).group(1))
    assert named == 5781 == 2709 + 6 * _E

    # ...and that total survives BOTH checks: the LRU floor first (boot B's refusal),
    assert named >= 4096  # 6 x 512 charged + the 1024-slot prefill-overlap floor
    assert lru_slots_after_owned_charge(
        moe_cache_size=named, owned_layers=6, num_experts=_E, floor=2 * _E
    ) == 2709
    # ...then the budget check that produced this very message.
    check_explicit_moe_cache_fits(
        moe_cache_size=named - 6 * _E, per_expert_bytes=_PER_EXPERT,
        budget_bytes=_LIVE_BUDGET, owned_layers=6, num_experts=_E,
        reserved_bytes=_LIVE_RESERVED, requested_total=named,
    )
    # one slot more does not fit, so 5781 really is the largest
    with pytest.raises(ValueError):
        check_explicit_moe_cache_fits(
            moe_cache_size=named + 1 - 6 * _E, per_expert_bytes=_PER_EXPERT,
            budget_bytes=_LIVE_BUDGET, owned_layers=6, num_experts=_E,
            reserved_bytes=_LIVE_RESERVED, requested_total=named + 1,
        )


def test_the_typed_total_is_reconstructed_when_the_caller_does_not_pass_it():
    """``requested_total`` is optional: the charge is exactly one expert layer per owned
    layer, so a caller that only has the post-charge LRU still gets the operator's total."""
    from freetoken.engine.cache_budget import check_explicit_moe_cache_fits

    with pytest.raises(ValueError) as excinfo:
        check_explicit_moe_cache_fits(
            moe_cache_size=3678, per_expert_bytes=_PER_EXPERT, budget_bytes=_LIVE_BUDGET,
            owned_layers=6, num_experts=_E, reserved_bytes=_LIVE_RESERVED,
        )
    assert "--moe-cache-size 6750" in str(excinfo.value)


def _live_engine_stubs():
    """A config + banks pair with the live box's MoE geometry, for driving
    ``Engine._init_offload_moe_cache`` up to the explicit-fit refusal on the CPU."""
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        is_moe = True
        num_experts = _E
        num_moe_layers = _L
        expert_quant = "nvfp4"
        moe_backend = "offload"
        nvfp4_backend = "triton"

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64,
                sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    config = SimpleNamespace(
        model_config=StubModelConfig(),
        model_path="/models/anon",
        dtype=torch.float16,
        page_size=64,
        max_running_req=1,
        hybrid_swa_cache_mode="auto",
        memory_ratio=0.9,
        max_seq_len=65536,
        swa_full_tokens_ratio=0.2,
        swa_num_pages_override=None,
        kv_reserve_tokens=0,
        moe_backend="offload",
        moe_cpu_layers=None,
        moe_gpu_owned_layers="auto",
        moe_prefill_overlap=True,
        moe_cache_size=_LIVE_TOTAL,
        moe_cache_auto=False,
        moe_cache_policy="lru",
        moe_prefill_hit_d2d=False,
        moe_hybrid_max_fetch=1,
        moe_vram_reserve_bytes=3 << 30,
        moe_cache_headroom_bytes=3 << 29,
        expert_load=None,
        use_dummy_weight=True,
        # the live boot ran integrated MTP speculation (the 2.17 GiB resident draft head)
        spec_decode=SimpleNamespace(enabled=True),
        tp_info=SimpleNamespace(size=1),
    )

    # one bank whose row is a whole live expert slot; layer 3 is the first STREAMING layer
    row = torch.zeros(2, _PER_EXPERT, dtype=torch.uint8)
    banks = SimpleNamespace(
        quant_format="nvfp4",
        sources={"gate_up": [row] * _L},
        layer_residency=None,
        gate_up_alpha=None,
        down_alpha=None,
    )
    return config, banks


def test_the_engine_refusal_carries_the_typed_total_through_the_charge(monkeypatch):
    """End to end at the engine seam: ``_charge_gpu_owned_layers_to_cache_size`` rewrites
    ``config.moe_cache_size``, and the refusal raised further down still names 6750.

    Also pins WHERE the refusal happens. It is raised from ``_init_offload_moe_cache``
    immediately after ``load_expert_banks`` -- i.e. after the ~40 s bank read boot A paid
    before being told its size was wrong. Every other term of the check is known at config
    time; the one that is not is ``expert_bytes_per_slot``, measured off the loaded bank
    rows. The config-time ``_BANK_BYTES_PER_EXPERT`` formula cannot stand in for it: it is
    documented as an over-estimate for the repacked marlin/b12x layouts and it is TP-blind,
    so a refusal built on it could refuse a size that fits. Hoisting the check therefore
    needs a new format- and TP-aware pre-load slot-byte estimate, not a move. When that
    estimate exists, the last assertion here becomes ``calls == []``.
    """
    from freetoken.engine import engine as engine_module
    from freetoken.engine.engine import Engine
    from freetoken.kvcache.mha_pool import MHAKVCache

    config, banks = _live_engine_stubs()
    calls: list[str] = []

    def fake_load_expert_banks(model_path, model_config, **kwargs):
        calls.append(model_path)
        return banks

    monkeypatch.setattr(engine_module, "load_expert_banks", fake_load_expert_banks)
    monkeypatch.setattr(engine_module, "_pin_budget_bytes", lambda _reserved: None)

    engine = Engine.__new__(Engine)
    engine.model = SimpleNamespace()          # no make_offload_moe_cache
    engine.device = torch.device("cpu")
    engine.dtype = torch.float16
    engine._host_tables_bytes = 0
    engine._pool_cls = MHAKVCache
    engine._baseline_free = 32_479_182_848    # 30.25 GiB, the live card
    engine._weights_bytes = 5_750_000_000     # ~5.36 GiB of weights: 6750 no longer fits

    with pytest.raises(ValueError) as excinfo:
        engine._init_offload_moe_cache(config)
    message = str(excinfo.value)

    assert config.moe_cache_size == 3678       # the charge did happen
    assert "--moe-cache-size 6750" in message  # ...and the refusal still quotes 6750
    named = int(re.search(r"to (\d+) slots", message).group(1))
    assert named >= 4096                       # the named total clears the LRU floor
    assert calls == ["/models/anon"]           # see the docstring: still one bank load


def test_the_vram_ledger_lists_every_term_in_one_block():
    from freetoken.engine.cache_budget import format_vram_ledger

    block = format_vram_ledger(
        total_bytes=32 << 30,
        weights_bytes=8 << 30,
        kv_bytes=1 << 30,
        gdn_state_bytes=2 << 30,
        gpu_owned_bytes=6 * _E * _PER_EXPERT,
        gpu_owned_layers=(0, 1, 2, 6, 7, 22),
        lru_slots=3678,
        lru_bytes=3678 * _PER_EXPERT,
        vram_reserve_bytes=3 << 30,
        headroom_bytes=3 << 29,
    )
    for label in (
        "VRAM ledger",
        "weights",
        "KV cache",
        "GDN state pool",
        "GPU-owned MoE layers",
        "MoE LRU cache",
        "post-cache reserve",
        "headroom",
        "unaccounted",
    ):
        assert label in block, label
    assert "[0, 1, 2, 6, 7, 22]" in block
    assert "3678 slots" in block
    # one line per term, no wall of text
    assert len(block.splitlines()) == 10


def test_engine_auto_sizing_charges_the_declared_post_cache_reserve():
    """Engine._resolve_auto_moe_cache_size must fold the declared reserve into
    fixed_cache_size, exactly as it folds the GDN state pool and the owned reservation."""
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        num_experts = 64
        num_moe_layers = 4

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64,
                sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_prefill_overlap = True
        kv_reserve_tokens = 32
        max_seq_len = 1 << 20
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        moe_vram_reserve_bytes = 0
        moe_cache_headroom_bytes = 0
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    class StubBanks:
        quant_format = "bf16"
        sources = {
            "gate_up": [torch.zeros(64, 32, 8, dtype=torch.float16)] * 4,  # row = 512 B
            "down": [torch.zeros(64, 8, 16, dtype=torch.float16)] * 4,     # row = 256 B
        }

    def resolve(reserve: int, headroom: int) -> int:
        config = StubConfig()
        config.moe_vram_reserve_bytes = reserve
        config.moe_cache_headroom_bytes = headroom
        engine = Engine.__new__(Engine)
        engine._baseline_free = 500_000
        engine._weights_bytes = 0
        engine._pool_cls = MHAKVCache
        size, _, _ = engine._resolve_auto_moe_cache_size(config, StubBanks())
        return size

    per_slot = 512 + 256
    assert resolve(0, 0) - resolve(4 * per_slot, 2 * per_slot) == 6


# ------------------------- the reserve is only charged for what the boot actually allocates


def test_the_auto_reserve_drops_the_draft_head_when_speculation_is_off():
    """The 2.17 GiB resident MTP draft head only exists when speculation is on. Charging it
    on every boot would cost ~800 expert slots to reserve bytes nothing will allocate."""
    from freetoken.engine.cache_budget import (
        DEFAULT_MOE_VRAM_RESERVE_BYTES,
        GRAPH_POOL_RESERVE_BYTES,
        MTP_DRAFT_HEAD_RESERVE_BYTES,
        auto_vram_reserve_bytes,
    )

    assert MTP_DRAFT_HEAD_RESERVE_BYTES + GRAPH_POOL_RESERVE_BYTES == (
        DEFAULT_MOE_VRAM_RESERVE_BYTES
    )
    assert MTP_DRAFT_HEAD_RESERVE_BYTES >= 2.17 * 2**30  # the measured draft head fits
    assert auto_vram_reserve_bytes(mtp_resident=True) == DEFAULT_MOE_VRAM_RESERVE_BYTES
    assert auto_vram_reserve_bytes(mtp_resident=False) == GRAPH_POOL_RESERVE_BYTES


def test_an_explicit_reserve_overrides_the_auto_composition():
    from freetoken.engine.cache_budget import (
        GRAPH_POOL_RESERVE_BYTES,
        resolve_vram_reserve_bytes,
    )

    assert resolve_vram_reserve_bytes(-1, mtp_resident=False) == GRAPH_POOL_RESERVE_BYTES
    assert resolve_vram_reserve_bytes(0, mtp_resident=True) == 0
    assert resolve_vram_reserve_bytes(1 << 20, mtp_resident=True) == 1 << 20
    with pytest.raises(ValueError):
        resolve_vram_reserve_bytes(-2, mtp_resident=True)


def _auto_sizing_stubs():
    """Engine stubs for _resolve_auto_moe_cache_size without a device (see the kwarg test)."""
    import torch

    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        num_experts = 64
        num_moe_layers = 4

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64,
                sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubSpecDecode:
        enabled = False

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_prefill_overlap = True
        kv_reserve_tokens = 32
        max_seq_len = 1 << 20
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        moe_vram_reserve_bytes = -1  # auto
        moe_cache_headroom_bytes = 0
        model_config = StubModelConfig()
        spec_decode = StubSpecDecode()

        class tp_info:
            size = 1

    class StubBanks:
        quant_format = "bf16"
        sources = {
            "gate_up": [torch.zeros(64, 32, 8, dtype=torch.float16)] * 4,  # row = 512 B
            "down": [torch.zeros(64, 8, 16, dtype=torch.float16)] * 4,     # row = 256 B
        }

    return StubConfig, StubBanks


def test_engine_auto_sizing_charges_the_draft_head_only_when_speculation_is_on():
    from freetoken.engine.cache_budget import MTP_DRAFT_HEAD_RESERVE_BYTES
    from freetoken.engine.engine import Engine
    from freetoken.kvcache.mha_pool import MHAKVCache

    StubConfig, StubBanks = _auto_sizing_stubs()

    def reserve(spec_enabled: bool) -> int:
        config = StubConfig()
        config.spec_decode.enabled = spec_enabled
        engine = Engine.__new__(Engine)
        engine._baseline_free = 1 << 33
        engine._weights_bytes = 0
        engine._pool_cls = MHAKVCache
        # the same number _resolve_auto_moe_cache_size folds into fixed_cache_size
        assert engine._resolve_auto_moe_cache_size(config, StubBanks())[0] >= 0
        return engine._post_cache_reserve(config)

    assert reserve(True) - reserve(False) == MTP_DRAFT_HEAD_RESERVE_BYTES


def test_the_vram_ledger_reports_the_kv_pages_the_boot_actually_took():
    """The ledger is emitted where num_pages is known. It used to be printed inside the MoE
    cache build -- before the KV pool is sized -- so its KV row read 0 pages on every boot
    that did not pass --num-tokens, and 'unaccounted' absorbed the whole KV pool."""
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.kvcache.mha_pool import MHAKVCache

    StubConfig, StubBanks = _auto_sizing_stubs()
    config = StubConfig()
    config.num_page_override = None
    lines: list[str] = []

    engine = Engine.__new__(Engine)
    engine._baseline_free = 1 << 33
    engine._weights_bytes = 1 << 30
    engine._pool_cls = MHAKVCache
    engine.num_pages = 321
    engine._vram_ledger_inputs = {
        "per_expert_bytes": 512 + 256,
        "gpu_owned_layers": (0, 1),
    }
    config.moe_cache_size = 100

    from freetoken.engine import engine as engine_module

    class _Recorder:
        def info_rank0(self, message, *args):
            lines.append(message % args if args else message)

    original, engine_module.logger = engine_module.logger, _Recorder()
    try:
        engine._log_vram_ledger(config)
    finally:
        engine_module.logger = original

    block = "\n".join(lines)
    assert "VRAM ledger" in block
    cache_per_page, kv_fixed, _, _ = MHAKVCache.kv_cost(config)
    expected = (kv_fixed + 321 * cache_per_page) / 2**30
    assert f"KV cache" in block
    assert f"{expected:.2f} GiB" in block
    assert torch  # keep the import honest for the stub tensors above

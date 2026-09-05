"""--moe-gpu-owned-layers: spec grammar, resolver, and the _adjust_config gate.

CPU-only: nothing here builds an engine, allocates a bank, or touches CUDA.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from freetoken.engine.engine import GPU_OWNED_LAYER_RANK
from freetoken.engine.engine import _parse_gpu_owned_layers_spec as parse
from freetoken.engine.engine import _resolve_gpu_owned_layers as resolve

L = 48
ANON_MODEL = "/models/anon"


def _cfg(**over):
    base = dict(
        moe_backend="offload",
        moe_cpu_layers=None,
        moe_gpu_owned_layers=None,
        model_path=ANON_MODEL,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------ the grammar


def test_the_ranked_list_is_the_measured_order_and_covers_every_layer():
    # derived from docs/research/routing-skew-2026-09-02/{code,prose,chat8k,toolcall}.json
    assert GPU_OWNED_LAYER_RANK[:8] == (1, 6, 0, 2, 7, 22, 10, 13)
    assert len(GPU_OWNED_LAYER_RANK) == L
    assert sorted(GPU_OWNED_LAYER_RANK) == list(range(L))


def test_auto_is_the_six_hungriest_layers():
    assert parse("auto", L) == frozenset({0, 1, 2, 6, 7, 22})


@pytest.mark.parametrize("n,expected", [(1, {1}), (3, {1, 6, 0}), (8, {1, 6, 0, 2, 7, 22, 10, 13})])
def test_auto_n_takes_the_first_n_of_the_ranked_list(n, expected):
    assert parse(f"auto:{n}", L) == frozenset(expected)


def test_auto_zero_is_empty_and_auto_all_is_every_layer():
    assert parse("auto:0", L) == frozenset()
    assert len(parse(f"auto:{L}", L)) == L


def test_explicit_list_count_and_fraction_reuse_the_cpu_layers_grammar():
    assert parse("3,7,11", L) == frozenset({3, 7, 11})
    assert parse("3, 7 ,11,", L) == frozenset({3, 7, 11})
    assert parse("5,5,5", L) == frozenset({5})
    assert parse("6", L) == frozenset({0, 8, 16, 24, 32, 40})
    assert len(parse("0.125", L)) == 6
    assert parse("", L) == frozenset()
    assert parse("   ", L) == frozenset()


@pytest.mark.parametrize("spec", ["99", "48,1", "-1", "1.5", "auto:-1", "auto:49", "auto:x"])
def test_out_of_range_specs_raise(spec):
    with pytest.raises(ValueError):
        parse(spec, L)


def test_the_error_names_the_flag_not_moe_cpu_layers():
    with pytest.raises(ValueError, match="--moe-gpu-owned-layers"):
        parse("99", L)


# ------------------------------------------------------------------ the resolver


def test_resolve_needs_an_offload_backend_and_a_spec():
    assert resolve(_cfg(moe_gpu_owned_layers="auto"), L) == frozenset({0, 1, 2, 6, 7, 22})
    assert resolve(_cfg(moe_gpu_owned_layers=None), L) == frozenset()
    assert resolve(_cfg(moe_backend="fused", moe_gpu_owned_layers="auto"), L) == frozenset()
    assert resolve(_cfg(moe_backend="cpu", moe_gpu_owned_layers="auto"), L) == frozenset()


# ------------------------------------------------------------ the learned order


def _learned_model(tmp_path, *, routes_per_expert, num_experts=512, ranked_first=(40, 41, 42)):
    """A checkpoint dir with a stats file whose broadest layers are ``ranked_first``."""
    from freetoken.moe.learned_routing import RoutingStatsRecorder

    freq = [[0] * num_experts for _ in range(L)]
    for layer in range(L):
        freq[layer][0] = routes_per_expert * 8  # narrow: one expert
    for layer in ranked_first:
        freq[layer] = [routes_per_expert] * num_experts  # broad: every expert
    rec = RoutingStatsRecorder(tmp_path, num_layers=L, num_experts=num_experts)
    rec.note(freq)
    assert rec.save()
    return _cfg(
        model_path=str(tmp_path),
        model_config=SimpleNamespace(num_experts=num_experts),
        moe_learn_routing=True,
    )


def test_auto_uses_the_learned_order_when_the_checkpoint_has_enough_routes(tmp_path):
    cfg = _learned_model(tmp_path, routes_per_expert=100)
    cfg.moe_gpu_owned_layers = "auto:3"
    assert resolve(cfg, L) == frozenset({40, 41, 42})
    cfg.moe_gpu_owned_layers = "auto"
    learned_six = resolve(cfg, L)
    assert {40, 41, 42} <= learned_six and len(learned_six) == 6


def test_auto_keeps_the_fixed_order_with_too_few_routes(tmp_path):
    cfg = _learned_model(tmp_path, routes_per_expert=1)
    cfg.moe_gpu_owned_layers = "auto:3"
    assert resolve(cfg, L) == frozenset({1, 6, 0})


def test_auto_keeps_the_fixed_order_when_learning_is_off_or_the_geometry_differs(tmp_path):
    cfg = _learned_model(tmp_path, routes_per_expert=100)
    cfg.moe_gpu_owned_layers = "auto:3"
    cfg.moe_learn_routing = False
    assert resolve(cfg, L) == frozenset({1, 6, 0})
    cfg.moe_learn_routing = True
    cfg.model_config = SimpleNamespace(num_experts=288)  # another model's file shape
    assert resolve(cfg, L) == frozenset({1, 6, 0})


def test_an_explicit_layer_list_ignores_the_stats_file(tmp_path):
    cfg = _learned_model(tmp_path, routes_per_expert=100)
    cfg.moe_gpu_owned_layers = "3,7"
    assert resolve(cfg, L) == frozenset({3, 7})


def test_configs_without_a_model_config_still_resolve_auto():
    # SimpleNamespace configs elsewhere in this file carry no model_config; the learned path
    # must step aside instead of raising.
    assert resolve(_cfg(moe_gpu_owned_layers="auto:2", moe_learn_routing=True), L) == frozenset({1, 6})


# ------------------------------------------------------------ the _adjust_config gate


class _HFConfig:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse_args(*extra: str):
    from freetoken.server.args import parse_args

    config = _HFConfig({"architectures": ["Qwen2ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        args, _run_shell = parse_args(["--model", ANON_MODEL, *extra])
    return args


def test_the_flag_round_trips_through_server_args():
    assert _parse_args().moe_gpu_owned_layers is None
    assert _parse_args("--moe-gpu-owned-layers", "auto").moe_gpu_owned_layers == "auto"
    assert _parse_args("--moe-gpu-owned-layers", "0,1,2").moe_gpu_owned_layers == "0,1,2"


def _adjust(**over):
    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        is_moe=True,
        num_moe_layers=L,
        num_experts=512,
        expert_quant="nvfp4",
        moe_backend="offload",
        nvfp4_backend="triton",
    )
    config = SimpleNamespace(
        model_config=model_config,
        model_path=ANON_MODEL,
        moe_backend="offload",
        moe_cpu_layers=None,
        moe_gpu_owned_layers="auto",
        moe_prefill_overlap=True,
        moe_cache_size=6750,
        moe_cache_auto=False,
    )
    for name, value in over.items():
        setattr(config, name, value)
    return config, _adjust_config


def test_validation_requires_the_offload_backend():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config, _ = _adjust(moe_backend="cpu")
    with pytest.raises(ValueError, match="requires --moe-backend offload"):
        _validate_gpu_owned_layers(config, L)


def test_validation_rejects_overlap_with_the_cpu_layer_set():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config, _ = _adjust(moe_cpu_layers="0,1", moe_gpu_owned_layers="auto")
    with pytest.raises(ValueError, match=r"both GPU-owned and CPU layers: \[0, 1\]"):
        _validate_gpu_owned_layers(config, L)


def test_validation_rejects_an_ftw_checkpoint(tmp_path):
    from freetoken.engine.engine import _validate_gpu_owned_layers

    (tmp_path / "freetoken_weight.json").write_text('{"tensors": []}', encoding="utf-8")
    config, _ = _adjust(model_path=str(tmp_path))
    with pytest.raises(ValueError, match="FTW packed checkpoint"):
        _validate_gpu_owned_layers(config, L)


def test_validation_keeps_two_expert_layers_of_lru_when_overlap_is_on():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    # --moe-cache-size is the TOTAL budget: 6 owned layers charge 6 * 512 = 3072 slots of it,
    # and prefill overlap needs 2 * 512 = 1024 left over, so the floor on the total is 4096.
    config, _ = _adjust(moe_cache_size=4095)
    with pytest.raises(ValueError, match="need at least 1024"):
        _validate_gpu_owned_layers(config, L)
    config, _ = _adjust(moe_cache_size=4096)
    assert _validate_gpu_owned_layers(config, L) == frozenset({0, 1, 2, 6, 7, 22})


def test_validation_does_not_charge_the_size_when_auto_sizing_owns_the_budget():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    # --moe-cache-auto charges the reservation through fixed_cache_size instead; a leftover
    # moe_cache_size must not be re-checked (or re-charged) against the overlap floor here.
    config, _ = _adjust(moe_cache_size=1024, moe_cache_auto=True)
    assert _validate_gpu_owned_layers(config, L) == frozenset({0, 1, 2, 6, 7, 22})


def test_the_engine_charges_the_owned_layers_to_an_explicit_cache_size_once():
    """The run-2 regression, in one assertion: 6750 + auto must stay a 6750-slot card."""
    from freetoken.engine.engine import Engine

    config, _ = _adjust(moe_cache_size=6750, moe_cache_auto=False)
    owned = frozenset({0, 1, 2, 6, 7, 22})

    Engine._charge_gpu_owned_layers_to_cache_size(None, config, owned)
    assert config.moe_cache_size == 3678  # 6750 - 6 * 512

    # idempotence is not claimed, so the guard is that the call site is single: assert the
    # engine only ever calls it from _init_offload_moe_cache.
    import inspect

    source = inspect.getsource(Engine)
    assert source.count("_charge_gpu_owned_layers_to_cache_size(") == 2  # def + one call


def test_the_engine_leaves_the_cache_size_alone_without_owned_layers_or_under_auto():
    from freetoken.engine.engine import Engine

    config, _ = _adjust(moe_cache_size=6750, moe_cache_auto=False)
    Engine._charge_gpu_owned_layers_to_cache_size(None, config, frozenset())
    assert config.moe_cache_size == 6750

    config, _ = _adjust(moe_cache_size=6750, moe_cache_auto=True)
    Engine._charge_gpu_owned_layers_to_cache_size(None, config, frozenset({0, 1}))
    assert config.moe_cache_size == 6750


def test_validation_is_inert_without_the_flag():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config, _ = _adjust(moe_gpu_owned_layers=None, moe_backend="fused")
    assert _validate_gpu_owned_layers(config, L) == frozenset()


def test_the_dense_override_table_clears_the_flag():
    """A dense checkpoint has no routed experts; the knob must not survive the reset."""
    from freetoken.engine.engine import _DENSE_MOE_SETTINGS

    assert _DENSE_MOE_SETTINGS["moe_gpu_owned_layers"] is None


# ------------------------------------------- issue 7: both refusals must name the fix


def test_the_cpu_layer_clash_says_a_layer_cannot_be_both_and_how_to_fix_it():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config, _ = _adjust(moe_cpu_layers="0,1", moe_gpu_owned_layers="auto")
    with pytest.raises(ValueError) as excinfo:
        _validate_gpu_owned_layers(config, L)
    message = str(excinfo.value)
    assert "[0, 1]" in message
    assert "cannot be both" in message
    # the fix, not just the diagnosis
    assert "--moe-cpu-layers" in message and "--moe-gpu-owned-layers" in message
    assert "Drop" in message


def test_the_lru_floor_refusal_names_the_size_to_raise_to_in_both_spellings():
    """The operator boots through the launcher, so the refusal has to name -MoECacheSize
    as well as the engine flag, and the number to raise it TO."""
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config, _ = _adjust(moe_cache_size=4095)
    with pytest.raises(ValueError) as excinfo:
        _validate_gpu_owned_layers(config, L)
    message = str(excinfo.value)
    assert "4096" in message  # 6 * 512 charged + the 1024-slot overlap floor
    assert "--moe-cache-size" in message
    assert "-MoECacheSize" in message
    assert "at least" in message


# ------------------------------------------- issue 7: auto:N beyond the first eight


@pytest.mark.parametrize("n", [9, 13, 24, 47, 48])
def test_auto_n_keeps_working_past_the_documented_first_eight(n):
    """The doc tables only print the first eight ranks; the list has all 48 and auto:N must
    take the first N of it, in ranked order, for any N up to the layer count."""
    owned = parse(f"auto:{n}", L)
    assert owned == frozenset(GPU_OWNED_LAYER_RANK[:n])
    assert len(owned) == n
    assert parse("auto:8", L) < owned  # ranked order, so each N is a superset of the last


def test_auto_n_larger_than_the_ranked_list_refuses_instead_of_silently_truncating():
    """A model with more MoE layers than the ranked list has entries: auto:N would have
    quietly returned len(list) layers instead of N."""
    with pytest.raises(ValueError, match="ranked list"):
        parse("auto:60", 96)
    # and the list itself still covers every layer of a model that IS 48 layers deep
    assert len(parse("auto:48", L)) == 48


# ------------------------------------------------------------------ the boot line


def test_the_boot_line_reports_the_owned_set_the_resident_bytes_and_the_lru():
    from freetoken.engine.engine import _gpu_owned_boot_line

    line = _gpu_owned_boot_line(
        owned=frozenset({0, 1, 2, 6, 7, 22}),
        num_moe_layers=48,
        num_experts=512,
        per_expert_bytes=2_772_480,
        cache_size=4400,
    )

    assert line == (
        "MoE GPU-owned layers: [0, 1, 2, 6, 7, 22] (6 x 1.32 GiB resident, no host bank); "
        "LRU cache 4400 slots for 42 streaming layers"
    )


# ------------------------------------------------------------------ the Windows launcher


def _launcher_text() -> str:
    from pathlib import Path

    return (
        Path(__file__).parents[2] / "scripts" / "start-qwen38-flash-next-mmap-windows.ps1"
    ).read_text(encoding="utf-8")


def test_the_launcher_exposes_gpu_owned_layers_and_defaults_to_off():
    launcher = _launcher_text()

    assert "[string]$GpuOwnedLayers = ''" in launcher
    assert "$env:FREETOKEN_MOE_GPU_OWNED_LAYERS" in launcher
    assert "'--moe-gpu-owned-layers', $GpuOwnedLayers" in launcher
    assert "if ($GpuOwnedLayers) {" in launcher
    # never a hard-coded set: the flag is only ever built from the parameter
    assert "'--moe-gpu-owned-layers', 'auto'" not in launcher


def test_the_launcher_banner_reports_the_owned_spec():
    assert "GPU-owned MoE layers: $(if ($GpuOwnedLayers) { $GpuOwnedLayers } else { 'off' })" in _launcher_text()


def test_the_launcher_exposes_the_vram_reserve_knobs():
    """The operator boots only through the launcher, and the post-cache reserve can refuse
    an explicit -MoECacheSize. Without a passthrough the only way past a wrong reserve would
    be editing the engine."""
    launcher = _launcher_text()

    assert "[long]$MoEVramReserveBytes = -1" in launcher
    assert "[long]$MoECacheHeadroomBytes = -1" in launcher
    assert "'--moe-vram-reserve-bytes', \"$MoEVramReserveBytes\"" in launcher
    assert "'--moe-cache-headroom-bytes', \"$MoECacheHeadroomBytes\"" in launcher
    # -1 means "leave the engine default alone", so the flag is not passed at all
    assert "if ($MoEVramReserveBytes -ge 0) {" in launcher
    assert "if ($MoECacheHeadroomBytes -ge 0) {" in launcher


def test_the_docs_describe_the_vram_reserve_knobs():
    from pathlib import Path

    root = Path(__file__).parents[2]
    cli = (root / "docs" / "cli.md").read_text(encoding="utf-8")
    windows = (root / "docs" / "windows-qwen38-flash-next-mmap.md").read_text(encoding="utf-8")
    for flag in ("--moe-vram-reserve-bytes", "--moe-cache-headroom-bytes"):
        assert flag in cli, flag
        assert flag in windows, flag
    assert "-MoEVramReserveBytes" in windows
    assert "-MoECacheHeadroomBytes" in windows
    assert "VRAM ledger" in windows


def test_the_docs_describe_the_flag():
    from pathlib import Path

    root = Path(__file__).parents[2]
    assert "--moe-gpu-owned-layers" in (root / "docs" / "cli.md").read_text(encoding="utf-8")
    windows = (root / "docs" / "windows-qwen38-flash-next-mmap.md").read_text(encoding="utf-8")
    assert "### GPU-owned MoE layers" in windows
    assert "-GpuOwnedLayers" in windows


# ------------------------------------------------------------------ the operator checklist


def test_the_operator_checklist_covers_every_live_check_the_spec_asks_for():
    from pathlib import Path

    status = (
        Path(__file__).parents[2] / "docs" / "plans"
        / "2026-09-02-qwen38-gpu-owned-moe-layers-status.md"
    ).read_text(encoding="utf-8")

    for heading in (
        "## Commits",
        "## CPU test results",
        "## Live verification",
        "## Only a live GPU run can decide this",
    ):
        assert heading in status
    for check in (
        "boot log shows owned set",
        # CORRECTED after run 2: the host banks are mapped pages, so the saving shows in
        # working set / physical in use, never in private bytes.
        "scheduler working set",
        "whole-system commit",
        "whole-system physical in-use",
        "boot peak host RAM",
        "8k-chat decode tok/s",
        "TTFT",
        "answers at temperature 0",
        "picture request",
        "/v1/cache/routing",
        "owned-layer rows on device",
    ):
        assert check in status, check


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))

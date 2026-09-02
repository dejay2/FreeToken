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

    # 512 experts, prefill overlap on -> the LRU floor is 2 * 512 = 1024 slots
    config, _ = _adjust(moe_cache_size=1023)
    with pytest.raises(ValueError, match="prefill overlap needs at least 1024"):
        _validate_gpu_owned_layers(config, L)
    config, _ = _adjust(moe_cache_size=1024)
    assert _validate_gpu_owned_layers(config, L) == frozenset({0, 1, 2, 6, 7, 22})


def test_validation_is_inert_without_the_flag():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config, _ = _adjust(moe_gpu_owned_layers=None, moe_backend="fused")
    assert _validate_gpu_owned_layers(config, L) == frozenset()


def test_the_dense_override_table_clears_the_flag():
    """A dense checkpoint has no routed experts; the knob must not survive the reset."""
    from freetoken.engine.engine import _DENSE_MOE_SETTINGS

    assert _DENSE_MOE_SETTINGS["moe_gpu_owned_layers"] is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))

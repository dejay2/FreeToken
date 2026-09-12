"""Flag plumbing for the dynamic KV pool (spec: Engine/args section)."""
from __future__ import annotations

import json

import pytest

from freetoken.server.args import parse_args

_CONTEXT = 4_194_304  # generous default so ceiling checks that pass an override never hit it


def _model_dir(tmp_path, max_position_embeddings: int = _CONTEXT) -> str:
    """A minimal local model folder, the way tests/server/test_parser_auto_selection.py's
    tests build one: just enough config.json for cached_load_hf_config to resolve."""
    folder = tmp_path / "model"
    folder.mkdir()
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "torch_dtype": "bfloat16",
        "max_position_embeddings": max_position_embeddings,
    }
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(folder)


def _base(tmp_path, **kwargs):
    return ["--model", _model_dir(tmp_path, **kwargs), "--moe-backend", "offload", "--kv-dtype", "fp8"]


def test_dynamic_off_by_default_changes_nothing(tmp_path):
    a, _ = parse_args(_base(tmp_path) + ["--num-tokens", "262208"])
    assert not a.kv_dynamic and a.num_token_override == 262208 and a.kv_ceiling_tokens is None


def test_dynamic_rewrites_the_boot_pool_to_the_floor_and_records_the_ceiling(tmp_path):
    a, _ = parse_args(
        _base(tmp_path)
        + ["--kv-dynamic", "--num-tokens", "262208", "--kv-reserve-tokens", "262144"]
    )
    assert a.kv_dynamic
    assert a.kv_ceiling_tokens == 262208
    assert a.num_token_override == 65_536 + 64      # floor plus the dummy page
    assert a.kv_reserve_tokens == 65_536
    assert a.kv_step_tokens == 32_768 and a.kv_shrink_idle_s == 600 and a.kv_park_ttl_s == 18_000


def test_dynamic_without_num_tokens_uses_the_context_as_ceiling(tmp_path):
    a, _ = parse_args(_base(tmp_path) + ["--kv-dynamic", "--max-seq-len-override", "131072"])
    assert a.kv_ceiling_tokens == 131072 + 64


def test_dynamic_without_num_tokens_or_override_uses_the_model_context(tmp_path):
    a, _ = parse_args(_base(tmp_path, max_position_embeddings=131072) + ["--kv-dynamic"])
    assert a.kv_ceiling_tokens == 131072 + 64


@pytest.mark.parametrize("extra", [
    ["--kv-dynamic", "--kv-step-tokens", "4096"],
    ["--kv-dynamic", "--num-tokens", "32768", "--kv-floor-tokens", "65536"],
    ["--kv-dynamic", "--moe-backend", "fused"],
    ["--kv-park-ttl-s", "-1"],
])
def test_bad_dynamic_flags_are_refused(tmp_path, extra):
    with pytest.raises(SystemExit):
        parse_args(_base(tmp_path) + extra)

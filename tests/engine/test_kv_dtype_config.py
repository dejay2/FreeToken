"""Startup KV dtype selection without changing the default BF16 boot."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from freetoken.server.args import parse_args

ANON_MODEL = "/models/anon"
LAUNCHER = (
    Path(__file__).parents[2] / "scripts" / "start-qwen38-flash-next-mmap-windows.ps1"
)


class _Config:
    def to_dict(self) -> dict:
        return {"architectures": ["Qwen2ForCausalLM"], "torch_dtype": "bfloat16"}


def _parse(*extra: str):
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        args, _run_shell = parse_args(["--model", ANON_MODEL, *extra])
    return args


def test_kv_dtype_defaults_to_bf16(monkeypatch):
    """An unset switch must retain the old two-byte QSA K/V storage path."""
    monkeypatch.delenv("FREETOKEN_KV_DTYPE", raising=False)
    assert _parse().kv_dtype == "bf16"


def test_kv_dtype_reads_the_environment(monkeypatch):
    monkeypatch.setenv("FREETOKEN_KV_DTYPE", "fp8")
    assert _parse().kv_dtype == "fp8"


def test_cli_kv_dtype_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("FREETOKEN_KV_DTYPE", "fp8")
    assert _parse("--kv-dtype", "bf16").kv_dtype == "bf16"


def test_kv_dtype_rejects_unknown_modes(monkeypatch):
    monkeypatch.delenv("FREETOKEN_KV_DTYPE", raising=False)
    with pytest.raises(SystemExit):
        _parse("--kv-dtype", "int8")


def test_kv_dtype_rejects_an_unknown_environment_value(monkeypatch):
    monkeypatch.setenv("FREETOKEN_KV_DTYPE", "int8")
    with pytest.raises(SystemExit):
        _parse()


def test_backend_cache_metadata_reports_the_selected_dtype_and_true_cost():
    from freetoken.kvcache.cache_status import compute_cache_status_meta

    config = SimpleNamespace(
        kv_dtype="fp8",
        page_size=64,
        memory_ratio=0.9,
        cache_type="radix",
        swa_full_tokens_ratio=0.2,
        model_config=SimpleNamespace(dsv4_args=None, has_swa_attention=False),
    )
    engine = SimpleNamespace(
        config=config,
        kv_cache=SimpleNamespace(unit_bytes=lambda: (13_248, 0)),
        moe_offload_cache=None,
        linear_state_pool=None,
        num_pages=100,
        _post_weights_free=0,
        _baseline_free=0,
        _weights_bytes=0,
    )

    meta = compute_cache_status_meta(engine)

    assert meta["kv_dtype"] == "fp8"
    assert meta["kv_bytes_per_token"] == 13_248


def test_cache_status_geometry_exposes_the_selected_kv_dtype():
    from freetoken.server.api_server import cache_geometry

    state = SimpleNamespace(
        stats=SimpleNamespace(kv_total_pages=0, mamba_total_slots=0),
        config=SimpleNamespace(
            kv_dtype="fp8",
            page_size=64,
            moe_cache_policy="lru",
            moe_cache_size=0,
            moe_cache_rate=None,
            model_config=SimpleNamespace(num_experts=0, num_moe_layers=0, dsv4_args=None),
        ),
        last_rebuild=None,
        cache_pools={"num_pages": 100, "page_size": 64},
        unit_bytes={"kv_bytes_per_token": 13_248},
        swa_full_tokens_ratio=0.0,
        cache_budget_bytes=0,
        free_vram_bytes=0,
        cache_floors={},
        _frontend_tokenizer=None,
        warm_frontend_tokenizer=lambda: None,
    )

    geometry = cache_geometry(state)

    assert geometry["kv_dtype"] == "fp8"
    assert geometry["unit_bytes"]["kv_per_token"] == 13_248


def test_launcher_exposes_fp8_without_changing_its_bf16_default():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "[ValidateSet('bf16', 'fp8')]" in launcher
    assert "[string]$KVDtype = 'bf16'" in launcher
    assert "if ($KVDtype -eq 'fp8')" in launcher
    assert "$serveArgs += @('--kv-dtype', 'fp8')" in launcher

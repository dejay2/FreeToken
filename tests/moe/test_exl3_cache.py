"""CPU-only EXL3 cache schema, sizing and provider boundary tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT, _BANK_SCHEMAS, OffloadMoeCache

_E, _H, _I = 2, 128, 256


def _sources(num_layers=2):
    return {
        "gate_trellis": [torch.zeros(_E, _H // 16, _I // 16, 32, dtype=torch.int16) for _ in range(num_layers)],
        "gate_suh": [torch.zeros(_E, _H, dtype=torch.float16) for _ in range(num_layers)],
        "gate_svh": [torch.zeros(_E, _I, dtype=torch.float16) for _ in range(num_layers)],
        "up_trellis": [torch.zeros(_E, _H // 16, _I // 16, 32, dtype=torch.int16) for _ in range(num_layers)],
        "up_suh": [torch.zeros(_E, _H, dtype=torch.float16) for _ in range(num_layers)],
        "up_svh": [torch.zeros(_E, _I, dtype=torch.float16) for _ in range(num_layers)],
        "down_trellis": [torch.zeros(_E, _I // 16, _H // 16, 32, dtype=torch.int16) for _ in range(num_layers)],
        "down_suh": [torch.zeros(_E, _I, dtype=torch.float16) for _ in range(num_layers)],
        "down_svh": [torch.zeros(_E, _H, dtype=torch.float16) for _ in range(num_layers)],
    }


def test_exl3_schema_and_aot_row_sizes_are_in_registration_order():
    from freetoken.kernel.aot_models import expert_bank_row_bytes

    expected_names = (
        "gate_trellis", "gate_suh", "gate_svh",
        "up_trellis", "up_suh", "up_svh",
        "down_trellis", "down_suh", "down_svh",
    )
    assert _BANK_SCHEMAS["exl3"] == expected_names
    rows = expert_bank_row_bytes("exl3", 4096, 2048)
    assert tuple(rows) == expected_names
    assert rows == {
        "gate_trellis": 2_097_152,
        "gate_suh": 8_192,
        "gate_svh": 4_096,
        "up_trellis": 2_097_152,
        "up_suh": 8_192,
        "up_svh": 4_096,
        "down_trellis": 2_097_152,
        "down_suh": 4_096,
        "down_svh": 8_192,
    }
    assert _BANK_BYTES_PER_EXPERT["exl3"](4096, 2048) == 6_328_320


def test_exl3_cache_accepts_all_nine_banks_and_reports_the_exact_row_bytes():
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=_E,
        cache_size=4,
        device=torch.device("cpu"),
        quant_format="exl3",
    )
    cache.set_bank_sources(_sources())

    assert cache.bank_schema == _BANK_SCHEMAS["exl3"]
    assert tuple(cache.bank_caches) == _BANK_SCHEMAS["exl3"]
    assert cache.bank_caches["gate_trellis"].shape == (4, 8, 16, 32)
    assert cache.bank_caches["down_trellis"].shape == (4, 16, 8, 32)
    expected = _BANK_BYTES_PER_EXPERT["exl3"](_H, _I)
    assert cache.bytes_per_expert_row() == expected
    assert expected == 3 * (8 * 16 * 32 * 2) + 6 * (_H + _I)
    assert all(size % 128 == 0 for size in (
        cache.bank_caches[name][0].numel() * cache.bank_caches[name][0].element_size()
        for name in _BANK_SCHEMAS["exl3"]
    ))


def test_exl3_provider_rejects_cpu_or_hybrid_before_opening_a_shard(monkeypatch):
    import freetoken.models.exl3_banks as exl3
    from freetoken.moe import expert_banks

    def should_not_read(*args, **kwargs):
        raise AssertionError("EXL3 provider opened a shard before rejecting decode_target")

    monkeypatch.setattr(exl3, "download_hf_weight", should_not_read)
    config = SimpleNamespace(expert_quant="exl3")
    for target in ("cpu", "hybrid"):
        with pytest.raises(ValueError, match="card-only"):
            expert_banks._exl3_banks(
                "not-a-model", config, torch.device("cpu"), torch.float16,
                False, parallel=False, decode_target=target,
            )


def test_forced_parallel_exl3_load_uses_the_established_serial_fallback(monkeypatch):
    from freetoken.moe import expert_banks

    calls = []

    def serial(model_path, model_config, **kwargs):
        calls.append((model_path, kwargs))
        return {name: [] for name in _BANK_SCHEMAS["exl3"]}

    import freetoken.models.exl3_banks as exl3
    monkeypatch.setattr(exl3, "load_exl3_expert_sources", serial)
    config = SimpleNamespace(expert_quant="exl3")
    with pytest.raises(NotImplementedError, match="serial-only"):
        expert_banks._exl3_banks(
            "fixture", config, torch.device("cpu"), torch.float16,
            False, parallel=True, decode_target="gpu",
        )
    assert calls == []

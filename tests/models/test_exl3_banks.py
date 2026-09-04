"""CPU-safe EXL3 bank schema and loader tests.

The fixture uses the GLM-5.3-Flash ``turboderp/GLM-5.3-Flash-exl3`` 2.05bpw
key layout with 128-wide dimensions, but stores only a few experts and layers.
The real checkpoint's routed experts are K=2/mul1; MTP layer 45 is included only
to prove that the main loader leaves it untouched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from freetoken.models.exl3_banks import (
    EXL3_BANK_NAMES,
    _EXL3_KEY_RE,
    _collect_records,
    load_exl3_expert_sources,
)

_E, _H, _I = 2, 128, 128
_FIRST, _NUM_LAYERS = 3, 2
_CONFIG = SimpleNamespace(
    num_layers=5,
    first_k_dense_replace=_FIRST,
    num_moe_layers=_NUM_LAYERS,
    num_experts=_E,
    hidden_size=_H,
    moe_intermediate_size=_I,
)
_PROJS = ("gate_proj", "up_proj", "down_proj")
_KINDS = ("trellis", "suh", "svh", "mul1")


def _shape(proj: str, kind: str, k: int = 2) -> tuple[int, ...]:
    if proj in ("gate_proj", "up_proj"):
        trellis = (_H // 16, _I // 16, 16 * k)
        factor = (_H,) if kind == "suh" else (_I,)
    else:
        trellis = (_I // 16, _H // 16, 16 * k)
        factor = (_I,) if kind == "suh" else (_H,)
    if kind == "trellis":
        return trellis
    if kind == "mul1":
        return ()
    return factor


def _tensor(layer: int, expert: int, proj: str, kind: str, *, k: int = 2) -> torch.Tensor:
    value = layer * 100 + expert * 10 + _PROJS.index(proj)
    if kind == "trellis":
        return torch.full(_shape(proj, kind, k), value, dtype=torch.int16)
    if kind in ("suh", "svh"):
        return torch.full(_shape(proj, kind), value / 10, dtype=torch.float16)
    return torch.tensor(value, dtype=torch.int32)


def _write_checkpoint(tmp_path, mutate=None, *, include_mtp=True) -> str:
    weight_map: dict[str, str] = {}
    for layer in range(_FIRST, _FIRST + _NUM_LAYERS + (1 if include_mtp else 0)):
        shard = f"model-{layer:05d}.safetensors"
        tensors = {}
        for expert in range(_E):
            for proj in _PROJS:
                for kind in _KINDS:
                    name = (
                        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                        f"{proj}.{kind}"
                    )
                    value = _tensor(layer, expert, proj, kind)
                    if mutate is not None:
                        name, value = mutate(name, value)
                    if value is None:
                        continue
                    tensors[name] = value
                    weight_map[name] = shard
        safetensors.torch.save_file(tensors, str(tmp_path / shard))

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    return str(tmp_path)


def _load(tmp_path, monkeypatch, mutate=None, *, include_mtp=True, sink=None):
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    return load_exl3_expert_sources(
        _write_checkpoint(tmp_path, mutate, include_mtp=include_mtp),
        _CONFIG,
        drop_page_cache=lambda path: None,
        primary=False,
        layer_sink=sink,
    )


def test_loader_preserves_the_nine_bank_order_shapes_and_bytes(tmp_path, monkeypatch):
    banks = _load(tmp_path, monkeypatch)

    assert tuple(banks) == EXL3_BANK_NAMES
    assert all(len(per_layer) == _NUM_LAYERS for per_layer in banks.values())
    assert banks["gate_trellis"][0].shape == (_E, 8, 8, 32)
    assert banks["gate_suh"][0].shape == (_E, _H)
    assert banks["gate_svh"][0].shape == (_E, _I)
    assert banks["down_trellis"][0].shape == (_E, 8, 8, 32)
    assert banks["down_suh"][0].shape == (_E, _I)
    assert banks["down_svh"][0].shape == (_E, _H)
    assert banks["gate_trellis"][0].dtype is torch.int16
    assert banks["gate_suh"][0].dtype is torch.float16
    assert banks["gate_trellis"][0].is_contiguous()

    expected_row_bytes = 3 * (8 * 8 * 32 * 2) + 6 * (_H + _I)
    measured_row_bytes = sum(
        banks[name][0][0].numel() * banks[name][0][0].element_size()
        for name in EXL3_BANK_NAMES
    )
    assert measured_row_bytes == expected_row_bytes == 13_824

    # Check the layer mapping and projection orientation, not only square shapes.
    assert torch.equal(
        banks["gate_trellis"][0][0], _tensor(3, 0, "gate_proj", "trellis")
    )
    assert torch.equal(
        banks["up_suh"][1][1], _tensor(4, 1, "up_proj", "suh")
    )
    assert torch.equal(
        banks["down_svh"][0][1], _tensor(3, 1, "down_proj", "svh")
    )


def test_loader_reads_every_shard_per_tensor_and_fires_each_layer_once(
    tmp_path, monkeypatch
):
    import freetoken.models.exl3_banks as exl3

    calls = []
    original = exl3._open_shard

    def wrapped(path, *, whole=False):
        calls.append((path, whole))
        return original(path, whole=whole)

    monkeypatch.setattr(exl3, "_open_shard", wrapped)
    seen = []

    def sink(layer_id, banks):
        seen.append((layer_id, tuple(banks)))

    _load(tmp_path, monkeypatch, sink=sink)

    assert calls and all(not whole for _path, whole in calls)
    assert [layer for layer, _banks in seen] == [0, 1]
    assert all(names == EXL3_BANK_NAMES for _layer, names in seen)


def test_layer_45_mtp_experts_are_excluded(tmp_path, monkeypatch):
    banks = _load(tmp_path, monkeypatch, include_mtp=True)
    # The MTP values use layer=5 and would be 500-series if they were copied into bank 0.
    assert banks["gate_suh"][0][0, 0].item() == pytest.approx(30.0, abs=0.1)
    assert banks["gate_suh"][1][0, 0].item() == pytest.approx(40.0, abs=0.1)


def test_detector_accepts_the_published_exl3_method():
    from freetoken.models.config import detect_expert_quant

    assert detect_expert_quant(
        SimpleNamespace(quantization_config={"quant_method": "exl3"})
    ) == "exl3"


@pytest.mark.parametrize(
    ("label", "mutate", "match"),
    [
        (
            "missing marker",
            lambda name, value: (name, None) if name.endswith(".mul1") and ".0." in name else (name, value),
            "missing.*mul1",
        ),
        (
            "mcg marker",
            lambda name, value: (name.replace(".mul1", ".mcg"), value)
            if name.endswith(".mul1") and ".0." in name
            else (name, value),
            "mcg codebook",
        ),
        (
            "K three",
            lambda name, value: (name, _tensor(3, 0, "gate_proj", "trellis", k=3))
            if name.endswith("layers.3.mlp.experts.0.gate_proj.trellis")
            else (name, value),
            r"K=3",
        ),
        (
            "wrong dtype",
            lambda name, value: (name, value.to(torch.float32))
            if name.endswith("layers.3.mlp.experts.0.gate_proj.suh")
            else (name, value),
            "wrong EXL3 dtype",
        ),
        (
            "wrong shape",
            lambda name, value: (name, torch.zeros((_H + 1,), dtype=torch.float16))
            if name.endswith("layers.3.mlp.experts.0.gate_proj.suh")
            else (name, value),
            "wrong EXL3 shape",
        ),
    ],
)
def test_loader_rejects_invalid_component_metadata(
    tmp_path, monkeypatch, label, mutate, match
):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, monkeypatch, mutate, include_mtp=False)


def test_record_scan_rejects_duplicate_component_completion():
    first = "model.language_model.layers.3.mlp.experts.0.gate_proj.trellis"
    shard = "one.safetensors"

    class DuplicateMap(dict):
        def items(self):
            return iter(((first, shard), (first, shard)))

    with pytest.raises(ValueError, match="duplicate EXL3 component"):
        _collect_records(DuplicateMap(), _CONFIG)


def test_key_pattern_keeps_the_glm_style_components_explicit():
    match = _EXL3_KEY_RE.fullmatch(
        "model.language_model.layers.3.mlp.experts.0.down_proj.svh"
    )
    assert match
    assert match.groupdict() == {
        "layer": "3",
        "expert": "0",
        "proj": "down_proj",
        "kind": "svh",
    }

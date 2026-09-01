"""The draft head's opt-in quantized expert placement.

``FREETOKEN_MTP_SPEC_EXPERT_FORMAT`` selects which banks the integrated draft head puts on
the card. It is parsed HERE, in ``spec_draft`` -- ``EngineConfig`` owns the speculation flags
and must not learn about a private bank placement. The default stays the exact bf16 banks
loaded straight from the target checkpoint, byte for byte the placement that shipped.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from freetoken.engine.spec_draft import (
    load_spec_expert_banks,
    resolve_spec_expert_placement,
    spec_expert_runner_type,
)
from freetoken.models.qwen4_exp.mtp_spike import (
    MTPBF16ExpertBanks,
    MTPGPUExpertRunner,
    MTPNVFP4ExpertBanks,
    MTPNVFP4GPUExpertRunner,
    quantize_nvfp4_rows,
)

EXPERTS, HIDDEN, INTERMEDIATE = 4, 64, 32


# ------------------------------------------------------------------------- env parsing


def test_the_default_placement_is_the_exact_bf16_banks():
    assert resolve_spec_expert_placement({}) == ("bf16", None)
    assert resolve_spec_expert_placement(
        {"FREETOKEN_MTP_SPEC_EXPERT_FORMAT": ""}
    ) == ("bf16", None)
    assert resolve_spec_expert_placement(
        {"FREETOKEN_MTP_SPEC_EXPERT_FORMAT": " BF16 "}
    ) == ("bf16", None)


def test_the_default_placement_reads_the_process_environment(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MTP_SPEC_EXPERT_FORMAT", raising=False)
    monkeypatch.delenv("FREETOKEN_MTP_SPEC_NVFP4_MANIFEST", raising=False)
    assert resolve_spec_expert_placement() == ("bf16", None)


@pytest.mark.parametrize("raw", ["fp8", "nvfp4 banks", "1", "int4"])
def test_an_unknown_placement_is_rejected_by_name(raw):
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_EXPERT_FORMAT"):
        resolve_spec_expert_placement({"FREETOKEN_MTP_SPEC_EXPERT_FORMAT": raw})


def test_the_quantized_placement_requires_a_manifest():
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_NVFP4_MANIFEST"):
        resolve_spec_expert_placement({"FREETOKEN_MTP_SPEC_EXPERT_FORMAT": "nvfp4"})
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_NVFP4_MANIFEST"):
        resolve_spec_expert_placement(
            {
                "FREETOKEN_MTP_SPEC_EXPERT_FORMAT": "nvfp4",
                "FREETOKEN_MTP_SPEC_NVFP4_MANIFEST": "   ",
            }
        )


def test_a_missing_manifest_file_is_named_in_the_error(tmp_path):
    missing = tmp_path / "nope" / "manifest.json"
    with pytest.raises(ValueError, match="manifest"):
        resolve_spec_expert_placement(
            {
                "FREETOKEN_MTP_SPEC_EXPERT_FORMAT": "nvfp4",
                "FREETOKEN_MTP_SPEC_NVFP4_MANIFEST": str(missing),
            }
        )
    directory = tmp_path / "banks"
    directory.mkdir()
    with pytest.raises(ValueError, match="manifest"):
        resolve_spec_expert_placement(
            {
                "FREETOKEN_MTP_SPEC_EXPERT_FORMAT": "nvfp4",
                "FREETOKEN_MTP_SPEC_NVFP4_MANIFEST": str(directory),
            }
        )


def test_a_present_manifest_resolves_to_an_absolute_path(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    placement, path = resolve_spec_expert_placement(
        {
            "FREETOKEN_MTP_SPEC_EXPERT_FORMAT": "NVFP4",
            "FREETOKEN_MTP_SPEC_NVFP4_MANIFEST": str(manifest),
        }
    )
    assert placement == "nvfp4"
    assert path == manifest.resolve() and path.is_absolute()


# ---------------------------------------------------------------------- runner selection


def test_the_default_placement_selects_todays_resident_bf16_runner():
    assert spec_expert_runner_type("bf16") is MTPGPUExpertRunner
    assert spec_expert_runner_type("nvfp4") is MTPNVFP4GPUExpertRunner
    with pytest.raises(ValueError, match="bf16 or nvfp4"):
        spec_expert_runner_type("fp8")


class _FakeStore:
    """Just the one call ``MTPBF16ExpertBanks.from_store`` makes."""

    def __init__(self):
        self.asked: list[str] = []

    def tensor(self, name: str) -> torch.Tensor:
        self.asked.append(name)
        if name.endswith("gate_up_proj"):
            return torch.zeros(EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
        return torch.zeros(EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)


def test_the_default_placement_loads_the_checkpoints_exact_banks():
    store = _FakeStore()
    banks = load_spec_expert_banks("bf16", None, store)
    assert isinstance(banks, MTPBF16ExpertBanks)
    assert store.asked == [
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj",
    ]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(root: Path) -> Path:
    generator = torch.Generator().manual_seed(3)

    def quantize(tensor):
        shape = tensor.shape
        packed, scale, glob = quantize_nvfp4_rows(tensor.reshape(-1, shape[-1]))
        return {
            "packed": packed.view(*shape[:-1], shape[-1] // 2).contiguous(),
            "scale": scale.view(*shape[:-1], shape[-1] // 16).contiguous(),
            "global": glob.view(*shape[:-1]).contiguous(),
        }

    gate = quantize(torch.randn(EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator))
    down = quantize(torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, generator=generator))
    tensors = {
        "gate_up_packed": gate["packed"],
        "gate_up_scale": gate["scale"],
        "gate_up_global": gate["global"],
        "down_packed": down["packed"],
        "down_scale": down["scale"],
        "down_global": down["global"],
    }
    banks = {}
    for name, tensor in tensors.items():
        path = root / f"{name}.bin"
        path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        banks[name] = {
            "file": path.name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "nbytes": path.stat().st_size,
            "sha256": _sha(path),
        }
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "format": "nvfp4", "banks": banks}),
        encoding="utf-8",
    )
    return manifest


def test_the_quantized_placement_loads_and_hash_validates_the_manifest(tmp_path):
    manifest = _write_manifest(tmp_path)

    banks = load_spec_expert_banks("nvfp4", manifest, None)

    assert isinstance(banks, MTPNVFP4ExpertBanks)
    assert banks.num_experts == EXPERTS
    assert banks.hidden_size == HIDDEN
    assert banks.intermediate_size == INTERMEDIATE
    assert banks.total_bytes < EXPERTS * 3 * HIDDEN * INTERMEDIATE * 2  # a quarter of bf16


def test_a_corrupt_quantized_manifest_fails_loudly(tmp_path):
    manifest = _write_manifest(tmp_path)
    (tmp_path / "down_packed.bin").write_bytes(b"broken")

    with pytest.raises(ValueError, match="size|hash"):
        load_spec_expert_banks("nvfp4", manifest, None)


def test_the_quantized_placement_runner_is_wired_to_the_manifest_banks(tmp_path):
    banks = load_spec_expert_banks("nvfp4", _write_manifest(tmp_path), None)
    runner_type = spec_expert_runner_type("nvfp4")

    runner = runner_type(
        banks,
        top_k=2,
        activation="silu",
        renormalize=True,
        max_tokens=8,
        num_threads=1,
        device=torch.device("cpu"),
        max_gather_tokens=4,
    )

    assert runner.resident_bytes == banks.total_bytes + runner.scratch_bytes
    assert runner.bank_bytes == banks.total_bytes
    hidden = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
    ids = torch.zeros(1, 2, dtype=torch.int32)
    assert runner.run_routed(hidden, torch.ones(1, 2), ids).shape == (1, HIDDEN)

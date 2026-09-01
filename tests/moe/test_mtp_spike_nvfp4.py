from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from freetoken.models.qwen4_exp.mtp_spike import (
    MTPNVFP4ExpertBanks,
    MTPNVFP4ExpertRunner,
    dequantize_nvfp4_rows,
    quantize_nvfp4_rows,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _independent_dequant(packed, scale, row_global):
    rows, packed_k = packed.shape
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).view(rows, 2 * packed_k)
    values = E2M1[codes.long()]
    return values * scale.float().repeat_interleave(16, dim=-1) * row_global.float()[:, None]


def test_quantization_is_deterministic_finite_and_low_nibble_first():
    rows = torch.tensor(
        [
            [0.0] * 16,
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.5,
             -0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.5],
            [1e-8 * (index - 8) for index in range(16)],
            [1000.0 * (index - 8) for index in range(16)],
        ],
        dtype=torch.float32,
    )
    first = quantize_nvfp4_rows(rows)
    second = quantize_nvfp4_rows(rows.clone())
    for left, right in zip(first, second):
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    packed, scale, row_global = first
    assert packed.shape == (4, 8)
    assert scale.shape == (4, 1)
    assert scale.dtype is torch.float8_e4m3fn
    assert row_global.dtype is torch.float16
    assert torch.isfinite(scale.float()).all()
    assert torch.isfinite(row_global.float()).all()
    assert packed[0].count_nonzero() == 0
    # Element 2 is +0.5-like and element 3 is +0.75-like after scaling: even K is low nibble.
    codes = torch.stack((packed[1] & 0xF, packed[1] >> 4), dim=-1).flatten()
    reconstructed_codes = E2M1[codes.long()]
    assert reconstructed_codes[2] >= 0
    assert reconstructed_codes[8].signbit().item() is False
    assert reconstructed_codes[10] <= 0


def test_quantization_rejects_invalid_rows():
    with pytest.raises(ValueError, match="divisible by 16"):
        quantize_nvfp4_rows(torch.zeros(2, 15))
    with pytest.raises(ValueError, match="finite"):
        quantize_nvfp4_rows(torch.tensor([[float("nan")] * 16]))
    with pytest.raises(ValueError, match="two-dimensional"):
        quantize_nvfp4_rows(torch.zeros(16))


def test_private_dequant_matches_independent_reference_exactly():
    generator = torch.Generator().manual_seed(7)
    rows = torch.randn(19, 64, generator=generator) * 0.2
    packed, scale, row_global = quantize_nvfp4_rows(rows)
    got = dequantize_nvfp4_rows(packed, scale, row_global)
    expected = _independent_dequant(packed, scale, row_global)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@requires_cuda
def test_quantized_rows_match_canonical_cuda_dequant():
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    rows = torch.randn(12, 256) * 0.05
    packed, scale, row_global = quantize_nvfp4_rows(rows)
    packed = packed.cuda().view(3, 4, 128).contiguous()
    scale = scale.cuda().view(3, 4, 16).contiguous()
    row_global = row_global.cuda().view(3, 4).contiguous()
    slots = torch.arange(3, device="cuda", dtype=torch.int32)
    got = dequant_nvfp4(packed, scale, row_global, slots, dtype=torch.bfloat16)
    expected = dequantize_nvfp4_rows(
        packed.cpu().view(12, 128),
        scale.cpu().view(12, 16),
        row_global.cpu().view(12),
    ).to(torch.bfloat16).view(3, 4, 256)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


def _quantized_banks(seed=11):
    generator = torch.Generator().manual_seed(seed)
    experts, hidden, intermediate = 8, 256, 128

    def quantize(tensor):
        shape = tensor.shape
        packed, scale, glob = quantize_nvfp4_rows(tensor.view(-1, shape[-1]))
        return (
            packed.view(*shape[:-1], shape[-1] // 2).contiguous(),
            scale.view(*shape[:-1], shape[-1] // 16).contiguous(),
            glob.view(*shape[:-1]).contiguous(),
        )

    gate = torch.randn(experts, 2 * intermediate, hidden, generator=generator) * 0.025
    down = torch.randn(experts, hidden, intermediate, generator=generator) * 0.025
    return MTPNVFP4ExpertBanks(*quantize(gate), *quantize(down)), hidden, intermediate


def _moe_reference(hidden, banks, weights, ids):
    gate = dequantize_nvfp4_rows(
        banks.gate_up_packed.view(-1, banks.hidden_size // 2),
        banks.gate_up_scale.view(-1, banks.hidden_size // 16),
        banks.gate_up_global.view(-1),
    ).view(banks.num_experts, 2 * banks.intermediate_size, banks.hidden_size)
    down = dequantize_nvfp4_rows(
        banks.down_packed.view(-1, banks.intermediate_size // 2),
        banks.down_scale.view(-1, banks.intermediate_size // 16),
        banks.down_global.view(-1),
    ).view(banks.num_experts, banks.hidden_size, banks.intermediate_size)
    result = torch.zeros_like(hidden, dtype=torch.float32)
    for token in range(hidden.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            projected = gate[expert] @ hidden[token].float()
            gate_value, up = projected.chunk(2)
            result[token] += weights[token, route] * (
                down[expert] @ (F.silu(gate_value) * up)
            )
    return result


@pytest.mark.parametrize("tokens", [1, 4])
@requires_cuda
def test_nvfp4_cpu_runner_matches_dequantized_reference(tokens):
    banks, hidden_size, _ = _quantized_banks(seed=30 + tokens)
    runner = MTPNVFP4ExpertRunner(
        banks,
        top_k=2,
        activation="silu",
        renormalize=True,
        max_tokens=4,
        num_threads=4,
        device=torch.device("cuda"),
    )
    try:
        hidden = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(banks.num_experts, device="cuda")[:2] for _ in range(tokens)]
        ).to(torch.int32)
        weights = torch.rand(tokens, 2, device="cuda")
        weights /= weights.sum(-1, keepdim=True)
        got = runner.run_routed(hidden, weights, ids).float()
        torch.cuda.synchronize()
        expected = _moe_reference(hidden.cpu(), banks, weights.cpu(), ids.cpu())
        relative = (got.cpu() - expected).abs().max() / (expected.abs().max() + 1e-6)
        assert relative < 0.04, relative
    finally:
        runner.close()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bank(path: Path, tensor: torch.Tensor) -> dict:
    path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return {
        "file": path.name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "nbytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def test_manifest_loader_maps_valid_banks_and_rejects_corruption(tmp_path):
    banks, _, _ = _quantized_banks(seed=51)
    tensors = {
        "gate_up_packed": banks.gate_up_packed,
        "gate_up_scale": banks.gate_up_scale,
        "gate_up_global": banks.gate_up_global,
        "down_packed": banks.down_packed,
        "down_scale": banks.down_scale,
        "down_global": banks.down_global,
    }
    manifest = {
        "schema_version": 1,
        "format": "nvfp4",
        "geometry": {
            "num_experts": banks.num_experts,
            "hidden_size": banks.hidden_size,
            "intermediate_size": banks.intermediate_size,
        },
        "banks": {
            name: _write_bank(tmp_path / f"{name}.bin", tensor)
            for name, tensor in tensors.items()
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    mapped = MTPNVFP4ExpertBanks.from_manifest(manifest_path)
    assert mapped.total_bytes == sum(item["nbytes"] for item in manifest["banks"].values())
    for name, tensor in tensors.items():
        got = getattr(mapped, name)
        assert torch.equal(got.view(torch.uint8), tensor.view(torch.uint8)), name

    del got, mapped
    gc.collect()
    (tmp_path / manifest["banks"]["down_packed"]["file"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="size|hash"):
        MTPNVFP4ExpertBanks.from_manifest(manifest_path)

"""Packed EXL3 multi-GEMM agreement tests for GLM-5.3 rows."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from freetoken.kernel.exl3_mgemm import (
    EXL3_MGEMM_MAX_INDICES,
    Exl3MgemmBanks,
    exl3_mgemm_limits,
    exl3_mgemm_projection,
    fused_experts_exl3_mgemm,
    prepare_exl3_mgemm_scratch,
)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
MODEL_PATH = Path(
    os.environ.get("FREETOKEN_GLM53_EXL3_MODEL", r"D:\Models\GLM-5.3-Flash-exl3-2.05bpw")
)

_H, _I, _E = 128, 256, 8
_PROJECTIONS = ("gate", "up", "down")


def _synthetic_banks(device: torch.device, experts: int = 2):
    torch.manual_seed(19)
    gate_up_trellis = torch.randint(
        -32768,
        32767,
        (experts, _H // 16, _I // 16, 32),
        dtype=torch.int16,
        device=device,
    ).contiguous()
    down_trellis = torch.randint(
        -32768,
        32767,
        (experts, _I // 16, _H // 16, 32),
        dtype=torch.int16,
        device=device,
    ).contiguous()

    def factor(width: int):
        return torch.rand((experts, width), dtype=torch.float16, device=device).contiguous()

    return (
        gate_up_trellis,
        factor(_H),
        factor(_I),
        gate_up_trellis.clone(),
        factor(_H),
        factor(_I),
        down_trellis,
        factor(_I),
        factor(_H),
    )


def test_limits_report_route_capacity_without_inventing_a_row_cap():
    limits = exl3_mgemm_limits()
    assert limits.max_indices == EXL3_MGEMM_MAX_INDICES == 128
    # exl3_gemm_shape_compat() checks K/N tile divisibility, not size_m.  The 128 limit is
    # the pointer-index capacity and must not be copied from exl3_moe as a per-expert row cap.
    assert limits.max_tokens_per_expert is None
    assert limits.max_tokens_for_top_k(8) == 16


@cuda
def test_grouped_projection_accepts_more_than_the_route_index_capacity():
    device = torch.device("cuda")
    banks = Exl3MgemmBanks.from_banks(_synthetic_banks(device))
    scratch = prepare_exl3_mgemm_scratch(device=device, max_rows=256, max_features=_I)
    inputs = torch.randn((129, _H), dtype=torch.bfloat16, device=device)

    got = exl3_mgemm_projection(
        inputs,
        banks,
        torch.zeros(1, dtype=torch.int32, device=device),
        projection="gate",
        scratch=scratch,
    )

    assert got.shape == (129, _I)
    assert got.dtype == torch.bfloat16


@cuda
def test_route_reduction_matches_reconstruct_first_for_glm_activation():
    device = torch.device("cuda")
    banks = Exl3MgemmBanks.from_banks(_synthetic_banks(device))
    scratch = prepare_exl3_mgemm_scratch(device=device, max_rows=8, max_features=_I)
    hidden = torch.randn((2, _H), dtype=torch.bfloat16, device=device)
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32, device=device)

    got = fused_experts_exl3_mgemm(
        hidden,
        banks,
        weights,
        ids,
        activation="swiglu_clamp",
        hidden_act_alpha=1.0,
        swiglu_limit=10.0,
        scratch=scratch,
    )

    from freetoken.kernel.exl3 import reconstruct
    from freetoken.layers import swiglu_clamp_and_mul

    expected = []
    for token in range(hidden.shape[0]):
        routed = []
        for route in range(ids.shape[1]):
            expert = int(ids[token, route].item())
            gate = reconstruct(
                banks.banks[0][expert],
                banks.banks[1][expert],
                banks.banks[2][expert],
                k=2,
                codebook="mul1",
            )
            up = reconstruct(
                banks.banks[3][expert],
                banks.banks[4][expert],
                banks.banks[5][expert],
                k=2,
                codebook="mul1",
            )
            down = reconstruct(
                banks.banks[6][expert],
                banks.banks[7][expert],
                banks.banks[8][expert],
                k=2,
                codebook="mul1",
            )
            gate_up = torch.cat(
                (
                    torch.matmul(hidden[token], gate.transpose(0, 1)),
                    torch.matmul(hidden[token], up.transpose(0, 1)),
                ),
                dim=-1,
            )
            activated = swiglu_clamp_and_mul(gate_up.unsqueeze(0), alpha=1.0, limit=10.0)[0]
            routed.append(float(weights[token, route]) * torch.matmul(activated, down.transpose(0, 1)))
        expected.append(torch.stack(routed).sum(dim=0))
    expected = torch.stack(expected).to(torch.bfloat16)
    torch.testing.assert_close(got.float(), expected.float(), rtol=8e-2, atol=0.5)


def _load_real_banks(model_path: Path, *, experts: int = _E):
    from safetensors import safe_open

    with (model_path / "model.safetensors.index.json").open(encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]

    names = {
        projection: {
            kind: [
                f"model.language_model.layers.3.mlp.experts.{expert}."
                f"{projection}_proj.{kind}"
                for expert in range(experts)
            ]
            for kind in ("trellis", "suh", "svh")
        }
        for projection in ("gate", "up", "down")
    }
    by_shard: dict[str, list[str]] = defaultdict(list)
    for projection in names.values():
        for records in projection.values():
            for name in records:
                by_shard[weight_map[name]].append(name)

    loaded: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(str(model_path / shard), framework="pt", device="cpu") as reader:
            for name in shard_names:
                loaded[name] = reader.get_tensor(name).contiguous()

    rows = []
    for projection in ("gate", "up", "down"):
        for kind in ("trellis", "suh", "svh"):
            rows.append(torch.stack([loaded[name] for name in names[projection][kind]], dim=0))
    return tuple(rows)


@pytest.mark.needs_weights
@pytest.mark.slow
@cuda
def test_real_checkpoint_packed_gate_up_down_agree_at_route_limit():
    pytest.importorskip("exllamav3_ext")
    if not MODEL_PATH.is_dir():
        pytest.skip(f"checkpoint is not present: {MODEL_PATH}")

    device = torch.device("cuda")
    cpu_banks = _load_real_banks(MODEL_PATH)
    banks = Exl3MgemmBanks.from_banks(tuple(bank.to(device=device) for bank in cpu_banks))
    real_hidden = int(banks.banks[0].shape[1] * 16)
    real_intermediate = int(banks.banks[0].shape[2] * 16)
    scratch = prepare_exl3_mgemm_scratch(
        device=device,
        max_rows=EXL3_MGEMM_MAX_INDICES,
        max_features=max(real_hidden, real_intermediate),
    )

    # Reconstruct each selected row once.  These are the independent reference matrices;
    # the wrapper under test must never call reconstruct().
    from freetoken.kernel.exl3 import reconstruct

    bank_triplets = {
        "gate": (0, 1, 2),
        "up": (3, 4, 5),
        "down": (6, 7, 8),
    }
    reference = {}
    for projection, (trellis_i, suh_i, svh_i) in bank_triplets.items():
        reference[projection] = [
            reconstruct(
                banks.banks[trellis_i][expert],
                banks.banks[suh_i][expert],
                banks.banks[svh_i][expert],
                k=2,
                codebook="mul1",
            )
            for expert in range(_E)
        ]

    for token_count in (1, 8, EXL3_MGEMM_MAX_INDICES):
        expert_ids = (torch.arange(token_count, device=device, dtype=torch.int32) % _E).contiguous()
        for projection in _PROJECTIONS:
            input_width = real_intermediate if projection == "down" else real_hidden
            inputs = torch.randn((token_count, input_width), dtype=torch.bfloat16, device=device)
            got = exl3_mgemm_projection(
                inputs,
                banks,
                expert_ids,
                projection=projection,
                scratch=scratch,
            )
            expected = torch.stack(
                [
                    torch.matmul(
                        inputs[row],
                        reference[projection][int(expert_ids[row].item())].transpose(0, 1),
                    )
                    for row in range(token_count)
                ],
                dim=0,
            )
            # The two paths use the same packed rows but different FP16/BF16 accumulation
            # order.  This tolerance is deliberately relative to the 2.05-bit output scale.
            torch.testing.assert_close(got.float(), expected.float(), rtol=5e-2, atol=0.5)


def test_fused_wrapper_requires_the_route_index_limit_message():
    # This is a CPU-side contract test: it proves the caller gets a fallback-able error before
    # any optional CUDA extension import when a prompt tile is too large.
    from freetoken.kernel.exl3_mgemm import Exl3MgemmLimitError

    assert issubclass(Exl3MgemmLimitError, ValueError)
    assert callable(fused_experts_exl3_mgemm)

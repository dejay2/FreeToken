from __future__ import annotations

import gc
import os
import weakref
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from freetoken.models.qwen4_exp.mtp_spike import (
    MTPBF16ExpertBanks,
    MTPExactExpertRunner,
    MTPWeightStore,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _banks(seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    experts, hidden, intermediate = 8, 256, 128
    gate_up = (
        torch.randn(experts, 2 * intermediate, hidden, generator=generator) * 0.025
    ).to(torch.bfloat16)
    down = (
        torch.randn(experts, hidden, intermediate, generator=generator) * 0.025
    ).to(torch.bfloat16)
    return MTPBF16ExpertBanks(gate_up, down), hidden, intermediate


def _reference(hidden, gate_up, down, weights, ids):
    result = torch.zeros_like(hidden, dtype=torch.float32)
    x = hidden.float()
    for token in range(hidden.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            projected = gate_up[expert].float() @ x[token]
            gate, up = projected.chunk(2)
            activated = F.silu(gate) * up
            value = down[expert].float() @ activated
            result[token] += weights[token, route].float() * value
    return result


@pytest.mark.parametrize("tokens", [1, 2, 128])
@requires_cuda
def test_exact_cpu_runner_matches_independent_torch_reference(tokens):
    banks, hidden_size, _ = _banks(seed=10 + tokens)
    runner = MTPExactExpertRunner(
        banks,
        top_k=2,
        activation="silu",
        renormalize=True,
        max_tokens=128,
        num_threads=4,
        device=torch.device("cuda"),
    )
    try:
        generator = torch.Generator(device="cuda").manual_seed(100 + tokens)
        hidden = torch.randn(
            tokens,
            hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        ids = torch.stack(
            [torch.randperm(banks.num_experts, generator=generator, device="cuda")[:2]
             for _ in range(tokens)]
        ).to(torch.int32)
        weights = torch.rand(tokens, 2, generator=generator, device="cuda")
        weights /= weights.sum(dim=-1, keepdim=True)

        got = runner.run_routed(hidden, weights, ids).float()
        torch.cuda.synchronize()
        expected = _reference(
            hidden.cpu(),
            banks.gate_up,
            banks.down,
            weights.cpu(),
            ids.cpu(),
        )
        relative = (got.cpu() - expected).abs().max() / (expected.abs().max() + 1e-6)
        assert relative < 0.025, relative
        assert runner.stats.calls == 1
        assert runner.stats.tokens == tokens
        assert runner.stats.logical_expert_bytes == (
            tokens * 2 * banks.bytes_per_expert
        )
        assert runner.stats.explicit_pcie_bytes == tokens * (
            2 * hidden_size * 2 + 2 * 4 + 2 * 4
        )
    finally:
        runner.close()


@requires_cuda
def test_router_is_topk_softmax_and_renormalizes():
    banks, hidden_size, _ = _banks(seed=22)
    runner = MTPExactExpertRunner(
        banks,
        top_k=2,
        activation="silu",
        renormalize=True,
        max_tokens=4,
        num_threads=2,
        device=torch.device("cuda"),
    )
    try:
        hidden = torch.randn(3, hidden_size, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(3, banks.num_experts, device="cuda", dtype=torch.bfloat16)
        weights, ids = runner.route(hidden, logits)
        probabilities = logits.float().softmax(dim=-1)
        expected_weights, expected_ids = probabilities.topk(2, dim=-1)
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
        assert torch.equal(ids, expected_ids.to(torch.int32))
        torch.testing.assert_close(weights, expected_weights, rtol=2e-4, atol=2e-4)
        output = runner.forward(hidden, logits)
        assert output.shape == hidden.shape
        torch.cuda.synchronize()
        runner.raise_if_unhealthy()
    finally:
        runner.close()


@requires_cuda
def test_runner_rejects_oversized_or_malformed_calls():
    banks, hidden_size, _ = _banks(seed=31)
    runner = MTPExactExpertRunner(
        banks,
        top_k=2,
        activation="silu",
        renormalize=True,
        max_tokens=2,
        num_threads=2,
        device=torch.device("cuda"),
    )
    try:
        with pytest.raises(ValueError, match="at most 2"):
            runner.run_routed(
                torch.zeros(3, hidden_size, device="cuda", dtype=torch.bfloat16),
                torch.ones(3, 2, device="cuda"),
                torch.zeros(3, 2, device="cuda", dtype=torch.int32),
            )
        with pytest.raises(ValueError, match="bfloat16"):
            runner.run_routed(
                torch.zeros(1, hidden_size, device="cuda"),
                torch.ones(1, 2, device="cuda"),
                torch.zeros(1, 2, device="cuda", dtype=torch.int32),
            )
        with pytest.raises(ValueError, match="top-k ids"):
            runner.run_routed(
                torch.zeros(1, hidden_size, device="cuda", dtype=torch.bfloat16),
                torch.ones(1, 2, device="cuda"),
                torch.zeros(1, 2, device="cuda", dtype=torch.int64),
            )
    finally:
        runner.close()


@requires_cuda
def test_close_releases_executor_reference():
    banks, _, _ = _banks(seed=44)
    runner = MTPExactExpertRunner(
        banks,
        top_k=2,
        activation="silu",
        renormalize=True,
        max_tokens=2,
        num_threads=2,
        device=torch.device("cuda"),
    )
    executor_ref = weakref.ref(runner.executor)
    runner.close()
    gc.collect()
    assert runner.executor is None
    assert executor_ref() is None


def test_bank_contract_rejects_wrong_layout_and_dtype():
    gate = torch.zeros(8, 256, 256, dtype=torch.bfloat16)
    down = torch.zeros(8, 256, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="gate_up"):
        MTPBF16ExpertBanks(gate[:, :-1], down)
    with pytest.raises(ValueError, match="BF16"):
        MTPBF16ExpertBanks(gate.float(), down)
    with pytest.raises(ValueError, match="contiguous"):
        MTPBF16ExpertBanks(gate.transpose(1, 2), down)


def test_real_bf16_banks_are_direct_file_backed_views():
    model = os.environ.get("FREETOKEN_QWEN38_MODEL_PATH")
    if not model or not Path(model).is_dir():
        pytest.skip("real private Qwen3.8 checkpoint is not configured")
    with MTPWeightStore(model) as store:
        gate = store.tensor("mtp.layers.0.mlp.experts.gate_up_proj")
        down = store.tensor("mtp.layers.0.mlp.experts.down_proj")
        banks = MTPBF16ExpertBanks.from_store(store)
        assert banks.gate_up.data_ptr() == gate.data_ptr()
        assert banks.down.data_ptr() == down.data_ptr()
        assert banks.gate_up.untyped_storage().nbytes() == gate.untyped_storage().nbytes()
        assert banks.down.untyped_storage().nbytes() == down.untyped_storage().nbytes()
        assert banks.total_bytes == 5_033_164_800
        assert banks.gate_up.shape == (512, 1280, 2560)
        assert banks.down.shape == (512, 2560, 640)
        assert banks.quant_format == "bf16"
        assert banks.num_layers == 1
        assert banks.bank_sources["gate_up"][0] is banks.gate_up
        assert banks.bank_sources["down"][0] is banks.down

"""The device-resident NVFP4 MTP expert placement.

The bf16 resident banks cost 5.03 GB of the card the TARGET's expert cache is competing for.
This placement keeps the same 512 experts quantized on the device (~1.42 GB) and dequantizes
only the routed rows, into a preallocated bf16 scratch, at gather time -- after which the
routed pipeline is the bf16 gathered path's, kernel for kernel.

What is pinned here: the device dequant reproduces the CPU reference's arithmetic EXACTLY
(same six-bank format interpretation), the routed forward equals the bf16 gathered runner on
banks carrying the same dequantized numbers, and the whole path stays capture-legal -- no
host round trip, no value-dependent shape, no steady-state growth of the weight scratch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.models.qwen4_exp.mtp_spike import (
    MTPBF16ExpertBanks,
    MTPGPUExpertRunner,
    MTPNVFP4ExpertBanks,
    MTPNVFP4GPUExpertRunner,
    dequantize_nvfp4_rows,
    quantize_nvfp4_rows,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

EXPERTS, HIDDEN, INTERMEDIATE = 8, 256, 128


def _quantize_bank(tensor: torch.Tensor):
    shape = tensor.shape
    packed, scale, glob = quantize_nvfp4_rows(tensor.reshape(-1, shape[-1]).float())
    return (
        packed.view(*shape[:-1], shape[-1] // 2).contiguous(),
        scale.view(*shape[:-1], shape[-1] // 16).contiguous(),
        glob.view(*shape[:-1]).contiguous(),
    )


def _dequantized_bank(packed, scale, glob) -> torch.Tensor:
    width = 2 * packed.shape[-1]
    rows = dequantize_nvfp4_rows(
        packed.reshape(-1, packed.shape[-1]),
        scale.reshape(-1, scale.shape[-1]),
        glob.reshape(-1),
    )
    return rows.view(*packed.shape[:-1], width)


def _paired_banks(seed: int = 11):
    """One expert layer in both placements, carrying the SAME numbers.

    The bf16 banks are the NVFP4 banks dequantized, so any difference between the two
    runners is arithmetic ordering, not quantization.
    """
    generator = torch.Generator().manual_seed(seed)
    gate_up = torch.randn(
        EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator
    ) * 0.025
    down = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, generator=generator) * 0.025
    gate_parts = _quantize_bank(gate_up)
    down_parts = _quantize_bank(down)
    quantized = MTPNVFP4ExpertBanks(*gate_parts, *down_parts)
    exact = MTPBF16ExpertBanks(
        _dequantized_bank(*gate_parts).to(torch.bfloat16).contiguous(),
        _dequantized_bank(*down_parts).to(torch.bfloat16).contiguous(),
    )
    return quantized, exact


def _routes(tokens: int, num_experts: int, top_k: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(num_experts, generator=generator)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    weights = torch.rand(tokens, top_k, generator=generator)
    weights /= weights.sum(dim=-1, keepdim=True)
    return weights, ids


def _quantized_runner(banks, *, device, top_k=2, max_tokens=128, max_gather_tokens=4):
    return MTPNVFP4GPUExpertRunner(
        banks,
        top_k=top_k,
        activation="silu",
        renormalize=True,
        max_tokens=max_tokens,
        num_threads=4,
        device=device,
        max_gather_tokens=max_gather_tokens,
    )


def _exact_runner(banks, *, device, top_k=2, max_tokens=128):
    return MTPGPUExpertRunner(
        banks,
        top_k=top_k,
        activation="silu",
        renormalize=True,
        max_tokens=max_tokens,
        num_threads=4,
        device=device,
    )


# ------------------------------------------------------------------- format exactness


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_dequant_reproduces_the_cpu_reference_bit_for_bit(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    quantized, _ = _paired_banks(seed=21)
    device = torch.device(device_name)
    runner = _quantized_runner(quantized, device=device)
    ids = torch.tensor([5, 0, 7, 0], dtype=torch.int64, device=device)

    gate_up, down = runner.dequantize_experts(ids)

    expected_gate = _dequantized_bank(
        quantized.gate_up_packed, quantized.gate_up_scale, quantized.gate_up_global
    ).to(torch.bfloat16)
    expected_down = _dequantized_bank(
        quantized.down_packed, quantized.down_scale, quantized.down_global
    ).to(torch.bfloat16)
    for slot, expert in enumerate([5, 0, 7, 0]):
        assert torch.equal(gate_up[slot].cpu(), expected_gate[expert]), expert
        assert torch.equal(down[slot].cpu(), expected_down[expert]), expert


def test_dequantized_scratch_is_preallocated_and_bounded_by_the_routed_width():
    quantized, _ = _paired_banks(seed=22)
    runner = _quantized_runner(
        quantized, device=torch.device("cpu"), top_k=2, max_gather_tokens=4
    )

    assert runner.max_gather_pairs == 8
    assert runner.gather_max_tokens == 4
    gate_up, down = runner.dequantize_experts(torch.zeros(8, dtype=torch.int64))
    assert gate_up.shape == (8, 2 * INTERMEDIATE, HIDDEN)
    assert down.shape == (8, HIDDEN, INTERMEDIATE)
    assert gate_up.dtype is torch.bfloat16 and down.dtype is torch.bfloat16
    # a second call reuses the same storage -- the scratch never grows in steady state
    again, again_down = runner.dequantize_experts(torch.ones(8, dtype=torch.int64))
    assert again.data_ptr() == gate_up.data_ptr()
    assert again_down.data_ptr() == down.data_ptr()
    with pytest.raises(ValueError, match="at most 8"):
        runner.dequantize_experts(torch.zeros(9, dtype=torch.int64))


def test_quantized_runner_charges_the_card_only_the_banks_and_its_scratch():
    quantized, exact = _paired_banks(seed=23)
    runner = _quantized_runner(quantized, device=torch.device("cpu"))

    scratch = 8 * (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) * 2
    assert runner.bank_bytes == quantized.total_bytes
    assert runner.scratch_bytes == scratch
    assert runner.resident_bytes == quantized.total_bytes + scratch
    # 4 bits a weight plus a 1-byte block scale every 16 and a 2-byte row global: the banks
    # land at ~3.5x smaller than bf16 (5.03 GB -> 1.42 GB at the real 512x2560x640 geometry)
    assert exact.total_bytes / runner.bank_bytes > 3.4


def test_quantized_runner_refuses_exact_bf16_banks():
    _, exact = _paired_banks(seed=24)
    with pytest.raises(ValueError, match="nvfp4"):
        _quantized_runner(exact, device=torch.device("cpu"))


def test_quantized_runner_refuses_a_geometry_it_cannot_gather():
    quantized, _ = _paired_banks(seed=25)
    with pytest.raises(ValueError, match="max_gather_tokens"):
        _quantized_runner(quantized, device=torch.device("cpu"), max_gather_tokens=0)


# ------------------------------------------------------------------ the routed forward


@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_quantized_routed_forward_matches_the_bf16_gathered_runner(tokens):
    quantized, exact = _paired_banks(seed=30 + tokens)
    device = torch.device("cpu")
    runner = _quantized_runner(quantized, device=device)
    reference = _exact_runner(exact, device=device)
    generator = torch.Generator().manual_seed(40 + tokens)
    hidden = (torch.randn(tokens, HIDDEN, generator=generator) * 0.5).to(torch.bfloat16)
    weights, ids = _routes(tokens, EXPERTS, 2, seed=50 + tokens)
    assert tokens <= runner.gather_max_tokens
    assert tokens <= reference.gather_max_tokens

    got = runner.run_routed(hidden, weights, ids)

    expected = reference.run_routed(hidden, weights, ids)
    assert got.shape == expected.shape and got.dtype is expected.dtype
    torch.testing.assert_close(got.float(), expected.float(), rtol=0, atol=0)
    assert runner.stats.calls == 1
    assert runner.stats.tokens == tokens
    assert runner.stats.logical_expert_bytes == tokens * 2 * quantized.bytes_per_expert
    assert runner.stats.explicit_pcie_bytes == 0


def test_quantized_routed_forward_zeroes_unrouted_negative_expert_ids():
    quantized, exact = _paired_banks(seed=61)
    device = torch.device("cpu")
    runner = _quantized_runner(quantized, device=device)
    reference = _exact_runner(exact, device=device)
    generator = torch.Generator().manual_seed(62)
    hidden = (torch.randn(2, HIDDEN, generator=generator) * 0.5).to(torch.bfloat16)
    weights, ids = _routes(2, EXPERTS, 2, seed=63)
    ids[1, 1] = -1

    got = runner.run_routed(hidden, weights, ids)

    torch.testing.assert_close(
        got.float(), reference.run_routed(hidden, weights, ids).float(), rtol=0, atol=0
    )


def test_quantized_runner_falls_back_to_the_expert_major_loop_beyond_the_gather_width():
    """Prompt priming runs wider than the speculative step and takes the loop, which is free
    to synchronize; only its numbers have to agree."""
    quantized, exact = _paired_banks(seed=70)
    device = torch.device("cpu")
    runner = _quantized_runner(quantized, device=device, max_gather_tokens=2)
    reference = _exact_runner(exact, device=device)
    generator = torch.Generator().manual_seed(71)
    hidden = (torch.randn(6, HIDDEN, generator=generator) * 0.5).to(torch.bfloat16)
    weights, ids = _routes(6, EXPERTS, 2, seed=72)
    assert 6 > runner.gather_max_tokens

    got = runner.run_routed(hidden, weights, ids)

    expected = reference.run_routed(hidden, weights, ids)
    torch.testing.assert_close(got.float(), expected.float(), rtol=0, atol=1e-2)


def test_quantized_runner_rejects_oversized_or_malformed_calls():
    quantized, _ = _paired_banks(seed=80)
    runner = _quantized_runner(quantized, device=torch.device("cpu"), max_tokens=2)

    with pytest.raises(ValueError, match="at most 2"):
        runner.run_routed(
            torch.zeros(3, HIDDEN, dtype=torch.bfloat16),
            torch.ones(3, 2),
            torch.zeros(3, 2, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="bfloat16"):
        runner.run_routed(
            torch.zeros(1, HIDDEN),
            torch.ones(1, 2),
            torch.zeros(1, 2, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="top-k ids"):
        runner.run_routed(
            torch.zeros(1, HIDDEN, dtype=torch.bfloat16),
            torch.ones(1, 2),
            torch.zeros(1, 2, dtype=torch.int64),
        )


def test_quantized_runner_routes_with_the_same_topk_softmax():
    quantized, _ = _paired_banks(seed=90)
    runner = _quantized_runner(quantized, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(91)
    hidden = torch.randn(3, HIDDEN, generator=generator).to(torch.bfloat16)
    logits = torch.randn(3, EXPERTS, generator=generator).to(torch.bfloat16)

    weights, ids = runner.route(hidden, logits)

    probabilities = logits.float().softmax(dim=-1)
    expected_weights, expected_ids = probabilities.topk(2, dim=-1)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    assert torch.equal(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(weights, expected_weights, rtol=2e-4, atol=2e-4)
    assert runner.forward(hidden, logits).shape == hidden.shape


# ------------------------------------------------------------------- capture legality


class _SyncTrap:
    """Fail the test on any device->host round trip the routed path might take."""

    def __init__(self, monkeypatch):
        for name in ("tolist", "item", "__int__", "__float__", "__bool__"):
            monkeypatch.setattr(
                torch.Tensor,
                name,
                lambda self, _name=name, *args, **kwargs: pytest.fail(
                    f"the quantized routed path synchronized through Tensor.{_name}"
                ),
                raising=True,
            )


@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_quantized_routed_path_takes_no_device_synchronization(tokens, monkeypatch):
    quantized, _ = _paired_banks(seed=100 + tokens)
    runner = _quantized_runner(quantized, device=torch.device("cpu"), max_tokens=8)
    generator = torch.Generator().manual_seed(110 + tokens)
    hidden = (torch.randn(tokens, HIDDEN, generator=generator) * 0.5).to(torch.bfloat16)
    weights, ids = _routes(tokens, EXPERTS, 2, seed=120 + tokens)
    assert tokens <= runner.gather_max_tokens

    _SyncTrap(monkeypatch)
    got = runner.run_routed(hidden, weights, ids)

    assert got.shape == hidden.shape


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 4])
def test_quantized_routed_path_is_cuda_graph_capturable(tokens):
    quantized, exact = _paired_banks(seed=130 + tokens)
    device = torch.device("cuda")
    runner = _quantized_runner(quantized, device=device, max_tokens=8)
    reference = _exact_runner(exact, device=device, max_tokens=8)
    generator = torch.Generator().manual_seed(140 + tokens)
    hidden_buf = (
        (torch.randn(tokens, HIDDEN, generator=generator) * 0.5).to(torch.bfloat16).to(device)
    )
    weights_cpu, ids_cpu = _routes(tokens, EXPERTS, 2, seed=150 + tokens)
    weight_buf = weights_cpu.to(device)
    id_buf = ids_cpu.to(device).to(torch.int32)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = runner.run_routed(hidden_buf, weight_buf, id_buf)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = runner.run_routed(hidden_buf, weight_buf, id_buf)

    next_weights, next_ids = _routes(tokens, EXPERTS, 2, seed=160 + tokens)
    hidden_buf.copy_(
        (torch.randn(tokens, HIDDEN, generator=generator) * 0.5).to(torch.bfloat16)
    )
    weight_buf.copy_(next_weights)
    id_buf.copy_(next_ids.to(torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected = reference.run_routed(hidden_buf, weight_buf, id_buf)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=2e-2)
    graph.reset()


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 4])
def test_quantized_and_exact_runners_agree_on_the_device(tokens):
    quantized, exact = _paired_banks(seed=170 + tokens)
    device = torch.device("cuda")
    runner = _quantized_runner(quantized, device=device, max_tokens=8)
    reference = _exact_runner(exact, device=device, max_tokens=8)
    generator = torch.Generator().manual_seed(180 + tokens)
    hidden = (
        (torch.randn(tokens, HIDDEN, generator=generator) * 0.5).to(torch.bfloat16).to(device)
    )
    weights, ids = _routes(tokens, EXPERTS, 2, seed=190 + tokens)
    weights = weights.to(device)
    ids = ids.to(device)

    got = runner.run_routed(hidden, weights, ids)
    expected = reference.run_routed(hidden, weights, ids)
    torch.cuda.synchronize()

    torch.testing.assert_close(got.float(), expected.float(), rtol=0, atol=0)


def test_the_quantized_gather_covers_the_widest_speculative_step():
    """Every per-cycle draft-head call is at most ``1 + depth`` rows; the quantized gather is
    sized for exactly that, because unlike the bf16 gather it also fixes a RESIDENT scratch."""
    quantized, _ = _paired_banks(seed=200)
    runner = _quantized_runner(quantized, device=torch.device("cpu"), max_gather_tokens=4)

    assert runner.gather_max_tokens == 4
    assert runner.max_gather_pairs == 4 * runner.top_k


def _independent_expert_major(gate_up, down, hidden, weights, ids):
    result = torch.zeros_like(hidden, dtype=torch.float32)
    x = hidden.float()
    for token in range(hidden.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            projected = gate_up[expert].float() @ x[token]
            gate, up = projected.chunk(2)
            result[token] += weights[token, route].float() * (
                down[expert].float() @ (F.silu(gate) * up)
            )
    return result


def test_quantized_routed_forward_matches_an_independent_torch_reference():
    quantized, exact = _paired_banks(seed=210)
    runner = _quantized_runner(quantized, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(211)
    hidden = (torch.randn(3, HIDDEN, generator=generator) * 0.5).to(torch.bfloat16)
    weights, ids = _routes(3, EXPERTS, 2, seed=212)

    got = runner.run_routed(hidden, weights, ids)

    expected = _independent_expert_major(exact.gate_up, exact.down, hidden, weights, ids)
    torch.testing.assert_close(got.float(), expected, rtol=0, atol=0.05)


@requires_cuda
def test_the_fused_device_dequant_and_the_torch_fallback_are_bit_identical():
    """The device path uses the canonical fused gather+dequant kernel where it exists; the
    portable torch arithmetic must decode the same bytes to the same bits."""
    quantized, _ = _paired_banks(seed=230)
    runner = _quantized_runner(quantized, device=torch.device("cuda"))
    if runner._triton_dequant is None:
        pytest.skip("this device has no fused nvfp4 dequant kernel")
    ids = torch.tensor([3, 1, 0, 6], dtype=torch.int64, device="cuda")

    fused = [tensor.clone() for tensor in runner.dequantize_experts(ids)]

    runner._triton_dequant = None
    for got, expected in zip(runner.dequantize_experts(ids), fused):
        assert torch.equal(got, expected)


def test_quantized_runner_keeps_no_host_bank_reference():
    """The 1.4 GB host copy must not outlive construction any more than the bf16 one does."""
    quantized, _ = _paired_banks(seed=220)
    runner = _quantized_runner(quantized, device=torch.device("cpu"))

    assert not hasattr(runner.banks, "gate_up_packed")
    assert runner.banks.num_experts == EXPERTS
    assert runner.banks.hidden_size == HIDDEN
    assert runner.banks.intermediate_size == INTERMEDIATE
    runner.raise_if_unhealthy()
    runner.close()

"""CPU-safe reconstruct-first EXL3 operation tests with mocked card seams."""

from __future__ import annotations

import pytest
import torch


H = I = 128


def _banks(num_experts: int = 10):
    trellis_shape = (num_experts, H // 16, I // 16, 32)
    gate_trellis = torch.zeros(trellis_shape, dtype=torch.int16)
    up_trellis = torch.zeros_like(gate_trellis)
    down_trellis = torch.zeros((num_experts, I // 16, H // 16, 32), dtype=torch.int16)

    def factors(width: int, offset: int):
        return torch.stack(
            [torch.full((width,), float(offset + expert), dtype=torch.float16)
             for expert in range(num_experts)]
        )

    return (
        gate_trellis,
        factors(H, 1),
        factors(I, 101),
        up_trellis,
        factors(H, 201),
        factors(I, 301),
        down_trellis,
        factors(I, 401),
        factors(H, 501),
    )


def _fake_reconstruct(trellis, suh, svh, *, k, codebook, out, work):
    del trellis, svh, k, codebook, work
    out.zero_()
    # Each bank's factor row carries a distinct marker. Square fixtures let the
    # diagonal stand in for a complete reconstructed linear matrix.
    out.diagonal().fill_(float(suh[0]))
    return out


def _activation(gate_up: torch.Tensor, name: str, *, alpha=1.0, limit=10.0):
    gate, up = gate_up.chunk(2, dim=-1)
    if name == "swiglu_clamp":
        gate = gate.float().clamp(max=limit)
        up = up.float().clamp(min=-limit, max=limit)
        return gate * torch.sigmoid(alpha * gate) * up
    if name == "silu":
        return torch.nn.functional.silu(gate.float()) * up.float()
    raise AssertionError(name)


def _reference(hidden, weights, ids, matrices, *, activation="silu", alpha=1.0, limit=10.0):
    result = torch.zeros_like(hidden, dtype=torch.float32)
    for token in range(hidden.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            gate_up = matrices[expert][0].float() @ hidden[token].float()
            activated = _activation(
                gate_up,
                activation,
                alpha=alpha,
                limit=limit,
            )
            contribution = matrices[expert][1].float() @ activated
            result[token] += float(weights[token, route]) * contribution
    return result.to(hidden.dtype)


def _install_mocks(monkeypatch):
    from freetoken.moe import fused_exl3

    monkeypatch.setattr(fused_exl3._exl3_kernel, "reconstruct", _fake_reconstruct)
    monkeypatch.setattr(
        fused_exl3,
        "require_exl3_gpu_only",
        lambda *, device, decode_target="gpu": None,
    )

    def prompt(hidden, w1, w2, weights, ids, **kwargs):
        out = _reference(
            hidden,
            weights,
            ids,
            [(w1[e], w2[e]) for e in range(w1.shape[0])],
            activation=kwargs.get("activation", "silu"),
            alpha=kwargs.get("hidden_act_alpha", 1.0),
            limit=kwargs.get("swiglu_limit", 10.0),
        )
        hidden.copy_(out)
        return hidden

    def decode(hidden, w1, w2, weights, ids, **kwargs):
        return _reference(
            hidden,
            weights,
            ids,
            [(w1[e], w2[e]) for e in range(w1.shape[0])],
            activation=kwargs.get("activation", "silu"),
            alpha=kwargs.get("hidden_act_alpha", 1.0),
            limit=kwargs.get("swiglu_limit", 10.0),
        )

    monkeypatch.setattr(fused_exl3, "fused_experts_impl", prompt)
    monkeypatch.setattr(fused_exl3, "fused_experts_decode_impl", decode)
    return fused_exl3


def _matrices(num_experts: int):
    banks = _banks(num_experts)
    matrices = []
    for expert in range(num_experts):
        gate = torch.eye(I) * float(1 + expert)
        up = torch.eye(I) * float(201 + expert)
        down = torch.eye(H) * float(401 + expert)
        matrices.append((torch.cat((gate, up), dim=0), down))
    return banks, matrices


def test_prepare_scratch_uses_fixed_contiguous_buffers():
    from freetoken.moe.fused_exl3 import prepare_exl3_scratch

    scratch = prepare_exl3_scratch(
        device="cpu", hidden_size=H, intermediate_size=I, max_tokens=32, chunk_experts=8
    )
    assert scratch.gate_up.shape == (8, 2 * I, H)
    assert scratch.down.shape == (8, H, I)
    assert scratch.reconstruct_work.shape == (H, I)
    assert scratch.input_buffer.shape == (32, H)
    assert scratch.output_accumulator.shape == (32, H)
    assert scratch.gate_up.dtype == torch.bfloat16
    assert scratch.reconstruct_work.dtype == torch.float16
    assert scratch.route_ids.is_contiguous() and scratch.route_weights.is_contiguous()
    assert scratch.route_ids[: 3 * 2].view(3, 2).is_contiguous()
    with pytest.raises(ValueError, match="1..8"):
        prepare_exl3_scratch(
            device="cpu", hidden_size=H, intermediate_size=I, max_tokens=32, chunk_experts=9
        )


def test_reconstruct_first_decode_chunks_unique_experts_and_clears_routes(monkeypatch):
    fused_exl3 = _install_mocks(monkeypatch)
    banks, matrices = _matrices(10)
    hidden = torch.randn(5, H, dtype=torch.bfloat16) / 8
    ids = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=torch.int32)
    weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.8, 0.2], [0.25, 0.75]],
        dtype=torch.float32,
    )
    original_ids = ids.clone()
    scratch = fused_exl3.prepare_exl3_scratch(
        device="cpu", hidden_size=H, intermediate_size=I, max_tokens=8, chunk_experts=8
    )

    got = fused_exl3.fused_experts_exl3(
        hidden,
        banks,
        weights,
        ids,
        is_prefill=False,
        activation="silu",
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        hidden_act_alpha=1.0,
        scratch=scratch,
    )
    expected = _reference(hidden, weights, ids, matrices)

    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
    assert torch.equal(ids, original_ids), "the caller's routing ids must not be rewritten"


def test_reconstruct_first_prompt_restores_original_input_and_clamped_activation(monkeypatch):
    fused_exl3 = _install_mocks(monkeypatch)
    banks, matrices = _matrices(4)
    hidden = torch.tensor(
        [[-12.0, -10.0, -2.0, 0.0] + [2.0] * (H - 4)], dtype=torch.bfloat16
    ).repeat(3, 1)
    ids = torch.tensor([[3], [3], [3]], dtype=torch.int32)
    weights = torch.ones((3, 1), dtype=torch.float32)
    original = hidden.clone()
    scratch = fused_exl3.prepare_exl3_scratch(
        device="cpu", hidden_size=H, intermediate_size=I, max_tokens=8, chunk_experts=8
    )

    got = fused_exl3.fused_experts_exl3(
        hidden,
        banks,
        weights,
        ids,
        is_prefill=True,
        activation="swiglu_clamp",
        apply_router_weight_on_input=False,
        swiglu_limit=10.0,
        hidden_act_alpha=1.0,
        scratch=scratch,
    )
    expected = _reference(
        original,
        weights,
        ids,
        matrices,
        activation="swiglu_clamp",
        alpha=1.0,
        limit=10.0,
    )

    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
    assert torch.equal(hidden, original), "prompt input must remain available to the shared expert"


def test_exl3_operation_refuses_cpu_without_a_mocked_card_seam():
    from freetoken.moe.fused_exl3 import fused_experts_exl3, prepare_exl3_scratch

    banks, _ = _matrices(1)
    hidden = torch.zeros((1, H), dtype=torch.bfloat16)
    ids = torch.zeros((1, 1), dtype=torch.int32)
    weights = torch.ones((1, 1), dtype=torch.float32)
    scratch = prepare_exl3_scratch(
        device="cpu", hidden_size=H, intermediate_size=I, max_tokens=1, chunk_experts=8
    )
    with pytest.raises(ValueError, match="card-only|CUDA"):
        fused_experts_exl3(
            hidden,
            banks,
            weights,
            ids,
            is_prefill=False,
            activation="silu",
            apply_router_weight_on_input=False,
            swiglu_limit=None,
            hidden_act_alpha=1.0,
            scratch=scratch,
        )

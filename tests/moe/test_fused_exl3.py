"""EXL3 operation tests with mocked CPU seams and optional card agreement checks."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest
import torch


H = I = 128
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
REAL_MODEL_PATH = Path(
    os.environ.get("FREETOKEN_GLM53_EXL3_MODEL", r"D:\Models\GLM-5.3-Flash-exl3-2.05bpw")
)


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


def _mgemm_banks(num_experts: int = 8):
    torch.manual_seed(19)
    gate_up_trellis = torch.randint(
        -32768,
        32767,
        (num_experts, H // 16, I // 16, 32),
        dtype=torch.int16,
    ).contiguous()
    down_trellis = torch.randint(
        -32768,
        32767,
        (num_experts, I // 16, H // 16, 32),
        dtype=torch.int16,
    ).contiguous()

    def factor(width: int):
        return torch.rand((num_experts, width), dtype=torch.float16).contiguous()

    return (
        gate_up_trellis,
        factor(H),
        factor(I),
        gate_up_trellis.clone(),
        factor(H),
        factor(I),
        down_trellis,
        factor(I),
        factor(H),
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


def _load_real_banks(model_path: Path, *, experts: int = 8):
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

    return tuple(
        torch.stack(
            [
                loaded[
                    f"model.language_model.layers.3.mlp.experts.{expert}."
                    f"{projection}_proj.{kind}"
                ]
                for expert in range(experts)
            ],
            dim=0,
        )
        for projection in ("gate", "up", "down")
        for kind in ("trellis", "suh", "svh")
    )


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
    # Ten unique experts force a second eight-expert prompt chunk, where restoring the
    # caller's input is required before the BF16 operation overwrites its working copy.
    banks, matrices = _matrices(10)
    hidden = torch.tensor(
        [[-12.0, -10.0, -2.0, 0.0] + [2.0] * (H - 4)], dtype=torch.bfloat16
    ).repeat(10, 1)
    ids = torch.arange(10, dtype=torch.int32).view(10, 1)
    weights = torch.ones((10, 1), dtype=torch.float32)
    original = hidden.clone()
    scratch = fused_exl3.prepare_exl3_scratch(
        device="cpu", hidden_size=H, intermediate_size=I, max_tokens=16, chunk_experts=8
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


def test_graph_decode_compacts_routes_on_device_and_matches_reference(monkeypatch):
    fused_exl3 = _install_mocks(monkeypatch)
    banks, matrices = _matrices(8)
    hidden = torch.randn(1, H, dtype=torch.bfloat16) / 8
    ids = torch.arange(8, dtype=torch.int32).view(1, 8)
    weights = torch.tensor(
        [[0.03, 0.07, 0.11, 0.15, 0.18, 0.19, 0.17, 0.10]], dtype=torch.float32
    )
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
    assert torch.equal(scratch.slot_list, torch.arange(8, dtype=torch.int32))
    assert scratch.slot_count.item() == 8


def test_graph_decode_reuses_fixed_workspace_without_host_reads_or_allocations(monkeypatch):
    fused_exl3 = _install_mocks(monkeypatch)
    banks, _ = _matrices(8)
    hidden = torch.randn(1, H, dtype=torch.bfloat16) / 8
    ids = torch.arange(8, dtype=torch.int32).view(1, 8)
    weights = torch.full((1, 8), 1 / 8, dtype=torch.float32)
    scratch = fused_exl3.prepare_exl3_scratch(
        device="cpu", hidden_size=H, intermediate_size=I, max_tokens=8, chunk_experts=8
    )

    first = fused_exl3.fused_experts_exl3(
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

    def fail(*args, **kwargs):
        raise AssertionError("graph-safe decode read host data or allocated a tensor")

    monkeypatch.setattr(torch.Tensor, "cpu", fail)
    monkeypatch.setattr(torch.Tensor, "tolist", fail)
    monkeypatch.setattr(torch, "empty", fail)
    second = fused_exl3.fused_experts_exl3(
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

    torch.testing.assert_close(second, first)
    assert second.data_ptr() == scratch.output_accumulator.data_ptr()


@cuda
def test_packed_wrapper_matches_reconstruct_first_on_synthetic_banks():
    """The fused module must preserve the complete GLM operation, not only one projection."""
    pytest.importorskip("exllamav3_ext")
    from freetoken.moe.fused_exl3 import fused_experts_exl3, prepare_exl3_scratch

    device = torch.device("cuda")
    banks = tuple(bank.to(device=device) for bank in _mgemm_banks(8))
    hidden = torch.randn((2, H), dtype=torch.bfloat16, device=device)
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device=device)
    weights = torch.tensor(
        [[0.10, 0.20, 0.30, 0.40], [0.40, 0.30, 0.20, 0.10]],
        dtype=torch.float32,
        device=device,
    )
    scratch = prepare_exl3_scratch(
        device=device,
        hidden_size=H,
        intermediate_size=I,
        max_tokens=8,
        chunk_experts=8,
        decode_max_tokens=2,
        enable_mgemm=True,
    )
    kwargs = dict(
        is_prefill=False,
        activation="swiglu_clamp",
        apply_router_weight_on_input=False,
        swiglu_limit=10.0,
        hidden_act_alpha=1.0,
        scratch=scratch,
        layer_id=0,
    )
    expected = fused_experts_exl3(
        hidden, banks, weights, ids, expert_op="reconstruct", **kwargs
    ).clone()
    got = fused_experts_exl3(hidden, banks, weights, ids, expert_op="mgemm", **kwargs)
    torch.testing.assert_close(got.float(), expected.float(), rtol=5e-2, atol=0.5)


@pytest.mark.needs_weights
@pytest.mark.slow
@cuda
def test_real_checkpoint_packed_full_operation_agrees_for_decode_and_grouped_prompt(
    monkeypatch,
):
    """Compare the complete packed operation with reconstruct-first on real GLM rows."""
    pytest.importorskip("exllamav3_ext")
    if not REAL_MODEL_PATH.is_dir():
        pytest.skip(f"checkpoint is not present: {REAL_MODEL_PATH}")

    from freetoken.moe.fused_exl3 import fused_experts_exl3, prepare_exl3_scratch

    device = torch.device("cuda")
    cpu_banks = _load_real_banks(REAL_MODEL_PATH)
    banks = tuple(bank.to(device=device) for bank in cpu_banks)
    hidden_size = int(banks[0].shape[1] * 16)
    intermediate_size = int(banks[0].shape[2] * 16)
    scratch = prepare_exl3_scratch(
        device=device,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_tokens=256,
        chunk_experts=8,
        decode_max_tokens=1,
        enable_mgemm=True,
    )

    def run(hidden, weights, ids, *, is_prefill, expert_op):
        return fused_experts_exl3(
            hidden,
            banks,
            weights,
            ids,
            is_prefill=is_prefill,
            activation="swiglu_clamp",
            apply_router_weight_on_input=False,
            swiglu_limit=10.0,
            hidden_act_alpha=1.0,
            scratch=scratch,
            expert_op=expert_op,
            layer_id=3,
        )

    decode_hidden = torch.randn((1, hidden_size), dtype=torch.bfloat16, device=device)
    decode_ids = torch.arange(8, dtype=torch.int32, device=device).view(1, 8)
    decode_weights = torch.tensor(
        [[0.03, 0.07, 0.11, 0.15, 0.18, 0.19, 0.17, 0.10]],
        dtype=torch.float32,
        device=device,
    )
    decode_expected = run(
        decode_hidden,
        decode_weights,
        decode_ids,
        is_prefill=False,
        expert_op="reconstruct",
    ).clone()
    decode_got = run(
        decode_hidden,
        decode_weights,
        decode_ids,
        is_prefill=False,
        expert_op="mgemm",
    ).clone()
    torch.testing.assert_close(
        decode_got.float(), decode_expected.float(), rtol=5e-2, atol=0.5
    )

    # One expert is deliberately repeated for 129 rows. Grouping keeps each packed call at one
    # expert index while the wrapper tiles the group to 128 rows, proving this is not a row cap.
    prompt_rows = 129
    prompt_hidden = torch.randn(
        (prompt_rows, hidden_size), dtype=torch.bfloat16, device=device
    )
    prompt_ids = torch.zeros((prompt_rows, 1), dtype=torch.int32, device=device)
    prompt_weights = torch.ones((prompt_rows, 1), dtype=torch.float32, device=device)
    prompt_expected = run(
        prompt_hidden,
        prompt_weights,
        prompt_ids,
        is_prefill=True,
        expert_op="reconstruct",
    ).clone()
    prompt_got = run(
        prompt_hidden,
        prompt_weights,
        prompt_ids,
        is_prefill=True,
        expert_op="mgemm",
    ).clone()
    torch.testing.assert_close(
        prompt_got.float(), prompt_expected.float(), rtol=5e-2, atol=0.5
    )

    # Pointer tables and every full-operation tensor are now warm. These guards catch a regression
    # that moves the old torch.cat/torch.empty/host-routing work back into graph replay.
    warm_decode = run(
        decode_hidden,
        decode_weights,
        decode_ids,
        is_prefill=False,
        expert_op="mgemm",
    ).clone()

    def fail(*args, **kwargs):
        raise AssertionError("packed decode allocated or read routing data on the host")

    monkeypatch.setattr(torch.Tensor, "cpu", fail)
    monkeypatch.setattr(torch.Tensor, "tolist", fail)
    monkeypatch.setattr(torch, "empty", fail)
    monkeypatch.setattr(torch, "tensor", fail)
    monkeypatch.setattr(torch, "cat", fail)
    replay = run(
        decode_hidden,
        decode_weights,
        decode_ids,
        is_prefill=False,
        expert_op="mgemm",
    )
    assert torch.equal(replay, warm_decode)
    assert replay.data_ptr() == scratch.output_accumulator.data_ptr()


def test_packed_decode_falls_back_once_at_the_route_index_limit(monkeypatch):
    """The real wrapper limit must choose reconstruct-first and emit one warning per shape."""
    fused_exl3 = _install_mocks(monkeypatch)
    from freetoken.kernel.exl3_mgemm import Exl3MgemmBanks

    banks, _ = _matrices(10)
    # Construct only the table shell: fused_experts_exl3_mgemm checks the route limit before it
    # needs CUDA or an extension, so this remains a CPU-side fallback test.
    tables = Exl3MgemmBanks(
        tuple(banks),
        tuple(torch.empty(10, dtype=torch.int64) for _ in range(9)),
        k=2,
    )
    monkeypatch.setattr(
        fused_exl3,
        "_mgemm_tables_for_views",
        lambda scratch, views: tables,
    )
    fused_exl3._MGEMM_FALLBACKS.clear()
    warnings = []
    monkeypatch.setattr(
        fused_exl3.logger,
        "warning_rank0",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    rows = 129  # top_k=1 => 129 route indices, one past the wheel's 128-entry limit.
    hidden = torch.randn((rows, H), dtype=torch.bfloat16)
    weights = torch.ones((rows, 1), dtype=torch.float32)
    ids = torch.zeros((rows, 1), dtype=torch.int32)
    scratch = fused_exl3.prepare_exl3_scratch(
        device="cpu", hidden_size=H, intermediate_size=I, max_tokens=rows, chunk_experts=8
    )

    first = fused_exl3.fused_experts_exl3(
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
        expert_op="mgemm",
        layer_id=991,
    ).clone()
    second = fused_exl3.fused_experts_exl3(
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
        expert_op="mgemm",
        layer_id=991,
    )

    assert torch.equal(second, first)
    assert torch.count_nonzero(first) > 0
    assert len(warnings) == 1
    assert "falling back" in warnings[0][0][0]

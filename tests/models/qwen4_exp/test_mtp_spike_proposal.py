from __future__ import annotations

import pytest
import torch

from .common import Fixture, parsed_config, requires_cuda
from freetoken.models.qwen4_exp.mtp_spike import (
    MTPDraftSampler,
    MTPProposalEngine,
    MTPShadowMetrics,
    accepted_draft_prefix,
)


def test_shifted_pairing_recursive_state_and_qsa_reuse():
    embedded_tokens = []
    state_inputs = []
    qsa_modes = []
    selected = torch.tensor([[5, 9, 13]], dtype=torch.int32)

    def embedding(token):
        embedded_tokens.append(token)
        return torch.tensor([[float(token)]])

    def step(embedding_row, hidden_state, *, saved_qsa_blocks):
        state_inputs.append(hidden_state.clone())
        qsa_modes.append("select" if saved_qsa_blocks is None else "reuse")
        if saved_qsa_blocks is not None:
            assert torch.equal(saved_qsa_blocks, selected)
        recursive = hidden_state + embedding_row.repeat(1, hidden_state.shape[1])
        # Force proposals 4, then 6, then 8.
        index = len(state_inputs)
        sample_hidden = torch.tensor([[float(index * 2 + 2)]])
        return sample_hidden, recursive, selected if saved_qsa_blocks is None else None

    def lm_head(sample_hidden):
        token = int(sample_hidden.item())
        logits = torch.full((1, 16), -1000.0)
        logits[0, token] = 1000.0
        return logits

    result = MTPProposalEngine(MTPDraftSampler(seed=1, device=torch.device("cpu"))).propose(
        initial_token=2,
        target_hidden=torch.ones(1, 4),
        draft_tokens=3,
        embedding=embedding,
        step=step,
        lm_head=lm_head,
        temperature=0.0,
    )

    assert result.tokens.tolist() == [4, 6, 8]
    assert embedded_tokens == [2, 4, 6]
    assert qsa_modes == ["select", "reuse", "reuse"]
    assert torch.equal(result.qsa_blocks, selected)
    torch.testing.assert_close(state_inputs[0], torch.ones(1, 4))
    torch.testing.assert_close(state_inputs[1], torch.full((1, 4), 3.0))
    torch.testing.assert_close(state_inputs[2], torch.full((1, 4), 7.0))
    torch.testing.assert_close(result.recursive_state, torch.full((1, 4), 13.0))


def test_dedicated_draft_rng_is_repeatable_and_does_not_touch_global_rng():
    logits = torch.tensor([0.2, 0.7, -0.1, 1.1, 0.3])
    global_before = torch.random.get_rng_state().clone()
    first = MTPDraftSampler(seed=12345, device=torch.device("cpu"))
    second = MTPDraftSampler(seed=12345, device=torch.device("cpu"))
    seq_a = [first.sample(logits, temperature=0.8, top_k=4, top_p=0.9) for _ in range(32)]
    seq_b = [second.sample(logits, temperature=0.8, top_k=4, top_p=0.9) for _ in range(32)]
    assert seq_a == seq_b
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert set(seq_a) <= {0, 1, 3, 4}


@requires_cuda
def test_cuda_draft_generator_does_not_consume_target_cuda_rng():
    target_before = torch.cuda.get_rng_state().clone()
    sampler = MTPDraftSampler(seed=909, device=torch.device("cuda", 0))
    logits = torch.tensor([0.1, 0.3, 0.2, 0.9], device="cuda")
    samples = [sampler.sample(logits, temperature=0.7, top_p=0.95) for _ in range(16)]
    assert len(samples) == 16
    assert torch.equal(torch.cuda.get_rng_state(), target_before)


def test_greedy_sampler_is_exact_and_validates_sampling_controls():
    sampler = MTPDraftSampler(seed=4, device=torch.device("cpu"))
    logits = torch.tensor([-2.0, 5.0, 3.0])
    assert sampler.sample(logits, temperature=0.0) == 1
    assert sampler.sample(logits, temperature=1.0, top_k=1) == 1
    with pytest.raises(ValueError, match="temperature"):
        sampler.sample(logits, temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        sampler.sample(logits, temperature=1.0, top_p=0.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        sampler.sample(logits.view(1, -1), temperature=1.0)


def test_device_native_sampling_matches_the_host_int_form_on_every_branch():
    """``sample`` is ``int(sample_device)``; drafting uses the latter to skip the per-token
    sync, so the two forms must select identically and advance the generator identically."""
    logits = torch.tensor([0.2, 0.7, -0.1, 1.1, 0.3])
    branches = (
        {"temperature": 0.0},
        {"temperature": 1.0, "top_k": 1},
        {"temperature": 0.8},
        {"temperature": 0.8, "top_k": 3},
        {"temperature": 0.8, "top_p": 0.9},
        {"temperature": 0.8, "top_k": 4, "top_p": 0.9},
    )
    for params in branches:
        host = MTPDraftSampler(seed=77, device=torch.device("cpu"))
        device = MTPDraftSampler(seed=77, device=torch.device("cpu"))
        for _ in range(16):
            picked = device.sample_device(logits, **params)
            assert isinstance(picked, torch.Tensor)
            assert picked.device == device.device
            assert picked.dtype == torch.int64
            assert picked.numel() == 1
            assert host.sample(logits, **params) == int(picked)


def test_device_native_sampling_validates_the_same_controls():
    sampler = MTPDraftSampler(seed=4, device=torch.device("cpu"))
    logits = torch.tensor([-2.0, 5.0, 3.0])
    assert int(sampler.sample_device(logits, temperature=0.0)) == 1
    with pytest.raises(ValueError, match="temperature"):
        sampler.sample_device(logits, temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        sampler.sample_device(logits, temperature=1.0, top_p=0.0)
    with pytest.raises(ValueError, match="top_k"):
        sampler.sample_device(logits, temperature=1.0, top_k=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        sampler.sample_device(logits.view(1, -1), temperature=1.0)


def test_exact_prefix_acceptance_stops_at_first_mismatch():
    assert accepted_draft_prefix([1, 2, 3], [1, 2, 3]) == 3
    assert accepted_draft_prefix([1, 2, 3], [1, 9, 3]) == 1
    assert accepted_draft_prefix([1, 2], [9, 2]) == 0
    assert accepted_draft_prefix([], []) == 0


def test_shadow_metrics_report_components_and_projection_without_threshold():
    metrics = MTPShadowMetrics()
    metrics.record(proposal_ms=8.0, verification_ms=10.0, state_ms=2.0, draft=3, accepted=2)
    metrics.record(proposal_ms=12.0, verification_ms=14.0, state_ms=4.0, draft=3, accepted=1)
    report = metrics.report()
    assert report["steps"] == 2
    assert report["draft_tokens"] == 6
    assert report["accepted_tokens"] == 3
    assert report["mean_accepted_per_step"] == 1.5
    assert report["mean_proposal_ms"] == 10.0
    assert report["mean_verification_ms"] == 12.0
    assert report["mean_state_ms"] == 3.0
    assert report["projected_sequential_tokens_per_second"] == pytest.approx(100.0)
    assert "pass" not in report
    assert "threshold" not in report


@requires_cuda
def test_qsa_backend_captures_once_then_reexpands_saved_blocks(monkeypatch):
    from freetoken.layers.rotary import get_rope

    get_rope.cache_clear()
    fixture = Fixture(parsed_config(), num_pages=16, max_running_req=2)
    layer_id = 3
    attention = fixture.layer(layer_id, seed=81)
    generator = torch.Generator(device="cuda").manual_seed(82)
    rows = torch.randn(
        132,
        fixture.config.hidden_size,
        dtype=fixture.dtype,
        device=fixture.device,
        generator=generator,
    )
    request = fixture.req(0, 0, 130)
    attention.forward(rows[:130], fixture.batch([request], "prefill"))

    fixture.step(request)
    capture_batch = fixture.batch([request], "decode")
    capture_batch.mtp_qsa_capture_blocks = {}
    attention.forward(rows[130:131], capture_batch)
    slot = fixture.backend._idx_slot[layer_id]
    saved = capture_batch.mtp_qsa_capture_blocks
    assert set(saved) == {slot}
    assert saved[slot].shape == (1, fixture.backend.block_topk)

    fixture.step(request)
    reuse_batch = fixture.batch([request], "decode")
    reuse_batch.mtp_qsa_saved_blocks = saved

    def forbidden_select(*args, **kwargs):
        raise AssertionError("QSA rescored a recursive MTP proposal")

    monkeypatch.setattr(fixture.backend, "_select", forbidden_select)
    output = attention.forward(rows[131:132], reuse_batch)
    assert output.shape == (1, fixture.config.hidden_size)
    assert torch.isfinite(output).all()


def test_proposal_depth_and_callback_contract_are_bounded():
    engine = MTPProposalEngine(MTPDraftSampler(seed=1, device=torch.device("cpu")), max_depth=2)
    with pytest.raises(ValueError, match="at most 2"):
        engine.propose(
            initial_token=1,
            target_hidden=torch.zeros(1, 2),
            draft_tokens=3,
            embedding=lambda token: torch.zeros(1, 1),
            step=lambda *args, **kwargs: None,
            lm_head=lambda hidden: hidden,
            temperature=0.0,
        )

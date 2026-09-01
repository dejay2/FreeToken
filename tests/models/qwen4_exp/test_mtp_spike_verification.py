from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen4_exp.mtp_spike import (
    MTPIsolatedTargetVerifier,
    MTPVerificationState,
)

REQUIRED = (
    "kv",
    "qsa_index",
    "qsa_ring",
    "recurrent",
    "ple",
    "page_table",
)


def _state() -> MTPVerificationState:
    return MTPVerificationState(
        components={
            name: (torch.full((8,), index, dtype=torch.int32),)
            for index, name in enumerate(REQUIRED)
        },
        api_state={"position": 0, "visible_tokens": [], "request_status": "active"},
    )


def _step(token: int, state: MTPVerificationState) -> torch.Tensor:
    position = state.api_state["position"]
    for index, name in enumerate(REQUIRED):
        state.components[name][0][position] = token * 10 + index
    state.api_state["position"] += 1
    state.api_state["visible_tokens"].append(token)
    # An unambiguous deterministic next-token distribution affected by all state families.
    checksum = sum(int(state.components[name][0][position]) for name in REQUIRED)
    logits = torch.full((64,), -1000.0)
    logits[(checksum + token) % logits.numel()] = 1000.0
    return logits


def test_one_isolated_verification_step_is_exactly_the_baseline_step():
    live = _state()
    before = live.digest()
    baseline = live.clone()
    expected_logits = _step(7, baseline)

    verifier = MTPIsolatedTargetVerifier(live, max_candidate_tokens=4)
    result = verifier.verify([7], _step)

    assert torch.equal(result.logits[0], expected_logits)
    assert result.state_after(1).digest() == baseline.digest()
    assert live.digest() == before
    assert live.api_state == {"position": 0, "visible_tokens": [], "request_status": "active"}


def test_rejected_suffix_never_leaks_into_live_or_accepted_prefix_state():
    live = _state()
    before = live.digest()
    result = MTPIsolatedTargetVerifier(live, max_candidate_tokens=4).verify(
        [3, 5, 9], _step
    )
    accepted = result.state_after(1)
    expected = live.clone()
    _step(3, expected)

    assert accepted.digest() == expected.digest()
    assert accepted.api_state["visible_tokens"] == [3]
    assert result.state_after(3).api_state["visible_tokens"] == [3, 5, 9]
    assert live.digest() == before


def test_failure_after_scratch_mutation_restores_live_bit_identity():
    live = _state()
    before = live.digest()

    def failing(token, state):
        _step(token, state)
        raise RuntimeError("verification fault")

    with pytest.raises(RuntimeError, match="verification fault"):
        MTPIsolatedTargetVerifier(live, max_candidate_tokens=2).verify([11], failing)
    assert live.digest() == before


def test_every_target_state_family_is_required_and_audited():
    live = _state()
    for missing in REQUIRED:
        components = dict(live.components)
        components.pop(missing)
        with pytest.raises(ValueError, match=missing):
            MTPVerificationState(components=components, api_state={})

    clone = live.clone()
    for name in REQUIRED:
        assert clone.components[name][0].data_ptr() != live.components[name][0].data_ptr()
        assert torch.equal(clone.components[name][0], live.components[name][0])


def test_candidate_depth_and_state_prefix_access_are_bounded():
    live = _state()
    verifier = MTPIsolatedTargetVerifier(live, max_candidate_tokens=2)
    with pytest.raises(ValueError, match="at most 2"):
        verifier.verify([1, 2, 3], _step)
    with pytest.raises(ValueError, match="at least one"):
        verifier.verify([], _step)
    result = verifier.verify([1, 2], _step)
    with pytest.raises(IndexError, match="verified prefix"):
        result.state_after(3)
    with pytest.raises(IndexError, match="verified prefix"):
        result.state_after(0)

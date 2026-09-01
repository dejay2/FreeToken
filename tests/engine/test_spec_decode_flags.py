"""Integrated speculative decode is one resolved value, read off the shared config object.

``FREETOKEN_MTP_SPECULATE`` / ``FREETOKEN_MTP_SPEC_DEPTH`` resolve on ``EngineConfig``, which
``SchedulerConfig`` (and so ``ServerArgs``) inherits -- the Engine and the Scheduler hold the
SAME instance, and it is the sole argument to both ``create_kv_pool`` and ``kv_cost``. That is
what makes the KV pool factory and the KV budget model structurally unable to disagree about
the pending-ring width (design risk #1).

Default-off is the contract: with the flag unset, ``num_speculative_tokens`` is 0 and every
downstream size is today's.
"""

from __future__ import annotations

import pytest

from freetoken.engine.config import SpecDecodeConfig, resolve_spec_decode


def test_speculation_is_off_and_costs_nothing_by_default():
    spec = resolve_spec_decode({})
    assert spec.enabled is False
    assert spec.num_speculative_tokens == 0
    assert spec.batch_width == 1


def test_the_flag_turns_it_on_at_the_default_depth():
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"})
    assert spec.enabled is True
    assert spec.depth == 3
    assert spec.num_speculative_tokens == 3
    assert spec.batch_width == 4


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_depth_is_honoured_over_its_whole_range(depth):
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_DEPTH": str(depth)}
    )
    assert spec.num_speculative_tokens == depth
    # w = 1 + k must stay a capturable MTP verify width (2..4).
    assert spec.batch_width in (2, 3, 4)


def test_depth_alone_reserves_nothing_while_speculation_is_off():
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPEC_DEPTH": "3"})
    assert spec.depth == 3
    assert spec.num_speculative_tokens == 0


@pytest.mark.parametrize("raw", ["0", "4", "-1", "x", ""])
def test_a_depth_outside_one_to_three_is_rejected(raw):
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_DEPTH"):
        resolve_spec_decode({"FREETOKEN_MTP_SPEC_DEPTH": raw})


@pytest.mark.parametrize("raw", ["2", "true", "yes", ""])
def test_a_non_boolean_speculate_flag_is_rejected(raw):
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPECULATE"):
        resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": raw})


def test_integrated_speculation_refuses_to_share_the_box_with_the_shadow_observer():
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SHADOW"):
        resolve_spec_decode(
            {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SHADOW": "1"}
        )
    # the observer alone is untouched
    assert resolve_spec_decode({"FREETOKEN_MTP_SHADOW": "1"}).enabled is False


def test_the_engine_config_resolves_the_flags_once_for_engine_and_scheduler(monkeypatch):
    from freetoken.scheduler.config import SchedulerConfig
    from freetoken.distributed import DistributedInfo
    import torch

    monkeypatch.setenv("FREETOKEN_MTP_SPECULATE", "1")
    monkeypatch.setenv("FREETOKEN_MTP_SPEC_DEPTH", "2")
    config = SchedulerConfig(
        model_path="unused", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16
    )
    assert config.spec_decode == SpecDecodeConfig(enabled=True, depth=2)
    assert config.num_speculative_tokens == 2
    # memoized: the engine and the scheduler read the same object, never the env twice
    monkeypatch.setenv("FREETOKEN_MTP_SPEC_DEPTH", "3")
    assert config.num_speculative_tokens == 2


def test_an_unset_environment_leaves_the_engine_config_at_zero(monkeypatch):
    from freetoken.scheduler.config import SchedulerConfig
    from freetoken.distributed import DistributedInfo
    import torch

    monkeypatch.delenv("FREETOKEN_MTP_SPECULATE", raising=False)
    monkeypatch.delenv("FREETOKEN_MTP_SPEC_DEPTH", raising=False)
    config = SchedulerConfig(
        model_path="unused", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16
    )
    assert config.num_speculative_tokens == 0
    assert config.spec_decode.enabled is False

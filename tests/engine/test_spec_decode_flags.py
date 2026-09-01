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

import torch
from types import SimpleNamespace

from freetoken.core import Batch
from freetoken.engine.config import SpecDecodeConfig, resolve_spec_decode

_CUDA = torch.cuda.is_available()


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


# ------------------------------------------------------------------ the engine's capture path
#
# Integrated speculation feeds its draft head the same (hidden, embeddings) pair the shadow
# observer captures, so a spec-enabled boot has to take the eager capture path for the ordinary
# forwards it still runs -- and, just as importantly, must not take it otherwise. With the flag
# off nothing here has a consumer, so ``forward_batch`` keeps its graph path byte for byte.


def _forward_engine(*, spec_draft=None):
    import torch

    from freetoken.core import Context, Req, SamplingParams, set_global_ctx
    from freetoken.engine.engine import Engine
    from freetoken.engine.sample import Sampler
    import freetoken.core as core

    core._GLOBAL_CTX = None
    ctx = Context(page_size=64)
    set_global_ctx(ctx)
    device = torch.device("cuda")
    engine = object.__new__(Engine)
    engine.ctx = ctx
    engine.device = device
    engine.stream = torch.cuda.current_stream()
    engine.mtp_shadow_observer = None
    engine.spec_draft = spec_draft
    engine.spec_sampler = None
    engine.spec_state_ladder = None
    engine.spec_graph_runner = None  # FREETOKEN_MTP_SPEC_GRAPH off: the eager capture path
    engine.cpu_moe_executor = None
    engine.sampler = Sampler(device=device, vocab_size=4)
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: False)

    calls: list[str] = []

    def _forward():
        calls.append("forward")
        return torch.zeros(1, 4, device=device)

    def _capture(*, all_row_logits=False):
        calls.append("capture")
        return (
            torch.zeros(1, 4, device=device),
            torch.zeros(1, 8, device=device),
            torch.zeros(1, 2, device=device),
        )

    engine.model = SimpleNamespace(forward=_forward, forward_mtp_capture=_capture)

    req = Req(
        input_ids=torch.arange(4, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=4,
        uid=1,
        sampling_params=SamplingParams(max_tokens=4),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = req.input_ids.to(device)
    return engine, batch, req, calls


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_the_flag_off_forward_keeps_the_ordinary_path():
    engine, batch, _req, calls = _forward_engine()
    engine.forward_batch(batch, engine.sampler.prepare(batch))
    assert calls == ["forward"]


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_spec_enabled_forward_captures_and_feeds_the_draft_head():
    seen: list = []
    draft = SimpleNamespace(
        observe_forward=lambda batch, capture, token: seen.append(
            (batch, tuple(t.shape for t in capture), int(token))
        )
    )
    engine, batch, _req, calls = _forward_engine(spec_draft=draft)

    engine.forward_batch(batch, engine.sampler.prepare(batch))

    assert calls == ["capture"]
    (observed_batch, shapes, token), = seen
    assert observed_batch is batch
    # (logits, multi_stream, inputs_embeds) -- exactly the observer's triple
    import torch

    assert shapes == (torch.Size([1, 4]), torch.Size([1, 8]), torch.Size([1, 2]))
    assert token == 0


def test_speculation_refuses_more_than_one_running_request_at_boot():
    """One running context in the draft head, one snapshot slot in the ladder, one stream per
    step in the sampler. A batch carrying two requests must not be where that is discovered."""
    from freetoken.engine.config import require_speculation_supported

    enabled = SpecDecodeConfig(enabled=True, depth=3)
    require_speculation_supported(SimpleNamespace(max_running_req=1, spec_decode=enabled))
    with pytest.raises(ValueError, match="max-running-requests 1"):
        require_speculation_supported(
            SimpleNamespace(max_running_req=2, spec_decode=enabled)
        )
    # off: any concurrency, no opinion
    require_speculation_supported(
        SimpleNamespace(max_running_req=4, spec_decode=SpecDecodeConfig())
    )


# --------------------------------------------------- FREETOKEN_MTP_SPEC_GRAPH (default off)


def test_the_verify_graph_is_off_by_default_so_speculation_stays_eager():
    assert resolve_spec_decode({}).graph is False
    assert resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"}).graph is False


def test_the_verify_graph_flag_turns_capture_on():
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_GRAPH": "1"}
    )
    assert spec.graph is True
    # width 1 is the graphed capture-decode: the ordinary forward a spec-enabled boot still
    # runs, which without a graph of its own falls back to eager at half the decode rate.
    assert spec.graph_widths == (1, 2, 3, 4)


@pytest.mark.parametrize("depth,widths", [(1, (1, 2)), (2, (1, 2, 3)), (3, (1, 2, 3, 4))])
def test_only_the_widths_the_depth_can_actually_produce_are_captured(depth, widths):
    """A graph costs a warm-up forward and a slab of pinned buffers; capturing a width the
    configured depth can never emit would spend both for nothing."""
    spec = resolve_spec_decode(
        {
            "FREETOKEN_MTP_SPECULATE": "1",
            "FREETOKEN_MTP_SPEC_GRAPH": "1",
            "FREETOKEN_MTP_SPEC_DEPTH": str(depth),
        }
    )
    assert spec.graph_widths == widths


@pytest.mark.parametrize("raw", ["2", "yes", "true", ""])
def test_a_non_binary_graph_flag_is_rejected(raw):
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_GRAPH"):
        resolve_spec_decode({"FREETOKEN_MTP_SPEC_GRAPH": raw})


def test_the_graph_flag_alone_captures_nothing_while_speculation_is_off():
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPEC_GRAPH": "1"})
    assert spec.enabled is False
    assert spec.graph_widths == ()


# ------------------------------------------------------- the adaptive fallback's three knobs
#
# A speculative cycle costs ~1.9 plain steps, so a draft that stops predicting the target is a
# net LOSS. The fallback is part of speculation, not a second feature behind its own flag:
# turning it off is ``FREETOKEN_MTP_SPEC_MIN_EMITTED=0``, which restores always-speculate.


def test_the_fallback_defaults_are_the_measured_breakeven():
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"})
    assert spec.ema_alpha == pytest.approx(0.3)
    assert spec.min_emitted == pytest.approx(2.0)
    assert spec.cooldown == 16
    assert spec.adaptive is True


def test_the_fallback_is_inert_while_speculation_is_off():
    spec = resolve_spec_decode({})
    assert spec.adaptive is False
    assert spec.min_emitted == pytest.approx(2.0)  # parsed, but nothing consults it


def test_a_zero_threshold_restores_always_speculate():
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_MIN_EMITTED": "0"}
    )
    assert spec.min_emitted == 0.0
    assert spec.adaptive is False


def test_each_fallback_knob_is_read_from_its_own_variable():
    spec = resolve_spec_decode(
        {
            "FREETOKEN_MTP_SPECULATE": "1",
            "FREETOKEN_MTP_SPEC_EMA_ALPHA": "0.5",
            "FREETOKEN_MTP_SPEC_MIN_EMITTED": "2.5",
            "FREETOKEN_MTP_SPEC_COOLDOWN": "8",
        }
    )
    assert (spec.ema_alpha, spec.min_emitted, spec.cooldown) == (0.5, 2.5, 8)


def test_a_fresh_request_is_seeded_at_a_full_cycles_emission():
    """Optimistic by construction: early noise must not lock speculation out before the
    request has produced any evidence of its own."""
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"})
    assert spec.ema_seed == pytest.approx(4.0) == pytest.approx(spec.batch_width)
    narrow = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_DEPTH": "1"}
    )
    assert narrow.ema_seed == pytest.approx(2.0)


def test_the_cooldown_backoff_is_bounded():
    """A cold request must keep probing: content changes mid-stream, and a permanently
    cold request would never discover that its draft went hot again. The cap is 4x rather
    than 8x now that two consecutive probes, not one, are what re-cools."""
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_COOLDOWN": "16"}
    )
    assert spec.cooldown_cap == 64


def test_the_probe_is_judged_on_its_own_threshold_not_the_averages():
    """``min_emitted`` is a bar for a MEAN; a probe is one integer sample from the
    distribution that mean describes, so it gets its own -- lower -- bar."""
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"})
    assert spec.probe_resume == pytest.approx(2.0)
    tuned = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_PROBE_RESUME": "1.5"}
    )
    assert tuned.probe_resume == pytest.approx(1.5)


@pytest.mark.parametrize("raw", ["0", "-1", "x", "", "nan", "4.5"])
def test_a_probe_resume_outside_the_emittable_range_is_rejected(raw):
    """Zero is rejected too: a probe that resumes on any emission is not a probe. Turning
    the fallback off is still ``FREETOKEN_MTP_SPEC_MIN_EMITTED=0``."""
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_PROBE_RESUME"):
        resolve_spec_decode({"FREETOKEN_MTP_SPEC_PROBE_RESUME": raw})


@pytest.mark.parametrize("raw", ["0", "-0.1", "1.1", "x", "", "nan"])
def test_an_ema_alpha_outside_zero_to_one_is_rejected(raw):
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_EMA_ALPHA"):
        resolve_spec_decode({"FREETOKEN_MTP_SPEC_EMA_ALPHA": raw})


@pytest.mark.parametrize("raw", ["-1", "x", "", "nan", "4.5"])
def test_a_threshold_outside_zero_to_the_full_width_is_rejected(raw):
    """Above ``1 + depth`` no cycle could ever clear the bar, so speculation would go cold
    and never come back -- a configuration that silently means 'off'."""
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_MIN_EMITTED"):
        resolve_spec_decode({"FREETOKEN_MTP_SPEC_MIN_EMITTED": raw})


def test_the_threshold_ceiling_follows_the_configured_depth():
    env = {"FREETOKEN_MTP_SPEC_DEPTH": "1", "FREETOKEN_MTP_SPEC_MIN_EMITTED": "2"}
    assert resolve_spec_decode(env).min_emitted == pytest.approx(2.0)
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_MIN_EMITTED"):
        resolve_spec_decode({**env, "FREETOKEN_MTP_SPEC_MIN_EMITTED": "2.5"})


@pytest.mark.parametrize("raw", ["0", "-1", "x", "", "1.5"])
def test_a_cooldown_below_one_step_is_rejected(raw):
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_COOLDOWN"):
        resolve_spec_decode({"FREETOKEN_MTP_SPEC_COOLDOWN": raw})

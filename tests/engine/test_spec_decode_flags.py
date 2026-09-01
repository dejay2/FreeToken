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
    assert spec.depth == 5
    assert spec.num_speculative_tokens == 5
    assert spec.batch_width == 6


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_depth_is_honoured_over_its_whole_range(depth):
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_DEPTH": str(depth)}
    )
    assert spec.num_speculative_tokens == depth
    # w = 1 + k must stay a capturable MTP verify width; the width-generic
    # SpecVerifyGraphRunner captures 2..6, so the whole range is emittable.
    assert 2 <= spec.batch_width <= 6


def test_the_ceiling_and_the_default_are_both_five():
    """The default followed the ceiling only once the paired sweep proved it: depth 5 with
    the confidence cut AND the cost-aware bar beat depth 3 on numbers/code/8k, while the
    flat-bar depth 5 had regressed long context -11..-24% -- both companions are load-bearing."""
    from freetoken.engine.config import _DEFAULT_SPEC_DEPTH, _MAX_SPEC_DEPTH

    assert (_MAX_SPEC_DEPTH, _DEFAULT_SPEC_DEPTH) == (5, 5)
    assert resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"}).depth == 5
    assert SpecDecodeConfig().depth == 5


def test_depth_alone_reserves_nothing_while_speculation_is_off():
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPEC_DEPTH": "3"})
    assert spec.depth == 3
    assert spec.num_speculative_tokens == 0


@pytest.mark.parametrize("raw", ["0", "6", "7", "-1", "x", ""])
def test_a_depth_outside_one_to_five_is_rejected(raw):
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
    # the unset MIN_EMITTED default is the cut-armed breakeven (the cut is on by default),
    # clamped to the 1+depth ceiling at shallow depths
    assert config.spec_decode == SpecDecodeConfig(enabled=True, depth=2, min_emitted=2.4)
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


# ------------------------------------------------ the CPU MoE executor's verify-width sizing
#
# The cpu/hybrid decode target cuts its C++ scratch and pinned IO buffers to ``max_tokens``
# once, before graph capture. A verify step submits the whole w = 1 + depth row block in ONE
# forward while max_running_req is pinned to 1, so the plain decode bounds miss it entirely.


def _sizing_config(*, max_running_req=1, cuda_graph_max_bs=None, spec=SpecDecodeConfig()):
    return SimpleNamespace(
        max_running_req=max_running_req,
        cuda_graph_max_bs=cuda_graph_max_bs,
        spec_decode=spec,
    )


def test_the_cpu_moe_executor_is_sized_for_the_speculative_verify_width():
    from freetoken.engine.engine import _cpu_moe_executor_tokens

    # the candidate launcher's shape: one running request, no captured decode graph --
    # without the verify width the pool would be cut to a single row.
    spec = SpecDecodeConfig(enabled=True, depth=3)
    assert _cpu_moe_executor_tokens(_sizing_config(spec=spec)) == 4
    assert _cpu_moe_executor_tokens(
        _sizing_config(cuda_graph_max_bs=0, spec=spec)
    ) == 4


@pytest.mark.parametrize("depth,tokens", [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
def test_the_sizing_follows_the_configured_depth(depth, tokens):
    from freetoken.engine.engine import _cpu_moe_executor_tokens

    spec = SpecDecodeConfig(enabled=True, depth=depth)
    assert _cpu_moe_executor_tokens(_sizing_config(spec=spec)) == tokens


def test_the_plain_decode_bounds_still_win_when_they_are_wider():
    from freetoken.engine.engine import _cpu_moe_executor_tokens

    spec = SpecDecodeConfig(enabled=True, depth=3)
    assert _cpu_moe_executor_tokens(_sizing_config(max_running_req=8, spec=spec)) == 8
    assert _cpu_moe_executor_tokens(
        _sizing_config(cuda_graph_max_bs=16, spec=spec)
    ) == 16


def test_speculation_off_leaves_the_sizing_exactly_as_it_was():
    from freetoken.engine.engine import _cpu_moe_executor_tokens

    off = SpecDecodeConfig()  # batch_width 1: inert
    assert _cpu_moe_executor_tokens(_sizing_config(spec=off)) == 1
    assert _cpu_moe_executor_tokens(_sizing_config(max_running_req=4, spec=off)) == 4
    assert _cpu_moe_executor_tokens(
        _sizing_config(max_running_req=4, cuda_graph_max_bs=8, spec=off)
    ) == 8
    # depth alone reserves nothing while the flag is off
    assert _cpu_moe_executor_tokens(
        _sizing_config(spec=SpecDecodeConfig(enabled=False, depth=3))
    ) == 1


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
    assert spec.graph_widths == (1, 2, 3, 4, 5, 6)


@pytest.mark.parametrize(
    "depth,widths",
    [
        (1, (1, 2)),
        (2, (1, 2, 3)),
        (3, (1, 2, 3, 4)),
        (4, (1, 2, 3, 4, 5)),
        (5, (1, 2, 3, 4, 5, 6)),
    ],
)
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
    """The cut is armed by default, so the bar a cycle is held to is the CUT cycle's."""
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"})
    assert spec.ema_alpha == pytest.approx(0.3)
    assert spec.min_emitted == pytest.approx(2.4)
    assert spec.cooldown == 16
    assert spec.adaptive is True


def test_the_fallback_is_inert_while_speculation_is_off():
    spec = resolve_spec_decode({})
    assert spec.adaptive is False
    assert spec.min_emitted == pytest.approx(2.4)  # parsed, but nothing consults it


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
    assert spec.ema_seed == pytest.approx(6.0) == pytest.approx(spec.batch_width)
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


def test_the_bar_is_cost_aware_by_default():
    """A static bar is measurably wrong at one end of a long request whichever end it was
    tuned for: cycles roughly double in cost from short context to 8-11k while a plain step
    barely moves, so the breakeven doubles with them."""
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"})
    assert spec.cost_aware is True
    assert spec.min_emitted == pytest.approx(2.4)  # now the FLOOR of a measured bar


def test_the_cost_aware_bar_has_its_own_off_switch():
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_COST_AWARE": "0"}
    )
    assert spec.cost_aware is False
    # the fallback itself is untouched: min_emitted is simply the bar again, flat
    assert spec.adaptive is True
    assert spec.min_emitted == pytest.approx(2.4)


@pytest.mark.parametrize("raw", ["", "yes", "2", "true"])
def test_a_non_binary_cost_aware_flag_is_rejected(raw):
    with pytest.raises(ValueError, match="COST_AWARE"):
        resolve_spec_decode({"FREETOKEN_MTP_SPEC_COST_AWARE": raw})


def test_the_min_emitted_doc_says_it_is_the_floor_of_a_measured_bar():
    """The knob's MEANING changed with the cut-over -- a tuning pass that still reads it as a
    flat bar will mis-tune it -- so the comment that a tuner reads has to say so."""
    import inspect

    source = inspect.getsource(SpecDecodeConfig)
    note = source.split("min_emitted:")[0].rsplit("ema_alpha:", 1)[-1]
    assert "FLOOR" in note and "cost_aware" in note


def test_the_probe_is_judged_on_its_own_threshold_not_the_averages():
    """``min_emitted`` is a bar for a MEAN; a probe is one integer sample from the
    distribution that mean describes, so it gets its own -- lower -- bar."""
    spec = resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"})
    assert spec.probe_resume == pytest.approx(2.0)
    tuned = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_PROBE_RESUME": "1.5"}
    )
    assert tuned.probe_resume == pytest.approx(1.5)


@pytest.mark.parametrize("raw", ["0", "-1", "x", "", "nan", "6.5"])
def test_a_probe_resume_outside_the_emittable_range_is_rejected(raw):
    """Zero is rejected too: a probe that resumes on any emission is not a probe. Turning
    the fallback off is still ``FREETOKEN_MTP_SPEC_MIN_EMITTED=0``."""
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_PROBE_RESUME"):
        resolve_spec_decode({"FREETOKEN_MTP_SPEC_PROBE_RESUME": raw})


@pytest.mark.parametrize("raw", ["0", "-0.1", "1.1", "x", "", "nan"])
def test_an_ema_alpha_outside_zero_to_one_is_rejected(raw):
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_EMA_ALPHA"):
        resolve_spec_decode({"FREETOKEN_MTP_SPEC_EMA_ALPHA": raw})


@pytest.mark.parametrize("raw", ["-1", "x", "", "nan", "6.5"])
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


# ----------------------------------------------------------------------- the confidence cut
#
# The draft head's own top-1 confidence predicts the target's verdict (accepted drafts average
# 0.91, rejected 0.46), and verify width is what a cycle costs on this box -- so stopping the
# chain at the first doubtful row is the knob, and 0.8 is where it was measured.


def test_the_cut_is_armed_at_the_measured_bar_by_default():
    assert resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"}).conf_cut == pytest.approx(0.8)
    assert resolve_spec_decode({}).conf_cut == pytest.approx(0.8)


def test_the_cut_reads_its_own_variable():
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_CONF_CUT": "0.55"}
    )
    assert spec.conf_cut == pytest.approx(0.55)


def test_a_zero_cut_restores_always_full_depth_drafting():
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_CONF_CUT": "0"}
    )
    assert spec.conf_cut == 0.0


@pytest.mark.parametrize("raw", ["-0.1", "1.1", "2", "x", "", "nan"])
def test_a_cut_outside_a_probability_is_rejected(raw):
    """It is compared against a softmax probability: a bar above 1 no row could ever clear
    would silently mean 'always draft exactly one token'."""
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_CONF_CUT"):
        resolve_spec_decode({"FREETOKEN_MTP_SPEC_CONF_CUT": raw})


def test_the_unset_emission_bar_follows_whether_the_cut_is_armed():
    """A cut cycle verifies fewer rows, so it is cheaper, so the bar it has to clear to be
    worth taking is lower: ~2.0 modelled, 2.4 chosen conservatively, against 3.6 uncut."""
    armed = resolve_spec_decode({"FREETOKEN_MTP_SPECULATE": "1"})
    assert armed.min_emitted == pytest.approx(2.4)
    disabled = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_CONF_CUT": "0"}
    )
    assert disabled.min_emitted == pytest.approx(3.6)


def test_the_cut_aware_default_still_clamps_to_the_full_width_ceiling():
    """At depth 1 the ceiling (1 + depth = 2) binds below the cut-armed 2.4."""
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_DEPTH": "1"}
    )
    assert spec.min_emitted == pytest.approx(2.0)


# ------------------------------------------------ the raised ceiling: depths 4 and 5
#
# Every bound in ``resolve_spec_decode`` is written against ``1 + depth`` rather than a literal,
# so the deep depths should need no code of their own. These pin that -- a hard 4 anywhere would
# either reject a legal configuration at boot or, worse, accept an illegal one.


@pytest.mark.parametrize("depth,width", [(4, 5), (5, 6)])
def test_the_deep_depths_resolve_to_their_full_verify_width(depth, width):
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_DEPTH": str(depth)}
    )
    assert (spec.depth, spec.num_speculative_tokens, spec.batch_width) == (
        depth,
        depth,
        width,
    )
    # the EMA seed is one full cycle's emission, whatever the width
    assert spec.ema_seed == pytest.approx(float(width))


@pytest.mark.parametrize("depth", [4, 5])
def test_the_unset_emission_bar_stops_clamping_once_the_width_clears_it(depth):
    """At depth 1-2 the ``1 + depth`` ceiling binds below the cut-armed 2.4; from depth 3 up
    the breakeven itself is the bar, and a deeper cycle does not raise it -- what a cycle
    costs is set by the rows the CUT lets it verify, not by the depth it was allowed."""
    spec = resolve_spec_decode(
        {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_DEPTH": str(depth)}
    )
    assert spec.min_emitted == pytest.approx(2.4)
    uncut = resolve_spec_decode(
        {
            "FREETOKEN_MTP_SPECULATE": "1",
            "FREETOKEN_MTP_SPEC_DEPTH": str(depth),
            "FREETOKEN_MTP_SPEC_CONF_CUT": "0",
        }
    )
    assert uncut.min_emitted == pytest.approx(3.6)


@pytest.mark.parametrize("depth,width", [(4, 5), (5, 6)])
def test_the_emission_and_probe_bounds_open_up_with_the_deeper_width(depth, width):
    """``0..1 + depth`` and ``(0, 1 + depth]``: exactly the full width is legal at the deeper
    depths (it was not at depth 3), and one step past it is still refused."""
    env = {"FREETOKEN_MTP_SPECULATE": "1", "FREETOKEN_MTP_SPEC_DEPTH": str(depth)}
    at_ceiling = resolve_spec_decode(
        {
            **env,
            "FREETOKEN_MTP_SPEC_MIN_EMITTED": str(float(width)),
            "FREETOKEN_MTP_SPEC_PROBE_RESUME": str(float(width)),
        }
    )
    assert at_ceiling.min_emitted == pytest.approx(float(width))
    assert at_ceiling.probe_resume == pytest.approx(float(width))
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_MIN_EMITTED"):
        resolve_spec_decode({**env, "FREETOKEN_MTP_SPEC_MIN_EMITTED": str(width + 0.1)})
    with pytest.raises(ValueError, match="FREETOKEN_MTP_SPEC_PROBE_RESUME"):
        resolve_spec_decode({**env, "FREETOKEN_MTP_SPEC_PROBE_RESUME": str(width + 0.1)})


@pytest.mark.parametrize("cut", ["0", "0.8"])
def test_an_explicit_emission_bar_wins_over_either_cut_default(cut):
    spec = resolve_spec_decode(
        {
            "FREETOKEN_MTP_SPECULATE": "1",
            "FREETOKEN_MTP_SPEC_CONF_CUT": cut,
            "FREETOKEN_MTP_SPEC_MIN_EMITTED": "3.1",
        }
    )
    assert spec.min_emitted == pytest.approx(3.1)

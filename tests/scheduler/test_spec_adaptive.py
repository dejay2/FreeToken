"""Adaptive speculation: stop speculating when the draft goes cold, probe to come back.

Measured on the live box (CUDA graphs on): a speculative cycle emits ~3.9 tokens on
predictable content (+16% over plain decode) but ~1.4-1.7 on prose -- a 32% LOSS, because the
fixed cycle cost is ~1.9 plain steps whatever the draft produces. The worst case has to become
"plain speed", so the dispatch consults a per-request acceptance EMA and falls back.

The seams here are the real ones: ``Scheduler._spec_candidate`` (which the loop already calls,
and whose None already means "decode plainly this step") and ``Scheduler._spec_record``, which
``_speculative_decode_step`` calls with the count it actually emitted. The end-to-end proof
that the cycle feeds that seam its real emitted count lives in test_spec_decode_loop.py.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.engine.config import SpecDecodeConfig
from freetoken.scheduler.scheduler import Scheduler, _SpecTimingProbe

CPU = torch.device("cpu")


def _stub(
    *, depth=3, min_emitted=2.0, probe_resume=2.0, cooldown=4, ema_alpha=0.5, ready=True,
    cost_aware=True,
):
    stub = Scheduler.__new__(Scheduler)
    stub.config = SimpleNamespace(
        spec_decode=SpecDecodeConfig(
            enabled=True,
            depth=depth,
            min_emitted=min_emitted,
            probe_resume=probe_resume,
            cooldown=cooldown,
            ema_alpha=ema_alpha,
            cost_aware=cost_aware,
        )
    )
    stub.engine = SimpleNamespace(spec_draft=SimpleNamespace(is_ready=lambda req: ready))
    stub.decode_manager = SimpleNamespace(running_reqs=set(), runnable=True)
    stub.prefill_manager = SimpleNamespace(runnable=False)
    stub.finished_reqs = set()
    return stub


def _req(stub, *, uid=1, prompt_len=8, output_len=4096, running=True):
    """A request parked exactly where the loop offers it to a speculative step."""
    req = Req(
        input_ids=torch.arange(1, prompt_len + 1, dtype=torch.int32),
        table_idx=0,
        cached_len=prompt_len - 1,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=None,
    )
    if running:
        stub.decode_manager.running_reqs = {req}
    return req


def _plain_steps_until_probe(stub, req, *, limit=64) -> int:
    plain = 0
    while stub._spec_candidate() is None:
        plain += 1
        assert plain <= limit, "the cooldown never allowed another probe"
    return plain


def _fail_a_probe_pair(stub, req) -> int:
    """Wait out the cooldown, then fail BOTH probes -- what it now takes to re-cool.

    Returns the plain steps waited before the first of them, and asserts on the way through
    that one bad probe alone did not re-cool.
    """
    gap = _plain_steps_until_probe(stub, req)
    stub._spec_record(req, 1)
    assert stub._spec_candidate() is req, "one bad probe must not re-cool on its own"
    stub._spec_record(req, 1)
    return gap


def _dispatches(stub, req, emissions):
    """Drive the scripted acceptance sequence through the real dispatch seam.

    Each element is what a cycle WOULD emit; a step the policy declines consumes none of
    them (it is a plain decode). Returns one bool per loop iteration: did we speculate?
    """
    decisions: list[bool] = []
    pending = list(emissions)
    while pending:
        speculated = stub._spec_candidate() is req
        decisions.append(speculated)
        if speculated:
            stub._spec_record(req, pending.pop(0))
    return decisions


# ------------------------------------------------------------------------------- the EMA


def test_a_fresh_request_starts_optimistic_and_speculates():
    stub = _stub()
    req = _req(stub)
    assert stub._spec_candidate() is req
    assert stub._spec_policy(req).ema == pytest.approx(4.0)  # 1 + depth


def test_the_ema_follows_the_emitted_counts():
    stub = _stub(ema_alpha=0.5)
    req = _req(stub)
    policy = stub._spec_policy(req)
    for emitted, expected in ((2, 3.0), (2, 2.5), (3, 2.75), (1, 1.875)):
        stub._spec_record(req, emitted)
        assert policy.ema == pytest.approx(expected)


def test_a_request_is_fresh_per_request_not_per_scheduler():
    """Content changes at request boundaries, so a new request inherits nothing."""
    stub = _stub()
    cold = _req(stub, uid=1)
    stub._spec_record(cold, 1)
    stub._spec_record(cold, 1)
    assert stub._spec_candidate() is None

    fresh = _req(stub, uid=2)  # the old request left; this one is running now
    assert stub._spec_candidate() is fresh
    assert stub._spec_policy(fresh).ema == pytest.approx(4.0)


def test_two_requests_keep_independent_acceptance():
    stub = _stub()
    a, b = _req(stub, uid=1), _req(stub, uid=2, running=False)
    stub.decode_manager.running_reqs = {a, b}
    for _ in range(4):
        stub._spec_record(a, 1)
    assert stub._spec_policy(a).ema < 2.0
    assert stub._spec_policy(b).ema == pytest.approx(4.0)
    assert stub._spec_policy(a) is not stub._spec_policy(b)


def test_a_departed_requests_policy_is_not_kept_forever():
    stub = _stub()
    gone = _req(stub, uid=1)
    stub._spec_record(gone, 1)
    live = _req(stub, uid=2)  # running_reqs no longer holds uid 1
    stub._spec_policy(live)
    assert set(stub._spec_policies) == {2}


# ---------------------------------------------------------------------- the cold threshold


def test_a_cold_ema_stops_the_dispatch():
    stub = _stub(ema_alpha=0.5)
    req = _req(stub)
    stub._spec_record(req, 1)  # 4.0 -> 2.5, still above the bar
    assert stub._spec_candidate() is req
    stub._spec_record(req, 1)  # -> 1.75
    assert stub._spec_candidate() is None


def test_the_threshold_is_the_ema_not_a_single_bad_cycle():
    """One rejected cycle inside a good run is noise; the fallback must not chase it."""
    stub = _stub(ema_alpha=0.3)
    req = _req(stub)
    for emitted in (4, 4, 1, 4, 4):
        assert stub._spec_candidate() is req
        stub._spec_record(req, emitted)
    assert stub._spec_candidate() is req


def test_the_cooldown_counts_plain_steps_then_allows_a_probe_pair():
    stub = _stub(cooldown=4, ema_alpha=0.5)
    req = _req(stub)
    stub._spec_record(req, 1)
    stub._spec_record(req, 1)  # cold

    assert [stub._spec_candidate() is req for _ in range(4)] == [False] * 4
    assert stub._spec_candidate() is req  # the first probe
    stub._spec_record(req, 1)
    # one unlucky probe is a single integer sample, not a verdict: the second follows
    # immediately rather than doubling the cooldown behind it
    assert stub._spec_candidate() is req
    stub._spec_record(req, 1)
    assert stub._spec_candidate() is None


def test_the_probe_resumes_on_its_own_threshold_not_the_averages():
    """``min_emitted`` is a bar for a MEAN. Judging one sample from that distribution by the
    mean's bar rejects roughly half of content that is comfortably worth speculating on."""
    stub = _stub(cooldown=2, min_emitted=3.0, probe_resume=2.0, ema_alpha=0.5)
    req = _req(stub)
    stub._spec_record(req, 1)  # 4.0 -> 2.5, below min_emitted
    _plain_steps_until_probe(stub, req)

    stub._spec_record(req, 2)  # below min_emitted, at probe_resume

    assert stub._spec_candidate() is req
    assert stub._spec_policy(req).ema == pytest.approx(2.0)


def test_the_second_probe_rescues_a_run_the_first_probe_missed():
    stub = _stub(cooldown=2, ema_alpha=0.5)
    req = _req(stub)
    stub._spec_record(req, 1)
    stub._spec_record(req, 1)
    _plain_steps_until_probe(stub, req)

    stub._spec_record(req, 1)  # first probe: unlucky
    assert stub._spec_candidate() is req
    stub._spec_record(req, 4)  # second probe: the content was fine all along

    assert stub._spec_candidate() is req
    assert stub._spec_policy(req).ema == pytest.approx(4.0)


def test_alternating_full_and_single_token_cycles_keep_speculating():
    """The live failure, encoded. Cycles that alternate a full run with a single token have a
    mean well clear of the bar, but every other probe sample lands under it -- and a
    single-sample verdict with doubling backoff turned that into 16 + 32 + 64 plain-step
    stretches on content that should have been speculating throughout."""
    stub = _stub(cooldown=4, ema_alpha=0.3, min_emitted=2.0)
    req = _req(stub)
    script = [1, 1, 1, 1] + [1, 4] * 10  # go cold, then alternate around a mean of 2.5

    decisions = _dispatches(stub, req, script)

    # one cooldown, then the probe pair resumes; no backoff ladder
    assert decisions.count(False) == 4
    assert stub._spec_policy(req).ema > 2.0


def test_a_good_probe_resumes_speculation():
    stub = _stub(cooldown=2, ema_alpha=0.5)
    req = _req(stub)
    stub._spec_record(req, 1)
    stub._spec_record(req, 1)
    for _ in range(2):
        assert stub._spec_candidate() is None
    assert stub._spec_candidate() is req  # the probe

    stub._spec_record(req, 4)  # the draft went hot again
    # the probe is judged on its OWN emission and replaces the stale average outright: the
    # EMA it inherited (1.75) would have vetoed a probe that just emitted a full run
    assert stub._spec_policy(req).ema == pytest.approx(4.0)
    assert stub._spec_candidate() is req
    assert stub._spec_candidate() is req  # and stays hot, with no cooldown in between


def test_a_bad_probe_backs_off_within_a_bounded_cooldown():
    stub = _stub(cooldown=2, ema_alpha=0.5)
    req = _req(stub)
    stub._spec_record(req, 1)
    stub._spec_record(req, 1)

    gaps = [_fail_a_probe_pair(stub, req) for _ in range(6)]

    # doubling, capped at cooldown_cap (4 * 2) -- a cold request never stops probing
    assert gaps == [2, 4, 8, 8, 8, 8]


def test_a_resumption_clears_the_backoff():
    stub = _stub(cooldown=2, ema_alpha=0.5)
    req = _req(stub)
    stub._spec_record(req, 1)
    stub._spec_record(req, 1)
    for _ in range(3):  # failed probe pairs push the cooldown out to the cap
        _fail_a_probe_pair(stub, req)
    _plain_steps_until_probe(stub, req)
    stub._spec_record(req, 4)  # a good probe

    stub._spec_record(req, 1)  # ... then straight back to cold
    stub._spec_record(req, 1)
    # the base cooldown, not the backed-off one
    assert _plain_steps_until_probe(stub, req) == 2


def test_the_structural_misses_do_not_burn_the_cooldown():
    """An unprimed draft head or a lagging host is not evidence about acceptance, and a
    step it declines is not a step the policy asked to spend plainly."""
    stub = _stub(cooldown=2, ema_alpha=0.5)
    req = _req(stub)
    stub._spec_record(req, 1)
    stub._spec_record(req, 1)
    stub.engine.spec_draft.is_ready = lambda r: False
    for _ in range(5):
        assert stub._spec_candidate() is None
    stub.engine.spec_draft.is_ready = lambda r: True
    assert [stub._spec_candidate() is req for _ in range(3)] == [False, False, True]


# ------------------------------------------------------------------ the off switch is exact


_SCRIPT = [4, 1, 1, 1, 1, 2, 1, 1, 4, 4, 1, 1, 1, 1, 1, 3]


def test_a_zero_threshold_reproduces_always_speculate():
    stub = _stub(min_emitted=0.0)
    req = _req(stub)
    assert _dispatches(stub, req, _SCRIPT) == [True] * len(_SCRIPT)
    # nothing was even constructed: the fallback is not on the path
    assert getattr(stub, "_spec_policies", None) in (None, {})


def test_the_same_script_does_fall_back_at_the_default_threshold():
    """The paired half of the test above: the script is one a live prose request produces,
    and the default policy must NOT dispatch every step of it."""
    stub = _stub(min_emitted=2.0, ema_alpha=0.3, cooldown=4)
    req = _req(stub)
    decisions = _dispatches(stub, req, _SCRIPT)
    assert decisions.count(True) == len(_SCRIPT)
    assert decisions.count(False) > 0


# ------------------------------------------------------------------- the cost-aware bar
#
# The static bar cannot be right at both ends of a long request. Measured on the live box: a
# speculative cycle costs ~28-45 ms at short context and ~70-100+ ms at 8-11k (verify AND draft
# both scale with the KV they read) while a plain step only goes ~15 -> ~20 ms, so the true
# breakeven roughly DOUBLES over one request. A bar pinned at the short-context number spends
# where cycles are dear: +11-13% at short context but -11..-24% at 8k on depth 5, ~-10% at 11k
# on depth 3. So the policy divides its own measured cycle by its own measured plain step.
#
# The seam is deliberately a value, not a clock: ``record_plain_ms`` / ``record_cycle_ms`` take
# the duration the caller measured, so nothing here monkeypatches time. That the loop feeds
# them the RIGHT durations, at points that add no device sync, is test_spec_decode_loop.py's.


def _warm(policy, *, plain_ms, cycle_ms, samples=3):
    """The minimum evidence the measured bar is allowed to act on, of each kind. A constant
    sample makes both EMAs equal that constant whatever alpha is."""
    for _ in range(samples):
        policy.record_plain_ms(plain_ms)
        policy.record_cycle_ms(cycle_ms)
    return policy


def test_the_bar_is_what_a_cycle_costs_in_plain_steps():
    stub = _stub(min_emitted=2.0, depth=3)
    policy = _warm(stub._spec_policy(_req(stub)), plain_ms=20.0, cycle_ms=70.0)
    assert policy.bar == pytest.approx(3.5)


def test_the_bar_is_the_static_floor_until_both_clocks_are_warm():
    """Two samples of each is not evidence; three is the bar this ships with."""
    stub = _stub(min_emitted=2.0)
    policy = _warm(stub._spec_policy(_req(stub)), plain_ms=20.0, cycle_ms=70.0, samples=2)
    assert policy.bar == pytest.approx(2.0)
    policy.record_plain_ms(20.0)
    policy.record_cycle_ms(70.0)
    assert policy.bar == pytest.approx(3.5)


def test_a_request_that_never_decoded_plainly_keeps_the_static_bar():
    """Speculating from the very first step leaves nothing to divide by."""
    stub = _stub(min_emitted=2.0)
    policy = stub._spec_policy(_req(stub))
    for _ in range(8):
        policy.record_cycle_ms(90.0)
    assert policy.plain_samples == 0
    assert policy.bar == pytest.approx(2.0)


def test_a_zero_plain_time_never_divides():
    """A coarse clock can hand back two identical timestamps; the bar is not infinity."""
    stub = _stub(min_emitted=2.0)
    policy = _warm(stub._spec_policy(_req(stub)), plain_ms=0.0, cycle_ms=70.0)
    assert policy.bar == pytest.approx(2.0)


def test_a_cheap_cycle_leaves_the_bar_on_the_static_floor():
    """The measured ratio times wall clock only; ``min_emitted`` also carries what a cycle
    costs the rest of the batch, so it stays the floor."""
    stub = _stub(min_emitted=2.0)
    policy = _warm(stub._spec_policy(_req(stub)), plain_ms=20.0, cycle_ms=30.0)
    assert policy.bar == pytest.approx(2.0)


def test_a_dear_cycle_cannot_raise_the_bar_past_what_a_cycle_could_emit():
    """Above ``1 + depth`` no cycle could ever clear it -- that is "off", not "cost aware"."""
    stub = _stub(min_emitted=2.0, depth=3)
    policy = _warm(stub._spec_policy(_req(stub)), plain_ms=20.0, cycle_ms=200.0)
    assert policy.bar == pytest.approx(4.0)


def test_the_wall_time_emas_track_at_the_emission_alpha():
    """Content and context shift together, so both sides move at the same rate."""
    stub = _stub(ema_alpha=0.5)
    policy = stub._spec_policy(_req(stub))
    policy.record_plain_ms(20.0)
    policy.record_plain_ms(40.0)
    policy.record_cycle_ms(60.0)
    policy.record_cycle_ms(100.0)
    assert (policy.plain_ms, policy.cycle_ms) == (pytest.approx(30.0), pytest.approx(80.0))


def test_a_dear_cycle_gates_off_an_ema_that_clears_the_static_bar():
    """The regression this exists for: at 8k the same emitted-EMA that paid at short context
    is a loss, and the static bar keeps spending on it."""
    stub = _stub(min_emitted=2.0, ema_alpha=0.5, depth=3)
    req = _req(stub)
    _warm(stub._spec_policy(req), plain_ms=20.0, cycle_ms=80.0)  # a cycle costs 4 plain steps
    assert stub._spec_policy(req).bar == pytest.approx(4.0)

    stub._spec_record(req, 3)  # 4.0 -> 3.5: twice the static bar, under the measured one

    assert stub._spec_candidate() is None


def test_the_same_ema_keeps_speculating_while_the_cycle_stays_cheap():
    """The paired half: identical emissions, cheap cycles, and the request stays hot."""
    stub = _stub(min_emitted=2.0, ema_alpha=0.5, depth=3)
    req = _req(stub)
    _warm(stub._spec_policy(req), plain_ms=20.0, cycle_ms=30.0)

    stub._spec_record(req, 3)

    assert stub._spec_candidate() is req


def test_plain_steps_during_a_cooldown_keep_moving_the_bar():
    """A cold request still decodes -- and its context still grows. The bar has to keep
    tracking that, or the probe at the end of the cooldown is judged by a stale price."""
    stub = _stub(min_emitted=2.0, ema_alpha=0.5, cooldown=4)
    req = _req(stub)
    policy = _warm(stub._spec_policy(req), plain_ms=40.0, cycle_ms=80.0)
    assert policy.bar == pytest.approx(2.0)
    stub._spec_record(req, 1)
    stub._spec_record(req, 1)  # cold

    for _ in range(2):  # the loop decodes plainly and reports each step it settles
        assert stub._spec_candidate() is None
        policy.record_plain_ms(20.0)

    assert policy.plain_ms == pytest.approx(25.0)
    assert policy.bar == pytest.approx(3.2)


# ------------------------------------------------------------ the cost-aware off switch


def test_cost_aware_off_ignores_the_clocks_entirely():
    stub = _stub(min_emitted=2.0, cost_aware=False)
    policy = _warm(stub._spec_policy(_req(stub)), plain_ms=20.0, cycle_ms=200.0)
    assert policy.bar == pytest.approx(2.0)


def test_cost_aware_off_reproduces_the_static_policy_step_for_step():
    """The A/B's control arm: fed the dearest cycles there are, the flat bar decides exactly
    what a policy that was never told any of it decides."""
    dear = _stub(min_emitted=2.0, ema_alpha=0.3, cooldown=4, cost_aware=False)
    dear_req = _req(dear)
    _warm(dear._spec_policy(dear_req), plain_ms=20.0, cycle_ms=200.0)
    blind = _stub(min_emitted=2.0, ema_alpha=0.3, cooldown=4, cost_aware=False)

    assert _dispatches(dear, dear_req, _SCRIPT) == _dispatches(blind, _req(blind), _SCRIPT)


def test_the_same_script_under_the_measured_bar_does_fall_back_further():
    """And the paired half again: cost-aware ON, the same dear cycles, and the policy spends
    strictly less of the script speculating."""
    dear = _stub(min_emitted=2.0, ema_alpha=0.3, cooldown=4)
    dear_req = _req(dear)
    _warm(dear._spec_policy(dear_req), plain_ms=20.0, cycle_ms=200.0)
    flat = _stub(min_emitted=2.0, ema_alpha=0.3, cooldown=4, cost_aware=False)

    measured = _dispatches(dear, dear_req, _SCRIPT)
    static = _dispatches(flat, _req(flat), _SCRIPT)

    assert measured.count(False) > static.count(False)


# ------------------------------------------------------------------------ stats visibility


def test_the_timing_probe_reports_the_ema_and_the_step_counts(caplog):
    stub = _stub(cooldown=2, ema_alpha=0.5)
    req = _req(stub)
    policy = stub._spec_policy(req)
    probe = _SpecTimingProbe(CPU)
    with caplog.at_level(logging.INFO):
        for _ in range(32):
            probe.start_cycle()
            stub._spec_record(req, 3)
            probe.finish_cycle(emitted=3, accepted=2, policy=policy)
    (record,) = [r for r in caplog.records if "spec timing" in r.getMessage()]
    message = record.getMessage()
    assert "ema 3.00" in message
    assert "spec/plain 32/0" in message


def test_the_timing_probe_reports_the_measured_bar_and_both_clocks(caplog):
    """A bar that has climbed away from ``min_emitted`` is the report saying "this request's
    cycles got dear", which is a different diagnosis from "this draft got worse"."""
    stub = _stub(cooldown=2, ema_alpha=0.5, min_emitted=2.0)
    req = _req(stub)
    policy = _warm(stub._spec_policy(req), plain_ms=20.0, cycle_ms=60.0)
    probe = _SpecTimingProbe(CPU)
    with caplog.at_level(logging.INFO):
        for _ in range(32):
            probe.start_cycle()
            stub._spec_record(req, 3)
            probe.finish_cycle(emitted=3, accepted=2, policy=policy)
    (record,) = [r for r in caplog.records if "spec timing" in r.getMessage()]
    message = record.getMessage()
    assert "bar 3.00" in message
    assert "cycle 60.0ms / plain 20.0ms" in message


def test_the_probe_survives_a_disabled_fallback(caplog):
    probe = _SpecTimingProbe(CPU)
    with caplog.at_level(logging.INFO):
        for _ in range(32):
            probe.start_cycle()
            probe.finish_cycle(emitted=3, accepted=2, policy=None)
    (record,) = [r for r in caplog.records if "spec timing" in r.getMessage()]
    assert "emitted/cycle 3.00" in record.getMessage()


def test_the_probe_reports_the_acceptance_split_a_decision_carries(caplog):
    decision = SimpleNamespace(filter_ms=2.0, decide_ms=4.0, sync_ms=30.0)
    probe = _SpecTimingProbe(CPU)
    with caplog.at_level(logging.INFO):
        for _ in range(32):
            probe.start_cycle()
            probe.finish_cycle(emitted=3, accepted=2, decision=decision)
    (record,) = [r for r in caplog.records if "spec timing" in r.getMessage()]
    message = record.getMessage()
    assert "'filter_ms': '2.0'" in message
    assert "'decide_ms': '4.0'" in message
    assert "'sync_ms': '30.0'" in message


def test_the_probe_reports_the_replay_split_the_graph_runner_measures(caplog):
    probe = _SpecTimingProbe(CPU)
    with caplog.at_level(logging.INFO):
        for _ in range(32):
            probe.start_cycle()
            probe.add_ms("replay.model", 12.0)
            probe.add_ms("replay.gpu", 3.5)
            probe.finish_cycle(emitted=3, accepted=2)

    assert probe.extra == {"replay.model": 32 * 12.0, "replay.gpu": 32 * 3.5}
    (record,) = [r for r in caplog.records if "spec timing" in r.getMessage()]
    message = record.getMessage()
    assert "replay {" in message
    assert "'replay.model': '12.0'" in message
    assert "'replay.gpu': '3.5'" in message


def test_a_dotted_sub_mark_does_not_eat_the_enclosing_stages_span():
    """The engine subdivides "verify+accept" from inside; the coarse stage must keep spanning
    the whole of it, or the report's stages stop summing to the cycle."""
    probe = _SpecTimingProbe(CPU)
    probe.start_cycle()
    probe.mark("verify.forward")
    probe.mark("verify.accept")
    probe.mark("verify+accept")

    subdivided = probe.stages["verify.forward"] + probe.stages["verify.accept"]
    assert probe.stages["verify+accept"] >= subdivided

"""Phase 5: the speculative step wired into the scheduler loop.

A cycle is ``draft k -> one w-row target forward -> accept -> emit -> settle -> feed the
draft``. It runs SYNCHRONOUSLY (design 6.2): the previous batch is drained before the step is
dispatched, which is what makes ``_prepare_spec_batch``'s "host ids caught up" guard hold and
what lets the draft consume the target hidden state of the row it just accepted.

The gate here is offline and end-to-end: a tiny deterministic fake target drives the REAL
``_prepare_spec_batch`` / ``SpecSampler`` / ``_emit_step_tokens`` / ``_rollback_spec_tokens``
seams, and the emitted stream must equal the same fake target decoded one row at a time --
across full acceptance, zero acceptance, partial acceptance, EOS, stop strings and the output
budget. What only Phase 6 can prove is that the REAL model's hidden states and logits behave;
everything between the model and the client is proven here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.config import SpecDecodeConfig
from freetoken.engine.sample import Sampler
from freetoken.engine.spec_sample import SpecSampler
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.scheduler import Scheduler

CPU = torch.device("cpu")
TABLE_WIDTH = 512
VOCAB = 16
EOS = 15
HIDDEN = 6


# --------------------------------------------------------------------------- the fake target
#
# Row i of any forward reads (token, position) and produces the distribution for position
# i + 1. Making the logits a pure function of (token, position) is what lets a w-row causal
# forward and w successive one-row forwards be compared token for token.


class _FakeTarget:
    """A deterministic 'model': argmax(logits(token, position)) is the next token."""

    def __init__(self, *, eos_at: int | None = None):
        self.eos_at = eos_at
        self.forwards: list[int] = []

    def next_token(self, token: int, position: int) -> int:
        if self.eos_at is not None and position >= self.eos_at:
            return EOS
        return (token * 7 + position * 3 + 1) % (VOCAB - 1)

    def row_logits(self, token: int, position: int) -> torch.Tensor:
        row = torch.full((VOCAB,), -8.0)
        row[self.next_token(token, position)] = 4.0
        row[(token + 1) % VOCAB] = 1.0  # a runner-up, so the row is not degenerate
        return row

    def forward_mtp_capture(self, *, all_row_logits: bool = False):
        """The engine's capture seam: all-row logits plus the target's hidden rows."""
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        ids = batch.input_ids.tolist()
        positions = batch.positions.tolist()
        self.forwards.append(len(ids))
        logits = torch.stack(
            [self.row_logits(int(t), int(p)) for t, p in zip(ids, positions)]
        )
        hidden = torch.stack(
            [torch.full((HIDDEN,), float(t) + 0.25 * float(p)) for t, p in zip(ids, positions)]
        )
        if not all_row_logits:
            logits = logits[-1:]
        return logits, hidden, hidden[:, :HIDDEN]


# ---------------------------------------------------------------------------- the fake draft


@dataclass(frozen=True)
class _Committed:
    tokens: tuple[int, ...]
    hidden: tuple[float, ...]  # the first element of each row the draft would consume


class _FakeDraft:
    """Proposes from ``policy`` and records what the cycle feeds back."""

    def __init__(self, target: _FakeTarget, *, policy: str = "perfect", cut_to=None):
        self.target = target
        self.policy = policy
        # what a confidence cut leaves behind: a proposal SHORTER than the depth it was asked
        # for. None = never cut, i.e. the always-full-width chain.
        self.cut_to = cut_to
        self.committed: list[_Committed] = []
        self.ready = True
        self.reset_uids: list[int] = []
        self.last_token: int | None = None
        self.last_position: int | None = None

    # the seam the scheduler dispatches on
    def is_ready(self, req: Req) -> bool:
        return self.ready

    def propose(self, req: Req, depth: int):
        from freetoken.engine.spec_draft import DraftProposal

        token = int(req.input_ids[-1])
        position = req.cached_len
        tokens: list[int] = []
        rows: list[torch.Tensor] = []
        if self.cut_to is not None:
            depth = min(depth, self.cut_to)
        for step in range(depth):
            if self.policy == "perfect":
                nxt = self.target.next_token(token, position)
            elif self.policy == "garbage":
                nxt = (self.target.next_token(token, position) + 1) % (VOCAB - 1)
            elif self.policy == "first-only":
                nxt = self.target.next_token(token, position) if step == 0 else 0
            else:  # pragma: no cover - guard
                raise ValueError(self.policy)
            row = torch.full((VOCAB,), -8.0)
            row[nxt] = 4.0
            rows.append(row)
            tokens.append(nxt)
            token, position = nxt, position + 1
        return DraftProposal(tokens=tuple(tokens), logits=torch.stack(rows))

    def commit(self, req: Req, *, hidden: torch.Tensor, token_ids) -> None:
        n = len(token_ids)
        assert hidden.shape[0] >= n, "the step's hidden rows must cover the emitted run"
        self.committed.append(
            _Committed(
                tuple(int(t) for t in token_ids),
                tuple(float(row[0]) for row in hidden[:n]),
            )
        )

    def reset_request(self, uid: int) -> None:
        self.reset_uids.append(uid)


# ------------------------------------------------------------------------------- the harness


class _RecordingLadder:
    """Stands in for ``SpecStateLadder`` so the CPU gate does not need the triton replay."""

    def __init__(self):
        self.calls: list[tuple] = []

    def begin(self, req: Req, batch: Batch) -> None:
        self.calls.append(("begin", req.cached_len, batch.emit_width))
        batch.spec_capture = self

    def rollback(self, req: Req, accepted: int) -> None:
        self.calls.append(("rollback", req.cached_len, req.device_len, accepted))


class _CharTokenizer:
    def decode(self, ids):
        return "".join(chr(i + 65) for i in ids)


def _scheduler(target, draft, *, enabled=True, depth=3, page_size=64, num_pages=16,
               eos=frozenset({EOS}), ladder=None):
    page_table = torch.zeros(2, TABLE_WIDTH, dtype=torch.int32)
    cm = CacheManager(num_pages, page_size, page_table, "naive")
    stub = Scheduler.__new__(Scheduler)
    stub.device = CPU
    stub.cache_manager = cm
    stub.token_pool = torch.zeros(2, TABLE_WIDTH + 1, dtype=torch.int32)
    stub.eos_token_ids = eos
    stub.toolcall_anchor_id = None
    stub.tokenizer = _CharTokenizer()
    stub.finished_reqs = set()
    stub.decode_manager = DecodeManager(page_size)
    stub.prefill_manager = SimpleNamespace(runnable=False, pending_list=[])
    stub.sent: list[list] = []
    stub.send_result = stub.sent.append
    stub.status_reporter = SimpleNamespace(report_batch=lambda *a, **k: None)
    stub.table_manager = SimpleNamespace(free=lambda idx: None)
    stub._forward_iter = 0
    spec = SpecDecodeConfig(enabled=enabled, depth=depth)
    stub.config = SimpleNamespace(page_size=page_size, spec_decode=spec, tp_info=None)
    stub.engine = SimpleNamespace(
        device=CPU,
        stream=None,
        page_table=page_table,
        linear_state_pool=None,
        model=target,
        ctx=_ctx_for(page_table),
        attn_backend=SimpleNamespace(prepare_metadata=lambda batch: None),
        sampler=Sampler(device=CPU, vocab_size=VOCAB),
        spec_sampler=SpecSampler(device=CPU, depth=depth),
        spec_draft=draft,
        spec_state_ladder=ladder,
        # eager, as FREETOKEN_MTP_SPEC_GRAPH=0 leaves it: the loop's contract is the same
        # either way, and the graph path has its own gate in tests/engine/test_spec_verify_graph
        spec_graph_runner=None,
        cpu_moe_executor=None,
        mtp_shadow_observer=None,
    )
    from freetoken.engine.engine import Engine

    stub.engine.speculative_decode_batch = (
        lambda batch, args, **kw: Engine.speculative_decode_batch(stub.engine, batch, args, **kw)
    )
    return stub


def _ctx_for(page_table):
    import freetoken.core as core
    from freetoken.core import Context

    from freetoken.core import set_global_ctx

    core._GLOBAL_CTX = None
    ctx = Context(page_size=64)
    ctx.page_table = page_table
    set_global_ctx(ctx)
    return ctx


@pytest.fixture(autouse=True)
def _no_ctx_leak():
    import freetoken.core as core

    yield
    core._GLOBAL_CTX = None


def _decode_req(stub, *, prompt_len=8, output_len=32, stop_strs=None, uid=1):
    """A request parked exactly where the loop hands it to a speculative step."""
    req = Req(
        input_ids=torch.arange(1, prompt_len + 1, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=output_len, stop_strs=stop_strs or []),
        cache_handle=stub.cache_manager.prefix_cache.match_prefix(
            torch.zeros(0, dtype=torch.int32)
        ).cuda_handle,
    )
    stub.token_pool[0, :prompt_len] = req.input_ids
    stub.cache_manager.allocate_paged([req])
    # the prefill's own sampled token, already appended by its (synchronous) drain
    first = stub.engine.model.next_token(int(req.input_ids[-1]), prompt_len - 1)
    req.complete_one()
    req.append_host(torch.tensor([first], dtype=torch.int32))
    stub.token_pool[0, prompt_len] = first
    stub.decode_manager.running_reqs = {req}
    return req


def _plain_decode(target, *, prompt_len=8, output_len=32, stop_strs=None,
                  page_size=64, num_pages=16, eos=frozenset({EOS})):
    """The reference: the same fake target decoded one row per step, through the real
    emission path, with the drain synchronous (the ``normal_loop`` shape)."""
    stub = _scheduler(target, _FakeDraft(target), page_size=page_size, num_pages=num_pages,
                      eos=eos)
    req = _decode_req(stub, prompt_len=prompt_len, output_len=output_len, stop_strs=stop_strs)
    msgs = [
        Scheduler._emit_step_tokens(
            stub, req, torch.tensor([int(req.input_ids[-1])], dtype=torch.int32)
        )
    ]
    # _decode_req already appended the prefill token; re-emit through the real path instead
    req.input_ids = req._ids_buf[: prompt_len + 1]
    while not msgs[-1].finished:
        stub.cache_manager.allocate_paged([req])
        position = req.device_len - 1
        token = target.next_token(int(stub.token_pool[0, position]), position)
        stub.token_pool[0, req.device_len] = token
        req.complete_one()
        msgs.append(
            Scheduler._emit_step_tokens(stub, req, torch.tensor([token], dtype=torch.int32))
        )
    with stub.cache_manager.lazy_free_region():
        stub._free_req_resources(req)  # what the drain does for a finished request
    return stub, req, msgs


def _run_spec(stub, req, *, cycles=None):
    msgs = []
    while req.can_decode and req in stub.decode_manager.running_reqs:
        stub._speculative_decode_step(req)
        msgs.extend(m for batch in stub.sent[-1:] for m in batch)
        if msgs and msgs[-1].finished:
            break
        if cycles is not None and len(msgs) >= cycles:
            break
    return msgs


def _tokens(msgs):
    return [t for m in msgs for t in m.next_tokens]


# ------------------------------------------------------------------------- dispatch predicate


def test_the_dispatch_is_off_while_the_flag_is_off():
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target), enabled=False)
    _decode_req(stub)
    assert stub._spec_dispatch_ready() is False


def test_the_dispatch_needs_a_runnable_decode_and_no_pending_prefill():
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    assert stub._spec_dispatch_ready() is False  # nothing running
    _decode_req(stub)
    assert stub._spec_dispatch_ready() is True
    stub.prefill_manager.runnable = True
    assert stub._spec_dispatch_ready() is False  # prefill wins, exactly as it always has


def test_the_dispatch_is_off_without_a_draft_head():
    target = _FakeTarget()
    stub = _scheduler(target, None)
    _decode_req(stub)
    assert stub._spec_dispatch_ready() is False


def test_the_candidate_predicate_is_the_requests_shape_not_the_batchs_phase():
    """Design 6.1/Phase 2: the spec batch is itself PREFILL-phase, so ``batch.is_decode``
    cannot be the predicate. The request being in decode shape is."""
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub)
    assert stub._spec_candidate() is req

    req.device_len += 1  # mid-prefill shape
    assert stub._spec_candidate() is None


def test_an_unprimed_draft_head_falls_back_to_plain_decode():
    target = _FakeTarget()
    draft = _FakeDraft(target)
    stub = _scheduler(target, draft)
    _decode_req(stub)
    draft.ready = False
    assert stub._spec_candidate() is None


def test_a_request_with_no_room_for_a_multi_token_run_falls_back():
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub, prompt_len=8, output_len=2)  # one token of budget left
    assert req.remain_len == 1
    assert stub._spec_candidate() is None


def test_a_request_whose_host_ids_lag_is_not_a_candidate():
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub)
    req.input_ids = req.input_ids[:-1]
    assert stub._spec_candidate() is None


# ----------------------------------------------------------------------------- the cycle


def test_one_cycle_emits_the_accepted_run_and_settles_the_request():
    target = _FakeTarget()
    draft = _FakeDraft(target)
    ladder = _RecordingLadder()
    stub = _scheduler(target, draft, ladder=ladder)
    req = _decode_req(stub, prompt_len=8)
    before = (req.cached_len, req.device_len)

    stub._speculative_decode_step(req)

    (msg,) = stub.sent[-1]
    assert len(msg.next_tokens) == 4  # 3 accepted drafts + the bonus token
    assert (req.cached_len, req.device_len) == (before[0] + 4, before[1] + 4)
    assert req.input_ids.numel() == req.device_len  # host stays caught up
    assert req.spec_inflight is None


def test_the_forward_is_one_w_row_pass_over_the_target():
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub)
    stub._speculative_decode_step(req)
    assert target.forwards == [4]  # k = 3 drafts -> w = 4 rows, one forward


def test_the_state_ladder_is_armed_before_the_forward_and_settled_after_the_lengths():
    target = _FakeTarget()
    ladder = _RecordingLadder()
    stub = _scheduler(target, _FakeDraft(target), ladder=ladder)
    req = _decode_req(stub, prompt_len=8)

    stub._speculative_decode_step(req)

    begin, rollback = ladder.calls
    assert begin == ("begin", 8, 4)
    # rollback runs on the SETTLED lengths, with the emitted run's row count
    assert rollback == ("rollback", 12, 13, 4)


@pytest.mark.parametrize("policy", ["perfect", "garbage", "first-only"])
def test_the_draft_head_is_fed_the_accepted_rows_hidden_states_and_tokens(policy):
    """Row i of the step read position ``cached_len + i`` and emitted the token at
    ``cached_len + i + 1``, so the pairs the draft commits are the step's OWN hidden rows
    0..n-1 -- the fake target stamps ``token + 0.25 * position`` into every row, which is
    what makes 'the right rows' checkable rather than merely 'n rows'."""
    target = _FakeTarget()
    draft = _FakeDraft(target, policy=policy)
    stub = _scheduler(target, draft)
    req = _decode_req(stub, prompt_len=8)
    row_inputs = [int(req.input_ids[-1])] + list(draft.propose(req, 3).tokens)

    stub._speculative_decode_step(req)

    (msg,) = stub.sent[-1]
    expected = tuple(
        float(row_inputs[i]) + 0.25 * (8 + i) for i in range(len(msg.next_tokens))
    )
    assert draft.committed == [_Committed(msg.next_tokens, expected)]


def test_a_wholly_rejected_cycle_still_emits_exactly_one_token():
    target = _FakeTarget()
    draft = _FakeDraft(target, policy="garbage")
    stub = _scheduler(target, draft)
    req = _decode_req(stub, prompt_len=8)

    stub._speculative_decode_step(req)

    (msg,) = stub.sent[-1]
    assert len(msg.next_tokens) == 1
    assert (req.cached_len, req.device_len) == (9, 10)


def test_a_partly_accepted_cycle_keeps_the_prefix_plus_the_correction():
    target = _FakeTarget()
    draft = _FakeDraft(target, policy="first-only")
    stub = _scheduler(target, draft)
    req = _decode_req(stub, prompt_len=8)

    stub._speculative_decode_step(req)

    (msg,) = stub.sent[-1]
    assert len(msg.next_tokens) == 2  # one accepted draft + the correction
    assert (req.cached_len, req.device_len) == (10, 11)


@pytest.mark.parametrize("k", [1, 2])
def test_a_cut_short_proposal_round_trips_the_whole_cycle(k):
    """A confidence cut hands the scheduler FEWER drafts than the configured depth. The step
    is already written for that (``depth`` is min'd with ``remain_len`` today), so nothing
    special happens: verify width is 1 + k, and a perfect short draft emits k + 1 tokens."""
    target = _FakeTarget()
    ladder = _RecordingLadder()
    stub = _scheduler(target, _FakeDraft(target, cut_to=k), depth=3, ladder=ladder)
    req = _decode_req(stub, prompt_len=8)

    stub._speculative_decode_step(req)

    assert target.forwards == [1 + k]  # the narrow verify, not the configured width of 4
    assert ladder.calls[0] == ("begin", 8, 1 + k)
    (msg,) = stub.sent[-1]
    assert len(msg.next_tokens) == k + 1  # k accepted drafts + the bonus token
    assert (req.cached_len, req.device_len) == (8 + k + 1, 9 + k + 1)
    assert req.input_ids.numel() == req.device_len
    assert req.spec_inflight is None


# ------------------------------------------------------ the raised depth ceiling (4 and 5)
#
# Nothing in the cycle is written per width: the batch builder derives ``w = 1 + k`` from the
# proposal, the write mapping walks ``emit_width``, the ladder's arena is sized from the config
# and acceptance is a cumprod over however many rows arrived. These run the WHOLE cycle at the
# new ceiling, because a width bound that only bites in one of those seams would otherwise show
# up first on a live boot.


@pytest.mark.parametrize("depth", [4, 5])
def test_a_full_width_cycle_at_the_raised_ceiling_round_trips(depth):
    target = _FakeTarget()
    ladder = _RecordingLadder()
    stub = _scheduler(target, _FakeDraft(target), depth=depth, ladder=ladder)
    req = _decode_req(stub, prompt_len=8)

    stub._speculative_decode_step(req)

    assert target.forwards == [1 + depth]  # k drafts -> one w-row forward
    assert ladder.calls[0] == ("begin", 8, 1 + depth)
    (msg,) = stub.sent[-1]
    assert len(msg.next_tokens) == depth + 1  # every draft accepted, plus the bonus
    assert (req.cached_len, req.device_len) == (9 + depth, 10 + depth)
    assert req.input_ids.numel() == req.device_len
    assert req.spec_inflight is None
    # the run lands at its own positions, one row per emitted token
    assert stub.token_pool[0, 9 : 10 + depth].tolist() == list(msg.next_tokens)
    assert ladder.calls[1] == ("rollback", 9 + depth, 10 + depth, depth + 1)


@pytest.mark.parametrize("depth", [4, 5])
@pytest.mark.parametrize("policy", ["perfect", "garbage", "first-only"])
def test_a_deep_cycle_emits_exactly_what_plain_decode_would_have(depth, policy):
    """The gate that matters: a deeper chain is a COST choice, so the emitted stream, the
    finish reason and the settled lengths must all be the plain decode's, whatever the draft
    guessed."""
    target = _FakeTarget()
    plain_stub, plain_req, plain = _plain_decode(target, prompt_len=8, output_len=20)

    spec_target = _FakeTarget()
    stub = _scheduler(spec_target, _FakeDraft(spec_target, policy=policy), depth=depth)
    req = _decode_req(stub, prompt_len=8, output_len=20)
    msgs = _run_spec(stub, req)

    assert _tokens(msgs) == _tokens(plain)[1:]
    assert msgs[-1].finished and msgs[-1].finish_reason == plain[-1].finish_reason
    assert req.input_ids.tolist() == plain_req.input_ids.tolist()
    assert (req.cached_len, req.device_len) == (plain_req.cached_len, plain_req.device_len)
    assert torch.equal(stub.cache_manager.free_slots, plain_stub.cache_manager.free_slots)


@pytest.mark.parametrize("depth", [4, 5])
def test_a_deep_cycle_near_the_budget_narrows_instead_of_overrunning(depth):
    """``depth = min(configured, remain_len)``: with fewer tokens of budget left than the
    ceiling, the cycle drafts narrower rather than forwarding rows whose sampled tokens would
    take the write mapping's -1 discard slot."""
    target = _FakeTarget()
    _, plain_req, plain = _plain_decode(target, prompt_len=8, output_len=4)

    spec_target = _FakeTarget()
    stub = _scheduler(spec_target, _FakeDraft(spec_target), depth=depth)
    req = _decode_req(stub, prompt_len=8, output_len=4)
    msgs = _run_spec(stub, req)

    assert max(spec_target.forwards) <= 1 + req.output_len
    assert _tokens(msgs) == _tokens(plain)[1:]
    assert msgs[-1].finish_reason == "length"


def test_a_deep_cut_short_proposal_still_verifies_only_what_was_drafted():
    """The confidence cut and the raised ceiling meet here: a depth-5 configuration whose cut
    left 2 drafts must forward 3 rows, not 6."""
    target = _FakeTarget()
    ladder = _RecordingLadder()
    stub = _scheduler(target, _FakeDraft(target, cut_to=2), depth=5, ladder=ladder)
    req = _decode_req(stub, prompt_len=8)

    stub._speculative_decode_step(req)

    assert target.forwards == [3]
    assert ladder.calls[0] == ("begin", 8, 3)
    (msg,) = stub.sent[-1]
    assert len(msg.next_tokens) == 3


def test_a_cut_short_cycle_emits_what_plain_decode_would_have():
    """The cut is a COST policy, never a correctness one: the target still decides every
    token, so a one-draft cycle's output is the plain decode's, token for token."""
    _, plain_req, plain = _plain_decode(_FakeTarget(), prompt_len=8, output_len=12)

    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target, cut_to=1))
    req = _decode_req(stub, prompt_len=8, output_len=12)
    msgs = _run_spec(stub, req)

    assert _tokens(msgs) == _tokens(plain)[1:]  # the prefill token is not a decode step
    assert req.input_ids.tolist() == plain_req.input_ids.tolist()


def test_the_sampled_run_lands_in_the_token_pool_at_its_own_positions():
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub, prompt_len=8)

    stub._speculative_decode_step(req)

    (msg,) = stub.sent[-1]
    assert stub.token_pool[0, 9:13].tolist() == list(msg.next_tokens)


def test_nothing_double_advances_the_device_length():
    """``forward_batch``'s ``complete_one`` loop must not run for a speculative step --
    ``_rollback_spec_tokens`` is the single length authority."""
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target, policy="garbage"))
    req = _decode_req(stub, prompt_len=8)
    stub._speculative_decode_step(req)
    assert (req.cached_len, req.device_len) == (9, 10)  # exactly one token, not two


# -------------------------------------------------------------------------- offline gate (a)


@pytest.mark.parametrize("policy", ["perfect", "garbage", "first-only"])
def test_greedy_output_is_token_for_token_the_plain_decode_of_the_same_target(policy):
    target = _FakeTarget()
    _, plain_req, plain = _plain_decode(target, prompt_len=8, output_len=20)

    spec_target = _FakeTarget()
    stub = _scheduler(spec_target, _FakeDraft(spec_target, policy=policy))
    req = _decode_req(stub, prompt_len=8, output_len=20)
    msgs = _run_spec(stub, req)

    assert _tokens(msgs) == _tokens(plain)[1:]  # the prefill token is not a decode step
    assert msgs[-1].finished and msgs[-1].finish_reason == plain[-1].finish_reason
    assert req.input_ids.tolist() == plain_req.input_ids.tolist()


@pytest.mark.parametrize("policy", ["perfect", "garbage", "first-only"])
def test_an_eos_inside_a_cycle_matches_the_plain_decode_ending(policy):
    target = _FakeTarget(eos_at=13)
    _, plain_req, plain = _plain_decode(target, prompt_len=8, output_len=20)
    assert plain[-1].finish_reason == "stop"

    spec_target = _FakeTarget(eos_at=13)
    stub = _scheduler(spec_target, _FakeDraft(spec_target, policy=policy))
    req = _decode_req(stub, prompt_len=8, output_len=20)
    msgs = _run_spec(stub, req)

    assert _tokens(msgs) == _tokens(plain)[1:]
    assert msgs[-1].finish_reason == "stop"
    assert req.input_ids.tolist() == plain_req.input_ids.tolist()


@pytest.mark.parametrize("output_len", [5, 6, 7, 8])
def test_the_output_budget_ends_the_run_exactly_where_plain_decode_does(output_len):
    target = _FakeTarget()
    _, plain_req, plain = _plain_decode(target, prompt_len=8, output_len=output_len)
    assert plain[-1].finish_reason == "length"

    spec_target = _FakeTarget()
    stub = _scheduler(spec_target, _FakeDraft(spec_target))
    req = _decode_req(stub, prompt_len=8, output_len=output_len)
    msgs = _run_spec(stub, req)

    assert _tokens(msgs) == _tokens(plain)[1:]
    assert msgs[-1].finish_reason == "length"
    assert req.input_ids.tolist() == plain_req.input_ids.tolist()


def test_a_stop_string_inside_a_cycle_matches_the_plain_decode_ending():
    # the char tokenizer maps id -> chr(id + 65), so a stop string is a run of ids
    target = _FakeTarget()
    _, plain_req, plain = _plain_decode(target, prompt_len=8, output_len=20, stop_strs=["I"])
    assert plain[-1].matched_stop == "I"

    spec_target = _FakeTarget()
    stub = _scheduler(spec_target, _FakeDraft(spec_target))
    req = _decode_req(stub, prompt_len=8, output_len=20, stop_strs=["I"])
    msgs = _run_spec(stub, req)

    assert _tokens(msgs) == _tokens(plain)[1:]
    assert msgs[-1].matched_stop == "I"
    assert req.input_ids.tolist() == plain_req.input_ids.tolist()


# -------------------------------------------------------------------------- offline gate (b)


@pytest.mark.parametrize("policy", ["perfect", "garbage", "first-only"])
@pytest.mark.parametrize("page_size,num_pages", [(64, 16), (1, 64)])
def test_lengths_and_pages_settle_equal_to_plain_decode_after_every_cycle(
    policy, page_size, num_pages
):
    target = _FakeTarget()
    plain_stub, plain_req, plain = _plain_decode(
        target, prompt_len=8, output_len=16, page_size=page_size, num_pages=num_pages
    )

    spec_target = _FakeTarget()
    stub = _scheduler(spec_target, _FakeDraft(spec_target, policy=policy),
                      page_size=page_size, num_pages=num_pages)
    req = _decode_req(stub, prompt_len=8, output_len=16)
    _run_spec(stub, req)

    assert (req.cached_len, req.device_len) == (plain_req.cached_len, plain_req.device_len)
    assert torch.equal(
        stub.engine.page_table[req.table_idx], plain_stub.engine.page_table[req.table_idx]
    )
    assert torch.equal(stub.cache_manager.free_slots, plain_stub.cache_manager.free_slots)


# ------------------------------------------------------------------- the adaptive fallback
#
# The unit gate for the policy is tests/scheduler/test_spec_adaptive.py. What is owed HERE is
# that the real cycle feeds it the count it really emitted, and that a request whose draft is
# useless converges to plain decoding without changing a single token of its output.


def _run_adaptive(stub, req, target, *, limit=64):
    """The loop's real alternation: speculate when ``_spec_candidate`` offers the request,
    decode one row plainly when it does not."""
    msgs, dispatches = [], []
    while req.can_decode and req in stub.decode_manager.running_reqs:
        candidate = stub._spec_candidate()
        dispatches.append(candidate is req)
        if candidate is req:
            stub._speculative_decode_step(req)
            msgs.extend(m for batch in stub.sent[-1:] for m in batch)
        else:
            stub.cache_manager.allocate_paged([req])
            position = req.device_len - 1
            token = target.next_token(int(stub.token_pool[0, position]), position)
            stub.token_pool[0, req.device_len] = token
            req.complete_one()
            msgs.append(
                Scheduler._emit_step_tokens(
                    stub, req, torch.tensor([token], dtype=torch.int32)
                )
            )
            if msgs[-1].finished:
                with stub.cache_manager.lazy_free_region():
                    stub.decode_manager.remove_req(req)
                    stub._free_req_resources(req)
        if msgs and msgs[-1].finished:
            break
        assert len(dispatches) <= limit, "the run never terminated"
    return msgs, dispatches


def test_a_garbage_draft_converges_to_plain_decoding_with_identical_output():
    target = _FakeTarget()
    _, plain_req, plain = _plain_decode(target, prompt_len=8, output_len=40)

    spec_target = _FakeTarget()
    stub = _scheduler(spec_target, _FakeDraft(spec_target, policy="garbage"))
    stub.config.spec_decode = SpecDecodeConfig(
        enabled=True, depth=3, min_emitted=2.0, cooldown=4, ema_alpha=0.5
    )
    req = _decode_req(stub, prompt_len=8, output_len=40)

    msgs, dispatches = _run_adaptive(stub, req, spec_target)

    assert _tokens(msgs) == _tokens(plain)[1:]  # the prefill token is not a decode step
    assert msgs[-1].finished and msgs[-1].finish_reason == plain[-1].finish_reason
    assert req.input_ids.tolist() == plain_req.input_ids.tolist()
    # a wholly rejected cycle emits 1, so at alpha 0.5 the seed (4.0) crosses 2.0 on the
    # second cycle -- and the cooldown then buys four plain steps
    assert dispatches[:2] == [True, True]
    assert dispatches[2:6] == [False] * 4
    # and the fallback dominates the rest of the run
    assert sum(dispatches) < len(dispatches) / 3


def test_a_perfect_draft_never_falls_back():
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    stub.config.spec_decode = SpecDecodeConfig(
        enabled=True, depth=3, min_emitted=2.0, cooldown=4, ema_alpha=0.5
    )
    req = _decode_req(stub, prompt_len=8, output_len=40)

    _msgs, dispatches = _run_adaptive(stub, req, target)

    assert all(dispatches)


# --------------------------------------------------------- the cost-aware bar's two clocks
#
# The policy's arithmetic is tests/scheduler/test_spec_adaptive.py's. What is owed HERE is that
# the loop feeds it the two wall times it divides -- a real cycle's, and a real plain decode
# step's -- from points that add no device sync of their own, and that an interval which
# straddled something other than a plain step is thrown away rather than priced.


class _Clock:
    """A ``perf_counter`` that advances a fixed amount per READING, so any span the code under
    test measures with one pair of calls is exactly ``step_ms`` -- and a span that took an
    extra reading is visibly not."""

    def __init__(self, step_ms: float) -> None:
        self.step_ms = step_ms
        self.now = 0.0

    def __call__(self) -> float:
        reading, self.now = self.now, self.now + self.step_ms / 1e3
        return reading


def _fake_clock(monkeypatch, step_ms: float) -> _Clock:
    """Rebind the scheduler module's own ``time``, not the stdlib's: nothing else in the
    process is timing anything with this."""
    from freetoken.scheduler import scheduler as sched_mod

    clock = _Clock(step_ms)
    monkeypatch.setattr(sched_mod, "time", SimpleNamespace(perf_counter=clock))
    return clock


def _timed_stub(monkeypatch, *, step_ms=30.0, cost_aware=True, min_emitted=2.0):
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    stub.config.spec_decode = SpecDecodeConfig(
        enabled=True, depth=3, min_emitted=min_emitted, cost_aware=cost_aware
    )
    # pin the two diagnosis flags off, so the cycle's readings are the timing hook's alone
    stub._spec_probe = None
    stub._spec_conf = None
    _fake_clock(monkeypatch, step_ms)
    req = _decode_req(stub, prompt_len=8, output_len=40)
    return stub, req, stub._spec_policy(req)


def test_a_cycle_reports_its_whole_wall_time_to_the_policy(monkeypatch):
    stub, req, policy = _timed_stub(monkeypatch, step_ms=30.0)

    stub._speculative_decode_step(req)

    assert policy.cycle_samples == 1
    assert policy.cycle_ms == pytest.approx(30.0)


def test_a_plain_decode_step_is_priced_as_the_interval_between_drains(monkeypatch):
    """Under overlap a decode step spans a whole loop iteration -- batch N is launched in the
    iteration that drains batch N-1 -- so the step's cost is the gap between drains, and the
    first drain can only mark."""
    stub, req, policy = _timed_stub(monkeypatch, step_ms=18.0)
    batch = Batch(reqs=[req], phase="decode")

    stub._spec_record_plain(batch)
    assert policy.plain_samples == 0

    stub._spec_record_plain(batch)
    assert policy.plain_samples == 1
    assert policy.plain_ms == pytest.approx(18.0)


def test_a_speculative_cycle_between_two_drains_voids_the_interval(monkeypatch):
    """Otherwise the plain step the bar divides by would be priced at a plain step PLUS a
    cycle -- which would drive the measured bar toward 1 exactly when cycles got dear."""
    stub, req, policy = _timed_stub(monkeypatch, step_ms=18.0)
    batch = Batch(reqs=[req], phase="decode")

    stub._spec_record_plain(batch)
    stub._speculative_decode_step(req)
    stub._spec_record_plain(batch)

    assert policy.plain_samples == 0
    # ... and the drain after the cycle marks again, so the next interval is priced
    stub._spec_record_plain(batch)
    assert policy.plain_ms == pytest.approx(18.0)


@pytest.mark.parametrize("phase,extra", [("prefill", 0), ("decode", 1)])
def test_only_a_single_request_decode_drain_prices_a_plain_step(monkeypatch, phase, extra):
    """A prefill, or a batch with another request in it, is not a plain decode step."""
    stub, req, policy = _timed_stub(monkeypatch, step_ms=18.0)
    other = Req(
        input_ids=torch.arange(1, 5, dtype=torch.int32), table_idx=1, cached_len=0,
        output_len=8, uid=99, sampling_params=SamplingParams(max_tokens=8), cache_handle=None,
    )
    decode = Batch(reqs=[req], phase="decode")
    intruder = Batch(reqs=[req] + [other] * extra, phase=phase)

    stub._spec_record_plain(decode)
    stub._spec_record_plain(intruder)
    stub._spec_record_plain(decode)

    assert policy.plain_samples == 0


def test_cost_aware_off_reads_no_clock_into_the_policy(monkeypatch):
    stub, req, policy = _timed_stub(monkeypatch, step_ms=30.0, cost_aware=False)
    batch = Batch(reqs=[req], phase="decode")

    stub._speculative_decode_step(req)
    stub._spec_record_plain(batch)
    stub._spec_record_plain(batch)

    assert (policy.cycle_samples, policy.plain_samples) == (0, 0)
    assert policy.bar == pytest.approx(2.0)


def test_a_disabled_fallback_prices_nothing_either(monkeypatch):
    """``FREETOKEN_MTP_SPEC_MIN_EMITTED=0`` constructs no policy at all; the plain hook must
    not construct one behind it."""
    stub, req, _policy = _timed_stub(monkeypatch, step_ms=30.0, min_emitted=0.0)

    stub._spec_record_plain(Batch(reqs=[req], phase="decode"))
    stub._spec_record_plain(Batch(reqs=[req], phase="decode"))

    assert getattr(stub, "_spec_policies", None) in (None, {})


def test_the_plain_step_mark_is_taken_on_the_drains_own_sync(monkeypatch):
    """Sync-free by placement: the drain has already waited for ``copy_done`` by the time the
    timestamp is taken, so the hook adds a ``perf_counter`` and no device work whatsoever."""
    from contextlib import nullcontext

    order: list[str] = []
    stub = Scheduler.__new__(Scheduler)
    stub.finished_reqs = set()
    stub.cache_manager = SimpleNamespace(lazy_free_region=nullcontext)
    stub._spec_record_plain = lambda batch: order.append("mark")
    stub._ship_replies = lambda *a, **k: order.append("ship")
    batch = Batch(reqs=[], phase="decode")
    copy_done = SimpleNamespace(synchronize=lambda: order.append("sync"))
    last_data = (SimpleNamespace(batch=batch), (None, None, copy_done))

    Scheduler._process_last_data(stub, last_data)

    assert order == ["sync", "mark", "ship"]


# ------------------------------------------------------------ finishing under an inflight step


def test_a_request_that_finishes_mid_cycle_settles_before_its_pages_are_freed():
    """The radix guard refuses a commit under in-flight speculative rows, and
    ``_free_req_resources`` commits. Settling first is what keeps the finish path legal."""
    target = _FakeTarget(eos_at=11)
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub, prompt_len=8, output_len=20)
    freed: list[Req] = []
    stub.table_manager = SimpleNamespace(free=lambda idx: freed.append(idx))

    stub._speculative_decode_step(req)

    (msg,) = stub.sent[-1]
    assert msg.finished and msg.finish_reason == "stop"
    assert req.spec_inflight is None  # settled before the free ran
    assert freed == [0] and req.table_idx == -1
    assert req not in stub.decode_manager.running_reqs


def test_a_finished_request_is_reset_on_the_draft_head():
    target = _FakeTarget(eos_at=11)
    draft = _FakeDraft(target)
    stub = _scheduler(target, draft)
    req = _decode_req(stub, prompt_len=8, output_len=20, uid=7)
    stub._speculative_decode_step(req)
    assert draft.reset_uids == [7]


# -------------------------------------------------------------------- the loop's own order
#
# Design 6.2: the cycle is serial, so a speculative step is DRAINED IN THE SAME ITERATION it is
# launched. Ordering the drain of the previous batch first is not a nicety -- it is what makes
# `_prepare_spec_batch`'s "host ids caught up" guard true, and it is the whole reason the
# observer's snapshot-skew bug cannot recur here.


def _loop_stub(*, enabled, draft, decode_runnable=True, prefill_runnable=False):
    from contextlib import nullcontext

    order: list[str] = []
    stub = Scheduler.__new__(Scheduler)
    stub.config = SimpleNamespace(spec_decode=SpecDecodeConfig(enabled=enabled, depth=3))
    stub.engine = SimpleNamespace(
        spec_draft=draft, stream=SimpleNamespace(wait_stream=lambda s: None)
    )
    stub.engine_stream_ctx = nullcontext()
    stub.stream = SimpleNamespace(wait_stream=lambda s: None)
    stub.prefill_manager = SimpleNamespace(runnable=prefill_runnable)
    stub.decode_manager = SimpleNamespace(runnable=decode_runnable)
    stub._pending_rebuild = None
    stub.receive_msg = lambda blocking: []
    stub._flush_abort_acks = lambda: None
    stub._process_last_data = lambda data: order.append(f"drain({data})")
    stub._restore_linear_states = lambda batch: None
    stub._forward = lambda fi: order.append("forward") or "out"
    stub._schedule_next_batch = lambda: order.append("schedule") or None
    stub._spec_candidate = lambda: order.append("candidate") or _SENTINEL_REQ
    stub._speculative_decode_step = lambda req: order.append("spec-step")
    return stub, order


_SENTINEL_REQ = object()


def test_the_loop_drains_the_previous_batch_before_it_speculates():
    stub, order = _loop_stub(enabled=True, draft=object())
    assert stub.overlap_loop("batch-n-1") is None
    assert order == ["drain(batch-n-1)", "candidate", "spec-step"]
    assert stub._last_data is None  # nothing is left in flight


def test_a_missed_candidate_falls_back_without_draining_twice():
    stub, order = _loop_stub(enabled=True, draft=object())
    stub._spec_candidate = lambda: order.append("candidate") or None
    stub.overlap_loop("batch-n-1")
    assert order == ["drain(batch-n-1)", "candidate", "schedule", "drain(None)"]


def test_the_flag_off_loop_keeps_the_overlap_order_exactly():
    stub, order = _loop_stub(enabled=False, draft=None)
    stub.overlap_loop("batch-n-1")
    assert order == ["schedule", "drain(batch-n-1)"]


def test_a_pending_prefill_keeps_the_overlap_order():
    stub, order = _loop_stub(enabled=True, draft=object(), prefill_runnable=True)
    stub.overlap_loop("batch-n-1")
    assert order == ["schedule", "drain(batch-n-1)"]


def test_the_non_overlap_loop_speculates_without_an_early_drain():
    stub, order = _loop_stub(enabled=True, draft=object())
    stub.normal_loop()
    assert order == ["candidate", "spec-step"]


# ------------------------------------------------------------------------ the real ladder
#
# Phase 3's kernel gate (tests/models/qwen4_exp/test_mtp_state_ladder.py) proves the replay is
# bit-exact. What Phase 5 owes is that the REAL ladder is on the path with the REAL spec batch,
# and that it settles on the emitted run rather than the accepted one. A pool with no GDN
# layers gives exactly that: the ladder's snapshot/restore and its arming are real, and the
# replay loop it drives is empty, so no triton launch is needed to check the wiring.


def _layerless_pool():
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(), num_key_heads=1, num_value_heads=1,
        key_head_dim=8, value_head_dim=8, conv_kernel_dim=4, output_gate="sigmoid",
    )
    return LinearStatePool(group, 4, torch.float32, CPU, tp_size=1)


def test_the_real_state_ladder_is_armed_on_the_real_speculative_batch():
    from freetoken.core import get_global_ctx
    from freetoken.engine.spec_state_ladder import SpecStateLadder

    target = _FakeTarget()
    pool = _layerless_pool()
    stub = _scheduler(target, _FakeDraft(target))
    stub.engine.linear_state_pool = pool
    stub.engine.ctx.linear_state_pool = pool
    ladder = SpecStateLadder(pool, 4)
    stub.engine.spec_state_ladder = ladder
    req = _decode_req(stub, prompt_len=8)

    seen: list = []
    plain_forward = target.forward_mtp_capture

    def _observing(**kwargs):
        batch = get_global_ctx().batch
        seen.append((batch.spec_capture, batch.emit_width, batch.mtp_verify))
        return plain_forward(**kwargs)

    target.forward_mtp_capture = _observing
    stub._speculative_decode_step(req)

    assert seen == [(ladder, 4, True)]  # armed before the forward, on the spec batch
    assert (req.cached_len, req.device_len) == (12, 13)


@pytest.mark.parametrize("policy", ["perfect", "garbage", "first-only"])
def test_the_real_ladder_settles_on_the_emitted_run(policy):
    from freetoken.engine.spec_state_ladder import SpecStateLadder

    target = _FakeTarget()
    pool = _layerless_pool()
    stub = _scheduler(target, _FakeDraft(target, policy=policy))
    stub.engine.linear_state_pool = pool
    stub.engine.ctx.linear_state_pool = pool
    ladder = SpecStateLadder(pool, 4)
    stub.engine.spec_state_ladder = ladder
    req = _decode_req(stub, prompt_len=8)

    stub._speculative_decode_step(req)

    (msg,) = stub.sent[-1]
    # the ladder released its step; the next cycle can arm again
    assert ladder._live is None and ladder._width == 0
    assert req.cached_len == 8 + len(msg.next_tokens)


# ------------------------------------------------------------------------- emission ordering


def test_a_settled_emission_reads_the_budget_from_the_host_not_the_device():
    """A speculative step's ``device_len`` is the last DRAFT row's, not the accepted run's, so
    ``hit_length`` cannot come from ``can_decode``. Plain decode's overlap-skewed reading is
    untouched -- ``settled`` is opt-in."""
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub, prompt_len=4, output_len=4)  # max_device_len 8, host at 5
    req.device_len = 8  # what _prepare_spec_batch leaves behind for a w = 4 step

    msg = Scheduler._emit_step_tokens(
        stub, req, torch.tensor([1, 2], dtype=torch.int32), settled=True
    )
    assert msg.next_tokens == (1, 2) and msg.finished is False

    msg = Scheduler._emit_step_tokens(
        stub, req, torch.tensor([3, 4], dtype=torch.int32), settled=True
    )
    assert msg.next_tokens == (3,) and msg.finish_reason == "length"


# ------------------------------------------------------------------------- the timing probe


class _RecordingProbe:
    """A ``_SpecTimingProbe`` in name only -- the engine duck-types on ``.mark``."""

    def __init__(self):
        self.marks: list[str] = []
        self.finished: list[tuple] = []

    def start_cycle(self) -> None:
        self.marks.clear()

    def mark(self, stage: str) -> None:
        self.marks.append(stage)

    def finish_cycle(self, *, emitted, accepted, policy=None, decision=None):
        self.finished.append((emitted, accepted, decision))


def test_the_cycle_subdivides_its_verify_stage_when_a_probe_is_armed():
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub)
    probe = _RecordingProbe()
    stub._spec_probe = probe

    stub._speculative_decode_step(req)

    assert probe.marks == [
        "draft",
        "prepare",
        "verify.forward",
        "verify.accept",
        "verify.pack",
        # the coarse stage still closes AFTER the engine returns, so it still spans the whole
        # verify -- the sub-marks only subdivide it
        "verify+accept",
        # the tail subdivides the same way: emit / rollback close before the coarse stage,
        # commit after it, and none of them moves what the coarse stages span
        "tail.emit",
        "tail.rollback",
        "emit+rollback",
        "tail.commit",
    ]


def test_the_probe_is_handed_the_untruncated_verdicts_acceptance_timings():
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub)
    probe = _RecordingProbe()
    stub._spec_probe = probe

    stub._speculative_decode_step(req)

    (emitted, accepted, decision) = probe.finished[-1]
    assert emitted == accepted == 4
    assert decision.accepted_rows == 4
    assert decision.filter_ms >= 0.0 and decision.decide_ms >= 0.0
    assert decision.sync_ms >= 0.0


# -------------------------------------------------------------------- the confidence-cut log


def test_an_unset_conf_log_flag_arms_nothing(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MTP_SPEC_CONF_LOG", raising=False)
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub)

    stub._speculative_decode_step(req)

    assert stub._spec_conf_log() is None
    (msg,) = stub.sent[-1]
    assert len(msg.next_tokens) == 4


def test_an_armed_conf_log_writes_one_well_formed_line_per_cycle(monkeypatch, tmp_path):
    monkeypatch.setenv("FREETOKEN_MTP_SPEC_CONF_LOG", str(tmp_path))
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub)
    cached_len = req.cached_len

    stub._speculative_decode_step(req)
    stub._spec_conf_log().flush()  # the buffer is drained every 100 cycles or at exit

    (path,) = list(tmp_path.glob("conf-log-*.jsonl"))
    (line,) = [json.loads(raw) for raw in path.read_text().splitlines()]
    assert set(line) == {
        "uid",
        "cached_len",
        "draft_q",
        "draft_top1",
        "draft_top1_gap",
        "accepted_drafts",
        "emitted",
        "greedy",
    }
    assert line["uid"] == req.uid
    # the length the drafts were made AT, not the one the cycle settled the request to
    assert line["cached_len"] == cached_len
    assert len(line["draft_q"]) == 3  # one per draft, whatever the run's own length
    assert all(0.0 <= q <= 1.0 for q in line["draft_q"])
    assert line["accepted_drafts"] == 3 and line["emitted"] == 4
    assert line["greedy"] is True
    # the fake draft head does not compute them; the field is present and null either way
    assert line["draft_top1"] is None and line["draft_top1_gap"] is None


def test_the_conf_log_carries_the_draft_heads_confidence_when_the_head_reports_it(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("FREETOKEN_MTP_SPEC_CONF_LOG", str(tmp_path))
    target = _FakeTarget()
    draft = _FakeDraft(target)
    inner = draft.propose

    def _with_confidence(req, depth):
        from freetoken.engine.spec_draft import DraftProposal

        proposal = inner(req, depth)
        return DraftProposal(
            tokens=proposal.tokens,
            logits=proposal.logits,
            draft_top1=tuple(0.9 - 0.1 * i for i in range(len(proposal.tokens))),
            draft_top1_gap=tuple(0.5 - 0.1 * i for i in range(len(proposal.tokens))),
        )

    draft.propose = _with_confidence
    stub = _scheduler(target, draft)
    req = _decode_req(stub)

    stub._speculative_decode_step(req)
    stub._spec_conf_log().flush()

    (path,) = list(tmp_path.glob("conf-log-*.jsonl"))
    (line,) = [json.loads(raw) for raw in path.read_text().splitlines()]
    assert line["draft_top1"] == pytest.approx([0.9, 0.8, 0.7])
    assert line["draft_top1_gap"] == pytest.approx([0.5, 0.4, 0.3])


def test_the_conf_log_buffers_rather_than_writing_a_line_at_a_time(monkeypatch, tmp_path):
    """A cycle is ~20 ms live; the log must not put a file write in the middle of every one."""
    monkeypatch.setenv("FREETOKEN_MTP_SPEC_CONF_LOG", str(tmp_path))
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub)

    _run_spec(stub, req, cycles=8)

    log = stub._spec_conf_log()
    assert log.cycles >= 2
    assert not list(tmp_path.glob("conf-log-*.jsonl"))  # nothing on disk before a flush
    log.flush()
    (path,) = list(tmp_path.glob("conf-log-*.jsonl"))
    assert len(path.read_text().splitlines()) == log.cycles


def test_an_unarmed_probe_leaves_the_cycle_exactly_as_it_was():
    target = _FakeTarget()
    stub = _scheduler(target, _FakeDraft(target))
    req = _decode_req(stub)
    stub._spec_probe = None

    stub._speculative_decode_step(req)

    (msg,) = stub.sent[-1]
    assert len(msg.next_tokens) == 4

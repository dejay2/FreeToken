"""CUDA graphs for the INTEGRATED speculative verify forward.

The observer's verify graph (``tests/engine/test_mtp_fast_verify_graph.py``) captures the same
model against a leased scratch page range and a shadow GDN slot, and needs only logits back.
The integrated step differs in three ways, and each one is a test below:

* it replays against the request's REAL page-table row and REAL linear slot, so every address
  the step reads is a per-replay refill rather than a capture-time constant;
* it needs the ``w`` hidden rows as well as the ``w`` logit rows -- the draft head consumes
  them -- which is exactly the slot the historical blocker was missing;
* it runs inside a live cycle, so capture's warm-up pass advances the request's linear state
  and must be wound back before the replay produces the step's real values.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.spec_graph import SpecVerifyGraphRunner, _SpecVerifyGraphBuffer

VOCAB = 6
HIDDEN = 5


class _Context:
    def __init__(self):
        self.batch = None

    @contextmanager
    def forward_batch(self, batch):
        self.batch = batch
        try:
            yield batch
        finally:
            self.batch = None


class _Attention:
    def __init__(self):
        self.prepared = []
        self.staged = []
        self.discarded = []
        self.reset_count = 0

    def prepare_mtp_verify_graph(self, batch):
        self.prepared.append(int(batch.input_ids.shape[0]))

    def stage_mtp_verify_graph(self, runtime_batch, static_batch):
        self.staged.append(int(runtime_batch.input_ids.shape[0]))

    def discard_mtp_verify_graph(self, width):
        self.discarded.append(width)

    def reset_mtp_verify_graph(self):
        self.reset_count += 1


class _Model:
    """Every per-step input reaches both outputs, so a missed refill cannot hide."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.capture_sizes = []
        self.replay_sizes = []

    def forward_mtp_capture(self, *, all_row_logits=False):
        assert all_row_logits, "the integrated step needs every row's logits"
        batch = self.ctx.batch
        logits, hidden = _outputs(batch)
        return logits, hidden, hidden  # (logits, multi_stream, inputs_embeds)

    def prepare_cuda_graph_capture(self, batch):
        self.capture_sizes.append(int(batch.input_ids.shape[0]))

    def prepare_cuda_graph_replay(self, batch):
        self.replay_sizes.append(int(batch.input_ids.shape[0]))


def _outputs(batch):
    width = int(batch.input_ids.shape[0])
    stream = torch.stack(
        (
            batch.input_ids.float(),
            batch.positions.float(),
            batch.out_loc.float(),
            batch.rope_positions[0].float(),
            batch.linear_table_idx[0].float().expand(width),
        ),
        dim=-1,
    )
    logits = torch.cat((stream, stream[:, :1] * 2.0), dim=-1)
    return logits, stream * 3.0


def _batch(width: int, offset: int, device: torch.device) -> Batch:
    cached_len = 5
    req = Req(
        input_ids=torch.arange(offset, offset + cached_len + width, dtype=torch.int32),
        table_idx=20 + offset,
        cached_len=cached_len,
        output_len=0,
        uid=offset,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    req.linear_slot_idx = 3 + offset
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.mtp_verify = True
    batch.emit_width = width
    batch.input_ids = torch.arange(offset, offset + width, dtype=torch.int32, device=device)
    batch.positions = torch.arange(
        100 + offset, 100 + offset + width, dtype=torch.int32, device=device
    )
    batch.out_loc = torch.arange(
        200 + offset, 200 + offset + width, dtype=torch.int32, device=device
    )
    batch.rope_positions = batch.positions.to(torch.int64).expand(3, -1).clone()
    batch.linear_table_idx = torch.tensor(
        [req.linear_slot_idx], dtype=torch.int32, device=device
    )
    batch.attn_metadata = None
    return batch


def _runner(device, ctx=None, model=None, attention=None):
    ctx = ctx if ctx is not None else _Context()
    return SpecVerifyGraphRunner(
        target_ctx=ctx,
        target_model=model if model is not None else _Model(ctx),
        attn_backend=attention if attention is not None else _Attention(),
        device=device,
        widths=(2, 3, 4),
        guard_bytes=0,
    )


# --------------------------------------------------------------------------- shape contract


@pytest.mark.parametrize("width", (2, 3, 4))
def test_the_buffer_refills_every_input_the_live_step_moves(width):
    device = torch.device("cpu")
    buffer = _SpecVerifyGraphBuffer.init(width, device)
    batch = _batch(width, 7, device)

    buffer.copy_from(batch)

    assert torch.equal(buffer.input_ids, batch.input_ids)
    assert torch.equal(buffer.positions, batch.positions)
    assert torch.equal(buffer.out_loc, batch.out_loc)
    assert torch.equal(buffer.rope_positions, batch.rope_positions)
    assert torch.equal(buffer.linear_table_idx, batch.linear_table_idx)
    assert buffer.fla_cu_seqlens.tolist() == [0, width]
    assert buffer.fla_cu_seqlens.dtype is torch.int64


def test_the_buffer_refuses_a_width_it_was_not_built_for():
    buffer = _SpecVerifyGraphBuffer.init(2, torch.device("cpu"))
    with pytest.raises(ValueError, match="replay width does not match"):
        buffer.copy_from(_batch(3, 7, torch.device("cpu")))


def test_the_runner_refuses_a_batch_that_is_not_one_speculative_request():
    runner = _runner(torch.device("cpu"))

    unmarked = _batch(2, 1, torch.device("cpu"))
    unmarked.mtp_verify = False
    with pytest.raises(ValueError, match="private verification marker"):
        runner.capture(unmarked)

    decode = _batch(2, 2, torch.device("cpu"))
    decode.phase = "decode"
    with pytest.raises(ValueError, match="one prefill request"):
        runner.capture(decode)

    narrow = _batch(3, 3, torch.device("cpu"))
    narrow.reqs[0].cached_len += 1
    with pytest.raises(ValueError, match="extend_len"):
        runner.capture(narrow)


def test_without_cuda_the_runner_allocates_nothing_and_says_so():
    runner = _runner(torch.device("cpu"))

    support = runner.capture(_batch(2, 1, torch.device("cpu")))

    assert support.status == "permanently-unsupported"
    assert support.reason == "CUDA_REQUIRED"
    assert runner.graph_count == 0
    assert runner.owned_buffer_bytes == 0
    assert runner.forward(_batch(2, 1, torch.device("cpu"))) is None


# ---------------------------------------------------------------------- graph == eager, live


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("width", (2, 3, 4))
def test_a_captured_replay_reproduces_the_eager_logits_and_hidden_rows(width):
    device = torch.device("cuda")
    attention = _Attention()
    ctx = _Context()
    model = _Model(ctx)
    runner = _runner(device, ctx=ctx, model=model, attention=attention)

    assert runner.capture(_batch(width, width, device)).status == "captured"

    batch = _batch(width, 40 + width, device)
    logits, hidden = runner.replay(batch)

    want_logits, want_hidden = _outputs(_batch(width, 40 + width, device))
    assert torch.equal(logits, want_logits)
    assert torch.equal(hidden, want_hidden)
    assert hidden.shape == (width, HIDDEN)
    assert attention.prepared == [width]
    assert attention.staged == [width]
    assert model.replay_sizes == [width]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("width", (2, 3, 4))
def test_two_consecutive_replays_with_different_refills_never_go_stale(width):
    device = torch.device("cuda")
    runner = _runner(device)
    assert runner.capture(_batch(width, width, device)).status == "captured"

    first_logits, first_hidden = runner.replay(_batch(width, 50 + width, device))
    second_logits, second_hidden = runner.replay(_batch(width, 90 + width, device))

    want_first = _outputs(_batch(width, 50 + width, device))
    want_second = _outputs(_batch(width, 90 + width, device))
    assert torch.equal(second_logits, want_second[0])
    assert torch.equal(second_hidden, want_second[1])
    # the first replay's results must survive the second: the caller holds them across the
    # cycle's accept / emit / rollback tail
    assert torch.equal(first_logits, want_first[0])
    assert torch.equal(first_hidden, want_first[1])
    assert first_logits.data_ptr() != second_logits.data_ptr()
    assert first_hidden.data_ptr() != second_hidden.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_a_replay_after_a_rollback_between_steps_still_matches_eager():
    """The settle path rewrites the linear slot and returns pages between two replays; a graph
    that had baked any of that would drift on the second cycle."""
    device = torch.device("cuda")
    runner = _runner(device)
    assert runner.capture(_batch(3, 3, device)).status == "captured"

    runner.replay(_batch(3, 60, device))
    rolled = _batch(3, 61, device)
    rolled.reqs[0].linear_slot_idx = 6  # a different slot, as a re-admission would give
    rolled.linear_table_idx = torch.tensor([6], dtype=torch.int32, device=device)
    rolled.out_loc = rolled.out_loc + 4096  # freed pages, re-leased elsewhere

    logits, hidden = runner.replay(rolled)

    want_logits, want_hidden = _outputs(rolled)
    assert torch.equal(logits, want_logits)
    assert torch.equal(hidden, want_hidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_forward_captures_lazily_then_replays_the_same_width():
    device = torch.device("cuda")
    ctx = _Context()
    model = _Model(ctx)
    runner = _runner(device, ctx=ctx, model=model)

    first = runner.forward(_batch(2, 10, device))
    second = runner.forward(_batch(2, 30, device))

    assert runner.graph_count == 1
    assert torch.equal(first[0], _outputs(_batch(2, 10, device))[0])
    assert torch.equal(second[1], _outputs(_batch(2, 30, device))[1])
    assert model.capture_sizes == [2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_capture_winds_the_live_state_back_before_the_step_produces_values():
    """Capture's warm-up pass EXECUTES the forward and so advances the request's GDN slot;
    the recorded pass does not execute at all. The runner must hand the caller a restore seam
    between the two, or the first speculative cycle at each width settles from a doubled state.
    """
    device = torch.device("cuda")
    ctx = _Context()
    model = _Model(ctx)
    runner = _runner(device, ctx=ctx, model=model)
    marks = []

    logits, _ = runner.forward(
        _batch(2, 12, device), restore_state=lambda: marks.append(len(model.capture_sizes))
    )

    assert marks, "capture never asked the caller to wind the live state back"
    assert torch.equal(logits, _outputs(_batch(2, 12, device))[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_a_failed_capture_falls_back_to_eager_and_still_restores():
    class _Broken(_Model):
        def forward_mtp_capture(self, *, all_row_logits=False):
            raise RuntimeError("synthetic transient capture failure")

    device = torch.device("cuda")
    ctx = _Context()
    runner = _runner(device, ctx=ctx, model=_Broken(ctx))
    marks = []

    result = runner.forward(_batch(2, 1, device), restore_state=lambda: marks.append(1))

    assert result is None
    assert marks == [1]
    assert runner.graph_count == 0
    assert runner.last_attempt(2).reason.startswith("CAPTURE_FAILED:RuntimeError:")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_destroy_releases_every_width():
    device = torch.device("cuda")
    attention = _Attention()
    runner = _runner(device, attention=attention)
    for width in (2, 3, 4):
        assert runner.capture(_batch(width, width, device)).status == "captured"
    assert runner.graph_count == 3
    assert runner.owned_buffer_bytes > 0

    runner.destroy()

    assert runner.graph_count == 0
    assert runner.owned_buffer_bytes == 0
    assert runner.live_graph_memory_bytes == 0
    assert attention.reset_count == 1


# ----------------------------------------------------------------- the replay timing seam


def test_the_replay_timing_seam_is_disarmed_by_default():
    assert _runner(torch.device("cpu")).replay_timings is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_an_armed_replay_splits_the_host_staging_from_the_gpu_duration():
    device = torch.device("cuda")
    runner = _runner(device)
    assert runner.capture(_batch(2, 2, device)).status == "captured"

    runner.replay(_batch(2, 70, device))
    assert runner.replay_timings is None
    # disarmed the replay must not create the timing events, and so never syncs on them
    assert runner._replay_events_pair is None

    runner.replay_timings = {}
    logits, hidden = runner.replay(_batch(2, 71, device))

    timings = runner.replay_timings
    assert set(timings) == {"copy_ms", "attn_ms", "model_ms", "launch_ms", "gpu_ms"}
    assert all(value >= 0.0 for value in timings.values())
    assert runner._replay_events_pair is not None
    # arming may not change what the replay produces
    want_logits, want_hidden = _outputs(_batch(2, 71, device))
    assert torch.equal(logits, want_logits)
    assert torch.equal(hidden, want_hidden)


# ------------------------------------------------------------------- the engine's dispatch


class _Sampler:
    def __init__(self):
        self.rows = []

    def step(self, *, uid, draft_tokens, draft_logits, target_logits, args):
        from types import SimpleNamespace

        self.rows.append(target_logits.clone())
        return SimpleNamespace(tokens=(7,))


class _Ladder:
    def __init__(self):
        self.restores = 0

    def restore_snapshot(self):
        self.restores += 1


class _Runner:
    def __init__(self, result):
        self.result = result
        self.restore_state = "unset"

    def forward(self, batch, *, restore_state=None):
        self.restore_state = restore_state
        return self.result


def _engine(runner, ladder, sampler, ctx, model):
    from types import SimpleNamespace

    return SimpleNamespace(
        device=torch.device("cpu"),
        ctx=ctx,
        model=model,
        cpu_moe_executor=None,
        spec_sampler=sampler,
        spec_state_ladder=ladder,
        spec_graph_runner=runner,
    )


def _step(engine, batch, probe=None):
    from freetoken.engine.engine import Engine

    return Engine.speculative_decode_batch(
        engine,
        batch,
        None,
        draft_tokens=(1,),
        draft_logits=torch.zeros(1, VOCAB + 1),
        probe=probe,
    )


def test_the_engine_step_uses_the_graph_and_hands_it_the_ladders_wind_back():
    device = torch.device("cpu")
    ctx = _Context()
    model = _Model(ctx)
    batch = _batch(2, 4, device)
    want_logits, want_hidden = _outputs(batch)
    ladder, sampler = _Ladder(), _Sampler()
    runner = _Runner((want_logits, want_hidden))

    output = _step(_engine(runner, ladder, sampler, ctx, model), batch)

    assert runner.restore_state == ladder.restore_snapshot
    assert torch.equal(sampler.rows[0], want_logits[:2])
    assert output.hidden is want_hidden


def test_the_engine_step_falls_back_to_the_eager_forward_when_the_graph_declines():
    device = torch.device("cpu")
    ctx = _Context()
    model = _Model(ctx)
    batch = _batch(3, 6, device)
    ladder, sampler = _Ladder(), _Sampler()

    output = _step(_engine(_Runner(None), ladder, sampler, ctx, model), batch)

    want_logits, want_hidden = _outputs(_batch(3, 6, device))
    assert torch.equal(sampler.rows[0], want_logits[:3])
    assert torch.equal(output.hidden, want_hidden)


class _TimedRunner(_Runner):
    def forward(self, batch, *, restore_state=None):
        assert self.replay_timings == {}, "the engine must arm the runner before the forward"
        self.replay_timings.update(
            copy_ms=1.0, attn_ms=2.0, model_ms=3.0, launch_ms=4.0, gpu_ms=5.0
        )
        return super().forward(batch, restore_state=restore_state)


class _MarkOnlyProbe:
    def __init__(self):
        self.marks = []

    def mark(self, name):
        self.marks.append(name)


class _FullProbe(_MarkOnlyProbe):
    def __init__(self):
        super().__init__()
        self.added = {}

    def add_ms(self, name, ms):
        self.added[name] = self.added.get(name, 0.0) + float(ms)


def test_a_probe_arms_the_runner_and_collects_the_replay_split():
    device = torch.device("cpu")
    ctx = _Context()
    batch = _batch(2, 4, device)
    runner = _TimedRunner(_outputs(batch))
    probe = _FullProbe()

    _step(_engine(runner, _Ladder(), _Sampler(), ctx, _Model(ctx)), batch, probe=probe)

    assert probe.added == {
        "replay.copy": 1.0,
        "replay.attn": 2.0,
        "replay.model": 3.0,
        "replay.launch": 4.0,
        "replay.gpu": 5.0,
    }
    # production must find the runner disarmed again
    assert runner.replay_timings is None


def test_a_probe_without_add_ms_still_steps():
    """The fake probes in these suites carry only .mark; arming must not require more."""
    device = torch.device("cpu")
    ctx = _Context()
    batch = _batch(2, 4, device)
    probe = _MarkOnlyProbe()

    _step(
        _engine(_TimedRunner(_outputs(batch)), _Ladder(), _Sampler(), ctx, _Model(ctx)),
        batch,
        probe=probe,
    )

    assert probe.marks == ["verify.forward", "verify.accept", "verify.pack"]


def test_without_a_graph_runner_the_step_is_the_eager_forward_it_always_was():
    device = torch.device("cpu")
    ctx = _Context()
    model = _Model(ctx)
    batch = _batch(4, 8, device)
    sampler = _Sampler()

    output = _step(_engine(None, None, sampler, ctx, model), batch)

    assert torch.equal(output.hidden, _outputs(_batch(4, 8, device))[1])

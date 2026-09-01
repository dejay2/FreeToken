"""The width-1 spec graph: an ORDINARY decode step, captured, that still feeds the draft head.

A spec-enabled boot cannot use ``graph_runner.replay`` for its ordinary forwards, because that
graph returns logits and nothing else while ``SpecDraftHead.observe_forward`` needs the hidden
rows and the input embeddings too. So every non-speculative forward went eager -- and once the
adaptive policy falls back to plain decode, that is EVERY step, at roughly half the graphed
rate. Measured live: mostly-fallback requests ran 17-27 tok/s against ~45 graphed.

Width 1 of the spec graph closes that: the same survivable-capture machinery, recording the
same ``forward_mtp_capture(all_row_logits=False)`` the eager path runs, into fixed logit,
hidden and embedding slots.

It is DECODE-SHAPED, not a one-row verify batch. The verify batch is prefill-phase (chunked GDN
kernel, ragged QSA), the ordinary decode batch is decode-phase (fused recurrent, paged QSA
decode); the draft head has always been fed from the decode-phase numbers, and a graph that
quietly swapped the kernels underneath it would change what a fallback step computes. So the
contract below is equality with the eager decode forward, not closeness to it.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.spec_graph import SpecVerifyGraphRunner, _SpecDecodeGraphBuffer

_CUDA = torch.cuda.is_available()


class _Context:
    def __init__(self):
        self.batch = None

    @contextmanager
    def forward_batch(self, batch):
        assert self.batch is None, "nested forward_batch"
        self.batch = batch
        try:
            yield batch
        finally:
            self.batch = None


class _Attention:
    """The DECODE capture hooks, not the verify ones: the width-1 graph stages its addressing
    through the same persistent buffers the plain decode graph replays against."""

    def __init__(self):
        self.captured = []
        self.replayed = []
        self.verify_prepared = []
        self.reset_count = 0

    def prepare_for_capture(self, batch):
        self.captured.append(batch.padded_size)

    def prepare_for_replay(self, batch):
        assert batch.active_table_idx is not None
        self.replayed.append(batch.padded_size)

    def prepare_mtp_verify_graph(self, batch):
        self.verify_prepared.append(int(batch.input_ids.shape[0]))

    def stage_mtp_verify_graph(self, runtime_batch, static_batch):
        pass

    def reset_mtp_verify_graph(self):
        self.reset_count += 1


def _outputs(batch):
    """Every per-step input reaches all three outputs, so a missed refill cannot hide."""
    row = torch.stack(
        (
            batch.input_ids.float(),
            batch.positions.float(),
            batch.out_loc.float(),
            batch.rope_positions[0].float(),
            batch.linear_table_idx[:1].float(),
        ),
        dim=-1,
    )
    logits = torch.cat((row, row[:, :1] * 2.0), dim=-1)
    return logits, row * 3.0, row * 5.0


class _Model:
    def __init__(self, ctx):
        self.ctx = ctx
        self.capture_sizes = []
        self.replay_sizes = []

    def forward_mtp_capture(self, *, all_row_logits=False):
        assert not all_row_logits, "a decode step samples one row"
        return _outputs(self.ctx.batch)

    def prepare_cuda_graph_capture(self, batch):
        self.capture_sizes.append(batch.padded_size)

    def prepare_cuda_graph_replay(self, batch):
        self.replay_sizes.append(batch.padded_size)


def _decode(offset: int, device: torch.device) -> Batch:
    cached_len = 5 + offset
    req = Req(
        input_ids=torch.arange(cached_len + 1, dtype=torch.int32),
        table_idx=2,
        cached_len=cached_len,
        output_len=32,
        uid=1,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    req.linear_slot_idx = 3 + offset
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.tensor([offset], dtype=torch.int32, device=device)
    batch.positions = torch.tensor([100 + offset], dtype=torch.int32, device=device)
    batch.out_loc = torch.tensor([200 + offset], dtype=torch.int32, device=device)
    batch.rope_positions = batch.positions.to(torch.int64).expand(3, -1).clone()
    batch.linear_table_idx = torch.tensor(
        [req.linear_slot_idx], dtype=torch.int32, device=device
    )
    batch.active_table_idx = torch.tensor([req.table_idx], dtype=torch.int32, device=device)
    batch.attn_metadata = None
    return batch


def _runner(device, ctx=None, model=None, attention=None, widths=(1, 2, 3, 4)):
    ctx = ctx if ctx is not None else _Context()
    return SpecVerifyGraphRunner(
        target_ctx=ctx,
        target_model=model if model is not None else _Model(ctx),
        attn_backend=attention if attention is not None else _Attention(),
        device=device,
        widths=widths,
        guard_bytes=0,
    )


# --------------------------------------------------------------------------- shape contract


def test_the_buffer_refills_every_input_a_decode_step_moves():
    device = torch.device("cpu")
    buffer = _SpecDecodeGraphBuffer.init(device)
    batch = _decode(7, device)

    buffer.copy_from(batch)

    assert torch.equal(buffer.input_ids, batch.input_ids)
    assert torch.equal(buffer.positions, batch.positions)
    assert torch.equal(buffer.out_loc, batch.out_loc)
    assert torch.equal(buffer.rope_positions, batch.rope_positions)
    assert torch.equal(buffer.linear_table_idx, batch.linear_table_idx)
    # the decode GDN indptr is the plain runner's constant arange, not the verify [0, w]
    assert buffer.fla_cu_seqlens.tolist() == [0, 1]


def test_the_bound_batch_is_a_decode_batch_and_never_a_verify_one():
    device = torch.device("cpu")
    buffer = _SpecDecodeGraphBuffer.init(device)
    batch = _decode(3, device)
    buffer.copy_from(batch)

    buffer.bind(batch)

    assert batch.mtp_verify is False
    assert batch.is_decode
    assert batch.fla_metadata.cache_indices is buffer.linear_table_idx
    assert batch.fla_metadata.has_initial_state is None
    assert batch.input_ids is buffer.input_ids


def test_the_width_one_graph_refuses_a_batch_that_is_not_a_plain_decode_step():
    runner = _runner(torch.device("cpu"))

    verify = _decode(1, torch.device("cpu"))
    verify.mtp_verify = True
    with pytest.raises(ValueError, match="ordinary decode"):
        runner.capture(verify)

    prefill = _decode(2, torch.device("cpu"))
    prefill.phase = "prefill"
    with pytest.raises(ValueError, match="one decode request"):
        runner.capture(prefill)


def test_without_cuda_the_width_one_graph_allocates_nothing_and_declines():
    runner = _runner(torch.device("cpu"))

    support = runner.capture(_decode(1, torch.device("cpu")))

    assert support.status == "permanently-unsupported"
    assert support.reason == "CUDA_REQUIRED"
    assert runner.forward_decode(_decode(1, torch.device("cpu"))) is None


def test_a_runner_without_width_one_declines_every_decode():
    runner = _runner(torch.device("cpu"), widths=(2, 3, 4))
    assert runner.forward_decode(_decode(1, torch.device("cpu"))) is None
    assert runner.capture_pending(1) is False


# ---------------------------------------------------------------------- graph == eager, live


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_captured_replay_reproduces_the_eager_logits_hidden_and_embeddings():
    device = torch.device("cuda")
    attention = _Attention()
    ctx = _Context()
    model = _Model(ctx)
    runner = _runner(device, ctx=ctx, model=model, attention=attention)

    assert runner.capture(_decode(1, device)).status == "captured"

    logits, hidden, embeds = runner.forward_decode(_decode(11, device))

    want = _outputs(_decode(11, device))
    assert torch.equal(logits, want[0])
    assert torch.equal(hidden, want[1])
    assert torch.equal(embeds, want[2])
    assert attention.captured == [1]
    assert attention.replayed == [1]
    assert attention.verify_prepared == []  # decode-shaped: no verify metadata anywhere
    assert model.replay_sizes == [1]


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_two_consecutive_replays_with_different_tokens_never_go_stale():
    device = torch.device("cuda")
    runner = _runner(device)
    assert runner.capture(_decode(1, device)).status == "captured"

    first = runner.forward_decode(_decode(21, device))
    second = runner.forward_decode(_decode(41, device))

    want_first = _outputs(_decode(21, device))
    want_second = _outputs(_decode(41, device))
    for got, want in ((first, want_first), (second, want_second)):
        for tensor, expected in zip(got, want):
            assert torch.equal(tensor, expected)
    # owned, not views into the buffers the next replay overwrites
    assert first[0].data_ptr() != second[0].data_ptr()
    assert first[1].data_ptr() != second[1].data_ptr()


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_the_sampled_token_is_the_one_the_eager_forward_would_have_produced():
    device = torch.device("cuda")
    runner = _runner(device)
    batch = _decode(13, device)

    replayed = runner.forward_decode(batch)  # captures, then replays

    eager = _outputs(_decode(13, device))
    assert int(replayed[0].argmax(dim=-1)) == int(eager[0].argmax(dim=-1))


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_capture_winds_the_live_state_back_before_the_step_produces_values():
    """The warm-up pass EXECUTES a real decode and advances the request's GDN slot; the
    recorded pass executes nothing at all. Without the wind-back the first graphed fallback
    step would settle from a state one row ahead of itself."""
    device = torch.device("cuda")
    ctx = _Context()
    model = _Model(ctx)
    runner = _runner(device, ctx=ctx, model=model)
    marks = []

    logits, _hidden, _embeds = runner.forward_decode(
        _decode(9, device), restore_state=lambda: marks.append(len(model.capture_sizes))
    )

    assert marks, "capture never asked the caller to wind the live state back"
    assert torch.equal(logits, _outputs(_decode(9, device))[0])


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_failed_width_one_capture_falls_back_to_eager_and_still_restores():
    class _Broken(_Model):
        def forward_mtp_capture(self, *, all_row_logits=False):
            raise RuntimeError("synthetic transient capture failure")

    device = torch.device("cuda")
    ctx = _Context()
    runner = _runner(device, ctx=ctx, model=_Broken(ctx))
    marks = []

    result = runner.forward_decode(_decode(1, device), restore_state=lambda: marks.append(1))

    assert result is None
    assert marks == [1]
    assert runner.graph_count == 0
    assert runner.last_attempt(1).reason.startswith("CAPTURE_FAILED:RuntimeError:")


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_the_decode_width_is_admitted_and_accounted_beside_the_verify_widths():
    """One more graph per boot, in the same runner, out of the same pool and the same guard."""
    device = torch.device("cuda")
    attention = _Attention()
    runner = _runner(device, attention=attention)

    assert runner.capture_pending(1) is True
    assert runner.capture(_decode(1, device)).status == "captured"
    assert runner.capture_pending(1) is False

    assert runner.graph_count == 1
    assert runner.owned_buffer_bytes > 0
    assert runner.live_graph_memory_bytes > 0
    runner.destroy()
    assert runner.graph_count == 0
    assert runner.owned_buffer_bytes == 0


# -------------------------------------------------------------------- the engine's dispatch


def _forward_engine(*, spec_draft=None, spec_graph_runner=None, ladder=None, graphable=True):
    from freetoken.core import Context, set_global_ctx
    from freetoken.engine.engine import Engine
    from freetoken.engine.sample import Sampler
    from types import SimpleNamespace
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
    engine.spec_state_ladder = ladder
    engine.spec_graph_runner = spec_graph_runner
    engine.cpu_moe_executor = None
    engine.sampler = Sampler(device=device, vocab_size=6)
    engine.graph_runner = SimpleNamespace(
        can_use_cuda_graph=lambda batch: graphable, replay=lambda batch: _eager_logits(device)
    )
    engine.model = SimpleNamespace(
        forward=lambda: _eager_logits(device),
        forward_mtp_capture=lambda *, all_row_logits=False: (
            _eager_logits(device),
            torch.ones(1, 8, device=device),
            torch.ones(1, 2, device=device),
        ),
    )
    return engine


def _eager_logits(device):
    return torch.zeros(1, 6, device=device)


class _StubRunner:
    def __init__(self, result, pending=True):
        self.result = result
        self.pending = pending
        self.restore_state = "unset"
        self.decodes = 0

    def capture_pending(self, width):
        assert width == 1
        return self.pending

    def forward_decode(self, batch, *, restore_state=None):
        self.decodes += 1
        self.restore_state = restore_state
        return self.result


class _StubLadder:
    def __init__(self):
        self.borrowed = []
        self.released = 0

    @contextmanager
    def borrow_snapshot(self, req):
        self.borrowed.append(req)
        try:
            yield self.restore_snapshot
        finally:
            self.released += 1

    def restore_snapshot(self):
        pass


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_spec_enabled_decode_replays_the_width_one_graph_and_feeds_the_draft_from_it():
    device = torch.device("cuda")
    seen = []
    draft = type(
        "_Draft", (), {"observe_forward": lambda self, b, capture, tok: seen.append(capture)}
    )()
    graphed = (
        torch.tensor([[0.0, 9.0, 0.0, 0.0, 0.0, 0.0]], device=device),
        torch.full((1, 8), 2.0, device=device),
        torch.full((1, 2), 3.0, device=device),
    )
    runner = _StubRunner(graphed, pending=False)
    engine = _forward_engine(spec_draft=draft, spec_graph_runner=runner)
    batch = _decode(1, device)

    output = engine.forward_batch(batch, engine.sampler.prepare(batch))

    assert runner.decodes == 1
    assert seen == [graphed]  # the draft is fed from the graph's own slots
    assert int(output.next_tokens_gpu[0]) == 1  # argmax of the graphed logits


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_the_capture_step_borrows_the_ladders_snapshot_around_the_warm_up():
    device = torch.device("cuda")
    draft = type("_Draft", (), {"observe_forward": lambda self, b, c, t: None})()
    graphed = (
        _eager_logits(device),
        torch.ones(1, 8, device=device),
        torch.ones(1, 2, device=device),
    )
    ladder = _StubLadder()
    runner = _StubRunner(graphed, pending=True)
    engine = _forward_engine(spec_draft=draft, spec_graph_runner=runner, ladder=ladder)
    batch = _decode(1, device)

    engine.forward_batch(batch, engine.sampler.prepare(batch))

    assert ladder.borrowed == [batch.reqs[0]]
    assert ladder.released == 1
    assert runner.restore_state == ladder.restore_snapshot


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_replay_only_step_does_not_pay_for_a_snapshot():
    device = torch.device("cuda")
    draft = type("_Draft", (), {"observe_forward": lambda self, b, c, t: None})()
    graphed = (
        _eager_logits(device),
        torch.ones(1, 8, device=device),
        torch.ones(1, 2, device=device),
    )
    ladder = _StubLadder()
    runner = _StubRunner(graphed, pending=False)
    engine = _forward_engine(spec_draft=draft, spec_graph_runner=runner, ladder=ladder)
    batch = _decode(1, device)

    engine.forward_batch(batch, engine.sampler.prepare(batch))

    assert ladder.borrowed == []
    assert runner.restore_state is None


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_declining_graph_leaves_the_eager_capture_path_exactly_as_it_was():
    device = torch.device("cuda")
    seen = []
    draft = type(
        "_Draft", (), {"observe_forward": lambda self, b, capture, tok: seen.append(capture)}
    )()
    runner = _StubRunner(None, pending=False)
    engine = _forward_engine(spec_draft=draft, spec_graph_runner=runner)
    batch = _decode(1, device)

    engine.forward_batch(batch, engine.sampler.prepare(batch))

    assert runner.decodes == 1
    (capture,) = seen
    assert tuple(t.shape for t in capture) == (
        torch.Size([1, 6]),
        torch.Size([1, 8]),
        torch.Size([1, 2]),
    )


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_prefill_forward_never_reaches_the_width_one_graph():
    device = torch.device("cuda")
    draft = type("_Draft", (), {"observe_forward": lambda self, b, c, t: None})()
    runner = _StubRunner(None)
    engine = _forward_engine(spec_draft=draft, spec_graph_runner=runner)
    batch = _decode(1, device)
    batch.phase = "prefill"

    engine.forward_batch(batch, engine.sampler.prepare(batch))

    assert runner.decodes == 0


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_batch_the_plain_path_could_not_graph_stays_eager():
    """The width-1 graph stages its addressing through the backend's persistent decode
    buffers, which exist only when the ordinary decode graphs were armed at boot."""
    device = torch.device("cuda")
    draft = type("_Draft", (), {"observe_forward": lambda self, b, c, t: None})()
    runner = _StubRunner(None)
    engine = _forward_engine(spec_draft=draft, spec_graph_runner=runner, graphable=False)
    batch = _decode(1, device)

    engine.forward_batch(batch, engine.sampler.prepare(batch))

    assert runner.decodes == 0


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_with_speculation_off_the_forward_keeps_its_ordinary_graph_path():
    device = torch.device("cuda")
    runner = _StubRunner(None)
    engine = _forward_engine(spec_graph_runner=runner)
    batch = _decode(1, device)

    engine.forward_batch(batch, engine.sampler.prepare(batch))

    assert runner.decodes == 0

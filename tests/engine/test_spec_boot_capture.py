"""Boot-time capture of every armed speculative graph width.

The widths used to capture LAZILY, on each one's first live step -- which is always AFTER a
prefill. At long context the first request's activation spike takes free VRAM below the capture
guard, so admission failed (MEMORY_ADMISSION), the retry budget drained, and the boot degenerated
to eager at roughly half the graphed rate without ever failing loudly.

``Engine._capture_spec_graphs_at_boot`` moves those captures beside the decode graphs, where
memory is still fresh. What that costs must be nothing:

* a width that cannot capture at boot reaches its first live step exactly as capturable as it
  was before (the attempt is refunded, no permanent verdict is recorded);
* the boot never aborts, whatever the capture does;
* the warm-up forwards execute against the dummy request, and every mark they leave -- the
  page-table row, the GDN slot, the MoE offload cache -- is wound back.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req
from freetoken.engine.engine import Engine
from freetoken.engine.spec_graph import SpecVerifyGraphRunner

_CUDA = torch.cuda.is_available()

PAGE_TABLE_WIDTH = 64
DUMMY_TABLE_IDX = 1
DUMMY_KV_SLOT = 999


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
    """Both capture seams: the decode one width 1 takes, the verify one the rest take."""

    def __init__(self):
        self.metadata = []
        self.decode_captures = []
        self.verify_prepared = []
        self.finished = []
        self.reset_count = 0

    def prepare_metadata(self, batch):
        self.metadata.append(int(batch.input_ids.shape[0]))
        batch.attn_metadata = SimpleNamespace(width=int(batch.input_ids.shape[0]))

    def prepare_for_capture(self, batch):
        self.decode_captures.append(batch.padded_size)
        self.prepare_metadata(batch)

    def prepare_for_replay(self, batch):
        pass

    def prepare_mtp_verify_graph(self, batch):
        self.verify_prepared.append(int(batch.input_ids.shape[0]))

    def stage_mtp_verify_graph(self, runtime_batch, static_batch):
        pass

    def finish_mtp_verify_graph_capture(self, width):
        self.finished.append(width)

    def discard_mtp_verify_graph(self, width):
        pass

    def reset_mtp_verify_graph(self):
        self.reset_count += 1

    def mtp_verify_graph_bytes(self):
        return 0


def _outputs(batch):
    """Every per-step input reaches every output, so a missed refill could not hide."""
    width = int(batch.input_ids.shape[0])
    row = torch.stack(
        (
            batch.input_ids.float(),
            batch.positions.float(),
            batch.out_loc.float(),
            batch.linear_table_idx[:1].float().expand(width),
        ),
        dim=-1,
    )
    return torch.cat((row, row[:, :1] * 2.0), dim=-1), row * 3.0, row * 5.0


class _Model:
    """The target, plus an optional GDN-shaped state the forward ADVANCES like the real one."""

    def __init__(self, ctx, state=None):
        self.ctx = ctx
        self.state = state
        self.calls = []
        self.ple_rows = []

    def forward_mtp_capture(self, *, all_row_logits=False):
        batch = self.ctx.batch
        width = int(batch.input_ids.shape[0])
        # width 1 records the ORDINARY decode forward; the verify widths need every row's logits
        assert all_row_logits == (width > 1)
        self.calls.append(width)
        if self.state is not None:
            self.state.add_(1)
        return _outputs(batch)

    def prepare_cuda_graph_capture(self, batch):
        # the PLE staging seam: one buffer per row count the graph will look up
        self.ple_rows.append(int(batch.input_ids.shape[0]))

    def prepare_cuda_graph_replay(self, batch):
        pass


class _Ladder:
    """``SpecStateLadder.borrow_snapshot``, reduced to what a boot capture uses."""

    def __init__(self, state=None):
        self.state = state
        self.snapshot = None
        self.borrowed = []
        self.released = 0
        self.restores = 0
        self.live = False

    @contextmanager
    def borrow_snapshot(self, req):
        assert not self.live, "a speculative step is already in flight on this ladder"
        self.borrowed.append(req)
        self.live = True
        if self.state is not None:
            self.snapshot = self.state.clone()
        try:
            yield self.restore_snapshot
        finally:
            self.live = False
            self.released += 1

    def restore_snapshot(self):
        self.restores += 1
        if self.state is not None:
            self.state.copy_(self.snapshot)


class _MoeCache:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class _SpyRunner:
    """A runner that records what boot capture asks of it and captures nothing."""

    def __init__(self, widths):
        self.widths = tuple(widths)
        self.captured = []
        self.refunded = []

    def capture_pending(self, width):
        return True

    def capture(self, batch, *, restore_state=None):
        from freetoken.engine.spec_graph import MTPGraphCaptureResult

        width = int(batch.input_ids.shape[0])
        self.captured.append((width, restore_state))
        return MTPGraphCaptureResult(width=width, status="captured", reason="")

    def refund_attempt(self, width):
        self.refunded.append(width)


def _dummy_req(device):
    req = Req(
        input_ids=torch.tensor([0], dtype=torch.int32),
        table_idx=DUMMY_TABLE_IDX,
        cached_len=0,
        output_len=1,
        uid=-1,
        sampling_params=None,
        cache_handle=None,
    )
    req.linear_slot_idx = 0
    return req


class _Forbidden:
    """Any touch is a failure: boot capture must never reach the draft head."""

    def __getattr__(self, name):
        raise AssertionError(f"boot capture touched the draft head ({name})")


def _engine(
    runner,
    *,
    device,
    ladder=None,
    moe_cache=None,
    graphable=True,
    max_seq_len=PAGE_TABLE_WIDTH,
    linear_pool=object(),
):
    engine = object.__new__(Engine)
    engine.device = device
    engine.stream = torch.cuda.current_stream() if device.type == "cuda" else None
    engine.max_seq_len = max_seq_len
    engine.page_table = torch.full(
        (DUMMY_TABLE_IDX + 1, PAGE_TABLE_WIDTH), DUMMY_KV_SLOT, dtype=torch.int32, device=device
    )
    engine.dummy_req = _dummy_req(device)
    engine.linear_state_pool = linear_pool
    engine.spec_state_ladder = ladder
    engine.spec_graph_runner = runner
    engine.moe_offload_cache = moe_cache
    engine.spec_draft = _Forbidden()
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: graphable)
    engine.attn_backend = runner.attn_backend if hasattr(runner, "attn_backend") else _Attention()
    return engine


def _runner(device, widths, *, ctx=None, model=None, attention=None, guard_bytes=0):
    ctx = ctx if ctx is not None else _Context()
    return SpecVerifyGraphRunner(
        target_ctx=ctx,
        target_model=model if model is not None else _Model(ctx),
        attn_backend=attention if attention is not None else _Attention(),
        device=device,
        widths=widths,
        guard_bytes=guard_bytes,
    )


# ------------------------------------------------------------------ the batch boot builds


def test_the_verify_batch_boot_builds_is_the_shape_the_scheduler_would_have_built():
    device = torch.device("cpu")
    engine = _engine(_SpyRunner((1, 2, 3)), device=device)
    row = engine.page_table[DUMMY_TABLE_IDX]

    batch = engine._spec_boot_batch(3, 8, row)

    assert batch.is_prefill and batch.size == 1 and batch.padded_size == 1
    assert batch.mtp_verify is True
    assert batch.emit_width == 3
    assert batch.reqs[0].extend_len == 3  # what the runner validates the width against
    assert batch.reqs[0].table_idx == engine.dummy_req.table_idx
    assert batch.reqs[0].linear_slot_idx == engine.dummy_req.linear_slot_idx
    assert batch.positions.tolist() == [8, 9, 10]
    assert batch.out_loc.tolist() == [8, 9, 10]  # read back from the row, as the scheduler does
    assert batch.rope_positions is None
    assert batch.linear_table_idx.tolist() == [0]


def test_the_width_one_batch_is_decode_shaped_and_keeps_the_dummy_kv_slot():
    device = torch.device("cpu")
    engine = _engine(_SpyRunner((1, 2)), device=device)
    row = engine.page_table[DUMMY_TABLE_IDX]

    batch = engine._spec_boot_batch(1, 8, row)

    assert batch.is_decode and batch.padded_size == 1
    assert batch.mtp_verify is False
    assert batch.reqs[0] is engine.dummy_req
    assert batch.out_loc.tolist() == [DUMMY_KV_SLOT]
    assert row.tolist() == [DUMMY_KV_SLOT] * PAGE_TABLE_WIDTH  # untouched by the decode shape


def test_the_width_one_batch_is_skipped_when_the_plain_decode_graphs_are_off():
    """The width-1 graph stages its addressing through the backend's persistent decode buffers,
    which exist only when the ordinary decode graphs were armed."""
    device = torch.device("cpu")
    engine = _engine(_SpyRunner((1, 2)), device=device, graphable=False)

    assert engine._spec_boot_batch(1, 8, engine.page_table[DUMMY_TABLE_IDX]) is None


# --------------------------------------------------------------------------- the boot pass


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_boot_capture_captures_every_armed_width():
    device = torch.device("cuda")
    ctx = _Context()
    model = _Model(ctx)
    attention = _Attention()
    runner = _runner(device, (1, 2, 3, 4), ctx=ctx, model=model, attention=attention)
    engine = _engine(runner, device=device)
    engine.attn_backend = attention

    engine._capture_spec_graphs_at_boot()

    assert runner.graph_count == 4
    for width in (1, 2, 3, 4):
        assert runner.support(width).status == "captured"
        assert runner.capture_pending(width) is False
    # one warm-up plus one recorded pass per width, and the PLE staging seam per row count
    assert sorted(model.calls) == [1, 1, 2, 2, 3, 3, 4, 4]
    assert sorted(set(model.ple_rows)) == [1, 2, 3, 4]
    assert attention.decode_captures == [1]  # only width 1 takes the decode seam
    assert attention.verify_prepared == [2, 3, 4]


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_the_widths_come_from_the_runner_and_nothing_hard_codes_two_to_four():
    device = torch.device("cuda")
    ctx = _Context()
    model = _Model(ctx)
    runner = _runner(device, (1, 2, 5), ctx=ctx, model=model)
    engine = _engine(runner, device=device)

    engine._capture_spec_graphs_at_boot()

    assert runner.graph_count == 3
    assert sorted(set(model.calls)) == [1, 2, 5]


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_replay_after_a_boot_capture_still_reproduces_the_eager_forward():
    """Boot capture must leave a graph the live step can replay, not merely a captured one."""
    device = torch.device("cuda")
    ctx = _Context()
    runner = _runner(device, (2,), ctx=ctx, model=_Model(ctx))
    engine = _engine(runner, device=device)
    engine._capture_spec_graphs_at_boot()
    assert runner.graph_count == 1

    live = engine._spec_boot_batch(2, 20, engine.page_table[DUMMY_TABLE_IDX])
    logits, hidden = runner.replay(live)

    want_logits, want_hidden, _ = _outputs(live)
    assert torch.equal(logits, want_logits)
    assert torch.equal(hidden, want_hidden)


# --------------------------------------------------------------- failure stays survivable


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_an_admission_failure_at_boot_leaves_the_width_exactly_as_capturable():
    device = torch.device("cuda")
    ctx = _Context()
    model = _Model(ctx)
    # a guard no card can satisfy: every width fails admission before it allocates anything
    runner = _runner(device, (1, 2, 3), ctx=ctx, model=model, guard_bytes=1 << 62)
    engine = _engine(runner, device=device)

    engine._capture_spec_graphs_at_boot()

    assert runner.graph_count == 0
    assert model.calls == []  # admission is decided before the warm-up forward
    for width in (1, 2, 3):
        assert runner.last_attempt(width).reason == "MEMORY_ADMISSION"
        assert runner.support(width) is None  # no permanent verdict from a boot-time failure
        assert runner.capture_pending(width) is True
        assert runner._attempts.get(width, 0) == 0  # the attempt was refunded

    # ...and the first live step captures exactly as it would have without boot capture
    runner.guard_bytes = 0
    batch = engine._spec_boot_batch(2, 20, engine.page_table[DUMMY_TABLE_IDX])
    assert runner.capture(batch).status == "captured"


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_permanently_unsupported_width_keeps_its_verdict_and_the_boot_survives():
    class _Broken(_Model):
        def forward_mtp_capture(self, *, all_row_logits=False):
            raise RuntimeError("synthetic capture failure")

    device = torch.device("cuda")
    ctx = _Context()
    runner = _runner(device, (2, 3), ctx=ctx, model=_Broken(ctx))
    engine = _engine(runner, device=device)

    engine._capture_spec_graphs_at_boot()  # must not raise

    assert runner.graph_count == 0
    for width in (2, 3):
        assert runner.last_attempt(width).reason.startswith("CAPTURE_FAILED:RuntimeError:")


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_batch_that_cannot_be_built_is_logged_and_the_boot_continues():
    class _RaisingAttention(_Attention):
        def prepare_metadata(self, batch):
            if int(batch.input_ids.shape[0]) == 2:
                raise RuntimeError("synthetic metadata failure")
            super().prepare_metadata(batch)

    device = torch.device("cuda")
    ctx = _Context()
    attention = _RaisingAttention()
    runner = _runner(device, (2, 3), ctx=ctx, model=_Model(ctx), attention=attention)
    engine = _engine(runner, device=device)
    engine.attn_backend = attention

    engine._capture_spec_graphs_at_boot()  # must not raise

    assert runner.graph_count == 1
    assert runner.support(3).status == "captured"
    assert runner.capture_pending(2) is True


# ------------------------------------------------------------------------ what boot restores


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_boot_capture_restores_the_dummy_row_the_gdn_state_and_the_moe_cache():
    device = torch.device("cuda")
    ctx = _Context()
    state = torch.arange(8, dtype=torch.float32, device=device)
    before_state = state.clone()
    model = _Model(ctx, state=state)
    ladder = _Ladder(state)
    moe_cache = _MoeCache()
    runner = _runner(device, (1, 2, 3), ctx=ctx, model=model)
    engine = _engine(runner, device=device, ladder=ladder, moe_cache=moe_cache)
    before_table = engine.page_table.clone()

    engine._capture_spec_graphs_at_boot()

    assert runner.graph_count == 3
    assert torch.equal(engine.page_table, before_table)  # the dummy row is back
    assert torch.equal(state, before_state)  # the warm-up's GDN advance is wound back
    assert moe_cache.resets == 1
    assert ladder.borrowed == [engine.dummy_req] * 3
    assert ladder.released == 3 and ladder.restores == 3
    assert not ladder.live


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_the_row_is_restored_even_when_a_width_fails_mid_capture():
    class _Broken(_Model):
        def forward_mtp_capture(self, *, all_row_logits=False):
            raise RuntimeError("synthetic capture failure")

    device = torch.device("cuda")
    ctx = _Context()
    runner = _runner(device, (2,), ctx=ctx, model=_Broken(ctx))
    engine = _engine(runner, device=device)
    before_table = engine.page_table.clone()

    engine._capture_spec_graphs_at_boot()

    assert torch.equal(engine.page_table, before_table)


def test_without_a_ladder_the_capture_still_runs_and_asks_for_no_wind_back():
    runner = _SpyRunner((2,))
    engine = _engine(runner, device=torch.device("cpu"))
    batch = engine._spec_boot_batch(2, 8, engine.page_table[DUMMY_TABLE_IDX])

    engine._capture_spec_width(runner, batch)

    assert runner.captured == [(2, None)]


def test_the_ladder_snapshot_wraps_every_boot_capture():
    ladder = _Ladder()
    runner = _SpyRunner((2,))
    engine = _engine(runner, device=torch.device("cpu"), ladder=ladder)
    batch = engine._spec_boot_batch(2, 8, engine.page_table[DUMMY_TABLE_IDX])

    engine._capture_spec_width(runner, batch)

    assert ladder.borrowed == [engine.dummy_req]
    assert runner.captured == [(2, ladder.restore_snapshot)]


# ------------------------------------------------------------------------- the off-switch


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_the_off_switch_leaves_every_width_to_lazy_capture(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MTP_SPEC_BOOT_CAPTURE", "0")
    device = torch.device("cuda")
    ctx = _Context()
    model = _Model(ctx)
    runner = _runner(device, (1, 2, 3), ctx=ctx, model=model)
    engine = _engine(runner, device=device)

    engine._capture_spec_graphs_at_boot()

    assert runner.graph_count == 0
    assert model.calls == []
    for width in (1, 2, 3):
        assert runner.capture_pending(width) is True
        assert runner.last_attempt(width) is None


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_boot_capture_is_on_by_default(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MTP_SPEC_BOOT_CAPTURE", raising=False)
    device = torch.device("cuda")
    ctx = _Context()
    runner = _runner(device, (2,), ctx=ctx, model=_Model(ctx))
    engine = _engine(runner, device=device)

    engine._capture_spec_graphs_at_boot()

    assert runner.graph_count == 1


# ------------------------------------------------------------------------- the quiet paths


def test_an_unarmed_boot_captures_nothing():
    engine = _engine(None, device=torch.device("cpu"))
    engine.spec_graph_runner = None

    engine._capture_spec_graphs_at_boot()  # must not raise


def test_without_cuda_boot_capture_does_nothing():
    runner = _SpyRunner((1, 2))
    engine = _engine(runner, device=torch.device("cpu"))

    engine._capture_spec_graphs_at_boot()

    assert runner.captured == []


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_a_context_too_short_for_the_widest_dummy_prefix_is_skipped():
    device = torch.device("cuda")
    runner = _SpyRunner((1, 2, 3, 4))
    engine = _engine(runner, device=device, max_seq_len=4)

    engine._capture_spec_graphs_at_boot()

    assert runner.captured == []

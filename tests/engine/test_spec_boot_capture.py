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

And one thing it must COST, which a first cut got wrong and only a live boot caught: a verify
capture has to run with the state ladder ARMED (``SpecStateLadder.begin``), exactly as the lazy
path did by accident because the scheduler begins the step before the forward that triggers the
capture. The per-layer stash is a CUDA copy, so it is part of what capture records; a graph
recorded unarmed replays a step nothing can roll back. The last section here pins that against
the REAL ladder over a real state pool, because a stub ladder is what let it through.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req
from freetoken.engine.engine import Engine
from freetoken.engine.spec_graph import SpecVerifyGraphRunner

# the shared qwen4_exp toy geometry lives beside its own suite; tests/ is not a package, so put
# it on the path explicitly rather than depending on which suite pytest collected first
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)

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
    """The target, plus an optional GDN-shaped state the forward ADVANCES like the real one.

    It stashes through ``batch.spec_capture`` exactly where the real layers do -- ``gdn.forward``
    on the PREFILL branch only, ``ple.forward`` on the mere presence of the ladder -- because the
    stash is a CUDA copy and therefore part of what capture RECORDS.
    """

    def __init__(self, ctx, state=None):
        self.ctx = ctx
        self.state = state
        self.calls = []
        self.ple_rows = []
        self.armed = []

    def forward_mtp_capture(self, *, all_row_logits=False):
        batch = self.ctx.batch
        width = int(batch.input_ids.shape[0])
        # width 1 records the ORDINARY decode forward; the verify widths need every row's logits
        assert all_row_logits == (width > 1)
        self.calls.append(width)
        self.armed.append(batch.spec_capture)
        if batch.spec_capture is not None:
            batch.spec_capture.stash(batch.input_ids)
        if self.state is not None:
            self.state.add_(1)
        return _outputs(batch)

    def prepare_cuda_graph_capture(self, batch):
        # the PLE staging seam: one buffer per row count the graph will look up
        self.ple_rows.append(int(batch.input_ids.shape[0]))

    def prepare_cuda_graph_replay(self, batch):
        pass


class _Ladder:
    """``SpecStateLadder``'s three entry points, reduced to what a boot capture uses.

    ``begin`` is the load-bearing one: it ARMS ``batch.spec_capture``, and a verify graph
    recorded without it contains no stash writes at all.
    """

    def __init__(self, state=None, arena=None):
        self.state = state
        self.arena = arena
        self.snapshot = None
        self.begun = []
        self.rolled = []
        self.borrowed = []
        self.released = 0
        self.restores = 0
        self.stashes = 0
        self.live = False

    def begin(self, req, batch):
        assert not self.live, "a speculative step is already in flight on this ladder"
        self.begun.append((req, batch.emit_width))
        self.live = True
        self._snapshot()
        batch.spec_capture = self

    def rollback(self, req, accepted):
        assert self.live, "no speculative step is in flight on this ladder"
        self.rolled.append((req, accepted))
        self.restore_snapshot()
        self.live = False

    @contextmanager
    def borrow_snapshot(self, req):
        assert not self.live, "a speculative step is already in flight on this ladder"
        self.borrowed.append(req)
        self.live = True
        self._snapshot()
        try:
            yield self.restore_snapshot
        finally:
            self.live = False
            self.released += 1

    def _snapshot(self):
        if self.state is not None:
            self.snapshot = self.state.clone()

    def restore_snapshot(self):
        self.restores += 1
        if self.state is not None:
            self.state.copy_(self.snapshot)

    def stash(self, rows):
        """The arena write the real ``stash_gdn`` / ``stash_ple`` do, and that capture records."""
        self.stashes += 1
        if self.arena is not None:
            self.arena[: rows.shape[0]].copy_(rows)


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
    # width 1 borrows; the verify widths take the full begin -> settle a live step takes
    assert ladder.borrowed == [engine.dummy_req]
    assert ladder.released == 1
    assert [width for _req, width in ladder.begun] == [2, 3]
    assert [accepted for _req, accepted in ladder.rolled] == [0, 0]  # the complete undo
    # one restore per capture (the warm-up's wind-back), plus the verify widths' settle
    assert ladder.restores == 3 + 2
    assert not ladder.live


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_every_verify_width_is_captured_with_the_stash_hooks_armed():
    """The regression this suite exists for. A verify graph recorded without ``begin`` carries
    no stash writes, so every later replay leaves the ladder's arena empty and the step's settle
    dies in ``rollback`` with "GDN layer index 0 never stashed". Width 1 must stay UNARMED: it is
    the ordinary decode forward, which never rolls back and must never touch the arena."""
    device = torch.device("cuda")
    ctx = _Context()
    model = _Model(ctx)
    ladder = _Ladder()
    runner = _runner(device, (1, 2, 3), ctx=ctx, model=model)
    engine = _engine(runner, device=device, ladder=ladder)

    engine._capture_spec_graphs_at_boot()

    armed = dict(zip(model.calls, model.armed))  # width -> the ladder the forward saw
    assert armed[1] is None
    assert armed[2] is ladder and armed[3] is ladder
    # one warm-up stash plus one recorded stash for each verify width
    assert ladder.stashes == 4


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_the_captured_verify_graph_replays_the_stash_writes():
    """Not just armed at capture time: the stash copies must be IN the graph, so a replay --
    which runs no Python at all -- still fills the arena with THIS step's rows."""
    device = torch.device("cuda")
    ctx = _Context()
    arena = torch.zeros(4, dtype=torch.int32, device=device)
    ladder = _Ladder(arena=arena)
    runner = _runner(device, (3,), ctx=ctx, model=_Model(ctx))
    engine = _engine(runner, device=device, ladder=ladder)
    engine._capture_spec_graphs_at_boot()
    assert runner.graph_count == 1

    arena.zero_()
    live = engine._spec_boot_batch(3, 20, engine.page_table[DUMMY_TABLE_IDX])
    live.input_ids = torch.tensor([11, 12, 13], dtype=torch.int32, device=device)
    runner.replay(live)

    assert arena.tolist() == [11, 12, 13, 0]


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


def test_a_verify_capture_begins_and_settles_the_ladder_like_a_live_step():
    ladder = _Ladder()
    runner = _SpyRunner((2,))
    engine = _engine(runner, device=torch.device("cpu"), ladder=ladder)
    batch = engine._spec_boot_batch(2, 8, engine.page_table[DUMMY_TABLE_IDX])

    engine._capture_spec_width(runner, batch)

    assert ladder.begun == [(batch.reqs[0], 2)]
    assert batch.spec_capture is ladder  # the stash hooks the capture records
    assert runner.captured == [(2, ladder.restore_snapshot)]
    assert ladder.rolled == [(batch.reqs[0], 0)]
    assert ladder.borrowed == []  # begin's own snapshot supersedes the borrow
    assert not ladder.live


def test_the_width_one_capture_borrows_instead_and_arms_nothing():
    ladder = _Ladder()
    runner = _SpyRunner((1,))
    engine = _engine(runner, device=torch.device("cpu"), ladder=ladder)
    batch = engine._spec_boot_batch(1, 8, engine.page_table[DUMMY_TABLE_IDX])

    engine._capture_spec_width(runner, batch)

    assert ladder.borrowed == [engine.dummy_req]
    assert ladder.begun == [] and ladder.rolled == []
    assert batch.spec_capture is None  # a decode graph must never write the arena
    assert runner.captured == [(1, ladder.restore_snapshot)]


def test_the_ladder_is_settled_even_when_the_capture_raises():
    class _Boom(_SpyRunner):
        def capture(self, batch, *, restore_state=None):
            raise RuntimeError("synthetic capture explosion")

    ladder = _Ladder()
    runner = _Boom((2,))
    engine = _engine(runner, device=torch.device("cpu"), ladder=ladder)
    batch = engine._spec_boot_batch(2, 8, engine.page_table[DUMMY_TABLE_IDX])

    with pytest.raises(RuntimeError, match="synthetic capture explosion"):
        engine._capture_spec_width(runner, batch)

    assert ladder.rolled == [(batch.reqs[0], 0)]
    assert not ladder.live  # a stuck ladder would refuse every later step


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


# ------------------------------------------------ the REAL ladder over a real state pool

LIVE_SLOT = 3
REAL_WIDTH = 3
REAL_DTYPE = torch.bfloat16


def _real_config():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.qwen4_exp.config import parse_config
    from models.qwen4_exp.common import toy_hf_config

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    return parse_config(toy_hf_config())


def _real_pool(config, device):
    from freetoken.kvcache.linear_state_pool import LinearStatePool

    return LinearStatePool(
        config.linear_attention_group(),
        8,
        REAL_DTYPE,
        device,
        tp_size=1,
        slot_states=config.slot_states,
    )


def _families(pool, slot):
    out = {
        "conv": pool.conv_states[:, slot].clone(),
        "recurrent": pool.recurrent_states[:, slot].clone(),
    }
    for name, tensor in pool.slot_states.items():
        out[name] = tensor[:, slot].clone()
    return out


def _seed_slot(pool, slot, seed):
    gen = torch.Generator(device=pool.device).manual_seed(seed)
    pool.conv_states[:, slot].normal_(0.0, 0.5, generator=gen)
    pool.recurrent_states[:, slot].normal_(0.0, 0.5, generator=gen)
    for tensor in pool.slot_states.values():
        if tensor.is_floating_point():
            tensor[:, slot].normal_(0.0, 0.5, generator=gen)
        else:
            tensor[:, slot].random_(0, 32, generator=gen)


class _LinearWorld:
    """A capturable stand-in for the qwen4_exp linear stack.

    It stashes what ``gdn.forward`` and ``ple.forward`` stash, with the same call shapes, and
    advances the pool slot the way the chunked prefill does -- through the batch's own
    ``cache_indices``, so the slot is a per-replay refill and never a capture-time constant.
    Everything it computes derives from ``batch.input_ids``, so a graph that failed to refill
    would stash visibly different rows.
    """

    def __init__(self, ctx, pool, config, device):
        from freetoken.models.qwen4_exp.config import PLE_CONV_STATE

        self.ctx = ctx
        self.pool = pool
        self.device = device
        self.layer_ids = tuple(config.linear_attention_group().layer_ids)
        _, _, self.conv_dim, self.km1 = pool.conv_states.shape
        _, _, self.v_heads, self.key_dim, self.value_dim = pool.recurrent_states.shape
        self.scale = float(self.key_dim) ** -0.5
        self.ple_layer_ids = pool.slot_state_layer_ids(PLE_CONV_STATE)
        self.ple_width = pool.slot_states[PLE_CONV_STATE].shape[2]
        gen = torch.Generator(device=device).manual_seed(11)
        self.params = {
            layer_id: (
                torch.empty(self.v_heads, device=device, dtype=torch.float32)
                .uniform_(0.01, 4.0, generator=gen)
                .log_(),
                torch.empty(self.v_heads, device=device, dtype=torch.float32).uniform_(
                    -1.0, 1.0, generator=gen
                ),
            )
            for layer_id in self.layer_ids
        }
        self._conv_ramp = torch.arange(
            self.conv_dim, device=device, dtype=torch.float32
        ).mul_(0.01)
        self._head_ramp = torch.arange(
            self.v_heads, device=device, dtype=torch.float32
        ).mul_(0.05)
        self._ple_ramp = torch.arange(
            self.ple_width, device=device, dtype=torch.float32
        ).mul_(0.02)

    def forward_mtp_capture(self, *, all_row_logits=False):
        batch = self.ctx.batch
        rows = int(batch.input_ids.shape[0])
        idx = batch.fla_metadata.cache_indices.to(torch.int64)
        seed = batch.input_ids.to(torch.float32).reshape(rows, 1).mul(0.03)
        for layer_id in self.layer_ids:
            li = self.pool.local_index(layer_id)
            conv_in = (seed + self._conv_ramp + li).to(REAL_DTYPE)
            mixed = (seed * 0.5 + self._conv_ramp * 0.25 + li).to(REAL_DTYPE)
            a = (seed + self._head_ramp).to(REAL_DTYPE)
            b = (seed * 0.25 + self._head_ramp).to(REAL_DTYPE)
            A_log, dt_bias = self.params[layer_id]
            if batch.spec_capture is not None:
                batch.spec_capture.stash_gdn(
                    layer_id,
                    conv_in=conv_in,
                    mixed=mixed,
                    a=a,
                    b=b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=self.scale,
                )
            # the forward leaves the slot holding every row; the settle discards this, but it
            # must MOVE or a missing restore could not show
            self.pool.conv_states[li].index_copy_(
                0,
                idx,
                conv_in[-1]
                .reshape(1, self.conv_dim, 1)
                .expand(1, self.conv_dim, self.km1)
                .contiguous(),
            )
            self.pool.recurrent_states[li].index_copy_(
                0,
                idx,
                conv_in[-1, : self.v_heads]
                .float()
                .reshape(1, self.v_heads, 1, 1)
                .expand(1, self.v_heads, self.key_dim, self.value_dim)
                .contiguous(),
            )
        for layer_id in self.ple_layer_ids:
            x = (seed + self._ple_ramp).to(REAL_DTYPE)
            if batch.spec_capture is not None:
                batch.spec_capture.stash_ple(layer_id, x)
        logits = seed.expand(rows, 4).contiguous()
        return logits, logits * 3.0, logits * 5.0

    def prepare_cuda_graph_capture(self, batch):
        pass

    def prepare_cuda_graph_replay(self, batch):
        pass


def _live_verify_batch(device, ids, *, cached_len=32):
    """One real speculative step's batch, on the LIVE slot -- the scheduler's shape."""
    from freetoken.attention.linear import build_fla_metadata

    width = len(ids)
    req = Req(
        input_ids=torch.zeros(cached_len + width, dtype=torch.int32),
        table_idx=DUMMY_TABLE_IDX,
        cached_len=cached_len,
        output_len=width,
        uid=0,
        sampling_params=None,
        cache_handle=None,
    )
    req.linear_slot_idx = LIVE_SLOT
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.mtp_verify = True
    batch.emit_width = width
    batch.input_ids = torch.tensor(ids, dtype=torch.int32, device=device)
    batch.positions = torch.arange(
        cached_len, cached_len + width, dtype=torch.int32, device=device
    )
    batch.out_loc = torch.arange(
        cached_len, cached_len + width, dtype=torch.int32, device=device
    )
    batch.linear_table_idx = torch.tensor([LIVE_SLOT], dtype=torch.int32, device=device)
    batch.fla_metadata = build_fla_metadata(batch, device)
    batch.attn_metadata = None
    return req, batch


def _real_world(device, widths):
    from freetoken.engine.spec_state_ladder import SpecStateLadder

    config = _real_config()
    pool = _real_pool(config, device)
    ctx = _Context()
    model = _LinearWorld(ctx, pool, config, device)
    ladder = SpecStateLadder(pool, max(widths))
    runner = _runner(device, widths, ctx=ctx, model=model)
    engine = _engine(runner, device=device, ladder=ladder, linear_pool=pool)
    return SimpleNamespace(
        engine=engine, runner=runner, ladder=ladder, pool=pool, model=model, ctx=ctx
    )


def _settle(world, ids, accepted):
    """One live speculative step through the boot-captured graph, then its settle."""
    req, batch = _live_verify_batch(world.engine.device, ids)
    world.ladder.begin(req, batch)
    world.runner.replay(batch)
    world.ladder.rollback(req, accepted)
    return _families(world.pool, LIVE_SLOT)


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_the_real_ladder_comes_out_of_boot_capture_stashed_and_settled():
    device = torch.device("cuda")
    world = _real_world(device, (REAL_WIDTH,))
    dummy_slot = world.engine.dummy_req.linear_slot_idx
    _seed_slot(world.pool, dummy_slot, seed=3)
    before = _families(world.pool, dummy_slot)

    world.engine._capture_spec_graphs_at_boot()

    assert world.runner.graph_count == 1
    # every GDN layer stashed -- exactly what rollback asserts on, and what an unarmed capture
    # leaves empty
    assert all(params is not None for params in world.ladder._params)
    # ...and the step is settled, so the next live begin is not refused
    assert world.ladder._live is None and world.ladder._width == 0
    after = _families(world.pool, dummy_slot)
    for name in before:
        assert torch.equal(after[name], before[name]), f"boot capture moved {name}"


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
@pytest.mark.parametrize("accepted", (0, 1, 2, REAL_WIDTH))
def test_a_live_step_on_a_boot_captured_graph_settles_like_the_eager_one(accepted):
    """The regression, end to end: boot-capture a verify width, then run a real live step
    (begin -> graph replay -> rollback) and require the settled slot to be what the SAME step
    run eagerly settles to. Before the ladder was armed at boot the replay stashed nothing and
    this died in ``_replay_recurrent`` with "GDN layer index 0 never stashed"."""
    device = torch.device("cuda")
    ids = [17, 23, 29]

    graphed = _real_world(device, (REAL_WIDTH,))
    _seed_slot(graphed.pool, graphed.engine.dummy_req.linear_slot_idx, seed=3)
    _seed_slot(graphed.pool, LIVE_SLOT, seed=9)
    graphed.engine._capture_spec_graphs_at_boot()
    assert graphed.runner.graph_count == 1
    got = _settle(graphed, ids, accepted)

    eager = _real_world(device, (REAL_WIDTH,))
    _seed_slot(eager.pool, LIVE_SLOT, seed=9)
    ref_req, ref_batch = _live_verify_batch(device, ids)
    eager.ladder.begin(ref_req, ref_batch)
    with eager.ctx.forward_batch(ref_batch):
        eager.model.forward_mtp_capture(all_row_logits=True)
    eager.ladder.rollback(ref_req, accepted)
    want = _families(eager.pool, LIVE_SLOT)

    assert set(got) == set(want)
    for name in want:
        assert torch.equal(got[name], want[name]), (
            f"{name} differs after a graphed step's rollback; max |delta| "
            f"{(got[name].float() - want[name].float()).abs().max().item()}"
        )


@pytest.mark.skipif(not _CUDA, reason="CUDA is required")
def test_the_boot_captured_graph_stashes_this_step_and_not_the_boot_dummys_rows():
    """A graph that baked the boot warm-up's rows instead of refilling them would settle every
    live step to the same state, whatever the drafts were."""
    device = torch.device("cuda")
    world = _real_world(device, (REAL_WIDTH,))
    _seed_slot(world.pool, world.engine.dummy_req.linear_slot_idx, seed=3)
    world.engine._capture_spec_graphs_at_boot()

    settled = []
    for ids in ([17, 23, 29], [2, 3, 5]):
        _seed_slot(world.pool, LIVE_SLOT, seed=9)
        settled.append(_settle(world, ids, 2))

    assert not torch.equal(settled[0]["recurrent"], settled[1]["recurrent"])
    assert not torch.equal(settled[0]["conv"], settled[1]["conv"])

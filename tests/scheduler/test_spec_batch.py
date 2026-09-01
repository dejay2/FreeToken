"""One speculative decode step: build a w-row batch on the request's REAL KV, then undo it.

``_prepare_spec_batch`` is the decode-batch builder generalized from one row to
``w = 1 + len(draft_tokens)``. Unlike the shadow observer's verify batch it redirects nothing:
the request's own ``table_idx``, its own page-table row, its own pages. That redirection lives
outside the ``mtp_verify`` flag, and integrated mode simply does not do it.

``_rollback_spec_tokens`` settles the step once the accepted run is known. It owns the
KV/page/length bookkeeping only -- the GDN recurrent/conv/PLE state still holds all w rows and
is Phase 3's problem, reached through the ``state_rollback`` seam.

Two invariants are pinned hard because nothing else would catch them:
  * keeping ``accepted`` rows leaves the request exactly where ``accepted`` plain decode steps
    would have -- free-list ORDER included, so a full rollback is byte-identical to never
    having run the step;
  * a request with speculative rows in flight can never reach a radix commit, which re-points
    its page-table row -- undoable by no device_len rewind (design risk #4).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.engine.config import SpecDecodeConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.scheduler import Scheduler

CPU = torch.device("cpu")
TABLE_WIDTH = 512


@pytest.fixture(autouse=True)
def _no_ctx_leak():
    import freetoken.core as core

    yield
    core._GLOBAL_CTX = None  # only the GDN-pool scenarios set one


def _gdn_pool():
    """A tiny CPU LinearStatePool -- enough for build_fla_metadata's track-checkpoint path,
    which reads the conv history width off the live pool."""
    import freetoken.core as core
    from freetoken.core import Context, set_global_ctx
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=1, num_value_heads=1,
        key_head_dim=8, value_head_dim=8, conv_kernel_dim=4, output_gate="sigmoid",
    )
    pool = LinearStatePool(group, 4, torch.float32, CPU, tp_size=1)
    core._GLOBAL_CTX = None
    ctx = Context(page_size=64)
    ctx.linear_state_pool = pool
    set_global_ctx(ctx)
    return pool


def _scheduler(*, page_size=64, num_pages=8, depth=3, enabled=True, cache_type="naive",
               gdn=False):
    page_table = torch.zeros(2, TABLE_WIDTH, dtype=torch.int32)
    cm = CacheManager(num_pages, page_size, page_table, cache_type)
    stub = Scheduler.__new__(Scheduler)
    stub.device = CPU
    stub.cache_manager = cm
    stub.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
    stub.prepared = []
    stub.engine = SimpleNamespace(
        page_table=page_table,
        linear_state_pool=_gdn_pool() if gdn else None,
        attn_backend=SimpleNamespace(prepare_metadata=stub.prepared.append),
        sampler=SimpleNamespace(prepare=lambda batch: "sample-args"),
    )
    stub.config = SimpleNamespace(
        page_size=page_size,
        spec_decode=SpecDecodeConfig(enabled=enabled, depth=depth),
    )
    return stub


def _decode_req(stub, *, prompt_len=64, output_len=64, table_idx=0, last_token=5):
    """A request parked exactly where a decode batch picks it up: (cached_len, device_len)
    == (P, P + 1), pages covering div_ceil(P, page_size), host ids caught up."""
    req = Req(
        input_ids=torch.arange(prompt_len, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=0,
        output_len=output_len,
        uid=1,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=stub.cache_manager.prefix_cache.match_prefix(
            torch.zeros(0, dtype=torch.int32)
        ).cuda_handle,
    )
    stub.cache_manager.allocate_paged([req])
    req.complete_one()
    req.append_host(torch.tensor([last_token], dtype=torch.int32))
    stub.token_pool[table_idx, req.cached_len] = last_token
    return req


def _plain_decode_steps(stub, req, n):
    """What n ordinary decode steps do to the same request: allocate, then complete_one."""
    for _ in range(n):
        stub.cache_manager.allocate_paged([req])
        req.complete_one()


def _state(stub, req):
    return (
        stub.cache_manager.free_slots.clone(),
        stub.engine.page_table[req.table_idx].clone(),
        req.cached_len,
        req.device_len,
    )


def _assert_same_state(stub, req, expected):
    free, row, cached_len, device_len = expected
    assert torch.equal(stub.cache_manager.free_slots, free)
    assert torch.equal(stub.engine.page_table[req.table_idx], row)
    assert (req.cached_len, req.device_len) == (cached_len, device_len)


# ----------------------------------------------------------------------- batch construction


def test_the_batch_has_one_row_per_speculative_token():
    stub = _scheduler()
    req = _decode_req(stub)
    batch = stub._prepare_spec_batch(req, [11, 12, 13]).batch

    assert batch.emit_width == 4
    assert batch.mtp_verify is True
    assert batch.reqs == [req] and batch.padded_reqs == [req]
    assert req.extend_len == 4
    # the MoE decode-movement gate wants prefill causal semantics with 2..4 rows
    assert batch.is_prefill
    assert batch.positions.tolist() == [64, 65, 66, 67]


@pytest.mark.parametrize(
    "drafts,width",
    [
        ([11], 2),
        ([11, 12], 3),
        ([11, 12, 13], 4),
        ([11, 12, 13, 14], 5),
        ([11, 12, 13, 14, 15], 6),
    ],
)
def test_every_configured_depth_produces_a_capturable_verify_width(drafts, width):
    stub = _scheduler(depth=len(drafts))
    req = _decode_req(stub)
    batch = stub._prepare_spec_batch(req, drafts).batch
    assert batch.emit_width == width and req.extend_len == width
    # the width-generic SpecVerifyGraphRunner captures 2..6 = 1 + the depth ceiling
    assert 2 <= width <= 6
    assert batch.positions.tolist() == list(range(64, 64 + width))
    assert stub.token_pool[batch.reqs[0].table_idx, 65 : 64 + width].tolist() == drafts


def test_the_rows_read_the_last_accepted_token_then_the_drafts():
    stub = _scheduler()
    req = _decode_req(stub, last_token=5)
    table_idx, positions = stub._prepare_spec_batch(req, [11, 12, 13]).input_tuple

    assert table_idx.tolist() == [0, 0, 0, 0]
    assert positions.tolist() == [64, 65, 66, 67]
    assert stub.token_pool[table_idx, positions].tolist() == [5, 11, 12, 13]


def test_the_batch_targets_the_requests_own_page_table_row():
    stub = _scheduler()
    req = _decode_req(stub)
    other_row = stub.engine.page_table[1].clone()
    forward_input = stub._prepare_spec_batch(req, [11, 12, 13])

    row = stub.engine.page_table[req.table_idx]
    assert forward_input.batch.out_loc.tolist() == row[64:68].tolist()
    assert torch.equal(stub.engine.page_table[1], other_row)  # no dummy/shadow row involved
    assert stub.prepared == [forward_input.batch]
    assert forward_input.sample_args == "sample-args"


def test_each_row_writes_the_token_it_samples_one_position_on():
    stub = _scheduler()
    req = _decode_req(stub, prompt_len=64, output_len=64)
    table_idx, write_idx = stub._prepare_spec_batch(req, [11, 12, 13]).write_tuple

    # row i reads position 64 + i and samples the token for 65 + i -- the write base is one
    # past row 0, NOT device_len (which the batch already advanced to the last draft row).
    assert table_idx.tolist() == [0, 0, 0, 0]
    assert write_idx.tolist() == [65, 66, 67, 68]


def test_write_slots_past_the_output_budget_are_discarded():
    stub = _scheduler()
    req = _decode_req(stub, prompt_len=64, output_len=3)  # max_device_len = 67
    _table_idx, write_idx = stub._prepare_spec_batch(req, [11, 12]).write_tuple
    assert write_idx.tolist() == [65, 66, -1]


# ------------------------------------------------------------------------- page arithmetic


def test_a_step_that_stays_inside_a_page_allocates_nothing():
    stub = _scheduler(page_size=64)
    req = _decode_req(stub, prompt_len=70)  # cached_len 70; page 1 already covers 64..127
    free_before = stub.cache_manager.free_slots.clone()
    stub._prepare_spec_batch(req, [11, 12, 13])
    assert torch.equal(stub.cache_manager.free_slots, free_before)


def test_a_step_that_straddles_a_page_boundary_allocates_the_next_page():
    stub = _scheduler(page_size=64)
    req = _decode_req(stub, prompt_len=64)  # cached_len 64: row 0 opens a new page
    free_before = stub.cache_manager.free_slots.clone()
    stub._prepare_spec_batch(req, [11, 12, 13])

    assert len(stub.cache_manager.free_slots) == len(free_before) - 1
    page = int(free_before[0])
    assert stub.engine.page_table[0, 64:68].tolist() == [page, page + 1, page + 2, page + 3]


def test_every_speculative_row_gets_a_slot_at_page_size_one():
    stub = _scheduler(page_size=1, num_pages=64)
    req = _decode_req(stub, prompt_len=5)
    free_before = stub.cache_manager.free_slots.clone()
    stub._prepare_spec_batch(req, [11, 12, 13])

    assert len(stub.cache_manager.free_slots) == len(free_before) - 4
    assert stub.engine.page_table[0, 5:9].tolist() == free_before[:4].tolist()


# ------------------------------------------------------------------------------- rejections


def test_speculation_off_refuses_to_build_a_batch():
    stub = _scheduler(enabled=False)
    req = _decode_req(stub)
    with pytest.raises(RuntimeError, match="FREETOKEN_MTP_SPECULATE"):
        stub._prepare_spec_batch(req, [11, 12, 13])


@pytest.mark.parametrize("drafts", [[], [11, 12, 13, 14]])
def test_a_draft_run_outside_the_configured_depth_is_refused(drafts):
    stub = _scheduler(depth=3)
    req = _decode_req(stub)
    with pytest.raises(ValueError, match="draft"):
        stub._prepare_spec_batch(req, drafts)


def test_a_request_that_is_not_in_decode_shape_is_refused():
    stub = _scheduler()
    req = _decode_req(stub)
    req.device_len += 2
    with pytest.raises(RuntimeError, match="decode shape"):
        stub._prepare_spec_batch(req, [11, 12, 13])


def test_a_request_whose_host_ids_lag_is_refused():
    stub = _scheduler()
    req = _decode_req(stub)
    req.input_ids = req.input_ids[:-1]  # the overlapped drain has not appended yet
    with pytest.raises(RuntimeError, match="host"):
        stub._prepare_spec_batch(req, [11, 12, 13])


def test_a_request_without_room_for_the_whole_run_is_refused():
    stub = _scheduler()
    req = _decode_req(stub, prompt_len=64, output_len=2)  # max_device_len 66, remain_len 1
    with pytest.raises(RuntimeError, match="budget"):
        stub._prepare_spec_batch(req, [11, 12, 13])


def test_two_speculative_steps_cannot_be_in_flight_at_once():
    stub = _scheduler()
    req = _decode_req(stub)
    stub._prepare_spec_batch(req, [11, 12, 13])
    with pytest.raises(RuntimeError, match="in flight"):
        stub._prepare_spec_batch(req, [21, 22, 23])


def test_rolling_back_a_step_that_never_ran_is_refused():
    stub = _scheduler()
    req = _decode_req(stub)
    with pytest.raises(RuntimeError, match="no speculative"):
        stub._rollback_spec_tokens(req, 1)


def test_keeping_more_rows_than_the_step_forwarded_is_refused():
    stub = _scheduler()
    req = _decode_req(stub)
    stub._prepare_spec_batch(req, [11, 12, 13])
    with pytest.raises(ValueError, match="accepted"):
        stub._rollback_spec_tokens(req, 5)


# --------------------------------------------------------------------------------- rollback


@pytest.mark.parametrize("page_size,prompt_len", [(64, 64), (64, 70), (1, 5)])
def test_a_full_rollback_is_byte_identical_to_never_having_run_the_step(page_size, prompt_len):
    stub = _scheduler(page_size=page_size, num_pages=64 if page_size == 1 else 8)
    req = _decode_req(stub, prompt_len=prompt_len)
    before = _state(stub, req)

    stub._prepare_spec_batch(req, [11, 12, 13])
    stub._rollback_spec_tokens(req, 0)

    _assert_same_state(stub, req, before)
    assert req.spec_inflight is None


@pytest.mark.parametrize("accepted", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("page_size,prompt_len", [(64, 64), (64, 70), (1, 5)])
def test_the_kept_run_lands_where_the_same_plain_decode_steps_would_have(
    accepted, page_size, prompt_len
):
    num_pages = 64 if page_size == 1 else 8
    plain = _scheduler(page_size=page_size, num_pages=num_pages)
    plain_req = _decode_req(plain, prompt_len=prompt_len)
    _plain_decode_steps(plain, plain_req, accepted)
    expected = _state(plain, plain_req)

    stub = _scheduler(page_size=page_size, num_pages=num_pages)
    req = _decode_req(stub, prompt_len=prompt_len)
    stub._prepare_spec_batch(req, [11, 12, 13])
    stub._rollback_spec_tokens(req, accepted)

    _assert_same_state(stub, req, expected)
    assert req.extend_len == 1  # back in decode shape for the next step


def test_full_acceptance_keeps_every_page_the_step_allocated():
    stub = _scheduler(page_size=1, num_pages=64)
    req = _decode_req(stub, prompt_len=5)
    free_after_alloc = stub.cache_manager.free_slots.clone()
    stub._prepare_spec_batch(req, [11, 12, 13])
    stub._rollback_spec_tokens(req, 4)

    assert (req.cached_len, req.device_len) == (9, 10)
    assert len(stub.cache_manager.free_slots) == len(free_after_alloc) - 4


def test_the_phase_three_state_seam_is_called_with_the_settled_run():
    stub = _scheduler()
    req = _decode_req(stub)
    seen = []
    stub._prepare_spec_batch(req, [11, 12, 13])
    stub._rollback_spec_tokens(
        req, 2, state_rollback=lambda r, n: seen.append((r.cached_len, r.device_len, n))
    )
    # GDN/PLE rollback (Phase 3) runs after the lengths are final, never before
    assert seen == [(66, 67, 2)]


# ------------------------------------------------------------- the track-checkpoint hazard
#
# A hybrid-radix track checkpoint freezes a request's GDN + PLE state into a DONATABLE pool
# slot on the forward stream. One taken during a speculative step would publish a state
# containing rejected rows into the prefix cache -- a rollback surface nothing else covers.
# It cannot happen, and not for the reason a x64-alignment intuition suggests: the boundary is
# counted in THIS FORWARD's rows (attention/linear.py:123-127, c = (extend_len - 1) // 64), so
# only a forward of at least 65 rows can schedule one. w <= 4 never can, at any cached_len.


@pytest.mark.parametrize("prompt_len", [62, 63, 64, 127, 128])
def test_no_track_checkpoint_can_ride_a_speculative_step(prompt_len):
    stub = _scheduler(gdn=True)
    req = _decode_req(stub, prompt_len=prompt_len, output_len=128)
    req.linear_slot_idx = 1
    req.mamba_ping_pong = (2, 3)  # eligible to snapshot; only the row count stops it

    fla = stub._prepare_spec_batch(req, [11, 12, 13]).batch.fla_metadata
    assert req.extend_len == 4
    assert (fla.track_dst, fla.track_h_row, fla.track_conv_src, fla.track_boundary_row) == (
        None, None, None, None
    )
    assert req.mamba_last_track_seqlen is None and req.mamba_next_track_idx == 0


def test_the_same_request_does_schedule_one_on_a_wide_enough_prefill():
    """The control: nothing about the request or the pool suppresses the checkpoint -- only the
    speculative step's row count does."""
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.core import Batch

    stub = _scheduler(gdn=True)
    req = _decode_req(stub, prompt_len=62, output_len=128)
    req.linear_slot_idx = 1
    req.mamba_ping_pong = (2, 3)
    req.cached_len, req.device_len = 62, 62 + 65

    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    fla = build_fla_metadata(batch, CPU)
    assert fla.track_dst is not None and fla.track_dst.tolist() == [2]
    assert req.mamba_last_track_seqlen == 62 + 64  # counted from cached_len, not from 0


# --------------------------------------------------------------- the radix-commit guard (#4)


def test_a_commit_under_in_flight_speculative_rows_is_refused():
    stub = _scheduler(cache_type="naive")
    req = _decode_req(stub)
    stub._prepare_spec_batch(req, [11, 12, 13])

    with pytest.raises(RuntimeError, match="speculative"):
        stub.cache_manager.cache_req(req, finished=False)
    with pytest.raises(RuntimeError, match="speculative"):
        stub.cache_manager.cache_req(req, finished=True)


def test_a_commit_is_allowed_again_once_the_step_is_settled():
    stub = _scheduler(cache_type="naive")
    req = _decode_req(stub)
    stub._prepare_spec_batch(req, [11, 12, 13])
    stub._rollback_spec_tokens(req, 2)

    stub.cache_manager.cache_req(req, finished=False)  # must not raise


def test_an_ordinary_request_is_untouched_by_the_guard():
    stub = _scheduler(cache_type="naive")
    req = _decode_req(stub)
    assert req.spec_inflight is None
    stub.cache_manager.cache_req(req, finished=False)  # must not raise

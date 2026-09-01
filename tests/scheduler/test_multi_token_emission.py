"""One scheduler step may emit a run of tokens for a request.

The emission protocol is width-generic: the sampled-token write pair carries
``Batch.emit_width`` slots per request, ``Req.complete_many`` advances the device length
by the run width, and the drain ships ONE DetokenizeMsg per uid per step carrying the
whole run. Plain decode keeps emitting width-1 runs, and must behave exactly as before.

Multi-token stop semantics, pinned here: the run is processed in order and the first stop
condition (EOS, stop string, output budget) truncates the remainder, so the emitted ids
and the finish reason equal what the same tokens emitted one per step would have produced.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.message import DetokenizeMsg
from freetoken.scheduler.scheduler import Scheduler, _make_write_tuple

EOS = 99
ANCHOR = 77


class _CharTokenizer:
    """Ids are code points, so a stop string is a run of tokens."""

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _req(*, prompt_len: int = 4, output_len: int = 8, table_idx: int = 3, **sp) -> Req:
    return Req(
        input_ids=torch.arange(prompt_len, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=0,
        output_len=output_len,
        uid=1,
        sampling_params=SamplingParams(max_tokens=output_len, **sp),
        cache_handle=None,
    )


def _scheduler(*, eos=frozenset({EOS}), anchor=None) -> Scheduler:
    stub = Scheduler.__new__(Scheduler)
    stub.eos_token_ids = eos
    stub.toolcall_anchor_id = anchor
    stub.tokenizer = _CharTokenizer()
    return stub


def _emit(stub: Scheduler, req: Req, tokens) -> DetokenizeMsg:
    return Scheduler._emit_step_tokens(
        stub, req, torch.tensor(tokens, dtype=torch.int32)
    )


# --- device-side width ------------------------------------------------------------


def test_complete_many_advances_the_device_length_by_the_run_width():
    req = _req(prompt_len=4, output_len=8)
    req.complete_many(3)
    assert (req.cached_len, req.device_len) == (4, 7)
    req.complete_one()
    assert (req.cached_len, req.device_len) == (7, 8)


def test_append_host_is_already_width_generic():
    req = _req(prompt_len=4, output_len=8)
    req.append_host(torch.tensor([10, 11, 12], dtype=torch.int32))
    assert req.input_ids.tolist() == [0, 1, 2, 3, 10, 11, 12]


def test_write_tuple_keeps_one_slot_per_request_for_plain_decode():
    req = _req(prompt_len=4, output_len=8, )
    req.complete_one()
    batch = Batch(reqs=[req], phase="decode")
    assert batch.emit_width == 1
    table_idx, write_idx = _make_write_tuple(batch, torch.device("cpu"))
    assert table_idx.tolist() == [3]
    assert write_idx.tolist() == [5]


def test_write_tuple_has_one_slot_per_emitted_token():
    req = _req(prompt_len=4, output_len=8)
    req.complete_one()
    batch = Batch(reqs=[req], phase="decode")
    batch.emit_width = 3
    table_idx, write_idx = _make_write_tuple(batch, torch.device("cpu"))
    assert table_idx.tolist() == [3, 3, 3]
    assert write_idx.tolist() == [5, 6, 7]


def test_write_tuple_discards_slots_past_the_output_budget():
    req = _req(prompt_len=4, output_len=3)  # max_device_len = 7
    req.complete_many(2)                    # device_len = 6, one slot of budget left
    batch = Batch(reqs=[req], phase="decode")
    batch.emit_width = 3
    _table_idx, write_idx = _make_write_tuple(batch, torch.device("cpu"))
    assert write_idx.tolist() == [6, -1, -1]


# --- host-side emission -----------------------------------------------------------


def test_a_run_emits_one_message_carrying_every_token():
    stub = _scheduler()
    req = _req(prompt_len=4, output_len=8)
    req.complete_many(3)

    msg = _emit(stub, req, [10, 11, 12])

    assert msg.next_tokens == (10, 11, 12)
    assert msg.finished is False and msg.finish_reason is None
    assert req.input_ids.tolist() == [0, 1, 2, 3, 10, 11, 12]


def test_a_run_matches_the_same_tokens_emitted_one_per_step():
    run_stub, step_stub = _scheduler(), _scheduler()
    run_req, step_req = _req(output_len=8), _req(output_len=8)
    run_req.complete_many(3)

    run = _emit(run_stub, run_req, [10, 11, 12])
    stepped = []
    for token in (10, 11, 12):
        step_req.complete_one()
        stepped.append(_emit(step_stub, step_req, [token]))

    assert run.next_tokens == tuple(t for m in stepped for t in m.next_tokens)
    assert run.finished == stepped[-1].finished
    assert run.finish_reason == stepped[-1].finish_reason
    assert run_req.input_ids.tolist() == step_req.input_ids.tolist()


def test_the_first_eos_in_a_run_truncates_the_rest():
    stub = _scheduler()
    req = _req(prompt_len=4, output_len=8)
    req.complete_many(4)

    msg = _emit(stub, req, [10, EOS, 11, 12])

    assert msg.next_tokens == (10, EOS)
    assert msg.finished is True and msg.finish_reason == "stop"
    assert msg.matched_stop is None
    # the rejected tail never reaches the host ids
    assert req.input_ids.tolist() == [0, 1, 2, 3, 10, EOS]


def test_ignore_eos_keeps_the_whole_run():
    stub = _scheduler()
    req = _req(prompt_len=4, output_len=8, ignore_eos=True)
    req.complete_many(3)

    msg = _emit(stub, req, [10, EOS, 11])

    assert msg.next_tokens == (10, EOS, 11)
    assert msg.finished is False


def test_the_first_stop_string_in_a_run_truncates_the_rest():
    stub = _scheduler()
    req = _req(prompt_len=4, output_len=8, stop_strs=["XY"])
    req.complete_many(3)

    msg = _emit(stub, req, [ord("X"), ord("Y"), ord("Z")])

    assert msg.next_tokens == (ord("X"), ord("Y"))
    assert msg.matched_stop == "XY"
    assert msg.finished is True and msg.finish_reason == "stop"
    assert msg.stop_strs == ["XY"]


def test_a_run_is_clamped_to_the_remaining_output_budget():
    stub = _scheduler()
    req = _req(prompt_len=4, output_len=2)  # max_device_len = 6
    req.complete_many(2)                    # the device wrote both tokens

    msg = _emit(stub, req, [10, 11, 12])

    assert msg.next_tokens == (10, 11)
    assert msg.finished is True and msg.finish_reason == "length"
    assert req.input_ids.tolist() == [0, 1, 2, 3, 10, 11]


def test_an_eos_inside_the_run_wins_over_the_length_finish():
    stub = _scheduler()
    req = _req(prompt_len=4, output_len=2)
    req.complete_many(2)

    msg = _emit(stub, req, [EOS, 11])

    assert msg.next_tokens == (EOS,)
    assert msg.finish_reason == "stop"


def test_the_toolcall_anchor_takes_the_first_anchor_token_in_the_run():
    stub = _scheduler(anchor=ANCHOR)
    req = _req(prompt_len=4, output_len=8)
    req.complete_many(3)

    _emit(stub, req, [10, ANCHOR, ANCHOR])

    assert req.toolcall_anchor_len == 6  # prompt + 10 + anchor


def test_a_terminal_anchor_token_does_not_set_the_anchor():
    stub = _scheduler(anchor=ANCHOR)
    req = _req(prompt_len=4, output_len=2)
    req.complete_many(2)

    _emit(stub, req, [10, ANCHOR])  # the anchor is the length-terminal token

    assert req.toolcall_anchor_len is None


def test_an_empty_budget_is_not_silently_emitted():
    stub = _scheduler()
    req = _req(prompt_len=4, output_len=1)
    req.append_host(torch.tensor([10], dtype=torch.int32))
    with pytest.raises(AssertionError):
        _emit(stub, req, [11])


# --- drain path -------------------------------------------------------------------


def _drain_stub(sent: list):
    return SimpleNamespace(
        cache_manager=SimpleNamespace(
            lazy_free_region=contextlib.nullcontext, cache_req=lambda *_a, **_k: None
        ),
        decode_manager=SimpleNamespace(remove_req=lambda _req: None, running_reqs=()),
        prefill_manager=SimpleNamespace(pending_list=()),
        config=SimpleNamespace(page_size=1),
        finished_reqs=set(),
        eos_token_ids=frozenset({EOS}),
        toolcall_anchor_id=None,
        tokenizer=_CharTokenizer(),
        status_reporter=SimpleNamespace(report_batch=lambda *_a, **kw: sent.append(kw)),
        send_result=sent.append,
        _free_req_resources=lambda _req: None,
        _kv_usage_pages=lambda: (1, 2),
        _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0,
    )


def test_the_drain_ships_one_message_per_uid_for_a_multi_token_step():
    sent: list = []
    stub = _drain_stub(sent)
    stub._emit_step_tokens = lambda req, tokens: Scheduler._emit_step_tokens(
        stub, req, tokens
    )
    stub._match_stop_str = lambda req: Scheduler._match_stop_str(stub, req)
    req = _req(prompt_len=4, output_len=8)
    req.complete_many(2)
    batch = Batch(reqs=[req], phase="decode")
    batch.emit_width = 2
    last_data = (
        SimpleNamespace(batch=batch),
        (
            None,
            torch.tensor([[10, 11]], dtype=torch.int32),
            SimpleNamespace(synchronize=lambda: None),
        ),
    )

    Scheduler._process_last_data(stub, last_data)

    reported = [item for item in sent if isinstance(item, dict)]
    replies = [item for item in sent if isinstance(item, list)]
    assert len(replies) == 1 and len(replies[0]) == 1
    assert replies[0][0].next_tokens == (10, 11)
    assert replies[0][0].kv_used_pages == 1 and replies[0][0].kv_total_pages == 2
    assert reported[0]["generated_tokens"] == 2

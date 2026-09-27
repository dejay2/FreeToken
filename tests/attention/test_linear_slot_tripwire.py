"""The linear-state slot tripwire (``check_linear_slots``).

2026-09-27 the serving box died on a device-side ``index_copy_`` out of bounds in the PLE conv
state at a prefill chunk boundary, and the dump could not say which owner handed out the slot.
The check turns the next bad id into a host-side error that names the request and the pool.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.core as core
from freetoken.attention.linear import build_fla_metadata, check_linear_slots
from freetoken.core import Batch, Context, set_global_ctx


class _Pool:
    num_slots = 8
    _free_slots = [5, 6]
    _reserved_slots = [7]


@pytest.fixture(autouse=True)
def _no_ctx_leak():
    yield
    core._GLOBAL_CTX = None


def _req(**fields):
    base = dict(uid=3, linear_slot_idx=2, mamba_ping_pong=None, table_idx=1, cached_len=0,
                extend_len=4, mamba_next_track_idx=0)
    base.update(fields)
    return SimpleNamespace(**base)


def test_slots_inside_the_pool_pass():
    check_linear_slots("decode", [_req(), _req()], [0, 7], _Pool())


@pytest.mark.parametrize("slot", [8, 9, -1])
def test_a_slot_outside_the_pool_raises_with_the_request_and_the_free_list(slot):
    with pytest.raises(RuntimeError) as err:
        check_linear_slots("track snapshot", [_req(linear_slot_idx=slot)], [slot], _Pool())
    message = str(err.value)
    assert "track snapshot" in message and f"slot={slot}" in message and "uid=3" in message
    assert "pool has 8 slots" in message and "free=[5, 6]" in message and "reserved=[7]" in message


def test_no_pool_means_nothing_to_check():
    core._GLOBAL_CTX = None
    check_linear_slots("prefill live", [_req()], [99])


def test_prefill_metadata_refuses_an_out_of_range_live_slot():
    core._GLOBAL_CTX = None
    set_global_ctx(Context(page_size=64, linear_state_pool=_Pool()))
    batch = Batch(reqs=[_req(linear_slot_idx=12)], phase="prefill")
    batch.padded_reqs = batch.reqs
    with pytest.raises(RuntimeError, match="prefill live"):
        build_fla_metadata(batch, torch.device("cpu"))

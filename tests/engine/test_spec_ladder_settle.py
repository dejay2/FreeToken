"""The settle path's LAUNCH shape -- what the ladder copies, and what it stops copying.

The bitwise gate on the ladder lives in ``tests/models/qwen4_exp/test_mtp_state_ladder.py``
and needs a GPU (the fla recurrent kernel is triton). These tests run on the CPU and pin the
part that has nothing to do with the kernel: a settle that accepts anything rewrites the conv
and PLE shift registers WHOLE out of the snapshot slot, so restoring them first is dead work,
and the settled bytes have to be identical either way. ``_launch_recurrent`` is stubbed for
exactly that reason -- the recurrent half is the kernel's business, and it is the only half
``rollback`` still restores.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.engine.spec_state_ladder import (
    LADDER_GRAPH_ENV,
    SpecStateLadder,
    ladder_graph_enabled,
)
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec
from freetoken.models.qwen4_exp.config import PLE_CONV_STATE, PLE_NGRAM_STATE

CPU = torch.device("cpu")
LAYERS = (0, 1, 2)
PLE_LAYERS = (0, 2)
NUM_SLOTS = 6
WIDTH = 4
LIVE = 3
KM1 = 3  # conv_kernel_dim - 1
PLE_STATE_LEN = 2
NGRAM_LEN = 3


def _pool() -> LinearStatePool:
    group = LinearGatedDeltaGroupConfig(
        name="gdn",
        layer_ids=LAYERS,
        num_key_heads=2,
        num_value_heads=2,
        key_head_dim=4,
        value_head_dim=4,
        conv_kernel_dim=KM1 + 1,
        output_gate="silu",
    )
    slot_states = (
        SlotStateSpec(
            name=PLE_CONV_STATE,
            shape=(5, PLE_STATE_LEN),
            layer_ids=PLE_LAYERS,
            dtype=torch.float32,
        ),
        SlotStateSpec(
            name=PLE_NGRAM_STATE,
            shape=(NGRAM_LEN,),
            layer_ids=(),
            dtype=torch.int32,
        ),
    )
    return LinearStatePool(
        group, NUM_SLOTS, torch.float32, CPU, tp_size=1, slot_states=slot_states
    )


class _Req:
    linear_slot_idx = LIVE
    table_idx = LIVE


class _Batch:
    def __init__(self, ids: torch.Tensor) -> None:
        self.emit_width = int(ids.numel())
        self.input_ids = ids
        self.spec_capture = None


def _fill(pool: LinearStatePool, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    pool.conv_states.copy_(torch.rand(pool.conv_states.shape, generator=generator))
    pool.recurrent_states.copy_(
        torch.rand(pool.recurrent_states.shape, generator=generator)
    )
    ple = pool.slot_states[PLE_CONV_STATE]
    ple.copy_(torch.rand(ple.shape, generator=generator))
    ngram = pool.slot_states[PLE_NGRAM_STATE]
    ngram.copy_(torch.randint(1, 900, ngram.shape, generator=generator).to(ngram.dtype))


def _armed_ladder(pool: LinearStatePool, ids: torch.Tensor) -> SpecStateLadder:
    """A ladder mid-step: snapshot taken, every layer's stash filled, no kernel needed."""
    ladder = SpecStateLadder(pool, WIDTH)
    # the recurrent replay is the fla kernel's job and needs a GPU; everything these tests
    # look at is the copying around it
    ladder._launch_recurrent = lambda indices, steps: []
    ladder.begin(_Req(), _Batch(ids))
    generator = torch.Generator().manual_seed(99)
    for layer_id in LAYERS:
        ladder.stash_gdn(
            layer_id,
            conv_in=torch.rand(
                (WIDTH, pool.conv_states.shape[2]), generator=generator
            ),
            mixed=torch.rand((WIDTH, pool.conv_states.shape[2]), generator=generator),
            a=torch.rand((WIDTH, 2), generator=generator),
            b=torch.rand((WIDTH, 2), generator=generator),
            A_log=torch.zeros(2),
            dt_bias=torch.zeros(2),
            scale=1.0,
        )
    for layer_id in PLE_LAYERS:
        ladder.stash_ple(layer_id, torch.rand((WIDTH, 5), generator=generator))
    return ladder


def _families(pool: LinearStatePool, slot: int) -> tuple[torch.Tensor, ...]:
    return (
        pool.conv_states[:, slot].clone(),
        pool.recurrent_states[:, slot].clone(),
        pool.slot_states[PLE_CONV_STATE][:, slot].clone(),
        pool.slot_states[PLE_NGRAM_STATE][:, slot].clone(),
    )


@pytest.mark.parametrize("accepted", [1, 2, WIDTH])
def test_skipping_the_shift_registers_restore_settles_the_same_bytes(accepted):
    """The optimisation's whole claim: reading the pre-step window out of the SNAPSHOT is the
    same as restoring the live slot first and reading it back."""
    ids = torch.tensor([11, 12, 13, 14], dtype=torch.int32)

    reference = _pool()
    _fill(reference, seed=7)
    ladder = _armed_ladder(reference, ids)
    # the pre-optimisation settle: restore EVERY family, then replay over the restored slot
    ladder.restore_snapshot()
    ladder._conv_hist[:, :, :KM1].copy_(reference.conv_states[:, LIVE])
    reference.conv_states[:, LIVE].copy_(
        ladder._conv_hist[:, :, accepted : accepted + KM1]
    )
    ladder._ple_hist[:, :, :PLE_STATE_LEN].copy_(
        reference.slot_states[PLE_CONV_STATE][:, LIVE]
    )
    reference.slot_states[PLE_CONV_STATE][:, LIVE].copy_(
        ladder._ple_hist[:, :, accepted : accepted + PLE_STATE_LEN]
    )
    ladder._ngram_hist[:NGRAM_LEN].copy_(
        reference.slot_states[PLE_NGRAM_STATE][0, LIVE]
    )
    reference.slot_states[PLE_NGRAM_STATE][0, LIVE].copy_(
        ladder._ngram_hist[accepted : accepted + NGRAM_LEN]
    )
    want = _families(reference, LIVE)

    pool = _pool()
    _fill(pool, seed=7)
    ladder = _armed_ladder(pool, ids)
    ladder.rollback(_Req(), accepted)
    got = _families(pool, LIVE)

    for expected, actual in zip(want, got):
        assert torch.equal(expected, actual)


def _trace_restores(ladder: SpecStateLadder) -> list[str]:
    """Which restore a settle reaches for: the whole snapshot, or the recurrent half."""
    seen: list[str] = []
    recurrent = ladder._restore_recurrent

    def _all() -> None:
        seen.append("snapshot")
        ladder.pool.copy_from(ladder.slot, ladder._live)

    def _recurrent() -> None:
        seen.append("recurrent")
        recurrent()

    ladder.restore_snapshot = _all
    ladder._restore_recurrent = _recurrent
    return seen


@pytest.mark.parametrize("accepted", [1, 2, WIDTH])
def test_a_settle_that_accepts_rows_restores_only_the_recurrent_state(accepted):
    """The three saved launches: conv, PLE conv and the n-gram context are not restored."""
    pool = _pool()
    _fill(pool, seed=13)
    ladder = _armed_ladder(pool, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    seen = _trace_restores(ladder)

    ladder.rollback(_Req(), accepted)

    assert seen == ["recurrent"]


def test_a_full_undo_is_still_the_whole_snapshot():
    """``rollback(req, 0)`` is boot capture's wind-back; it must stay ``copy_from``."""
    pool = _pool()
    _fill(pool, seed=21)
    before = _families(pool, LIVE)
    ladder = _armed_ladder(pool, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    seen = _trace_restores(ladder)
    # a "forward" that advanced every family
    _fill_slot(pool, LIVE, seed=22)

    ladder.rollback(_Req(), 0)

    assert seen == ["snapshot"]
    for expected, actual in zip(before, _families(pool, LIVE)):
        assert torch.equal(expected, actual)


def _fill_slot(pool: LinearStatePool, slot: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    pool.conv_states[:, slot] = torch.rand(
        pool.conv_states[:, slot].shape, generator=generator
    )
    pool.recurrent_states[:, slot] = torch.rand(
        pool.recurrent_states[:, slot].shape, generator=generator
    )
    ple = pool.slot_states[PLE_CONV_STATE]
    ple[:, slot] = torch.rand(ple[:, slot].shape, generator=generator)
    ngram = pool.slot_states[PLE_NGRAM_STATE]
    ngram[:, slot] = torch.randint(
        1, 900, ngram[:, slot].shape, generator=generator
    ).to(ngram.dtype)


# ----------------------------------------------------------------- the graphed replay switch


def test_the_graphed_replay_is_on_by_default_and_switchable():
    assert ladder_graph_enabled({}) is True
    assert ladder_graph_enabled({LADDER_GRAPH_ENV: "1"}) is True
    assert ladder_graph_enabled({LADDER_GRAPH_ENV: "0"}) is False
    with pytest.raises(ValueError, match=LADDER_GRAPH_ENV):
        ladder_graph_enabled({LADDER_GRAPH_ENV: "yes"})


def test_a_cpu_ladder_never_arms_a_graph():
    """Graphing is a CUDA-only path; on the CPU the settle is the eager loop it always was."""
    ladder = SpecStateLadder(_pool(), WIDTH)
    assert ladder._graph_enabled is False


def test_the_off_switch_keeps_the_eager_per_layer_replay(monkeypatch):
    monkeypatch.setenv(LADDER_GRAPH_ENV, "0")
    pool = _pool()
    _fill(pool, seed=5)
    ladder = _armed_ladder(pool, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    assert ladder._graph_enabled is False

    seen: list[int] = []
    ladder._launch_recurrent = lambda indices, steps: seen.append(steps) or []
    ladder._capture_replay = lambda steps: pytest.fail("capture must not be attempted")
    ladder.rollback(_Req(), 3)

    assert seen == [3]


def test_a_pool_rebuild_drops_every_recorded_replay():
    """``rebuild`` reallocates the state tensors, so a graph that baked their addresses is
    pointing at freed storage; ``rebind`` is the seam that has to throw them away."""
    pool = _pool()
    ladder = SpecStateLadder(pool, WIDTH)
    ladder._graphs[2] = object()
    ladder._graph_outputs[2] = [torch.zeros(1)]

    pool.rebuild(NUM_SLOTS)
    ladder.rebind()

    assert ladder._graphs == {} and ladder._graph_outputs == {}

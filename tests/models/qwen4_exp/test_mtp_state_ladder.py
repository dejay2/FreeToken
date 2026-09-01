"""Phase 3 -- the speculative linear-state ladder (design section 4, Strategy R).

A ``w``-row speculative forward drives the GDN layers through the CHUNKED prefill kernel and
the PLE layer through its packed prefill conv, leaving every linear state holding all ``w``
rows. ``SpecStateLadder`` snapshots the slot first, stashes each layer's per-row recurrent
inputs during the forward, and on settle restores the snapshot and re-advances the accepted
prefix through the SAME recurrent kernel plain decode uses.

The gate: for every accepted count and several boundary geometries, the settled slot must hold
exactly what the same number of plain decode steps would have left there -- recurrent, conv and
both PLE families, BIT for BIT.

That is only achievable because every rung of the ladder is exact (pinned below): the varlen
conv equals successive decode convs, the conv/PLE states are shift registers of raw inputs, and
the recurrent kernel at ``T = n`` equals ``n`` launches at ``T = 1``. The one thing that is NOT
exact is upstream of all of it -- a bf16 GEMM is not row-stable in its batch size -- so the
bitwise gate uses an input projection that is (a scaled selection matrix: every dot product is
one product plus exact zeros), and a second test tracks the shipping dense-weight case within
that projection noise.
"""

from __future__ import annotations

import pytest
import torch

import freetoken.core as core
from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.spec_state_ladder import SpecStateLadder
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.qwen4_exp.config import PLE_CONV_STATE, PLE_NGRAM_STATE, parse_config
from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet
from freetoken.models.qwen4_exp.ple import GpuResidentTable, PLELayer, commit_ngram_context
from freetoken.utils.torch_utils import torch_dtype

from .common import EOS, VOCAB, fresh_ctx, hash_constants, requires_cuda, toy_hf_config

pytestmark = requires_cuda

DEV = torch.device("cuda")
DTYPE = torch.bfloat16
WIDTH = 4
SLOT = 3


# --------------------------------------------------------------------------------- fixtures


def _config():
    # head_k_dim 64: the fla chunk kernel's tile shapes want a real head dim, and the ladder's
    # q/k/v views are derived from conv_dim, so a non-square split would go unnoticed at 32.
    return parse_config(toy_hf_config(linear_key_head_dim=64, linear_value_head_dim=64))


def _fill(op, gen, *, row_stable):
    """Random weights, except that ``row_stable`` makes the fused input projection a scaled
    selection matrix -- one nonzero per output row, so its GEMM is a single product plus exact
    zeros and therefore reproduces bit for bit at any batch width."""
    for name, tensor in op.state_dict().items():
        if not tensor.is_floating_point():
            tensor.zero_()
        elif name == "A_log":
            tensor.uniform_(0.01, 16.0, generator=gen).log_()
        elif name == "dt_bias":
            tensor.uniform_(-1.0, 1.0, generator=gen)
        elif name == "norm.weight":
            tensor.normal_(1.0, 0.1, generator=gen)
        elif name == "conv1d.weight":
            tensor.normal_(0.0, 0.4, generator=gen)
        elif name == "in_proj.weight" and row_stable:
            tensor.zero_()
            rows, cols = tensor.shape
            picks = torch.randint(0, cols, (rows,), generator=gen, device=tensor.device)
            tensor[torch.arange(rows, device=tensor.device), picks] = torch.empty(
                rows, device=tensor.device, dtype=tensor.dtype
            ).normal_(0.0, 0.5, generator=gen)
        else:
            tensor.normal_(0.0, 0.05, generator=gen)


def _make_ple(config, gen):
    args = config.qwen4_args
    with torch.device(DEV), torch_dtype(DTYPE):
        layer = PLELayer(config, args.ple_layer_ids[0])
    for tensor in layer.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, 0.05, generator=gen)
    multipliers, sizes, offsets = hash_constants(args)
    layer.ple_embedding.layer_multipliers.copy_(multipliers)
    layer.ple_embedding.ngram_heads_vocab_sizes.copy_(sizes)
    layer.ple_embedding.ngram_heads_offsets.copy_(offsets)
    rows = int(sizes.sum())
    rows = -(-rows // args.make_ngram_vocab_size_divisible_by) * args.make_ngram_vocab_size_divisible_by
    weight = torch.empty(rows, args.ngram_head_dim, device=DEV, dtype=DTYPE).normal_(
        0.0, 0.05, generator=gen
    )
    layer.ple_embedding.attach_table(GpuResidentTable(weight, dtype=DTYPE))
    return layer


class _World:
    """One coherent toy qwen4_exp linear stack: every GDN layer plus the PLE layer, all on one
    LinearStatePool, exercised through their real forwards."""

    def __init__(self, *, row_stable=True, seed=0, num_slots=8):
        config = _config()
        self.config = config
        self.args = config.qwen4_args
        group = config.linear_attention_group()
        self.group = group
        self.pool = LinearStatePool(
            group, num_slots, DTYPE, DEV, tp_size=1, slot_states=config.slot_states
        )
        self.ctx = fresh_ctx(page_size=64, linear_state_pool=self.pool)
        gen = torch.Generator(device=DEV).manual_seed(seed)
        self.gdn = {}
        for layer_id in group.layer_ids:
            with torch.device(DEV), torch_dtype(DTYPE):
                op = Qwen4ExpGatedDeltaNet(
                    hidden_size=config.hidden_size,
                    num_k_heads=group.num_key_heads,
                    num_v_heads=group.num_value_heads,
                    head_k_dim=group.key_head_dim,
                    head_v_dim=group.value_head_dim,
                    conv_kernel_size=group.conv_kernel_dim,
                    rms_norm_eps=config.rms_norm_eps,
                    layer_id=layer_id,
                    output_gate=group.output_gate,
                )
            _fill(op, gen, row_stable=row_stable)
            self.gdn[layer_id] = op
        self.ple = _make_ple(config, gen)

    # -- state families ------------------------------------------------------------------
    def families(self, slot):
        out = {
            "conv": self.pool.conv_states[:, slot].clone(),
            "recurrent": self.pool.recurrent_states[:, slot].clone(),
        }
        for name, tensor in self.pool.slot_states.items():
            out[name] = tensor[:, slot].clone()
        return out

    # -- forwards ------------------------------------------------------------------------
    def _req(self, cached_len, extend, *, ids):
        req = Req(
            input_ids=ids[: cached_len + extend].cpu(),
            table_idx=SLOT,
            cached_len=cached_len,
            output_len=1,
            uid=0,
            sampling_params=SamplingParams(),
            cache_handle=None,
        )
        req.linear_slot_idx = SLOT
        return req

    def prefill(self, hidden, ple_in, ids, *, cached_len, mtp_verify=False, ladder=None):
        """One prefill-phase forward over ``hidden.shape[0]`` rows."""
        n = hidden.shape[0]
        req = self._req(cached_len, n, ids=ids)
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        batch.input_ids = ids[cached_len : cached_len + n].to(DEV)
        batch.mtp_verify = mtp_verify
        batch.emit_width = n if mtp_verify else 1
        from freetoken.attention.linear import build_fla_metadata

        batch.linear_table_idx = torch.tensor([SLOT], dtype=torch.int32, device=DEV)
        batch.fla_metadata = build_fla_metadata(batch, DEV)
        if ladder is not None:
            ladder.begin(req, batch)
        self._run(batch, hidden, ple_in)
        return req, batch

    def decode(self, hidden, ple_in, ids, *, cached_len):
        """One plain decode step -- the ground truth rung of the ladder."""
        req = self._req(cached_len, 1, ids=ids)
        batch = Batch(reqs=[req], phase="decode")
        batch.padded_reqs = batch.reqs
        batch.input_ids = ids[cached_len : cached_len + 1].to(DEV)
        batch.linear_table_idx = torch.tensor([SLOT], dtype=torch.int32, device=DEV)
        self._run(batch, hidden, ple_in)
        return req, batch

    def _run(self, batch, hidden, ple_in):
        from freetoken.models.qwen4_exp.ple import build_ple_metadata

        with self.ctx.forward_batch(batch):
            meta = build_ple_metadata(batch, self.args, DEV)
            for layer_id in self.group.layer_ids:
                self.gdn[layer_id].forward(hidden)
            self.ple.forward(ple_in, batch, meta)
            commit_ngram_context(meta, batch.fla_metadata)


def _inputs(n, hidden_size, ple_width, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    hidden = torch.empty(n, hidden_size, device=DEV, dtype=DTYPE).normal_(generator=gen)
    ple_in = torch.empty(n, ple_width, device=DEV, dtype=DTYPE).normal_(generator=gen)
    return hidden, ple_in


def _token_ids(n, seed=5):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=gen, dtype=torch.int32)


def _warm(world, cached_len, ids):
    """Prime the slot with a real prefill so the ladder starts from a non-trivial state."""
    hidden, ple_in = _inputs(cached_len, world.config.hidden_size, world.args.ple_state_width, 1)
    world.prefill(hidden, ple_in, ids, cached_len=0)


def _plain_decode_families(world, cached_len, ids, hidden, ple_in, steps):
    for i in range(steps):
        world.decode(
            hidden[i : i + 1], ple_in[i : i + 1], ids, cached_len=cached_len + i
        )
    return world.families(SLOT)


def _spec_then_rollback(world, cached_len, ids, hidden, ple_in, accepted):
    ladder = SpecStateLadder(world.pool, WIDTH)
    req, _ = world.prefill(
        hidden, ple_in, ids, cached_len=cached_len, mtp_verify=True, ladder=ladder
    )
    req.cached_len = cached_len + accepted
    req.device_len = req.cached_len + 1
    ladder.rollback(req, accepted)
    return ladder, world.families(SLOT)


def _scenario(cached_len, accepted, *, row_stable=True, seed=0):
    """Same start state, same rows, two paths: plain decode vs speculate-then-settle."""
    ids = _token_ids(cached_len + WIDTH + 4)
    plain = _World(row_stable=row_stable, seed=seed)
    hidden, ple_in = _inputs(WIDTH, plain.config.hidden_size, plain.args.ple_state_width, 2)
    _warm(plain, cached_len, ids)
    want = _plain_decode_families(plain, cached_len, ids, hidden, ple_in, accepted)

    spec = _World(row_stable=row_stable, seed=seed)
    _warm(spec, cached_len, ids)
    ladder, got = _spec_then_rollback(spec, cached_len, ids, hidden, ple_in, accepted)
    return want, got, ladder


def _assert_bitwise(want, got):
    assert set(want) == set(got) == {"conv", "recurrent", PLE_CONV_STATE, PLE_NGRAM_STATE}
    for name in want:
        assert torch.equal(got[name], want[name]), (
            f"{name} differs; max |delta| "
            f"{(got[name].float() - want[name].float()).abs().max().item()}"
        )


# ------------------------------------------------------------------------------- the gate


@pytest.mark.parametrize("accepted", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("cached_len", [7, 62, 64, 127], ids=["short", "x64-cross", "x64-open", "x64-fill"])
def test_the_settled_state_is_what_the_same_plain_decode_steps_would_have_left(
    cached_len, accepted
):
    want, got, _ = _scenario(cached_len, accepted)
    _assert_bitwise(want, got)


def test_a_fully_accepted_step_replays_too_rather_than_keeping_the_chunk_state():
    """Full acceptance is NOT a no-op. The forward's own state came from the chunked kernel,
    which is a different (equally valid) approximation of the recurrence; keeping it would put
    every full-acceptance cycle -- the common case -- off the decode ladder."""
    want, got, _ = _scenario(cached_len=64, accepted=WIDTH)
    _assert_bitwise(want, got)


def test_the_chunk_kernels_own_state_is_not_the_decode_ladders_state():
    """Why the test above matters: measured, the w-row chunked forward leaves a recurrent state
    that differs from four plain decode steps far above any rounding tolerance."""
    ids = _token_ids(64 + WIDTH + 4)
    world = _World()
    hidden, ple_in = _inputs(WIDTH, world.config.hidden_size, world.args.ple_state_width, 2)
    _warm(world, 64, ids)
    world.pool.copy_from(SLOT, 7)  # keep the start state
    plain = _plain_decode_families(world, 64, ids, hidden, ple_in, WIDTH)["recurrent"]
    world.pool.copy_from(7, SLOT)
    world.prefill(hidden, ple_in, ids, cached_len=64, mtp_verify=True)
    chunk = world.families(SLOT)["recurrent"]

    scale = plain.abs().max()
    assert (chunk - plain).abs().max() > 1e-3 * scale


# ------------------------------------------------- what the bitwise gate had to hold still


def test_a_bf16_gemm_is_not_row_stable_in_its_batch_width():
    """The only inexactness between a w-row forward and w decode steps, and it sits upstream of
    every state family: the fused input projection. Pinned so the tolerance below is evidence,
    not a guess."""
    gen = torch.Generator(device=DEV).manual_seed(0)
    weight = torch.empty(3088, 256, device=DEV, dtype=DTYPE).normal_(0, 0.05, generator=gen)
    rows = torch.empty(WIDTH, 256, device=DEV, dtype=DTYPE).normal_(generator=gen)
    wide = torch.nn.functional.linear(rows, weight)
    assert any(
        not torch.equal(torch.nn.functional.linear(rows[i : i + 1], weight)[0], wide[i])
        for i in range(WIDTH)
    )


@pytest.mark.parametrize("accepted", [0, 2, 4])
def test_shipping_dense_weights_track_plain_decode_within_that_projection_noise(accepted):
    want, got, _ = _scenario(64, accepted, row_stable=False, seed=4)
    # the shift-register families carry raw projection rows, so they stay bitwise-close;
    # only the recurrent state accumulates the GEMM's last-place differences
    for name in (PLE_NGRAM_STATE,):
        assert torch.equal(got[name], want[name])
    for name in ("conv", "recurrent", PLE_CONV_STATE):
        w, g = want[name].float(), got[name].float()
        assert (g - w).abs().max() <= 2e-2 * w.abs().max().clamp_min(1e-3)


# ------------------------------------------------------------------------------ mechanics


def test_the_snapshot_slot_comes_out_of_the_pool_once_and_is_not_a_live_slot():
    world = _World()
    free_before = world.pool.num_free_slots
    ladder = SpecStateLadder(world.pool, WIDTH)
    assert world.pool.num_free_slots == free_before - 1
    assert ladder.slot not in (world.pool.padding_slot, SLOT)


def test_a_snapshot_and_a_full_undo_allocate_nothing():
    """Strategy R's per-cycle cost has to be preallocated: the snapshot, the restore and the
    replay all run out of buffers the ladder took at enable time."""
    ids = _token_ids(64 + WIDTH + 4)
    world = _World()
    hidden, ple_in = _inputs(WIDTH, world.config.hidden_size, world.args.ple_state_width, 2)
    _warm(world, 64, ids)
    ladder = SpecStateLadder(world.pool, WIDTH)
    # one warm cycle first: triton autotune / kernel caches allocate on their first launch
    req, _ = world.prefill(hidden, ple_in, ids, cached_len=64, mtp_verify=True, ladder=ladder)
    ladder.rollback(req, 2)

    torch.cuda.synchronize()
    req, batch = world.prefill(hidden, ple_in, ids, cached_len=64, mtp_verify=True, ladder=ladder)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    ladder.rollback(req, 2)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


def test_restoring_with_nothing_accepted_returns_the_slot_to_the_snapshot():
    ids = _token_ids(64 + WIDTH + 4)
    world = _World()
    hidden, ple_in = _inputs(WIDTH, world.config.hidden_size, world.args.ple_state_width, 2)
    _warm(world, 64, ids)
    before = world.families(SLOT)
    ladder = SpecStateLadder(world.pool, WIDTH)
    req, _ = world.prefill(hidden, ple_in, ids, cached_len=64, mtp_verify=True, ladder=ladder)
    assert not torch.equal(world.families(SLOT)["recurrent"], before["recurrent"])
    ladder.rollback(req, 0)
    _assert_bitwise(before, world.families(SLOT))


def test_an_ordinary_forward_stashes_nothing():
    ids = _token_ids(64 + WIDTH + 4)
    world = _World()
    hidden, ple_in = _inputs(WIDTH, world.config.hidden_size, world.args.ple_state_width, 2)
    _warm(world, 64, ids)
    ladder = SpecStateLadder(world.pool, WIDTH)
    _, batch = world.prefill(hidden, ple_in, ids, cached_len=64)
    assert batch.spec_capture is None
    with pytest.raises(RuntimeError, match="no speculative"):
        ladder.rollback(batch.reqs[0], 1)


def test_keeping_more_rows_than_the_step_forwarded_is_refused():
    ids = _token_ids(64 + WIDTH + 4)
    world = _World()
    hidden, ple_in = _inputs(WIDTH, world.config.hidden_size, world.args.ple_state_width, 2)
    _warm(world, 64, ids)
    ladder = SpecStateLadder(world.pool, WIDTH)
    req, _ = world.prefill(hidden, ple_in, ids, cached_len=64, mtp_verify=True, ladder=ladder)
    with pytest.raises(ValueError, match="accepted"):
        ladder.rollback(req, WIDTH + 1)


def test_a_step_wider_than_the_preallocated_arena_is_refused():
    ids = _token_ids(64 + WIDTH + 4)
    world = _World()
    hidden, ple_in = _inputs(WIDTH, world.config.hidden_size, world.args.ple_state_width, 2)
    _warm(world, 64, ids)
    ladder = SpecStateLadder(world.pool, 2)
    with pytest.raises(ValueError, match="width"):
        world.prefill(hidden, ple_in, ids, cached_len=64, mtp_verify=True, ladder=ladder)


def test_a_slot_state_the_ladder_cannot_roll_back_is_refused_at_construction():
    """A future per-request slot state would ride the same slot and be silently left holding
    the rejected rows. The ladder knows two shift registers and refuses anything else."""
    from freetoken.models.config import SlotStateSpec

    world = _World()
    pool = LinearStatePool(
        world.group, 8, DTYPE, DEV, tp_size=1,
        slot_states=(*world.config.slot_states, SlotStateSpec(name="mystery", shape=(4,))),
    )
    with pytest.raises(ValueError, match="mystery"):
        SpecStateLadder(pool, WIDTH)


def test_a_narrowed_recurrent_state_is_refused(monkeypatch):
    """The replay advances ``accepted`` rows in one launch, keeping the state in fp32
    registers; plain decode rounds to the pool dtype between steps. Off fp32 the two ladders
    part, so the ladder refuses rather than drift."""
    from freetoken.env import ENV

    world = _World()
    monkeypatch.setattr(ENV, "MAMBA_SSM_DTYPE", "bfloat16")
    pool = LinearStatePool(
        world.group, 8, DTYPE, DEV, tp_size=1, slot_states=world.config.slot_states
    )
    with pytest.raises(ValueError, match="fp32"):
        SpecStateLadder(pool, WIDTH)


def test_a_pool_rebuild_hands_the_ladder_a_fresh_slot():
    world = _World()
    ladder = SpecStateLadder(world.pool, WIDTH)
    world.pool.rebuild(6)
    ladder.rebind()
    assert 0 < ladder.slot < 6
    assert ladder.slot not in world.pool._free_slots


def teardown_module():
    core._GLOBAL_CTX = None


# ------------------------------------------------- the ladder under a captured verify graph


def test_the_stash_hooks_are_cuda_graph_capturable_and_refill_on_replay():
    """The stash hooks run INSIDE the speculative forward, so a captured verify graph bakes
    them. Both are pure fixed-shape D2D copies into a preallocated arena -- no allocation, no
    host round trip, no shape that depends on a value -- which is what makes that legal."""
    world = _World()
    ladder = SpecStateLadder(world.pool, WIDTH)
    ladder._live = SLOT
    ladder._width = WIDTH
    gen = torch.Generator(device=DEV).manual_seed(11)
    layer_id = world.group.layer_ids[0]
    gdn = world.gdn[layer_id]
    conv_dim = world.pool.conv_states.shape[2]
    conv_in = torch.empty(WIDTH, conv_dim, device=DEV, dtype=DTYPE).normal_(generator=gen)
    mixed = torch.empty_like(conv_in).normal_(generator=gen)
    a = torch.empty(WIDTH, world.group.num_value_heads, device=DEV, dtype=DTYPE).normal_(
        generator=gen
    )
    b = torch.empty_like(a).normal_(generator=gen)
    ple_in = torch.empty(
        WIDTH, world.args.ple_state_width, device=DEV, dtype=DTYPE
    ).normal_(generator=gen)

    def stash():
        ladder.stash_gdn(
            layer_id, conv_in=conv_in, mixed=mixed, a=a, b=b,
            A_log=gdn.A_log, dt_bias=gdn.dt_bias, scale=1.0,
        )
        ladder.stash_ple(world.ple.layer_id, ple_in)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stash()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        stash()

    conv_in.normal_(generator=gen)
    mixed.normal_(generator=gen)
    ple_in.normal_(generator=gen)
    graph.replay()
    torch.cuda.synchronize()

    li = world.pool.local_index(layer_id)
    km1 = world.pool.conv_states.shape[-1]
    assert torch.equal(
        ladder._conv_hist[li, :, km1 : km1 + WIDTH], conv_in.transpose(0, 1)
    )
    assert torch.equal(ladder._mixed[li, :WIDTH], mixed)
    row = ladder._ple_rows[world.ple.layer_id]
    state_len = ladder._ple_state_len
    assert torch.equal(
        ladder._ple_hist[row, :, state_len : state_len + WIDTH], ple_in.transpose(0, 1)
    )
    graph.reset()


def test_a_step_whose_forward_was_a_graph_replay_still_settles():
    """A captured forward never re-runs the Python stash hooks, so the per-layer kernel
    parameters they record must survive from the capture step to every replay step. They are
    module weights and a constant, so persisting them is also the only correct thing."""
    cached_len = 64
    ids = _token_ids(cached_len + WIDTH + 4)
    world = _World()
    hidden, ple_in = _inputs(WIDTH, world.config.hidden_size, world.args.ple_state_width, 2)
    _warm(world, cached_len, ids)
    ladder = SpecStateLadder(world.pool, WIDTH)

    req, _ = world.prefill(
        hidden, ple_in, ids, cached_len=cached_len, mtp_verify=True, ladder=ladder
    )
    ladder.rollback(req, WIDTH)
    # second cycle: begin arms the ladder, but the "forward" is a graph replay that refills
    # the arena without ever calling stash_gdn again
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = ids[cached_len : cached_len + WIDTH].to(DEV)
    batch.mtp_verify = True
    batch.emit_width = WIDTH
    ladder.begin(req, batch)

    ladder.rollback(req, 2)  # must not trip the "never stashed" assertion


def test_restore_snapshot_rewinds_the_slot_without_ending_the_step():
    """Capture's warm-up pass executes the forward and advances the live slot; the recorded
    pass executes nothing. This is the seam between them, and the step stays in flight."""
    cached_len = 64
    ids = _token_ids(cached_len + WIDTH + 4)
    world = _World()
    hidden, ple_in = _inputs(WIDTH, world.config.hidden_size, world.args.ple_state_width, 2)
    _warm(world, cached_len, ids)
    ladder = SpecStateLadder(world.pool, WIDTH)
    before = world.families(SLOT)

    req, _ = world.prefill(
        hidden, ple_in, ids, cached_len=cached_len, mtp_verify=True, ladder=ladder
    )
    advanced = world.families(SLOT)
    ladder.restore_snapshot()

    _assert_bitwise(before, world.families(SLOT))
    assert not torch.equal(advanced["recurrent"], before["recurrent"])
    ladder.rollback(req, 0)  # the step was still in flight and settles normally
    _assert_bitwise(before, world.families(SLOT))

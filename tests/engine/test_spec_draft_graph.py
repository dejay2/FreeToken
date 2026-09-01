"""The DRAFT side of a cycle, recorded into CUDA graphs.

``SpecDraftHead.propose`` in ``chain`` mode is 843 launches over 7.2 ms of kernels and the
commit forward that follows it another ~120 over 1 ms, so between them roughly two thirds of a
cycle's wall clock is the host pushing launch packets for a fixed sequence. Recording them is
only legal once every value that MOVES between cycles is a device input rather than a host
scalar baked into the record, and that is what this file pins:

* the CPU half -- the keying (mrope in the key, row count in the commit key, and NOTHING about
  the request's sampling parameters), the device-position staging arithmetic, the request
  filter's landing in the sampler's three cells, and every arm of the fallback, which must
  leave the eager chain byte for byte what it was;
* the GPU half -- capture and replay against a toy MTP head, where the graphed chain and the
  graphed commit must equal the eager ones BIT for BIT, including after ``committed_len``
  moves (which is the whole point of the ``_graph_base`` cell) and including a SAMPLED
  request, whose N replays must draw what N eager chains from the same seed draw.

The toy head is the ``tests/engine/test_spec_draft.py`` fixture moved onto the device: the
same hand-built ``SpecDraftHead`` with a fake staged model, a fake LM head and a fake
attention backend, except that every one of them now runs REAL device kernels whose outputs
depend on the batch's device-resident addressing. That is deliberate -- a fake that ignored
``positions`` / ``out_loc`` / ``seq_lens`` would pass whether or not those became device
inputs, which is the one thing that must not be assumable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Context, SamplingParams, set_global_ctx
from freetoken.engine.spec_draft import SpecDraftHead
from freetoken.engine.spec_draft_graph import (
    DRAFT_GRAPH_ENV,
    SpecDraftGraphRunner,
    draft_graph_enabled,
)

CPU = torch.device("cpu")
VOCAB = 16
HIDDEN = 8
HC = 2
WIDTH = HC * HIDDEN
TOPK = 2
RING = 8
PAGES = 4
PAGE_SIZE = 64
SLOTS = PAGES * PAGE_SIZE

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _cuda() -> torch.device:
    """The INDEXED device. ``MTPDraftSampler`` compares ``logits.device`` to its own, and a
    bare ``cuda`` never equals the ``cuda:0`` a tensor reports."""
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(autouse=True)
def _no_ctx_leak():
    import freetoken.core as core

    yield
    core._GLOBAL_CTX = None


# ------------------------------------------------------------------------------- the fakes


class _KVCache:
    """Only what the head reads: the two pending rings and the pool dtype."""

    def __init__(self, device):
        self.dtype = torch.float32
        self._kv_buffer = torch.zeros(SLOTS, HIDDEN, device=device)
        self._cmp_k_buffer = torch.zeros(SLOTS, HIDDEN, device=device)
        self._pending_ring = torch.zeros(1, 1, RING, HIDDEN, device=device)
        self._pending_position_ring = torch.zeros(
            1, 1, RING, 3, dtype=torch.int64, device=device
        )


class _Attention:
    """``prepare_metadata`` in the shape ``QSASparseAttnBackend._stage_step`` stages it.

    The one thing that matters here is the arm this whole change turns on: the sequence length
    is ``fill_``-ed from a host scalar unless the caller supplies a device cell, in which case
    it is copied from that cell so a replay re-reads it.
    """

    def __init__(self, device):
        self.device = device
        self.block_topk = TOPK
        self._idx_slot = {0: 0}
        self.step_seq_len_source = None
        self.seq_lens = torch.zeros(1, dtype=torch.int32, device=device)
        self.host_fills: list[int] = []

    def prepare_metadata(self, batch) -> None:
        length = int(batch.reqs[0].device_len)
        source = self.step_seq_len_source
        if source is None:
            self.host_fills.append(length)
            self.seq_lens.fill_(length)
        else:
            self.seq_lens.copy_(source)
        batch.attn_metadata = SimpleNamespace(seq_lens=self.seq_lens)


class _Staged:
    """A forward whose every output depends on the batch's DEVICE addressing.

    ``positions`` decide the arithmetic, ``out_loc`` decides which KV rows are written, and
    ``seq_lens`` (which only the metadata carries) is added in -- so a graph that baked any of
    the three would disagree with eager the moment ``committed_len`` moved.
    """

    def __init__(self, kv_cache, device):
        self.kv_cache = kv_cache
        self.device = device
        self.calls: list[tuple[int, int]] = []

    def forward(self, embeds, hidden, batch):
        rows = embeds.shape[0]
        self.calls.append((rows, batch.reqs[0].cached_len))
        positions = batch.positions.to(torch.float32).unsqueeze(1)
        lengths = batch.attn_metadata.seq_lens.to(torch.float32).reshape(1, 1)
        sample = embeds + hidden[:, :HIDDEN] * 0.5 + positions + lengths
        rope = batch.rope_positions
        if rope is not None:
            # a picture request's three-axis coordinates, which the chain derives per step
            sample = sample + rope[0].to(sample.dtype).unsqueeze(1)
        saved = getattr(batch, "mtp_qsa_saved_blocks", None)
        if isinstance(saved, dict):
            sample = sample + saved[0][:rows, :1].to(sample.dtype)
        # the private KV write, at the physical slots ``out_loc`` names
        self.kv_cache._kv_buffer.index_copy_(
            0, batch.out_loc.to(torch.int64), sample.detach()
        )
        ring_rows = batch.positions.to(torch.int64) % RING
        self.kv_cache._pending_ring[0, 0].index_copy_(0, ring_rows, sample.detach())
        capture = getattr(batch, "mtp_qsa_capture_blocks", None)
        if isinstance(capture, dict):
            capture[0] = (
                batch.positions.reshape(rows, 1).repeat(1, TOPK).to(torch.int32)
            )
        recursive = sample.repeat(1, HC)
        return sample, recursive


class _TargetModel:
    def __init__(self, device):
        generator = torch.Generator(device="cpu").manual_seed(4242)
        table = torch.randn(VOCAB, HIDDEN, generator=generator).to(device)
        head = torch.randn(VOCAB, HIDDEN, generator=generator).to(device)
        self.embed_table = table
        self.model = SimpleNamespace(
            embed_tokens=SimpleNamespace(
                forward=lambda ids: table.index_select(0, ids.to(torch.int64))
            )
        )
        self.lm_head = SimpleNamespace(forward_all=lambda hidden: hidden @ head.t())


def _head(device, *, depth=3, graphs=True, conf_cut=0.0, cut_mode="chain") -> SpecDraftHead:
    import freetoken.core as core

    core._GLOBAL_CTX = None
    ctx = Context(page_size=PAGE_SIZE)
    ctx.page_table = torch.zeros((2, SLOTS), dtype=torch.int32, device=device)
    ctx.kv_cache = None
    ctx.attn_backend = None
    set_global_ctx(ctx)

    head = object.__new__(SpecDraftHead)
    head.device = device
    head.depth = depth
    head.conf_cut = conf_cut
    head.draft_cut_mode = cut_mode
    head.seed = 1729
    head.num_pages = PAGES
    head.target_ctx = ctx
    head.target_model = _TargetModel(device)
    head.draft_lm_head = head.target_model.lm_head
    head.mtp_config = SimpleNamespace(
        hidden_size=HIDDEN, qwen4_args=SimpleNamespace(hc_count=HC)
    )
    # identity page table, exactly as ``_init_private_state`` builds it
    head.page_table = torch.arange(SLOTS, dtype=torch.int32, device=device).expand(2, -1)
    head.page_table = head.page_table.contiguous()
    head.kv_cache = _KVCache(device)
    head.attn_backend = _Attention(device)
    head.staged_model = _Staged(head.kv_cache, device)
    head._uid = None
    head.committed_len = 0
    head._pending_hidden = None
    head._pending_rope = None
    head._sample = None
    head._recursive = None
    head._saved_blocks = None
    head._buffered = []
    # one generator for the head's life, as ``SpecDraftHead.__init__`` builds it
    head._sampler = _sampler(device)
    if graphs and device.type == "cuda":
        head._graph_runner = SpecDraftGraphRunner(device=device)
    return head


def _prime(head, *, base: int) -> None:
    """Give the head a committed context to draft from, without a graph in sight."""
    device = head.device
    head.committed_len = base
    head._sample = torch.full((1, HIDDEN), 0.25, device=device)
    head._recursive = torch.full((1, WIDTH), -0.5, device=device)
    head._saved_blocks = {
        0: torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    }
    head._sampler.reseed(7)


def _sampler(device):
    from freetoken.models.qwen4_exp.mtp_spike import MTPDraftSampler

    return MTPDraftSampler(seed=7, device=device)


def _req(*, mrope=False, temperature=0.0, top_k=-1, top_p=1.0):
    req = SimpleNamespace(
        uid=1,
        sampling_params=SamplingParams(
            temperature=temperature, top_k=top_k, top_p=top_p
        ),
        mrope_position_ids=torch.zeros(3, 1) if mrope else None,
        mrope_position_delta=11 if mrope else 0,
    )
    return req


def _snapshot(head):
    return {
        "kv": head.kv_cache._kv_buffer.clone(),
        "ring": head.kv_cache._pending_ring.clone(),
        "positions": head.kv_cache._pending_position_ring.clone(),
        "len": head.committed_len,
    }


# ------------------------------------------------------------------------------ the CPU half


def test_the_off_switch_leaves_the_head_ungraphed(monkeypatch):
    monkeypatch.setenv(DRAFT_GRAPH_ENV, "0")
    assert not draft_graph_enabled()
    head = _head(CPU, graphs=False)
    head._init_graph_runner()
    assert head._graph_runner is None
    assert head._graph_disabled_reason == f"{DRAFT_GRAPH_ENV}=0"
    assert not head.graphs_enabled


def test_the_off_switch_refuses_anything_but_zero_or_one(monkeypatch):
    monkeypatch.setenv(DRAFT_GRAPH_ENV, "yes")
    with pytest.raises(ValueError, match=DRAFT_GRAPH_ENV):
        draft_graph_enabled()


def test_a_cpu_head_is_never_graphed():
    head = _head(CPU, graphs=False)
    head._init_graph_runner()
    assert head._graph_runner is None
    assert head._graph_disabled_reason == "CUDA_REQUIRED"


def test_step_cut_mode_is_never_graphed():
    """``step`` reads a confidence back to the host per drafted token, which is a device
    synchronization inside the chain and capture-illegal."""
    head = _head(CPU, graphs=False, conf_cut=0.5, cut_mode="step")
    head._init_graph_runner()
    assert head._graph_disabled_reason == "draft_cut_mode=step"


def test_an_ungraphed_head_takes_the_eager_chain_and_commit():
    """Every fallback arm must be the pre-graph path exactly: no key, no buffers, no runner."""
    head = _head(CPU, graphs=False)
    _prime(head, base=4)
    assert not head._chain_graphable()
    assert not head._commit_graphable(2)
    assert head._commit_graph(2, mrope=False) is None
    assert not head._graph_buffers_ready


def test_a_sampled_request_is_graphable_like_any_other():
    """The key carries nothing about the request. ``MTPDraftSampler.draw`` runs one fixed
    sequence of kernels over three device cells, so temperature / top-k / top-p are replay
    inputs, and the head's single generator is registered with the graph."""
    head = _head(CPU, graphs=False)
    head._graph_runner = object()  # armed as far as the key is concerned
    _prime(head, base=4)
    assert head._chain_graphable()
    assert head._chain_graph_key(mrope=False) == "chain:mrope=0"


def test_a_head_without_a_primed_context_is_not_graphable():
    head = _head(CPU, graphs=False)
    head._graph_runner = object()
    assert not head._chain_graphable()


def test_the_keys_separate_the_shapes_that_change_the_kernels():
    head = _head(CPU, graphs=False)
    keys = {
        head._chain_graph_key(mrope=False),
        head._chain_graph_key(mrope=True),
        head._commit_graph_key(1, mrope=False),
        head._commit_graph_key(2, mrope=False),
        head._commit_graph_key(2, mrope=True),
    }
    assert len(keys) == 5


def test_a_commit_wider_than_the_accepted_run_is_never_graphed():
    """Buffered flushes of other row counts stay eager -- their widths are unbounded."""
    head = _head(CPU, graphs=False)
    head._graph_runner = object()
    assert not head._commit_graphable(head.depth + 2)
    assert not head._commit_graphable(0)
    assert head._commit_graphable(head.depth + 1)


def test_the_runner_refuses_a_key_it_never_captured():
    runner = SpecDraftGraphRunner(device=CPU)
    with pytest.raises(RuntimeError, match="NOT_CAPTURED"):
        runner.replay("chain:mrope=0")


def test_a_cpu_runner_classifies_capture_as_permanently_unsupported():
    runner = SpecDraftGraphRunner(device=CPU)
    result = runner.capture("chain:mrope=0", lambda: None)
    assert result.status == "permanently-unsupported"
    assert result.reason == "CUDA_REQUIRED"
    assert not runner.capture_pending("chain:mrope=0")
    # ...and the verdict stands rather than being re-attempted
    assert runner.capture("chain:mrope=0", lambda: None) is result


def test_a_refunded_attempt_leaves_the_key_as_capturable_as_it_was():
    runner = SpecDraftGraphRunner(device=CPU)
    runner._attempts["commit:2:mrope=0"] = 3
    assert not runner.capture_pending("commit:2:mrope=0")
    runner.refund_attempt("commit:2:mrope=0")
    assert runner._attempts["commit:2:mrope=0"] == 2


def test_destroy_drops_everything_and_refuses_further_capture():
    runner = SpecDraftGraphRunner(device=CPU)
    runner.destroy()
    assert runner.graph_count == 0
    assert runner.capture("x", lambda: None).reason == "RUNNER_DESTROYED"


def test_the_device_position_staging_reproduces_the_host_arithmetic():
    """``positions``/``out_loc``/``device_len`` must be exactly what the host slice wrote."""
    head = _head(CPU, graphs=False)
    head._graph_base = torch.zeros(1, dtype=torch.int32)
    head._graph_len = torch.zeros(1, dtype=torch.int32)
    head._graph_index64 = {}
    head._graph_offsets = {}
    positions = torch.zeros(3, dtype=torch.int32)
    out_loc = torch.zeros(3, dtype=torch.int32)

    head._graph_base.fill_(9)
    length = head._stage_device_positions(3, 2, positions, out_loc)

    assert positions.tolist() == [11, 12, 13]
    assert out_loc.tolist() == head.page_table[0, 11:14].tolist()
    assert int(length[0]) == 9 + 2 + 3
    # the cell IS the input: moving it and restaging moves everything downstream
    head._graph_base.fill_(0)
    head._stage_device_positions(3, 2, positions, out_loc)
    assert positions.tolist() == [2, 3, 4]


# ------------------------------------------------- the request's filter, as device cells
#
# The chain graph carries no sampling parameters in its key, because it carries none in its
# record either: ``request_filter_params`` decides the triple on the host and
# ``MTPDraftSampler.stage`` writes it into three cells the replay reads.


@pytest.mark.parametrize(
    "params,expected",
    [
        (SamplingParams(temperature=0.0), (0.0, -1, 1.0)),
        (SamplingParams(temperature=0.9), (0.9, -1, 1.0)),
        # top_k == 1 IS greedy to the server (SamplingParams.is_greedy), so it lands on the
        # argmax cell rather than on a one-wide filter
        (SamplingParams(temperature=0.9, top_k=1), (0.0, -1, 1.0)),
        (SamplingParams(temperature=0.9, top_k=0), (0.9, -1, 1.0)),
        (SamplingParams(temperature=0.9, top_k=40, top_p=0.95), (0.9, 40, 0.95)),
        # temperature 0 with a top_p is SAMPLED by the server (Sampler.prepare floors it)
        (SamplingParams(temperature=0.0, top_p=0.8), (1e-6, -1, 0.8)),
    ],
)
def test_the_request_filter_lands_in_the_samplers_three_cells(params, expected):
    from freetoken.engine.spec_sample import request_filter_params

    sampler = _sampler(CPU)
    triple = request_filter_params(params)
    assert triple == pytest.approx(expected)

    sampler.stage(temperature=triple[0], top_k=triple[1], top_p=triple[2])
    assert float(sampler.temperature_cell[0]) == pytest.approx(expected[0])
    assert int(sampler.top_k_cell[0]) == expected[1]
    assert float(sampler.top_p_cell[0]) == pytest.approx(expected[2])


def test_a_zero_temperature_cell_draws_the_argmax_bit_for_bit():
    """The greedy id is still ``torch.argmax(logits)`` -- selected by a ``where`` on the cell
    rather than by a host branch, which is what let the record cover both kinds of request."""
    sampler = _sampler(CPU)
    generator = torch.Generator().manual_seed(5)
    for _ in range(16):
        logits = torch.randn(VOCAB, generator=generator) * 3
        sampler.stage(temperature=0.0, top_k=-1, top_p=1.0)
        assert torch.equal(sampler.draw(logits), torch.argmax(logits))
        sampler.stage(temperature=0.9, top_k=1, top_p=1.0)
        assert torch.equal(sampler.draw(logits), torch.argmax(logits))


def test_an_absent_top_k_or_top_p_masks_nothing():
    """``top_k <= 0`` and ``top_p >= 1`` are the "whole vocabulary" spellings, and branchless
    they have to reach every token the unfiltered softmax gives mass to."""
    sampler = _sampler(CPU)
    logits = torch.zeros(VOCAB)  # a uniform distribution: every token must be reachable
    sampler.stage(temperature=1.0, top_k=None, top_p=None)
    seen = {int(sampler.draw(logits)) for _ in range(4000)}
    assert seen == set(range(VOCAB))

    sampler.stage(temperature=1.0, top_k=-1, top_p=1.0)
    assert {int(sampler.draw(logits)) for _ in range(4000)} == set(range(VOCAB))

    # ...and a real top_k does bite
    sampler.stage(temperature=1.0, top_k=2, top_p=1.0)
    ordered = torch.arange(VOCAB, dtype=torch.float32)
    assert {int(sampler.draw(ordered)) for _ in range(400)} == {VOCAB - 1, VOCAB - 2}


def test_a_batch_outside_a_graph_body_stages_from_the_host_length():
    head = _head(CPU, graphs=False)
    head.committed_len = 6
    head._batch(2)
    assert head.attn_backend.host_fills == [8]
    assert head.attn_backend.step_seq_len_source is None


# ------------------------------------------------------------------------------ the GPU half


@requires_cuda
def test_a_graphed_chain_equals_the_eager_chain_bit_for_bit():
    device = _cuda()
    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=12)
    req = _req()
    expected = eager.propose(req, 3)
    after_eager = _snapshot(eager)

    head = _head(device, graphs=True, depth=3)
    _prime(head, base=12)
    graphed = head.propose(req, 3)

    assert head._graph_runner.available(head._chain_graph_key(mrope=False))
    assert graphed.tokens == expected.tokens
    assert torch.equal(graphed.logits, expected.logits)
    # and the undo is the same undo: the length is back and both rings are restored
    assert head.committed_len == after_eager["len"] == 12
    assert torch.equal(head.kv_cache._pending_ring, after_eager["ring"])


@requires_cuda
def test_a_replay_follows_committed_len_because_the_base_is_a_device_cell():
    """The one value a naive record would bake. Replaying at a DIFFERENT base has to equal the
    eager chain at that base, or the graph is serving the capture's context forever."""
    device = _cuda()
    head = _head(device, graphs=True, depth=3)
    _prime(head, base=12)
    req = _req()
    head.propose(req, 3)  # captures at base 12

    _prime(head, base=40)
    moved = head.propose(req, 3)

    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=40)
    expected = eager.propose(req, 3)

    assert moved.tokens == expected.tokens
    assert torch.equal(moved.logits, expected.logits)


@requires_cuda
def test_the_graphed_chain_reads_this_cycles_seed_rows_and_saved_blocks():
    """The other three moving inputs. Feeding different rows through the SAME graph must give
    what the eager chain gives for those rows."""
    device = _cuda()
    head = _head(device, graphs=True, depth=3)
    _prime(head, base=12)
    req = _req()
    head.propose(req, 3)

    head._sample = torch.full((1, HIDDEN), -1.75, device=device)
    head._recursive = torch.full((1, WIDTH), 0.875, device=device)
    head._saved_blocks = {0: torch.tensor([[1, 2]], dtype=torch.int32, device=device)}
    replayed = head.propose(req, 3)

    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=12)
    eager._sample = head._sample.clone()
    eager._recursive = head._recursive.clone()
    eager._saved_blocks = {0: head._saved_blocks[0].clone()}
    expected = eager.propose(req, 3)

    assert replayed.tokens == expected.tokens
    assert torch.equal(replayed.logits, expected.logits)


@requires_cuda
def test_one_recorded_chain_serves_every_shallower_depth():
    """The graph always drafts the full depth; a shallower proposal keeps the prefix, which is
    the trade ``chain`` mode already makes for the confidence cut."""
    device = _cuda()
    head = _head(device, graphs=True, depth=3)
    _prime(head, base=12)
    req = _req()
    full = head.propose(req, 3)
    short = head.propose(req, 2)

    assert head._graph_runner.graph_count == 1
    assert short.tokens == full.tokens[:2]
    assert torch.equal(short.logits, full.logits[:2])


@requires_cuda
def test_a_graphed_commit_equals_the_eager_commit_bit_for_bit():
    device = _cuda()
    hidden = torch.arange(2 * WIDTH, dtype=torch.float32, device=device).reshape(2, WIDTH)
    embeds = torch.linspace(-1, 1, 2 * HIDDEN, device=device).reshape(2, HIDDEN)

    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=12)
    eager._commit_pairs(hidden, embeds, None)

    head = _head(device, graphs=True, depth=3)
    _prime(head, base=12)
    head._commit_pairs(hidden, embeds, None)

    assert head._graph_runner.available(head._commit_graph_key(2, mrope=False))
    assert torch.equal(head._sample, eager._sample)
    assert torch.equal(head._recursive, eager._recursive)
    assert torch.equal(head._saved_blocks[0], eager._saved_blocks[0])
    assert head.committed_len == eager.committed_len == 14
    assert torch.equal(head.kv_cache._kv_buffer, eager.kv_cache._kv_buffer)
    assert torch.equal(head.kv_cache._pending_ring, eager.kv_cache._pending_ring)


@requires_cuda
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_every_accepted_run_width_replays_like_eager(rows):
    device = _cuda()
    hidden = torch.arange(rows * WIDTH, dtype=torch.float32, device=device).reshape(
        rows, WIDTH
    )
    embeds = torch.linspace(-1, 1, rows * HIDDEN, device=device).reshape(rows, HIDDEN)

    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=9)
    eager._commit_pairs(hidden, embeds, None)

    head = _head(device, graphs=True, depth=3)
    _prime(head, base=9)
    head._commit_pairs(hidden, embeds, None)  # captures
    head.committed_len = 9
    head._commit_pairs(hidden, embeds, None)  # replays

    assert torch.equal(head._sample, eager._sample)
    assert head.committed_len == 9 + rows


@requires_cuda
def test_a_sampled_chain_replays_exactly_what_the_eager_chain_would_draw():
    """THE point of registering the generator: N replays draw what N eager chains draw.

    The state is matched by reseeding both heads AFTER the graph is captured -- capture's
    warm-up draws too, and a graph is recorded once while the stream goes on advancing.
    """
    device = _cuda()
    req = _req(temperature=0.9, top_k=8, top_p=0.95)

    head = _head(device, graphs=True, depth=3)
    head.capture_graphs_at_boot()
    _prime(head, base=12)
    head._sampler.reseed(31)
    graphed = [head.propose(req, 3).tokens for _ in range(6)]

    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=12)
    eager._sampler.reseed(31)
    expected = [eager.propose(req, 3).tokens for _ in range(6)]

    assert graphed == expected
    # ...and the philox offset really is advancing: a frozen one would repeat the first chain
    assert len(set(graphed)) > 1


@requires_cuda
def test_a_registered_generator_still_draws_outside_the_graph():
    """Registration must not strand the eager path. Every fallback the head has -- a buffered
    flush, a width the graph does not cover, a capture that never succeeded -- runs the eager
    chain through the SAME generator the record holds."""
    device = _cuda()
    head = _head(device, graphs=True, depth=3)
    head.capture_graphs_at_boot()
    logits = torch.randn(VOCAB, device=device)

    drawn = [int(head._sampler.sample(logits, temperature=0.8)) for _ in range(8)]

    assert len(drawn) == 8
    assert all(0 <= token < VOCAB for token in drawn)
    # ...and an eager proposal on the graphed head still works after a replay
    _prime(head, base=12)
    head.propose(_req(temperature=0.8), 3)
    head._graph_runner = None
    assert len(head.propose(_req(temperature=0.8), 3).tokens) == 3


@requires_cuda
def test_a_sampled_chain_and_a_greedy_one_share_one_recorded_graph():
    """The key carries no sampling parameters, so the same record serves both -- and the greedy
    replay is still bit-for-bit the eager argmax chain."""
    device = _cuda()
    head = _head(device, graphs=True, depth=3)
    head.capture_graphs_at_boot()
    _prime(head, base=12)

    sampled = head.propose(_req(temperature=0.9, top_k=8), 3)
    greedy = head.propose(_req(), 3)

    chains = [k for k in head._graph_runner.keys if k.startswith("chain:")]
    assert chains == [head._chain_graph_key(mrope=False)]
    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=12)
    assert greedy.tokens == eager.propose(_req(), 3).tokens
    # both chains start from the same seed row, so row 0's logits are the same tensor of
    # numbers; from row 1 they diverge, because the sampled draw fed a different token back in
    assert torch.equal(greedy.logits[0], sampled.logits[0])
    assert greedy.tokens != sampled.tokens


@requires_cuda
def test_a_graphed_sampled_proposal_is_drawn_from_the_q_acceptance_divides_by():
    """The speculative-sampling theorem's precondition: every proposed token has to carry mass
    under ``filtered_probs`` of the row it was drawn from, or acceptance divides by zero."""
    from freetoken.engine.sample import BatchSamplingArgs
    from freetoken.engine.spec_sample import filtered_probs

    device = _cuda()
    temperature, top_k, top_p = 0.9, 6, 0.9
    req = _req(temperature=temperature, top_k=top_k, top_p=top_p)
    head = _head(device, graphs=True, depth=3)
    head.capture_graphs_at_boot()
    _prime(head, base=12)

    args = BatchSamplingArgs(
        temperatures=torch.tensor([temperature], device=device),
        top_k=torch.tensor([top_k], device=device),
        top_p=torch.tensor([top_p], device=device),
    )
    for _ in range(8):
        proposal = head.propose(req, 3)
        q = filtered_probs(proposal.logits, args)
        for row, token in enumerate(proposal.tokens):
            assert float(q[row, token]) > 0.0


@requires_cuda
def test_a_picture_requests_chain_derives_its_rope_origin_on_the_device():
    """``committed_len + mrope_position_delta + step`` -- three values, one of which moves per
    request -- so the origin is a cell and the per-step coordinate is derived inside the graph.
    """
    device = _cuda()
    req = _req(mrope=True)

    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=12)
    expected = eager.propose(req, 3)

    head = _head(device, graphs=True, depth=3)
    _prime(head, base=12)
    graphed = head.propose(req, 3)

    assert head._graph_runner.keys == (head._chain_graph_key(mrope=True),)
    assert graphed.tokens == expected.tokens
    assert torch.equal(graphed.logits, expected.logits)


@requires_cuda
def test_a_picture_requests_commit_replays_its_staged_coordinates():
    device = _cuda()
    hidden = torch.arange(2 * WIDTH, dtype=torch.float32, device=device).reshape(2, WIDTH)
    embeds = torch.linspace(-1, 1, 2 * HIDDEN, device=device).reshape(2, HIDDEN)
    rope = torch.arange(23, 25, device=device).expand(3, -1).contiguous()

    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=12)
    eager._commit_pairs(hidden, embeds, rope)

    head = _head(device, graphs=True, depth=3)
    _prime(head, base=12)
    head._commit_pairs(hidden, embeds, rope)

    assert head._graph_runner.keys == (head._commit_graph_key(2, mrope=True),)
    assert torch.equal(head._sample, eager._sample)
    assert torch.equal(head._recursive, eager._recursive)


@requires_cuda
def test_boot_capture_records_the_chain_and_every_commit_width():
    device = _cuda()
    head = _head(device, graphs=True, depth=3)

    results = head.capture_graphs_at_boot()

    assert set(results) == {
        head._chain_graph_key(mrope=False),
        *(head._commit_graph_key(n, mrope=False) for n in range(1, 5)),
    }
    assert set(results.values()) == {"captured"}
    # ...and the head comes out of it pristine
    assert head.committed_len == 0 and head._saved_blocks is None


@requires_cuda
def test_a_boot_captured_chain_serves_the_first_live_proposal():
    device = _cuda()
    head = _head(device, graphs=True, depth=3)
    head.capture_graphs_at_boot()
    calls = len(head.staged_model.calls)

    _prime(head, base=12)
    proposal = head.propose(_req(), 3)

    # the replay ran no Python at all: the staged model's call log did not grow
    assert len(head.staged_model.calls) == calls
    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=12)
    expected = eager.propose(_req(), 3)
    assert proposal.tokens == expected.tokens


@requires_cuda
def test_a_capture_failure_falls_back_to_the_eager_chain_without_raising(monkeypatch):
    device = _cuda()
    head = _head(device, graphs=True, depth=3)
    _prime(head, base=12)

    def _explode(*_args, **_kwargs):
        raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(SpecDraftHead, "_chain_graph_body", _explode)
    proposal = head.propose(_req(), 3)

    key = head._chain_graph_key(mrope=False)
    assert not head._graph_runner.available(key)
    assert head._graph_runner.last_attempt(key).status in {
        "retryable",
        "permanently-unsupported",
    }
    eager = _head(device, graphs=False, depth=3)
    _prime(eager, base=12)
    assert proposal.tokens == eager.propose(_req(), 3).tokens


@requires_cuda
def test_the_probe_sees_the_replay_and_the_readback_separately():
    from freetoken.scheduler.scheduler import _SpecTimingProbe

    device = _cuda()
    head = _head(device, graphs=True, depth=3)
    head.capture_graphs_at_boot()
    _prime(head, base=12)
    probe = _SpecTimingProbe(device)
    probe.start_cycle()

    head.propose(_req(), 3, probe=probe)
    head._commit_pairs(
        torch.zeros(2, WIDTH, device=device),
        torch.zeros(2, HIDDEN, device=device),
        None,
        probe=probe,
    )

    assert {"draft.stage", "draft.replay", "draft.readback", "tail.commit.replay"} <= set(
        probe.stages
    )

"""The QSA backend's reusable per-forward staging.

Armed only by the private MTP draft head (``SpecDraftHead._init_private_state``). That head
runs an UNGRAPHED chain of tiny one-request forwards whose shapes repeat exactly -- one row per
recursive draft step, N rows per buffered flush -- and, having no ``init_capture_graph``, it
took the fully allocating path every time: two pinned host tensors in ``prepare_metadata``, a
third in ``_snapshot_decode``, and half a dozen ``torch.empty`` transients in ``_scratch``.
Pinned allocation is the expensive one -- a miss in the caching host allocator is a
``cudaHostAlloc``, which synchronizes the device.

Pinned here: armed staging produces the SAME metadata the allocating path produces (this is
the correctness bar -- the addressing decides which K/V rows a forward reads), it allocates
nothing after the first forward of a shape, and an unarmed backend is untouched.
"""

from __future__ import annotations

import pytest
import torch

import freetoken.attention.qsa_sparse as qsa
from freetoken.attention.qsa_sparse import QSASparseAttnBackend

from .common import Fixture, parsed_config

PAGES = 8


@pytest.fixture(autouse=True)
def _host_memory_is_not_pinnable_on_cpu(monkeypatch):
    """``pin_memory=True`` needs CUDA; the staging logic does not."""
    monkeypatch.setitem(qsa._CPU_PINNED, "pin_memory", False)


@pytest.fixture
def pair():
    """One CPU pool and page table, an armed backend and an unarmed reference over it."""
    fixture = Fixture(parsed_config(), num_pages=PAGES, device="cpu")
    reference = QSASparseAttnBackend(fixture.config)
    fixture.backend.enable_step_workspace()
    return fixture, reference


def _decode_req(fixture, table_idx: int, prefix: int = 5):
    """A one-row request whose page row is really allocated (``allocate`` only fills WHOLE
    pages, so a request that never starts at zero leaves the table zeroed)."""
    req = fixture.req(table_idx, 0, prefix)
    fixture.step(req)
    return req


def _metadata(backend, fixture, req, phase):
    batch = fixture.batch([req], phase)
    # ``Fixture.batch`` prepares through the fixture's own (armed) backend; re-prepare through
    # whichever one this call is about, over the identical request.
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    if md.block_table is None:
        backend._snapshot_decode(md, batch)
    return md


_STAGED = ("last_indices", "token_to_req", "cu_seqlens", "seq_lens", "ring_slots", "block_table")


@pytest.mark.parametrize("rows,phase", [(1, "decode"), (3, "prefill")])
def test_armed_staging_reproduces_the_allocating_paths_addressing(pair, rows, phase):
    fixture, reference = pair
    req = _decode_req(fixture, 0) if rows == 1 else fixture.req(0, 0, rows)

    # the reference first: the armed metadata is ONE reused object, so preparing again would
    # overwrite the values being compared
    plain = {
        name: getattr(_metadata(reference, fixture, req, phase), name).clone()
        for name in _STAGED
    }
    armed = _metadata(fixture.backend, fixture, req, phase)

    assert armed.is_decode == (rows == 1)
    for name in _STAGED:
        assert torch.equal(getattr(armed, name), plain[name]), name
    assert armed.qo_indptr_cpu.tolist() == [0, rows]
    assert armed.kv_len_cpu.tolist() == [req.device_len]


@pytest.mark.parametrize("rows,phase", [(1, "decode"), (3, "prefill")])
def test_a_second_forward_of_a_shape_allocates_nothing(pair, rows, phase):
    fixture, _ = pair
    backend = fixture.backend
    first_req = _decode_req(fixture, 0) if rows == 1 else fixture.req(0, 0, rows)
    second_req = _decode_req(fixture, 1) if rows == 1 else fixture.req(1, 0, rows)

    first = _metadata(backend, fixture, first_req, phase)
    addresses = {name: getattr(first, name).data_ptr() for name in _STAGED}
    pinned = (first.qo_indptr_cpu.data_ptr(), first.kv_len_cpu.data_ptr())

    second = _metadata(backend, fixture, second_req, phase)

    assert second is first  # one metadata object per shape, refilled in place
    for name in _STAGED:
        assert getattr(second, name).data_ptr() == addresses[name], name
    assert (second.qo_indptr_cpu.data_ptr(), second.kv_len_cpu.data_ptr()) == pinned


def test_the_lengths_and_the_page_row_still_move_between_forwards(pair):
    """Reuse must not mean staleness: everything that CAN change between two forwards of one
    shape -- the sequence length, the ring slot and the page row -- is refilled."""
    fixture, _ = pair
    backend = fixture.backend
    first = _metadata(backend, fixture, _decode_req(fixture, 0, prefix=5), "decode")
    length, table = int(first.seq_lens[0]), first.block_table.clone()

    second = _metadata(backend, fixture, _decode_req(fixture, 1, prefix=9), "decode")

    assert int(second.seq_lens[0]) == 10 != length
    assert int(second.ring_slots[0]) == 1
    assert not torch.equal(second.block_table, table)


def test_each_row_count_gets_its_own_staging(pair):
    fixture, _ = pair
    backend = fixture.backend
    one = _metadata(backend, fixture, _decode_req(fixture, 0), "decode")
    three = _metadata(backend, fixture, fixture.req(1, 0, 3), "prefill")

    assert one is not three
    assert one.token_to_req.numel() == 1 and three.token_to_req.numel() == 3
    assert one.is_decode and not three.is_decode


def test_the_transient_scratch_is_kept_at_its_high_water_mark(pair):
    """The other half of the per-step allocation: ``_scratch`` reallocated every forward
    because an ungraphed backend has no static buffers. Armed, it grows once per shape."""
    fixture, _ = pair
    backend = fixture.backend

    first = backend._scratch("pooled", 4, 16, dtype=torch.bfloat16)
    again = backend._scratch("pooled", 4, 16, dtype=torch.bfloat16)
    narrower = backend._scratch("pooled", 2, 16, dtype=torch.bfloat16)
    assert again.data_ptr() == first.data_ptr()
    assert narrower.data_ptr() == first.data_ptr() and narrower.shape[0] == 2

    wider = backend._scratch("pooled", 9, 16, dtype=torch.bfloat16)
    assert wider.data_ptr() != first.data_ptr()
    assert backend._scratch("pooled", 4, 16, dtype=torch.bfloat16).data_ptr() == wider.data_ptr()


def test_a_growing_high_water_mark_retires_the_old_buffer_rather_than_freeing_it(pair):
    """A GRAPHED chain baked the address of the buffer it was captured against.

    Freeing it when a later, wider eager flush grows the mark would leave every replay writing
    through a dangling pointer, so the replaced buffer is retired (kept alive) instead. Every
    one of these transients is written before it is read within a forward, so the graph's
    retired copy stays a correct private scratch.
    """
    fixture, _ = pair
    backend = fixture.backend

    first = backend._scratch("pooled", 4, 16, dtype=torch.bfloat16)
    wider = backend._scratch("pooled", 9, 16, dtype=torch.bfloat16)

    assert wider.data_ptr() != first.data_ptr()
    assert any(held.data_ptr() == first.data_ptr() for held in backend._step_scratch_retired)
    # ...and the retired buffer is still writable and readable at its captured address
    first.fill_(1.0)
    assert float(first[0, 0]) == 1.0


def test_a_device_length_source_replaces_the_baked_host_scalar(pair):
    """``fill_`` bakes its host argument into the launch, which a captured chain would then
    serve forever. Armed with a source cell, the length becomes a device-to-device copy the
    replay re-reads -- and the value is the same one the host scalar would have written."""
    fixture, _ = pair
    backend = fixture.backend
    req = _decode_req(fixture, 0, prefix=5)

    md = _metadata(backend, fixture, req, "decode")
    baked = int(md.seq_lens[0])

    source = torch.tensor([baked + 7], dtype=torch.int32)
    backend.step_seq_len_source = source
    try:
        armed = _metadata(backend, fixture, req, "decode")
        assert int(armed.seq_lens[0]) == baked + 7
        # the cell is the live one: rewriting it and restaging follows it
        source.fill_(baked)
        restaged = _metadata(backend, fixture, req, "decode")
        assert int(restaged.seq_lens[0]) == baked
    finally:
        backend.step_seq_len_source = None

    # cleared, the host scalar is back and nothing about the unarmed path has moved
    assert int(_metadata(backend, fixture, req, "decode").seq_lens[0]) == baked


def test_an_unarmed_backend_has_no_length_source(pair):
    """The target's path must be exactly what it was: the class default reads None, so a
    backend built by hand (the shadow tooling does) never reaches the copy at all."""
    _, reference = pair

    assert reference.step_seq_len_source is None
    assert QSASparseAttnBackend.step_seq_len_source is None
    assert QSASparseAttnBackend._step_scratch_retired is None


def test_an_unarmed_backend_keeps_allocating_per_forward(pair):
    """The target's path must be exactly what it was: no workspace, no scratch cache."""
    _, reference = pair

    assert reference._step_ws is None and reference._step_scratch is None
    first = reference._scratch("pooled", 4, 16, dtype=torch.bfloat16)
    assert reference._scratch("pooled", 4, 16, dtype=torch.bfloat16).data_ptr() != first.data_ptr()


def test_arming_is_idempotent(pair):
    fixture, _ = pair
    backend = fixture.backend
    _metadata(backend, fixture, _decode_req(fixture, 0), "decode")
    workspaces = backend._step_ws

    backend.enable_step_workspace()

    assert backend._step_ws is workspaces


def test_a_multi_request_batch_falls_back_to_the_allocating_path(pair):
    """The workspace is keyed by row count alone, which is only a key while there is one
    request; a batch of two must take the ragged path rather than be mis-staged."""
    fixture, _ = pair
    reqs = [_decode_req(fixture, 0, prefix=5), _decode_req(fixture, 1, prefix=9)]
    batch = fixture.batch(reqs, "decode")

    md = batch.attn_metadata

    assert fixture.backend._step_ws == {}
    assert md.kv_len_cpu.tolist() == [6, 10]

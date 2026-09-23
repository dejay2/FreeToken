"""RAM-tier page-major gather (2026-09-23): byte identity with the per-view serialization,
chunking on page boundaries, the on-device prefix check and no per-save pinning."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import freetoken.kernel.pinned as pinned_module
import freetoken.kvcache.park_store as park_module
from freetoken.kvcache.park_store import ParkStore, _PageSource

from .test_park_store import (  # noqa: F401  (_tp is an autouse fixture)
    _bases,
    _entry_for,
    _fill_raw,
    _page_views,
    _qsa_pool,
    _raw_bytes,
    _state_pool,
    _store,
    _tp,
)

KV_DTYPES = [torch.bfloat16, torch.float8_e4m3fn]


def _old_serialization(store: ParkStore, bases: torch.Tensor, slot: int) -> torch.Tensor:
    """The pre-2026-09-23 bytes: one copy per page view, then the state views."""
    span = store._entry_views(bases, slot)
    out = torch.empty(span.nbytes, dtype=torch.uint8)
    ParkStore._copy_span_to_window(span, 0, span.nbytes, out)
    return out


def _page_bytes(store: ParkStore) -> int:
    return store.page_size * store._kv_bytes_per_token()


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
def test_regions_match_page_byte_views_row_for_row(kv_dtype):
    kv = _qsa_pool(num_pages=9, kv_dtype=kv_dtype)
    regions = kv.page_byte_regions()
    for page in range(9):
        views = kv.page_byte_views(page)
        assert len(views) == len(regions)
        for region, view in zip(regions, views, strict=True):
            assert region.dtype == torch.uint8 and region.dim() == 2
            assert region[page].data_ptr() == view.data_ptr()
            assert torch.equal(region[page], view.view(torch.uint8).reshape(-1))


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
def test_gather_is_byte_identical_to_per_view_serialization(tmp_path: Path, kv_dtype):
    kv, state = _qsa_pool(num_pages=16, kv_dtype=kv_dtype), _state_pool()
    slot = state.alloc(1)[0]
    pages = [11, 3, 7, 0, 14, 5]  # shuffled and non-contiguous
    _fill_raw(_page_views(kv, range(16)), seed=4)
    _fill_raw(state.slot_byte_views(slot), seed=5)
    store = _store("ram", tmp_path, kv, state)
    try:
        bases = _bases(pages)
        source = store._page_source(bases, slot)
        assert isinstance(source, _PageSource)
        want = _old_serialization(store, bases, slot)
        assert torch.equal(store._copy_to_ram(source), want)
        # A page-aligned suffix is the same bytes as the old span suffix.
        cut = 2 * _page_bytes(store)
        assert torch.equal(store._copy_to_ram(source.suffix(cut)), want[cut:])
        with pytest.raises(ValueError):
            source.suffix(cut + 1)
    finally:
        store.close()


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
def test_ram_buffer_equals_the_ssd_file_regions(tmp_path: Path, kv_dtype):
    kv, state = _qsa_pool(num_pages=16, kv_dtype=kv_dtype), _state_pool()
    slot = state.alloc(1)[0]
    pages = [9, 2, 13, 4]
    _fill_raw(_page_views(kv, pages), seed=8)
    _fill_raw(state.slot_byte_views(slot), seed=9)
    tokens = torch.arange(16, dtype=torch.int32)
    ram = _store("ram", tmp_path / "ram", kv, state)
    ssd = _store("ssd", tmp_path / "ssd", kv, state)
    try:
        assert ram.save(tokens, _bases(pages), slot)
        assert ssd.save(tokens, _bases(pages), slot)
        ram_entry, ssd_entry = _entry_for(ram, tokens), _entry_for(ssd, tokens)
        raw = ssd_entry.path.read_bytes()
        kv_part = raw[ssd_entry.kv_offset : ssd_entry.kv_offset + ssd_entry.kv_bytes]
        state_part = raw[ssd_entry.state_offset : ssd_entry.state_offset + ssd_entry.state_bytes]
        assert bytes(ram_entry.ram_buffer.numpy()) == kv_part + state_part
    finally:
        ram.close()
        ssd.close()


@pytest.mark.parametrize(
    "chunk_pages", [0.5, 1, 1.5, 2, 5, 6, 7, 1000], ids=lambda v: f"chunk{v}"
)
def test_chunking_stays_on_page_boundaries(tmp_path: Path, monkeypatch, chunk_pages):
    """Sub-page, fractional, exact, entry-sized and larger-than-entry chunks."""
    kv, state = _qsa_pool(num_pages=16, kv_dtype=torch.float8_e4m3fn), _state_pool()
    slot = state.alloc(1)[0]
    pages = [15, 1, 8, 2, 12, 6]
    _fill_raw(_page_views(kv, range(16)), seed=21)
    _fill_raw(state.slot_byte_views(slot), seed=22)
    store = _store("ram", tmp_path, kv, state)
    page_bytes = _page_bytes(store)
    store._stage_chunk_bytes = int(chunk_pages * page_bytes)
    calls: list[tuple[int, int, tuple[int, ...]]] = []
    real_gather = _PageSource.gather

    def recording_gather(self, first, count, out):
        calls.append((first, count, tuple(out.shape)))
        return real_gather(self, first, count, out)

    monkeypatch.setattr(_PageSource, "gather", recording_gather)
    try:
        bases = _bases(pages)
        got = store._copy_to_ram(store._page_source(bases, slot))
        assert torch.equal(got, _old_serialization(store, bases, slot))
        per_chunk = max(1, int(chunk_pages * page_bytes) // page_bytes)
        expected_calls = -(-len(pages) // per_chunk)
        assert len(calls) == expected_calls
        assert sum(count for _first, count, _shape in calls) == len(pages)
        for first, count, shape in calls:
            assert first % per_chunk == 0
            assert shape == (count, page_bytes), "a chunk split a page"
        assert store._staging is not None
        assert store._staging.numel() >= 2 * per_chunk * page_bytes
    finally:
        store.close()
    assert store._staging is None


@pytest.mark.parametrize("chunk_pages", [1, 2, 100])
@pytest.mark.parametrize("page_offset", [0, 1, 3])
@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
def test_ram_chain_restores_exactly_across_chunks(
    tmp_path: Path, chunk_pages, page_offset, kv_dtype
):
    kv, state = _qsa_pool(num_pages=32, kv_dtype=kv_dtype), _state_pool(num_slots=8)
    source_slot, target_slot = state.alloc(2)
    store = _store("ram", tmp_path, kv, state)
    store._stage_chunk_bytes = chunk_pages * _page_bytes(store)
    try:
        tokens = torch.arange(24, dtype=torch.int32)
        _fill_raw(_page_views(kv, [3, 9, 1]), seed=1)
        _fill_raw(state.slot_byte_views(source_slot), seed=2)
        assert store.save(tokens[:12], _bases([3, 9, 1]), source_slot)
        _fill_raw(_page_views(kv, [20, 7, 30]), seed=3)
        _fill_raw(state.slot_byte_views(source_slot), seed=4)
        assert store.save(tokens, _bases([3, 9, 1, 20, 7, 30]), source_slot)
        entry = _entry_for(store, tokens)
        assert entry.parent_key is not None and entry.parent_token_count == 12
        expected_kv = _raw_bytes(_page_views(kv, [3, 9, 1, 20, 7, 30][page_offset:]))
        expected_state = _raw_bytes(state.slot_byte_views(source_slot))
        target = [25, 11, 17, 5, 28, 14][page_offset:]
        _fill_raw(_page_views(kv, target), seed=99)
        store.restore(entry, _bases(target), target_slot, page_offset=page_offset)
        assert all(
            torch.equal(a, b)
            for a, b in zip(_raw_bytes(_page_views(kv, target)), expected_kv, strict=True)
        )
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                _raw_bytes(state.slot_byte_views(target_slot)), expected_state, strict=True
            )
        )
        assert store.status()["last_restore_breakdown_ms"]["kv_bytes"] == float(
            len(target) * _page_bytes(store)
        )
    finally:
        store.close()


@pytest.mark.parametrize("region_index", [0, 4, -1])  # K/V, index, fp8 scale
@pytest.mark.parametrize("chunk_pages", [1, 100])
def test_continuation_prefix_check_on_device(tmp_path: Path, region_index, chunk_pages):
    """A match links to the parent; one flipped byte in any region of a borrowed page makes
    the save a standalone root carrying the new bytes."""
    kv = _qsa_pool(num_pages=32, kv_dtype=torch.float8_e4m3fn)
    state = _state_pool(num_slots=8)
    slot = state.alloc(1)[0]
    store = _store("ram", tmp_path, kv, state)
    store._stage_chunk_bytes = chunk_pages * _page_bytes(store)
    try:
        tokens = torch.arange(24, dtype=torch.int32)
        prefix_pages = [4, 12, 2, 19]
        _fill_raw(_page_views(kv, prefix_pages), seed=6)
        _fill_raw(state.slot_byte_views(slot), seed=7)
        assert store.save(tokens[:16], _bases(prefix_pages), slot)
        parent = _entry_for(store, tokens[:16])

        # Matching bytes: the continuation borrows the parent's pages.
        _fill_raw(_page_views(kv, [8]), seed=10)
        assert store.save(tokens[:20], _bases(prefix_pages + [8]), slot)
        child = _entry_for(store, tokens[:20])
        assert child.parent_key == parent.key and child.parent_token_count == 16

        # One flipped byte in the third borrowed page (a later chunk at chunk_pages=1).
        views = kv.page_byte_views(prefix_pages[2])
        views[region_index].view(torch.uint8).reshape(-1)[-1] ^= 0x5A
        _fill_raw(_page_views(kv, [27]), seed=11)
        pages = prefix_pages + [8, 27]
        assert store.save(tokens, _bases(pages), slot)
        root = _entry_for(store, tokens)
        assert root.parent_key is None and root.parent_token_count == 0
        want = _old_serialization(store, _bases(pages), slot)
        assert torch.equal(root.ram_buffer, want)
        assert store.status()["disabled"] is False
    finally:
        store.close()


def test_ram_mode_never_pins_host_memory(tmp_path: Path, monkeypatch):
    """No pinned allocation per save, per prefix check or per restore (2026-09-23)."""
    kv, state = _qsa_pool(num_pages=32), _state_pool(num_slots=8)
    source_slot, target_slot = state.alloc(2)
    store = _store("ram", tmp_path, kv, state)

    def no_pinning(*_args, **_kwargs):
        raise AssertionError("RAM parking pinned host memory")

    monkeypatch.setattr(pinned_module, "alloc_pinned_tensor", no_pinning)
    monkeypatch.setattr(ParkStore, "_allocate_window", no_pinning)
    try:
        tokens = torch.arange(20, dtype=torch.int32)
        _fill_raw(_page_views(kv, [1, 2, 3]), seed=1)
        _fill_raw(state.slot_byte_views(source_slot), seed=2)
        assert store.save(tokens[:12], _bases([1, 2, 3]), source_slot)
        _fill_raw(_page_views(kv, [4, 5]), seed=3)
        assert store.save(tokens, _bases([1, 2, 3, 4, 5]), source_slot)
        entry = _entry_for(store, tokens)
        assert entry.parent_key is not None
        assert not entry.ram_buffer.is_pinned()
        store.restore(entry, _bases([6, 7, 8, 9, 10]), target_slot)
        assert store.status()["disabled"] is False
    finally:
        store.close()


def test_mismatched_regions_fall_back_to_the_per_view_path(tmp_path: Path, monkeypatch):
    kv, state = _qsa_pool(num_pages=16), _state_pool()
    source_slot, target_slot = state.alloc(2)
    real_regions = kv.page_byte_regions
    monkeypatch.setattr(kv, "page_byte_regions", lambda: tuple(reversed(real_regions())))
    store = _store("ram", tmp_path, kv, state)
    try:
        assert store._page_source(_bases([1, 2]), source_slot) is None
        _fill_raw(_page_views(kv, [1, 2]), seed=41)
        _fill_raw(state.slot_byte_views(source_slot), seed=42)
        tokens = torch.arange(8, dtype=torch.int32)
        want = _old_serialization(store, _bases([1, 2]), source_slot)
        assert store.save(tokens, _bases([1, 2]), source_slot)
        entry = _entry_for(store, tokens)
        assert torch.equal(entry.ram_buffer, want)
        store.restore(entry, _bases([9, 10]), target_slot)
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                _raw_bytes(_page_views(kv, [9, 10])),
                _raw_bytes(_page_views(kv, [1, 2])),
                strict=True,
            )
        )
    finally:
        store.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
def test_cuda_ram_save_is_pageable_and_byte_identical(tmp_path: Path, monkeypatch, kv_dtype):
    kv = _qsa_pool(num_pages=32, kv_dtype=kv_dtype, device="cuda:0")
    state = _state_pool(num_slots=8, device="cuda:0")
    source_slot, target_slot = state.alloc(2)
    store = _store("ram", tmp_path, kv, state)
    store._stage_chunk_bytes = 2 * _page_bytes(store)
    calls = []
    real_alloc = pinned_module.alloc_pinned_tensor
    monkeypatch.setattr(
        pinned_module,
        "alloc_pinned_tensor",
        lambda *a, **k: calls.append(a) or real_alloc(*a, **k),
    )
    try:
        tokens = torch.arange(24, dtype=torch.int32)
        pages = [30, 2, 17, 5, 11, 23]
        _fill_raw(_page_views(kv, pages), seed=51)
        _fill_raw(state.slot_byte_views(source_slot), seed=52)
        assert store.save(tokens[:12], _bases(pages[:3]), source_slot)
        assert store.save(tokens, _bases(pages), source_slot)
        entry = _entry_for(store, tokens)
        assert entry.parent_key is not None and not entry.ram_buffer.is_pinned()
        suffix_want = _old_serialization(store, _bases(pages), source_slot)[
            12 * store._kv_bytes_per_token() :
        ]
        assert torch.equal(entry.ram_buffer, suffix_want)
        store.restore(entry, _bases([1, 3, 4, 6, 7, 8]), target_slot)
        torch.cuda.synchronize()
        for a, b in zip(
            _raw_bytes(_page_views(kv, [1, 3, 4, 6, 7, 8])),
            _raw_bytes(_page_views(kv, pages)),
            strict=True,
        ):
            assert torch.equal(a, b)
        assert calls == []
    finally:
        store.close()


def test_module_exposes_one_staging_chunk_constant():
    # Guards the VRAM bound the design relies on: two halves of this size at most.
    assert park_module._STAGE_CHUNK_BYTES == 32 << 20

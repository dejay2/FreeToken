"""Mapped picture weights against the real RadixArk/Qwen3.8-Flash-Next-NVFP4 checkpoint.

Everything here is CPU-only and reads no more than the 897,862,112-byte picture extent, so
it runs while another job owns the GPU. The synthetic tests in ``test_weight.py`` pin the
contract; these pin that the contract holds against the actual file -- the layout facts the
design was measured on, and byte identity with ``safetensors.safe_open`` for all 333
tensors.

The checkpoint comes from ``FREETOKEN_QWEN4EXP_MODEL`` when set, else the standard local
path. Skipped when neither exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import safetensors
import torch

from freetoken.models.qwen4_exp import weight as weight_mod

_DEFAULT_MODEL = r"D:\Models\Qwen3.8-Flash-Next-NVFP4"
MODEL_PATH = os.environ.get("FREETOKEN_QWEN4EXP_MODEL") or _DEFAULT_MODEL

pytestmark = [
    pytest.mark.needs_weights,
    pytest.mark.skipif(
        not Path(MODEL_PATH).is_dir(), reason=f"checkpoint {MODEL_PATH} is not present"
    ),
]

# Measured 2026-09-02 on the real checkpoint with the server stopped. These are the numbers
# the design's latency and RAM claims rest on, so a checkpoint re-export that moves them
# should fail here rather than silently invalidate the measurements.
PICTURE_TENSORS = 333
PICTURE_BYTES = 897_862_112
PICTURE_SHARD = "model-bf16-00001.safetensors"
POS_EMBED_BYTES = 5_308_416


@pytest.fixture(scope="module")
def layout() -> weight_mod.VisionLayout:
    return weight_mod._vision_layout(MODEL_PATH)


@pytest.fixture(scope="module")
def source(layout):
    holder = weight_mod.MmapVisionWeights(layout)
    try:
        yield holder
    finally:
        holder.close()


def test_the_picture_weights_are_one_contiguous_extent_of_one_shard(layout):
    assert len(layout.shards) == 1, "every picture tensor lives in one bf16 shard"
    shard = layout.shards[0]
    assert os.path.basename(shard.path) == PICTURE_SHARD
    assert len(shard.tensors) == PICTURE_TENSORS
    assert layout.nbytes == PICTURE_BYTES
    # span == bytes: no gaps and no non-picture tensor interleaved, which is why one
    # prefetch over the whole window reads nothing the encode will not use.
    assert layout.span == PICTURE_BYTES
    assert layout.dtype is torch.bfloat16


def test_the_extent_ends_at_the_last_byte_of_the_shard(layout):
    shard = layout.shards[0]
    assert shard.end == os.path.getsize(shard.path)


def test_every_picture_tensor_starts_at_an_odd_offset(layout):
    """The shard's data base is odd (header 47,581 B -> base 47,589), so every bf16 tensor
    is 2-byte misaligned. That is legal for a ``cudaMemcpy`` source and is the reason
    ``pos_embed`` -- the one CPU compute operand -- is carved out as a resident tensor."""
    offsets = {spec.offset % 2 for spec in layout.shards[0].tensors}
    assert offsets == {1}


def test_the_component_byte_budget_matches_the_design(layout):
    by_component = {"blocks": 0, "merger": 0, "patch_embed": 0, "pos_embed": 0}
    counts = dict.fromkeys(by_component, 0)
    for spec in layout.shards[0].tensors:
        component = spec.name.split(".")[1]
        by_component[component] += spec.nbytes
        counts[component] += 1
    assert counts == {"blocks": 324, "merger": 6, "patch_embed": 2, "pos_embed": 1}
    assert by_component == {
        "blocks": 822_933_216,
        "merger": 66_079_232,
        "patch_embed": 3_541_248,
        "pos_embed": POS_EMBED_BYTES,
    }
    assert by_component["blocks"] % 27 == 0, "27 identical transformer blocks"


def test_every_mapped_view_is_byte_identical_to_the_checkpoint(source, layout):
    """Acceptance criterion 8, off-GPU: the bytes are the same bytes. All 333 tensors, not
    a sample -- a layout bug would most likely hit exactly the one that was not checked."""
    shard = layout.shards[0]
    with safetensors.safe_open(shard.path, framework="pt", device="cpu") as handle:
        for spec in shard.tensors:
            view = source.tensor(spec.name)
            reference = handle.get_tensor(spec.raw_name)
            assert view.shape == reference.shape, spec.name
            assert view.dtype is reference.dtype, spec.name
            # through int16, so a bf16 NaN payload still compares as bytes
            assert torch.equal(view.view(torch.int16), reference.view(torch.int16)), spec.name


def test_every_view_but_pos_embed_lives_inside_the_mapping(source, layout):
    """Acceptance criterion 2. A picture tensor outside the mapping has been copied, and the
    856 MiB saving with it."""
    outside = [
        spec.name
        for spec in layout.shards[0].tensors
        if not source.contains(source.tensor(spec.name).data_ptr())
    ]
    assert outside == ["visual.pos_embed.weight"]
    pos_embed = source.tensor("visual.pos_embed.weight")
    assert pos_embed.data_ptr() % 2 == 0, "the CPU F.embedding operand must be aligned"
    assert source.mapped_bytes == PICTURE_BYTES - POS_EMBED_BYTES


def test_the_mapping_charges_only_the_extent_not_the_whole_shard(source, layout):
    """One window, aligned down to the allocation granularity: ~857 MiB of commit charge
    instead of the 1.27 GiB the whole file would take."""
    assert len(source.windows) == 1
    window = source.windows[0]
    shard = layout.shards[0]
    assert window.file_offset % (64 << 10) == 0
    assert window.file_offset <= shard.start
    assert shard.start - window.file_offset < (64 << 10)
    assert window.span == shard.end - window.file_offset
    assert window.span < PICTURE_BYTES + (64 << 10)


def test_every_mapped_view_is_a_valid_copy_source(source, layout):
    """The riskiest assumption in the design, against the real file: a 2-byte-misaligned
    bf16 view of a mapped page is a correct ``copy_`` source, and copying FROM it does not
    write THROUGH it. A copy-on-write fault there would make the page private and dirty and
    silently give back the 856 MiB this mode exists to save.

    This faults the whole extent in, so it doubles as the off-GPU read smoke test. The
    target is CPU here; production copies into a CUDA workspace, which is the same ``copy_``
    with a different target device and is on the operator's live checklist.
    """
    shard = layout.shards[0]
    with torch.no_grad():
        for spec in shard.tensors:
            view = source.tensor(spec.name)
            address = view.data_ptr()
            target = torch.empty_like(view)
            target.copy_(view, non_blocking=False)
            assert torch.equal(target.view(torch.int16), view.view(torch.int16)), spec.name
            assert view.data_ptr() == address, f"{spec.name} moved"
            if spec.name not in source.resident_names:
                assert source.contains(view.data_ptr()), f"{spec.name} left the mapping"


def test_the_real_extent_is_mapped_read_only(source):
    """Read-only, not copy-on-write: on Windows that is PAGE_READONLY, which charges no
    commit, where the ACCESS_COPY reservation charged the full 856 MiB and left the live
    saving at 347 MiB instead of the 800 the design asked for (live-results section 3.2).

    Asserted against the mapping, not through a view: torch has no read-only tensor, so a
    write through a view would take the process down with an access violation rather than
    raise. The views are only ever ``copy_`` sources -- proved for all 333 of them by
    ``test_every_mapped_view_is_a_valid_copy_source`` above, which runs over this same
    read-only mapping.
    """
    assert len(source._maps) == 1
    mapping = next(iter(source._maps.values()))
    view = memoryview(mapping)
    try:
        assert view.readonly is True
    finally:
        view.release()  # an exported pointer would block mapping.close()
    with pytest.raises(TypeError, match="readonly"):
        mapping[0:1] = b"\x00"


def test_a_prefetch_over_the_real_extent_succeeds_and_is_not_reissued(source):
    """The real syscall over the real 856 MiB, no GPU involved. Asynchronous: it returns
    while the reads continue, which is the whole basis of the overlap argument."""
    if weight_mod._prefetch_virtual_memory is None:
        pytest.skip("PrefetchVirtualMemory is Windows-only")
    weight_mod._vision_prefetch_failed = False
    source.release_prefetch()

    assert source.prefetch() is True
    assert source.prefetch() is True, "idempotent while the picture is still being encoded"
    source.release_prefetch()
    assert source.prefetch() is True

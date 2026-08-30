from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.scheduler import Scheduler, _make_rope_positions
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

IMAGE_TOKEN = 99
CHUNK = 8


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _stack():
    _setup_context()
    page_table = torch.zeros((5, 64), dtype=torch.int32)
    cache = CacheManager(num_pages=64, page_size=1, page_table=page_table, type="radix")
    table = TableManager(max_running_reqs=4, page_table=page_table)
    decode = DecodeManager(page_size=1)
    prefill = PrefillManager(cache, table, decode)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(
        model_config=SimpleNamespace(image_token_id=IMAGE_TOKEN)
    )
    return cache, table, prefill, scheduler


def _drive_private_picture_prompt(input_ids: torch.Tensor):
    cache, table, prefill, scheduler = _stack()
    picture_rows = int((input_ids == IMAGE_TOKEN).sum().item())
    features = torch.arange(picture_rows * 3, dtype=torch.float32).view(picture_rows, 3)
    positions = torch.stack(
        (
            torch.arange(len(input_ids), dtype=torch.int64),
            torch.arange(len(input_ids), dtype=torch.int64) + 100,
            torch.arange(len(input_ids), dtype=torch.int64) + 200,
        )
    )
    pending = PendingReq(
        uid=7,
        input_ids=input_ids,
        sampling_params=SamplingParams(max_tokens=1),
        mm_embeds=features,
        cache_private=True,
        mrope_position_ids=positions,
        mrope_position_delta=-3,
    )
    prefill.pending_list = [pending]

    feature_slices = []
    position_slices = []
    admissions = []
    final_req = None
    while prefill.runnable:
        batch = prefill.schedule_next_batch(CHUNK)
        assert batch is not None
        admissions.append(list(batch.prompt_admissions))
        cache.allocate_paged(batch.reqs)
        batch.padded_reqs = batch.reqs
        Scheduler._gather_multimodal(scheduler, batch)
        feature_slices.append(None if batch.mm_embeds is None else batch.mm_embeds.clone())
        position_slices.append(_make_rope_positions(batch, torch.device("cpu")).clone())
        for req in batch.reqs:
            assert req.mm_embeds is None
            req.complete_one()
            if not isinstance(req, ChunkedReq):
                final_req = req
        if prefill.runnable:
            assert prefill.pending_list[0].mm_embeds is features
            assert prefill.pending_list[0].cache_private

    assert final_req is not None
    cache.cache_req(final_req, finished=False)
    assert cache.prefix_cache.size_info.evictable_size == 0
    cache.cache_req(final_req, finished=True)
    table.free(final_req.table_idx)
    cache.check_integrity()
    assert cache.page_usage()[0] == 0
    return features, positions, feature_slices, position_slices, admissions


def test_chunked_picture_features_and_positions_follow_each_token_range():
    # Two picture spans both cross an eight-token step boundary: [6, 10) and [15, 17).
    input_ids = torch.arange(24, dtype=torch.int32) + 1000
    input_ids[[6, 7, 8, 9, 15, 16]] = IMAGE_TOKEN

    features, positions, feature_slices, position_slices, admissions = (
        _drive_private_picture_prompt(input_ids)
    )

    assert [part[:, 0].tolist() for part in feature_slices] == [
        features[:2, 0].tolist(),
        features[2:5, 0].tolist(),
        features[5:, 0].tolist(),
    ]
    assert torch.equal(torch.cat(feature_slices), features)
    assert torch.equal(torch.cat(position_slices, dim=1), positions)
    assert admissions[0] == [(7, len(input_ids), 0)]
    assert admissions[1:] == [[], []]


def test_chunk_without_picture_placeholders_gets_no_picture_features():
    input_ids = torch.arange(16, dtype=torch.int32) + 1000
    input_ids[[8, 9]] = IMAGE_TOKEN

    features, _positions, feature_slices, _position_slices, _admissions = (
        _drive_private_picture_prompt(input_ids)
    )

    assert feature_slices[0] is None
    assert torch.equal(feature_slices[1], features)

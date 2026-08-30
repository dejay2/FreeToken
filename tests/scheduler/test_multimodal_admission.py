from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.message import ErrorReplyMsg, UserMsg
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.utils import PendingReq


IMAGE_TOKEN = 99


def _message(uid: int = 1) -> UserMsg:
    return UserMsg(
        uid=uid,
        input_ids=torch.tensor([5, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 6]),
        sampling_params=SamplingParams(max_tokens=4),
        mm_pixel_values=torch.randn(16, 24, dtype=torch.bfloat16),
        mm_image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.int64),
        mm_token_type_ids=torch.tensor([0, 1, 1, 1, 1, 0], dtype=torch.int64),
    )


def _scheduler(feature_rows: int = 4):
    calls = []

    class Model:
        def encode_images(self, pixels, grid):
            calls.append((pixels.clone(), grid.clone()))
            return torch.arange(feature_rows * 8, dtype=torch.float32).view(feature_rows, 8)

    added = []
    sent = []
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.device = torch.device("cpu")
    scheduler.prefill_budget = 8192
    scheduler.engine = SimpleNamespace(max_seq_len=262_144, model=Model())
    scheduler.config = SimpleNamespace(
        model_config=SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=IMAGE_TOKEN,
        )
    )
    scheduler.prefill_manager = SimpleNamespace(add_one_req=added.append)
    scheduler.send_result = sent.extend
    return scheduler, calls, added, sent


def test_scheduler_encodes_picture_builds_mrope_and_releases_raw_tensors():
    scheduler, calls, added, sent = _scheduler()
    message = _message()

    Scheduler._process_one_msg(scheduler, message)

    assert len(calls) == 1 and sent == [] and added == [message]
    assert message.mm_embeds.shape == (4, 8)
    assert message.mrope_position_ids.shape == (3, 6)
    assert isinstance(message.mrope_position_delta, int)
    assert message.mm_pixel_values is None
    assert message.mm_image_grid_thw is None
    assert message.mm_token_type_ids is None


def test_scheduler_rejects_feature_placeholder_mismatch_without_stopping_server():
    scheduler, calls, added, sent = _scheduler(feature_rows=3)
    message = _message(uid=2)

    Scheduler._process_one_msg(scheduler, message)

    assert len(calls) == 1 and added == []
    assert len(sent) == 1 and isinstance(sent[0], ErrorReplyMsg)
    assert "picture-token slots" in sent[0].error
    assert message.mm_pixel_values is None
    assert message.mm_image_grid_thw is None
    assert message.mm_token_type_ids is None


def test_scheduler_admits_picture_prompt_above_one_prefill_batch():
    scheduler, calls, added, sent = _scheduler()
    message = _message(uid=3)
    message.input_ids = torch.full((8193,), 5, dtype=torch.int32)
    message.input_ids[-4:] = IMAGE_TOKEN
    message.mm_token_type_ids = torch.zeros(8193, dtype=torch.int64)
    message.mm_token_type_ids[-4:] = 1

    Scheduler._process_one_msg(scheduler, message)

    assert len(calls) == 1 and added == [message] and sent == []
    assert message.mm_embeds.shape == (4, 8)
    assert message.mrope_position_ids.shape == (3, 8193)
    assert message.mm_pixel_values is None
    assert message.mm_image_grid_thw is None
    assert message.mm_token_type_ids is None


def test_scheduler_still_rejects_picture_prompt_at_combined_context_limit():
    scheduler, calls, added, sent = _scheduler()
    message = _message(uid=30)
    message.input_ids = torch.full((262_144,), 5, dtype=torch.int32)

    Scheduler._process_one_msg(scheduler, message)

    assert calls == [] and added == []
    assert len(sent) == 1 and sent[0].code == "context_length_exceeded"
    assert "prompt is too long" in sent[0].error
    assert message.mm_pixel_values is None
    assert message.mm_image_grid_thw is None
    assert message.mm_token_type_ids is None


def test_prefill_marks_picture_request_private_then_releases_soft_embeddings_after_gather():
    message = _message(uid=4)
    message.mm_embeds = torch.ones(4, 8)
    manager = PrefillManager(None, None, None)
    manager.add_one_req(message)
    pending = manager.pending_list[0]
    assert pending.cache_private

    request = SimpleNamespace(
        input_ids=message.input_ids,
        cached_len=0,
        device_len=len(message.input_ids),
        mm_embeds=pending.mm_embeds,
        cache_private=True,
    )
    batch = SimpleNamespace(reqs=[request], mm_embeds=None)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(
        model_config=SimpleNamespace(image_token_id=IMAGE_TOKEN)
    )
    Scheduler._gather_multimodal(scheduler, batch)

    assert batch.mm_embeds.shape == (4, 8)
    assert request.mm_embeds is None
    assert request.cache_private


def test_private_picture_placeholder_tokens_never_match_shared_prefix_cache():
    matched = []

    class Prefix:
        def match_prefix(self, ids):
            matched.append(ids.clone())
            return "empty-handle"

    cache = CacheManager.__new__(CacheManager)
    cache.prefix_cache = Prefix()
    cache.is_swa = False
    cache.is_hybrid = False
    pending = PendingReq(
        uid=5,
        input_ids=torch.tensor([1, IMAGE_TOKEN, IMAGE_TOKEN, 2]),
        sampling_params=SamplingParams(max_tokens=1),
        mm_embeds=torch.ones(2, 8),
        cache_private=True,
    )

    assert CacheManager.match_req(cache, pending) == "empty-handle"
    assert matched[0].numel() == 0

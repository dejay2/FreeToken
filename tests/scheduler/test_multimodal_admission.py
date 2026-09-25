from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.message import ErrorReplyMsg, UserMsg
from freetoken.scheduler import scheduler as scheduler_module
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


def _scheduler(feature_rows: int = 4, encode_error: Exception | None = None):
    calls = []

    class Model:
        def encode_images(self, pixels, grid):
            calls.append((pixels.device, grid.device, tuple(pixels.shape), grid.tolist()))
            if encode_error is not None:
                raise encode_error
            return torch.arange(feature_rows * 8, dtype=torch.float32).view(feature_rows, 8)

    added = []
    sent = []
    scheduler = Scheduler.__new__(Scheduler)
    # A non-CPU engine device proves that the scheduler leaves transport tensors on
    # the CPU and lets the model own placement.
    scheduler.device = torch.device("meta")
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

    assert calls == [
        (torch.device("cpu"), torch.device("cpu"), (16, 24), [[1, 4, 4]])
    ]
    assert sent == [] and added == [message]
    assert message.mm_embeds.shape == (4, 8)
    assert message.mrope_position_ids.shape == (3, 6)
    assert isinstance(message.mrope_position_delta, int)
    assert message.mm_pixel_values is None
    assert message.mm_image_grid_thw is None
    assert message.mm_token_type_ids is None


def test_scheduler_logs_encoder_time_without_changing_admission(monkeypatch):
    scheduler, _calls, added, sent = _scheduler()
    message = _message(uid=77)
    logs = []
    ticks = iter((100.0, 104.25))
    monkeypatch.setattr(scheduler_module.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        scheduler_module.logger, "info_rank0", lambda *args: logs.append(args)
    )

    Scheduler._process_one_msg(scheduler, message)

    assert added == [message] and sent == []
    assert logs == [("Picture encoder request %d: %.3f seconds", 77, 4.25)]


def test_scheduler_releases_raw_tensors_when_streamed_encoding_fails():
    scheduler, calls, added, sent = _scheduler(
        encode_error=RuntimeError("injected streamed failure")
    )
    message = _message(uid=78)

    Scheduler._process_one_msg(scheduler, message)

    assert len(calls) == 1 and added == []
    assert len(sent) == 1 and isinstance(sent[0], ErrorReplyMsg)
    assert "injected streamed failure" in sent[0].error
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


# --------------------------------------------------------------------------------------
# The picture-weight prefetch hook
# --------------------------------------------------------------------------------------


def _scheduler_with_prefetch(prefetch_error: Exception | None = None):
    """A model that also offers the optional ``prefetch_picture_weights`` hook."""
    order = []

    class Model:
        def prefetch_picture_weights(self):
            order.append("prefetch")
            if prefetch_error is not None:
                raise prefetch_error

        def encode_images(self, pixels, grid):
            order.append("encode")
            return torch.arange(4 * 8, dtype=torch.float32).view(4, 8)

    scheduler, _calls, added, sent = _scheduler()
    scheduler.engine = SimpleNamespace(max_seq_len=262_144, model=Model())
    return scheduler, order, added, sent


def test_scheduler_prefetches_picture_weights_once_before_encoding():
    """In mmap mode the ~856 MiB extent may be entirely non-resident. The syscall returns
    while the reads continue, so issuing it at admission overlaps the read with the encode's
    own GPU work -- which only helps if it happens before the encode, exactly once."""
    scheduler, order, added, sent = _scheduler_with_prefetch()

    Scheduler._process_one_msg(scheduler, _message(uid=11))

    assert order == ["prefetch", "encode"]
    assert len(added) == 1 and not sent


def test_scheduler_prefetches_once_per_picture_request():
    scheduler, order, added, _sent = _scheduler_with_prefetch()

    Scheduler._process_one_msg(scheduler, _message(uid=12))
    Scheduler._process_one_msg(scheduler, _message(uid=13))

    assert order == ["prefetch", "encode", "prefetch", "encode"]
    assert len(added) == 2


def test_a_failing_prefetch_never_costs_the_request():
    """The prefetch is an optimization. Losing it costs latency, not the picture."""
    scheduler, order, added, sent = _scheduler_with_prefetch(
        prefetch_error=OSError("no working set quota")
    )

    Scheduler._process_one_msg(scheduler, _message(uid=14))

    assert order == ["prefetch", "encode"]
    assert len(added) == 1 and not sent


def test_a_model_without_the_prefetch_hook_still_admits_pictures():
    """``ram`` mode and every non-Qwen multimodal model: the hook is optional."""
    scheduler, calls, added, sent = _scheduler()
    assert not hasattr(scheduler.engine.model, "prefetch_picture_weights")

    Scheduler._process_one_msg(scheduler, _message(uid=15))

    assert len(calls) == 1 and len(added) == 1 and not sent


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
        extend_len=len(message.input_ids),
        mm_embeds=pending.mm_embeds,
        cache_private=True,
    )
    batch = SimpleNamespace(reqs=[request], mm_embeds=None)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.device = torch.device("cpu")
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


class _Stream:
    def __init__(self, name):
        self.name, self.waited_on = name, []

    def wait_stream(self, other):
        self.waited_on.append(other.name)


def _stream_harness(monkeypatch, scheduler, *, exl3: bool):
    """Fake CUDA streams: the scheduler runs on ``sched`` while ``engine`` may be busy."""
    from freetoken.kernel import exl3_launch

    sched, engine = _Stream("sched"), _Stream("engine")
    state = {"current": sched}

    class _StreamCtx:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.prev, state["current"] = state["current"], self.stream

        def __exit__(self, *exc):
            state["current"] = self.prev

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: state["current"])
    monkeypatch.setattr(torch.cuda, "stream", _StreamCtx)
    scheduler.engine.stream = engine
    scheduler.config.model_config.vision_config.exl3 = exl3
    seen = []
    model = scheduler.engine.model
    original = model.encode_images

    def encode(pixels, grid):
        seen.append((state["current"].name, exl3_launch._full_label(None)))
        return original(pixels, grid)

    model.encode_images = encode
    return sched, engine, seen


def test_exl3_picture_encode_runs_on_the_engine_stream_after_the_scheduling_stream(monkeypatch):
    # H-D (hang investigation 2026-09-25): encoding an EXL3 tower on the scheduling stream
    # while a batch runs on the engine stream puts ExLlamaV3 kernels on two streams at once.
    scheduler, _calls, added, sent = _scheduler()
    sched, engine, seen = _stream_harness(monkeypatch, scheduler, exl3=True)

    Scheduler._process_one_msg(scheduler, _message(uid=90))

    assert sent == [] and len(added) == 1
    assert seen == [("engine", "picture")]
    assert engine.waited_on == ["sched"]


def test_bf16_picture_encode_keeps_its_stream(monkeypatch):
    scheduler, _calls, added, sent = _scheduler()
    sched, engine, seen = _stream_harness(monkeypatch, scheduler, exl3=False)

    Scheduler._process_one_msg(scheduler, _message(uid=91))

    assert sent == [] and len(added) == 1
    assert seen == [("sched", "?")] and engine.waited_on == []

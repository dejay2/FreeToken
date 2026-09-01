from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.mtp_fast_verify import MTPVerifyGraphRunner, _MTPVerifyGraphBuffer
from freetoken.engine.mtp_shadow import tensor_sha256


class _Context:
    @contextmanager
    def forward_batch(self, batch):
        yield batch


class _Attention:
    def __init__(self):
        self.prepared = []
        self.staged = []
        self.discarded = []
        self.reset_count = 0

    def prepare_mtp_verify_graph(self, batch):
        self.prepared.append(int(batch.input_ids.shape[0]))

    def stage_mtp_verify_graph(self, runtime_batch, static_batch):
        self.staged.append(int(runtime_batch.input_ids.shape[0]))

    def discard_mtp_verify_graph(self, width):
        self.discarded.append(width)

    def reset_mtp_verify_graph(self):
        self.reset_count += 1


class _Inner:
    def forward(self, input_ids, batch):
        values = torch.stack(
            (
                input_ids.float(),
                batch.positions.float(),
                batch.out_loc.float(),
                batch.linear_table_idx[0].float().expand_as(input_ids),
            ),
            dim=-1,
        )
        return values


class _Head:
    def forward_all(self, hidden):
        return torch.cat((hidden, hidden[:, :2] + hidden[:, 2:4]), dim=-1)


class _Model:
    def __init__(self):
        self.model = _Inner()
        self.lm_head = _Head()
        self.capture_sizes = []
        self.replay_sizes = []

    def prepare_cuda_graph_capture(self, batch):
        self.capture_sizes.append(int(batch.input_ids.shape[0]))

    def prepare_cuda_graph_replay(self, batch):
        self.replay_sizes.append(int(batch.input_ids.shape[0]))


def _batch(width: int, offset: int, device: torch.device) -> Batch:
    cached_len = 5
    req = Req(
        input_ids=torch.arange(
            offset, offset + cached_len + width, dtype=torch.int32
        ),
        table_idx=20 + offset,
        cached_len=cached_len,
        output_len=0,
        uid=offset,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    req.linear_slot_idx = 300 + offset
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.arange(
        offset, offset + width, dtype=torch.int32, device=device
    )
    batch.positions = torch.arange(
        100 + offset, 100 + offset + width, dtype=torch.int32, device=device
    )
    batch.out_loc = torch.arange(
        200 + offset, 200 + offset + width, dtype=torch.int32, device=device
    )
    batch.rope_positions = batch.positions.to(torch.int64).expand(3, -1).clone()
    batch.linear_table_idx = torch.tensor(
        [req.linear_slot_idx], dtype=torch.int32, device=device
    )
    batch.attn_metadata = None
    batch.mtp_verify = True
    return batch


def _expected(batch):
    hidden = _Inner().forward(batch.input_ids, batch)
    return _Head().forward_all(hidden).float()


@pytest.mark.parametrize("width", (2, 3, 4))
def test_graph_runner_accepts_one_request_with_two_to_four_verify_tokens(width):
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=_Model(),
        attn_backend=_Attention(),
        moe_cache=None,
        device=torch.device("cpu"),
        vocab_size=6,
        guard_bytes=0,
    )
    batch = _batch(width, 1, torch.device("cpu"))

    assert batch.size == batch.padded_size == 1
    assert batch.reqs[0].extend_len == width
    support = runner.capture(batch)

    assert support.width == width
    assert support.status == "permanently-unsupported"
    assert support.reason == "CUDA_REQUIRED"


@pytest.mark.parametrize("width", (2, 3, 4))
def test_graph_buffer_replay_accepts_one_request_with_two_to_four_verify_tokens(width):
    batch = _batch(width, 7, torch.device("cpu"))
    buffer = _MTPVerifyGraphBuffer.init(width, 6, torch.device("cpu"))

    buffer.copy_from(batch)

    assert torch.equal(buffer.input_ids, batch.input_ids)
    assert buffer.linear_table_idx.tolist() == [batch.reqs[0].linear_slot_idx]
    assert buffer.fla_cu_seqlens.tolist() == [0, width]


def test_graph_runner_rejects_invalid_private_verify_shapes():
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=_Model(),
        attn_backend=_Attention(),
        moe_cache=None,
        device=torch.device("cpu"),
        vocab_size=6,
        guard_bytes=0,
    )

    wrong_width = _batch(2, 1, torch.device("cpu"))
    wrong_width.input_ids = wrong_width.input_ids[:1]
    with pytest.raises(ValueError, match="2, 3, or 4 token rows"):
        runner.capture(wrong_width)

    too_wide = _batch(4, 1, torch.device("cpu"))
    too_wide.input_ids = torch.cat((too_wide.input_ids, too_wide.input_ids[:1]))
    with pytest.raises(ValueError, match="2, 3, or 4 token rows"):
        runner.capture(too_wide)

    decode = _batch(2, 2, torch.device("cpu"))
    decode.phase = "decode"
    with pytest.raises(ValueError, match="one prefill request"):
        runner.capture(decode)

    multiple = _batch(2, 3, torch.device("cpu"))
    multiple.reqs.append(_batch(2, 4, torch.device("cpu")).reqs[0])
    multiple.padded_reqs = multiple.reqs
    with pytest.raises(ValueError, match="one prefill request"):
        runner.capture(multiple)

    unmarked = _batch(2, 5, torch.device("cpu"))
    unmarked.mtp_verify = False
    with pytest.raises(ValueError, match="private verification marker"):
        runner.capture(unmarked)


def test_graph_buffer_rejects_replay_token_width_mismatch():
    buffer = _MTPVerifyGraphBuffer.init(2, 6, torch.device("cpu"))
    with pytest.raises(ValueError, match="replay width does not match"):
        buffer.copy_from(_batch(3, 7, torch.device("cpu")))


def test_graph_runner_rejects_non_cuda_without_allocating():
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=_Model(),
        attn_backend=_Attention(),
        moe_cache=None,
        device=torch.device("cpu"),
        vocab_size=6,
        guard_bytes=0,
    )

    support = runner.capture(_batch(2, 1, torch.device("cpu")))

    assert support.status == "permanently-unsupported"
    assert support.reason == "CUDA_REQUIRED"
    assert runner.support(2) is support
    assert runner.last_attempt(2) is support
    assert runner.graph_count == 0
    assert runner.owned_buffer_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_graphs_stage_runtime_values_replay_exactly_and_destroy():
    device = torch.device("cuda")
    attention = _Attention()
    model = _Model()
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=model,
        attn_backend=attention,
        moe_cache=None,
        device=device,
        vocab_size=6,
        guard_bytes=0,
    )

    for width in (2, 3, 4):
        support = runner.capture(_batch(width, width, device))
        assert support.status == "captured"
        assert support.width == width

    owned_after_capture = runner.owned_buffer_bytes
    live_memory_after_capture = runner.live_graph_memory_bytes
    assert runner.graph_count == 3
    assert owned_after_capture > 0
    assert live_memory_after_capture >= owned_after_capture
    assert attention.prepared == [2, 3, 4]

    hashes = []
    for width in (2, 3, 4):
        batch = _batch(width, 20 + width, device)
        for repeat in range(3):
            result = runner.replay(batch)
            assert result.mode == "fast-graph"
            assert result.required_synchronizations == 1
            assert result.instrumentation_synchronizations == 1
            assert result.synchronizations == 2
            assert torch.equal(result.logits, _expected(batch))
            hashes.append((repeat, width, tensor_sha256(result.logits)))
        changed = _batch(width, 70 + width, device)
        changed_result = runner.replay(changed)
        assert torch.equal(changed_result.logits, _expected(changed))
        assert tensor_sha256(changed_result.logits) != hashes[-1][2]

    for width in (2, 3, 4):
        assert len({value for _, got_width, value in hashes if got_width == width}) == 1
    assert runner.owned_buffer_bytes == owned_after_capture
    assert runner.live_graph_memory_bytes == live_memory_after_capture
    assert attention.staged == [2] * 4 + [3] * 4 + [4] * 4
    assert model.capture_sizes == [2, 3, 4]
    assert model.replay_sizes == [2] * 4 + [3] * 4 + [4] * 4

    runner.destroy()
    assert runner.graph_count == 0
    assert runner.owned_buffer_bytes == 0
    assert runner.live_graph_memory_bytes == 0
    assert runner.support(2) is None
    assert attention.reset_count == 1
    with pytest.raises(RuntimeError, match="destroyed"):
        runner.capture(_batch(2, 1, device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_replay_returns_owned_logits():
    device = torch.device("cuda")
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=_Model(),
        attn_backend=_Attention(),
        moe_cache=None,
        device=device,
        vocab_size=6,
        guard_bytes=0,
    )
    assert runner.capture(_batch(2, 1, device)).status == "captured"

    first = runner.replay(_batch(2, 20, device))
    first_snapshot = first.logits.clone()
    first_hash = tensor_sha256(first.logits)
    second = runner.replay(_batch(2, 70, device))

    assert torch.equal(first.logits, first_snapshot)
    assert tensor_sha256(first.logits) == first_hash
    assert first.logits.data_ptr() != runner._buffers[2].logits.data_ptr()
    assert first.logits.data_ptr() != second.logits.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_memory_admission_is_retryable_and_later_capture_can_succeed(monkeypatch):
    device = torch.device("cuda")
    original_mem_get_info = torch.cuda.mem_get_info
    admitted = False

    def bounded_mem_get_info(got_device):
        if not admitted:
            return (0, original_mem_get_info(got_device)[1])
        return original_mem_get_info(got_device)

    monkeypatch.setattr(torch.cuda, "mem_get_info", bounded_mem_get_info)
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=_Model(),
        attn_backend=_Attention(),
        moe_cache=None,
        device=device,
        vocab_size=6,
        guard_bytes=0,
    )

    first = runner.capture(_batch(2, 1, device))
    assert first.status == "retryable"
    assert first.reason == "MEMORY_ADMISSION"
    assert runner.support(2) is None
    assert runner.last_attempt(2) is first
    assert runner.graph_count == runner.live_graph_memory_bytes == 0

    admitted = True
    second = runner.capture(_batch(2, 1, device))
    assert second.status == "captured"
    assert runner.support(2) is second
    assert runner.replay(_batch(2, 20, device)).logits.shape == (2, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_capture_exception_discards_qsa_and_can_retry():
    class FailOnceInner(_Inner):
        fail = True

        def forward(self, input_ids, batch):
            if self.fail:
                raise RuntimeError("synthetic transient capture failure")
            return super().forward(input_ids, batch)

    device = torch.device("cuda")
    attention = _Attention()
    model = _Model()
    model.model = FailOnceInner()
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=model,
        attn_backend=attention,
        moe_cache=None,
        device=device,
        vocab_size=6,
        guard_bytes=0,
    )

    first = runner.capture(_batch(2, 1, device))
    assert first.status == "retryable"
    assert first.reason.startswith("CAPTURE_FAILED:RuntimeError:")
    assert "synthetic transient capture failure" in first.reason
    assert runner.support(2) is None
    assert runner.last_attempt(2) is first
    assert attention.discarded == [2]
    assert runner.graph_count == 0
    assert runner.live_graph_memory_bytes == 0
    assert 2 not in runner._buffers
    assert 2 not in runner._batches
    assert 2 not in runner._events

    model.model.fail = False
    second = runner.capture(_batch(2, 1, device))
    assert second.status == "captured"
    assert runner.support(2) is second
    assert torch.equal(
        runner.replay(_batch(2, 20, device)).logits,
        _expected(_batch(2, 20, device)),
    )


class _CaptureBodyError(RuntimeError):
    """Marker raised from inside the capture region so the recorded reason is unambiguous."""


class _PinDuringCaptureInner(_Inner):
    """Succeeds on the warm-up pass, then does the capture-illegal pinned-host allocation."""

    def __init__(self):
        self.calls = 0

    def forward(self, input_ids, batch):
        self.calls += 1
        if self.calls > 1:
            try:
                torch.empty(4).pin_memory()
            except Exception:
                pass
            raise _CaptureBodyError("pinned host allocation inside the capture region")
        return super().forward(input_ids, batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_in_capture_failure_is_survivable_and_names_the_body_exception():
    device = torch.device("cuda")
    model = _Model()
    model.model = _PinDuringCaptureInner()
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=model,
        attn_backend=_Attention(),
        moe_cache=None,
        device=device,
        vocab_size=6,
        guard_bytes=0,
    )
    before_stream = torch.cuda.current_stream(device)

    support = runner.capture(_batch(2, 1, device))

    assert support.status == "permanently-unsupported"
    assert support.reason.startswith("CAPTURE_FAILED:_CaptureBodyError:")
    assert "pinned host allocation inside the capture region" in support.reason
    assert model.model.calls == 2
    assert runner.graph_count == 0
    assert torch.cuda.current_stream(device) == before_stream
    torch.zeros(4, device=device)
    torch.cuda.synchronize(device)

    if runner._graphs_disabled:
        assert runner.support(3).status == "permanently-unsupported"
        assert runner.support(4).status == "permanently-unsupported"
    else:
        clean = MTPVerifyGraphRunner(
            target_ctx=_Context(),
            target_model=_Model(),
            attn_backend=_Attention(),
            moe_cache=None,
            device=device,
            vocab_size=6,
            guard_bytes=0,
        )
        assert clean.capture(_batch(2, 1, device)).status == "captured"
        assert torch.equal(
            clean.replay(_batch(2, 20, device)).logits, _expected(_batch(2, 20, device))
        )
        clean.destroy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_in_capture_failure_is_never_retried():
    device = torch.device("cuda")
    model = _Model()
    model.model = _PinDuringCaptureInner()
    attention = _Attention()
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=model,
        attn_backend=attention,
        moe_cache=None,
        device=device,
        vocab_size=6,
        guard_bytes=0,
    )

    first = runner.capture(_batch(2, 1, device))
    second = runner.capture(_batch(2, 1, device))

    assert first.status == "permanently-unsupported"
    assert second is first
    assert model.model.calls == 2
    assert attention.discarded == [2]
    assert 2 not in runner._buffers
    assert 2 not in runner._batches
    assert 2 not in runner._events


def test_capture_forward_without_capture_allocates_no_device_logits():
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=_Model(),
        attn_backend=_Attention(),
        moe_cache=None,
        device=torch.device("cpu"),
        vocab_size=6,
        guard_bytes=0,
    )

    result = runner.capture_forward(_batch(3, 1, torch.device("cpu")))

    assert result.mode == "graph-capture"
    assert runner.support(3).status == "permanently-unsupported"
    assert result.logits.shape == (3, 0)
    assert result.logits.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_capture_forward_returns_device_logits_only_when_captured():
    device = torch.device("cuda")
    model = _Model()
    model.model = _PinDuringCaptureInner()
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=model,
        attn_backend=_Attention(),
        moe_cache=None,
        device=device,
        vocab_size=6,
        guard_bytes=0,
    )

    failed = runner.capture_forward(_batch(2, 1, device))

    assert failed.logits.device.type == "cpu"
    assert failed.logits.shape == (2, 0)
    torch.zeros(4, device=device)
    torch.cuda.synchronize(device)

    clean = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=_Model(),
        attn_backend=_Attention(),
        moe_cache=None,
        device=device,
        vocab_size=6,
        guard_bytes=0,
    )
    captured = clean.capture_forward(_batch(2, 1, device))
    assert captured.logits.device.type == "cuda"
    assert captured.logits.shape == (2, 6)
    clean.destroy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_raising_capture_end_neither_masks_the_body_error_nor_leaks_the_stream(monkeypatch):
    device = torch.device("cuda")
    original_end = torch.cuda.CUDAGraph.capture_end

    def poisoned_capture_end(self):
        original_end(self)
        raise RuntimeError("CUDA error: operation failed due to a previous error during capture")

    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_end", poisoned_capture_end)
    model = _Model()
    model.model = _PinDuringCaptureInner()
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=model,
        attn_backend=_Attention(),
        moe_cache=None,
        device=device,
        vocab_size=6,
        guard_bytes=0,
    )
    before_stream = torch.cuda.current_stream(device)

    support = runner.capture(_batch(2, 1, device))

    assert support.status == "permanently-unsupported"
    assert support.reason.startswith("CAPTURE_FAILED:_CaptureBodyError:")
    assert torch.cuda.current_stream(device) == before_stream
    torch.zeros(4, device=device)
    torch.cuda.synchronize(device)


def test_graph_runner_rejects_rows_that_disagree_with_extend_len():
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=_Model(),
        attn_backend=_Attention(),
        moe_cache=None,
        device=torch.device("cpu"),
        vocab_size=6,
        guard_bytes=0,
    )
    batch = _batch(3, 1, torch.device("cpu"))
    # the overlap-skewed capture reconstructs a request one token short of its verify rows
    batch.reqs[0].cached_len += 1
    assert batch.reqs[0].extend_len == 2

    with pytest.raises(ValueError, match="extend_len"):
        runner.capture(batch)
    with pytest.raises(ValueError, match="extend_len"):
        runner.replay(batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_capture_pins_the_fla_chunk_index_tensors_for_every_width():
    from freetoken.kernel.fla.chunk import CHUNK_SIZE
    from freetoken.kernel.fla.index import prepare_chunk_indices

    device = torch.device("cuda")
    runner = MTPVerifyGraphRunner(
        target_ctx=_Context(),
        target_model=_Model(),
        attn_backend=_Attention(),
        moe_cache=None,
        device=device,
        vocab_size=6,
        guard_bytes=0,
    )
    for width in (2, 3, 4):
        assert runner.capture(_batch(width, width, device)).status == "captured"

    # six entries across a four-slot identity LRU: without its own reference the runner would
    # let the tensors its graphs read be freed out from under them
    for width in (2, 3, 4):
        pinned = runner._fla_index_pins[width]
        cu_seqlens = runner._buffers[width].fla_cu_seqlens
        assert len(pinned) == 3
        assert torch.equal(pinned[0], prepare_chunk_indices(cu_seqlens, CHUNK_SIZE))

    runner.destroy()
    assert runner._fla_index_pins == {}

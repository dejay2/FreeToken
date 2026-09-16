from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.llm.llm import LLM, RequestAllFinished
from freetoken.message import MMItem


def _offline_stub():
    llm = LLM.__new__(LLM)
    llm.prefill_budget = 128
    llm.tokenizer = SimpleNamespace(decode=lambda ids: "")
    received = []

    def run():
        received.extend(llm.offline_receive_msg())
        raise RequestAllFinished()

    llm.run_forever = run
    return llm, received


def test_legacy_positional_mm_inputs_keep_precomputed_embeddings():
    llm, received = _offline_stub()
    features = torch.ones(2, 4)
    calls = []
    llm.encode_images = lambda pixels, positions: calls.append((pixels, positions)) or features
    pixels, positions = torch.zeros(1, 2, 3), torch.zeros(1, 2, 2)
    result = llm.generate([[1, 5, 5]], SamplingParams(), [{"pixel_values": pixels, "image_position_ids": positions}])
    assert result == [{"text": "", "token_ids": []}]
    assert received[0].mm_embeds is features
    assert received[0].mm_items is None
    assert calls[0][0] is pixels and calls[0][1] is positions


def test_raw_images_use_processor_items():
    llm, received = _offline_stub()
    item = MMItem("image", 1, 1000001, [[1, 3]], feature=torch.ones(2, 4))
    llm._mm_processor = SimpleNamespace(apply=lambda ids, images: SimpleNamespace(
        input_ids=torch.tensor([1, 1000001, 1000001], dtype=torch.int32),
        mm_items=[item], mrope_positions=torch.zeros(3, 3, dtype=torch.int32), mrope_delta=-1,
    ))
    llm.generate([[1, 5, 5]], SamplingParams(), images=[[b"image"]])
    assert received[0].mm_items == [item]
    assert received[0].mm_embeds is None
    assert received[0].mrope_delta == -1


def test_raw_and_preprocessed_images_cannot_be_combined():
    llm, _ = _offline_stub()
    with pytest.raises(ValueError, match="either images or mm_inputs"):
        llm.generate([[1]], SamplingParams(), [None], images=[[b"image"]])

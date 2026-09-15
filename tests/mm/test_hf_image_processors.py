"""Exercise actual HF image processor APIs without checkpoints or model weights."""
from __future__ import annotations

import io

import pytest
import torch

from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processors.gemma4 import Gemma4MMProcessor, Gemma4UnifiedMMProcessor
from freetoken.mm.processors.glm5_next import Glm5NextMMProcessor
from freetoken.mm.processors.minimax_m3 import MiniMaxM3MMProcessor
from freetoken.mm.processors.muse_glimmer import MuseGlimmerMMProcessor
from freetoken.mm.processors.qwen_vl import QwenVLMMProcessor


@pytest.mark.parametrize("max_tokens", [None, 70])
@pytest.mark.parametrize("config_name,processor_name,wrapper", [
    ("Qwen3VLConfig", "Qwen2VLImageProcessor", QwenVLMMProcessor),
    ("Gemma4Config", "Gemma4ImageProcessor", Gemma4MMProcessor),
    ("Gemma4UnifiedConfig", "Gemma4UnifiedImageProcessor", Gemma4UnifiedMMProcessor),
    ("Glm5NextConfig", "Glm5NextImageProcessor", Glm5NextMMProcessor),
    ("MuseGlimmerConfig", "MuseGlimmerImageProcessor", MuseGlimmerMMProcessor),
    ("MiniMaxM3VLConfig", "MiniMaxM3VLImageProcessor", MiniMaxM3MMProcessor),
])
def test_actual_hf_image_processing(tmp_path, config_name, processor_name, wrapper, max_tokens):
    pytest.importorskip("torchvision")
    transformers = pytest.importorskip("transformers")
    Image = pytest.importorskip("PIL.Image")
    config_cls = getattr(transformers, config_name)
    # Gemma's top-level config defaults to a text-only checkpoint.
    config = config_cls(vision_config={}) if config_name.startswith("Gemma") else config_cls()
    real = getattr(transformers, processor_name)()
    vc = config.vision_config
    # Real checkpoints save processor dimensions alongside the model. Reproduce
    # that contract: Qwen2VL's processor defaults differ from Qwen3VL's config.
    real.patch_size = vc.patch_size
    for source, target in (
        ("temporal_patch_size", "temporal_patch_size"),
        ("spatial_merge_size", "merge_size"),
        ("merge_size", "merge_size"),
        ("patch_temporal", "temporal_patch_size"),
    ):
        if hasattr(vc, source):
            setattr(real, target, getattr(vc, source))
    real.save_pretrained(tmp_path)
    processor = wrapper(config, str(tmp_path), MultimodalConfig(image_max_tokens=max_tokens))
    buf = io.BytesIO()
    with Image.new("RGB", (112, 84), (60, 20, 200)) as image:
        image.save(buf, format="PNG")
    ids = torch.tensor([1, *processor.placeholder, 2], dtype=torch.int32)
    result = processor.apply(ids, [buf.getvalue()])
    (item,) = result.mm_items
    item.validate()
    assert item.feature.device.type == "cpu"
    assert item.feature.ndim == 2 and item.feature.shape[-1] == processor.patch_dim
    assert item.num_tokens > 0
    if max_tokens is not None:
        assert item.num_tokens <= max_tokens
    assert int((result.input_ids == item.pad_value).sum()) == item.num_tokens
    assert result.input_ids[0] == 1 and result.input_ids[-1] == 2
    repeated = processor.apply(ids, [buf.getvalue()])
    assert repeated.mm_items[0].hash == item.hash
    assert torch.equal(result.input_ids, repeated.input_ids)

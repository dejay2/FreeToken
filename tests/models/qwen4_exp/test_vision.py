from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen4_exp.config import Qwen4VisionConfig
from freetoken.models.qwen4_exp.model import Qwen4ExpModel
from freetoken.models.qwen4_exp.vision import Qwen4VisionModel


def _config() -> Qwen4VisionConfig:
    return Qwen4VisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        num_position_embeddings=16,
        out_hidden_size=24,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=(),
    )


def test_qwen_picture_reader_matches_transformers_qwen3_vl_reference():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    torch.manual_seed(9)
    ours_config = _config()
    reference_config = Qwen3VLVisionConfig(
        depth=ours_config.depth,
        hidden_size=ours_config.hidden_size,
        intermediate_size=ours_config.intermediate_size,
        num_heads=ours_config.num_heads,
        num_position_embeddings=ours_config.num_position_embeddings,
        out_hidden_size=ours_config.out_hidden_size,
        patch_size=ours_config.patch_size,
        spatial_merge_size=ours_config.spatial_merge_size,
        temporal_patch_size=ours_config.temporal_patch_size,
        in_channels=ours_config.in_channels,
        hidden_act=ours_config.hidden_act,
        deepstack_visual_indexes=[],
        _attn_implementation="sdpa",
    )
    reference = Qwen3VLVisionModel(reference_config).eval().cpu()
    with torch.device("cpu"):
        ours = Qwen4VisionModel(ours_config)
    ours.load_state_dict(dict(reference.state_dict()))

    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixels = torch.randn(
        16,
        ours_config.in_channels
        * ours_config.temporal_patch_size
        * ours_config.patch_size
        * ours_config.patch_size,
    )
    with torch.inference_mode():
        expected = reference(pixels, grid).pooler_output
        actual = ours.forward(pixels, grid)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_qwen_picture_reader_derived_rope_survives_meta_construction():
    torch.manual_seed(17)
    config = _config()
    with torch.device("cpu"):
        reference = Qwen4VisionModel(config)
    with torch.no_grad():
        for tensor in reference.state_dict().values():
            if tensor.is_floating_point():
                tensor.uniform_(-0.02, 0.02)
            else:
                tensor.zero_()
    with torch.device("meta"):
        model = Qwen4VisionModel(config)
    with torch.device("cpu"):
        model.load_state_dict(dict(reference.state_dict()))

    assert not hasattr(model, "_inv_freq")
    assert model._inv_dim == config.hidden_size // config.num_heads // 2

    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixels = torch.randn(
        16,
        config.in_channels
        * config.temporal_patch_size
        * config.patch_size
        * config.patch_size,
    )
    with torch.inference_mode():
        expected = reference.forward(pixels, grid)
        actual = model.forward(pixels, grid)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _model_shell(image_token_id: int = 99) -> Qwen4ExpModel:
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    model._image_token_id = image_token_id
    return model


def test_picture_features_replace_exact_placeholder_rows_before_hc_expansion():
    model = _model_shell()
    input_ids = torch.tensor([5, 99, 6, 99], dtype=torch.int32)
    hidden = torch.arange(16, dtype=torch.float32).view(4, 4)
    features = torch.tensor([[101, 102, 103, 104], [201, 202, 203, 204]], dtype=torch.float32)

    actual = model._merge_multimodal(input_ids, hidden, features)

    assert torch.equal(actual[0], hidden[0])
    assert torch.equal(actual[1], features[0])
    assert torch.equal(actual[2], hidden[2])
    assert torch.equal(actual[3], features[1])


@pytest.mark.parametrize("feature_rows", [0, 1, 3])
def test_picture_feature_count_must_equal_placeholder_count(feature_rows: int):
    model = _model_shell()
    input_ids = torch.tensor([99, 1, 99], dtype=torch.int32)
    hidden = torch.zeros(3, 4)
    features = torch.zeros(feature_rows, 4)

    with pytest.raises(ValueError, match="picture-token slots.*picture features"):
        model._merge_multimodal(input_ids, hidden, features)

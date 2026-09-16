"""CPU startup regressions using real Gemma parsers and checkpoint readers."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models.config import SWAAttentionGroupConfig
from freetoken.models.gemma4.config import UnifiedVisionConfig, VisionConfig, parse_config


def _hf_config(unified=False, text_only=False, bidirectional=False):
    text = SimpleNamespace(
        model_type='gemma4_unified_text' if unified else 'gemma4_text',
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, hidden_size=16, vocab_size=32, intermediate_size=32,
        rms_norm_eps=1e-6, max_position_embeddings=128, sliding_window=32,
        layer_types=['sliding_attention', 'full_attention'],
        rope_parameters={
            'sliding_attention': {'rope_theta': 10000},
            'full_attention': {'rope_theta': 10000},
        },
        use_bidirectional_attention='vision' if bidirectional else None,
    )
    vision = (SimpleNamespace(model_patch_size=4, mm_embed_dim=16,
                             mm_posemb_size=8, rms_norm_eps=1e-6)
              if unified else SimpleNamespace(
                  hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                  intermediate_size=32, patch_size=4, position_embedding_size=8,
                  pooling_kernel_size=2, rms_norm_eps=1e-6, standardize=True))
    return SimpleNamespace(
        text_config=text, vision_config=None if text_only else vision,
        architectures=['Gemma4UnifiedForConditionalGeneration' if unified
                       else 'Gemma4ForConditionalGeneration'],
        image_token_id=31,
    )


@pytest.mark.parametrize('unified', [False, True])
@pytest.mark.parametrize('text_only', [False, True])
@pytest.mark.parametrize('bidirectional', [False, True])
def test_gemma_config_parses_real_attention_groups(unified, text_only, bidirectional):
    config = parse_config(_hf_config(unified, text_only, bidirectional))
    swa = config.attention_group_for_layer(0)
    assert isinstance(swa, SWAAttentionGroupConfig)
    assert swa.bidirectional_mm_blocks is bidirectional
    assert swa.sliding_window == 32
    assert config.is_multimodal is (not text_only)
    if not text_only:
        assert isinstance(config.vision_config, UnifiedVisionConfig if unified else VisionConfig)


def test_swa_existing_callers_default_to_causal():
    group = SWAAttentionGroupConfig(name='swa', layer_ids=(0,), num_kv_heads=1,
                                   head_dim=8, rotary_config=None, sliding_window=32)
    assert group.bidirectional_mm_blocks is False


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_standardized_gemma_vision_weights_load_from_safetensors(tmp_path, monkeypatch, dtype):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.gemma4 import weight

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    # Replace only config retrieval; parsing, file IO, key conversion and validation
    # all use the production path.
    monkeypatch.setattr(weight, 'cached_load_hf_config', lambda _: _hf_config())
    tensors = {
        'model.vision_tower.std_scale': torch.full((16,), 0.5, dtype=dtype),
        'model.vision_tower.std_bias': torch.ones(16, dtype=dtype),
        'model.vision_tower.patch_embedder.input_proj.weight': torch.ones(16, 48, dtype=dtype),
    }
    save_file(tensors, str(tmp_path / 'model.safetensors'))
    loaded = dict(weight.iter_weights(str(tmp_path), torch.device('cpu'),
                  include_moe_experts=False, include_non_moe=True, include_vision=True))
    assert set(loaded) == {name.removeprefix('model.') for name in tensors}
    for name, tensor in tensors.items():
        actual = loaded[name.removeprefix('model.')]
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, tensor)


@pytest.mark.parametrize('leaf,dtype', [
    ('std_scale', torch.uint8),
    ('patch_embedder.input_proj.weight', torch.uint8),
    ('patch_embedder.input_proj.weight', torch.float8_e4m3fn),
    ('patch_embedder.input_proj.weight_scale', torch.float32),
    ('patch_embedder.input_proj.weight_scale_inv', torch.float32),
    ('patch_embedder.input_proj.qweight', torch.int32),
])
def test_gemma_loader_still_rejects_encoded_vision(tmp_path, monkeypatch, leaf, dtype):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.gemma4 import weight

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(weight, 'cached_load_hf_config', lambda _: _hf_config())
    save_file({'model.vision_tower.' + leaf: torch.zeros(16, dtype=dtype)},
              str(tmp_path / 'model.safetensors'))
    with pytest.raises(NotImplementedError, match='Quantized vision weight'):
        list(weight.iter_weights(str(tmp_path), torch.device('cpu'),
             include_moe_experts=False, include_non_moe=True, include_vision=True))

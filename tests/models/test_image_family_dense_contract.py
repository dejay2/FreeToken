"""CPU checks for image tower construction and encoded-weight rejection."""
import importlib.util
from pathlib import Path

import pytest
import torch
from freetoken.models.vision_weight import require_dense_vision_weight


@pytest.mark.parametrize('family,cls', [('glm5_next', 'Glm5NextVisionModel'), ('minimax_m3', 'MiniMaxM3VisionModel'), ('muse_glimmer', 'MuseGlimmerVisionModel')])
def test_image_tower_constructs_with_legacy_dense_layers(family, cls):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    spec = importlib.util.spec_from_file_location('vision_fixture', Path(__file__).with_name(f'test_{family}_vision.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    config = fixture._tiny_vc()
    model = getattr(fixture, cls)(config)
    state = model.state_dict()
    assert state and all(t.dtype == torch.float32 for t in state.values())
    if family == 'glm5_next':
        assert model.blocks.op_list[0].mlp.swiglu_limit == config.swiglu_limit
        assert state['blocks.0.mlp.gate_proj.bias'].shape == (config.intermediate_size,)
        assert 'merger.gate_proj.bias' not in state


@pytest.mark.parametrize('name,dtype', [('visual.x.weight', torch.uint8), ('visual.x.weight', torch.float8_e4m3fn), ('visual.x.weight_scale', torch.float32), ('visual.x.trellis', torch.int16), ('visual.x.qweight', torch.int32)])
def test_quantized_vision_is_explicitly_rejected(name, dtype):
    with pytest.raises(NotImplementedError, match='Quantized vision weight'):
        require_dense_vision_weight(name, torch.empty(1, dtype=dtype))


def test_dense_vision_validation_preserves_tensor():
    tensor = torch.randn(2, 3, dtype=torch.bfloat16)
    assert require_dense_vision_weight('visual.x.weight', tensor) is tensor


@pytest.mark.parametrize('family,prefix', [('glm5_next', 'model.visual.'), ('minimax_m3', 'multi_modal_projector.')])
@pytest.mark.parametrize('leaf,dtype', [('weight', torch.float8_e4m3fn), ('weight_scale', torch.float32)])
def test_family_reader_rejects_encoded_vision_before_cast(family, prefix, leaf, dtype):
    import importlib
    from types import SimpleNamespace
    module = importlib.import_module(f'freetoken.models.{family}.weight')
    name = prefix + 'linear.' + leaf
    reader = SimpleNamespace(get=lambda key: torch.empty(2, 2, dtype=dtype))
    args = (reader, {name: 'shard'}) + ((0,) if family == 'minimax_m3' else ())
    with pytest.raises(NotImplementedError, match='Quantized vision weight'):
        list(module._iter_vision(*args))


def test_gemma_unified_embedder_cpu_shape_and_finiteness():
    from freetoken.models.gemma4.config import UnifiedVisionConfig
    from freetoken.models.gemma4.vision import Gemma4UnifiedVisionEmbedder
    config = UnifiedVisionConfig(hidden_size=16, patch_dim=48, posemb_size=8, layer_norm_eps=1e-5, rms_norm_eps=1e-6, text_hidden_size=32)
    tower = Gemma4UnifiedVisionEmbedder(config)
    for tensor in tower.state_dict().values():
        tensor.normal_(0, 0.02)
    pixels = torch.rand(1, 6, 48)
    positions = torch.tensor([[[x, y] for y in range(2) for x in range(3)]])
    output = tower.forward(pixels, positions)
    assert output.shape == (1, 6, 16)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize('unified', [False, True])
def test_gemma_legacy_preprocessed_image_adapter(unified):
    from types import SimpleNamespace
    from freetoken.models.gemma4.model import Gemma4ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration
    pixels = torch.randn(2, 3, 4)
    positions = torch.zeros(2, 3, 2, dtype=torch.long)
    calls = []
    def encode(feature, coordinates):
        assert feature is pixels and coordinates is positions
        calls.append(True)
        return feature if unified else feature.reshape(-1, 4)
    model = SimpleNamespace(embed_vision=SimpleNamespace(forward=lambda value: value * 2))
    setattr(model, 'vision_embedder' if unified else 'vision_tower', SimpleNamespace(forward=encode))
    cls = Gemma4UnifiedForConditionalGeneration if unified else Gemma4ForConditionalGeneration
    output = cls.encode_images(model, pixels, positions)
    assert calls == [True]
    torch.testing.assert_close(output, pixels.reshape(-1, 4) * 2)

"""EXL3 picture tower (spec 2026-09-25 section 5)."""

from __future__ import annotations

import torch
import pytest

from freetoken.kernel.exl3_linear import Exl3Linear
from freetoken.models.qwen4_exp import weight as W
from freetoken.models.vision_weight import require_dense_vision_weight


def test_exl3_vision_skips_packed_qkv_parts():
    for part in ("q_proj", "k_proj", "v_proj"):
        for comp in ("trellis", "suh", "svh", "mul1", "bias"):
            name = f"model.visual.blocks.0.attn.{part}.{comp}"
            assert W._rename(name, include_vision=True, exl3=True) is None


def test_exl3_vision_keeps_bf16_qkv_and_packed_proj():
    assert W._rename("model.visual.blocks.0.attn.qkv.weight", include_vision=True, exl3=True) == \
        "visual.blocks.0.attn.qkv.weight"
    assert W._rename("model.visual.blocks.0.attn.proj.trellis", include_vision=True, exl3=True) == \
        "visual.blocks.0.attn.proj.trellis"


def test_vision_weight_check_allows_named_exl3_linears():
    t = torch.zeros(4, dtype=torch.int16)
    require_dense_vision_weight("visual.blocks.0.mlp.linear_fc1.trellis", t, exl3=True)
    with pytest.raises(NotImplementedError):
        require_dense_vision_weight("visual.patch_embed.proj.trellis", t, exl3=True)
    with pytest.raises(NotImplementedError):
        require_dense_vision_weight("visual.blocks.0.mlp.linear_fc1.trellis", t)  # not exl3


def test_exl3_vision_modules(exl3_vision_config):
    from freetoken.models.qwen4_exp.vision import Qwen4VisionBlock, Qwen4VisionPatchMerger
    block = Qwen4VisionBlock(exl3_vision_config)
    assert isinstance(block.attn.proj, Exl3Linear)
    assert isinstance(block.mlp.linear_fc1, Exl3Linear) and isinstance(block.mlp.linear_fc2, Exl3Linear)
    assert not isinstance(block.attn.qkv, Exl3Linear)
    merger = Qwen4VisionPatchMerger(exl3_vision_config)
    assert isinstance(merger.linear_fc1, Exl3Linear)


def test_exl3_vision_k_hint_matches_config(exl3_vision_config):
    """The workspace block's Exl3Linear shapes come from ``k_hint`` at construction time
    (not from a load), so a wrong config value would silently build the wrong trellis
    shape -- forward_layer_streamed relies on the constructor shape matching the checkpoint."""
    from freetoken.models.qwen4_exp.vision import Qwen4VisionBlock

    block = Qwen4VisionBlock(exl3_vision_config)
    assert block.attn.proj.k == exl3_vision_config.exl3_k == 5
    assert block.attn.proj.trellis.shape[-1] == 16 * exl3_vision_config.exl3_k


# --------------------------------------------------------------------------------------
# CPU-runnable companion to the box's CUDA layer-stream copy test (Step 4): the copy
# helper (_copy_component_state_) only touches tensors' shape/dtype/address, never the
# device, so this exercises the same EXL3 path without a GPU.
# --------------------------------------------------------------------------------------


def _randomize_exl3_block_(block) -> None:
    with torch.no_grad():
        for tensor in block.state_dict().values():
            if tensor.dtype == torch.int16:
                tensor.random_(-32768, 32767)
            elif tensor.is_floating_point():
                tensor.uniform_(-0.02, 0.02)
            else:
                tensor.zero_()


def test_exl3_layer_stream_block_copy(exl3_vision_config):
    from freetoken.models.qwen4_exp.vision import Qwen4VisionBlock, _copy_component_state_

    torch.manual_seed(5)
    with torch.device("cpu"):
        source = Qwen4VisionBlock(exl3_vision_config)  # stands in for the CPU-held tower
        target = Qwen4VisionBlock(exl3_vision_config)  # stands in for the GPU workspace block
    _randomize_exl3_block_(source)
    addresses = {key: value.data_ptr() for key, value in target.state_dict().items()}

    _copy_component_state_(target, source)

    # in place: the workspace's own storage never moved
    assert {key: value.data_ptr() for key, value in target.state_dict().items()} == addresses
    # every packed component landed unchanged, including attn.proj.trellis specifically
    source_state = source.state_dict()
    for key, value in target.state_dict().items():
        assert torch.equal(value, source_state[key]), key
    assert torch.equal(target.attn.proj.trellis, source.attn.proj.trellis)


def test_exl3_vision_mlp_pads_intermediate_to_128(exl3_vision_config):
    """Qwen3.8-Flash-Next's tower has intermediate 4304; the EXL3 checkpoint stores
    linear_fc1 as [1152 -> 4352] (bias 4352) and linear_fc2 as [4352 -> 1152], exllamav3's
    MLP pad_to=128 (found on the first box boot, 2026-09-25: 4304 % 128 != 0 refused)."""
    import dataclasses

    from freetoken.models.qwen4_exp.vision import Qwen4VisionMLP

    cfg = dataclasses.replace(exl3_vision_config, intermediate_size=200)
    mlp = Qwen4VisionMLP(cfg)
    assert mlp.linear_fc1.out_features == 256 and mlp.linear_fc2.in_features == 256
    assert mlp.linear_fc1.bias.shape == (256,)


def test_exl3_vision_mlp_loads_the_real_padded_checkpoint_shapes(exl3_vision_config):
    """Review of b3cfda1 against the 3.05bpw_h5_ng5 checkpoint facts: intermediate 4304 in the
    config, fc1 stored [1152 -> 4352] (trellis 72x272, bias 4352: exllamav3 Linear zero-pads
    the bias to the padded width, modules/linear.py pad1) and fc2 [4352 -> 1152] (trellis
    272x72, bias 1152; exllamav3 MLP builds down with allow_input_padding and
    trim_padded_out, so fc1's padded outputs feed fc2 untrimmed)."""
    import dataclasses

    from freetoken.models.qwen4_exp.vision import Qwen4VisionMLP

    cfg = dataclasses.replace(exl3_vision_config, hidden_size=1152, intermediate_size=4304)
    mlp = Qwen4VisionMLP(cfg)

    def packed(fin, fout, k=5):
        return {
            "trellis": torch.zeros(fin // 16, fout // 16, 16 * k, dtype=torch.int16),
            "suh": torch.zeros(fin, dtype=torch.float16),
            "svh": torch.zeros(fout, dtype=torch.float16),
            "mul1": torch.tensor(0, dtype=torch.int32),
            "bias": torch.zeros(fout, dtype=torch.float16),
        }

    state = {f"linear_fc1.{n}": t for n, t in packed(1152, 4352).items()}
    state.update({f"linear_fc2.{n}": t for n, t in packed(4352, 1152).items()})
    mlp.linear_fc1.load_state_dict(state, prefix="linear_fc1", _internal=True)
    mlp.linear_fc2.load_state_dict(state, prefix="linear_fc2", _internal=True)
    assert state == {}
    assert mlp.linear_fc1.trellis.shape == (72, 272, 80)
    assert mlp.linear_fc2.trellis.shape == (272, 72, 80)
    assert mlp.linear_fc1.bias.shape == (4352,) and mlp.linear_fc2.bias.shape == (1152,)


def test_bf16_vision_mlp_keeps_the_unpadded_intermediate_width():
    """The padding is EXL3-only: the NVFP4/bf16 tower keeps the config's 4304."""
    import dataclasses

    from freetoken.models.qwen4_exp.vision import Qwen4VisionMLP

    from .test_vision import _config

    cfg = dataclasses.replace(_config(), intermediate_size=4304)
    assert not getattr(cfg, "exl3", False)
    mlp = Qwen4VisionMLP(cfg)
    assert mlp.linear_fc1.weight.shape[0] == 4304
    assert mlp.linear_fc2.weight.shape[1] == 4304

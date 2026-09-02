from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.config import Qwen4VisionConfig
from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM, Qwen4ExpModel
from freetoken.models.qwen4_exp.vision import (
    Qwen4VisionBlock,
    Qwen4VisionModel,
    Qwen4VisionPatchEmbed,
    _copy_component_state_,
)
from freetoken.utils.torch_utils import torch_dtype


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


def test_stream_component_copy_is_in_place_and_validates_shape_and_dtype():
    config = _config()
    with torch.device("cpu"):
        source = Qwen4VisionPatchEmbed(config)
        target = Qwen4VisionPatchEmbed(config)
    with torch.no_grad():
        for tensor in source.state_dict().values():
            tensor.fill_(3.25)
    addresses = {key: value.data_ptr() for key, value in target.state_dict().items()}

    _copy_component_state_(target, source)

    assert {key: value.data_ptr() for key, value in target.state_dict().items()} == addresses
    assert all(torch.all(value == 3.25) for value in target.state_dict().values())

    target.proj.weight = target.proj.weight[:, :, :, :, :-1]
    with pytest.raises(ValueError, match="shape"):
        _copy_component_state_(target, source)

    with torch.device("cpu"), torch_dtype(torch.float64):
        wrong_dtype = Qwen4VisionPatchEmbed(config)
    with pytest.raises(ValueError, match="dtype"):
        _copy_component_state_(wrong_dtype, source)


def _stream_config(
    *, depth: int = 27, deepstack_visual_indexes: tuple[int, ...] = ()
) -> Qwen4VisionConfig:
    return Qwen4VisionConfig(
        depth=depth,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        num_position_embeddings=16,
        out_hidden_size=8,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=deepstack_visual_indexes,
    )


def _cpu_stream_source(config: Qwen4VisionConfig) -> Qwen4VisionModel:
    with torch.device("cpu"), torch_dtype(torch.bfloat16):
        model = Qwen4VisionModel(config)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.uniform_(-0.02, 0.02)
        for layer_id, block in enumerate(model.blocks.op_list):
            for tensor in block.state_dict().values():
                tensor.fill_((layer_id + 1) / 4096)
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_layer_stream_matches_gpu_reference_uses_all_blocks_once_and_cleans_up(monkeypatch):
    from freetoken.models.qwen4_exp import vision as vision_module

    torch.manual_seed(29)
    config = _stream_config()
    source = _cpu_stream_source(config)
    with torch.device("cuda"), torch_dtype(torch.bfloat16):
        reference = Qwen4VisionModel(config)
    reference.load_state_dict(
        {key: value.to("cuda") for key, value in source.state_dict().items()}
    )
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixels = torch.randn(16, 24, dtype=torch.bfloat16)
    source_layers = {id(block): layer_id for layer_id, block in enumerate(source.blocks.op_list)}
    copied_blocks = []
    empty_cache_calls = []
    original_copy = vision_module._copy_component_state_
    original_empty_cache = torch.cuda.empty_cache

    def record_copy(target, origin):
        if id(origin) in source_layers:
            copied_blocks.append((source_layers[id(origin)], id(target)))
        return original_copy(target, origin)

    def record_empty_cache():
        empty_cache_calls.append(True)
        original_empty_cache()

    monkeypatch.setattr(vision_module, "_copy_component_state_", record_copy)
    monkeypatch.setattr(torch.cuda, "empty_cache", record_empty_cache)
    with torch.inference_mode():
        expected = reference.forward(pixels.to("cuda"), grid.to("cuda"))
        first = source.forward_layer_streamed(pixels, grid, device=torch.device("cuda"))
        second = source.forward_layer_streamed(pixels, grid, device=torch.device("cuda"))

    torch.testing.assert_close(first, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(second, expected, rtol=2e-3, atol=2e-3)
    assert [layer for layer, _target in copied_blocks] == list(range(27)) * 2
    assert len({target for _layer, target in copied_blocks[:27]}) == 1
    assert len({target for _layer, target in copied_blocks[27:]}) == 1
    assert source._active_stream_workspace is None
    assert len(empty_cache_calls) == 2
    assert first.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
@pytest.mark.parametrize("failure_stage", ("patch", "block", "merger"))
def test_layer_stream_failure_cleans_workspace_and_next_encode_succeeds(
    monkeypatch, failure_stage
):
    from freetoken.models.qwen4_exp import vision as vision_module

    config = _stream_config(depth=3)
    source = _cpu_stream_source(config)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixels = torch.randn(16, 24, dtype=torch.bfloat16)
    original_copy = vision_module._copy_component_state_
    original_block_forward = Qwen4VisionBlock.forward

    if failure_stage == "block":
        def fail_block(*_args, **_kwargs):
            raise RuntimeError("injected block failure")

        monkeypatch.setattr(Qwen4VisionBlock, "forward", fail_block)
    else:
        failed_source = source.patch_embed if failure_stage == "patch" else source.merger

        def fail_component(target, origin):
            if origin is failed_source:
                raise RuntimeError(f"injected {failure_stage} failure")
            return original_copy(target, origin)

        monkeypatch.setattr(vision_module, "_copy_component_state_", fail_component)

    with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
        source.forward_layer_streamed(pixels, grid, device=torch.device("cuda"))
    assert source._active_stream_workspace is None

    monkeypatch.setattr(vision_module, "_copy_component_state_", original_copy)
    monkeypatch.setattr(Qwen4VisionBlock, "forward", original_block_forward)
    recovered = source.forward_layer_streamed(pixels, grid, device=torch.device("cuda"))
    assert recovered.shape == (4, config.out_hidden_size)
    assert source._active_stream_workspace is None


def test_layer_stream_rejects_deepstack_before_allocating_workspace():
    source = _cpu_stream_source(_stream_config(depth=2, deepstack_visual_indexes=(0,)))
    with pytest.raises(ValueError, match="deepstack_visual_indexes"):
        source.forward_layer_streamed(
            torch.randn(16, 24),
            torch.tensor([[1, 4, 4]], dtype=torch.long),
            device=torch.device("cpu"),
        )
    assert source._active_stream_workspace is None


# --------------------------------------------------------------------------------------
# Picture weights served from a mapping: the prefetch handshake
# --------------------------------------------------------------------------------------


class _RecordingSource:
    """Stands in for ``MmapVisionWeights``: records the prefetch handshake, maps nothing."""

    def __init__(self, calls: list[str], *, succeeds: bool = True) -> None:
        self.calls = calls
        self._succeeds = succeeds

    def prefetch(self) -> bool:
        self.calls.append("prefetch")
        return self._succeeds

    def release_prefetch(self) -> None:
        self.calls.append("release")


def test_picture_weight_prefetch_is_a_no_op_in_resident_mode():
    """``ram`` mode attaches no source, and the encode must not care."""
    source = _cpu_stream_source(_stream_config(depth=1))
    assert source.prefetch_weights() is False
    assert source.release_weight_prefetch() is None


def test_attached_weight_source_receives_the_prefetch():
    calls: list[str] = []
    source = _cpu_stream_source(_stream_config(depth=1))
    source.attach_weight_source(_RecordingSource(calls))

    assert source.prefetch_weights() is True
    source.release_weight_prefetch()
    assert calls == ["prefetch", "release"]


def test_a_failed_prefetch_is_reported_but_never_raises():
    source = _cpu_stream_source(_stream_config(depth=1))
    source.attach_weight_source(_RecordingSource([], succeeds=False))
    assert source.prefetch_weights() is False


def test_the_weight_source_is_invisible_to_the_state_dict():
    """It hangs off a ``_``-prefixed attribute, so ``BaseOP.state_dict`` skips it and
    ``load_state_dict``'s strict key check never sees it."""
    source = _cpu_stream_source(_stream_config(depth=1))
    before = set(source.state_dict())
    source.attach_weight_source(_RecordingSource([]))

    state = source.state_dict()
    assert set(state) == before
    assert all(isinstance(value, torch.Tensor) for value in state.values())


def test_streamed_encode_prefetches_at_entry_and_releases_on_every_exit():
    """The prefetch is the encode's first act -- in ``mmap`` mode the whole 856 MiB extent
    may be non-resident and the syscall returns while the reads continue -- and it is
    released however the encode ends, so the next picture issues its own."""
    calls: list[str] = []
    source = _cpu_stream_source(_stream_config(depth=1))
    source.attach_weight_source(_RecordingSource(calls))

    with pytest.raises(ValueError, match="requires a CUDA device"):
        source.forward_layer_streamed(
            torch.randn(16, 24),
            torch.tensor([[1, 4, 4]], dtype=torch.long),
            device=torch.device("cpu"),
        )

    assert calls == ["prefetch", "release"]
    assert source._active_stream_workspace is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_streamed_encode_from_mapped_sources_prefetches_exactly_once():
    torch.manual_seed(31)
    calls: list[str] = []
    source = _cpu_stream_source(_stream_config(depth=2))
    source.attach_weight_source(_RecordingSource(calls))

    features = source.forward_layer_streamed(
        torch.randn(16, 24, dtype=torch.bfloat16),
        torch.tensor([[1, 4, 4]], dtype=torch.long),
        device=torch.device("cuda"),
    )

    assert features.device.type == "cuda"
    assert calls == ["prefetch", "release"]


def test_qwen_picture_encoding_owns_input_and_output_placement():
    calls = []

    class Visual:
        def forward(self, pixels, grid):
            calls.append(("gpu", pixels.device, grid.device))
            return torch.ones(4, 8, device=pixels.device)

        def forward_layer_streamed(self, pixels, grid, *, device):
            calls.append(("layer-stream", pixels.device, grid.device, device))
            return torch.ones(4, 8, device=device)

    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model.visual = Visual()
    model.model = SimpleNamespace(
        embed_tokens=SimpleNamespace(weight=torch.empty(2, 8, device="cpu"))
    )
    pixels = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)

    model._vision_execution = "gpu"
    gpu_result = model.encode_images(pixels, grid)
    model._vision_execution = "layer-stream"
    stream_result = model.encode_images(pixels, grid)

    assert calls == [
        ("gpu", torch.device("cpu"), torch.device("cpu")),
        ("layer-stream", torch.device("cpu"), torch.device("cpu"), torch.device("cpu")),
    ]
    assert gpu_result.device.type == stream_result.device.type == "cpu"


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

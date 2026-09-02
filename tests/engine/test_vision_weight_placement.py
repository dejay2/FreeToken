from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.engine import engine as engine_module
from freetoken.engine.engine import Engine, _materialize_loaded_weight_state_dict
from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM


class _Recorder:
    """Stands in for the engine's module logger."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def info_rank0(self, message, *args):
        self.lines.append(message % args if args else message)


def _placement_config(**over):
    """The engine-owned half of the placement report needs a config to read."""
    model_config = SimpleNamespace(
        dense_quant=over.pop("dense_quant", "none"),
        num_moe_layers=over.pop("num_moe_layers", 0),
        num_experts=over.pop("num_experts", 0),
    )
    return SimpleNamespace(
        model_config=model_config,
        moe_backend=over.pop("moe_backend", "fused"),
        moe_cache_size=over.pop("moe_cache_size", 0),
        **over,
    )


def test_weight_sources_are_adopted_before_the_placement_report(monkeypatch):
    """The report describes what the weight sources decided, so it has to run after they are
    adopted. It used to run twenty lines earlier than the adoption, which made every mmap
    boot log ``backing=ram`` while the mapping was demonstrably live.

    Since the expert placement joined it, the report is emitted later still -- after the
    host tables and the offload MoE cache exist -- so installing the weights logs nothing.
    """
    order: list[str] = []

    class Model:
        backing = "ram"

        def load_state_dict(self, state):
            order.append("load_state_dict")

        def adopt_weight_sources(self, engine_config):
            order.append("adopt_weight_sources")
            self.backing = "mmap"

        def weight_placement_report(self):
            order.append("weight_placement_report")
            return f"Picture weights: backing={self.backing}"

    recorder = _Recorder()
    monkeypatch.setattr(engine_module, "logger", recorder)
    monkeypatch.setattr(Engine, "_load_weight_state_dict", lambda self, config: {})
    engine = Engine.__new__(Engine)
    engine.model = Model()

    Engine._install_model_weights(engine, SimpleNamespace())
    assert order == ["load_state_dict", "adopt_weight_sources"]
    assert recorder.lines == []

    Engine._log_weight_placement_report(engine, _placement_config())
    assert order == ["load_state_dict", "adopt_weight_sources", "weight_placement_report"]
    assert recorder.lines == ["Picture weights: backing=mmap"]


def test_a_model_with_no_weight_sources_to_adopt_still_reports(monkeypatch):
    """Both hooks are optional: every non-Qwen model has neither."""
    recorder = _Recorder()
    monkeypatch.setattr(engine_module, "logger", recorder)
    monkeypatch.setattr(Engine, "_load_weight_state_dict", lambda self, config: {})
    engine = Engine.__new__(Engine)
    engine.model = SimpleNamespace(load_state_dict=lambda state: None)

    Engine._install_model_weights(engine, SimpleNamespace())
    Engine._log_weight_placement_report(engine, _placement_config())

    assert recorder.lines == []


def test_an_empty_report_is_not_logged(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(engine_module, "logger", recorder)
    monkeypatch.setattr(Engine, "_load_weight_state_dict", lambda self, config: {})
    engine = Engine.__new__(Engine)
    engine.model = SimpleNamespace(
        load_state_dict=lambda state: None, weight_placement_report=lambda: ""
    )

    Engine._install_model_weights(engine, SimpleNamespace())
    Engine._log_weight_placement_report(engine, _placement_config())

    assert recorder.lines == []


def test_the_placement_report_carries_the_expert_placement(monkeypatch):
    """Issue 4: the report was emitted from _install_model_weights, twenty lines before the
    offload cache existed, so it could never name a GPU-owned layer. One block now."""
    recorder = _Recorder()
    monkeypatch.setattr(engine_module, "logger", recorder)
    engine = Engine.__new__(Engine)
    engine.model = SimpleNamespace(
        weight_placement_report=lambda: "Token embedding: host-resident, bytes=8, device=cpu"
    )
    engine._gpu_owned_layer_ids = frozenset({0, 1, 2, 6, 7, 22})
    engine.moe_offload_cache = SimpleNamespace(cpu_layer_ids=frozenset())

    Engine._log_weight_placement_report(
        engine,
        _placement_config(
            dense_quant="int8", num_moe_layers=48, num_experts=512,
            moe_backend="offload", moe_cache_size=3678,
        ),
    )

    assert len(recorder.lines) == 1  # ONE block, not a line per subsystem
    block = recorder.lines[0].splitlines()
    assert block[0] == "Token embedding: host-resident, bytes=8, device=cpu"
    assert "Dense weights: quant=int8" in block[1]
    assert block[2] == (
        "Expert placement: backend=offload, moe_layers=48, experts=512, "
        "gpu_owned_layers=[0, 1, 2, 6, 7, 22], cpu_layers=[], streaming_layers=42, "
        "lru_slots=3678"
    )


def test_a_dense_model_reports_no_expert_placement(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(engine_module, "logger", recorder)
    engine = Engine.__new__(Engine)
    engine.model = SimpleNamespace(weight_placement_report=lambda: "Picture weights: x")

    Engine._log_weight_placement_report(engine, _placement_config(dense_quant="int8"))

    assert recorder.lines == ["Picture weights: x\nDense weights: quant=int8"]


def _model_state():
    return {
        "model.embed_tokens.weight": torch.empty(2, 3, dtype=torch.bfloat16, device="meta"),
        "visual.patch_embed.proj.weight": torch.empty(
            4, 3, dtype=torch.bfloat16, device="meta"
        ),
    }


def _weights():
    return [
        ("model.embed_tokens.weight", torch.arange(6, dtype=torch.float32).view(2, 3)),
        ("visual.patch_embed.proj.weight", torch.arange(12, dtype=torch.float32).view(4, 3)),
    ]


def test_materializer_uses_per_key_destination_and_preserves_expected_dtype():
    seen = []

    def destination(key: str, engine_device: torch.device) -> torch.device:
        seen.append((key, engine_device))
        return torch.device("cpu") if key.startswith("visual.") else engine_device

    state = _materialize_loaded_weight_state_dict(
        _model_state(),
        _weights(),
        device=torch.device("meta"),
        device_for_key=destination,
    )

    assert [key for key, _device in seen] == [
        "model.embed_tokens.weight",
        "visual.patch_embed.proj.weight",
    ]
    assert all(device == torch.device("meta") for _key, device in seen)
    assert state["model.embed_tokens.weight"].device.type == "meta"
    assert state["visual.patch_embed.proj.weight"].device.type == "cpu"
    assert state["visual.patch_embed.proj.weight"].dtype is torch.bfloat16
    assert torch.equal(
        state["visual.patch_embed.proj.weight"],
        torch.arange(12, dtype=torch.bfloat16).view(4, 3),
    )


def test_materializer_without_destination_hook_keeps_one_device_behavior():
    state = _materialize_loaded_weight_state_dict(
        _model_state(), _weights(), device=torch.device("meta")
    )
    assert {tensor.device.type for tensor in state.values()} == {"meta"}


def _model_shell(mode: str) -> Qwen4ExpForCausalLM:
    model = Qwen4ExpForCausalLM.__new__(Qwen4ExpForCausalLM)
    model._vision_execution = mode
    return model


def test_qwen_layer_stream_routes_only_visual_keys_to_cpu():
    engine_device = torch.device("meta")
    model = _model_shell("layer-stream")

    visual_device = model.weight_device_for_key(
        "visual.blocks.0.attn.qkv.weight", engine_device
    )
    assert visual_device == torch.device("cpu")
    assert model.weight_device_for_key("model.embed_tokens.weight", engine_device) == engine_device
    assert model.weight_device_for_key("lm_head.weight", engine_device) == engine_device


def test_qwen_gpu_mode_keeps_every_key_on_engine_device():
    engine_device = torch.device("meta")
    model = _model_shell("gpu")
    for key in ("visual.blocks.0.attn.qkv.weight", "model.embed_tokens.weight", "lm_head.weight"):
        assert model.weight_device_for_key(key, engine_device) == engine_device


def test_qwen_picture_weight_report_records_exact_count_bytes_and_devices():
    model = _model_shell("layer-stream")
    state = {
        "one": torch.zeros(2, 3, dtype=torch.bfloat16),
        "two": torch.zeros(5, dtype=torch.float32),
    }
    model.visual = SimpleNamespace(state_dict=lambda: state, weight_backing=lambda: "ram")

    assert model.weight_placement_report() == (
        "Picture weights: mode=layer-stream, backing=ram, tensors=2, bytes=32, devices=cpu"
    )


def test_qwen_picture_weight_report_states_the_mapped_backing():
    """Acceptance criterion 1: the boot log says which backing is live without a new probe."""
    model = _model_shell("layer-stream")
    state = {"one": torch.zeros(2, 3, dtype=torch.bfloat16)}
    model.visual = SimpleNamespace(state_dict=lambda: state, weight_backing=lambda: "mmap")

    assert model.weight_placement_report() == (
        "Picture weights: mode=layer-stream, backing=mmap, tensors=1, bytes=12, devices=cpu"
    )


def test_prefetch_picture_weights_delegates_to_the_tower():
    calls = []
    model = _model_shell("layer-stream")
    model.visual = SimpleNamespace(prefetch_weights=lambda: calls.append("prefetch"))

    model.prefetch_picture_weights()

    assert calls == ["prefetch"]


def test_prefetch_picture_weights_is_a_no_op_on_a_text_only_model():
    """A text-only boot builds no tower; the scheduler hook must still be safe to call."""
    model = _model_shell("gpu")
    assert not hasattr(model, "visual")
    assert model.prefetch_picture_weights() is None


def test_adopt_weight_sources_hands_the_tower_the_mapping_the_loader_built(monkeypatch):
    from freetoken.models.qwen4_exp import weight as weight_mod

    holder = object()
    adopted = []
    model = _model_shell("layer-stream")
    model.visual = SimpleNamespace(attach_weight_source=adopted.append)
    monkeypatch.setattr(weight_mod, "mmap_vision_weights", lambda path: holder)

    model.adopt_weight_sources(SimpleNamespace(model_path="whatever"))

    assert adopted == [holder]


def test_adopt_weight_sources_attaches_nothing_in_resident_mode(monkeypatch):
    """``ram``, an FTW checkpoint, or a mapping the OS refused: no holder, no attachment."""
    from freetoken.models.qwen4_exp import weight as weight_mod

    adopted = []
    model = _model_shell("layer-stream")
    model.visual = SimpleNamespace(attach_weight_source=adopted.append)
    monkeypatch.setattr(weight_mod, "mmap_vision_weights", lambda path: None)

    model.adopt_weight_sources(SimpleNamespace(model_path="whatever"))

    assert adopted == []


def test_adopt_weight_sources_is_a_no_op_on_a_text_only_model():
    model = _model_shell("gpu")
    assert not hasattr(model, "visual")
    assert model.adopt_weight_sources(SimpleNamespace(model_path="whatever")) is None


def test_qwen_report_names_the_ple_table_backing():
    """Issue 4: the PLE table is 47.7 GiB of the boot's host residency and had no line at
    all. It can only be reported once load_host_tables has run, which is why the report
    moved after it."""
    model = _model_shell("gpu")
    model._ple_placement = "backend=mmap, mapped_bytes=51204915200, layers=1"

    assert model.weight_placement_report() == (
        "PLE table: backend=mmap, mapped_bytes=51204915200, layers=1"
    )


def test_qwen_report_without_a_ple_table_says_nothing_about_one():
    assert _model_shell("gpu").weight_placement_report() == ""

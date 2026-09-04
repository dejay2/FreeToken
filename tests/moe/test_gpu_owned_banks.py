"""GPU-owned MoE layers, loader half (--moe-gpu-owned-layers).

CPU-only. The "device" is ``torch.device("cpu")``, so the direct fill of the owned layer's
device tensor, the layer-completion sink and every refusal are exercised without CUDA.

Two of the tests here run the REAL NVFP4 shard loaders over a synthetic checkpoint inside
``torch.inference_mode()``. That context manager is the whole difference between a green
suite and the 2026-09-02 live run, which hung: inference mode is thread-local, so the owned
layers' device banks are inference tensors and any thread but the loading one is forbidden
to write them (``docs/research/measurements-gpu-owned-layers-2026-09-02.md``).

What is NOT covered here and only a live GPU run proves (see the operator checklist,
docs/plans/2026-09-02-qwen38-gpu-owned-moe-layers-status.md): the real pageable H2D copy,
and that the resident VRAM rows are byte-identical to a normal host-bank load.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import threading
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

import freetoken.moe.host_banks as hb

_SPECS = {
    "gate_up": ((2, 4), torch.float32),
    "down": ((2, 3), torch.float32),
}
CPU = torch.device("cpu")


# --------------------------------------------------------------------------------------
# A synthetic NVFP4 expert checkpoint, so the real loaders can run without D:\Models.
# Shapes follow models/nvfp4_banks.py::_alloc_nvfp4_host_banks (H % 16 == 0, I % 16 == 0).
# --------------------------------------------------------------------------------------

_E, _H, _I, _NL = 2, 32, 16, 3

_CKPT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_CKPT_CONFIG = SimpleNamespace(
    num_experts=_E, hidden_size=_H, moe_intermediate_size=_I, num_moe_layers=_NL
)


def _ckpt_spec():
    from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec

    return Nvfp4ExpertSourceSpec(
        key_pattern=_CKPT_KEY_RE,
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,
        desc="synthetic NVFP4 experts",
    )


def _write_synthetic_checkpoint(root: str) -> str:
    """One shard per MoE layer plus the safetensors index, deterministic contents."""
    g = torch.Generator().manual_seed(1234)
    weight_map: dict[str, str] = {}
    for layer in range(_NL):
        shard = f"model-{layer:05d}-of-{_NL:05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        for expert in range(_E):
            for proj, rows, cols in (
                ("gate_proj", _I, _H),
                ("up_proj", _I, _H),
                ("down_proj", _H, _I),
            ):
                base = (
                    f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
                )
                tensors[base + ".weight"] = torch.randint(
                    0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g
                )
                tensors[base + ".weight_scale"] = (
                    torch.rand((rows, cols // 16), generator=g) + 0.5
                ).to(torch.float8_e4m3fn)
                tensors[base + ".weight_scale_2"] = torch.tensor(
                    [0.125 * (layer + 1) + 0.0625 * expert], dtype=torch.float32
                )
        safetensors.torch.save_file(tensors, os.path.join(root, shard))
        weight_map.update({name: shard for name in tensors})
    with open(
        os.path.join(root, "model.safetensors.index.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump({"metadata": {}, "weight_map": weight_map}, fh)
    return root


def _no_drop(path: str) -> None:  # the loaders' DropPageCache hook
    return None


def _load_ckpt(folder, *, owned, inference=True, parallel=False, sink=None):
    """Run a real NVFP4 loader exactly as the engine does: inside ``inference_mode()``,
    with the ambient residency plan naming the GPU-owned layers."""
    import contextlib

    from freetoken.models.nvfp4_banks import (
        load_nvfp4_expert_source_banks,
        load_nvfp4_expert_source_banks_parallel,
    )

    loader = (
        load_nvfp4_expert_source_banks_parallel if parallel
        else load_nvfp4_expert_source_banks
    )
    labels = [
        hb.HostResidency.GPU_OWNED.value if i in owned else hb.HostResidency.PINNED.value
        for i in range(_NL)
    ]
    plan = hb.requested_residency(labels, device=CPU) if owned else contextlib.nullcontext()
    with torch.inference_mode(inference), plan:
        return loader(
            folder,
            _CKPT_CONFIG,
            _ckpt_spec(),
            drop_page_cache=_no_drop,
            primary=False,
            layer_sink=sink,
        )


def _assert_same_bytes(got: dict, want: dict) -> None:
    assert got.keys() == want.keys()
    for name in want:
        assert len(got[name]) == len(want[name])
        for layer, (a, b) in enumerate(zip(got[name], want[name])):
            assert a.shape == b.shape and a.dtype == b.dtype, (name, layer)
            assert torch.equal(
                a.contiguous().flatten().view(torch.uint8),
                b.contiguous().flatten().view(torch.uint8),
            ), f"{name} layer {layer} differs"


@pytest.fixture(scope="module")
def synthetic_ckpt(tmp_path_factory):
    return _write_synthetic_checkpoint(str(tmp_path_factory.mktemp("nvfp4-ckpt")))


@pytest.fixture(scope="module")
def reference_banks(synthetic_ckpt):
    """The same checkpoint loaded with no owned layers: the byte-identity oracle."""
    os.environ["FREETOKEN_SKIP_BANK_PIN"] = "1"  # no CUDA here; pin-after-fill is a no-op
    try:
        return _load_ckpt(synthetic_ckpt, owned=frozenset(), inference=False)
    finally:
        os.environ.pop("FREETOKEN_SKIP_BANK_PIN", None)


# --------------------------------------------------------------------------------------
# The 2026-09-02 live failure, as tests.
# --------------------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_the_serial_loader_fills_owned_layers_inside_inference_mode(
    synthetic_ckpt, reference_banks, monkeypatch
):
    # The live hang: the engine loads weights inside torch.inference_mode(), which is
    # THREAD-LOCAL. Owned layers must therefore be filled by the placement thread itself --
    # nothing may be handed to the PinPipeline drain thread, which is not in inference mode.
    settled: list = []
    monkeypatch.setattr(
        hb.PinPipeline, "submit", lambda self, *a, **k: settled.append(a)
    )

    banks = _load_ckpt(synthetic_ckpt, owned=frozenset({0, 1, 2}))

    _assert_same_bytes(banks, reference_banks)
    assert settled == [], "a GPU-owned layer must never reach the settle pipeline"


@pytest.mark.timeout(60)
def test_the_parallel_reader_may_interleave_three_owned_layers(
    synthetic_ckpt, reference_banks, monkeypatch
):
    # The other half of the live defect: the placement loop is single-threaded, so any
    # bounded per-layer resource it must wait on deadlocks once the reader has more owned
    # layers in flight than the bound. Deliver strict round-robin across all three.
    import freetoken.models.weight as weight_mod

    def interleaved(model_path, is_expert, **kwargs):
        by_layer: dict[int, list[str]] = collections.defaultdict(list)
        tensors: dict[str, torch.Tensor] = {}
        for path in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            for name, tensor in safetensors.torch.load_file(path).items():
                if is_expert(name):
                    tensors[name] = tensor
                    by_layer[int(_CKPT_KEY_RE.match(name).group("layer"))].append(name)
        for names in by_layer.values():
            names.sort()
        for i in range(max(len(v) for v in by_layer.values())):
            for layer in sorted(by_layer):
                if i < len(by_layer[layer]):
                    name = by_layer[layer][i]
                    yield name, tensors[name]

    monkeypatch.setattr(weight_mod, "iter_expert_tensors_parallel", interleaved)

    banks = _load_ckpt(synthetic_ckpt, owned=frozenset({0, 1, 2}), parallel=True)

    _assert_same_bytes(banks, reference_banks)


@pytest.mark.timeout(60)
def test_an_owned_layer_is_already_complete_when_its_completion_sink_fires(
    synthetic_ckpt, reference_banks
):
    # No staging, no deferred flush: the device tensor holds the whole layer the moment the
    # last of its E*6 writes lands, on the thread that wrote them.
    seen: list[tuple[int, int, bool]] = []

    def sink(layer_id, banks):
        complete = all(
            torch.equal(
                bank.tensor.contiguous().flatten().view(torch.uint8),
                reference_banks[name][layer_id].contiguous().flatten().view(torch.uint8),
            )
            for name, bank in banks.items()
        )
        seen.append((layer_id, threading.get_ident(), complete))

    _load_ckpt(synthetic_ckpt, owned=frozenset({0, 1, 2}), sink=sink)

    assert [s[0] for s in seen] == [0, 1, 2]
    assert {s[1] for s in seen} == {threading.get_ident()}, "filled off the placement thread"
    assert all(s[2] for s in seen), "the device tensor was not complete at layer completion"


# --------------------------------------------------------------------------------------
# Unit-level behaviour of the primitives.
# --------------------------------------------------------------------------------------


def test_alloc_layer_banks_puts_owned_layers_on_the_device_and_allocates_no_host_bank():
    banks = hb.alloc_layer_banks(_SPECS, 3, gpu_owned=frozenset({1}), device=CPU)

    for name, (shape, dtype) in _SPECS.items():
        assert isinstance(banks[name][0], hb.HostBank)
        assert isinstance(banks[name][2], hb.HostBank)
        owned = banks[name][1]
        assert isinstance(owned, hb.GpuOwnedBank)
        assert owned.tensor.shape == shape and owned.tensor.dtype == dtype
        assert owned.tensor.device == CPU
        assert owned.residency is hb.HostResidency.GPU_OWNED
        # the per-layer list keeps length num_layers so every consumer still sees one entry
        assert len(banks[name]) == 3
    # both kinds of bank are written where they live: no staging indirection anywhere
    assert banks["gate_up"][0].fill is banks["gate_up"][0].tensor
    assert banks["gate_up"][1].fill is banks["gate_up"][1].tensor


def test_alloc_layer_banks_reads_the_owned_set_from_the_ambient_plan():
    labels = [
        hb.HostResidency.PINNED.value,
        hb.HostResidency.GPU_OWNED.value,
    ]
    with hb.requested_residency(labels, device=CPU):
        banks = hb.alloc_layer_banks(_SPECS, 2)

    assert isinstance(banks["gate_up"][0], hb.HostBank)
    assert isinstance(banks["gate_up"][1], hb.GpuOwnedBank)


def test_the_layer_completion_sink_settles_nothing_for_a_gpu_owned_layer():
    banks = hb.alloc_layer_banks(_SPECS, 2, gpu_owned=frozenset({1}), device=CPU)
    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.GPU_OWNED.value]
    payload = {
        "gate_up": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "down": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    for name, value in payload.items():
        banks[name][1].fill.copy_(value)

    with hb.requested_residency(labels, device=CPU):
        with hb.PinPipeline() as pins:
            pins(1, {name: per[1] for name, per in banks.items()})

    # the fill already landed in the device tensor; the sink had nothing to do
    for name, value in payload.items():
        assert torch.equal(banks[name][1].tensor, value)


def test_a_gpu_owned_layer_at_the_plain_settle_path_fails_loudly():
    # a provider without a per-layer completion sink would leave the device banks unfilled
    banks = hb.alloc_layer_banks(_SPECS, 2, gpu_owned=frozenset({1}), device=CPU)
    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.GPU_OWNED.value]

    with hb.requested_residency(labels, device=CPU):
        with pytest.raises(RuntimeError, match="no per-layer completion sink"):
            hb.pin_banks(banks)


@pytest.mark.timeout(60)
def test_a_failing_settle_surfaces_at_pin_pipeline_wait_and_blocks_nobody():
    # No caller ever waits on the drain thread now that owned layers fill in place, so
    # _run's "drain without settling after a failure" cannot strand anyone; the stored
    # exception is raised by wait()/__exit__ and later banks are left unsettled.
    class Boom:
        def pin(self):
            raise RuntimeError("boom")

    later = SimpleNamespace(pinned=False)

    class Later:
        def pin(self):
            later.pinned = True

    pins = hb.PinPipeline()
    pins.submit(Boom())
    pins.submit(Later())

    with pytest.raises(RuntimeError, match="boom"):
        pins.wait()
    assert later.pinned is False


def test_bank_bytes_estimate_subtracts_gpu_owned_layers():
    from freetoken.moe.expert_banks import bank_bytes_estimate

    config = SimpleNamespace(
        expert_quant="nvfp4", moe_weight_format=None,
        num_moe_layers=48, num_experts=512, hidden_size=4096, moe_intermediate_size=512,
    )
    full = bank_bytes_estimate(config)
    owned6 = bank_bytes_estimate(config, gpu_owned=6)

    assert full is not None
    assert owned6 == full * 42 // 48


def test_echo_residency_refuses_an_unapplied_gpu_owned_request():
    from freetoken.moe.expert_banks import ExpertBanks, _echo_residency

    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.GPU_OWNED.value]
    banks = ExpertBanks("bf16", {"gate_up": [], "down": []})
    stale = hb._ResidencyPlan(labels)  # never consulted -> the layers got host banks

    with pytest.raises(RuntimeError, match="--moe-gpu-owned-layers"):
        _echo_residency(banks, labels, stale)


def test_ftw_bank_loader_refuses_gpu_owned_layers(tmp_path):
    from freetoken.checkpoint.ftw import load_ftw_banks

    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.GPU_OWNED.value]
    with pytest.raises(ValueError, match="FTW packed checkpoint"):
        load_ftw_banks(str(tmp_path), num_layers=2, layer_residency=labels)


def test_cpu_moe_executor_refuses_cuda_bank_sources():
    # _make_table hands C++ raw data_ptr()s it dereferences on the CPU; a CUDA source
    # (a GPU-owned layer) would be a silent wrong-memory read
    from freetoken.moe.cpu_executor import _reject_cuda_sources

    ok = {"gate_up": [SimpleNamespace(is_cuda=False)], "down": [SimpleNamespace(is_cuda=False)]}
    _reject_cuda_sources(ok)

    bad = {
        "gate_up": [SimpleNamespace(is_cuda=False), SimpleNamespace(is_cuda=True)],
        "down": [SimpleNamespace(is_cuda=False), SimpleNamespace(is_cuda=False)],
    }
    with pytest.raises(ValueError, match=r"bank 'gate_up' layer 1"):
        _reject_cuda_sources(bad)


# ------------------------------------------------------ the expert quant-format guard
# Only reviewed providers (currently NVFP4 and EXL3) fill an owned layer's device banks
# correctly. Since 8a63977 removed the staging indirection, any other provider writes straight
# THROUGH .fill / .tensor into the device tensor, so a wrong-geometry load no longer trips an
# assert -- it may silently appear to work. Refuse the format instead of relying on that accident.


def test_the_config_validator_refuses_a_non_nvfp4_expert_quant():
    from freetoken.engine.engine import _validate_gpu_owned_layers

    config = SimpleNamespace(
        moe_gpu_owned_layers="auto:2",
        moe_backend="offload",
        moe_cpu_layers=None,
        moe_cache_size=0,
        moe_cache_auto=True,
        moe_prefill_overlap=True,
        model_path="",
        model_config=SimpleNamespace(
            num_moe_layers=48, num_experts=512, expert_quant="q4_0",
        ),
    )
    with pytest.raises(ValueError) as excinfo:
        _validate_gpu_owned_layers(config, 48)
    message = str(excinfo.value)
    assert "--moe-gpu-owned-layers" in message
    assert "q4_0" in message  # the refusal names the format it found
    assert "nvfp4" in message

    config.model_config.expert_quant = "nvfp4"
    assert _validate_gpu_owned_layers(config, 48) == frozenset({1, 6})


def test_the_loader_refuses_owned_banks_for_a_non_nvfp4_provider():
    """The load-time half: a fake provider that asks the ambient plan for owned layers must
    be refused BEFORE any device tensor exists."""
    labels = [hb.HostResidency.GPU_OWNED.value, hb.HostResidency.PINNED.value]

    with hb.requested_residency(labels, device=CPU, expert_quant="q4_0"):
        with pytest.raises(ValueError) as excinfo:
            hb.plan_gpu_owned()
        assert "q4_0" in str(excinfo.value)
        with pytest.raises(ValueError, match="q4_0"):
            hb.alloc_layer_banks(_SPECS, 2)

    # ...and the NVFP4 providers still get their owned banks
    with hb.requested_residency(labels, device=CPU, expert_quant="nvfp4"):
        owned, device = hb.plan_gpu_owned()
        assert owned == frozenset({0})
        assert device == CPU
        banks = hb.alloc_layer_banks(_SPECS, 2)
        assert isinstance(banks["gate_up"][0], hb.GpuOwnedBank)
        assert isinstance(banks["gate_up"][1], hb.HostBank)


def test_a_plan_without_a_declared_quant_format_is_not_second_guessed():
    """Hand-built plans (tests, the shadow tooling) declare no format and keep working."""
    labels = [hb.HostResidency.GPU_OWNED.value, hb.HostResidency.PINNED.value]

    with hb.requested_residency(labels, device=CPU):
        assert hb.plan_gpu_owned() == (frozenset({0}), CPU)


def test_load_expert_banks_refuses_owned_layers_for_a_non_nvfp4_checkpoint(monkeypatch):
    """End to end through the real dispatch, with a provider that would have written the
    device tensor: the refusal names the format and no bank is built."""
    from freetoken.moe import expert_banks

    built = []

    def fake_provider(model_path, model_config, device, dtype, dummy, **kwargs):
        built.append(model_path)
        hb.alloc_layer_banks(_SPECS, model_config.num_moe_layers)
        raise AssertionError("unreachable: the guard must fire first")

    monkeypatch.setitem(expert_banks._PROVIDERS, "q4_0", fake_provider)
    config = SimpleNamespace(
        expert_quant="q4_0", num_moe_layers=2, num_experts=_E, architectures=["Fake"],
    )
    labels = [hb.HostResidency.GPU_OWNED.value, hb.HostResidency.PINNED.value]

    with pytest.raises(ValueError) as excinfo:
        expert_banks.load_expert_banks(
            "", config, device=CPU, dtype=torch.float32, dummy=True,
            parallel=False, layer_residency=labels,
        )
    assert "q4_0" in str(excinfo.value)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))

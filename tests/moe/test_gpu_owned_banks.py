"""GPU-owned MoE layers, loader half (--moe-gpu-owned-layers).

CPU-only. The "device" is ``torch.device("cpu")``, so the staging -> destination copy, the
two-slot staging pool with back-pressure, the layer-completion flush and every refusal are
exercised without CUDA.

What is NOT covered here and only a live GPU run proves (see the operator checklist,
docs/plans/2026-09-02-qwen38-gpu-owned-moe-layers-status.md): the cudaHostAlloc'd staging
banks, the ``copy_(non_blocking=True)`` H2D itself, the CUDA event that gates staging reuse,
and that the resident VRAM rows are byte-identical to a normal host-bank load.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

import freetoken.moe.host_banks as hb

_SPECS = {
    "gate_up": ((2, 4), torch.float32),
    "down": ((2, 3), torch.float32),
}
CPU = torch.device("cpu")


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
    # a plain host bank writes where it lives; an owned bank writes into shared staging
    assert banks["gate_up"][0].fill is banks["gate_up"][0].tensor
    assert banks["gate_up"][1].fill.data_ptr() != banks["gate_up"][1].tensor.data_ptr()


def test_alloc_layer_banks_reads_the_owned_set_from_the_ambient_plan():
    labels = [
        hb.HostResidency.PINNED.value,
        hb.HostResidency.GPU_OWNED.value,
    ]
    with hb.requested_residency(labels, device=CPU):
        banks = hb.alloc_layer_banks(_SPECS, 2)

    assert isinstance(banks["gate_up"][0], hb.HostBank)
    assert isinstance(banks["gate_up"][1], hb.GpuOwnedBank)


def test_the_staging_pool_hands_out_two_layers_and_back_pressures_the_third():
    # the parallel reader interleaves layers, so the pool must bound in-flight staging
    # (spec section 4.2 choice (a): cap 2) instead of serializing the read
    pool = hb.GpuOwnedStagingPool(_SPECS, cap=2, device=CPU)
    first = pool.fill_view(0, "gate_up")
    second = pool.fill_view(1, "gate_up")

    assert first.data_ptr() != second.data_ptr()
    assert pool.fill_view(0, "gate_up").data_ptr() == first.data_ptr()  # sticky per layer

    started, got = threading.Event(), []

    def third() -> None:
        started.set()
        got.append(pool.fill_view(2, "gate_up").data_ptr())

    worker = threading.Thread(target=third, daemon=True)
    worker.start()
    started.wait(timeout=5)
    worker.join(timeout=0.5)
    assert worker.is_alive(), "a third owned layer must wait for a staging slot"

    pool.flush(0, {"gate_up": torch.zeros(2, 4), "down": torch.zeros(2, 3)})
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert got == [first.data_ptr()], "the freed slot must be recycled, not a third allocation"


def test_the_layer_completion_sink_flushes_staging_into_the_device_tensor():
    banks = hb.alloc_layer_banks(_SPECS, 2, gpu_owned=frozenset({1}), device=CPU)
    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.GPU_OWNED.value]
    payload = {
        "gate_up": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "down": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    for name, value in payload.items():
        banks[name][1].fill.copy_(value)
    staging_ptr = banks["gate_up"][1].fill.data_ptr()

    with hb.requested_residency(labels, device=CPU):
        with hb.PinPipeline() as pins:
            pins(1, {name: per[1] for name, per in banks.items()})

    for name, value in payload.items():
        assert torch.equal(banks[name][1].tensor, value)
        assert banks[name][1].tensor.data_ptr() != staging_ptr
    # the staging slot came back to the pool, so the next owned layer reuses it
    pool = banks["gate_up"][1].pool
    assert pool.fill_view(7, "gate_up").data_ptr() == staging_ptr


def test_a_gpu_owned_layer_at_the_plain_settle_path_fails_loudly():
    # a provider without a per-layer completion sink would leave the device banks unfilled
    banks = hb.alloc_layer_banks(_SPECS, 2, gpu_owned=frozenset({1}), device=CPU)
    labels = [hb.HostResidency.PINNED.value, hb.HostResidency.GPU_OWNED.value]

    with hb.requested_residency(labels, device=CPU):
        with pytest.raises(RuntimeError, match="no per-layer completion sink"):
            hb.pin_banks(banks)


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))

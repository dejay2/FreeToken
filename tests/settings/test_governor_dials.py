from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from freetoken.daemon.settings import dials
from freetoken.daemon.settings.boot_parser import BootFile
from freetoken.daemon.settings.linux_launch import ModelFacts, build_launch
from freetoken.daemon.settings.memory_fit import MemoryFitService, MachineSnapshot


REAL_BOOT = Path(__file__).parents[2] / "boot-2020.ps1"


def _copy_boot(tmp_path: Path) -> Path:
    target = tmp_path / "boot-2020.ps1"
    shutil.copy2(REAL_BOOT, target)
    return target


def _facts(**over) -> ModelFacts:
    base = dict(
        is_moe=True,
        expert_count=24576,
        max_context=262144,
        has_ple=True,
        has_vision=True,
        has_mtp=True,
        parking_supported=True,
        model_type="qwen4_exp",
    )
    base.update(over)
    return ModelFacts(**base)


def _arg(plan, flag: str) -> str | None:
    argv = plan.argv
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_governor_dials_metadata_and_validation() -> None:
    gov = dials.DIAL_BY_NAME["MemoryGovernor"]
    assert gov.control == "toggle"
    assert gov.default is True
    assert gov.group == "Expert slots and card memory"
    assert list(gov.effects) == ["speed:down"]

    vram = dials.DIAL_BY_NAME["GovernorVRAMFreeGB"]
    assert vram.control == "number"
    assert vram.default == 1.5
    assert vram.unit == "GB"
    assert vram.minimum == 0.0

    ram = dials.DIAL_BY_NAME["GovernorRAMFreeGB"]
    assert ram.control == "number"
    assert ram.default == 4.0
    assert ram.unit == "GB"
    assert ram.minimum == 0.0

    errors = dials.validate_settings({
        "MemoryGovernor": True,
        "GovernorVRAMFreeGB": 2.0,
        "GovernorRAMFreeGB": 8.0,
    })
    assert not errors
    assert dials.canonical_value(gov, True) is True
    assert dials.canonical_value(vram, 2.0) == 2.0
    assert dials.canonical_value(ram, 8.0) == 8.0


def test_governor_dials_save_and_load(tmp_path: Path) -> None:
    boot_path = _copy_boot(tmp_path)
    boot = BootFile(boot_path)

    # Save MemoryGovernor=True, GovernorVRAMFreeGB=2.5, GovernorRAMFreeGB=8.0
    saved = boot.save({
        "MemoryGovernor": True,
        "GovernorVRAMFreeGB": 2.5,
        "GovernorRAMFreeGB": 8.0,
    })
    assert saved["MemoryGovernor"] is True
    assert saved["GovernorVRAMFreeGB"] == 2.5
    assert saved["GovernorRAMFreeGB"] == 8.0

    # Reload from disk to verify persistence
    reloaded = BootFile(boot_path).load()
    assert reloaded["MemoryGovernor"] is True
    assert reloaded["GovernorVRAMFreeGB"] == 2.5
    assert reloaded["GovernorRAMFreeGB"] == 8.0

    content = boot_path.read_text(encoding="utf-8")
    assert "-MemoryGovernor" in content
    assert "-GovernorVRAMFreeGB 2.5" in content
    assert "-GovernorRAMFreeGB 8" in content

    # Now turn MemoryGovernor off and save again
    saved_off = boot.save({"MemoryGovernor": False})
    assert saved_off["MemoryGovernor"] is False
    reloaded_off = BootFile(boot_path).load()
    assert reloaded_off["MemoryGovernor"] is False
    content_off = boot_path.read_text(encoding="utf-8")
    assert "-MemoryGovernor" not in content_off


def test_governor_reserve_mapping_in_build_launch() -> None:
    # 1. When MemoryGovernor is True and MoEVramReserveBytes is 0:
    # Cushion bytes = round(1.5 * 1024**3) = 1610612736
    plan1 = build_launch(
        {
            "ModelPath": "/models/demo",
            "MoEVramReserveBytes": 0,
            "MemoryGovernor": True,
            "GovernorVRAMFreeGB": 1.5,
        },
        python="py",
        base_env={},
        facts=_facts(),
        wsl=False,
    )
    expected_cushion = str(int(round(1.5 * (1024 ** 3))))
    assert _arg(plan1, "--moe-vram-reserve-bytes") == expected_cushion

    # 2. When MemoryGovernor is False and MoEVramReserveBytes is 0:
    # Pass 0 directly
    plan2 = build_launch(
        {
            "ModelPath": "/models/demo",
            "MoEVramReserveBytes": 0,
            "MemoryGovernor": False,
        },
        python="py",
        base_env={},
        facts=_facts(),
        wsl=False,
    )
    assert _arg(plan2, "--moe-vram-reserve-bytes") == "0"

    # 3. When MoEVramReserveBytes is explicitly > 0 (e.g. 2 GiB):
    # Preserved as-is
    plan3 = build_launch(
        {
            "ModelPath": "/models/demo",
            "MoEVramReserveBytes": 2 * (1024 ** 3),
            "MemoryGovernor": True,
            "GovernorVRAMFreeGB": 1.5,
        },
        python="py",
        base_env={},
        facts=_facts(),
        wsl=False,
    )
    assert _arg(plan3, "--moe-vram-reserve-bytes") == str(2 * (1024 ** 3))

    # 4. When MoEVramReserveBytes is -1 (auto):
    # Flag is omitted
    plan4 = build_launch(
        {
            "ModelPath": "/models/demo",
            "MoEVramReserveBytes": -1,
            "MemoryGovernor": True,
        },
        python="py",
        base_env={},
        facts=_facts(),
        wsl=False,
    )
    assert "--moe-vram-reserve-bytes" not in plan4.argv

    # 5. Custom GovernorVRAMFreeGB (e.g. 2.0 GiB) with MoEVramReserveBytes=0:
    plan5 = build_launch(
        {
            "ModelPath": "/models/demo",
            "MoEVramReserveBytes": 0,
            "MemoryGovernor": True,
            "GovernorVRAMFreeGB": 2.0,
        },
        python="py",
        base_env={},
        facts=_facts(),
        wsl=False,
    )
    assert _arg(plan5, "--moe-vram-reserve-bytes") == str(int(round(2.0 * (1024 ** 3))))


def test_estimate_counts_governor_cushion(tmp_path: Path) -> None:
    # Model config stub
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "num_hidden_layers": 48,
            "num_experts": 512,
            "max_position_embeddings": 262144,
            "hidden_size": 16,
            "moe_intermediate_size": 32,
            "quantization_config": {"quant_algo": "NVFP4"},
        }),
        encoding="utf-8",
    )

    boot_path = _copy_boot(tmp_path)

    child_requests: list[dict] = []

    class FakeRunner:
        def __call__(self, argv, **kwargs):
            input_data = json.loads(kwargs.get("input", "{}"))
            child_requests.append(input_data)
            need = 50
            phase = {
                "need_bytes": need,
                "resident_bytes": need - 10,
                "boot_peak_bytes": need,
                "shortfall_bytes": 0,
            }
            result = {
                "version": 1,
                "status": "ok",
                "fits": True,
                "fits_now": True,
                "fits_empty": True,
                "sampled_at": None,
                "effective_settings": {},
                "machine": {},
                "effective": {
                    "page_size": 64,
                    "attention_backend": "qsa_sparse",
                    "ple_backend": "mmap",
                    "pin_budget_bytes": 80_000,
                    "bank_cuda_alloc": True,
                },
                "resources": {
                    "ram": {
                        "free_bytes": 100,
                        "total_bytes": 200,
                        "now": dict(phase),
                        "empty": dict(phase),
                    },
                    "vram": {
                        "free_bytes": 100,
                        "total_bytes": 200,
                        "now": dict(phase),
                        "empty": dict(phase),
                    },
                },
                "geometry": {
                    "now": {
                        "total_slots": 120,
                        "lru_slots": 116,
                        "owned_layers": [0],
                        "num_pages": 8,
                        "page_size": 64,
                        "usable_kv_tokens": 448,
                        "prefill_overlap": True,
                    },
                    "empty": {
                        "total_slots": 120,
                        "lru_slots": 116,
                        "owned_layers": [0],
                        "num_pages": 8,
                        "page_size": 64,
                        "usable_kv_tokens": 448,
                        "prefill_overlap": True,
                    },
                },
                "pinning": {"need_bytes": 50, "cap_bytes": 80_000, "shortfall_bytes": 0, "cpu_layers": []},
                "components": [],
                "issues": [],
                "assumptions": [],
                "suggestion": None,
            }
            return type("Result", (), {"stdout": json.dumps(result), "stderr": "", "returncode": 0})()

    def snapshot() -> dict:
        return {
            "ram_free_bytes": 16 * 1024**3,
            "ram_total_bytes": 32 * 1024**3,
            "vram_free_bytes": 8 * 1024**3,
            "vram_total_bytes": 16 * 1024**3,
        }

    service = MemoryFitService(snapshot=snapshot, runner=FakeRunner())

    # Estimate with MemoryGovernor=True and MoEVramReserveBytes=0
    service.estimate_settings(
        {
            "ModelPath": str(model_dir),
            "MoEVramReserveBytes": 0,
            "MemoryGovernor": True,
            "GovernorVRAMFreeGB": 1.5,
        },
        boot_file=BootFile(boot_path),
        environ={"PATH": "/bin"},
    )

    assert len(child_requests) == 1
    child_argv = child_requests[0]["argv"]
    expected_cushion = str(int(round(1.5 * (1024 ** 3))))
    assert "--moe-vram-reserve-bytes" in child_argv
    cushion_idx = child_argv.index("--moe-vram-reserve-bytes") + 1
    assert child_argv[cushion_idx] == expected_cushion

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


def _plan(settings: dict) -> object:
    return build_launch({"ModelPath": "/models/demo", **settings}, python="py", base_env={}, facts=_facts(), wsl=False)


def test_governor_cushion_maps_onto_headroom_never_the_reserve() -> None:
    cushion_1_5 = str(int(round(1.5 * (1024 ** 3))))
    two_gib = 2 * (1024 ** 3)

    # Governor on, headroom dial auto (-1): the cushion becomes the headroom; the reserve flag
    # keeps its own auto (-1 = omitted, so the engine still adds its graph/MTP reserve on top).
    plan = _plan({"MemoryGovernor": True, "GovernorVRAMFreeGB": 1.5})
    assert _arg(plan, "--moe-cache-headroom-bytes") == cushion_1_5
    assert "--moe-vram-reserve-bytes" not in plan.argv

    # The settings dict may lack the keys entirely: the dial defaults (on, 1.5) apply.
    plan = _plan({})
    assert _arg(plan, "--moe-cache-headroom-bytes") == cushion_1_5
    assert "--moe-vram-reserve-bytes" not in plan.argv

    # A larger explicit headroom dial wins over the cushion; a smaller one gives way.
    plan = _plan({"MemoryGovernor": True, "GovernorVRAMFreeGB": 1.5, "MoECacheHeadroomBytes": two_gib})
    assert _arg(plan, "--moe-cache-headroom-bytes") == str(two_gib)
    plan = _plan({"MemoryGovernor": True, "GovernorVRAMFreeGB": 2.0, "MoECacheHeadroomBytes": 1024})
    assert _arg(plan, "--moe-cache-headroom-bytes") == str(two_gib)

    # An explicit 0 headroom is not "positive": the cushion still applies while the governor is on.
    plan = _plan({"MemoryGovernor": True, "GovernorVRAMFreeGB": 1.5, "MoECacheHeadroomBytes": 0})
    assert _arg(plan, "--moe-cache-headroom-bytes") == cushion_1_5

    # The reserve dial is passed through untouched whatever the governor does.
    plan = _plan({"MemoryGovernor": True, "MoEVramReserveBytes": 0})
    assert _arg(plan, "--moe-vram-reserve-bytes") == "0"
    plan = _plan({"MemoryGovernor": True, "MoEVramReserveBytes": two_gib})
    assert _arg(plan, "--moe-vram-reserve-bytes") == str(two_gib)

    # Governor off: today's behaviour exactly (auto headroom omitted, explicit values as typed).
    plan = _plan({"MemoryGovernor": False})
    assert "--moe-cache-headroom-bytes" not in plan.argv and "--moe-vram-reserve-bytes" not in plan.argv
    plan = _plan({"MemoryGovernor": False, "MoECacheHeadroomBytes": 0, "MoEVramReserveBytes": 0})
    assert _arg(plan, "--moe-cache-headroom-bytes") == "0" and _arg(plan, "--moe-vram-reserve-bytes") == "0"

    # A dense model gets neither flag.
    plan = build_launch({"ModelPath": "/models/demo", "MemoryGovernor": True}, python="py", base_env={},
                        facts=_facts(is_moe=False, expert_count=0, model_type="dense"), wsl=False)
    assert "--moe-cache-headroom-bytes" not in plan.argv


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

    # Estimate with the governor on: the child gets the cushion as headroom, the same flag the
    # boot gets, so memory_plan counts it (fixed_with_post = ... + policy_reserve + headroom).
    service.estimate_settings(
        {
            "ModelPath": str(model_dir),
            "MemoryGovernor": True,
            "GovernorVRAMFreeGB": 1.5,
        },
        boot_file=BootFile(boot_path),
        environ={"PATH": "/bin"},
    )

    assert len(child_requests) == 1
    child_argv = child_requests[0]["argv"]
    expected_cushion = str(int(round(1.5 * (1024 ** 3))))
    assert "--moe-vram-reserve-bytes" not in child_argv
    cushion_idx = child_argv.index("--moe-cache-headroom-bytes") + 1
    assert child_argv[cushion_idx] == expected_cushion

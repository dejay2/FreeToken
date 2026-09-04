from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "start-freetoken-windows.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


def _run_dry_run(model: dict, model_path: Path, *arguments: str) -> str:
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(model), encoding="utf-8")
    result = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(SCRIPT),
            "-ModelPath",
            str(model_path),
            "-DryRun",
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_dense_models_drop_all_expert_only_memory_options(tmp_path):
    text = SCRIPT.read_text(encoding="utf-8")
    assert "if ($isMoe -and $MoEVramReserveBytes -ge 0)" in text
    assert "if ($isMoe -and $MoECacheHeadroomBytes -ge 0)" in text

    if POWERSHELL is None:
        pytest.skip("Windows PowerShell is not available on this box")
    output = _run_dry_run(
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 12,
            "max_position_embeddings": 131072,
        },
        tmp_path / "dense",
        "-MoEVramReserveBytes",
        "1",
        "-MoECacheHeadroomBytes",
        "2",
        "-GpuOwnedLayers",
        "3",
        "-MoECacheSize",
        "4188",
    )
    command = output.split("DRY RUN: the engine command would be", 1)[1]
    assert "--moe-vram-reserve-bytes" not in command
    assert "--moe-cache-headroom-bytes" not in command
    assert "--moe-gpu-owned-layers" not in command
    assert "--moe-cache-size" not in command
    assert "Expert-slot settings ignored" in output


def test_moe_cache_size_is_clamped_to_the_model_expert_count(tmp_path):
    if POWERSHELL is None:
        pytest.skip("Windows PowerShell is not available on this box")
    output = _run_dry_run(
        {
            "architectures": ["GptOssForCausalLM"],
            "model_type": "gpt_oss",
            "num_hidden_layers": 24,
            "num_local_experts": 32,
            "max_position_embeddings": 131072,
        },
        tmp_path / "gpt-oss",
        "-MoECacheSize",
        "4188",
    )
    command = output.split("DRY RUN: the engine command would be", 1)[1]
    assert "--moe-cache-size 768" in command
    assert "--moe-cache-size 4188" not in command
    assert "clamped from 4188 to 768" in output

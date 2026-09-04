from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from freetoken.daemon.settings.boot_parser import BootFile, BootParseError


REAL_BOOT = Path(__file__).parents[2] / "boot-2020.ps1"


def copy_boot(tmp_path: Path) -> Path:
    target = tmp_path / "boot-2020.ps1"
    shutil.copy2(REAL_BOOT, target)
    return target


def test_real_boot_round_trips_and_preserves_comments_and_expressions(tmp_path):
    path = copy_boot(tmp_path)
    original = path.read_text(encoding="utf-8")
    boot = BootFile(path)

    parsed = boot.load()
    assert parsed["ModelPath"] == r"D:\Models\Qwen3.8-Flash-Next-NVFP4"
    assert parsed["DesktopPython"].startswith("(Join-Path $env:LOCALAPPDATA")
    assert parsed["EmbedHost"] is True
    assert parsed["FREETOKEN_MTP_SPECULATE"] == "0"
    assert "Optional FP8 candidate" in original
    assert "-KVDtype fp8" in original

    result = boot.save({"CudaGraphMaxBS": 3, "EnableCacheReport": False})
    updated = path.read_text(encoding="utf-8")
    assert result["CudaGraphMaxBS"] == 3
    assert "-CudaGraphMaxBS 3" in updated
    assert "-EnableCacheReport" not in updated.split("# Optional parking", 1)[0]
    assert "Optional FP8 candidate" in updated
    assert "-KVDtype fp8" in updated
    assert "(Join-Path $env:LOCALAPPDATA 'FreeToken\\venv\\Scripts\\python.exe')" in updated
    assert (tmp_path / "boot-2020.ps1.bak").read_text(encoding="utf-8") == original
    assert BootFile(path).load()["CudaGraphMaxBS"] == 3


def test_save_rejects_invalid_boot_without_rewriting(tmp_path):
    path = copy_boot(tmp_path)
    path.write_text("$env:FREETOKEN_MTP_SPECULATE = '0'\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(BootParseError, match="launcher"):
        BootFile(path).save({"Port": 2021})
    assert path.read_bytes() == before
    assert not (tmp_path / "boot-2020.ps1.bak").exists()


def test_boot_file_can_enable_commented_candidate_without_moving_comments(tmp_path):
    path = copy_boot(tmp_path)
    boot = BootFile(path)
    boot.save({"KVPark": "ssd"})
    text = path.read_text(encoding="utf-8")
    active = text.split("# Optional parking", 1)[0]
    assert "-KVPark ssd" in active
    assert "#    -KVPark ssd" in text
    assert "# Optional parking candidate" in text

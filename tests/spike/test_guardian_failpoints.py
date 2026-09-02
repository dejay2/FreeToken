from __future__ import annotations

import os
from pathlib import Path

import pytest

_PRIVATE_ROOT = os.environ.get("FREETOKEN_MTP_PRIVATE_ROOT", "").strip()
ROOT = Path(_PRIVATE_ROOT) / "prototypes" if _PRIVATE_ROOT else None
pytestmark = pytest.mark.skipif(
    ROOT is None or not ROOT.is_dir(),
    reason="needs FREETOKEN_MTP_PRIVATE_ROOT pointing at the private MTP spike root",
)


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_guarded_matrix_uses_separate_exact_port_ranges_and_unconditional_restore():
    script = _text("run_guarded_live_mtp_matrix.ps1")

    assert "$candidatePort = 2030" in script
    assert "$acceptedPort = 2020" in script
    assert "-ge 2030 -and $_.LocalPort -le 2035" in script
    assert "-ge 2020 -and $_.LocalPort -le 2025" in script
    assert "'-Port', '2030'" in script
    assert "http://127.0.0.1:2030/health" in script
    assert "http://127.0.0.1:2020/health" in script
    assert "finally" in script
    assert "New-Item -ItemType File -Path $signalPath -Force" in script
    assert "FREETOKEN_MTP_VERIFY_MODE" in script


def test_all_declared_failpoints_still_flow_through_guardian_restoration():
    script = _text("run_guarded_live_mtp_matrix.ps1")
    required = {
        "after-accepted-stop",
        "candidate-startup",
        "correctness-gate",
        "mid-matrix",
        "summary-failure",
    }

    for failpoint in required:
        assert f"'{failpoint}'" in script
        marker = f"Invoke-Failpoint '{failpoint}'"
        assert marker in script
        assert script.index(marker) < script.index("finally")


def test_matrix_runs_correctness_gate_before_any_phase():
    script = _text("run_guarded_live_mtp_matrix.ps1")

    assert "[switch]$ValidateOnly" in script
    assert "run_one_mtp_repair_correctness.py" in script
    gate = script.index("$correctnessGate")
    matrix = script.index("foreach ($placement in $Placements)")
    assert gate < matrix
    assert script.index("if ($ValidateOnly)") < script.index("$guardian = Start-Process")


def test_correctness_gate_failpoint_restores_without_matrix():
    script = _text("run_guarded_live_mtp_matrix.ps1")

    marker = "Invoke-Failpoint 'correctness-gate'"
    assert marker in script
    assert script.index(marker) < script.index("foreach ($placement in $Placements)")
    assert script.index(marker) < script.index("finally")
    assert "New-Item -ItemType File -Path $signalPath -Force" in script


def test_guardian_cleans_candidate_then_restores_only_accepted_mtp_off_service():
    guardian = _text("restore_accepted_server_guardian.ps1")

    assert "-ge 2030 -and $_.LocalPort -le 2035" in guardian
    assert "-ge 2020 -and $_.LocalPort -le 2025" in guardian
    assert "Remove-Item Env:FREETOKEN_MTP_VERIFY_MODE" in guardian
    assert "listener = '127.0.0.1:2020'" in guardian
    assert "mtp_shadow = $false" in guardian
    assert "source = $acceptedRoot" in guardian


def test_state_diagnostic_wrapper_arms_guardian_before_downtime_and_always_restores():
    script = _text("run_guarded_mtp_state_diagnostic.ps1")

    assert "[switch]$ValidateOnly" in script
    assert "$candidatePort = 2030" in script
    assert "$acceptedPort = 2020" in script
    assert "-ge 2030 -and $_.LocalPort -le 2035" in script
    assert "-ge 2020 -and $_.LocalPort -le 2025" in script
    assert "http://127.0.0.1:2030/health" in script
    assert "http://127.0.0.1:2020/health" in script
    assert "$env:FREETOKEN_MTP_EXPERT_FORMAT = 'bf16'" in script
    assert "$env:FREETOKEN_MTP_VERIFY_MODE = 'compare'" in script
    assert "MTP verifier changed live target state families:" in script
    assert "NO_REPRODUCTION" in script
    assert "finally" in script
    assert "New-Item -ItemType File -Path $signalPath -Force" in script
    guardian_start = script.index("$guardian = Start-Process")
    downtime = script.index("    Stop-AcceptedTrees", guardian_start)
    assert guardian_start < downtime
    assert script.index("if ($ValidateOnly)") < guardian_start

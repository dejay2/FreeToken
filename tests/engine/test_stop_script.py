"""scripts/stop-qwen38-flash-next-windows.ps1: the selection helpers.

Live runs left ``python.exe ... spawn_main`` orphans holding the ZMQ port, which broke the
next boot with ``ZMQError: Address in use (tcp://127.0.0.1:2033)``. The script that clears
them decides what to kill from a process snapshot; the decision is what is tested here, by
dot-sourcing the script (``-DotSourceOnly`` returns before it touches a real process) and
handing its helpers a synthetic snapshot. Nothing here starts, signals or enumerates a
process, and nothing here needs a GPU.

Test style follows tests/engine/test_expert_load_flag.py: the script is also read as text,
so its PowerShell-5.1 constraints (no ``&&``, no ternary) are pinned without running it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "stop-qwen38-flash-next-windows.ps1"
)
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(
    POWERSHELL is None, reason="Windows PowerShell is not available on this box"
)

# One synthetic snapshot, in the shape Get-CimInstance Win32_Process returns.
#   2000 a FreeToken server on port 2020            -> kill
#   2001 its spawned child (parent alive)           -> kill, as a descendant
#   2002 a grandchild                               -> kill, as a descendant
#   3000 a FreeToken server on port 2030            -> kill only when no port is named
#   4000 an orphaned spawn_main (parent 999 is gone) -> kill, always
#   5000 ft.exe, the Desktop daemon                  -> NEVER
#   6000 an unrelated python                         -> NEVER
#   7000 a spawn_main whose parent IS alive          -> only via its parent's descendants
SNAPSHOT = [
    {"ProcessId": 2000, "ParentProcessId": 1500, "Name": "python.exe",
     "CommandLine": "python.exe -m freetoken.cli serve --model D:\\Models\\Q --port 2020"},
    {"ProcessId": 2001, "ParentProcessId": 2000, "Name": "python.exe",
     "CommandLine": "python.exe -c from multiprocessing.spawn import spawn_main; spawn_main(...)"},
    {"ProcessId": 2002, "ParentProcessId": 2001, "Name": "python.exe",
     "CommandLine": "python.exe -c worker"},
    {"ProcessId": 3000, "ParentProcessId": 1500, "Name": "python.exe",
     "CommandLine": "python.exe -m freetoken.cli serve --model D:\\Models\\Q --port 2030"},
    {"ProcessId": 4000, "ParentProcessId": 999, "Name": "python.exe",
     "CommandLine": "python.exe -c from multiprocessing.spawn import spawn_main; spawn_main(...)"},
    {"ProcessId": 5000, "ParentProcessId": 1, "Name": "ft.exe",
     "CommandLine": "ft.exe --daemon --port 1900 -m freetoken.cli serve"},
    {"ProcessId": 6000, "ParentProcessId": 1, "Name": "python.exe",
     "CommandLine": "python.exe -m http.server 8000"},
    {"ProcessId": 7000, "ParentProcessId": 6000, "Name": "python.exe",
     "CommandLine": "python.exe -c from multiprocessing.spawn import spawn_main; spawn_main(...)"},
]


def _run(body: str) -> str:
    """Dot-source the script and run ``body`` against it, returning its stdout."""
    snapshot = json.dumps(SNAPSHOT).replace("'", "''")
    script = (
        "$ErrorActionPreference = 'Stop'; "
        f". '{SCRIPT}' -DotSourceOnly; "
        f"$Snapshot = ConvertFrom-Json '{snapshot}'; "
        + body
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _ids(body: str) -> list[int]:
    out = _run(body + " | Sort-Object | ForEach-Object { Write-Output $_ }")
    return [int(line) for line in out.splitlines() if line.strip()]


def test_the_script_exists_and_dot_sources_without_touching_a_process():
    assert SCRIPT.is_file()
    assert _run("Write-Output 'sourced'") == "sourced"


def test_a_named_port_selects_that_server_its_descendants_and_every_orphan():
    assert _ids("Select-FreeTokenKillSet -Processes $Snapshot -Port 2020") == [
        2000, 2001, 2002, 4000
    ]


def test_no_port_selects_every_freetoken_server_and_its_tree():
    assert _ids("Select-FreeTokenKillSet -Processes $Snapshot -Port 0") == [
        2000, 2001, 2002, 3000, 4000
    ]


def test_the_desktop_daemon_and_unrelated_python_are_never_selected():
    """ft.exe holds port 1900 and its command line even mentions freetoken.cli serve."""
    for port in (0, 2020, 2030):
        selected = _ids(f"Select-FreeTokenKillSet -Processes $Snapshot -Port {port}")
        assert 5000 not in selected  # ft.exe, the Desktop daemon
        assert 6000 not in selected  # someone else's python
        assert 7000 not in selected  # a spawn_main whose parent is alive and unrelated


def test_the_serve_matcher_keys_on_the_module_the_port_and_the_process_name():
    def match(name: str, cmdline: str, port: int) -> bool:
        cmdline = cmdline.replace("'", "''")
        out = _run(
            f"Test-FreeTokenServeProcess -Name '{name}' -CommandLine '{cmdline}' -Port {port}"
        )
        return out == "True"

    serve = "python.exe -m freetoken.cli serve --model X --port 2020"
    assert match("python.exe", serve, 2020)
    assert match("pythonw.exe", serve, 2020)
    assert match("python.exe", serve, 0)  # 0 = any port
    assert not match("python.exe", serve, 2030)  # a different server
    assert not match("ft.exe", serve, 2020)  # never a non-python process
    assert not match("python.exe", "python.exe -m freetoken.cli bench", 0)
    assert not match("python.exe", "python.exe -m http.server 2020", 2020)
    # --port=2020 is the same request as --port 2020
    assert match("python.exe", "python.exe -m freetoken.cli serve --port=2020", 2020)
    # and 2020 must not match 20200
    assert not match("python.exe", "python.exe -m freetoken.cli serve --port 20200", 2020)


def test_an_orphan_is_a_spawn_main_python_whose_parent_is_gone():
    def orphan(pid: int) -> bool:
        return pid in _ids("Get-FreeTokenOrphanIds -Processes $Snapshot")

    assert orphan(4000)  # parent 999 is not in the snapshot
    assert not orphan(2001)  # parent 2000 is alive
    assert not orphan(7000)  # parent 6000 is alive
    assert not orphan(6000)  # not a spawn_main at all


def test_the_parent_walk_collects_every_generation():
    assert _ids("Get-FreeTokenDescendantIds -Processes $Snapshot -Id 2000") == [2001, 2002]
    assert _ids("Get-FreeTokenDescendantIds -Processes $Snapshot -Id 2001") == [2002]
    assert _ids("Get-FreeTokenDescendantIds -Processes $Snapshot -Id 3000") == []
    # a cycle in the snapshot (a recycled pid) must not hang the walk
    cyclic = "$c = @([pscustomobject]@{ProcessId=1;ParentProcessId=2;Name='python.exe';CommandLine='x'},[pscustomobject]@{ProcessId=2;ParentProcessId=1;Name='python.exe';CommandLine='x'}); "
    assert _ids(cyclic + "Get-FreeTokenDescendantIds -Processes $c -Id 1") == [2]


def test_the_ports_waited_on_cover_the_zmq_side_range():
    out = _run("(Get-FreeTokenWatchPorts -Port 2020) -join ','")
    assert out == ",".join(str(p) for p in range(2020, 2030))
    assert _run("(Get-FreeTokenWatchPorts -Port 0).Count") == "0"


def test_the_script_is_powershell_51_compatible_and_kills_nothing_by_name():
    text = SCRIPT.read_text(encoding="utf-8")
    # the operators, in CODE: the comments name them precisely because they are banned
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    for banned in ("&&", "||", "??", " ? "):
        assert banned not in code, banned
    # never kill by image name -- that is how ft.exe dies
    assert "taskkill" not in text.lower()
    assert "Stop-Process -Name" not in text
    # the process-name allowlist, and the daemon it exists to protect
    assert "python.exe" in text and "pythonw.exe" in text
    assert "ft.exe" in text


def test_the_windows_guide_documents_the_stop_script():
    guide = (
        Path(__file__).parents[2] / "docs" / "windows-qwen38-flash-next-mmap.md"
    ).read_text(encoding="utf-8")
    assert "stop-qwen38-flash-next-windows.ps1" in guide
    assert "spawn_main" in guide
    assert "Address in use" in guide

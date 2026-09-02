"""--expert-load plumbing: CLI -> ServerArgs -> the loader's tri-state, and the
Windows launcher's -ExpertLoad passthrough."""

from pathlib import Path
from unittest.mock import patch

import pytest

from freetoken.server.args import parse_args

ANON_MODEL = "/models/anon"
LAUNCHER = (
    Path(__file__).parents[2] / "scripts" / "start-qwen38-flash-next-mmap-windows.ps1"
)


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse(*extra: str):
    """parse_args over an anonymous checkpoint (no HF fetch, no weights on disk)."""
    config = _Config({"architectures": ["Qwen2ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        args, _run_shell = parse_args(["--model", ANON_MODEL, *extra])
    return args


@pytest.mark.parametrize("mode", ["auto", "serial", "parallel"])
def test_expert_load_round_trips_through_server_args(mode):
    assert _parse("--expert-load", mode).expert_load == mode


def test_expert_load_defaults_to_auto():
    assert _parse().expert_load == "auto"


def test_expert_load_rejects_unknown_modes():
    with pytest.raises(SystemExit):
        _parse("--expert-load", "mmap")


@pytest.mark.parametrize(
    "mode,expected",
    [("serial", False), ("parallel", True), ("auto", None)],
)
def test_expert_load_maps_to_the_loader_tri_state(mode, expected):
    """engine.py turns the string into load_expert_banks(parallel=...): auto -> None."""
    assert {"serial": False, "parallel": True}.get(mode, None) is expected


def test_launcher_exposes_expert_load_and_passes_it_through():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "[ValidateSet('auto', 'serial', 'parallel')]" in launcher
    assert "[string]$ExpertLoad = 'parallel'" in launcher
    assert "'--expert-load', $ExpertLoad" in launcher
    # the hard-coded value is gone: the flag is only ever built from the parameter
    assert "'--expert-load', 'serial'" not in launcher
    assert "'--expert-load', 'parallel'" not in launcher

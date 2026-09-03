from pathlib import Path

from freetoken.engine.config import EngineConfig


_LAUNCHER = (
    Path(__file__).parents[2]
    / "scripts"
    / "start-qwen38-flash-next-mmap-windows.ps1"
)


def test_parking_engine_defaults_are_off_and_bounded():
    fields = EngineConfig.__dataclass_fields__
    assert fields["kv_park"].default == "off"
    assert fields["kv_park_idle_ms"].default == 0
    assert fields["kv_park_min_tokens"].default == 8192
    assert fields["kv_park_ram_gib"].default == 2.0
    assert fields["kv_park_ssd_dir"].default == "~/.cache/freetoken/kv-park"
    assert fields["kv_park_ssd_gib"].default == 32.0
    assert fields["kv_park_window_mib"].default == 256


def test_launcher_explicit_off_overrides_an_inherited_parking_environment():
    text = _LAUNCHER.read_text(encoding="utf-8")

    assert "$serveArgs += @('--kv-park', $KVPark)" in text
    mode_line = text.index("$serveArgs += @('--kv-park', $KVPark)")
    options_guard = text.index("if ($KVPark -ne 'off')", mode_line)
    assert mode_line < options_guard

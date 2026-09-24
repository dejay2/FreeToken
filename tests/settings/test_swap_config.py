from __future__ import annotations

import copy
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from freetoken.daemon.settings.ninfer_dials import clamp_params
from freetoken.daemon.settings.swap_config import (
    HEADER, SwapConfigWriter, SwitcherRefused, config_sha256, extract_model_blocks, render_config,
)
from tests.settings.registry_fixtures import five

REPO = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).with_name("golden") / "config_five_models.yaml"


def test_five_models_render_the_golden_file():
    assert render_config(five(), {}) == GOLDEN.read_text(encoding="utf-8")


def test_same_registry_same_bytes():
    assert render_config(five()) == render_config(copy.deepcopy(five()))
    assert render_config(five()).splitlines()[0] == HEADER


def test_clamp_params_follow_the_catalogue():
    parsed = yaml.safe_load(render_config(five()))
    assert parsed["models"]["quasar-27b"]["filters"]["clampParams"] == clamp_params()
    assert "filters" not in parsed["models"]["qwen3.8-flash"]


def test_quotes_hashes_and_colons_in_names_stay_valid_yaml():
    doc = five()
    doc["models"][2]["name"] = 'QUASAR "fast": #1 \\ path'
    doc["models"][2]["aliases"] = ["org/quasar:q4"]
    parsed = yaml.safe_load(render_config(doc))
    assert parsed["models"]["quasar-27b"]["name"] == 'QUASAR "fast": #1 \\ path'
    assert parsed["models"]["quasar-27b"]["aliases"] == ["org/quasar:q4"]


def test_paths_with_spaces_are_quoted_for_the_command_line():
    doc = five()
    doc["models"][2]["artifact"] = "~/my models/q.ninfer"
    cmd = yaml.safe_load(render_config(doc))["models"]["quasar-27b"]["cmd"]
    assert "'${env.HOME}/my models/q.ninfer'" in cmd
    assert shlex.split(cmd.replace("${env.HOME}", "/home/jay"))[1] == "/home/jay/my models/q.ninfer"


def test_guess_ahead_off_drops_its_flags():
    doc = five()
    doc["models"][2]["overrides"]["spec"] = "off"
    cmd = yaml.safe_load(render_config(doc))["models"]["quasar-27b"]["cmd"]
    assert "--spec" not in cmd and "--draft-tokens" not in cmd and "--lm-head-draft" not in cmd


def test_chat_reuse_off_drops_the_context_cache_flags():
    doc = five()
    doc["engines"]["ninfer"]["defaults"]["host-kv-mib"] = 4096
    doc["models"][2]["overrides"]["no-prefix-reuse"] = True
    cmd = yaml.safe_load(render_config(doc))["models"]["quasar-27b"]["cmd"]
    assert "--no-prefix-reuse" in cmd and "--host-kv-mib" not in cmd


def test_idle_minutes_null_uses_the_system_setting():
    doc = five()
    doc["system"]["defaultIdleMinutes"] = 15
    doc["models"][3]["idleMinutes"] = None
    parsed = yaml.safe_load(render_config(doc))
    assert parsed["models"]["fable-27b"]["ttl"] == 900
    assert parsed["models"]["quasar-27b"]["ttl"] == 0


def test_holds_keep_the_old_block_for_that_model_only():
    blocks = extract_model_blocks(render_config(five()))
    doc = five()
    doc["models"][2]["overrides"]["draft-tokens"] = 5
    doc["models"][3]["overrides"]["draft-tokens"] = 3
    held = extract_model_blocks(render_config(doc, {"quasar-27b": blocks["quasar-27b"]}))
    assert held["quasar-27b"] == blocks["quasar-27b"]
    assert "--draft-tokens 3" in held["fable-27b"]


def test_blocks_join_back_into_the_same_file():
    text = render_config(five())
    blocks = extract_model_blocks(text)
    assert list(blocks) == ["qwen3.8-flash", "qwen3.8-flash-abliterated", "quasar-27b", "fable-27b", "twin-27b"]
    head = text.split("  # --- model ", 1)[0]
    assert head + "".join(blocks.values()) == text


def test_files_we_did_not_write_have_no_blocks():
    assert extract_model_blocks("healthCheckTimeout: 600\nmodels: {}\n") == {}
    assert extract_model_blocks("") == {}


def fake_runner(ok=True, message="config is valid: 5 model(s), 0 peer(s)"):
    calls = []

    def run(args, **kwargs):
        calls.append(list(args))
        yaml.safe_load(Path(args[2]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(args, 0 if ok else 1, stdout=message + "\n", stderr="")

    run.calls = calls
    return run


def make_writer(tmp_path, runner):
    binary = tmp_path / "llama-swap"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return SwapConfigWriter(tmp_path / "config.yaml", binary=binary, runner=runner)


def test_write_checks_then_swaps_in_and_skips_identical_text(tmp_path):
    runner = fake_runner()
    writer = make_writer(tmp_path, runner)
    text = render_config(five())
    assert writer.write(text) is True
    assert writer.current_text() == text
    assert runner.calls[0][1:] == ["--config", str(tmp_path / "config.yaml.new"), "--check-config"]
    assert not (tmp_path / "config.yaml.new").exists()
    assert writer.write(text) is False
    assert len(runner.calls) == 1
    assert config_sha256(text) == config_sha256((tmp_path / "config.yaml").read_text())


def test_failed_check_keeps_the_old_file(tmp_path):
    writer = make_writer(tmp_path, fake_runner())
    writer.write("old: 1\n")
    writer._runner = fake_runner(ok=False, message="config validation failed: models.quasar-27b: bad")
    with pytest.raises(SwitcherRefused, match="bad"):
        writer.write(render_config(five()))
    assert writer.current_text() == "old: 1\n"
    assert not (tmp_path / "config.yaml.new").exists()


def test_missing_switcher_program_refuses(tmp_path):
    writer = SwapConfigWriter(tmp_path / "config.yaml", binary=tmp_path / "nope")
    with pytest.raises(SwitcherRefused, match="not found"):
        writer.write(render_config(five()))


def test_backup_before_registry_is_made_once(tmp_path):
    writer = make_writer(tmp_path, fake_runner())
    (tmp_path / "config.yaml").write_text("hand: made\n")
    first = writer.backup_before_registry()
    assert first == tmp_path / "config.yaml.bak-before-registry" and first.read_text() == "hand: made\n"
    (tmp_path / "config.yaml").write_text("second: file\n")
    assert writer.backup_before_registry() is None
    assert first.read_text() == "hand: made\n"


def test_real_switcher_accepts_the_golden_file(tmp_path):
    binary = REPO / "engines" / "llama-swap" / "build" / "llama-swap"
    if not binary.is_file():
        pytest.skip("build the switcher first: cd engines/llama-swap && ~/.local/go/bin/go build -o build/llama-swap .")
    SwapConfigWriter(tmp_path / "config.yaml", binary=binary).write(GOLDEN.read_text(encoding="utf-8"))

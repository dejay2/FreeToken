"""The NInfer catalogue against the frozen runtimes' own option parser."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from freetoken.daemon.settings.dials import DIALS
from freetoken.daemon.settings.ninfer_dials import (
    BY_NAME, FIXED_FLAGS, GROUP_INFO, RUNTIMES, clamp_params, dial_dicts, normalized,
    parse_flags, render_flags, settings_for, validate,
)

REPO = Path(__file__).resolve().parents[2]
SOURCES = {rt: REPO / "engines" / rt / "src" / "serve" / "serve_options.cpp" for rt in RUNTIMES}


def source_flags(runtime: str) -> set[str]:
    text = SOURCES[runtime].read_text(encoding="utf-8")
    return set(re.findall(r'arg == "(--[a-z0-9-]+)"', text)) - set(FIXED_FLAGS)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_catalogue_matches_each_frozen_runtime_source(runtime):
    ours = {setting.flag for setting in settings_for(runtime)}
    assert ours == source_flags(runtime)


def test_chat_template_is_upstream_only():
    assert BY_NAME["chat-template"].runtimes == ("ninfer-upstream",)
    assert "--chat-template" not in source_flags("ninfer")


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_every_flag_is_in_the_built_runtime_help(runtime):
    binary = REPO / "engines" / runtime / "build" / "apps" / "ninfer-serve"
    if not binary.is_file():
        pytest.skip(f"{runtime} is not built here (serving box only: scripts/engines/build.sh ninfer)")
    run = subprocess.run([str(binary), "--help"], capture_output=True, text=True, timeout=30)
    text = run.stdout + run.stderr
    missing = [setting.flag for setting in settings_for(runtime) if setting.flag not in text]
    assert not missing, f"{runtime} --help lacks {missing}"


def test_answer_style_and_limit_ranges_match_the_runtime_parser():
    text = SOURCES["ninfer"].read_text(encoding="utf-8")
    pairs = re.findall(
        r'"(temperature|top-p|min-p|presence-penalty|frequency-penalty)", (-?[\d.]+)f, (-?[\d.]+)f', text
    )
    assert len(pairs) == 5
    for name, low, high in pairs:
        dial = BY_NAME[name].dial
        assert (dial.minimum, dial.maximum) == (float(low), float(high)), name
    assert "top_k > 20" in text and BY_NAME["top-k"].dial.maximum == 20
    assert "kMaximumConcurrency" in text and BY_NAME["max-concurrency"].dial.maximum == 8
    assert "threads > 64" in text and BY_NAME["media-preprocess-threads"].dial.maximum == 64
    assert "% 128" in text and BY_NAME["prefill-chunk"].step == 128


def test_clamp_params_come_from_the_answer_style_ranges():
    assert clamp_params() == {
        "temperature": [0, 2], "top_p": [0, 1], "top_k": [0, 20], "min_p": [0, 1],
        "presence_penalty": [-2, 2], "frequency_penalty": [-2, 2],
    }


def test_cross_field_rules_speak_plainly():
    def messages(settings, runtime=None):
        return {error["field"]: error["message"] for error in validate(settings, runtime)}

    assert "at least the longest chat" in messages({"max-context": 150000, "kv-capacity": 100000})["kv-capacity"]
    assert validate({"max-context": 150000, "kv-capacity": 0}) == []  # 0 = fill the card
    assert "1 to 5" in messages({"spec": "mtp", "draft-tokens": 6})["draft-tokens"]
    assert "needs" in messages({"spec": "dflash2"})["draft-tokens"]
    assert "multiple of 128" in messages({"prefill-chunk": 1000})["prefill-chunk"]
    assert "upstream runtime" in messages({"chat-template": "/t.jinja"}, "ninfer")["chat-template"]
    assert "at most 2" in messages({"temperature": 2.5})["temperature"]
    assert "Unknown" in messages({"bogus": 1})["bogus"]
    assert validate({"draft-tokens": 7}, cross=False) == []


def test_render_and_parse_round_trip():
    settings = {"max-context": 150000, "kv-capacity": 0, "kv-dtype": "int8", "spec": "dflash2",
                "draft-tokens": 7, "lm-head-draft": True, "temperature": 0.6, "vision": True}
    flags = render_flags(settings, "ninfer")
    assert flags[:4] == ["--max-context", "150000", "--kv-capacity", "auto"]
    assert "--max-pending-requests" not in flags  # equal to the runtime's own default
    parsed, fixed = parse_flags(["--host", "127.0.0.1", "--port", "8090", *flags])
    assert fixed == {"--host": "127.0.0.1", "--port": "8090"}
    assert normalized(parsed) == normalized(settings)


def test_guess_ahead_off_leaves_its_flags_out():
    flags = render_flags({"spec": "off", "draft-tokens": 7, "lm-head-draft": True}, "ninfer")
    assert flags == []


def test_page_contract_matches_the_freetoken_dials():
    ours = dial_dicts({}, "ninfer")
    theirs = DIALS[1].as_dict(DIALS[1].default)
    assert set(ours[0]) == set(theirs)
    assert all(row["group"] in GROUP_INFO for row in ours)
    assert "chat-template" not in {row["name"] for row in ours}
    assert "chat-template" in {row["name"] for row in dial_dicts({}, "ninfer-upstream")}

"""Pi sync. Review focus 1 lives here. The fixtures follow the shape of Jay's Pi files
(C:\\Users\\jay\\.pi\\agent\\models.json and settings.json); the real ones are never read here."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from freetoken.daemon.settings.pi_sync import FALLBACK, PROVIDER, PiSync

COST = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
MODELS = {"providers": {
    "anthropic": {"apiKey": "sk-ant-secret", "models": [{"id": "claude-x"}]},
    PROVIDER: {
        "name": "FreeToken (local)", "baseUrl": "http://127.0.0.1:2040/v1", "apiKey": "local",
        "api": "openai-completions", "compat": {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
        "models": [
            {"id": "qwen3.8-flash", "name": "Qwen3.8 Flash (FreeToken)", "reasoning": True, "input": ["text", "image"],
             "contextWindow": 262144, "maxTokens": 32768, "cost": COST, "samplingParams": {"temperature": 0.6},
             "thinkingLevelMap": {"off": "none", "high": "high"}},
            {"id": "quasar-27b", "name": "QUASAR 27B", "reasoning": True, "input": ["text", "image"],
             "contextWindow": 150000, "maxTokens": 16384, "cost": COST,
             "samplingParams": {"temperature": 0.7, "top_p": 0.95},
             "thinkingLevelMap": {"off": "none", "low": "low", "high": "high"}},
        ]},
    "openrouter": {"baseUrl": "https://openrouter.ai/api/v1", "apiKey": "sk-or-secret", "models": [{"id": "x/y"}]},
}}
SETTINGS = {"defaultProvider": PROVIDER, "defaultModel": "quasar-27b",
            "enabledModels": [f"{PROVIDER}/qwen3.8-flash", f"{PROVIDER}/quasar-27b", "openrouter/x/y"],
            "theme": "dark", "packages": ["pi-subagents"]}
ENGINES = {"qwen3.8-flash": "freetoken", "quasar-27b": "ninfer"}


def write_pi(folder: Path, models=MODELS, settings=SETTINGS) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    (folder / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return folder


class Clock:
    def __init__(self):
        self.t = dt.datetime(2026, 9, 25, 12, 0, 0)

    def __call__(self):
        self.t += dt.timedelta(seconds=1)
        return self.t


def load(folder: Path):
    return (json.loads((folder / "models.json").read_text(encoding="utf-8")),
            json.loads((folder / "settings.json").read_text(encoding="utf-8")))


def backups(folder: Path, name: str) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.startswith(name + ".bak-"))


def test_a_new_model_copies_limits_from_a_same_engine_neighbour(tmp_path):
    folder = write_pi(tmp_path / "agent")
    result = PiSync(folder, now=Clock()).add("small-9b", "Small 9B (NInfer)", "ninfer", ENGINES)
    assert result["status"] == "updated", result
    models, settings = load(folder)
    assert models["providers"][PROVIDER]["models"][-1] == {
        "id": "small-9b", "name": "Small 9B (NInfer)", "reasoning": True, "input": ["text", "image"], "cost": COST,
        "contextWindow": 150000, "maxTokens": 16384, "thinkingLevelMap": {"off": "none", "low": "low", "high": "high"}}
    assert settings["enabledModels"][-1] == f"{PROVIDER}/small-9b"


def test_other_providers_and_settings_are_never_touched(tmp_path):
    folder = write_pi(tmp_path / "agent")
    pi = PiSync(folder, now=Clock())
    pi.add("small-9b", "Small", "ninfer", ENGINES)
    models, settings = load(folder)
    others = lambda doc: {k: v for k, v in doc["providers"].items() if k != PROVIDER}  # noqa: E731
    assert others(models) == others(MODELS)
    assert {k: v for k, v in models["providers"][PROVIDER].items() if k != "models"} == \
        {k: v for k, v in MODELS["providers"][PROVIDER].items() if k != "models"}
    assert {k: v for k, v in settings.items() if k != "enabledModels"} == {k: v for k, v in SETTINGS.items() if k != "enabledModels"}
    assert settings["enabledModels"][:3] == SETTINGS["enabledModels"]
    assert pi.remove("small-9b")["status"] == "updated"
    assert load(folder) == (MODELS, SETTINGS)
    assert (folder / "models.json").read_text(encoding="utf-8").startswith('{\n  "providers"')  # own indent kept


def test_both_files_are_backed_up_before_every_change(tmp_path):
    folder = write_pi(tmp_path / "agent")
    original = {name: (folder / name).read_bytes() for name in ("models.json", "settings.json")}
    pi = PiSync(folder, now=Clock())
    pi.add("small-9b", "Small", "ninfer", ENGINES)
    pi.remove("small-9b")
    for name in ("models.json", "settings.json"):
        names = backups(folder, name)
        assert len(names) == 2, names
        assert (folder / names[0]).read_bytes() == original[name]


def test_no_change_writes_nothing(tmp_path):
    folder = write_pi(tmp_path / "agent")
    pi = PiSync(folder, now=Clock())
    assert pi.add("quasar-27b", "QUASAR", "ninfer", ENGINES)["status"] == "unchanged"
    assert pi.remove("not-there")["status"] == "unchanged"
    assert backups(folder, "models.json") == [] and backups(folder, "settings.json") == []


@pytest.mark.parametrize("damage", ["missing", "bad_json", "no_provider"])
def test_unreachable_or_odd_files_say_pi_not_updated_and_change_nothing(tmp_path, damage):
    folder = tmp_path / "agent"
    if damage != "missing":
        write_pi(folder, models={"providers": {"openrouter": {}}} if damage == "no_provider" else MODELS)
        if damage == "bad_json":
            (folder / "settings.json").write_text("{oops", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in folder.iterdir()} if folder.exists() else {}
    result = PiSync(folder, now=Clock()).add("small-9b", "Small", "ninfer", ENGINES)
    assert result["status"] == "not_updated" and result["message"]
    assert ({p.name: p.read_bytes() for p in folder.iterdir()} if folder.exists() else {}) == before


def test_no_same_engine_neighbour_uses_the_fallback_and_says_so(tmp_path):
    folder = write_pi(tmp_path / "agent")
    result = PiSync(folder, now=Clock()).add("tiny", "Tiny", "ninfer", {"qwen3.8-flash": "freetoken"})
    assert load(folder)[0]["providers"][PROVIDER]["models"][-1] == {"id": "tiny", "name": "Tiny", **FALLBACK}
    assert result["notes"]


def test_removing_pis_default_model_leaves_the_default_and_says_so(tmp_path):
    folder = write_pi(tmp_path / "agent")
    result = PiSync(folder, now=Clock()).remove("quasar-27b")
    models, settings = load(folder)
    assert "quasar-27b" not in [m["id"] for m in models["providers"][PROVIDER]["models"]]
    assert settings["defaultModel"] == "quasar-27b" and f"{PROVIDER}/quasar-27b" not in settings["enabledModels"]
    assert any("default" in note for note in result["notes"])


def test_backups_are_pruned_to_twenty(tmp_path):
    folder = write_pi(tmp_path / "agent")
    pi = PiSync(folder, now=Clock())
    for _ in range(12):
        pi.add("small-9b", "Small", "ninfer", ENGINES)
        pi.remove("small-9b")
    assert len(backups(folder, "models.json")) == 20 and len(backups(folder, "settings.json")) == 20


def test_crlf_and_bom_are_kept(tmp_path):
    folder = write_pi(tmp_path / "agent")
    for name in ("models.json", "settings.json"):
        path = folder / name
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes().replace(b"\n", b"\r\n"))
    assert PiSync(folder, now=Clock()).add("small-9b", "Small", "ninfer", ENGINES)["status"] == "updated"
    raw = (folder / "models.json").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")


def test_a_failed_second_write_puts_the_first_file_back(tmp_path, monkeypatch):
    """models.json was replaced, then settings.json could not be: Pi must not be left half-changed."""
    folder = write_pi(tmp_path / "agent")
    original = {name: (folder / name).read_bytes() for name in ("models.json", "settings.json")}
    real = PiSync._atomic_write.__func__

    def flaky(cls, path, data):
        if path.name == "settings.json":
            (folder / "settings.json.tmp-freetoken").write_bytes(data)  # the temp file got written...
            raise PermissionError(13, "Permission denied")  # ...but the replace did not
        real(cls, path, data)

    monkeypatch.setattr(PiSync, "_atomic_write", classmethod(flaky))
    result = PiSync(folder, now=Clock()).add("small-9b", "Small", "ninfer", ENGINES)
    assert result["status"] == "not_updated" and "Permission denied" in result["message"], result
    assert "left as it was" in result["message"]
    assert (folder / "models.json").read_bytes() == original["models.json"]
    assert (folder / "settings.json").read_bytes() == original["settings.json"]
    assert not list(folder.glob("*.tmp-freetoken"))


def test_bytes_that_are_not_utf8_are_never_rewritten(tmp_path):
    folder = write_pi(tmp_path / "agent")
    path = folder / "settings.json"
    damaged = path.read_bytes().replace(b'"dark"', b'"d\xffrk"')
    path.write_bytes(damaged)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    result = PiSync(folder, now=Clock()).add("small-9b", "Small", "ninfer", ENGINES)
    assert result["status"] == "not_updated" and "UTF-8" in result["message"], result
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert b"\xef\xbf\xbd" not in path.read_bytes()


def test_odd_rows_do_not_raise_and_the_sync_never_raises(tmp_path):
    models = json.loads(json.dumps(MODELS))
    models["providers"][PROVIDER]["models"].insert(0, {"id": ["not", "a", "string"], "name": "odd"})
    models["providers"][PROVIDER]["models"].insert(0, "just a string")
    folder = write_pi(tmp_path / "agent", models=models)
    result = PiSync(folder, now=Clock()).add("small-9b", "Small", "ninfer", ENGINES)
    assert result["status"] == "updated", result
    rows = load(folder)[0]["providers"][PROVIDER]["models"]
    assert rows[0] == "just a string" and rows[1] == {"id": ["not", "a", "string"], "name": "odd"}
    assert rows[-1]["id"] == "small-9b" and rows[-1]["contextWindow"] == 150000  # the real neighbour still found

    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    result = PiSync(folder, now=Clock()).add("tiny", "Tiny", "ninfer", None)  # a broken caller: .get on None
    assert result["status"] == "not_updated" and "AttributeError" in result["message"], result
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_a_single_line_file_stays_single_line(tmp_path):
    folder = write_pi(tmp_path / "agent")
    (folder / "models.json").write_text(json.dumps(MODELS), encoding="utf-8")  # compact, no newline
    (folder / "settings.json").write_text(json.dumps(SETTINGS) + "\n", encoding="utf-8")  # compact, final newline
    assert PiSync(folder, now=Clock()).add("small-9b", "Small", "ninfer", ENGINES)["status"] == "updated"
    models_text = (folder / "models.json").read_text(encoding="utf-8")
    settings_text = (folder / "settings.json").read_text(encoding="utf-8")
    assert "\n" not in models_text and json.loads(models_text)["providers"][PROVIDER]["models"][-1]["id"] == "small-9b"
    assert settings_text.endswith("\n") and "\n" not in settings_text[:-1]


def test_sync_off_says_so(tmp_path):
    result = PiSync(tmp_path, enabled=False).add("x", "X", "ninfer", {})
    assert result == {"status": "not_updated", "message": "Pi sync is off on this helper.", "notes": []}

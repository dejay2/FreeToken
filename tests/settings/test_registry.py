from __future__ import annotations

import datetime as dt
import json
import os
import stat

import pytest

from freetoken.daemon.settings import registry as reg
from freetoken.daemon.settings.registry import (
    RegistryCorrupt, RegistryMissing, RegistryStore, StaleRevision, base_settings, differences,
    effective_settings, find_model, validate_registry,
)
from tests.settings.registry_fixtures import five


def ticking_clock():
    state = {"n": 0}

    def now():
        state["n"] += 1
        return dt.datetime(2026, 9, 24, 12, 0, 0) + dt.timedelta(seconds=state["n"])

    return now


def make_store(tmp_path):
    return RegistryStore(tmp_path / "registry.json", now=ticking_clock())


def test_fixture_is_valid():
    assert validate_registry(five()) == []


def test_missing_file_is_reported_as_missing(tmp_path):
    with pytest.raises(RegistryMissing):
        make_store(tmp_path).load()


def test_save_then_load_round_trips_and_is_private(tmp_path):
    store = make_store(tmp_path)
    revision = store.save(five(), expected_revision=None)
    doc, again = store.load()
    assert revision == again
    assert [m["id"] for m in doc["models"]] == [
        "qwen3.8-flash", "qwen3.8-flash-abliterated", "quasar-27b", "fable-27b", "twin-27b"]
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600


def test_layering_defaults_then_active_preset_then_overrides():
    doc = five()
    quasar = find_model(doc, "quasar-27b")
    quasar["presets"] = {"Fast agents": {"max-concurrency": 6, "kv-dtype": "fp8"}}
    quasar["activePreset"] = "Fast agents"
    effective = effective_settings(doc, quasar)
    assert effective["max-context"] == 150000  # engine default
    assert effective["max-concurrency"] == 6  # the preset beats the default (4)
    assert effective["kv-dtype"] == "int8"  # the model's own change beats the preset
    assert effective["max-pending-requests"] == 16  # the runtime's built-in value
    assert base_settings(doc, quasar)["kv-dtype"] == "fp8"
    assert base_settings(doc, quasar, None)["kv-dtype"] == "bf16"  # no preset


def test_freetoken_effective_settings_carry_the_model_folder():
    doc = five()
    flash = find_model(doc, "qwen3.8-flash")
    effective = effective_settings(doc, flash)
    assert effective["ModelPath"] == os.path.expanduser("~/models/Qwen3.8-Flash-Next-NVFP4")
    assert effective["KVDtype"] == "fp8"
    assert effective["KVDynamic"] is True  # a catalogue default under the imported ones


def test_differences_keep_only_changed_values():
    assert differences({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 5, "c": 3}) == {"b": 2}


def test_unchanged_save_writes_no_backup(tmp_path):
    store = make_store(tmp_path)
    revision = store.save(five(), expected_revision=None)
    assert store.save(five(), expected_revision=revision) == revision
    assert store.backups() == []


def test_stale_revision_is_refused(tmp_path):
    store = make_store(tmp_path)
    first = store.save(five(), expected_revision=None)
    doc = five()
    doc["system"]["floorGB"] = 7
    store.save(doc, expected_revision=first)
    with pytest.raises(StaleRevision):
        store.save(five(), expected_revision=first)


def test_hand_edited_invalid_json_is_corrupt_and_lists_backups(tmp_path):
    store = make_store(tmp_path)
    first = store.save(five(), expected_revision=None)
    doc = five()
    doc["system"]["floorGB"] = 7
    store.save(doc, expected_revision=first)
    store.path.write_text('{"version": 1, "models": [', encoding="utf-8")
    with pytest.raises(RegistryCorrupt) as caught:
        store.load()
    assert "not valid JSON" in str(caught.value)
    assert len(caught.value.backups) == 1
    with pytest.raises(StaleRevision):
        store.save(five(), expected_revision=first)


def test_valid_json_with_bad_content_is_corrupt_too(tmp_path):
    store = make_store(tmp_path)
    store.path.write_text(json.dumps({"version": 1, "models": "nope"}), encoding="utf-8")
    with pytest.raises(RegistryCorrupt) as caught:
        store.load()
    assert "problems" in str(caught.value)


def test_restore_brings_back_a_backup_and_keeps_the_broken_file(tmp_path):
    store = make_store(tmp_path)
    first = store.save(five(), expected_revision=None)
    doc = five()
    doc["system"]["floorGB"] = 8
    store.save(doc, expected_revision=first)
    store.path.write_text("{broken", encoding="utf-8")
    store.restore(store.backups()[0])
    restored, _ = store.load()
    assert restored["system"]["floorGB"] == 6
    # The broken file is kept, but outside the backup list, so "restore the newest backup"
    # never brings the damage back (final review, open item).
    assert not any((tmp_path / name).read_text() == "{broken" for name in store.backups())
    kept = [p for p in tmp_path.iterdir() if p.name.startswith("registry.json.corrupt-")]
    assert len(kept) == 1 and kept[0].read_text() == "{broken"
    store.restore(store.backups()[0])  # a good current file still becomes a normal backup
    assert not any(p.name.startswith("registry.json.corrupt-") and p != kept[0] for p in tmp_path.iterdir())
    with pytest.raises(KeyError):
        store.restore("registry.json.bak-nope")


def test_backups_keep_the_newest_20(tmp_path):
    store = make_store(tmp_path)
    revision = store.save(five(), expected_revision=None)
    for index in range(25):
        doc = five()
        doc["system"]["waitSeconds"] = 100 + index
        revision = store.save(doc, expected_revision=revision)
    assert len(store.backups()) == reg.BACKUPS_KEPT == 20
    oldest_kept = json.loads((tmp_path / store.backups()[-1]).read_text())
    assert oldest_kept["system"]["waitSeconds"] == 104


@pytest.mark.parametrize("bad_id", ["Quasar", "quasar 27b", "a/b", "a:b", "../x", "-x", "", "ü-model",
                                    "x" * 64, "x\ny", 'a"b', "a#b"])
def test_unsafe_model_ids_are_refused(bad_id):
    doc = five()
    doc["models"][2]["id"] = bad_id
    errors = validate_registry(doc)
    assert any(e["field"] == "models[2].id" and "small letters" in e["message"] for e in errors), errors


def test_duplicate_ids_and_alias_clashes_are_refused():
    doc = five()
    doc["models"][3]["aliases"] = ["quasar-27b"]
    assert any("already used" in e["message"] for e in validate_registry(doc))
    doc = five()
    doc["models"][4]["id"] = "fable-27b"
    assert any("already used" in e["message"] for e in validate_registry(doc))
    doc = five()
    doc["models"][3]["aliases"] = ["has space"]
    assert any(e["field"] == "model.aliases" for e in validate_registry(doc))


def test_names_may_hold_quotes_but_not_line_breaks():
    doc = five()
    doc["models"][0]["name"] = 'Flash "fast" #1: yes'
    assert validate_registry(doc) == []
    doc["models"][0]["name"] = "two\nlines"
    assert any(e["field"] == "model.name" for e in validate_registry(doc))


def test_model_folder_cannot_be_overridden():
    doc = five()
    doc["models"][0]["overrides"] = {"ModelPath": "/elsewhere"}
    assert any(e["field"] == "ModelPath" for e in validate_registry(doc))


def test_effective_ninfer_settings_are_checked_together():
    doc = five()
    doc["models"][3]["overrides"]["draft-tokens"] = 9  # fable-27b uses MTP: 1..5
    errors = validate_registry(doc)
    assert any(e["field"] == "draft-tokens" and "1 to 5" in e["message"] and e["where"] == "fable-27b"
               for e in errors), errors


def test_schema_file_matches_the_validator():
    schema = json.loads(reg.SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["required"] == ["version", "system", "engines", "models"]
    model = schema["$defs"]["model"]["properties"]
    assert model["id"]["pattern"] == reg.MODEL_ID_RE.pattern
    assert model["aliases"]["items"]["pattern"] == reg.ALIAS_RE.pattern
    assert model["engine"]["enum"] == list(reg.ENGINES)

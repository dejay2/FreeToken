# Control Panel, Stage B: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Jay can add a model to the control panel and remove one, from the page. He adds a
model by browsing to a file or folder on the PC, or by pasting a Hugging Face link that is
downloaded and checked. The page works out which engine (and which NInfer runtime) runs it,
suggests an id, a name and a memory figure, and saves it. Remove asks first, puts a loaded
model away before removing it, and deletes the files only when asked. Pi's model list stays in
step with both.

**Architecture:**
- `daemon/settings/model_detect.py` (new) says what a path is. A `.ninfer` file goes to the
  NInfer engine, and its 8-byte magic picks the runtime: v2 → `ninfer` (QUASAR fork), v3 →
  `ninfer-upstream`. It lists a split v3 model's part files. A folder whose `config.json`
  architecture FreeToken serves goes to FreeToken. Anything else is "not supported by your
  engines". It also suggests the id, name and memory need.
- `daemon/settings/download.py` (modify) gains an "add" download:
  - it plans the download from the repo listing: one `.ninfer` entry plus its parts, or a
    model folder;
  - it downloads into a hidden staging folder beside the target;
  - it checks every file against the repo's `SHA256SUMS` when there is one, otherwise against
    the Hub's published LFS sha256;
  - it moves the files into place without ever overwriting, and deletes the staging folder on
    failure or cancel.
- `daemon/settings/pi_sync.py` (new) edits only the `freetoken-local` provider's `models` list
  in Pi's `models.json` and the `freetoken-local/<id>` rows of `settings.json` `enabledModels`.
  It backs up both files before every change. When they are out of reach it says "Pi not
  updated" and the rest carries on.
- `daemon/settings/panel.py` (modify) adds detect, plan/download, add and remove on the
  Stage A save path (`PanelService._save`). There are two small `_save` changes: a brand-new
  model may be added while the switcher state is unknown, and a removed model's hold is
  dropped.
- `static/panel.js` + `static/index.html` (modify) add the three-step **Add a model** wizard
  and the **Remove this model…** question with the "Also delete the model files" checkbox
  (off by default).

**Tech Stack:** Python 3.13 (FastAPI, pydantic, pytest, huggingface_hub 1.30), plain browser
JavaScript (tested with node 24), systemd --user in WSL `vllm`.

**Spec:** `docs/superpowers/specs/2026-09-24-control-panel-part2-design.md`, section 7
("Stage B: add and remove"), plus its Error handling and Testing sections.
**Builds on:** Stage A (`docs/superpowers/plans/2026-09-24-control-panel-stage-a.md`, merged
fedf802).

## Global Constraints

- **Where things go on the box (never tracked):**
  - FreeToken model folders go in `~/models/<repo name>` (env `FREETOKEN_MODELS_DIR` overrides).
  - NInfer files go in `~/ninfer-work/models/` (env `FREETOKEN_NINFER_MODELS_DIR` overrides).
  - A download lands first in the staging folder `<target folder>/.incoming-<job id>/`. The
    dot keeps it out of the page's Browse list.
  - Pi's agent folder is `/mnt/c/Users/<user>/.pi/agent` (env `FREETOKEN_PI_AGENT_DIR`
    overrides; `<user>` is the Linux user, `jay` on the box). The files are `models.json` and
    `settings.json`.
  - Pi backups are `<file>.bak-<YYYYmmdd-HHMMSS-ffffff>`, one of each file before every change.
    The newest **20** of each are kept.
- **Detection (the header decides, never the file name):**
  - v2 magic `NINFER\0\x02`, prefix `<8sQ` → runtime `ninfer`.
  - v3 magic `NINFER\0\x03`, header `<8sQ16s` → runtime `ninfer-upstream`.
  - Part magic `NINPRT\0\x03`; parts are listed in the v3 directory's `files[1:]`.
  - A folder is FreeToken when its `config.json` architecture is in
    `model_info.SUPPORTED_ARCHITECTURES` and it holds `.safetensors` files.
  - The server detects again at Save; the page never chooses the engine.
- **Identity:**
  - The suggested id is the file or folder name, lower-cased, with runs of other characters
    turned into `-`. It is made unique against every id and alias (case-insensitive) with
    `-2`, `-3`… and must match `registry.MODEL_ID_RE`.
  - The suggested name is the name with `_` shown as spaces, plus ` (NInfer)` or ` (FreeToken)`.
  - Suggested memory: for NInfer, ⌈file bytes / GiB⌉ (QUASAR 19,782,132,224 → 19, against 18
    in use). For FreeToken, ⌈expert bytes / GiB⌉ (Qwen3.8 Flash 68,136,468,480 → 64, against
    61-62 measured on 2026-09-24/25). All three stay editable.
  - A new model is stored with `idleMinutes: null`, `aliases: []`, `overrides: {}`,
    `presets: {}` and `activePreset: null`. `artifact` is written as `~/…` when it is under HOME.
- **FreeToken defaults a new model cannot use** become that model's own overrides at its limit:
  a longer chat than it reads, or more expert slots than it has. The save reports each one in
  plain words. Without this, the add would fail. A registry holding such a model would also
  refuse every later save, because `_save` checks every FreeToken model's limits.
- **Checksums:**
  - A repo's `SHA256SUMS` (or `SHA256SUMS.txt`) wins.
  - Otherwise a file is checked against the Hub's LFS `sha256`.
  - Small git files with neither are not checked, and the page says how many files were.
  - A mismatch deletes the download.
- **Never overwrite.** Planning refuses a target that exists. Moving into place uses `os.link`,
  which fails if the file appeared meanwhile, or a checked `rename` for folders.
- **Remove:**
  - The switcher state must be known: an unknown state is refused.
  - A loaded model is unloaded first; a failed unload removes nothing.
  - Then the list is saved and the config regenerated. Then the model's `model-<id>` profile is
    deleted, moving the helper back to its default boot file when that profile was active.
  - Then the files are deleted, only when asked. Then Pi is updated.
- **Deleting files:**
  - It is off by default.
  - It covers only the model's own files: a NInfer entry plus its v3 parts, or a FreeToken
    folder that holds `config.json`.
  - It never touches a path another model's files are, contain or sit inside. It never
    touches HOME or any folder above it.
  - The safety checks run before anything changes.
- **Pi sync:**
  - It never touches another provider, the provider's own fields or other `settings.json` keys.
  - A new entry copies `reasoning`, `input`, `cost`, `contextWindow`, `maxTokens` and
    `thinkingLevelMap` from the first Pi entry of a registry model on the same engine.
    `samplingParams` is per model and is not copied. With no such neighbour it uses
    `pi_sync.FALLBACK` and says so.
  - Removing Pi's default model leaves `defaultModel` alone and says so.
  - Files keep their indent, final newline, CRLF and BOM.
- **Plain words on the page** for Jay (non-technical), as in Stage A. The browser's own
  confirm/alert/prompt are never used.
- **Process rules:**
  - Implementers (subagents) run no git write commands (no
    add/commit/checkout/switch/reset/clean/stash/rm/mv/push). The controller commits.
  - Live tests on the box go only through the controller, after asking Jay. Check
    `curl -s 127.0.0.1:2040/running` first and never evict a model Jay is using.
  - At most one FreeToken boot per live session.
  - Never force-push. Push target `origin` (`dejay2/FreeToken`); `upstream` is fetch-only.
  - Branch `feat/control-panel-b` (from `mtp-upstream-merge` fedf802); the PR targets
    `mtp-upstream-merge` and is squash-merged.
  - Commits are `type: subject` and every message ends with
    `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - Live acceptance captures screenshots of the page with `~/.npm-global/bin/chrome-devtools-axi`
    as proof.
  - Merge gate: a Codex review (Astra, xhigh) with its findings fixed, then Jay's yes.
- **Commands:** from the worktree root,
  `PYTHONPATH=python /home/jay/projects/FreeToken/.venv/bin/python -m pytest <paths> -q`
  (below shortened to `PYTEST <paths>`).

## Review Focus

1. **Pi's files are someone else's.**
   - Another provider, the provider's own fields, and every `settings.json` key other than
     `enabledModels` stay exactly as they were.
   - Both files are backed up before every change.
   - A missing folder, bad JSON or a missing `freetoken-local` provider means "Pi not updated":
     nothing is written, and the add or remove itself still succeeds.
   - Pinned in Task 4 (`test_other_providers_and_settings_are_never_touched`,
     `test_both_files_are_backed_up_before_every_change`,
     `test_unreachable_or_odd_files_say_pi_not_updated_and_change_nothing`) and Task 5
     (`test_pi_out_of_reach_still_adds_and_says_so`).
2. **Removing a model that is, or may be, loaded.**
   - It is unloaded first.
   - An unknown switcher state is refused.
   - A failed unload changes nothing: not the list, the config, the files or Pi.
   - A "next time" hold on it goes with it.
   - Pinned in Task 5 (`test_removing_a_loaded_model_unloads_it_first`,
     `test_remove_is_refused_while_the_switcher_state_is_unknown`,
     `test_a_failed_unload_removes_nothing`, `test_a_removed_model_leaves_no_hold_behind`).
3. **"Also delete the model files" deletes only that model's files.**
   - It is off by default.
   - It takes the entry plus its v3 parts and nothing else.
   - It never deletes files another model reads, including a split model's parts, a folder
     that is not a model, or HOME.
   - It runs only after the list is saved.
   - Pinned in Task 5 (`test_delete_files_takes_the_entry_and_its_parts_and_nothing_else`,
     `test_files_another_model_uses_are_never_deleted`,
     `test_parts_another_model_reads_are_never_deleted`,
     `test_delete_files_refuses_a_folder_that_is_not_a_model`,
     `test_the_home_folder_is_never_deleted`) and Task 6
     (`test_remove_asks_with_delete_files_off_and_cancel_sends_nothing`).
4. **A download that fails, is cancelled or fails its checksum leaves nothing behind.**
   - The staging folder is deleted, no final file appears, and nothing is added to the list.
   - An existing file, or one that appears during the download, is never overwritten.
   - Pinned in Task 3 (`test_sha256sums_wins_and_a_mismatch_deletes_everything`,
     `test_failure_and_cancel_delete_the_partial_files`, `test_nothing_is_ever_overwritten`,
     `test_a_file_that_appears_during_the_download_is_not_overwritten`) and Task 5
     (`test_a_bad_checksum_leaves_nothing_behind`).
5. **The header picks the engine and runtime, and the server checks again at Save.**
   - v2 goes to `ninfer` and v3 to `ninfer-upstream`.
   - A part file, a missing part, garbage or a truncated header is refused, and so is an
     unsupported architecture.
   - A page that sends another engine, a taken id or a duplicate path is refused.
   - Pinned in Task 1 (`test_magic_bytes_match_the_frozen_runtimes`, `test_v2_goes_to_the_quasar_runtime`,
     `test_v3_goes_to_upstream_and_lists_its_parts`, `test_parts_missing_parts_and_garbage_are_not_models`)
     and Task 5 (`test_the_server_checks_everything_again_at_save`). Checked live in Task 8 Step 3
     against every artifact on the box.

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `python/freetoken/daemon/settings/model_detect.py` | new | NInfer header reader, detection, id/name/memory suggestions, a model's own files, CLI |
| `python/freetoken/daemon/settings/browse.py` | modify | Browse kind `add`: folders plus `.ninfer` files, model folders and `.ninfer` files marked |
| `python/freetoken/daemon/settings/download.py` | modify | LFS sha256 on `RemoteFile`; `plan_add` / `start_add` / `latest_add`; staging, checksums, place, clean-up |
| `python/freetoken/daemon/settings/pi_sync.py` | new | `PiSync.add/remove` with backups, format-keeping writes, "Pi not updated" |
| `python/freetoken/daemon/settings/panel.py` | modify | detect / add-info / plan / downloads / add / remove routes and methods; `_save` tweaks; `_fit_to_model`; `aliases` in the list, `artifact` in the view |
| `python/freetoken/daemon/settings/app.py` | modify | wire downloads, `PiSync`, the profile-deleted hook; version 2.1.0 |
| `python/freetoken/daemon/settings/static/panel.js` | modify | wizard, remove question, plain-words helpers |
| `python/freetoken/daemon/settings/static/index.html` | modify | Add button, wizard and remove dialogs, browser `onPick`, Escape handling |
| `tests/settings/test_model_detect.py` | new | Task 1 |
| `tests/settings/test_browse.py` | modify | Task 2 |
| `tests/settings/test_download_add.py` | new | Task 3 |
| `tests/settings/test_pi_sync.py` | new | Task 4 |
| `tests/settings/test_panel_add_remove.py` | new | Task 5 |
| `tests/settings/test_routes.py` | modify | version 2.1.0 |
| `tests/settings/test_panel_page.py` | modify | Task 6 |
| `README.md` | modify | fork section: add and remove models, Pi sync |
| `docs/research/control-panel-b-acceptance-<date>.md` + `…/<date>/*.png` | new (Task 8) | live acceptance record and screenshots |

---

### Task 1: What a path is (`model_detect.py`)

**Files:**
- Create: `python/freetoken/daemon/settings/model_detect.py`
- Test: `tests/settings/test_model_detect.py`

**Interfaces:**
- Consumes: `model_info.read_model`, `model_info.GIB`, `model_info.ModelInfo`,
  `registry.ENGINE_LABELS`, `ninfer_dials.RUNTIME_LABELS`.
- Produces:
  - constants `V2_MAGIC`, `V3_MAGIC`, `PART_MAGIC`, `V2_PREFIX`, `V3_HEADER`, `PART_RE`,
    `RUNTIME_BY_VERSION`, `NINFER_SUFFIX`;
  - `NotAModel(ValueError)`;
  - `read_ninfer(path) -> {"version", "runtime", "files": [str], "bytes"}` (raises `NotAModel`);
  - `detect(path, *, taken=()) -> {"kind": "ninfer"|"freetoken"|"unsupported", "path", "engine",
    "runtime", "engineLabel", "runtimeLabel", "format", "bytes", "files", "reason", "suggested":
    {"id","name","ramNeedGB"} | None}`;
  - `model_files(engine, artifact) -> [str]`;
  - `suggest_id(stem, taken) -> str`, `suggest_name(stem, engine) -> str`,
    `suggest_ram_gb(engine, total_bytes, info=None) -> int`;
  - `main(argv) -> int`.
- Test helpers other tasks import: `write_v2(path, payload=4096)`,
  `write_v3(path, parts=0, payload=4096, part_names=None)`,
  `write_folder(path, architecture="LlamaForCausalLM", **config)`.

- [ ] **Step 1: Write the failing tests** (`tests/settings/test_model_detect.py`)

```python
"""What a file or folder is, for the Add model wizard. Review focus 5 lives here."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from freetoken.daemon.settings import model_detect as md
from freetoken.daemon.settings.model_info import ModelInfo
from freetoken.daemon.settings.registry import MODEL_ID_RE

REPO = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3
PAD = 4096


def write_v2(path: Path, payload: int = 4096) -> Path:
    """A minimal v2 artifact: magic + u64 JSON length + directory, payload at 4 KiB."""
    directory = json.dumps({"identity": {"model_id": "m", "weights_id": "w"}, "objects": []}).encode()
    head = md.V2_PREFIX.pack(md.V2_MAGIC, len(directory)) + directory
    path.write_bytes(head + b"\0" * (PAD - len(head)) + b"\1" * payload)
    return path


def write_v3(path: Path, parts: int = 0, payload: int = 4096, part_names: list[str] | None = None) -> Path:
    """A minimal v3 entry (magic, JSON length, 16-byte id, directory) and its parts."""
    names = part_names if part_names is not None else [f"{path.name}.part-{i:04d}" for i in range(1, parts + 1)]
    files = [{"path": None, "payload_bytes": payload}] + [{"path": name, "payload_bytes": payload} for name in names]
    directory = json.dumps({"files": files, "objects": []}).encode() + b"   "  # the writer pads with spaces
    ident = b"\x07" * 16
    head = md.V3_HEADER.pack(md.V3_MAGIC, len(directory), ident) + directory
    path.write_bytes(head + b"\0" * (PAD - len(head)) + b"\1" * payload)
    for index, name in enumerate(names, start=1):
        part = path.with_name(name)
        if not part.exists():
            part.write_bytes(md.V3_HEADER.pack(md.PART_MAGIC, index, ident) + b"\0" * (PAD - 32) + b"\2" * payload)
    return path


def write_folder(path: Path, architecture: str = "LlamaForCausalLM", **config) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    body = {"architectures": [architecture], "num_hidden_layers": 2, "hidden_size": 64, **config}
    (path / "config.json").write_text(json.dumps(body), encoding="utf-8")
    (path / "model.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")
    return path


def test_magic_bytes_match_the_frozen_runtimes():
    fork = (REPO / "engines/ninfer/tools/artifact/container.py").read_text(encoding="utf-8")
    upstream = (REPO / "engines/ninfer-upstream/tools/artifact/framing.py").read_text(encoding="utf-8")
    assert r'MAGIC = b"NINFER\x00\x02"' in fork and 'PREFIX = struct.Struct("<8sQ")' in fork
    assert r'MAGIC = b"NINFER\x00\x03"' in upstream and r'PART_MAGIC = b"NINPRT\x00\x03"' in upstream
    assert 'HEADER = struct.Struct("<8sQ16s")' in upstream
    assert (md.V2_MAGIC, md.V3_MAGIC, md.PART_MAGIC) == (b"NINFER\x00\x02", b"NINFER\x00\x03", b"NINPRT\x00\x03")
    assert md.RUNTIME_BY_VERSION == {2: "ninfer", 3: "ninfer-upstream"}


def test_v2_goes_to_the_quasar_runtime(tmp_path):
    entry = write_v2(tmp_path / "quasar_27b_nvfp4.ninfer")
    found = md.detect(str(entry))
    assert (found["kind"], found["engine"], found["runtime"]) == ("ninfer", "ninfer", "ninfer")
    assert found["runtimeLabel"] == "QUASAR runtime" and found["format"] == "NInfer v2 file"
    assert found["files"] == [str(entry)] and found["bytes"] == entry.stat().st_size
    assert found["suggested"] == {"id": "quasar_27b_nvfp4", "name": "quasar 27b nvfp4 (NInfer)", "ramNeedGB": 1}


def test_v3_goes_to_upstream_and_lists_its_parts(tmp_path):
    entry = write_v3(tmp_path / "fable.ninfer", parts=2)
    found = md.detect(str(entry))
    assert (found["engine"], found["runtime"]) == ("ninfer", "ninfer-upstream")
    assert found["files"] == [str(entry), str(tmp_path / "fable.ninfer.part-0001"), str(tmp_path / "fable.ninfer.part-0002")]
    assert found["bytes"] == sum(Path(item).stat().st_size for item in found["files"])


def test_parts_missing_parts_and_garbage_are_not_models(tmp_path):
    entry = write_v3(tmp_path / "split.ninfer", parts=2)
    assert "one part of a split" in md.detect(str(tmp_path / "split.ninfer.part-0001"))["reason"]
    (tmp_path / "split.ninfer.part-0002").unlink()
    missing = md.detect(str(entry))
    assert missing["kind"] == "unsupported" and "split.ninfer.part-0002 is missing" in missing["reason"]
    (tmp_path / "junk.ninfer").write_bytes(b"hello there")
    assert "does not start like a NInfer model" in md.detect(str(tmp_path / "junk.ninfer"))["reason"]
    cut = md.V3_HEADER.pack(md.V3_MAGIC, 10_000, b"\0" * 16) + b"{}"
    (tmp_path / "cut.ninfer").write_bytes(cut)
    assert "damaged" in md.detect(str(tmp_path / "cut.ninfer"))["reason"]
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert "not supported by your engines" in md.detect(str(tmp_path / "notes.txt"))["reason"]
    assert md.detect(str(tmp_path / "nothing-here"))["reason"] == "Nothing was found at that path."
    assert md.detect("relative/path")["reason"] == "Use a full path, starting with / or ~/."


def test_folders_follow_freetokens_model_registry(tmp_path):
    good = md.detect(str(write_folder(tmp_path / "Tiny-Llama")))
    assert (good["engine"], good["runtime"], good["format"]) == ("freetoken", "freetoken", "LlamaForCausalLM model folder")
    assert good["suggested"]["name"] == "Tiny-Llama (FreeToken)"
    bert = md.detect(str(write_folder(tmp_path / "bert", "BertModel")))
    assert bert["kind"] == "unsupported" and "(BertModel) is not supported by your engines" in bert["reason"]
    (tmp_path / "empty").mkdir()
    assert "not supported by your engines" in md.detect(str(tmp_path / "empty"))["reason"]
    bare = write_folder(tmp_path / "no-weights")
    (bare / "model.safetensors").unlink()
    assert "no .safetensors" in md.detect(str(bare))["reason"]


def test_ids_are_safe_and_unique():
    assert md.suggest_id("Quasar 27B NVFP4", []) == "quasar-27b-nvfp4"
    assert md.suggest_id("quasar_27b_nvfp4", ["quasar_27b_nvfp4", "quasar_27b_nvfp4-2"]) == "quasar_27b_nvfp4-3"
    assert md.suggest_id("Qwen3.8-Flash-Next-NVFP4", ["Qwen3.8-Flash-Next-NVFP4"]) == "qwen3.8-flash-next-nvfp4-2"
    assert md.suggest_id("...!!!", []) == "model"
    long = md.suggest_id("x" * 100, ["x" * 63])
    assert len(long) <= 63 and MODEL_ID_RE.fullmatch(long) and long.endswith("-2")


def test_memory_suggestions_match_the_measured_models():
    assert md.suggest_ram_gb("ninfer", 19_782_132_224) == 19          # QUASAR, 18 in use
    flash = ModelInfo(path="", name="", total_expert_bytes=68_136_468_480)
    assert md.suggest_ram_gb("freetoken", 0, flash) == 64              # Qwen3.8 Flash, 61-62 measured
    dense = ModelInfo(path="", name="", weight_bytes=10 * GIB + 1, ple_bytes=GIB)
    assert md.suggest_ram_gb("freetoken", 0, dense) == 10
    assert md.suggest_ram_gb("ninfer", 5) == 1


def test_a_models_own_files(tmp_path):
    entry = write_v3(tmp_path / "split.ninfer", parts=1)
    assert md.model_files("ninfer", str(entry)) == [str(entry), str(tmp_path / "split.ninfer.part-0001")]
    folder = write_folder(tmp_path / "Tiny")
    assert md.model_files("freetoken", str(folder)) == [str(folder)]
    (tmp_path / "split.ninfer.part-0002").write_bytes(b"orphan")  # damaged listing: fall back to the name pattern
    entry.write_bytes(b"broken")
    assert md.model_files("ninfer", str(entry)) == [str(entry), str(tmp_path / "split.ninfer.part-0001"),
                                                    str(tmp_path / "split.ninfer.part-0002")]
    assert md.model_files("ninfer", str(tmp_path / "gone.ninfer")) == []


def test_command_line(tmp_path, capsys):
    entry = write_v2(tmp_path / "q.ninfer")
    assert md.main([str(entry)]) == 0
    assert "ninfer runtime=ninfer" in capsys.readouterr().out
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert md.main([str(tmp_path / "notes.txt")]) == 1
```

- [ ] **Step 2: Run to see them fail**

`PYTEST tests/settings/test_model_detect.py`
Expected: `ModuleNotFoundError: No module named 'freetoken.daemon.settings.model_detect'`.

- [ ] **Step 3: Write `python/freetoken/daemon/settings/model_detect.py`**

```python
"""What a file or folder on this PC is, for the control panel's Add model wizard.

Spec: docs/superpowers/specs/2026-09-24-control-panel-part2-design.md, section 7 (detection).

A ``.ninfer`` file is a NInfer model, and its first 8 bytes say which runtime can read it:

- ``NINFER\\0\\x02`` (v2): only the QUASAR fork, engines/ninfer (src/artifact/reader.cpp
  ``kMagic``; tools/artifact/container.py ``MAGIC`` and ``PREFIX = "<8sQ"``: magic + JSON
  length). Its reader refuses anything else with "artifact magic is not NInfer v2".
- ``NINFER\\0\\x03`` (v3): only engines/ninfer-upstream (tools/artifact/framing.py ``MAGIC`` and
  ``HEADER = "<8sQ16s"``: magic, JSON length, 16-byte id). Its reader refuses v2 with "NInfer v2
  artifact is not supported" (src/artifact/reader.cpp).

So the header, not the file name, picks the runtime. A v3 model may be split: the entry's JSON
directory lists its continuation files in ``files[1:]`` (named ``<entry>.part-NNNN`` by
tools/artifact/writer.py), each starting with ``NINPRT\\0\\x03``. Only the header and the
directory are read here, never the weights.

A folder is a FreeToken model when its config.json names an architecture FreeToken's model
registry serves (model_info.SUPPORTED_ARCHITECTURES, kept in step with models/register.py by
tests/settings/test_model_info.py) and it holds .safetensors weights. Anything else is "not
supported by your engines".

``python -m freetoken.daemon.settings.model_detect PATH...`` prints what each path is; the live
acceptance runs it over every artifact on the box.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

from .model_info import GIB, ModelInfo, read_model
from .ninfer_dials import RUNTIME_LABELS
from .registry import ENGINE_LABELS

NINFER_SUFFIX = ".ninfer"
V2_MAGIC = b"NINFER\x00\x02"
V3_MAGIC = b"NINFER\x00\x03"
PART_MAGIC = b"NINPRT\x00\x03"
V2_PREFIX = struct.Struct("<8sQ")
V3_HEADER = struct.Struct("<8sQ16s")
RUNTIME_BY_VERSION = {2: "ninfer", 3: "ninfer-upstream"}
PART_RE = re.compile(r"^(?P<entry>.+\.ninfer)\.part-\d{4}$")
# A v3 directory's own sibling-name rule is stricter (tools/artifact/schema.py identifier);
# this one only has to keep a listed part inside the entry's folder.
_SIBLING_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
# The largest directory read; QUASAR-size artifacts carry well under 1 MiB of JSON.
MAX_DIRECTORY_BYTES = 64 * 1024 * 1024
NOT_SUPPORTED = "It is not supported by your engines."
PART_WORDS = "This is one part of a split NInfer model. Pick the file ending in .ninfer next to it."
DAMAGED_WORDS = "This NInfer file is damaged: its header does not match the file. " + NOT_SUPPORTED


class NotAModel(ValueError):
    """Plain words for why a path cannot be added."""


def read_ninfer(path: str | os.PathLike[str]) -> dict[str, Any]:
    entry = Path(path)
    try:
        size = entry.stat().st_size
        with entry.open("rb") as fh:
            head = fh.read(V3_HEADER.size)
            magic = head[:8]
            if magic == V2_MAGIC:
                return _v2(entry, head, size)
            if magic == PART_MAGIC:
                raise NotAModel(PART_WORDS)
            if magic != V3_MAGIC:
                raise NotAModel("This file does not start like a NInfer model. " + NOT_SUPPORTED)
            if len(head) < V3_HEADER.size:
                raise NotAModel(DAMAGED_WORDS)
            _, json_bytes, _ = V3_HEADER.unpack(head)
            if not 0 < json_bytes <= min(MAX_DIRECTORY_BYTES, size - V3_HEADER.size):
                raise NotAModel(DAMAGED_WORDS)
            directory = fh.read(json_bytes)
    except OSError as exc:
        raise NotAModel(f"The file could not be read ({exc.strerror or exc}).") from exc
    return _v3(entry, directory, size)


def _v2(entry: Path, head: bytes, size: int) -> dict[str, Any]:
    if len(head) < V2_PREFIX.size:
        raise NotAModel(DAMAGED_WORDS)
    _, json_bytes = V2_PREFIX.unpack(head[:V2_PREFIX.size])
    if not 0 < json_bytes <= size - V2_PREFIX.size:
        raise NotAModel(DAMAGED_WORDS)
    return {"version": 2, "runtime": RUNTIME_BY_VERSION[2], "files": [str(entry)], "bytes": size}


def _v3(entry: Path, directory: bytes, size: int) -> dict[str, Any]:
    try:
        doc = json.loads(directory.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise NotAModel(DAMAGED_WORDS) from exc
    records = doc.get("files") if isinstance(doc, dict) else None
    if (not isinstance(records, list) or not records or not isinstance(records[0], dict)
            or records[0].get("path") is not None):
        raise NotAModel(DAMAGED_WORDS)
    files, total = [str(entry)], size
    for record in records[1:]:
        name = record.get("path") if isinstance(record, dict) else None
        if not isinstance(name, str) or not _SIBLING_RE.fullmatch(name):
            raise NotAModel(DAMAGED_WORDS)
        part = entry.with_name(name)
        try:
            total += part.stat().st_size
        except OSError as exc:
            raise NotAModel(f"This NInfer model is split into parts, and the part {name} is missing next to it.") from exc
        files.append(str(part))
    return {"version": 3, "runtime": RUNTIME_BY_VERSION[3], "files": files, "bytes": total}


def model_files(engine: str, artifact: str) -> list[str]:
    """Everything that belongs to a model on disk: a NInfer entry plus its v3 parts, or a
    FreeToken folder. Paths that do not exist are left out."""
    path = Path(os.path.expanduser(artifact))
    if engine != "ninfer":
        return [str(path)] if path.is_dir() else []
    try:
        return read_ninfer(path)["files"]
    except NotAModel:
        # Damaged or half-deleted: the entry (if any) plus the parts named after it.
        found = [path] if path.is_file() else []
        found += sorted(path.parent.glob(glob.escape(path.name) + ".part-[0-9][0-9][0-9][0-9]"))
        return [str(item) for item in found]


def suggest_id(stem: str, taken: Iterable[str]) -> str:
    used = {str(item).lower() for item in taken}
    base = re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip("-._")[:63].rstrip("-._") or "model"
    candidate, number = base, 2
    while candidate in used:
        suffix = f"-{number}"
        candidate = base[:63 - len(suffix)].rstrip("-._") + suffix
        number += 1
    return candidate


def suggest_name(stem: str, engine: str) -> str:
    words = re.sub(r"[_\s]+", " ", stem).strip() or "New model"
    return f"{words} ({ENGINE_LABELS[engine]})"[:120]


def suggest_ram_gb(engine: str, total_bytes: int, info: ModelInfo | None = None) -> int:
    """PC memory to wait for before loading, rounded up to whole GB (1024^3, like the page).

    NInfer maps the whole file: QUASAR's 19,782,132,224 bytes give 19 (config has run on 18
    since 2026-09-24). FreeToken keeps the routed experts in host RAM: Qwen3.8 Flash's
    68,136,468,480 expert bytes give 64 against 61-62 GB measured on the Windows side
    (docs/research/own-switcher-acceptance-2026-09-24.md, control-panel-a-acceptance-2026-09-25.md).
    Without an expert count, the weights minus the demand-paged PLE table."""
    if engine == "ninfer":
        return max(1, math.ceil(total_bytes / GIB))
    experts = int(getattr(info, "total_expert_bytes", 0) or 0)
    if experts:
        return max(1, math.ceil(experts / GIB))
    weights = max(0, int(info.weight_bytes) - int(info.ple_bytes)) if info is not None else total_bytes
    return max(1, math.ceil(weights / GIB))


def detect(path: str, *, taken: Iterable[str] = ()) -> dict[str, Any]:
    text = str(path or "").strip().strip('"').strip("'")
    out: dict[str, Any] = {"kind": "unsupported", "path": text, "engine": None, "runtime": None, "engineLabel": "",
                           "runtimeLabel": "", "format": "", "bytes": 0, "files": [], "reason": "", "suggested": None}
    if not text:
        out["reason"] = "Choose a file or folder first."
        return out
    target = Path(os.path.expanduser(text))
    if not target.is_absolute():
        out["reason"] = "Use a full path, starting with / or ~/."
        return out
    target = Path(os.path.abspath(target))
    out["path"] = str(target)
    try:
        if target.is_file():
            if PART_RE.match(target.name):
                raise NotAModel(PART_WORDS)
            if not target.name.endswith(NINFER_SUFFIX):
                raise NotAModel("This file is not a NInfer model (those end in .ninfer). " + NOT_SUPPORTED)
            found = read_ninfer(target)
            engine, runtime, size, files = "ninfer", found["runtime"], found["bytes"], found["files"]
            stem, ram = target.name[:-len(NINFER_SUFFIX)], suggest_ram_gb("ninfer", found["bytes"])
            out["format"] = f"NInfer v{found['version']} file"
        elif target.is_dir():
            info = read_model(target)
            if not info.found:
                raise NotAModel(f"{info.error} {NOT_SUPPORTED}")
            if not info.supported:
                raise NotAModel(f"This model's design ({info.architecture or 'not named in its config.json'}) "
                                "is not supported by your engines.")
            if not info.weight_files:
                raise NotAModel("This folder has a config.json but no .safetensors weight files. " + NOT_SUPPORTED)
            engine, runtime, size, files = "freetoken", "freetoken", info.weight_bytes, [str(target)]
            stem, ram = target.name, suggest_ram_gb("freetoken", info.weight_bytes, info)
            out["format"] = f"{info.architecture} model folder"
        else:
            raise NotAModel("Nothing was found at that path.")
    except NotAModel as exc:
        out["reason"] = str(exc)
        return out
    out.update(kind=engine, engine=engine, runtime=runtime, engineLabel=ENGINE_LABELS[engine],
               runtimeLabel=RUNTIME_LABELS.get(runtime, ""), bytes=int(size), files=files,
               suggested={"id": suggest_id(stem, taken), "name": suggest_name(stem, engine), "ramNeedGB": ram})
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m freetoken.daemon.settings.model_detect PATH...", file=sys.stderr)
        return 2
    status = 0
    for arg in args:
        found = detect(arg)
        if found["kind"] == "unsupported":
            print(f"{arg}: not supported: {found['reason']}")
            status = 1
        else:
            print(f"{arg}: {found['engine']} runtime={found['runtime']} bytes={found['bytes']} "
                  f"files={len(found['files'])} id={found['suggested']['id']} ramNeedGB={found['suggested']['ramNeedGB']}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [name for name in dir() if not name.startswith("_")]
```

- [ ] **Step 4: Run the tests**

`PYTEST tests/settings/test_model_detect.py tests/settings/test_settings_import_safety.py`
Expected: all pass. The import-safety test confirms the helper still imports no torch.

- [ ] **Step 5: Commit (controller)**

```bash
git add python/freetoken/daemon/settings/model_detect.py tests/settings/test_model_detect.py
git commit -m "feat(settings): tell NInfer v2/v3 files and FreeToken folders apart for Add model

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Browse for a model file or folder (`browse.py`)

**Files:**
- Modify: `python/freetoken/daemon/settings/browse.py`
- Test: `tests/settings/test_browse.py`

**Interfaces:**
- Produces: `BROWSE_KINDS` gains `"add"`. `list_directory(path, "add")` lists folders and
  `.ninfer` files. It marks model folders and `.ninfer` files `isModel`. Part files, other
  files and dot entries (including `.incoming-*` staging folders) stay out.
  `/api/browse?kind=add` accepts it through the existing route.

- [ ] **Step 1: Write the failing test** (append to `tests/settings/test_browse.py`)

```python
def test_add_kind_lists_ninfer_files_and_model_folders(tmp_path):
    _tree(tmp_path)
    for name in ("quasar.ninfer", "twin.ninfer", "twin.ninfer.part-0001"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / ".incoming-download-1").mkdir()
    listing = list_directory(str(tmp_path), "add")
    names = {entry["name"]: entry for entry in listing["entries"]}
    assert set(names) == {"Some-Model", "Pictures", "quasar.ninfer", "twin.ninfer"}
    assert names["Some-Model"]["isModel"] is True and names["quasar.ninfer"]["isModel"] is True
    assert names["quasar.ninfer"]["kind"] == "file" and names["Pictures"]["isModel"] is False
    assert list_directory(str(tmp_path / "Some-Model"), "add")["isModel"] is True
```

- [ ] **Step 2: Run to see it fail**

`PYTEST tests/settings/test_browse.py -k add_kind`
Expected: `ValueError: kind must be one of folder, model, file`.

- [ ] **Step 3: Edit `browse.py`**
  - Module docstring: append the sentence "Kind ``add`` (the control panel's Add model wizard)
    lists folders and ``.ninfer`` files and marks model folders and ``.ninfer`` files."
  - Replace `BROWSE_KINDS = ("folder", "model", "file")` with
    `BROWSE_KINDS = ("folder", "model", "file", "add")`.
  - In `list_directory`, replace

```python
        if not is_dir and kind != "file":
            continue
```
  with

```python
        if not is_dir and kind != "file" and not (kind == "add" and child.name.endswith(".ninfer")):
            continue
```
  - Replace the entry's `"isModel": bool(is_dir and kind == "model" and is_model_folder(child)),` with

```python
                "isModel": bool((is_dir and kind in ("model", "add") and is_model_folder(child))
                                or (not is_dir and kind == "add")),
```
  - Replace the listing's `"isModel": kind == "model" and is_model_folder(start),` with
    `"isModel": kind in ("model", "add") and is_model_folder(start),`.

- [ ] **Step 4: Run the tests**

`PYTEST tests/settings/test_browse.py`
Expected: all pass.

- [ ] **Step 5: Commit (controller)**

```bash
git add python/freetoken/daemon/settings/browse.py tests/settings/test_browse.py
git commit -m "feat(settings): browse kind for picking a NInfer file or model folder

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Downloads for the wizard: single files, parts, checksums (`download.py`)

**Files:**
- Modify: `python/freetoken/daemon/settings/download.py`
- Test: `tests/settings/test_download_add.py` (the existing `tests/settings/test_download.py`
  must keep passing unchanged)

**Interfaces:**
- Consumes: `model_info.describe_config`, `memory_reclaim.release_completed_file` (both already
  imported).
- Produces:
  - `RemoteFile.sha256: str | None`, `AddUnsupported(ValueError)`,
    `parse_sha256sums(text) -> {name: hex}`, `sha256_file(path) -> hex`;
  - `DownloadManager.plan_add(value, entry=None, *, folder_root, ninfer_root) -> dict` with keys
    `repo, name, engine ("ninfer"|"freetoken"), kind ("ninfer"|"folder"), entries, entry,
    architecture, files [{name, bytes, check: "SHA256SUMS"|"published"|None}], sumsFile,
    totalBytes, root, target, finals, exists, diskFreeBytes, diskFits`;
  - `DownloadManager.start_add(value, entry=None, *, folder_root, ninfer_root) -> DownloadJob`;
  - `DownloadManager.latest_add() -> dict | None`;
  - `DownloadJob` gains `kind` ("folder" default, "add"), `engine`, `staging`, `target`,
    `fetch`, `finals`, `sums_name`, `verified`. `as_dict()` gains `kind`, `engine`, `verified`
    and `resultPath` (the entry file or folder once `done`, else `None`).
  - Add-job stages: `queued → downloading → verifying → moving → done`, or `failed` / `cancelled`.
- Test helper other tasks import: `Hub(files, published=True, config=None, fail_on=None, gate=None)`.

- [ ] **Step 1: Write the failing tests** (`tests/settings/test_download_add.py`)

```python
"""The Add model wizard's downloads (DownloadManager.plan_add/start_add). Review focus 4 lives here."""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from freetoken.daemon.settings.download import (
    AddUnsupported, DownloadConflict, DownloadManager, InvalidRepository, parse_sha256sums,
)

GIB = 1024 ** 3


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Hub:
    """A fake Hub. model_info lists the files (LFS sha256 on all but config.json and
    SHA256SUMS when published); snapshot writes one file like snapshot_download(local_dir=…),
    including its .cache folder."""

    def __init__(self, files: dict[str, bytes], published: bool = True, config=None, fail_on=None, gate=None):
        self.files, self.published, self.config, self.fail_on, self.gate = files, published, config, fail_on, gate
        self.fetched: list[str] = []

    def model_info(self, repo, files_metadata=False):
        rows = []
        for name, data in self.files.items():
            lfs = (SimpleNamespace(size=len(data), sha256=sha(data))
                   if self.published and name not in ("config.json", "SHA256SUMS") else None)
            rows.append(SimpleNamespace(rfilename=name, size=len(data), lfs=lfs))
        return SimpleNamespace(siblings=rows)

    def snapshot(self, repo, *, local_dir, allow_patterns=None, **_kwargs):
        name = allow_patterns[0]
        self.fetched.append(name)
        if name == self.fail_on:
            raise OSError("connection reset")
        target = Path(local_dir)
        (target / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(self.files[name])
        if self.gate is not None and name == self.gate[0]:
            self.gate[1].set()
            self.gate[2].wait(2)


def manager(tmp_path, hub, free=10 ** 12):
    return DownloadManager(tmp_path / "models", api_factory=hub, config_fetcher=lambda _: hub.config or {},
                           snapshot_downloader=hub.snapshot, pc_memory=64 * GIB, card_memory=32 * GIB,
                           disk_free=lambda _: free)


def roots(tmp_path):
    folder, ninfer = tmp_path / "models", tmp_path / "ninfer"
    folder.mkdir(exist_ok=True)
    ninfer.mkdir(exist_ok=True)
    return {"folder_root": folder, "ninfer_root": ninfer}


def finish(m, job):
    for _ in range(400):
        current = m.get(job.job_id)
        if current.stage in ("done", "failed", "cancelled"):
            return current.as_dict()
        time.sleep(0.005)
    raise AssertionError(m.get(job.job_id).as_dict())


def leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.startswith((".incoming-", ".cache")))


def test_a_ninfer_repo_fetches_the_entry_and_its_parts_and_checks_them(tmp_path):
    hub = Hub({"small.ninfer": b"entry" * 10, "small.ninfer.part-0001": b"part" * 5, "README.md": b"hi"})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    plan = m.plan_add("owner/small-repo", **r)
    assert (plan["engine"], plan["kind"], plan["entry"]) == ("ninfer", "ninfer", "small.ninfer")
    assert [f["name"] for f in plan["files"]] == ["small.ninfer", "small.ninfer.part-0001"]
    assert {f["check"] for f in plan["files"]} == {"published"}
    assert plan["target"] == str(r["ninfer_root"] / "small.ninfer") and plan["exists"] is False
    done = finish(m, m.start_add("https://huggingface.co/owner/small-repo", **r))
    assert done["stage"] == "done", done
    assert done["resultPath"] == str(r["ninfer_root"] / "small.ninfer")
    assert sorted(done["verified"]) == ["small.ninfer", "small.ninfer.part-0001"]
    assert (r["ninfer_root"] / "small.ninfer.part-0001").read_bytes() == b"part" * 5
    assert hub.fetched == ["small.ninfer", "small.ninfer.part-0001"]
    assert leftovers(r["ninfer_root"]) == []


def test_parse_sha256sums_reads_both_common_forms():
    text = f"{'a' * 64}  one.ninfer\n{'B' * 64} *./two.ninfer\nnot a line\n"
    assert parse_sha256sums(text) == {"one.ninfer": "a" * 64, "two.ninfer": "b" * 64}


def test_sha256sums_wins_and_a_mismatch_deletes_everything(tmp_path):
    hub = Hub({"small.ninfer": b"weights" * 100, "SHA256SUMS": f"{'0' * 64}  small.ninfer\n".encode()})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    (r["ninfer_root"] / "other.ninfer").write_bytes(b"keep me")
    plan = m.plan_add("owner/repo", **r)
    assert plan["sumsFile"] == "SHA256SUMS" and plan["files"] == [{"name": "small.ninfer", "bytes": 700, "check": "SHA256SUMS"}]
    done = finish(m, m.start_add("owner/repo", **r))
    assert done["stage"] == "failed" and "checksum" in done["error"], done
    assert sorted(p.name for p in r["ninfer_root"].iterdir()) == ["other.ninfer"]
    assert (r["ninfer_root"] / "other.ninfer").read_bytes() == b"keep me"


def test_two_ninfer_files_need_a_pick_and_only_the_pick_is_fetched(tmp_path):
    hub = Hub({"a.ninfer": b"a", "b.ninfer": b"b", "b.ninfer.part-0001": b"bb"})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    plan = m.plan_add("owner/repo", **r)
    assert plan["entries"] == ["a.ninfer", "b.ninfer"] and plan["entry"] is None and plan["files"] == []
    with pytest.raises(InvalidRepository, match="Pick"):
        m.start_add("owner/repo", **r)
    with pytest.raises(InvalidRepository):
        m.plan_add("owner/repo", "c.ninfer", **r)
    done = finish(m, m.start_add("owner/repo", "b.ninfer", **r))
    assert done["stage"] == "done" and hub.fetched == ["b.ninfer", "b.ninfer.part-0001"]
    assert not (r["ninfer_root"] / "a.ninfer").exists()


def test_nothing_is_ever_overwritten(tmp_path):
    hub = Hub({"small.ninfer": b"new"})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    (r["ninfer_root"] / "small.ninfer").write_bytes(b"old")
    assert m.plan_add("owner/repo", **r)["exists"] is True
    with pytest.raises(DownloadConflict, match="already on this PC"):
        m.start_add("owner/repo", **r)
    assert (r["ninfer_root"] / "small.ninfer").read_bytes() == b"old" and hub.fetched == []


def test_a_file_that_appears_during_the_download_is_not_overwritten(tmp_path):
    started, release = threading.Event(), threading.Event()
    hub = Hub({"small.ninfer": b"new"}, gate=("small.ninfer", started, release))
    m, r = manager(tmp_path, hub), roots(tmp_path)
    job = m.start_add("owner/repo", **r)
    assert started.wait(2)
    (r["ninfer_root"] / "small.ninfer").write_bytes(b"old")
    release.set()
    done = finish(m, job)
    assert done["stage"] == "failed" and "appeared" in done["error"], done
    assert (r["ninfer_root"] / "small.ninfer").read_bytes() == b"old" and leftovers(r["ninfer_root"]) == []


def test_a_model_folder_repo_lands_in_models_name(tmp_path):
    config = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2, "hidden_size": 64}
    hub = Hub({"config.json": b"{}", "model.safetensors": b"w" * 32, "tokenizer.json": b"{}", "notes.md": b"x"},
              config=config)
    m, r = manager(tmp_path, hub), roots(tmp_path)
    plan = m.plan_add("owner/Tiny-Llama", **r)
    assert (plan["engine"], plan["kind"], plan["architecture"]) == ("freetoken", "folder", "LlamaForCausalLM")
    assert plan["target"] == str(r["folder_root"] / "Tiny-Llama")
    assert [f["name"] for f in plan["files"]] == ["config.json", "model.safetensors", "tokenizer.json"]
    done = finish(m, m.start_add("owner/Tiny-Llama", **r))
    assert done["stage"] == "done", done
    folder = r["folder_root"] / "Tiny-Llama"
    assert done["resultPath"] == str(folder) and (folder / "model.safetensors").is_file()
    assert not (folder / ".cache").exists() and leftovers(r["folder_root"]) == []
    assert sorted(done["verified"]) == ["model.safetensors", "tokenizer.json"]


def test_designs_and_repos_the_engines_cannot_run_are_refused_before_downloading(tmp_path):
    bert = Hub({"config.json": b"{}", "model.safetensors": b"w"}, config={"architectures": ["BertModel"]})
    with pytest.raises(AddUnsupported, match=r"BertModel\) is not supported by your engines"):
        manager(tmp_path, bert).plan_add("owner/bert", **roots(tmp_path))
    gguf = Hub({"README.md": b"hi", "model.gguf": b"x"})
    with pytest.raises(AddUnsupported, match="not supported by your engines"):
        manager(tmp_path, gguf).start_add("owner/gguf", **roots(tmp_path))
    assert bert.fetched == [] and gguf.fetched == []


def test_failure_and_cancel_delete_the_partial_files(tmp_path):
    r = roots(tmp_path)
    broken = Hub({"small.ninfer": b"a" * 10, "small.ninfer.part-0001": b"b" * 10}, fail_on="small.ninfer.part-0001")
    m = manager(tmp_path, broken)
    failed = finish(m, m.start_add("owner/repo", **r))
    assert failed["stage"] == "failed" and "connection reset" in failed["error"]
    assert list(r["ninfer_root"].iterdir()) == []

    started, release = threading.Event(), threading.Event()
    slow = Hub({"small.ninfer": b"a" * 10, "small.ninfer.part-0001": b"b" * 10}, gate=("small.ninfer", started, release))
    m2 = manager(tmp_path, slow)
    job = m2.start_add("owner/repo", **r)
    assert started.wait(2)
    assert m2.cancel(job.job_id).cancel_requested is True
    release.set()
    assert finish(m2, job)["stage"] == "cancelled"
    assert list(r["ninfer_root"].iterdir()) == [] and slow.fetched == ["small.ninfer"]


def test_not_enough_drive_space_refuses_to_start(tmp_path):
    m = manager(tmp_path, Hub({"small.ninfer": b"a" * 100}), free=10)
    with pytest.raises(DownloadConflict, match="drive space"):
        m.start_add("owner/repo", **roots(tmp_path))


def test_latest_add_reports_the_newest_wizard_download(tmp_path):
    hub = Hub({"small.ninfer": b"a"})
    m, r = manager(tmp_path, hub), roots(tmp_path)
    assert m.latest_add() is None
    job = m.start_add("owner/repo", **r)
    finish(m, job)
    assert m.latest_add()["id"] == job.job_id and m.latest_add()["kind"] == "add"
```

- [ ] **Step 2: Run to see them fail**

`PYTEST tests/settings/test_download_add.py`
Expected: `ImportError: cannot import name 'AddUnsupported'`.

- [ ] **Step 3: Edit `download.py`**

(a) Imports: add `import hashlib`.

(b) After `_TERMINAL_STAGES = …` add:

```python
# The Add model wizard (control panel Stage B). A repo's SHA256SUMS wins over the Hub's
# per-file LFS sha256; files with neither (small git files) are not checked.
_SUMS_NAMES = ("SHA256SUMS", "SHA256SUMS.txt")
_NINFER_ENTRY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.ninfer$")
_NINFER_PART = re.compile(r"^(?P<entry>.+\.ninfer)\.part-\d{4}$")  # tools/artifact/writer.py naming
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SUMS_LINE = re.compile(r"^\s*([0-9A-Fa-f]{64})\s+\*?(\S.*?)\s*$")
_HASH_CHUNK = 8 * 1024 * 1024
```

(c) After `class DownloadConflict` add:

```python
class AddUnsupported(ValueError):
    """The repo holds nothing the engines can run."""


class ChecksumMismatch(RuntimeError):
    """A downloaded file does not match the checksum the repo publishes."""
```

(d) `RemoteFile`: add the field `sha256: str | None = None` after `size`.

(e) `DownloadJob`: add these fields after `completed_at`:

```python
    # Add model wizard jobs (kind "add"): staging folder, final target, what to fetch.
    kind: str = "folder"
    engine: str | None = None
    staging: Path | None = None
    target: Path | None = None
    fetch: list[RemoteFile] = field(default_factory=list)
    finals: list[str] = field(default_factory=list)
    sums_name: str | None = None
    verified: list[str] = field(default_factory=list)
```

and in `as_dict` add these keys to the returned dict:

```python
            "kind": self.kind,
            "engine": self.engine,
            "verified": list(self.verified),
            "resultPath": str(self.target) if self.kind == "add" and self.stage == "done" and self.target else None,
```

(f) `_remote_files`: replace `result.append(RemoteFile(name=name, size=size))` with

```python
        digest = _field(lfs, "sha256") if lfs is not None else None
        digest = digest.lower() if isinstance(digest, str) and _SHA_RE.fullmatch(digest.lower()) else None
        result.append(RemoteFile(name=name, size=size, sha256=digest))
```

(g) After `_downloadable_files` add:

```python
def parse_sha256sums(text: str) -> dict[str, str]:
    """``<hex>  <name>`` or ``<hex> *<name>`` lines (sha256sum's text and binary forms)."""
    sums: dict[str, str] = {}
    for line in text.splitlines():
        match = _SUMS_LINE.match(line)
        if match:
            name = match.group(2)
            sums[name[2:] if name.startswith("./") else name] = match.group(1).lower()
    return sums


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(_HASH_CHUNK):
            digest.update(block)
    return digest.hexdigest()
```

(h) Replace `_refresh` with:

```python
    def _refresh(self, job: DownloadJob) -> None:
        if job.kind == "add":
            # Staging is gone once the files are placed (or deleted); count from the plan then.
            if job.stage in ("verifying", "moving", "done"):
                job.received_bytes = job.total_bytes
            else:
                received = _folder_size(job.staging) if job.staging is not None else 0
                job.received_bytes = min(received, job.total_bytes) if job.total_bytes else received
        else:
            job.received_bytes = _folder_size(job.target_folder)
        if job.total_bytes:
            job.percent = min(100.0, job.received_bytes * 100.0 / job.total_bytes)
        elif job.stage == "done":
            job.percent = 100.0
        else:
            job.percent = 0.0
```

(i) In `_finish`, wrap the `_partial_targets` block so wizard jobs never register a resumable
partial target (their staging folder is deleted instead):

```python
            if job.kind != "add":
                key = self._target_key(job.target_folder)
                if stage in {"cancelled", "failed"} and job.target_folder.exists():
                    self._partial_targets.add(key)
                elif stage == "done":
                    self._partial_targets.discard(key)
```

(j) Add these methods to `DownloadManager` (after `_download_one`):

```python
    # ---- Add model wizard (control panel Stage B) -------------------------

    def _plan_add(self, value: str, entry: str | None, folder_root, ninfer_root
                  ) -> tuple[dict[str, Any], list[RemoteFile], RemoteFile | None]:
        repo = parse_repo(value)
        _, name = repo.split("/", 1)
        listing = [item for item in self._hub_files(repo) if _is_top_level_file(item.name)]
        by_name = {item.name: item for item in listing}
        sums = next((by_name[n] for n in _SUMS_NAMES if n in by_name), None)
        entries = sorted(n for n in by_name if _NINFER_ENTRY.fullmatch(n))
        architecture = None
        if entries:
            if entry is not None and entry not in entries:
                raise InvalidRepository(f"{entry} is not in that repo.")
            engine, root = "ninfer", Path(ninfer_root)
            chosen = entry or (entries[0] if len(entries) == 1 else None)
            parts = sorted(n for n in by_name if (m := _NINFER_PART.match(n)) and m.group("entry") == chosen)
            fetch = [by_name[chosen], *(by_name[n] for n in parts)] if chosen else []
            finals = [chosen, *parts] if chosen else []
            target = root / chosen if chosen else None
        elif "config.json" in by_name:
            info = describe_config(self._config(repo), name)
            if not info.supported:
                raise AddUnsupported(f"This model's design ({info.architecture or 'not named in its config.json'}) "
                                     "is not supported by your engines.")
            engine, root, chosen, architecture = "freetoken", Path(folder_root), None, info.architecture
            fetch, finals, target = _downloadable_files(listing), [name], Path(folder_root) / name
        else:
            raise AddUnsupported("This repo has no NInfer file (.ninfer) and no model folder (config.json), "
                                 "so it is not supported by your engines.")
        total = sum(item.size or 0 for item in fetch) + (sums.size or 0 if sums is not None and fetch else 0)
        try:
            free = max(0, int(self._disk_free(root)))
        except (TypeError, ValueError, OSError):
            free = 0
        plan = {
            "repo": repo, "name": name, "engine": engine, "kind": "ninfer" if engine == "ninfer" else "folder",
            "entries": entries, "entry": chosen, "architecture": architecture,
            "files": [{"name": item.name, "bytes": int(item.size or 0),
                       "check": "SHA256SUMS" if sums is not None else ("published" if item.sha256 else None)}
                      for item in fetch],
            "sumsFile": sums.name if sums is not None else None,
            "totalBytes": int(total), "root": str(root), "target": str(target) if target else None,
            "finals": finals, "exists": any((root / final).exists() for final in finals),
            "diskFreeBytes": free, "diskFits": free >= total,
        }
        return plan, fetch, sums

    def plan_add(self, value: str, entry: str | None = None, *, folder_root, ninfer_root) -> dict[str, Any]:
        return self._plan_add(value, entry, folder_root, ninfer_root)[0]

    def start_add(self, value: str, entry: str | None = None, *, folder_root, ninfer_root) -> DownloadJob:
        plan, fetch, sums = self._plan_add(value, entry, folder_root, ninfer_root)
        if not fetch:
            raise InvalidRepository("Pick which NInfer file to download.")
        if plan["exists"]:
            raise DownloadConflict("It is already on this PC, so it was not downloaded again. Add it from “On this PC”.")
        if not plan["diskFits"]:
            raise DownloadConflict(f"Not enough drive space: it needs {plan['totalBytes'] / GIB:.1f} GB and "
                                   f"{plan['diskFreeBytes'] / GIB:.1f} GB is free.")
        with self._lock:
            if self._active_id is not None:
                active = self._jobs.get(self._active_id)
                raise DownloadConflict(f"Another model download is already {active.stage if active else 'running'}.")
            job_id = f"download-{uuid.uuid4().hex[:12]}"
            staging = Path(plan["root"]) / f".incoming-{job_id}"
            job = DownloadJob(job_id=job_id, repo=plan["repo"], target_folder=staging, total_bytes=plan["totalBytes"],
                              kind="add", engine=plan["engine"], staging=staging, target=Path(plan["target"]),
                              fetch=[*([sums] if sums is not None else []), *fetch], finals=plan["finals"],
                              sums_name=sums.name if sums is not None else None)
            self._jobs[job_id] = job
            self._active_id = job_id
            threading.Thread(target=self._run_add, args=(job_id,), name=f"settings-add-{job_id[-6:]}", daemon=True).start()
            return job

    def latest_add(self) -> dict[str, Any] | None:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.kind == "add"]
            if not jobs:
                return None
            self._refresh(jobs[-1])
            return jobs[-1].as_dict()

    def _abandon(self, job_id: str, stage: str, error: str | None = None) -> None:
        """Spec error table: a failed or cancelled download leaves no partial files."""
        job = self._jobs[job_id]
        if job.staging is not None:
            shutil.rmtree(job.staging, ignore_errors=True)
        self._finish(job_id, stage, error)

    def _run_add(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
        self._set_stage(job_id, "downloading")
        try:
            job.staging.mkdir(parents=True)
            for item in job.fetch:
                if self._cancelled(job_id):
                    self._abandon(job_id, "cancelled")
                    return
                self._download_one(job.repo, job.staging, item.name)
                path = job.staging / item.name
                if not path.is_file():
                    raise RuntimeError(f"{item.name} did not arrive from Hugging Face.")
                if item.size is not None and path.stat().st_size != item.size:
                    raise RuntimeError(f"{item.name} arrived with the wrong size ({path.stat().st_size:,} bytes, "
                                       f"expected {item.size:,}).")
                if item.name.endswith((".safetensors", ".ninfer")) or _NINFER_PART.match(item.name):
                    release_completed_file(path)
                with self._lock:
                    job.files.append(item.name)
            self._set_stage(job_id, "verifying")
            expected = {item.name: item.sha256 for item in job.fetch if item.sha256}
            if job.sums_name:
                expected.update(parse_sha256sums((job.staging / job.sums_name).read_text(encoding="utf-8", errors="replace")))
            for item in job.fetch:
                want = expected.get(item.name)
                if want is None or item.name == job.sums_name:
                    continue
                if self._cancelled(job_id):
                    self._abandon(job_id, "cancelled")
                    return
                if sha256_file(job.staging / item.name) != want:
                    raise ChecksumMismatch(f"{item.name} does not match the checksum the repo publishes, "
                                           "so the download was deleted.")
                with self._lock:
                    job.verified.append(item.name)
            self._set_stage(job_id, "moving")
            self._place(job)
            self._finish(job_id, "done")
        except Exception as exc:  # noqa: BLE001 - a failed download must not kill the helper
            self._abandon(job_id, "failed", str(exc))

    @staticmethod
    def _place(job: DownloadJob) -> None:
        """Move the checked files into place without ever overwriting: os.link fails when the
        name exists (a plain rename would replace it); a folder is renamed only when its name
        is free. Anything already placed is taken back if a later file cannot be."""
        staging, target = job.staging, job.target
        if job.engine == "freetoken":
            shutil.rmtree(staging / ".cache", ignore_errors=True)
            if target.exists():
                raise DownloadConflict(f"{target.name} appeared in {target.parent} while downloading; nothing was overwritten.")
            staging.rename(target)
            return
        placed: list[Path] = []
        try:
            for name in job.finals:
                destination = target.parent / name
                try:
                    os.link(staging / name, destination)
                except FileExistsError as exc:
                    raise DownloadConflict(f"{name} appeared in {destination.parent} while downloading; "
                                           "nothing was overwritten.") from exc
                placed.append(destination)
        except BaseException:
            for path in placed:
                path.unlink(missing_ok=True)
            raise
        shutil.rmtree(staging, ignore_errors=True)
```

(k) `__all__`: add `"AddUnsupported"`, `"ChecksumMismatch"`, `"parse_sha256sums"`, `"sha256_file"`.

- [ ] **Step 4: Run the tests**

`PYTEST tests/settings/test_download_add.py tests/settings/test_download.py tests/settings/test_settings_import_safety.py`
Expected: all pass. The old download tests are unchanged, which confirms the resumable
"folder" path still works as before.

- [ ] **Step 5: Commit (controller)**

```bash
git add python/freetoken/daemon/settings/download.py tests/settings/test_download_add.py
git commit -m "feat(settings): checked, never-overwriting downloads of NInfer files and model folders

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Pi sync (`pi_sync.py`)

**Files:**
- Create: `python/freetoken/daemon/settings/pi_sync.py`
- Test: `tests/settings/test_pi_sync.py`

**Interfaces:**
- Produces:
  - `PROVIDER = "freetoken-local"`, `COPIED_FIELDS`, `FALLBACK`, `BACKUPS_KEPT = 20`,
    `default_agent_dir()`;
  - `PiSync(agent_dir=None, *, now=None, enabled=True)`;
  - `.add(model_id, name, engine, engines_by_id) -> result`;
  - `.remove(model_id) -> result`;
  - result = `{"status": "updated"|"unchanged"|"not_updated", "message": str, "notes": [str]}`.
- Test helpers other tasks import: `MODELS`, `SETTINGS`, `ENGINES`, `write_pi(folder, models=MODELS, settings=SETTINGS)`.

- [ ] **Step 1: Write the failing tests** (`tests/settings/test_pi_sync.py`)

```python
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


def test_sync_off_says_so(tmp_path):
    result = PiSync(tmp_path, enabled=False).add("x", "X", "ninfer", {})
    assert result == {"status": "not_updated", "message": "Pi sync is off on this helper.", "notes": []}
```

- [ ] **Step 2: Run to see them fail**

`PYTEST tests/settings/test_pi_sync.py`
Expected: `ModuleNotFoundError: No module named 'freetoken.daemon.settings.pi_sync'`.

- [ ] **Step 3: Write `python/freetoken/daemon/settings/pi_sync.py`**

```python
"""Keep Pi's model list in step with the control panel (spec section 7, "Pi sync").

Pi reads Windows files: C:\\Users\\<user>\\.pi\\agent\\models.json and settings.json, which
WSL sees under /mnt/c/Users/<user>/.pi/agent (FREETOKEN_PI_AGENT_DIR overrides). These are
Jay's files, so this module:

- changes only the ``freetoken-local`` provider's ``models`` list and the
  ``freetoken-local/<id>`` rows of settings.json ``enabledModels``; every other provider, the
  provider's own fields and every other settings key stay exactly as they were;
- backs up both files (byte for byte, as <file>.bak-<time>) before every change, newest 20 kept;
- keeps each file's indent, final newline, CRLF line ends and UTF-8 BOM;
- never raises: a missing folder, bad JSON or a missing provider answers "not_updated" with
  the reason, and the page shows "Pi not updated" while the add or remove itself stands.

A new entry copies ``COPIED_FIELDS`` from the first Pi entry whose id is a registry model on
the same engine. ``samplingParams`` is per model and is not copied (NInfer's clampParams keeps
an app's values in range anyway). With no such neighbour, ``FALLBACK`` is used and a note says
so. Removing Pi's default model leaves ``defaultModel`` alone and a note says so: picking
another default is Jay's call.
"""

from __future__ import annotations

import codecs
import copy
import datetime as _dt
import getpass
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

PROVIDER = "freetoken-local"
COPIED_FIELDS = ("reasoning", "input", "cost", "contextWindow", "maxTokens", "thinkingLevelMap")
FALLBACK = {"reasoning": True, "input": ["text"], "contextWindow": 131072, "maxTokens": 32768,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}
BACKUPS_KEPT = 20


def default_agent_dir() -> Path:
    configured = os.environ.get("FREETOKEN_PI_AGENT_DIR")
    if configured:
        return Path(configured)
    return Path("/mnt/c/Users") / getpass.getuser() / ".pi" / "agent"


def _result(status: str, message: str, notes: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "message": message, "notes": list(notes or [])}


def _indent(text: str) -> int | str:
    for line in text.splitlines()[1:]:
        stripped = line.lstrip(" \t")
        if stripped and len(stripped) != len(line):
            lead = line[: len(line) - len(stripped)]
            return "\t" if lead.startswith("\t") else len(lead)
    return 2


def _render(doc: Any, original: str, bom: bool) -> bytes:
    text = json.dumps(doc, indent=_indent(original), ensure_ascii=False)
    if original.endswith("\n"):
        text += "\n"
    if "\r\n" in original:
        text = text.replace("\n", "\r\n")
    return (codecs.BOM_UTF8 if bom else b"") + text.encode("utf-8")


class PiSync:
    def __init__(self, agent_dir: str | os.PathLike[str] | None = None, *,
                 now: Callable[[], _dt.datetime] | None = None, enabled: bool = True) -> None:
        self.agent_dir = Path(agent_dir) if agent_dir is not None else default_agent_dir()
        self.models_path = self.agent_dir / "models.json"
        self.settings_path = self.agent_dir / "settings.json"
        self._now = now or _dt.datetime.now
        self.enabled = enabled

    def add(self, model_id: str, name: str, engine: str, engines_by_id: Mapping[str, str]) -> dict[str, Any]:
        key = f"{PROVIDER}/{model_id}"

        def edit(rows: list, settings: dict) -> list[str]:
            notes: list[str] = []
            if not any(isinstance(row, dict) and row.get("id") == model_id for row in rows):
                neighbour = next((row for row in rows if isinstance(row, dict)
                                  and engines_by_id.get(row.get("id")) == engine), None)
                entry: dict[str, Any] = {"id": model_id, "name": name}
                if neighbour is not None:
                    entry.update({field: copy.deepcopy(neighbour[field]) for field in COPIED_FIELDS if field in neighbour})
                    notes.append(f"Pi's limits for it were copied from {neighbour.get('name') or neighbour.get('id')}.")
                else:
                    entry.update(copy.deepcopy(FALLBACK))
                    notes.append("No other model on the same engine was in Pi to copy limits from, so Pi uses "
                                 f"{FALLBACK['contextWindow']:,} tokens of context for it.")
                rows.append(entry)
            enabled = settings.get("enabledModels")
            if isinstance(enabled, list) and key not in enabled:
                enabled.append(key)
            return notes

        return self._change(edit)

    def remove(self, model_id: str) -> dict[str, Any]:
        key = f"{PROVIDER}/{model_id}"

        def edit(rows: list, settings: dict) -> list[str]:
            rows[:] = [row for row in rows if not (isinstance(row, dict) and row.get("id") == model_id)]
            enabled = settings.get("enabledModels")
            if isinstance(enabled, list):
                enabled[:] = [item for item in enabled if item != key]
            if settings.get("defaultProvider") == PROVIDER and settings.get("defaultModel") == model_id:
                return [f"Pi still starts with {model_id} by default; pick another default model in Pi."]
            return []

        return self._change(edit)

    def _change(self, edit: Callable[[list, dict], list[str]]) -> dict[str, Any]:
        if not self.enabled:
            return _result("not_updated", "Pi sync is off on this helper.")
        raw: dict[Path, bytes] = {}
        try:
            for path in (self.models_path, self.settings_path):
                raw[path] = path.read_bytes()
        except OSError as exc:
            return _result("not_updated", f"Pi's files could not be read in {self.agent_dir} ({exc.strerror or exc}).")
        texts = {path: data.decode("utf-8-sig", errors="replace") for path, data in raw.items()}
        try:
            models, settings = json.loads(texts[self.models_path]), json.loads(texts[self.settings_path])
        except ValueError as exc:
            return _result("not_updated", f"One of Pi's files is not valid JSON ({exc}).")
        providers = models.get("providers") if isinstance(models, dict) else None
        provider = providers.get(PROVIDER) if isinstance(providers, dict) else None
        if not isinstance(settings, dict) or not isinstance(provider, dict) or not isinstance(provider.get("models"), list):
            return _result("not_updated", f"Pi's models.json has no {PROVIDER} model list.")
        new_models, new_settings = copy.deepcopy(models), copy.deepcopy(settings)
        notes = edit(new_models["providers"][PROVIDER]["models"], new_settings)
        writes = [(path, doc) for path, doc, old in ((self.models_path, new_models, models),
                                                     (self.settings_path, new_settings, settings)) if doc != old]
        if not writes:
            return _result("unchanged", "Pi already matched.", notes)
        try:
            stamp = self._now().strftime("%Y%m%d-%H%M%S-%f")
            for path in (self.models_path, self.settings_path):
                path.with_name(f"{path.name}.bak-{stamp}").write_bytes(raw[path])
            for path, doc in writes:
                self._atomic_write(path, _render(doc, texts[path], raw[path].startswith(codecs.BOM_UTF8)))
            self._prune()
        except OSError as exc:
            return _result("not_updated", f"Pi's files could not be written ({exc.strerror or exc}); "
                                          f"backups are in {self.agent_dir}.", notes)
        return _result("updated", "Pi's model list was updated.", notes)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp-freetoken")
        temporary.write_bytes(data)
        os.replace(temporary, path)

    def _prune(self) -> None:
        for path in (self.models_path, self.settings_path):
            prefix = path.name + ".bak-"
            names = sorted((p.name for p in self.agent_dir.iterdir() if p.name.startswith(prefix)), reverse=True)
            for name in names[BACKUPS_KEPT:]:
                (self.agent_dir / name).unlink(missing_ok=True)


__all__ = ["BACKUPS_KEPT", "COPIED_FIELDS", "FALLBACK", "PROVIDER", "PiSync", "default_agent_dir"]
```

- [ ] **Step 4: Run the tests**

`PYTEST tests/settings/test_pi_sync.py tests/settings/test_settings_import_safety.py`
Expected: all pass.

- [ ] **Step 5: Commit (controller)**

```bash
git add python/freetoken/daemon/settings/pi_sync.py tests/settings/test_pi_sync.py
git commit -m "feat(settings): keep Pi's freetoken-local model list in step, with backups

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: The panel's server side for add and remove (`panel.py`, `app.py`)

**Files:**
- Modify: `python/freetoken/daemon/settings/panel.py`
- Modify: `python/freetoken/daemon/settings/app.py`
- Modify: `tests/settings/test_routes.py` (version `2.1.0`)
- Test: `tests/settings/test_panel_add_remove.py`

**Interfaces:**
- Consumes: Task 1 (`detect`, `model_files`, `PART_RE`), Task 3 (`plan_add`, `start_add`,
  `latest_add`, `get`, `cancel`, `InvalidRepository`, `AddUnsupported`, `DownloadConflict`),
  Task 4 (`PiSync.add/remove`), and Stage A (`_save`, `_ask`, `profiles.delete`, `profile_id`).
- Produces:
  - `PanelService(..., downloads=None, pi=None, add_roots=None, profile_deleted=None)`, with
    public attributes `downloads`, `pi`, `add_roots` and `profile_deleted`;
  - `default_add_roots()`;
  - methods `add_info()`, `detect_path(path)`, `plan_download(link, entry)`,
    `start_download(link, entry)`, `download_status(job_id)`, `cancel_download(job_id)`,
    `add_model(path, model_id, name, ram_need, revision)`,
    `remove_model(model_id, delete_files, revision)`;
  - routes, all under `/api/panel`:
    - `GET /add/info`;
    - `POST /add/detect {path}`;
    - `POST /add/plan {link, entry}`;
    - `POST /add/downloads {link, entry}`;
    - `GET /add/downloads/{id}`;
    - `POST /add/downloads/{id}/cancel`;
    - `POST /models {revision, path, id, name, ramNeedGB}`;
    - `POST /models/{id}/remove {revision, deleteFiles}`.
  - The add result is `{status: "added", id, name, revision, adjusted: [str], pi, restarting, held}`.
  - The remove result is `{status: "removed", id, name, revision, profile, files: null|{deleted,
    paths, message}, pi, …}`.
  - New error codes:
    - 422 `not_supported`, `bad_link`;
    - 409 `already_added`, `files_shared`, `files_unsafe`, `files_missing`, `download_conflict`;
    - 503 `switcher_unknown`, `unload_failed`, `downloads_off`;
    - 502 `hub_error`.
  - `models()` rows gain `aliases`; `model_view()` gains `artifact`.

- [ ] **Step 1: Write the failing tests** (`tests/settings/test_panel_add_remove.py`)

```python
"""Add and remove models on the control panel (Stage B). Review focus 2, 3 and 5 live here."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from freetoken.daemon.settings.app import create_app
from freetoken.daemon.settings.boot_parser import BootFile
from freetoken.daemon.settings.download import DownloadManager
from freetoken.daemon.settings.panel import PanelService
from freetoken.daemon.settings.pi_sync import PROVIDER, PiSync
from freetoken.daemon.settings.process_manager import ProcessManager
from freetoken.daemon.settings.profiles_manager import ProfilesManager
from freetoken.daemon.settings.registry import RegistryStore, find_model
from freetoken.daemon.settings.swap_config import SwapConfigWriter, extract_model_blocks, render_config
from tests.settings.registry_fixtures import five
from tests.settings.test_download_add import Hub
from tests.settings.test_model_detect import write_folder, write_v2, write_v3
from tests.settings.test_panel_routes import FakeEstimates, FakeSwitcher, checker
from tests.settings.test_pi_sync import write_pi

GIB = 1024 ** 3
FIVE_IDS = [m["id"] for m in five()["models"]]


@pytest.fixture
def box(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    ninfer, models = home / "ninfer-work" / "models", home / "models"
    ninfer.mkdir(parents=True)
    models.mkdir(parents=True)
    write_v2(ninfer / "quasar_27b_nvfp4.ninfer")
    write_v3(ninfer / "fable_27b_nvfp4.ninfer")
    write_v3(ninfer / "twin_nvfp4.ninfer")
    boot = tmp_path / "boot-2020.ps1"
    boot.write_text("& $launcher `\n    -KVDtype 'fp8' `\n    -Port 2020\n", encoding="utf-8")
    cfg = tmp_path / "llama-swap" / "config.yaml"
    cfg.parent.mkdir()
    binary = tmp_path / "llama-swap-bin"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    writer = SwapConfigWriter(cfg, binary=binary, runner=checker())
    switcher = FakeSwitcher(cfg)
    profiles = ProfilesManager(tmp_path / "boot-profiles.json", boot_file=boot)
    store = RegistryStore(tmp_path / "freetoken" / "registry.json")
    pi_dir = write_pi(tmp_path / "pi")
    hub = Hub({})
    downloads = DownloadManager(models, api_factory=hub, config_fetcher=lambda _: hub.config or {},
                                snapshot_downloader=hub.snapshot, pc_memory=64 * GIB, card_memory=32 * GIB,
                                disk_free=lambda _: 10 ** 12)
    service = PanelService(
        store=store, writer=writer, switcher=switcher, profiles=profiles,
        boot_file=lambda: BootFile(boot), default_boot=lambda: boot, estimate_service=FakeEstimates(),
        card_probe=lambda: {"totalBytes": 32 * GIB, "usedBytes": 2 * GIB}, windows_free_probe=lambda: 40 * GIB,
        spawn=lambda fn, *args: None, sleep=lambda _: None,
        downloads=downloads, pi=PiSync(pi_dir), add_roots={"folder": models, "ninfer": ninfer},
    )
    proc = ProcessManager(boot_file=boot, stop_script=tmp_path / "stop.ps1", log_path=tmp_path / "server.log",
                          lock_path=tmp_path / "gpu.lock", runner=lambda *a, **k: None,
                          readiness=lambda: {"state": "unreachable"}, sleep=lambda _: None, poll_interval=0)
    app = create_app(boot_file=boot, process_manager=proc, profiles=profiles, log_path=tmp_path / "server.log",
                     static_path=tmp_path / "missing.html", panel=service)
    store.save(five(), expected_revision=None)
    writer.write(render_config(store.load()[0], {}))
    return SimpleNamespace(client=TestClient(app), app=app, service=service, store=store, cfg=cfg, switcher=switcher,
                           profiles=profiles, pi=pi_dir, ninfer=ninfer, models=models, hub=hub, boot=boot, home=home)


def revision(box):
    return box.client.get("/api/panel/registry").json()["revision"]


def pi_ids(box):
    return [m["id"] for m in json.loads((box.pi / "models.json").read_text())["providers"][PROVIDER]["models"]]


def enabled(box):
    return json.loads((box.pi / "settings.json").read_text())["enabledModels"]


def add(box, path, **identity):
    body = {"revision": revision(box), "path": str(path), "id": "small_9b", "name": "Small 9B (NInfer)",
            "ramNeedGB": 6, **identity}
    return box.client.post("/api/panel/models", json=body)


def remove(box, model_id, **body):
    return box.client.post(f"/api/panel/models/{model_id}/remove", json={"revision": revision(box), **body})


def ids(box):
    return [m["id"] for m in box.store.load()[0]["models"]]


# ---- detect and add ----
def test_detect_says_what_a_path_is_and_suggests_an_identity(box):
    write_v3(box.ninfer / "small_9b.ninfer", parts=1)
    found = box.client.post("/api/panel/add/detect", json={"path": "~/ninfer-work/models/small_9b.ninfer"}).json()
    assert (found["kind"], found["runtime"], found["already"]) == ("ninfer", "ninfer-upstream", None)
    assert found["suggested"]["id"] == "small_9b" and len(found["files"]) == 2
    write_v2(box.ninfer / "quasar-27b.ninfer")
    clash = box.client.post("/api/panel/add/detect", json={"path": str(box.ninfer / "quasar-27b.ninfer")}).json()
    assert clash["suggested"]["id"] == "quasar-27b-2"
    known = box.client.post("/api/panel/add/detect", json={"path": str(box.ninfer / "quasar_27b_nvfp4.ninfer")}).json()
    assert known["runtime"] == "ninfer" and known["already"] == "Qwen3.8 27B QUASAR NVFP4 (NInfer, DFlash2)"
    info = box.client.get("/api/panel/add/info").json()
    assert info == {"roots": {"folder": str(box.models), "ninfer": str(box.ninfer)}, "download": None}


def test_adding_a_ninfer_model_updates_the_list_the_switcher_and_pi(box):
    write_v3(box.ninfer / "small_9b.ninfer")
    answer = add(box, "~/ninfer-work/models/small_9b.ninfer")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert (body["status"], body["name"], body["pi"]["status"]) == ("added", "Small 9B (NInfer)", "updated")
    model = find_model(box.store.load()[0], "small_9b")
    assert (model["engine"], model["runtime"], model["artifact"]) == ("ninfer", "ninfer-upstream", "~/ninfer-work/models/small_9b.ninfer")
    assert model["ramNeedGB"] == 6 and model["idleMinutes"] is None and model["overrides"] == {}
    assert "small_9b" in extract_model_blocks(box.cfg.read_text())
    assert pi_ids(box)[-1] == "small_9b" and enabled(box)[-1] == f"{PROVIDER}/small_9b"
    pi_entry = json.loads((box.pi / "models.json").read_text())["providers"][PROVIDER]["models"][-1]
    assert pi_entry["contextWindow"] == 150000  # copied from quasar-27b, the NInfer neighbour
    row = next(r for r in box.client.get("/api/panel/models").json()["models"] if r["id"] == "small_9b")
    assert row["aliases"] == [] and row["runtime"] == "ninfer-upstream"


def test_the_server_checks_everything_again_at_save(box):
    (box.ninfer / "notes.txt").write_text("hi")
    refused = add(box, box.ninfer / "notes.txt")
    assert refused.status_code == 422 and refused.json()["code"] == "not_supported"
    again = add(box, box.ninfer / "quasar_27b_nvfp4.ninfer")
    assert again.status_code == 409 and again.json()["code"] == "already_added"
    write_v2(box.ninfer / "small_9b.ninfer")
    for identity, field in (({"id": "Bad Id"}, "add.id"), ({"id": "qwen3.8-flash-next-nvfp4"}, "add.id"),
                            ({"id": "quasar-27b"}, "add.id"), ({"name": ""}, "add.name"),
                            ({"ramNeedGB": "lots"}, "add.ramNeedGB"), ({"ramNeedGB": 900}, "add.ramNeedGB")):
        answer = add(box, box.ninfer / "small_9b.ninfer", **identity)
        assert answer.status_code == 422 and answer.json()["detail"][0]["field"] == field, (identity, answer.text)
    stale = box.client.post("/api/panel/models", json={"revision": "old", "path": str(box.ninfer / "small_9b.ninfer"),
                                                       "id": "small_9b", "name": "S", "ramNeedGB": 6})
    assert stale.json()["code"] == "stale_revision"
    assert ids(box) == FIVE_IDS and "small_9b" not in pi_ids(box)


def test_adding_works_while_the_switcher_state_is_unknown(box):
    box.switcher.up = False  # unknown: a model new to the list has no entry yet, so it cannot be loaded
    write_v2(box.ninfer / "small_9b.ninfer")
    assert add(box, box.ninfer / "small_9b.ninfer").status_code == 200


def test_a_small_freetoken_model_gets_the_defaults_fitted_to_it(box):
    folder = write_folder(box.models / "Tiny-Llama", max_position_embeddings=8192)
    answer = add(box, folder, id="tiny-llama", name="Tiny Llama (FreeToken)", ramNeedGB=2)
    assert answer.status_code == 200, answer.text
    assert find_model(box.store.load()[0], "tiny-llama")["overrides"] == {"ContextTokens": 8192}
    assert "8,192" in answer.json()["adjusted"][0]
    assert box.profiles.get("model-tiny-llama")["settings"]["ModelPath"] == str(folder)


def test_pi_out_of_reach_still_adds_and_says_so(box):
    for path in box.pi.iterdir():
        path.unlink()
    box.pi.rmdir()
    write_v2(box.ninfer / "small_9b.ninfer")
    body = add(box, box.ninfer / "small_9b.ninfer").json()
    assert body["status"] == "added" and body["pi"]["status"] == "not_updated"
    assert "small_9b" in ids(box)


# ---- remove ----
def test_removing_a_loaded_model_unloads_it_first(box):
    box.switcher.states = {"quasar-27b": "ready"}
    answer = remove(box, "quasar-27b")
    assert answer.status_code == 200, answer.text
    assert box.switcher.calls == [("unload", "quasar-27b")]
    assert "quasar-27b" not in ids(box) and "quasar-27b" not in extract_model_blocks(box.cfg.read_text())
    assert "quasar-27b" not in pi_ids(box) and f"{PROVIDER}/quasar-27b" not in enabled(box)
    assert (box.ninfer / "quasar_27b_nvfp4.ninfer").is_file(), "files stay unless asked"
    assert answer.json()["files"] is None and answer.json()["pi"]["notes"]  # quasar was Pi's default


def test_remove_is_refused_while_the_switcher_state_is_unknown(box):
    box.switcher.up = False
    before = box.store.load()[1]
    answer = remove(box, "quasar-27b")
    assert answer.status_code == 503 and answer.json()["code"] == "switcher_unknown"
    assert box.store.load()[1] == before and "quasar-27b" in pi_ids(box)


def test_a_failed_unload_removes_nothing(box):
    box.switcher.states, box.switcher.unload_ok = {"quasar-27b": "ready"}, False
    before, text = box.store.load()[1], box.cfg.read_text()
    answer = remove(box, "quasar-27b", deleteFiles=True)
    assert answer.status_code == 503 and answer.json()["code"] == "unload_failed"
    assert box.store.load()[1] == before and box.cfg.read_text() == text
    assert (box.ninfer / "quasar_27b_nvfp4.ninfer").is_file() and "quasar-27b" in pi_ids(box)


def test_delete_files_takes_the_entry_and_its_parts_and_nothing_else(box):
    write_v3(box.ninfer / "small_9b.ninfer", parts=2)
    assert add(box, box.ninfer / "small_9b.ninfer").status_code == 200
    answer = remove(box, "small_9b", deleteFiles=True)
    assert answer.status_code == 200, answer.text
    assert answer.json()["files"]["deleted"] is True and len(answer.json()["files"]["paths"]) == 3
    assert sorted(p.name for p in box.ninfer.iterdir()) == [
        "fable_27b_nvfp4.ninfer", "quasar_27b_nvfp4.ninfer", "twin_nvfp4.ninfer"]


def test_files_another_model_uses_are_never_deleted(box):
    doc, rev = box.store.load()
    doc["models"].append({**find_model(doc, "quasar-27b"), "id": "quasar-copy", "name": "QUASAR copy"})
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "quasar-copy", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_shared"
    assert "quasar-copy" in ids(box) and (box.ninfer / "quasar_27b_nvfp4.ninfer").is_file()


def test_parts_another_model_reads_are_never_deleted(box):
    write_v3(box.ninfer / "twin_nvfp4.ninfer", parts=1)          # twin is now split
    write_v3(box.ninfer / "copy.ninfer", part_names=["twin_nvfp4.ninfer.part-0001"])  # an entry reading twin's part
    doc, rev = box.store.load()
    doc["models"].append({**find_model(doc, "twin-27b"), "id": "twin-copy", "name": "Twin copy",
                          "artifact": "~/ninfer-work/models/copy.ninfer"})
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "twin-copy", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_shared"
    assert (box.ninfer / "twin_nvfp4.ninfer.part-0001").is_file() and (box.ninfer / "copy.ninfer").is_file()


def test_delete_files_refuses_a_folder_that_is_not_a_model(box):
    odd = box.models / "odd"
    odd.mkdir()
    (odd / "keep.txt").write_text("x")
    doc, rev = box.store.load()
    find_model(doc, "qwen3.8-flash")["artifact"] = "~/models/odd"
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "qwen3.8-flash", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_unsafe"
    assert (odd / "keep.txt").is_file() and "qwen3.8-flash" in ids(box)


def test_the_home_folder_is_never_deleted(box):
    (box.home / "config.json").write_text("{}")
    doc, rev = box.store.load()
    find_model(doc, "qwen3.8-flash")["artifact"] = "~/"
    box.store.save(doc, expected_revision=rev)
    answer = remove(box, "qwen3.8-flash", deleteFiles=True)
    assert answer.status_code == 409 and answer.json()["code"] == "files_unsafe"
    assert (box.home / "config.json").is_file()


def test_removing_a_freetoken_model_drops_its_profile_and_moves_the_helper_off_it(box):
    box.service._ensure_profiles(box.store.load()[0])
    activated = box.client.post("/api/profiles/model-qwen3.8-flash/activate")
    assert activated.status_code == 200, activated.text
    assert Path(box.app.state.boot_file.path) != box.boot
    answer = remove(box, "qwen3.8-flash")
    assert answer.status_code == 200, answer.text
    assert box.profiles.get("model-qwen3.8-flash") is None
    assert Path(box.app.state.boot_file.path) == box.boot


def test_a_removed_model_leaves_no_hold_behind(box):
    box.switcher.states = {"quasar-27b": "ready"}
    view = box.client.get("/api/panel/views/model/quasar-27b").json()
    assert view["artifact"] == "~/ninfer-work/models/quasar_27b_nvfp4.ninfer"
    settings = {k: v for k, v in view["settings"].items() if not k.startswith("model.")}
    settings["draft-tokens"] = 5
    held = box.client.put("/api/panel/models/quasar-27b", json={
        "revision": view["revision"], "settings": settings, "identity": {}, "activePreset": None, "whenLoaded": "next-time"})
    assert held.json()["held"] == ["quasar-27b"]
    box.switcher.up, box.switcher.refused = False, True   # down: nothing is loaded, holds are kept
    assert remove(box, "quasar-27b").status_code == 200
    assert "quasar-27b" not in json.loads(box.service.holds_path.read_text())


# ---- downloads through the panel ----
def wait_job(box, job_id):
    for _ in range(400):
        job = box.client.get(f"/api/panel/add/downloads/{job_id}").json()
        if job["stage"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.005)
    raise AssertionError(job)


def test_a_link_downloads_checks_and_adds(box, tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    write_v3(source / "small_9b.ninfer", parts=1)
    box.hub.files = {p.name: p.read_bytes() for p in sorted(source.iterdir())}
    plan = box.client.post("/api/panel/add/plan", json={"link": "https://huggingface.co/owner/small-9b"}).json()
    assert plan["engine"] == "ninfer" and plan["target"] == str(box.ninfer / "small_9b.ninfer")
    job = box.client.post("/api/panel/add/downloads", json={"link": "owner/small-9b"}).json()
    done = wait_job(box, job["id"])
    assert done["stage"] == "done" and len(done["verified"]) == 2, done
    assert box.client.get("/api/panel/add/info").json()["download"]["id"] == job["id"]
    found = box.client.post("/api/panel/add/detect", json={"path": done["resultPath"]}).json()
    assert found["runtime"] == "ninfer-upstream" and found["already"] is None
    assert add(box, done["resultPath"], id=found["suggested"]["id"]).status_code == 200


def test_a_bad_checksum_leaves_nothing_behind(box):
    box.hub.files = {"small_9b.ninfer": b"x" * 64, "SHA256SUMS": f"{'0' * 64}  small_9b.ninfer\n".encode()}
    job = box.client.post("/api/panel/add/downloads", json={"link": "owner/small-9b"}).json()
    done = wait_job(box, job["id"])
    assert done["stage"] == "failed" and "checksum" in done["error"]
    assert not (box.ninfer / "small_9b.ninfer").exists()
    assert not [p for p in box.ninfer.iterdir() if p.name.startswith(".incoming-")]
    assert ids(box) == FIVE_IDS


def test_link_errors_are_plain(box):
    assert box.client.post("/api/panel/add/plan", json={"link": "https://example.com/x/y"}).json()["code"] == "bad_link"
    box.hub.files = {"README.md": b"hi"}
    answer = box.client.post("/api/panel/add/plan", json={"link": "owner/nothing"})
    assert answer.status_code == 422 and "not supported by your engines" in answer.json()["message"]
    assert box.client.get("/api/panel/add/downloads/nope").status_code == 404
```

- [ ] **Step 2: Run to see them fail**

`PYTEST tests/settings/test_panel_add_remove.py`
Expected: `TypeError: PanelService.__init__() got an unexpected keyword argument 'downloads'`.

- [ ] **Step 3: Edit `panel.py`**

(a) Docstring: add a paragraph at the end:

```
Stage B adds models (a file or folder on the PC, or a Hugging Face download checked against
the repo's published checksums) and removes them (unload first, then save, then the model's
profile, then its files when asked, then Pi). Detection runs again on the server at Save.
Removing needs a known switcher state: P5 stops a removed model that is still loaded, so an
unknown state could otherwise pull a model out from under a chat.
```

(b) Imports. Add `import shutil`. Extend the existing imports as shown:

```python
from .dials import (DIAL_BY_NAME, DIALS, GROUP_INFO, MODEL_AWARE_DIALS, Dial, adapt_dial, dial_value_for_display,
                    validate_settings)
from .download import AddUnsupported, DownloadConflict, InvalidRepository
from .model_detect import detect, model_files
from .pi_sync import PiSync
from .profiles_manager import ProfileError, ProfileValidationError
from .registry import (
    ACTIVE, ENGINE_LABELS, ENGINES, ID_RULE, MODEL_ID_RE, NAME_MAX, RegistryCorrupt, RegistryError, RegistryMissing,
    RegistryValidationError, StaleRevision, base_settings, canonical_engine_settings, canonicalize, differences,
    effective_settings, engine_defaults, expand, find_model, freetoken_profile_settings, idle_minutes, preset_values,
    same, validate_registry,
)
```

(c) After `_default_spawn` add these module-level helpers:

```python
def default_add_roots() -> dict[str, Path]:
    """Spec section 7: ~/models/<name> for FreeToken folders, ~/ninfer-work/models/ for NInfer."""
    home = Path.home()
    return {"folder": Path(os.environ.get("FREETOKEN_MODELS_DIR") or home / "models"),
            "ninfer": Path(os.environ.get("FREETOKEN_NINFER_MODELS_DIR") or home / "ninfer-work" / "models")}


def _home_path(path: str) -> str:
    home = str(Path.home()).rstrip("/")
    return "~/" + path[len(home) + 1:] if path.startswith(home + "/") else path


def _fit_to_model(doc: Mapping[str, Any], entry: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """FreeToken defaults a new model cannot use become its own values at its limit.

    Without this the add fails, and a registry holding such a model would refuse every later
    save (_save checks every FreeToken model's limits). Checked 2026-09-25 against five() +
    a Llama config with max_position_embeddings 8192: only ContextTokens (262,144 by default)
    is over, and 8192 clears validate_settings; a 64-expert, 24-layer Qwen3-MoE also needs
    MoECacheSize 5332 -> 1536. Dials stored in another form (GpuOwnedLayers "auto:{n}") are
    left for Jay: the save then names them."""
    info = read_model(expand(entry["artifact"]))
    effective = effective_settings(doc, entry)
    effective.pop("ModelPath", None)
    overrides: dict[str, Any] = {}
    notes: list[str] = []
    for error in validate_settings(effective, info):
        dial = DIAL_BY_NAME.get(error["field"])
        if dial is None or dial.stored_as or dial.name not in MODEL_AWARE_DIALS or dial.name in overrides:
            continue
        value, bounds = effective.get(dial.name), adapt_dial(dial, info)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if bounds.get("max") is not None and value > bounds["max"]:
            overrides[dial.name], word = bounds["max"], "most"
        elif bounds.get("min") is not None and value < bounds["min"]:
            overrides[dial.name], word = bounds["min"], "least"
        else:
            continue
        notes.append(f"{dial.plain or dial.name} set to {overrides[dial.name]:,}, the {word} this model allows.")
    return overrides, notes
```

(d) `PanelService.__init__`: add the keyword parameters
`downloads=None, pi=None, add_roots=None, profile_deleted=None` after `restart_wait_s`, and
in the body:

```python
        self.downloads = downloads
        self.pi = pi if pi is not None else PiSync(enabled=False)
        self.add_roots = {k: Path(v) for k, v in (add_roots or default_add_roots()).items()}
        # Called with profiles.delete()'s result so the app can move the helper back to its
        # default boot file when the removed model's profile was the active one.
        self.profile_deleted = profile_deleted
```

(e) `_save`, two changes. Replace

```python
            if known is None:
                moved = [m for m in after if m not in holds and before.get(m) != after[m]]
```
with

```python
            # A removed model's hold goes with it (Stage B): its entry is no longer in the list.
            holds = {model_id: block for model_id, block in holds.items() if model_id in after}
            if known is None:
                # A model new to the list has no entry in the switcher yet, so it cannot be
                # loaded: adding one while the state is unknown is safe (Stage B).
                moved = [m for m in after if m in before and m not in holds and before[m] != after[m]]
```

(f) `model_view`: add `"artifact": model["artifact"],` to the returned dict (after
`"runtimeLabel"`). `models()`: add `"aliases": list(model.get("aliases") or []),` to each row.

(g) Add a new section to `PanelService`, before `# ---- holds and start-up ----`:

```python
    # ---- add and remove (Stage B) ----
    @staticmethod
    def _taken(doc: Mapping[str, Any]) -> list[str]:
        return [m["id"] for m in doc["models"]] + [a for m in doc["models"] for a in m.get("aliases") or []]

    @staticmethod
    def _owner(doc: Mapping[str, Any], name: str) -> str | None:
        wanted = name.lower()
        for model in doc["models"]:
            if model["id"].lower() == wanted or any(str(a).lower() == wanted for a in model.get("aliases") or []):
                return model["name"]
        return None

    @staticmethod
    def _already(doc: Mapping[str, Any], path: str) -> str | None:
        real = os.path.realpath(path)
        for model in doc["models"]:
            if os.path.realpath(expand(model["artifact"])) == real:
                return model["name"]
        return None

    def add_info(self) -> dict[str, Any]:
        return {"roots": {name: str(path) for name, path in self.add_roots.items()},
                "download": self.downloads.latest_add() if self.downloads is not None else None}

    def detect_path(self, path: str) -> dict[str, Any]:
        doc, _ = self.store.load()
        found = detect(path, taken=self._taken(doc))
        found["already"] = self._already(doc, found["path"]) if found["kind"] != "unsupported" else None
        return found

    def _hub(self, method: str, *args: Any) -> Any:
        if self.downloads is None:
            raise PanelError(503, "downloads_off", "Downloads are not available on this helper.")
        try:
            return getattr(self.downloads, method)(*args, folder_root=self.add_roots["folder"],
                                                   ninfer_root=self.add_roots["ninfer"])
        except AddUnsupported as exc:
            raise PanelError(422, "not_supported", str(exc)) from exc
        except InvalidRepository as exc:
            raise PanelError(422, "bad_link", str(exc)) from exc
        except DownloadConflict as exc:
            raise PanelError(409, "download_conflict", str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - the Hub's own errors (no such repo, no network), in plain words
            raise PanelError(502, "hub_error", f"Couldn't read that repo from Hugging Face: {exc}") from exc

    def plan_download(self, link: str, entry: str | None = None) -> dict[str, Any]:
        return self._hub("plan_add", link, entry)

    def start_download(self, link: str, entry: str | None = None) -> dict[str, Any]:
        return self._hub("start_add", link, entry).as_dict()

    def _add_job(self, job: Any, job_id: str) -> dict[str, Any]:
        if job is None or job.kind != "add":
            raise KeyError(job_id)
        return job.as_dict()

    def download_status(self, job_id: str) -> dict[str, Any]:
        return self._add_job(self.downloads.get(job_id) if self.downloads is not None else None, job_id)

    def cancel_download(self, job_id: str) -> dict[str, Any]:
        return self._add_job(self.downloads.cancel(job_id) if self.downloads is not None else None, job_id)

    def add_model(self, path: str, model_id: Any, name: Any, ram_need: Any, revision: str | None) -> dict[str, Any]:
        with self._lock:
            doc, current_revision = self.store.load()
            if revision != current_revision:
                raise StaleRevision("The model list was changed somewhere else.")
            found = detect(path, taken=self._taken(doc))
            if found["kind"] == "unsupported":
                raise PanelError(422, "not_supported", found["reason"])
            already = self._already(doc, found["path"])
            if already:
                raise PanelError(409, "already_added", f"This is already in the list as {already}.")
            model_id, name = str(model_id or "").strip(), str(name or "").strip()
            errors: list[dict[str, str]] = []
            if not MODEL_ID_RE.fullmatch(model_id):
                errors.append({"field": "add.id", "message": f"The id: {ID_RULE}"})
            elif (owner := self._owner(doc, model_id)) is not None:
                errors.append({"field": "add.id", "message": f"The id {model_id} is already used by {owner}."})
            if not name or len(name) > NAME_MAX or any(ord(c) < 32 or ord(c) == 127 for c in name):
                errors.append({"field": "add.name", "message": f"The name must be 1 to {NAME_MAX} characters on one line."})
            ram: int | float | None = None
            try:
                ram = _number(ram_need)
                if not 0 <= ram <= 512:
                    raise ValueError("out of range")
            except (TypeError, ValueError):
                errors.append({"field": "add.ramNeedGB", "message": "The PC memory it needs must be a number from 0 to 512 GB."})
            if errors:
                raise RegistryValidationError(errors)
            entry = {"id": model_id, "name": name, "engine": found["engine"], "runtime": found["runtime"],
                     "artifact": _home_path(found["path"]), "ramNeedGB": ram, "idleMinutes": None,
                     "aliases": [], "overrides": {}, "presets": {}, "activePreset": None}
            adjusted: list[str] = []
            if entry["engine"] == "freetoken":
                entry["overrides"], adjusted = _fit_to_model(doc, entry)

            def mutate(proposed: dict) -> dict:
                proposed["models"].append(copy.deepcopy(entry))
                return proposed

            result = self._save(mutate, current_revision, None)
            engines = {m["id"]: m["engine"] for m in doc["models"]}
            result.update(status="added", id=model_id, name=name, adjusted=adjusted,
                          pi=self.pi.add(model_id, name, entry["engine"], engines))
            return result

    def _check_deletable(self, doc: Mapping[str, Any], model: Mapping[str, Any], files: list[str]) -> None:
        """Runs before anything changes. Only the model's own files: a NInfer entry or part, or
        a folder holding config.json; never HOME or above; never a path another model's files
        are, contain or sit inside (a v3 entry can read another model's parts)."""
        name = model["name"]
        if not files:
            raise PanelError(409, "files_missing", f"{name}'s files were not found, so there is nothing to delete. "
                                                   "Remove it without deleting files.")
        home = Path(os.path.realpath(Path.home()))
        targets = [Path(os.path.realpath(item)) for item in files]
        for original, real in zip(files, targets):
            if real == home or real in home.parents or real == Path("/"):
                raise PanelError(409, "files_unsafe", f"{original} is your home folder or above it, so nothing was deleted.")
            if model["engine"] == "freetoken":
                if not (Path(original) / "config.json").is_file():
                    raise PanelError(409, "files_unsafe", f"{original} does not look like a model folder, so nothing was deleted.")
            elif not (real.name.endswith(".ninfer") or PART_RE.match(real.name)):
                raise PanelError(409, "files_unsafe", f"{original} is not a NInfer file, so nothing was deleted.")
        used: list[tuple[Path, str]] = []
        for other in doc["models"]:
            if other["id"] == model["id"]:
                continue
            paths = model_files(other["engine"], other["artifact"]) or [expand(other["artifact"])]
            used += [(Path(os.path.realpath(path)), other["name"]) for path in paths]
        for real in targets:
            for other_path, other_name in used:
                if other_path == real or real in other_path.parents or other_path in real.parents:
                    raise PanelError(409, "files_shared", f"{other_name} uses the same files, so nothing was removed. "
                                                          f"Remove it without deleting files, or remove {other_name} first.")

    @staticmethod
    def _delete_files(files: list[str]) -> dict[str, Any]:
        gone, failed = [], []
        for item in files:
            path = Path(item)
            try:
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                gone.append(str(path))
            except OSError as exc:
                failed.append(f"{path} ({exc.strerror or exc})")
        if failed:
            return {"deleted": False, "paths": gone,
                    "message": "The model was removed, but some of its files could not be deleted: " + "; ".join(failed) + "."}
        return {"deleted": True, "paths": gone, "message": "Its files were deleted."}

    def _drop_profile(self, model: Mapping[str, Any]) -> dict[str, Any] | None:
        if model["engine"] != "freetoken":
            return None
        try:
            result = self.profiles.delete(profile_id(model["id"]))
        except ProfileError as exc:
            return {"deleted": False, "message": f"Its FreeToken settings profile could not be deleted: {exc}"}
        if result.get("deleted") and result.get("activeProfileId") and result.get("bootFilePath") and self.profile_deleted:
            self.profile_deleted(result)
        return {"deleted": bool(result.get("deleted"))}

    def remove_model(self, model_id: str, delete_files: bool, revision: str | None) -> dict[str, Any]:
        with self._lock:
            doc, current_revision = self.store.load()
            if revision != current_revision:
                raise StaleRevision("The model list was changed somewhere else.")
            model = find_model(doc, model_id)
            files = model_files(model["engine"], model["artifact"]) if delete_files else []
            if delete_files:
                self._check_deletable(doc, model, files)
            known, _down = self._ask()
            if known is None:
                raise PanelError(503, "switcher_unknown", f"Can't tell whether {model['name']} is loaded right now, "
                                                          "so it was not removed. Try again in a moment.")
            if model_id in known and not self.switcher.unload(model_id):
                raise PanelError(503, "unload_failed", f"Couldn't put {model['name']} away, so nothing was removed. "
                                                       "Try again in a moment.")

            def mutate(proposed: dict) -> dict:
                proposed["models"] = [m for m in proposed["models"] if m["id"] != model_id]
                return proposed

            result = self._save(mutate, current_revision, None)
            result.update(status="removed", id=model_id, name=model["name"], profile=self._drop_profile(model),
                          files=self._delete_files(files) if delete_files else None, pi=self.pi.remove(model_id))
            return result
```

Also import `PART_RE`: `from .model_detect import PART_RE, detect, model_files`.

(h) Route bodies (next to the others):

```python
class DetectBody(BaseModel):
    path: str = ""


class LinkBody(BaseModel):
    link: str = ""
    entry: str | None = None


class AddBody(_SaveBody):
    path: str = ""
    id: str = ""
    name: str = ""
    ramNeedGB: Any = None


class RemoveBody(BaseModel):
    revision: str | None = None
    deleteFiles: bool = False
```

and, inside `create_panel_router` before `return router`:

```python
    @router.get("/add/info")
    async def add_info():
        return await call(service.add_info)

    @router.post("/add/detect")
    async def add_detect(body: DetectBody):
        return await call(service.detect_path, body.path)

    @router.post("/add/plan")
    async def add_plan(body: LinkBody):
        return await call(service.plan_download, body.link, body.entry)

    @router.post("/add/downloads")
    async def add_download(body: LinkBody):
        return await call(service.start_download, body.link, body.entry)

    @router.get("/add/downloads/{job_id}")
    async def add_download_status(job_id: str):
        return await call(service.download_status, job_id)

    @router.post("/add/downloads/{job_id}/cancel")
    async def add_download_cancel(job_id: str):
        return await call(service.cancel_download, job_id)

    @router.post("/models")
    async def add_model(body: AddBody):
        return await call(service.add_model, body.path, body.id, body.name, body.ramNeedGB, body.revision)

    @router.post("/models/{model_id}/remove")
    async def remove_model(model_id: str, body: RemoveBody):
        return await call(service.remove_model, model_id, body.deleteFiles, body.revision)
```

Extend `__all__` with `"default_add_roots"`.

- [ ] **Step 4: Edit `app.py`**
  - `from .pi_sync import PiSync`.
  - `HELPER_VERSION = "2.1.0"`.
  - In `create_app`, the `PanelService(...)` call gains `downloads=download_manager, pi=PiSync(),`.
    Right after `app.state.panel = panel`, add:

```python
    if panel.downloads is None:
        panel.downloads = download_manager
```
  - After `def set_active_boot(...)` is defined, add:

```python
    if panel.profile_deleted is None:
        # Removing a FreeToken model deletes its model-<id> profile; when that profile was the
        # active one, profiles.delete() falls back to the default boot file, and the helper must
        # follow it as the DELETE /api/profiles route does.
        panel.profile_deleted = lambda result: set_active_boot(result["bootFilePath"])
```
  - `tests/settings/test_routes.py`: change `"2.0.0"` to `"2.1.0"`.

- [ ] **Step 5: Run the tests**

```bash
PYTEST tests/settings/test_panel_add_remove.py tests/settings/test_panel_routes.py tests/settings/test_routes.py \
       tests/settings/test_settings_import_safety.py
```
Expected: all pass. The Stage A route tests still pass, which includes
`test_a_save_moving_a_possibly_loaded_model_is_refused_while_unknown`: the `moved` change only
exempts models new to the list.

- [ ] **Step 6: Commit (controller)**

```bash
git add python/freetoken/daemon/settings/panel.py python/freetoken/daemon/settings/app.py \
        tests/settings/test_panel_add_remove.py tests/settings/test_routes.py
git commit -m "feat(settings): add and remove models on the control panel, with Pi kept in step

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The page: Add a model wizard and Remove (`panel.js`, `index.html`)

**Files:**
- Modify: `python/freetoken/daemon/settings/static/panel.js`
- Modify: `python/freetoken/daemon/settings/static/index.html`
- Test: `tests/settings/test_panel_page.py`

**Interfaces:**
- Consumes: Task 5 routes; Stage A's `json`, `$`, `setNotice`, `leaveGuard`, `clearView`,
  `showMain`, `loadModels`, `loadNow`, `reopenView`, `registryProblem`,
  `backToRegistryProblem`, `panelErrorText`, `isLoadedState`, and `openBrowser`/`closeBrowser`
  (index.html).
- Produces:
  - pure helpers exported for node: `idProblem`, `detectionText`, `planSummary`,
    `downloadLine`, `removeQuestion`, `piNote`, `addedNote`, `removedNote`;
  - browser functions exported for node: `addCheckPath`, `addValidate`, `addSave`,
    `openRemove`, `answerRemove`;
  - page ids: `add-model`, `add-wizard` (`add-close`, `add-cancel`, `data-add-source`,
    `add-pc`, `add-path`, `add-browse`, `add-check`, `add-link`, `add-repo`, `add-plan`,
    `add-entry`, `add-plan-card`, `add-download-actions`, `add-download`, `add-progress`,
    `add-progress-stage`, `add-progress-bar`, `add-progress-detail`, `add-download-cancel`,
    `add-step-found`, `add-found-text`, `add-step-identity`, `add-id`, `add-id-error`,
    `add-name`, `add-ram`, `add-errors`, `add-save`), `model-remove`, and `remove-ask`
    (`remove-ask-text`, `remove-files`, `remove-files-note`, `remove-ask-ok`,
    `remove-ask-cancel`).
  - `openBrowser(name, kind, {title, start, onPick})`.

- [ ] **Step 1: Write the failing tests** (append to `tests/settings/test_panel_page.py`)

```python
# ---- Stage B: add and remove ----
def test_add_and_remove_speak_plain_words():
    _node(r"""
const rows = [{id: 'quasar-27b', name: 'QUASAR', aliases: []}, {id: 'qwen3.8-flash', name: 'Flash', aliases: ['Qwen3.8-Flash-Next-NVFP4']}];
assert.equal(p.idProblem('small_9b', rows), '');
assert.equal(p.idProblem('Bad Id', rows), 'Use 1 to 63 small letters, numbers, dots, dashes or underscores, starting with a letter or number.');
assert.equal(p.idProblem('qwen3.8-flash-next-nvfp4', rows), 'That id is already used by Flash.');
assert.equal(p.idProblem('QUASAR-27B', rows), 'Use 1 to 63 small letters, numbers, dots, dashes or underscores, starting with a letter or number.');
assert.equal(p.detectionText({kind: 'unsupported', reason: 'This file is not a NInfer model (those end in .ninfer). It is not supported by your engines.'}),
  'This file is not a NInfer model (those end in .ninfer). It is not supported by your engines.');
assert.equal(p.detectionText({kind: 'ninfer', format: 'NInfer v3 file', engineLabel: 'NInfer', runtimeLabel: 'upstream runtime', bytes: 5 * G}),
  'This is a NInfer v3 file for NInfer (upstream runtime), 5.0 GB.');
assert.equal(p.detectionText({kind: 'ninfer', format: 'NInfer v2 file', engineLabel: 'NInfer', runtimeLabel: 'QUASAR runtime', bytes: 18.4 * G, already: 'QUASAR'}),
  'This is a NInfer v2 file for NInfer (QUASAR runtime), 18.4 GB. It is already in the list as QUASAR.');
const plan = {kind: 'ninfer', entry: 'small.ninfer', entries: ['small.ninfer'], totalBytes: 5 * G, target: '/h/ninfer-work/models/small.ninfer',
  files: [{name: 'small.ninfer', check: 'published'}, {name: 'small.ninfer.part-0001', check: 'published'}], diskFits: true, diskFreeBytes: 900 * G, exists: false};
assert.equal(p.planSummary(plan), 'Downloads the NInfer file small.ninfer (5.0 GB) into /h/ninfer-work/models/small.ninfer. All 2 files will be checked against the checksums the repo publishes.');
assert.match(p.planSummary({...plan, files: [{name: 'a', check: null}]}), /publishes no checksums/);
assert.match(p.planSummary({...plan, diskFits: false, diskFreeBytes: 2 * G}), /Not enough drive space: 2\.0 GB free\./);
assert.match(p.planSummary({...plan, exists: true}), /already on this PC/);
assert.equal(p.planSummary({kind: 'ninfer', entry: null, entries: ['a.ninfer', 'b.ninfer'], files: []}), 'This repo has 2 NInfer files. Pick one.');
assert.equal(p.downloadLine({stage: 'downloading', percent: 41.6, receivedBytes: 2 * G, totalBytes: 5 * G}), 'Downloading · 42% · 2.0 GB of 5.0 GB');
assert.equal(p.downloadLine({stage: 'failed', error: 'small.ninfer does not match the checksum the repo publishes, so the download was deleted.'}),
  'Download failed: small.ninfer does not match the checksum the repo publishes, so the download was deleted. Its partial files were deleted.');
assert.equal(p.downloadLine({stage: 'done', percent: 100, receivedBytes: 5 * G, totalBytes: 5 * G, verified: ['a', 'b']}),
  'Downloaded · 100% · 5.0 GB of 5.0 GB · 2 file(s) matched the published checksums');
assert.equal(p.removeQuestion({id: 'q', name: 'QUASAR'}, true), 'Remove QUASAR from the list? It is loaded now and will be put away first. Apps will no longer see it.');
assert.equal(p.removeQuestion({id: 'q'}, false), 'Remove q from the list? Apps will no longer see it.');
assert.equal(p.piNote({status: 'not_updated', message: "Pi's files could not be read."}), "Pi not updated: Pi's files could not be read.");
assert.equal(p.addedNote({id: 't', name: 'Tiny', adjusted: ['Longest chat set to 8,192, the most this model allows.'], pi: {status: 'updated', notes: []}}),
  'Added Tiny. Longest chat set to 8,192, the most this model allows. Pi updated.');
assert.equal(p.removedNote({id: 'q', name: 'QUASAR', files: {deleted: true, message: 'Its files were deleted.'}, pi: {status: 'updated', notes: ['Pi still starts with q by default; pick another default model in Pi.']}}),
  'Removed QUASAR. Its files were deleted. Pi updated. Pi still starts with q by default; pick another default model in Pi.');
""")


_FAKE_PAGE = r"""
const nodes = {};
global.$ = (id) => (nodes[id] ||= {hidden: true, textContent: '', innerHTML: '', value: '', className: '', disabled: false, checked: false, style: {}, addEventListener() {}, querySelectorAll: () => [], focus() {}});
global.document = {hidden: false, querySelectorAll: () => [], querySelector: () => null};
const notes = []; global.setNotice = (message) => notes.push(message);
global.changedNames = () => [];
const posts = [];
const tick = () => new Promise((resolve) => setImmediate(resolve));
"""


def test_the_wizard_checks_a_path_then_adds_it():
    _node(_FAKE_PAGE + r"""
global.state = {view: null};
global.json = async (url, options = {}) => {
  if (options.method === 'POST') posts.push({url, body: JSON.parse(options.body || '{}')});
  if (url === '/api/panel/add/detect') return {response: {ok: true, status: 200}, body: {kind: 'ninfer', path: '/h/ninfer-work/models/small_9b.ninfer',
    engine: 'ninfer', runtime: 'ninfer-upstream', engineLabel: 'NInfer', runtimeLabel: 'upstream runtime', format: 'NInfer v3 file', bytes: 5 * G,
    already: null, suggested: {id: 'small_9b', name: 'small 9b (NInfer)', ramNeedGB: 5}}};
  if (url === '/api/panel/models' && options.method === 'POST') return {response: {ok: true, status: 200},
    body: {status: 'added', id: 'small_9b', name: 'small 9b (NInfer)', revision: 'r2', adjusted: [], pi: {status: 'updated', notes: []}}};
  return {response: {ok: true, status: 200}, body: {models: [], switcher: {up: true, running: []}}};
};
(async () => {
  p.panel.revision = 'r1';
  p.panel.models = [{id: 'quasar-27b', name: 'QUASAR', aliases: []}];
  $('add-wizard').hidden = false;
  await p.addCheckPath('/h/ninfer-work/models/small_9b.ninfer');
  assert.equal($('add-found-text').textContent, 'This is a NInfer v3 file for NInfer (upstream runtime), 5.0 GB.');
  assert.equal($('add-step-identity').hidden, false);
  assert.equal($('add-id').value, 'small_9b');
  assert.equal($('add-save').disabled, false);
  $('add-id').value = 'quasar-27b'; p.addValidate();
  assert.equal($('add-save').disabled, true);
  assert.equal($('add-id-error').textContent, 'That id is already used by QUASAR.');
  $('add-id').value = 'small_9b'; p.addValidate();
  await p.addSave();
  const sent = posts.find((row) => row.url === '/api/panel/models').body;
  assert.deepEqual(sent, {revision: 'r1', path: '/h/ninfer-work/models/small_9b.ninfer', id: 'small_9b', name: 'small 9b (NInfer)', ramNeedGB: '5'});
  assert.equal($('add-wizard').hidden, true);
  assert.equal(p.panel.revision, 'r2');
  assert.ok(notes.includes('Added small 9b (NInfer). Pi updated.'), notes.join(' / '));
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_an_unsupported_or_known_path_offers_no_save():
    _node(_FAKE_PAGE + r"""
global.state = {view: null};
let answer = {kind: 'unsupported', path: '/x/notes.txt', reason: 'This file is not a NInfer model (those end in .ninfer). It is not supported by your engines.', suggested: null};
global.json = async (url) => ({response: {ok: true, status: 200}, body: answer});
(async () => {
  await p.addCheckPath('/x/notes.txt');
  assert.equal($('add-step-identity').hidden, true);
  assert.equal($('add-save').disabled, true);
  assert.equal($('add-found-text').className, 'error');
  answer = {kind: 'ninfer', path: '/q.ninfer', format: 'NInfer v2 file', engineLabel: 'NInfer', runtimeLabel: 'QUASAR runtime', bytes: G, already: 'QUASAR', suggested: {id: 'q-2', name: 'q', ramNeedGB: 1}};
  await p.addCheckPath('/q.ninfer');
  assert.equal($('add-save').disabled, true);
  assert.match($('add-found-text').textContent, /already in the list as QUASAR/);
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_remove_asks_with_delete_files_off_and_cancel_sends_nothing():
    _node(_FAKE_PAGE + r"""
global.state = {view: {kind: 'model', id: 'q', name: 'QUASAR', state: 'ready', artifact: '~/ninfer-work/models/q.ninfer', url: '/api/panel/views/model/q'}, settings: {}, saved: {}};
global.json = async (url, options = {}) => {
  if (options.method === 'POST') posts.push({url, body: JSON.parse(options.body || '{}')});
  return {response: {ok: true, status: 200}, body: {status: 'removed', id: 'q', name: 'QUASAR', revision: 'r2', files: null,
    pi: {status: 'updated', notes: []}, models: [], switcher: {up: true, running: []}}};
};
(async () => {
  p.panel.revision = 'r1';
  $('remove-files').checked = true;              // left over from an earlier question
  const first = p.openRemove(); await tick();
  assert.equal($('remove-ask').hidden, false);
  assert.equal($('remove-files').checked, false);
  assert.equal($('remove-ask-text').textContent, 'Remove QUASAR from the list? It is loaded now and will be put away first. Apps will no longer see it.');
  assert.match($('remove-files-note').textContent, /~\/ninfer-work\/models\/q\.ninfer/);
  p.answerRemove(false); await first;
  assert.deepEqual(posts, []);
  const second = p.openRemove(); await tick();
  p.answerRemove(true); await second;
  assert.deepEqual(posts[0], {url: '/api/panel/models/q/remove', body: {revision: 'r1', deleteFiles: false}});
  assert.ok(notes.includes('Removed QUASAR. Pi updated.'), notes.join(' / '));
})().catch((error) => { console.error(error); process.exitCode = 1; });
""")


def test_stage_b_page_contract():
    from freetoken.daemon.settings.registry import MODEL_ID_RE

    page, js = PAGE.read_text(encoding="utf-8"), PANEL_JS.read_text(encoding="utf-8")
    for present in ('id="add-model"', 'id="add-wizard"', 'data-add-source="pc"', 'data-add-source="link"', 'id="add-path"',
                    'id="add-browse"', 'id="add-check"', 'id="add-repo"', 'id="add-plan"', 'id="add-entry"',
                    'id="add-download"', 'id="add-download-cancel"', 'id="add-progress"', 'id="add-id"', 'id="add-name"',
                    'id="add-ram"', 'id="add-save"', 'id="model-remove"', 'id="remove-ask"', 'id="remove-ask-ok"',
                    'id="remove-ask-cancel"', 'Also delete the model files'):
        assert present in page, present
    assert '<input id="remove-files" type="checkbox">' in page, "the delete checkbox starts unticked"
    for route in ("/api/panel/add/info", "/api/panel/add/detect", "/api/panel/add/plan", "/api/panel/add/downloads",
                  "/remove", "'/api/panel/models'"):
        assert route in js, route
    assert MODEL_ID_RE.pattern == "^[a-z0-9][a-z0-9._-]{0,62}$"
    assert "/^[a-z0-9][a-z0-9._-]{0,62}$/" in js, "the page's id rule must match registry.MODEL_ID_RE"
    assert "options.onPick" in page and "if (onPick)" in page
    for banned in ("window.confirm", "window.alert", "window.prompt"):
        assert banned not in js
```

- [ ] **Step 2: Run to see them fail**

`PYTEST tests/settings/test_panel_page.py -k "add or remove or stage_b"`
Expected: failures (`p.idProblem is not a function`, missing ids).

- [ ] **Step 3: Edit `panel.js`**

(a) Header comment: append the line "Stage B (spec section 7): the Add a model wizard and the
Remove question." In the `panel` object literal, add `removeResolve: null`.

(b) Before the `if (typeof module !== 'undefined') module.exports = …` line, add the pure helpers:

```js
// ---- Stage B: add and remove (spec section 7) ----
const ADD_ID_RE = /^[a-z0-9][a-z0-9._-]{0,62}$/; // registry.MODEL_ID_RE (a test keeps them equal)
const ADD_ID_RULE = 'Use 1 to 63 small letters, numbers, dots, dashes or underscores, starting with a letter or number.';
const ADD_STAGE_WORDS = { queued: 'Waiting to start', downloading: 'Downloading', verifying: 'Checking the download', moving: 'Putting it in place', done: 'Downloaded', failed: 'Download failed', cancelled: 'Download cancelled' };
const ADD_TERMINAL = new Set(['done', 'failed', 'cancelled']);
const ADD_POLL_MS = 1500;
function idProblem(id, rows) {
  const text = String(id ?? '').trim();
  if (!ADD_ID_RE.test(text)) return ADD_ID_RULE;
  const wanted = text.toLowerCase();
  const owner = (rows || []).find((row) => String(row.id).toLowerCase() === wanted || (row.aliases || []).some((alias) => String(alias).toLowerCase() === wanted));
  return owner ? `That id is already used by ${owner.name || owner.id}.` : '';
}
function detectionText(found) {
  if (!found) return '';
  if (found.kind === 'unsupported') return found.reason || 'Not supported by your engines.';
  const engine = found.runtimeLabel ? `${found.engineLabel} (${found.runtimeLabel})` : found.engineLabel;
  const already = found.already ? ` It is already in the list as ${found.already}.` : '';
  return `This is a ${found.format || 'model'} for ${engine}, ${fmtGB(found.bytes)}.${already}`;
}
function planSummary(plan) {
  if (!plan) return '';
  if (plan.kind === 'ninfer' && !plan.entry) return `This repo has ${(plan.entries || []).length} NInfer files. Pick one.`;
  const what = plan.kind === 'ninfer' ? `NInfer file ${plan.entry}` : `model folder ${plan.name}`;
  const files = plan.files || [];
  const checked = files.filter((file) => file.check).length;
  const sums = !checked ? ' The repo publishes no checksums, so the files cannot be checked.' : checked === files.length ? ` All ${files.length} files will be checked against the checksums the repo publishes.` : ` ${checked} of ${files.length} files will be checked against the checksums the repo publishes.`;
  const space = plan.diskFits === false ? ` Not enough drive space: ${fmtGB(plan.diskFreeBytes)} free.` : '';
  const exists = plan.exists ? ' It is already on this PC, so use “On this PC” to add it.' : '';
  return `Downloads the ${what} (${fmtGB(plan.totalBytes)}) into ${plan.target}.${sums}${space}${exists}`;
}
function downloadLine(job) {
  if (!job) return '';
  const words = ADD_STAGE_WORDS[job.stage] || String(job.stage || '');
  if (job.stage === 'failed') return `${words}: ${job.error || 'no reason was given.'} Its partial files were deleted.`;
  if (job.stage === 'cancelled') return `${words}. Its partial files were deleted.`;
  const sizes = job.totalBytes ? ` · ${fmtGB(job.receivedBytes)} of ${fmtGB(job.totalBytes)}` : '';
  const checked = job.stage === 'done' && (job.verified || []).length ? ` · ${job.verified.length} file(s) matched the published checksums` : '';
  return `${words} · ${Math.round(Number(job.percent) || 0)}%${sizes}${checked}`;
}
function removeQuestion(row, loaded) {
  return `Remove ${row.name || row.id} from the list?${loaded ? ' It is loaded now and will be put away first.' : ''} Apps will no longer see it.`;
}
function piNote(pi) {
  if (!pi) return '';
  const notes = (pi.notes || []).join(' ');
  if (pi.status === 'not_updated') return `Pi not updated: ${pi.message}`;
  if (pi.status === 'updated') return `Pi updated.${notes ? ` ${notes}` : ''}`;
  return notes;
}
function addedNote(body) {
  return [`Added ${body.name || body.id}.`, ...(body.adjusted || []), piNote(body.pi)].filter(Boolean).join(' ');
}
function removedNote(body) {
  return [`Removed ${body.name || body.id}.`, body.files && body.files.message, body.profile && body.profile.message, piNote(body.pi)].filter(Boolean).join(' ');
}
```

`downloadLine`'s failed text: server errors end with a full stop, so
`${words}: ${job.error} Its partial…` reads as two sentences. That is the form the test pins.

(c) Extend the `module.exports` object with
`idProblem, detectionText, planSummary, downloadLine, removeQuestion, piNote, addedNote, removedNote, addCheckPath, addValidate, addSave, openRemove, answerRemove`.

(d) In the browser section, add after `answerConfirm`:

```js
/* ---------- Stage B: add a model, remove a model ---------- */
const addState = { found: null, plan: null, job: null, timer: null, roots: {} };
function addSource(which) {
  $('add-pc').hidden = which !== 'pc';
  $('add-link').hidden = which !== 'link';
  document.querySelectorAll('[data-add-source]').forEach((button) => button.setAttribute('aria-selected', String(button.dataset.addSource === which)));
}
function addReset() {
  addState.found = null;
  $('add-step-found').hidden = true;
  $('add-step-identity').hidden = true;
  $('add-errors').textContent = '';
  $('add-save').disabled = true;
}
async function openAdd() {
  if (!leaveGuard()) return;
  addReset();
  addState.plan = null;
  $('add-path').value = ''; $('add-repo').value = '';
  ['add-plan-card', 'add-entry', 'add-download-actions', 'add-progress'].forEach((id) => { $(id).hidden = true; });
  addSource('pc');
  $('add-wizard').hidden = false;
  const { response, body } = await json('/api/panel/add/info');
  if (!response.ok) return;
  addState.roots = body.roots || {};
  // A download started earlier keeps running in the helper; reopening the wizard picks it up.
  const job = body.download;
  if (job && !ADD_TERMINAL.has(job.stage)) { addSource('link'); addState.job = job; renderAddJob(job); pollAddJob(); }
}
function closeAdd() { $('add-wizard').hidden = true; clearTimeout(addState.timer); addState.timer = null; }
function addBrowse() {
  openBrowser('add', 'add', { title: 'Choose a model file or folder', start: addState.roots.ninfer || '', onPick: (path) => { $('add-path').value = path; addCheckPath(path); } });
}
async function addCheckPath(path) {
  addReset();
  const text = String(path ?? '').trim();
  if (!text) return;
  $('add-step-found').hidden = false;
  $('add-found-text').className = '';
  $('add-found-text').textContent = 'Checking…';
  const { response, body } = await postJson('/api/panel/add/detect', { path: text });
  if (registryProblem(response, body)) { closeAdd(); await backToRegistryProblem(body); return; }
  if (!response.ok) { $('add-found-text').textContent = panelErrorText(body, 'Could not check that path.'); $('add-found-text').className = 'error'; return; }
  showFound(body);
}
function showFound(found) {
  addState.found = found;
  const ok = found.kind !== 'unsupported' && !found.already;
  $('add-step-found').hidden = false;
  $('add-found-text').textContent = detectionText(found);
  $('add-found-text').className = ok ? '' : 'error';
  $('add-step-identity').hidden = !ok;
  if (ok) { $('add-id').value = found.suggested.id; $('add-name').value = found.suggested.name; $('add-ram').value = String(found.suggested.ramNeedGB); }
  addValidate();
}
function addValidate() {
  const found = addState.found;
  const usable = !!found && found.kind !== 'unsupported' && !found.already;
  const problem = usable ? idProblem($('add-id').value, panel.models) : '';
  $('add-id-error').textContent = problem;
  $('add-save').disabled = !usable || !!problem || !String($('add-name').value ?? '').trim();
}
async function addPlan() {
  addReset();
  $('add-download-actions').hidden = true;
  const entry = $('add-entry').hidden ? null : ($('add-entry').value || null);
  $('add-plan-card').hidden = false;
  $('add-plan-card').textContent = 'Reading the repo…';
  const { response, body } = await postJson('/api/panel/add/plan', { link: String($('add-repo').value ?? '').trim(), entry });
  if (!response.ok) { addState.plan = null; $('add-plan-card').textContent = panelErrorText(body, 'Could not read that link.'); return; }
  addState.plan = body;
  $('add-plan-card').textContent = planSummary(body);
  const pick = body.kind === 'ninfer' && (body.entries || []).length > 1;
  $('add-entry').hidden = !pick;
  if (pick && !entry) $('add-entry').innerHTML = `<option value="">Pick a NInfer file…</option>${body.entries.map((name) => `<option value="${panelEsc(name)}">${panelEsc(name)}</option>`).join('')}`;
  const ready = body.kind === 'folder' || !!body.entry;
  $('add-download-actions').hidden = !ready;
  $('add-download').disabled = !ready || !!body.exists || !body.diskFits;
}
async function addDownload() {
  const plan = addState.plan;
  if (!plan || $('add-download').disabled) return;
  $('add-download').disabled = true;
  const { response, body } = await postJson('/api/panel/add/downloads', { link: plan.repo, entry: plan.entry || null });
  if (!response.ok) { $('add-download').disabled = false; $('add-plan-card').textContent = panelErrorText(body, 'Could not start the download.'); return; }
  addState.job = body;
  renderAddJob(body);
  pollAddJob();
}
function renderAddJob(job) {
  $('add-progress').hidden = false;
  $('add-progress-stage').textContent = ADD_STAGE_WORDS[job.stage] || job.stage;
  $('add-progress-bar').style.width = `${Math.max(0, Math.min(100, Number(job.percent) || 0))}%`;
  $('add-progress-detail').textContent = downloadLine(job);
  $('add-download-cancel').disabled = ADD_TERMINAL.has(job.stage);
}
async function pollAddJob() {
  clearTimeout(addState.timer); addState.timer = null;
  const job = addState.job;
  if (!job) return;
  const { response, body } = await json(`/api/panel/add/downloads/${encodeURIComponent(job.id)}`);
  if (response.status === 404) { $('add-progress-detail').textContent = 'The settings page restarted and lost this download. Start it again.'; $('add-download').disabled = false; return; }
  const current = response.ok ? body : job;
  addState.job = current;
  renderAddJob(current);
  if (current.stage === 'done') { $('add-path').value = current.resultPath || ''; await addCheckPath(current.resultPath); return; }
  if (current.stage === 'failed' || current.stage === 'cancelled') { $('add-download').disabled = false; return; }
  if (!$('add-wizard').hidden) addState.timer = setTimeout(pollAddJob, ADD_POLL_MS);
}
async function addCancelDownload() {
  const job = addState.job;
  if (!job) return;
  $('add-download-cancel').disabled = true;
  const { response, body } = await postJson(`/api/panel/add/downloads/${encodeURIComponent(job.id)}/cancel`, {});
  if (response.ok) { addState.job = body; renderAddJob(body); }
}
async function addSave() {
  const found = addState.found;
  if (!found || $('add-save').disabled) return;
  $('add-save').disabled = true;
  $('add-errors').textContent = '';
  const { response, body } = await postJson('/api/panel/models', { revision: panel.revision, path: found.path, id: String($('add-id').value ?? '').trim(), name: String($('add-name').value ?? '').trim(), ramNeedGB: $('add-ram').value });
  if (response.status === 409 && body.code === 'stale_revision') { await loadModels(); $('add-errors').textContent = 'The model list changed meanwhile. Check the details and press Add model again.'; addValidate(); return; }
  if (registryProblem(response, body)) { closeAdd(); await backToRegistryProblem(body); return; }
  if (!response.ok) { $('add-errors').textContent = panelErrorText(body, 'Could not add the model.'); addValidate(); return; }
  panel.revision = body.revision || panel.revision;
  closeAdd();
  setNotice(addedNote(body), body.pi && body.pi.status === 'not_updated' ? 'warn' : 'good');
  await loadModels();
  loadNow();
}

async function openRemove() {
  const view = state.view;
  if (!view || view.kind !== 'model' || !leaveGuard()) return;
  const row = (panel.models || []).find((item) => item.id === view.id) || { id: view.id, name: view.name };
  $('remove-ask-text').textContent = removeQuestion(row, isLoadedState(view.state));
  $('remove-files').checked = false; // spec: "also delete the model files" is off by default, every time
  $('remove-files-note').textContent = `Ticked, this also deletes ${view.artifact || 'its files'} from the drive. That cannot be undone.`;
  $('remove-ask').hidden = false;
  const yes = await new Promise((resolve) => { panel.removeResolve = resolve; });
  if (!yes) return;
  const { response, body } = await postJson(`/api/panel/models/${encodeURIComponent(view.id)}/remove`, { revision: panel.revision, deleteFiles: !!$('remove-files').checked });
  if (response.status === 409 && body.code === 'stale_revision') { setNotice(body.message, 'bad'); await reopenView(); return; }
  if (registryProblem(response, body)) { await backToRegistryProblem(body); return; }
  if (!response.ok) { setNotice(panelErrorText(body, 'Could not remove the model.'), 'bad'); return; }
  panel.revision = body.revision || panel.revision;
  clearView();
  showMain('models');
  setNotice(removedNote(body), (body.files && !body.files.deleted) || (body.pi && body.pi.status === 'not_updated') ? 'warn' : 'good');
  loadNow();
}
function answerRemove(yes) {
  $('remove-ask').hidden = true;
  const resolve = panel.removeResolve;
  panel.removeResolve = null;
  if (resolve) resolve(!!yes);
}
```

(e) `applyView`: after `$('panel-fit').hidden = body.kind !== 'model';` add
`$('model-remove').hidden = body.kind !== 'model';`.

(f) `wirePanel`: append

```js
  $('add-model').addEventListener('click', openAdd);
  $('add-close').addEventListener('click', closeAdd);
  $('add-cancel').addEventListener('click', closeAdd);
  document.querySelectorAll('[data-add-source]').forEach((button) => button.addEventListener('click', () => { addReset(); addSource(button.dataset.addSource); }));
  $('add-browse').addEventListener('click', addBrowse);
  $('add-check').addEventListener('click', () => addCheckPath($('add-path').value));
  $('add-path').addEventListener('keydown', (event) => { if (event.key === 'Enter') addCheckPath($('add-path').value); });
  $('add-plan').addEventListener('click', () => { $('add-entry').hidden = true; addPlan(); });
  $('add-repo').addEventListener('keydown', (event) => { if (event.key === 'Enter') { $('add-entry').hidden = true; addPlan(); } });
  $('add-entry').addEventListener('change', addPlan);
  $('add-download').addEventListener('click', addDownload);
  $('add-download-cancel').addEventListener('click', addCancelDownload);
  ['add-id', 'add-name', 'add-ram'].forEach((id) => $(id).addEventListener('input', addValidate));
  $('add-save').addEventListener('click', addSave);
  $('model-remove').addEventListener('click', openRemove);
  $('remove-ask-ok').addEventListener('click', () => answerRemove(true));
  $('remove-ask-cancel').addEventListener('click', () => answerRemove(false));
```

- [ ] **Step 4: Edit `index.html`**

(a) CSS: after the `.modal-foot { … }` rule add

```css
    #browser { z-index: 50; } /* opened from the Add model wizard, it must sit on top of it */
    .add-field { display: grid; gap: 4px; margin-top: 10px; }
    .add-field input { width: 100%; }
    .add-source { margin-bottom: 10px; }
    #add-wizard section { margin-top: 14px; }
```

(b) Models head: replace

```html
      <div class="panel-head" id="models-head"><div><h2>Models</h2><p class="small">Load or unload a model, or open its settings. Apps keep using the same address whichever model is loaded.</p></div></div>
```
with

```html
      <div class="panel-head" id="models-head"><div><h2>Models</h2><p class="small">Load or unload a model, or open its settings. Apps keep using the same address whichever model is loaded.</p></div><button class="button primary" id="add-model" type="button">Add a model</button></div>
```

(c) View head: replace

```html
        <div><button class="button small" id="view-back" type="button" hidden>← All models</button><h2 id="view-title"></h2><p class="small" id="view-note"></p></div>
```
with

```html
        <div><button class="button small" id="view-back" type="button" hidden>← All models</button><h2 id="view-title"></h2><p class="small" id="view-note"></p><button class="button small danger" id="model-remove" type="button" hidden>Remove this model…</button></div>
```

(d) Immediately before `<div class="modal" id="browser" …>` insert:

```html
  <div class="modal" id="add-wizard" hidden role="dialog" aria-modal="true" aria-labelledby="add-title">
    <div class="modal-card">
      <div class="modal-head"><h2 id="add-title">Add a model</h2><button class="button small ghost" type="button" id="add-close" aria-label="Close">✕</button></div>
      <div class="modal-body">
        <p class="small">Step 1 of 3: where is the model?</p>
        <nav class="tabs add-source" role="tablist" aria-label="Where the model is">
          <button class="tab-btn" role="tab" type="button" data-add-source="pc" aria-selected="true">On this PC</button>
          <button class="tab-btn" role="tab" type="button" data-add-source="link" aria-selected="false">Download from Hugging Face</button>
        </nav>
        <div id="add-pc">
          <p class="small">Pick a NInfer file (it ends in .ninfer) or a model folder (config.json plus .safetensors files).</p>
          <div class="path-row"><input id="add-path" type="text" spellcheck="false" placeholder="~/ninfer-work/models/… or ~/models/…" aria-label="Model file or folder"><button class="button" type="button" id="add-browse">Browse…</button><button class="button primary" type="button" id="add-check">Check</button></div>
        </div>
        <div id="add-link" hidden>
          <p class="small">Paste a Hugging Face link. NInfer files go to the NInfer models folder and model folders to the models folder. Nothing already on the PC is overwritten, and the files are checked against the checksums the repo publishes.</p>
          <div class="inline-form"><input id="add-repo" type="text" spellcheck="false" placeholder="https://huggingface.co/owner/name" aria-label="Hugging Face link"><button class="button primary" type="button" id="add-plan">Check link</button></div>
          <select id="add-entry" hidden aria-label="Which NInfer file"></select>
          <p class="small" id="add-plan-card" hidden aria-live="polite"></p>
          <div class="actions" id="add-download-actions" hidden><button class="button primary" type="button" id="add-download">Download</button></div>
          <div class="job" id="add-progress" hidden aria-live="polite"><div style="flex:1;min-width:220px"><strong id="add-progress-stage">Waiting</strong><div class="bar"><span id="add-progress-bar" style="width:0%"></span></div><span class="small" id="add-progress-detail"></span></div><button class="button small" type="button" id="add-download-cancel">Cancel download</button></div>
        </div>
        <section id="add-step-found" hidden><p class="small">Step 2 of 3: what it is</p><p id="add-found-text" aria-live="polite"></p></section>
        <section id="add-step-identity" hidden>
          <p class="small">Step 3 of 3: how it shows up in apps</p>
          <div class="add-field"><label for="add-id">Id (the name apps ask for)</label><input id="add-id" type="text" maxlength="63" spellcheck="false"><p class="small error" id="add-id-error"></p></div>
          <div class="add-field"><label for="add-name">Name in apps</label><input id="add-name" type="text" maxlength="120"></div>
          <div class="add-field"><label for="add-ram">PC memory it needs (GB)</label><input id="add-ram" type="number" min="0" max="512" step="1"><p class="small">Suggested from the model's size. The switcher waits until this much PC memory is free before loading it.</p></div>
        </section>
        <p class="error" id="add-errors" aria-live="polite"></p>
      </div>
      <div class="modal-foot"><button class="button" type="button" id="add-cancel">Close</button><div class="actions"><button class="button primary" type="button" id="add-save" disabled>Add model</button></div></div>
    </div>
  </div>

  <div class="modal" id="remove-ask" hidden role="dialog" aria-modal="true" aria-labelledby="remove-ask-title">
    <div class="modal-card" style="width:min(560px,100%)">
      <div class="modal-head"><h2 id="remove-ask-title">Remove this model?</h2></div>
      <div class="modal-body"><p id="remove-ask-text"></p><label class="check"><input id="remove-files" type="checkbox"> Also delete the model files</label><p class="small" id="remove-files-note"></p></div>
      <div class="modal-foot"><button class="button" type="button" id="remove-ask-cancel">Cancel</button><div class="actions"><button class="button danger" type="button" id="remove-ask-ok">Remove</button></div></div>
    </div>
  </div>
```

(e) Folder browser. Replace `openBrowser` with

```js
    // options.onPick (the Add model wizard): hand the chosen path back instead of filling a dial.
    async function openBrowser(name, kind, options = {}) {
      const dial = options.onPick ? null : dialByName(name);
      state.browser = { name, kind, path: '', isModel: false, onPick: options.onPick || null };
      $('browser-title').textContent = options.title || (kind === 'file' ? `Choose a file for “${dial.plain}”` : `Choose a folder for “${dial.plain}”`);
      $('browser-choose').textContent = kind === 'file' ? 'Use this file' : 'Use this folder';
      $('browser').hidden = false;
      await browseTo(options.onPick ? String(options.start || '') : String(valueFor(dial) || ''));
    }
```
  In `pickPath`, replace its first line `const { name } = state.browser;` with

```js
      const { name, onPick } = state.browser;
      if (onPick) { closeBrowser(); onPick(path); return; }
```
  In `browseTo`, before `else if (kind === 'file') { hint.textContent = 'Click a file to choose it.'; … }`, insert
  `else if (kind === 'add') { hint.textContent = 'Click a .ninfer file, or open a folder marked Model and press Choose.'; hint.className = 'hint'; }`.

(f) Replace the Escape handler line
`document.addEventListener('keydown', (event) => { if (event.key === 'Escape') { closeBrowser(); if (!$('restart-ask').hidden) answerRestart(null); if (!$('confirm-ask').hidden) answerConfirm(false); } });`
with

```js
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      if (!$('browser').hidden) { closeBrowser(); return; } // one dialog per Escape: the browser sits on the wizard
      if (!$('restart-ask').hidden) answerRestart(null);
      if (!$('confirm-ask').hidden) answerConfirm(false);
      if (!$('remove-ask').hidden) answerRemove(false);
      if (!$('add-wizard').hidden) closeAdd();
    });
```

(g) In `syncTabExtras`, change the comment on `modelTools.hidden = true` to
`// replaced by the Add a model wizard on the Models tab (Stage B)`. The old hidden block and
its `/api/downloads` routes stay; `test_static_page.py` still pins them.

- [ ] **Step 5: Run the page tests and a browser check on the devbox**

```bash
PYTEST tests/settings/test_panel_page.py tests/settings/test_static_page.py
```
Expected: all pass.

Then start a scratch helper on the devbox against temp files, the Stage A "2032 scratch
helper" pattern, and click through once:

```bash
SCR=$(mktemp -d); mkdir -p $SCR/ninfer $SCR/models $SCR/pi
PYTHONPATH=python /home/jay/projects/FreeToken/.venv/bin/python - <<EOF
import json, sys; sys.path.insert(0, "tests/settings")
from tests.settings.test_model_detect import write_v3; from tests.settings.test_pi_sync import write_pi
from pathlib import Path; write_v3(Path("$SCR/ninfer/small_9b.ninfer"), parts=1); write_pi(Path("$SCR/pi"))
EOF
FREETOKEN_REGISTRY=$SCR/registry.json FREETOKEN_SWAP_CONFIG=$SCR/config.yaml FREETOKEN_PI_AGENT_DIR=$SCR/pi \
FREETOKEN_NINFER_MODELS_DIR=$SCR/ninfer FREETOKEN_MODELS_DIR=$SCR/models FREETOKEN_SWITCHER_URL=http://127.0.0.1:9 \
PYTHONPATH=python /home/jay/projects/FreeToken/.venv/bin/python -m freetoken.daemon.settings.server --port 2032 \
  --boot-file $SCR/boot.ps1 --profiles-file $SCR/profiles.json --log-file $SCR/s.log --gpu-lock $SCR/gpu.lock &
```
  Before starting, seed `$SCR/registry.json` with `five()` and `$SCR/config.yaml` with its
  render. The switcher binary check needs `FREETOKEN_LLAMA_SWAP_BIN` pointing at a built
  llama-swap (`scripts/engines/build.sh llama-swap` on the devbox), or at a shell script that
  prints `config is valid` and exits 0.
  Use `chrome-devtools-axi` on `http://127.0.0.1:2032`:
  1. Add a model → Browse → `small_9b.ninfer` → Add. Check it appears in the list and in
     `$SCR/pi/models.json`.
  2. Settings → Remove this model… → Remove.
  3. Escape closes one dialog at a time.

  Kill the helper and `rm -rf $SCR` afterwards. Record anything odd for the controller; this
  check writes nothing tracked.

- [ ] **Step 6: Commit (controller)**

```bash
git add python/freetoken/daemon/settings/static/panel.js python/freetoken/daemon/settings/static/index.html \
        tests/settings/test_panel_page.py
git commit -m "feat(settings): Add a model wizard and Remove question on the control panel page

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Docs and the full suite

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README.** In the fork section's settings-page paragraph, replace the sentence
  "The Models tab can also preview a Hugging Face model, check memory and drive space, and
  download it without overwriting an existing folder." with:

  "**Add a model** on the Models tab takes a NInfer file or a model folder on the PC, or a
  Hugging Face link. A link is downloaded into `~/ninfer-work/models/` or `~/models/<name>`,
  checked against the repo's published checksums, and never overwrites anything. The page
  reads the file's header to pick the engine and the NInfer runtime (v2 files run on the QUASAR
  runtime, v3 on upstream). It suggests an id, a name and the PC memory it needs. **Remove this
  model…** in a model's settings puts it away first if it is loaded. It deletes the files only
  when you tick the box. Both keep Pi's `freetoken-local` model list (`C:\Users\<you>\.pi\agent`)
  in step, with a backup of Pi's files before every change."

- [ ] **Step 2: Full suite on the devbox**

```bash
PYTEST tests/settings tests/daemon tests/engines 2>&1 | tail -3
```
Expected: all pass. The only skips are the NInfer `--help` tests and anything needing a GPU.

- [ ] **Step 3: Commit and push (controller)**

```bash
git add README.md
git commit -m "docs: adding and removing models on the control panel

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push -u origin feat/control-panel-b
```

---

### Task 8: Live acceptance on the box, with screenshots (controller only; ask Jay first)

**Files:**
- Create: `docs/research/control-panel-b-acceptance-<date>.md`
- Create: `docs/research/control-panel-b-acceptance-<date>/NN-<what>.png`

Box access follows the route in memory: `ssh 5090`, then a stdin script into
`wsl -d vllm -e bash -l`. Paths below are inside WSL. Screenshots are taken from the devbox
against the tailnet page `https://5090.tail45ff04.ts.net`:

```bash
AXI=~/.npm-global/bin/chrome-devtools-axi
OUT=docs/research/control-panel-b-acceptance-$(date +%F); mkdir -p $OUT
$AXI open https://5090.tail45ff04.ts.net && $AXI snapshot     # read the @uids, then click/fill by uid
$AXI screenshot $OUT/01-models-add-button.png
```

- [ ] **Step 1: Ask Jay for the window.** Tell him:
  - it takes about 40 minutes;
  - there is no FreeToken boot unless he agrees to Step 7's optional load;
  - the helper restarts once;
  - Pi's two files are edited twice, with backups.

  Then:
  - ask which models he is using, and run `curl -s 127.0.0.1:2040/running`;
  - agree the test repo (Step 5);
  - ask him to close Pi on Windows during Steps 5-8, so Pi does not rewrite `settings.json`
    over the change.

  Do not start without a yes.

- [ ] **Step 2: Deploy and back up**

```bash
cd ~/FreeToken && git fetch -q && git checkout -q feat/control-panel-b && git pull -q
PI=/mnt/c/Users/jay/.pi/agent; ls -l $PI/models.json $PI/settings.json
STAMP=$(date +%Y%m%d-%H%M%S)
cp $PI/models.json $PI/models.json.bak-before-stage-b-$STAMP; cp $PI/settings.json $PI/settings.json.bak-before-stage-b-$STAMP
cp ~/.config/freetoken/registry.json ~/registry.before-stage-b-$STAMP.json
systemctl --user restart freetoken-settings && sleep 5 && curl -s 127.0.0.1:2031/api/status | python3 -c "import json,sys; print(json.load(sys.stdin)['helper']['version'])"
curl -s 127.0.0.1:2031/api/panel/add/info
```
Expected:
- both Pi files are listed;
- the version is `2.1.0`;
- `add/info` shows the roots `/home/jay/models` and `/home/jay/ninfer-work/models` and
  `"download": null`.

If the helper's user cannot read `$PI`, set `FREETOKEN_PI_AGENT_DIR` in the unit
(`systemctl --user edit freetoken-settings`) and restart.

- [ ] **Step 3: Detection against every artifact on the box** (Review focus 5)

```bash
cd ~/FreeToken && PYTHONPATH=python .venv/bin/python -m freetoken.daemon.settings.model_detect \
  ~/ninfer-work/models/*.ninfer ~/models/*/
PYTHONPATH=python .venv/bin/python - <<'EOF'
from freetoken.daemon.settings.registry import RegistryStore, expand
from freetoken.daemon.settings.model_detect import detect
doc, _ = RegistryStore().load()
for m in doc["models"]:
    found = detect(expand(m["artifact"]))
    print(m["id"], m["runtime"], found["runtime"], "OK" if found["runtime"] == m["runtime"] else "MISMATCH", found["suggested"])
EOF
```
Expected:
- every registry model prints `OK`: QUASAR `ninfer` (v2); Fable and Twin `ninfer-upstream` (v3);
- both FreeToken folders print `freetoken`;
- the suggested `ramNeedGB` is recorded against the registry's value (18 NInfer, 62 FreeToken)
  and is within 10%.

If a mismatch appears, stop: it is a detection bug.

- [ ] **Step 4: "On this PC" with a known model**
  - Models → **Add a model**. Screenshot `01-add-wizard.png`.
  - Browse to `~/ninfer-work/models`; screenshot `02-browse-marks-models.png`, which shows
    `.ninfer` files marked Model and no `.part-` files.
  - Pick `quasar_27b_nvfp4.ninfer`. Expected: "This is a NInfer v2 file for NInfer (QUASAR
    runtime), 18.4 GB. It is already in the list as …", and Add model is off. Screenshot
    `03-already-in-the-list.png`.

- [ ] **Step 5: Add a small NInfer model from a link** (the spec's acceptance item)
  - Choose the repo with Jay. List candidates with
    `curl -s "https://huggingface.co/api/models?search=ninfer&full=true&limit=50" | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)]"`
    and then `curl -s "https://huggingface.co/api/models/<id>?blobs=true"` for the file sizes.
    Pick the smallest repo with a `.ninfer` file of 12 GB or less. Check that its magic
    matches a runtime the box has.
  - **If no public small NInfer repo exists:** use a small FreeToken-supported folder repo
    (`Qwen/Qwen3-0.6B`, Qwen3ForCausalLM) for the link path, and record the deviation in the
    acceptance doc.
  - Paste the link and press **Check link**. Screenshot `04-link-plan.png`, which shows the
    size, the target and "All N files will be checked…".
  - Press **Download**. Screenshot `05-download-progress.png` mid-way.
  - Expected when it is done:
    - "Downloaded · 100% · … · N file(s) matched the published checksums";
    - step 2 shows the engine and runtime;
    - step 3 is filled in. Screenshot `06-identity.png`.
  - Press **Add model**. Screenshot `07-models-list-added.png`.
  - Then check on the box:

```bash
curl -s 127.0.0.1:2040/v1/models | python3 -c "import json,sys; print(sorted(m['id'] for m in json.load(sys.stdin)['data']))"
curl -s 127.0.0.1:2040/api/config/hash; sha256sum ~/llama-swap/config.yaml
ls -a ~/ninfer-work/models | grep -c '^\.incoming' ; ls -l $PI/*.bak-2* | tail -4
PYTHONPATH=python .venv/bin/python - <<EOF
import json
old = json.load(open("$PI/models.json.bak-before-stage-b-$STAMP")); new = json.load(open("$PI/models.json"))
print("others same:", {k: v for k, v in old["providers"].items() if k != "freetoken-local"} == {k: v for k, v in new["providers"].items() if k != "freetoken-local"})
print("new entry:", new["providers"]["freetoken-local"]["models"][-1])
s_old = json.load(open("$PI/settings.json.bak-before-stage-b-$STAMP")); s_new = json.load(open("$PI/settings.json"))
print("settings same but enabledModels:", {k: v for k, v in s_old.items() if k != "enabledModels"} == {k: v for k, v in s_new.items() if k != "enabledModels"})
print("enabled:", s_new["enabledModels"][-1])
EOF
```
  Expected:
  - the new id is in `/v1/models`;
  - the hash equals the sha256 of the file;
  - `0` staging folders are left;
  - two new `.bak-` files, one per Pi file;
  - `others same: True`;
  - the new entry copies QUASAR's or Fable's limits;
  - `settings same but enabledModels: True`;
  - `enabled: freetoken-local/<new id>`.

  Ask Jay to open Pi and confirm the model shows in its picker.

- [ ] **Step 6: Cancel leaves nothing**
  - Start the same download again under a second target. It is refused as "already on this
    PC", which is correct; screenshot `08-no-overwrite.png`.
  - Then start a download of another repo (any model repo from Step 5's list) and press
    **Cancel download** within a few seconds.
  - Expected: "Download cancelled. Its partial files were deleted.", and
    `ls -a ~/ninfer-work/models ~/models | grep incoming` prints nothing. Screenshot
    `09-cancelled.png`.

- [ ] **Step 7 (optional, only with Jay's yes): load the new model, then remove it while loaded**
  - Load it from the list and ask it one chat through `http://100.106.5.124:12020/v1`.
  - Open its settings → **Remove this model…**.
  - Expected: the question says "It is loaded now and will be put away first", and the
    checkbox is unticked. Screenshot `10-remove-loaded.png`.
  - Tick **Also delete the model files** and press Remove.
  - Expected:
    - `/running` no longer lists it;
    - the files are gone (entry and parts);
    - the Pi entry and `enabledModels` row are gone;
    - a new pair of Pi backups exists.
  - Screenshot `11-models-list-removed.png`.
  - If a FreeToken folder was used in Step 5, loading it is this session's one FreeToken boot.

- [ ] **Step 8: Remove without loading (when Step 7 was skipped)**
  - Remove the new model with **Also delete the model files** ticked; screenshots `10`/`11`
    as above.
  - Expected: the same checks as Step 7, without the unload.

- [ ] **Step 9: Put things back and compare**

```bash
PYTHONPATH=python .venv/bin/python - <<EOF
import json
for name in ("models.json", "settings.json"):
    print(name, json.load(open(f"$PI/{name}")) == json.load(open(f"$PI/{name}.bak-before-stage-b-$STAMP")))
EOF
diff <(python3 -m json.tool ~/.config/freetoken/registry.json) <(python3 -m json.tool ~/registry.before-stage-b-$STAMP.json) && echo registry same
```
  Expected: `True`, `True` and `registry same`. If any differs, restore from the
  `*-before-stage-b-*` copies and record why.

- [ ] **Step 10: Record and commit.** Write the acceptance doc. It holds:
  - a table of every step with pass or fail and the numbers: sizes, download time,
    checksum count, hashes, suggested against registry `ramNeedGB`;
  - the repo used;
  - the screenshot list, linked;
  - anything left over.

  Leave the box on `feat/control-panel-b` until the merge.

```bash
git add docs/research/control-panel-b-acceptance-*.md docs/research/control-panel-b-acceptance-*/
git commit -m "docs: control panel stage B live acceptance

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push
```

---

### Task 9: Codex review, fixes, PR, merge

- [ ] **Step 1: Codex review, round 1 (the merge gate).**
  - Run the `codex-review` skill with model **Astra** at thinking level **xhigh**. It is
    read-only, with no git writes. It covers
    `git diff origin/mtp-upstream-merge...feat/control-panel-b -- python/freetoken/daemon/settings tests/settings README.md`,
    with the spec (section 7, Error handling, Testing) and this plan as context.
  - Ask it to check, in order:
    1. the five Review Focus items;
    2. every path that deletes files (`_check_deletable`, `_delete_files`, `_abandon`,
       `_place`), for a way to delete or overwrite something that is not the model's own;
    3. the `_save` changes against Stage A's hold and unknown-state rules;
    4. the Pi writes on a Windows (drvfs) path;
    5. plain words on the page.
  - It lists every finding at once. Fix every must-fix, with the controller dispatching an
    implementer, then rerun Task 7 Step 2.
- [ ] **Step 2: Round 2** covers the fixes only, with the same model and level. Two rounds at
  most (feedback 2026-09-07): judge the rest on the card, not the reviewer's bar.
- [ ] **Step 3: Open the PR** from `feat/control-panel-b` to `mtp-upstream-merge`, titled
  `feat: control panel stage B (add and remove models, Pi sync)`. The body lists:
  - what changed for Jay, in plain words;
  - the detection rule (magic → runtime);
  - the checksum rule;
  - the delete-safety rules;
  - the Pi rules;
  - the test commands and results;
  - the acceptance table with the screenshots linked;
  - the Codex review findings and fixes.

  It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- [ ] **Step 4: Ask Jay to merge.** Squash-merge only on his yes, and only when the Codex
  review passed and the acceptance passed. Then run
  `git checkout mtp-upstream-merge && git pull` on the box and
  `systemctl --user restart freetoken-settings`; this is the same code.
- [ ] **Step 5: Update memory.** Add `project-control-panel-b.md` and an index line. It records:
  - the merge commit;
  - the Pi path and backup names;
  - the staging-folder name;
  - the rollback: the `*-before-stage-b-*` copies;
  - the follow-ups below.

---

## Self-review against the spec (Stage B)

| Spec item (section 7 + error handling + testing) | Where |
|---|---|
| Add wizard step 1: browse the PC with the existing `browse.py` | Task 2 (kind `add`), Task 6 (`openBrowser` `onPick`) |
| Step 1: paste a HuggingFace link, download with the existing `download.py` into `~/models/<name>` or `~/ninfer-work/models/` | Task 3 (`plan_add`/`start_add` on `DownloadManager`), Task 5 (`default_add_roots`) |
| Progress shown | Task 3 (`_refresh` for add jobs), Task 6 (`renderAddJob`, 1.5 s poll, resumes on reopen) |
| Checksum verified where the repo publishes one | Task 3 (`SHA256SUMS` first, else LFS sha256; the page says how many files were checked) |
| Detection: `.ninfer` → NInfer; header v2 → `ninfer`, v3 → `ninfer-upstream` | Task 1 (magic bytes pinned against both frozen runtimes' sources), Task 8 Step 3 live |
| Detection: folder + config.json with a supported architecture → FreeToken | Task 1 (`read_model(...).supported`, the `SUPPORTED_ARCHITECTURES` mirror of `models/register.py`) |
| Anything else → "not supported by your engines" | Tasks 1, 3 (`AddUnsupported` before downloading), 5 (`not_supported`) |
| Identity: id suggested from the file name, unique, registry id rules | Task 1 `suggest_id`; Task 5 server check (`MODEL_ID_RE`, ids and aliases); Task 6 `idProblem` (regex pinned equal) |
| Name; memory need suggested from the file size, editable | Task 1 `suggest_name`/`suggest_ram_gb` (calibrated on QUASAR and Flash); Task 6 editable fields |
| Save regenerates the config; the model appears in the list and in Pi | Task 5 (`add_model` through `_save`, then `pi.add`) |
| Remove: confirm, with "also delete the model files" off by default | Task 6 (`remove-ask`, unticked every time), Task 5 (`deleteFiles` defaults to false) |
| Removing a loaded model unloads it first | Task 5 (`remove_model`; unknown state refused; a failed unload removes nothing) |
| Its FreeToken profile is deleted | Task 5 (`_drop_profile`, and the helper follows the default boot file when it was active) |
| Pi sync: only `freetoken-local` models and `enabledModels` rows; never other providers | Task 4 |
| Timestamped backup of both files before every change | Task 4 (`<file>.bak-<time>`, newest 20 kept) |
| New entries copy contextWindow/maxTokens/thinkingLevelMap from a same-engine neighbour | Task 4 (also reasoning/input/cost; not samplingParams; fallback with a note) |
| Unreachable Pi path → "Pi not updated" and carry on | Task 4 (never raises), Task 5 (`test_pi_out_of_reach_still_adds_and_says_so`), Task 6 (`piNote`, warn notice) |
| Error table: download fails / checksum mismatch → wizard stops, partial files deleted, reason shown | Task 3 (`_abandon`), Task 6 (`downloadLine`) |
| Error table: Pi can't be reached | as above |
| Error table: an impossible value → the field is marked and Save stays off | Task 5 (`add.id`/`add.name`/`add.ramNeedGB` errors), Task 6 (`addValidate`, `add-errors`) |
| Testing: unit tests; page tests in the helper page style; live: add a small NInfer model from a link, it appears in Pi, then remove it | Tasks 1-6; Task 8 Steps 5-8 |
| Reviews before merge | Task 9 (Codex Astra xhigh, two rounds at most, as the merge gate given for this stage) |

**Spec gaps and how this plan closes them:**
1. **Split v3 artifacts.** Upstream v3 models can be one entry plus `<entry>.part-NNNN` files,
   and the spec speaks of one file. Detection lists and sizes the parts. The download fetches
   and checks them. Remove deletes them. A part file picked on its own is refused with a
   pointer to the entry. Delete safety also covers a v3 entry that reads another model's parts.
2. **"Checksum where the repo publishes one".** A repo's `SHA256SUMS` wins. Otherwise the
   Hub's LFS sha256, which the Hub publishes for every LFS file, is used. Small git files with
   neither are not checked, and the page says how many files were.
3. **`download.py` cannot fetch a single `.ninfer` file.** Its allow-list covers only
   safetensors and tokenizer files, and its progress counts the whole target folder, which for
   NInfer is shared with other models. The plan adds a separate "add" job: a staging folder,
   checks, and a never-overwriting place step. The old resumable folder download is left as
   it is.
4. **FreeToken defaults a small model cannot use.** An example is the 262,144-token longest
   chat against a model that reads 8,192. Left as they are, the add fails, and a registry
   holding such a model would block every later save. They become that model's overrides at
   its limit, and the save names each one.
5. **Removing the active FreeToken profile.** `profiles.delete` falls back to the default boot
   file, but the app's `boot_file` did not follow; only the DELETE route called
   `set_active_boot`. A `profile_deleted` hook now does the same.
6. **Removing while the switcher state is unknown.** P5 stops a removed model that is still
   loaded, so an unknown state is refused, like Stage A's unknown-state rules. Adding while
   unknown is allowed: a model new to the list has no switcher entry. This is a one-line
   `_save` change, and Stage A's test for it still passes.
7. **A "next time" hold on a removed model** would stay in `held-models.json`. `_save` now
   drops holds for models no longer in the list.
8. **Removing Pi's default model.** The spec is silent. `defaultModel` is left alone (picking
   another is Jay's call), and the page says so.
9. **Pi entry fields the spec does not name** (`reasoning`, `input`, `cost`): they are copied
   from the same neighbour. `samplingParams` is per model and is not copied. With no
   same-engine neighbour, a fallback is used with a note.
10. **Pi backup retention.** The spec does not say. The newest 20 of each are kept, like the
    registry.
11. **Delete safety.** The spec says only "also delete the model files". The plan deletes only
    the model's own files. It never deletes HOME or above, a non-model folder, or anything
    another model's files are, contain or sit inside. The checks run before anything changes,
    and deletion happens only after the list is saved.

**Left for later (out of this stage):**
- NInfer fit estimates for a newly added model use QUASAR's KV geometry (Stage A's
  `ninfer_fit.py` note). Reading the geometry from the artifact header is a follow-up. Until
  then a new family's fit line can be off, and "Save anyway" still works.
- Renaming a model on its settings view does not rename its Pi entry.
- The old hidden "Model downloads and folders" block in `index.html` and its `/api/downloads`
  routes can be deleted once Stage B has been live for a while.
- NInfer repos that keep the `.ninfer` file in a subfolder are not planned. Top-level files
  only, as `download.py` does today.

**Checks run while writing this plan (2026-09-25).** This plan was not dry-run end to end. It
was written under a no-files rule. Spot checks run on the devbox:
- `suggest_id`: the results match those listed in Task 1.
- The ⌈bytes/GiB⌉ memory suggestions: QUASAR gives 19 and Flash gives 64.
- Pi indent detection keeps a 2-space file at 2 spaces.
- A padded v3 directory parses.
- `five()` FreeToken defaults against a Llama config (8192 positions, 2 layers, 64 hidden)
  fail only on ContextTokens and pass with 8192. They pass the whole `validate_registry` with
  that override.
- A 64-expert, 24-layer Qwen3-MoE also needs MoECacheSize 1536.
- huggingface_hub 1.30.0's `BlobLfsInfo` has `sha256`.
- node v24.20.0 is present.

**Placeholder scan:** none left, apart from `<date>` in the acceptance file names and the test
repo, which is chosen with Jay (Task 8 Step 5, with a stated fallback).

**Type consistency (names used across tasks):**
- `model_detect`:
  - `detect(path, *, taken)`, `read_ninfer`, `model_files(engine, artifact)`, `PART_RE`,
    `suggest_id`, `suggest_name`, `suggest_ram_gb`, `NotAModel`;
  - used in Tasks 1 and 5; the test helpers `write_v2`/`write_v3`/`write_folder` in Tasks 1
    and 5.
- `download`:
  - `DownloadManager.plan_add/start_add(value, entry=None, *, folder_root, ninfer_root)`,
    `latest_add`, `get`, `cancel`;
  - `DownloadJob.kind/engine/staging/target/fetch/finals/sums_name/verified`;
  - `as_dict()["resultPath"/"verified"/"kind"]`;
  - `AddUnsupported`, `InvalidRepository`, `DownloadConflict`, `parse_sha256sums`;
  - used in Tasks 3 and 5; the test helper `Hub` in Tasks 3 and 5.
- `pi_sync`:
  - `PiSync(agent_dir, *, now, enabled).add(model_id, name, engine, engines_by_id)` and
    `.remove(model_id)` return `{status, message, notes}`;
  - used in Tasks 4, 5 and 6 (`piNote`); the test helpers `write_pi`/`MODELS`/`SETTINGS` in
    Tasks 4 and 5.
- `panel`:
  - `PanelService(..., downloads, pi, add_roots, profile_deleted)`, `detect_path`,
    `add_info`, `plan_download`, `start_download`, `download_status`, `cancel_download`,
    `add_model`, `remove_model`;
  - routes `/api/panel/add/{info,detect,plan,downloads[/id[/cancel]]}`, `POST /api/panel/models`,
    `POST /api/panel/models/{id}/remove`;
  - used in Tasks 5 and 6.
- Page:
  - `openBrowser(name, kind, {title, start, onPick})`, `openAdd`, `closeAdd`, `addCheckPath`,
    `addValidate`, `addSave`, `openRemove`, `answerRemove`;
  - helpers `idProblem`, `detectionText`, `planSummary`, `downloadLine`, `removeQuestion`,
    `piNote`, `addedNote`, `removedNote`;
  - used in Task 6.

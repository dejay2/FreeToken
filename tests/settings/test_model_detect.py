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

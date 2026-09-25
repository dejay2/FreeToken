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
directory are read here, never the weights (each listed part's first 8 bytes are checked too).

A folder is a FreeToken model when its config.json names an architecture FreeToken's model
registry serves (model_info.SUPPORTED_ARCHITECTURES, kept in step with models/register.py by
tests/settings/test_model_info.py) and it holds .safetensors weights. Anything else is "not
supported by your engines".

``python -m freetoken.daemon.settings.model_detect PATH...`` prints what each path is; the live
acceptance runs it over every artifact on the box.

This module stays torch-free like the rest of the settings helper (tests/settings/
test_settings_import_safety.py).
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import struct
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
# this one only has to keep a listed part inside the entry's folder (no separators, no "..").
_SIBLING_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
# The largest directory read; QUASAR-size artifacts carry well under 1 MiB of JSON.
MAX_DIRECTORY_BYTES = 64 * 1024 * 1024
NOT_SUPPORTED = "It is not supported by your engines."
PART_WORDS = "This is one part of a split NInfer model. Pick the file ending in .ninfer next to it."
DAMAGED_WORDS = "This NInfer file is damaged: its header does not match the file. " + NOT_SUPPORTED


class NotAModel(ValueError):
    """Plain words for why a path cannot be added."""


def read_ninfer(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a NInfer entry's header (and, for v3, its directory) and say which runtime reads it.

    Returns ``{"version", "runtime", "files", "bytes"}`` where ``files`` is the entry followed
    by its v3 parts and ``bytes`` is their total size. Raises :class:`NotAModel` with plain
    words for a part file, a foreign or truncated header, a damaged directory or a missing part.
    """
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


def _part_names(directory: bytes) -> list[str]:
    """The part names a v3 directory lists, checked for shape only (nothing on disk is looked at)."""
    try:
        doc = json.loads(directory.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise NotAModel(DAMAGED_WORDS) from exc
    records = doc.get("files") if isinstance(doc, dict) else None
    if (not isinstance(records, list) or not records or not isinstance(records[0], dict)
            or records[0].get("path") is not None):
        raise NotAModel(DAMAGED_WORDS)
    names: list[str] = []
    for record in records[1:]:
        name = record.get("path") if isinstance(record, dict) else None
        if not isinstance(name, str) or not _SIBLING_RE.fullmatch(name):
            raise NotAModel(DAMAGED_WORDS)
        names.append(name)
    return names


def _v3(entry: Path, directory: bytes, size: int) -> dict[str, Any]:
    files, total = [str(entry)], size
    for name in _part_names(directory):
        part = entry.with_name(name)
        try:
            total += part.stat().st_size
            with part.open("rb") as fh:
                magic = fh.read(len(PART_MAGIC))
        except OSError as exc:
            raise NotAModel(f"This NInfer model is split into parts, and the part {name} is missing next to it.") from exc
        # The directory only names the part; the part's own header proves it is one (a file of
        # the right name from another model or a broken copy would otherwise pass as a model).
        if magic != PART_MAGIC:
            raise NotAModel(f"This NInfer model is split into parts, and the part {name} next to it is not a NInfer "
                            f"part file. {NOT_SUPPORTED}")
        files.append(str(part))
    return {"version": 3, "runtime": RUNTIME_BY_VERSION[3], "files": files, "bytes": total}


class OwnershipUnknown(ValueError):
    """A registered model's files cannot be listed (its NInfer header cannot be read)."""


def model_files(engine: str, artifact: str) -> list[str]:
    """Everything that belongs to a model on disk: a NInfer entry plus its v3 parts, or a
    FreeToken folder. Paths that do not exist are left out. A symlinked NInfer entry is only
    the link: its target's parts belong to whatever the target is (review round, PR #17)."""
    path = Path(os.path.expanduser(artifact))
    if engine != "ninfer":
        return [str(path)] if path.is_dir() else []
    if path.is_symlink():
        return [str(path)]
    try:
        return read_ninfer(path)["files"]
    except NotAModel:
        # Damaged or half-deleted: the entry (if any) plus the parts named after it.
        found = [path] if path.is_file() else []
        found += sorted(path.parent.glob(glob.escape(path.name) + ".part-[0-9][0-9][0-9][0-9]"))
        return [str(item) for item in found]


def _listed_parts(entry: Path) -> list[str]:
    """The part names an entry's header lists, whether or not those parts exist. Raises
    OSError (unreadable) or NotAModel (a NInfer header that cannot be parsed)."""
    with entry.open("rb") as fh:
        head = fh.read(V3_HEADER.size)
        if head[:8] != V3_MAGIC:
            return []  # v2 (one file) or not NInfer at all: it references nothing else
        if len(head) < V3_HEADER.size:
            raise NotAModel(DAMAGED_WORDS)
        _, json_bytes, _ = V3_HEADER.unpack(head)
        if not 0 < json_bytes <= MAX_DIRECTORY_BYTES:
            raise NotAModel(DAMAGED_WORDS)
        directory = fh.read(json_bytes)
    if len(directory) != json_bytes:
        raise NotAModel(DAMAGED_WORDS)
    return _part_names(directory)


def referenced_files(engine: str, artifact: str) -> list[str]:
    """Every path a registered model may read, for deciding what another model's delete must
    not touch. Unlike model_files this does not need the model to be loadable: a v3 header's
    part list counts even when a listed part is missing, and a symlinked entry counts both its
    own folder and its target's. Raises OwnershipUnknown when an entry exists but its header
    cannot be read, because then which parts it uses cannot be known (review round, PR #17)."""
    path = Path(os.path.expanduser(artifact))
    if engine != "ninfer":
        return [str(path)]
    entries = [path]
    if path.is_symlink():
        entries.append(Path(os.path.realpath(path)))
    out: list[str] = [str(path)]
    for entry in entries:
        out += [str(item) for item in sorted(entry.parent.glob(glob.escape(entry.name) + ".part-[0-9][0-9][0-9][0-9]"))]
        if not entry.exists():
            continue
        try:
            names = _listed_parts(entry)
        except (OSError, NotAModel) as exc:
            raise OwnershipUnknown(str(path)) from exc
        out += [str(entry.with_name(name)) for name in names]
        if path.is_symlink():
            out += [str(path.with_name(name)) for name in names]
    return list(dict.fromkeys(out))


def suggest_id(stem: str, taken: Iterable[str]) -> str:
    """A registry id from a file or folder name: lower-case, runs of other characters become
    ``-``, made unique against ``taken`` (ids and aliases, case-insensitive) with ``-2``, ``-3``…
    Always matches registry.MODEL_ID_RE."""
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
    """Say what ``path`` is. ``kind`` is ``ninfer``, ``freetoken`` or ``unsupported``; an
    unsupported result carries plain words in ``reason`` and no ``suggested`` block. The
    suggested id is made unique against ``taken`` (every registry id and alias)."""
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


__all__ = [
    "DAMAGED_WORDS", "MAX_DIRECTORY_BYTES", "NINFER_SUFFIX", "NOT_SUPPORTED", "NotAModel", "OwnershipUnknown", "PART_MAGIC",
    "PART_RE", "PART_WORDS", "RUNTIME_BY_VERSION", "V2_MAGIC", "V2_PREFIX", "V3_HEADER", "V3_MAGIC",
    "detect", "main", "model_files", "read_ninfer", "referenced_files", "suggest_id", "suggest_name", "suggest_ram_gb",
]

"""Folder and file listing for the page's Browse buttons.

The helper only ever answers on 127.0.0.1, and this module only lists names: it never opens a
file. A folder counts as a model folder when it holds ``config.json`` next to at least one
``.safetensors`` file, which is what every checkpoint this fork serves looks like.
Kind ``add`` (the control panel's Add model wizard) lists folders and ``.ninfer`` files and
marks model folders and ``.ninfer`` files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MAX_ENTRIES = 500
BROWSE_KINDS = ("folder", "model", "file", "add")


def drives() -> list[str]:
    if hasattr(os, "listdrives"):
        return [drive for drive in os.listdrives()]
    return ["/"]


def resolve_start(path: str | None) -> Path | None:
    """Turn the dial's stored text into an existing directory, or None to show the drives."""
    if not path:
        return None
    text = os.path.expandvars(os.path.expanduser(path.strip().strip('"').strip("'")))
    if "$" in text or "(" in text:
        # PowerShell expressions such as (Join-Path $env:LOCALAPPDATA ...) are not paths.
        return None
    candidate = Path(text)
    for probe in (candidate, candidate.parent):
        if probe == Path("."):
            # A bare name (or a Windows path read on POSIX) has "." for a parent; the
            # helper's working directory is never what the page asked for.
            continue
        try:
            if probe.is_dir():
                return probe.resolve()
        except OSError:
            continue
    return None


def is_model_folder(folder: Path) -> bool:
    try:
        if not (folder / "config.json").is_file():
            return False
        return any(child.suffix == ".safetensors" for child in folder.iterdir() if child.is_file())
    except OSError:
        return False


def list_directory(path: str | None, kind: str = "folder") -> dict[str, Any]:
    if kind not in BROWSE_KINDS:
        raise ValueError(f"kind must be one of {', '.join(BROWSE_KINDS)}")
    start = resolve_start(path)
    if start is None:
        return {
            "path": "",
            "parent": None,
            "drives": drives(),
            "entries": [{"name": drive, "path": drive, "kind": "dir", "isModel": False} for drive in drives()],
            "isModel": False,
            "truncated": False,
        }
    entries: list[dict[str, Any]] = []
    truncated = False
    try:
        children = sorted(start.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError:
        children = []
    for child in children:
        try:
            if child.name.startswith(("$", ".")) or child.name.lower() in {"system volume information"}:
                continue
            is_dir = child.is_dir()
        except OSError:
            continue
        if not is_dir and kind != "file" and not (kind == "add" and child.name.endswith(".ninfer")):
            continue
        if len(entries) >= MAX_ENTRIES:
            truncated = True
            break
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "kind": "dir" if is_dir else "file",
                "isModel": bool((is_dir and kind in ("model", "add") and is_model_folder(child))
                                or (not is_dir and kind == "add")),
            }
        )
    parent = start.parent
    return {
        "path": str(start),
        "parent": None if parent == start else str(parent),
        "drives": drives(),
        "entries": entries,
        "isModel": kind in ("model", "add") and is_model_folder(start),
        "truncated": truncated,
    }


__all__ = ["BROWSE_KINDS", "MAX_ENTRIES", "drives", "is_model_folder", "list_directory", "resolve_start"]

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


def _indent(text: str) -> int | str | None:
    """The file's own indent: None for a single-line (compact) file, a tab, or the width of the
    first indented line (2 when unsure)."""
    if "\n" not in text.rstrip("\r\n"):
        return None
    for line in text.splitlines()[1:]:
        stripped = line.lstrip(" \t")
        if stripped and len(stripped) != len(line):
            lead = line[: len(line) - len(stripped)]
            return "\t" if lead.startswith("\t") else len(lead)
    return 2


def _render(doc: Any, original: str, bom: bool) -> bytes:
    """Serialise ``doc`` in the original file's shape: indent, final newline, CRLF and BOM.

    json.dumps never emits a raw newline inside a string (it escapes them), so the CRLF
    replacement only touches the line ends it produced.
    """
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
                # A row whose id is not a string (a hand-edited file) is not a neighbour and
                # must not raise (an unhashable id would in a plain dict lookup).
                neighbour = next((row for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)
                                  and engines_by_id.get(row["id"]) == engine), None)
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
        """Read both files, apply ``edit`` to copies, and write only what changed, after backing
        both files up. Every failure answers "not_updated"; nothing is written before the
        files have been read and parsed and the provider found."""
        if not self.enabled:
            return _result("not_updated", "Pi sync is off on this helper.")
        raw: dict[Path, bytes] = {}
        try:
            for path in (self.models_path, self.settings_path):
                raw[path] = path.read_bytes()
        except OSError as exc:
            return _result("not_updated", f"Pi's files could not be read in {self.agent_dir} ({exc.strerror or exc}).")
        # Strict decoding: a byte that is not UTF-8 must never be written back as U+FFFD.
        try:
            texts = {path: data.decode("utf-8-sig") for path, data in raw.items()}
        except UnicodeDecodeError as exc:
            return _result("not_updated", f"One of Pi's files is not UTF-8 text ({exc.reason} at byte {exc.start}).")
        try:
            models, settings = json.loads(texts[self.models_path]), json.loads(texts[self.settings_path])
        except ValueError as exc:
            return _result("not_updated", f"One of Pi's files is not valid JSON ({exc}).")
        providers = models.get("providers") if isinstance(models, dict) else None
        provider = providers.get(PROVIDER) if isinstance(providers, dict) else None
        if not isinstance(settings, dict) or not isinstance(provider, dict) or not isinstance(provider.get("models"), list):
            return _result("not_updated", f"Pi's models.json has no {PROVIDER} model list.")
        new_models, new_settings = copy.deepcopy(models), copy.deepcopy(settings)
        try:
            notes = edit(new_models["providers"][PROVIDER]["models"], new_settings)
        except Exception as exc:  # noqa: BLE001 - Jay's files can hold anything; the sync never raises
            return _result("not_updated", f"Pi's files hold something unexpected ({exc.__class__.__name__}: {exc}).")
        writes = [(path, doc) for path, doc, old in ((self.models_path, new_models, models),
                                                     (self.settings_path, new_settings, settings)) if doc != old]
        if not writes:
            return _result("unchanged", "Pi already matched.", notes)
        written: list[Path] = []
        try:
            stamp = self._now().strftime("%Y%m%d-%H%M%S-%f")
            for path in (self.models_path, self.settings_path):
                path.with_name(f"{path.name}.bak-{stamp}").write_bytes(raw[path])
            for path, doc in writes:
                self._atomic_write(path, _render(doc, texts[path], raw[path].startswith(codecs.BOM_UTF8)))
                written.append(path)
            self._prune()
        except OSError as exc:
            # A half-applied change (models.json new, settings.json old) would leave Pi listing
            # a model it cannot enable; put back what this change replaced, from its own bytes.
            restored = self._restore(raw, written)
            state = "Pi was left as it was" if restored else f"restore it from the backups in {self.agent_dir}"
            return _result("not_updated", f"Pi's files could not be written ({exc.strerror or exc}); {state}.", notes)
        return _result("updated", "Pi's model list was updated.", notes)

    def _restore(self, raw: Mapping[Path, bytes], written: list[Path]) -> bool:
        """Undo this change's writes (best effort) and drop any temp file; True when all undone."""
        ok = True
        for path in written:
            try:
                self._atomic_write(path, raw[path])
            except OSError:
                ok = False
        for path in (self.models_path, self.settings_path):
            try:
                self._temp_path(path).unlink(missing_ok=True)
            except OSError:
                pass
        return ok

    @staticmethod
    def _temp_path(path: Path) -> Path:
        return path.with_name(path.name + ".tmp-freetoken")

    @classmethod
    def _atomic_write(cls, path: Path, data: bytes) -> None:
        temporary = cls._temp_path(path)
        temporary.write_bytes(data)
        os.replace(temporary, path)

    def _prune(self) -> None:
        for path in (self.models_path, self.settings_path):
            prefix = path.name + ".bak-"
            names = sorted((p.name for p in self.agent_dir.iterdir() if p.name.startswith(prefix)), reverse=True)
            for name in names[BACKUPS_KEPT:]:
                (self.agent_dir / name).unlink(missing_ok=True)


__all__ = ["BACKUPS_KEPT", "COPIED_FIELDS", "FALLBACK", "PROVIDER", "PiSync", "default_agent_dir"]

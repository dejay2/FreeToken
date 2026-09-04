"""Line-preserving parser and safe writer for the machine-local PowerShell boot file."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .dials import DIALS, DIAL_BY_NAME, ENV_DIALS, canonical_value, validate_settings


class BootParseError(RuntimeError):
    """The boot script is not in the small, supported line-oriented shape."""

    def __init__(self, message: str, *, line: int | None = None, path: str | None = None) -> None:
        self.message = message
        self.line = line
        self.path = path
        location = ""
        if path:
            location += path
        if line is not None:
            location += f":{line}"
        super().__init__(f"{location}: {message}" if location else message)


class BootValidationError(ValueError):
    """A caller attempted to save a value outside the dial catalogue."""

    def __init__(self, errors: list[dict[str, str]]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))


@dataclass
class LauncherArg:
    name: str
    raw_value: str | None
    is_switch: bool
    indent: str
    line_no: int
    line_index: int
    source_line: str

    @property
    def trailing_backtick(self) -> bool:
        return _has_continuation(self.source_line)


@dataclass
class EnvAssignment:
    name: str
    value: str
    quote: str
    indent: str
    line_no: int
    line_index: int
    source_line: str


@dataclass
class BootToken:
    kind: str
    text: str
    line_no: int
    line_index: int


@dataclass
class BootDocument:
    text: str
    lines: list[str]
    newline: str
    launcher_start: int
    launcher_end: int
    launcher_args: list[LauncherArg]
    env_assignments: dict[str, EnvAssignment]
    tokens: list[BootToken] = field(default_factory=list)

    @property
    def args_by_name(self) -> dict[str, LauncherArg]:
        return {arg.name: arg for arg in self.launcher_args}


_ENV_RE = re.compile(
    r"^(?P<indent>\s*)\$env:(?P<name>FREETOKEN_[A-Za-z0-9_]+)\s*=\s*"
    r"(?P<quote>['\"]?)(?P<value>.*?)(?P=quote)\s*$"
)
_LAUNCHER_START_RE = re.compile(r"^\s*&\s*\$launcher\s*`?\s*$")
_ARG_RE = re.compile(
    r"^(?P<indent>\s*)-(?P<name>[A-Za-z0-9_]+)"
    r"(?:\s+(?P<value>.*?))?\s*(?P<tick>`)?\s*$"
)
_COMMENTED_ARG_RE = re.compile(r"^\s*#\s*-(?P<name>[A-Za-z0-9_]+)(?:\s+(?P<value>.*?))?\s*$")
_LAUNCHER_ORDER = (
    "ModelPath",
    "Port",
    "ContextTokens",
    "KVCacheTokens",
    "MaxRunningRequests",
    "MoECacheSize",
    "GpuOwnedLayers",
    "CudaGraphMaxBS",
    "KVDtype",
    "DesktopPython",
    "DenseQuant",
    "EmbedHost",
    "EnableVision",
    "VisionPackagesPath",
    "VisionExecution",
    "VisionWeights",
    "ExpertLoad",
    "EnableCacheReport",
    "CollectRoutingStats",
    "KVPark",
    "KVParkIdleMs",
    "KVParkMinTokens",
    "KVParkRAMGiB",
    "KVParkSSDDir",
    "KVParkSSDGiB",
    "KVParkWindowMiB",
    "MoEVramReserveBytes",
    "MoECacheHeadroomBytes",
)
_ORDER_INDEX = {name: index for index, name in enumerate(_LAUNCHER_ORDER)}


def _without_eol(line: str) -> str:
    return line.rstrip("\r\n")


def _ending(line: str, fallback: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return "" if not fallback else fallback


def _has_continuation(line: str) -> bool:
    return _without_eol(line).rstrip().endswith("`")


def _unquote(value: str, quote: str) -> str:
    if not quote:
        return value.strip()
    value = value.strip()
    if len(value) < 2 or value[-1] != quote:
        return value
    value = value[1:-1]
    if quote == "'":
        return value.replace("''", "'")
    return value.replace('""', '"')


def _value_from_raw(raw: str | None) -> Any:
    if raw is None or not raw.strip():
        return True
    value = raw.strip()
    if len(value) >= 2 and value[0] in "'\"" and value[-1] == value[0]:
        return _unquote(value, value[0])
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass
    return value


def _quote_literal(value: str) -> str:
    if value.startswith("(") or value.startswith("$"):
        return value
    if re.fullmatch(r"[A-Za-z0-9_.:+-]+", value):
        return value
    return "'" + value.replace("'", "''") + "'"


def _serialize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "" if value else ""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return _quote_literal(value)
    return _quote_literal(str(value))


def _make_tokens(
    lines: list[str],
    *,
    launcher_start: int,
    launcher_end: int,
    launcher_args: list[LauncherArg],
) -> list[BootToken]:
    arg_indexes = {arg.line_index for arg in launcher_args}
    tokens: list[BootToken] = []
    for index, line in enumerate(lines):
        content = _without_eol(line)
        if index == launcher_start:
            kind = "LAUNCHER_START"
        elif index in arg_indexes:
            kind = "LAUNCHER_ARG"
        elif _ENV_RE.match(content):
            kind = "ENV_VAR"
        elif _COMMENTED_ARG_RE.match(content):
            kind = "COMMENTED_ARG"
        elif not content.strip():
            kind = "BLANK"
        elif content.lstrip().startswith("#"):
            kind = "COMMENT"
        else:
            kind = "RAW_SCRIPT"
        tokens.append(BootToken(kind, line, index + 1, index))
    return tokens


def _parse_text(text: str, path: str | None = None) -> BootDocument:
    lines = text.splitlines(keepends=True)
    if not lines and text:
        lines = [text]
    newline = "\r\n" if "\r\n" in text else "\n"
    env: dict[str, EnvAssignment] = {}
    launcher_start: int | None = None
    launcher_end: int | None = None
    args: list[LauncherArg] = []

    for index, line in enumerate(lines):
        content = _without_eol(line)
        match = _ENV_RE.match(content)
        if match:
            quote = match.group("quote")
            raw_value = match.group("value")
            if quote and not content.rstrip().endswith(quote):
                raise BootParseError("unterminated environment value", line=index + 1, path=path)
            env_name = match.group("name")
            if env_name in env:
                raise BootParseError(f"duplicate environment assignment {env_name}", line=index + 1, path=path)
            env[env_name] = EnvAssignment(
                name=env_name,
                value=_unquote(raw_value, quote),
                quote=quote,
                indent=match.group("indent"),
                line_no=index + 1,
                line_index=index,
                source_line=line,
            )
        if launcher_start is None and _LAUNCHER_START_RE.match(content):
            launcher_start = index
            if not _has_continuation(line):
                launcher_end = index
                continue
            cursor = index + 1
            previous = line
            while _has_continuation(previous):
                if cursor >= len(lines):
                    raise BootParseError("launcher continuation reaches end of file", line=index + 1, path=path)
                candidate = lines[cursor]
                arg_match = _ARG_RE.match(_without_eol(candidate))
                if not arg_match:
                    raise BootParseError(
                        "launcher continuation is not a parameter line", line=cursor + 1, path=path
                    )
                value = arg_match.group("value")
                args.append(
                    LauncherArg(
                        name=arg_match.group("name"),
                        raw_value=value.strip() if value is not None else None,
                        is_switch=value is None or not value.strip(),
                        indent=arg_match.group("indent"),
                        line_no=cursor + 1,
                        line_index=cursor,
                        source_line=candidate,
                    )
                )
                previous = candidate
                cursor += 1
            launcher_end = cursor - 1
            break

    if launcher_start is None or launcher_end is None:
        raise BootParseError("launcher call is missing", path=path)
    if not args:
        raise BootParseError("launcher call has no parameter lines", line=launcher_start + 1, path=path)
    seen: set[str] = set()
    for arg in args:
        if arg.name in seen:
            raise BootParseError(f"duplicate launcher parameter {arg.name}", line=arg.line_no, path=path)
        seen.add(arg.name)
    return BootDocument(
        text=text,
        lines=lines,
        newline=newline,
        launcher_start=launcher_start,
        launcher_end=launcher_end,
        launcher_args=args,
        env_assignments=env,
        tokens=_make_tokens(
            lines,
            launcher_start=launcher_start,
            launcher_end=launcher_end,
            launcher_args=args,
        ),
    )


def parse_boot_text(text: str, *, path: str | None = None) -> BootDocument:
    """Parse already-read text; useful for the staged-file self-check and tests."""
    return _parse_text(text, path)


class BootFile:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    @property
    def backup_path(self) -> Path:
        return self.path.with_name(self.path.name + ".bak")

    @property
    def temp_path(self) -> Path:
        return self.path.with_name(self.path.name + ".tmp")

    def read_document(self) -> BootDocument:
        try:
            # ``newline=''`` keeps CRLF/LF exactly as the owner wrote it; PowerShell accepts both,
            # but a settings save should not turn an otherwise untouched file into a whole-file diff.
            with self.path.open("r", encoding="utf-8", newline="") as fh:
                text = fh.read()
        except FileNotFoundError as exc:
            raise BootParseError("boot file does not exist", path=str(self.path)) from exc
        except UnicodeDecodeError as exc:
            raise BootParseError(f"boot file is not UTF-8: {exc}", path=str(self.path)) from exc
        except OSError as exc:
            raise BootParseError(f"cannot read boot file: {exc}", path=str(self.path)) from exc
        return _parse_text(text, str(self.path))

    def parse(self) -> BootDocument:
        return self.read_document()

    def tokenize(self) -> list[BootToken]:
        return self.read_document().tokens

    def load(self) -> dict[str, Any]:
        return self.settings_from_document(self.read_document())

    current_settings = load

    @staticmethod
    def settings_from_document(document: BootDocument) -> dict[str, Any]:
        active = document.args_by_name
        values: dict[str, Any] = {}
        for dial in DIALS:
            if dial.source == "env":
                assignment = document.env_assignments.get(dial.name)
                values[dial.name] = assignment.value if assignment is not None else dial.default
                continue
            arg = active.get(dial.name)
            if arg is not None:
                values[dial.name] = _value_from_raw(arg.raw_value)
            elif dial.control == "toggle":
                values[dial.name] = False
            elif dial.name == "KVPark":
                values[dial.name] = "off"
            else:
                values[dial.name] = dial.default
        return values

    def save(self, changes: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(changes, dict):
            raise BootValidationError([{"field": "settings", "message": "must be an object"}])
        errors = validate_settings(changes, ceilings_only=True)  # the app already applied the model limits
        if errors:
            raise BootValidationError(errors)
        document = self.read_document()  # parse before backup: a broken source is never rewritten
        normalized = {name: canonical_value(DIAL_BY_NAME[name], value) for name, value in changes.items()}
        merged = self.settings_from_document(document)
        merged.update(normalized)
        staged = self._serialize(document, normalized)

        # The backup is deliberately made before the temp write. If the staged self-check fails,
        # the source stays byte-for-byte untouched and the backup still records the last good file.
        try:
            shutil.copy2(self.path, self.backup_path)
            self._write_bytes(self.temp_path, staged)
            with self.temp_path.open("r", encoding="utf-8", newline="") as fh:
                staged_text = fh.read()
            _parse_text(staged_text, str(self.temp_path))
            os.replace(self.temp_path, self.path)
        except BootParseError:
            try:
                self.temp_path.unlink()
            except FileNotFoundError:
                pass
            raise
        except Exception:
            try:
                self.temp_path.unlink()
            except FileNotFoundError:
                pass
            raise
        return merged

    def write(self, changes: dict[str, Any]) -> dict[str, Any]:
        return self.save(changes)

    def _serialize(self, document: BootDocument, changes: dict[str, Any]) -> str:
        lines = list(document.lines)
        args = list(document.launcher_args)
        by_name = {arg.name: arg for arg in args}
        result_args: list[tuple[str, str | None, str]] = []
        for arg in args:
            if arg.name in changes:
                dial = DIAL_BY_NAME[arg.name]
                value = changes[arg.name]
                if dial.control == "toggle":
                    if dial.source == "env":
                        # env assignments are handled below; an env dial cannot be in this block
                        continue
                    if not value:
                        continue
                    raw = None
                elif arg.name == "KVPark" and value == "off":
                    # The baseline represents parking-off by omission; keep that shape instead of
                    # turning the setting into an unnecessary explicit argument.
                    continue
                else:
                    raw = _serialize_value(value)
                result_args.append((arg.name, raw, arg.indent))
            else:
                result_args.append((arg.name, arg.raw_value if not arg.is_switch else None, arg.indent))

        # A missing active switch/value is appended, preserving the existing parameter order and all
        # profile comments below the launcher. This is what activates the commented KVPark candidate.
        existing_names = set(by_name)
        for dial in DIALS:
            if dial.source == "env" or dial.name in existing_names or dial.name not in changes:
                continue
            value = changes[dial.name]
            if dial.control == "toggle":
                if not value:
                    continue
                raw = None
            elif dial.name == "KVPark" and value == "off":
                continue
            else:
                raw = _serialize_value(value)
            indent = args[0].indent if args else "    "
            entry = (dial.name, raw, indent)
            insert_at = len(result_args)
            dial_order = _ORDER_INDEX.get(dial.name, len(_LAUNCHER_ORDER))
            for index, existing in enumerate(result_args):
                if _ORDER_INDEX.get(existing[0], len(_LAUNCHER_ORDER)) > dial_order:
                    insert_at = index
                    break
            result_args.insert(insert_at, entry)

        start_line = lines[document.launcher_start]
        start_end = _ending(start_line, document.newline)
        final_original = lines[document.launcher_end]
        final_end = _ending(final_original, start_end or document.newline)
        block: list[str] = [start_line]
        if result_args:
            for index, (name, raw, indent) in enumerate(result_args):
                value_part = "" if raw is None else f" {raw}"
                is_final = index == len(result_args) - 1
                continuation = "`" if not is_final else ""
                separator = "" if is_final else " "
                ending = final_end if is_final else (start_end or document.newline)
                block.append(f"{indent}-{name}{value_part}{separator}{continuation}{ending}")
        else:
            # Retain a syntactically valid call even if every switch was turned off.
            block[0] = _without_eol(start_line).rstrip("`").rstrip() + (start_end or document.newline)

        lines[document.launcher_start : document.launcher_end + 1] = block
        # Env values use their original quote style and indentation. Missing env assignments are
        # inserted immediately before the launcher so the header remains a recognisable segment.
        inserted: list[str] = []
        for dial in DIALS:
            if dial.source != "env" or dial.name not in changes:
                continue
            assignment = document.env_assignments.get(dial.name)
            quote = assignment.quote if assignment is not None and assignment.quote else "'"
            indent = assignment.indent if assignment is not None else ""
            value = str(changes[dial.name]).replace(quote, quote + quote) if quote else str(changes[dial.name])
            rendered = f"{indent}$env:{dial.name} = {quote}{value}{quote}{start_end or document.newline}"
            if assignment is None:
                inserted.append(rendered)
            else:
                # The launcher replacement may have shifted indices, so locate by original line
                # identity before applying insertions.
                old = lines[assignment.line_index]
                old_end = _ending(old, document.newline)
                lines[assignment.line_index] = f"{indent}$env:{dial.name} = {quote}{value}{quote}{old_end}"
        if inserted:
            # Recompute the launcher index after the block replacement; insert before it.
            launcher_index = next(
                i for i, line in enumerate(lines) if _LAUNCHER_START_RE.match(_without_eol(line))
            )
            lines[launcher_index:launcher_index] = inserted
        return "".join(lines)

    @staticmethod
    def _write_bytes(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())


# Names that make the small module convenient to use from callers and tests.
BootParser = BootFile


__all__ = [
    "BootDocument",
    "BootToken",
    "BootFile",
    "BootParseError",
    "BootParser",
    "BootValidationError",
    "EnvAssignment",
    "LauncherArg",
    "parse_boot_text",
]

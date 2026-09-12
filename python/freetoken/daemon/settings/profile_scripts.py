"""Render a named settings snapshot as a runnable PowerShell boot file."""

from __future__ import annotations

from typing import Any

from .boot_parser import _LAUNCHER_ORDER, _serialize_value
from .dials import DIALS, DIAL_BY_NAME, ENV_DIALS, canonical_value, validate_settings


_DEFAULT_LAUNCHER = r"..\scripts\start-freetoken-windows.ps1"


def _enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _env_text(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return "1"
    if text in {"0", "false", "no", "off", ""}:
        return "0"
    return str(value)


def _single_quote(value: str) -> str:
    return value.replace("'", "''")


def _setting(settings: dict[str, Any], name: str) -> Any:
    dial = DIAL_BY_NAME[name]
    return settings[name] if name in settings else dial.default


def render_boot_script(
    settings: dict[str, Any], *, launcher_relative: str = _DEFAULT_LAUNCHER, title: str = "FreeToken profile"
) -> str:
    """Return a self-contained boot file for a settings snapshot.

    The line shape intentionally matches :class:`BootFile`: environment assignments come first,
    followed by a variable holding the launcher and one continued ``-Parameter`` line per
    non-switch dial. This makes a profile both runnable by PowerShell and editable by the helper.
    """
    if not isinstance(settings, dict):
        raise TypeError("settings must be an object")
    errors = validate_settings(settings, ceilings_only=True)
    if errors:
        raise ValueError("; ".join(item["message"] for item in errors))

    safe_title = " ".join(str(title).replace("\r", " ").replace("\n", " ").split()) or "FreeToken profile"
    lines = [f"# {safe_title}", "$ErrorActionPreference = 'Stop'"]

    # The old default boot file names this path with a local PowerShell variable. Keep that
    # spelling so the snapshot round-trips, but bind it relative to the repository when the
    # generated file lives under boot-profiles/.
    vision_path = _setting(settings, "VisionPackagesPath")
    if vision_path == "$visionPackages":
        lines.extend(
            [
                "",
                "$visionPackages = Join-Path (Split-Path $PSScriptRoot -Parent) '.local\\vision-packages'",
            ]
        )

    for dial in DIALS:
        if dial.source != "env" or dial.name not in ENV_DIALS:
            continue
        value = _setting(settings, dial.name)
        # Toggles render as 1/0; an env amount (the guess depth) keeps its number.
        text = _env_text(value) if dial.control == "toggle" else str(canonical_value(dial, value))
        lines.append(f"$env:{dial.name} = '{_single_quote(text)}'")

    launcher = _single_quote(str(launcher_relative))
    lines.extend(["", f"$launcher = Join-Path $PSScriptRoot '{launcher}'", "", "& $launcher `"])

    rendered: list[tuple[str, str | None]] = []
    for name in _LAUNCHER_ORDER:
        dial = DIAL_BY_NAME.get(name)
        if dial is None:
            continue
        if name not in settings:
            # A dial the caller's settings snapshot never mentions is left out of the
            # rendered launcher line entirely, not padded with today's dial default: a
            # profile that never opted into a knob (e.g. the dynamic KV pool dials) must
            # not start carrying that knob's default the next time it is regenerated from
            # a partial update. Previously this leniency applied only to helper-only knobs;
            # widened to every dial after a profile-update review found a Port-only save
            # baking KVDynamic/KVFloorTokens/... defaults into an otherwise plain profile.
            continue
        value = _setting(settings, name)
        if dial.control == "toggle":
            if _enabled(value):
                rendered.append((name, None))
            continue
        rendered.append((name, _serialize_value(value)))

    for index, (name, raw) in enumerate(rendered):
        value = "" if raw is None else f" {raw}"
        continuation = " `" if index < len(rendered) - 1 else ""
        lines.append(f"    -{name}{value}{continuation}")

    return "\n".join(lines) + "\n"


__all__ = ["render_boot_script"]

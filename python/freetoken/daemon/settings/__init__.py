"""Torch-free Windows settings helper for the FreeToken serving boot file."""

from __future__ import annotations

__all__ = ["BootFile", "BootParseError", "ProcessManager", "ProfilesManager", "create_app"]


def __getattr__(name: str):
    if name in {"BootFile", "BootParseError"}:
        from .boot_parser import BootFile, BootParseError

        return {"BootFile": BootFile, "BootParseError": BootParseError}[name]
    if name == "ProcessManager":
        from .process_manager import ProcessManager

        return ProcessManager
    if name == "ProfilesManager":
        from .profiles_manager import ProfilesManager

        return ProfilesManager
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(name)

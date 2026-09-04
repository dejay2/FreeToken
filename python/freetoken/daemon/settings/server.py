"""Command-line runner for the loopback-only settings helper."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .app import HELPER_VERSION, create_app, default_paths
from .process_manager import ProcessManager
from .profiles_manager import ProfilesManager


def _parser() -> argparse.ArgumentParser:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="FreeToken Windows settings helper")
    parser.add_argument("--port", type=int, default=int(os.environ.get("FREETOKEN_SETTINGS_PORT", "2021")))
    parser.add_argument("--boot-file", default=str(paths["boot"]), help="PowerShell boot file to edit")
    parser.add_argument("--stop-script", default=str(paths["stop"]), help="PowerShell stop script")
    parser.add_argument("--log-file", default=str(paths["log"]), help="Captured server log")
    parser.add_argument("--profiles-file", default=str(paths["profiles"]), help="JSON profile store")
    parser.add_argument("--gpu-lock", default=str(paths["lock"]), help="GPU ownership file")
    parser.add_argument("--job-id", default=os.environ.get("FREETOKEN_SETTINGS_JOB_ID"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    boot = Path(args.boot_file)
    process_manager = ProcessManager(
        boot_file=boot,
        stop_script=args.stop_script,
        log_path=args.log_file,
        lock_path=args.gpu_lock,
        port=2020,
        owner_id=args.job_id,
    )
    profiles = ProfilesManager(args.profiles_file, boot_file=boot)
    app = create_app(
        boot_file=boot,
        process_manager=process_manager,
        profiles=profiles,
        log_path=args.log_file,
        version=HELPER_VERSION,
    )
    try:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
    finally:
        process_manager.close()
    return 0


__all__ = ["main"]

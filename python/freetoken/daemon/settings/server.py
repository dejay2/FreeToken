"""Command-line runner for the loopback-only settings helper."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .app import HELPER_VERSION, create_app, default_paths
from .process_manager import ProcessManager
from .profiles_manager import ProfilesManager

logger = logging.getLogger("freetoken.daemon.settings.server")


def _parser() -> argparse.ArgumentParser:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="FreeToken Windows settings helper")
    # 2031 avoids engine worker ports 2020-2029; measured by the stop script's port sweep.
    parser.add_argument("--port", type=int, default=int(os.environ.get("FREETOKEN_SETTINGS_PORT", "2031")))
    parser.add_argument("--boot-file", default=str(paths["boot"]), help="PowerShell boot file to edit")
    parser.add_argument("--stop-script", default=str(paths["stop"]), help="PowerShell stop script")
    parser.add_argument("--log-file", default=str(paths["log"]), help="Captured server log")
    parser.add_argument("--profiles-file", default=str(paths["profiles"]), help="JSON profile store")
    parser.add_argument("--gpu-lock", default=str(paths["lock"]), help="GPU ownership file")
    parser.add_argument("--job-id", default=os.environ.get("FREETOKEN_SETTINGS_JOB_ID"))
    return parser


def start_panel(app) -> None:
    """Control panel start-up. First put away a model an interrupted Test-tab test left on test
    settings (playground.recover), then make the switcher file match the registry, then keep
    releasing "next time" holds once their model unloads (panel.py). recover() runs before
    sync_config() because sync rewrites the switcher file from the registry, and the marker
    tells recover which model was on test settings. A failing recover must never stop the
    helper: the Right-now strip still shows what is loaded, and the marker's model is noted
    as a leftover so the strip says it may still sit on test settings (the marker stays, so
    the next helper start tries again)."""
    panel = app.state.panel
    try:
        app.state.playground.recover()
    except Exception:  # noqa: BLE001
        logger.exception("Test-tab recovery failed at helper start; the marker is kept for the next start")
        try:
            marker = panel.read_test_marker()
            if marker:
                panel.note_test_leftover(str(marker.get("model") or ""), marker.get("preset"))
        except Exception:  # noqa: BLE001
            logger.exception("The test marker could not be read after the failed recovery")
    app.state.panel.sync_config()
    app.state.panel.start_hold_watcher()


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
    panel = app.state.panel
    start_panel(app)
    # Start the memory governor with the helper, not only from a page-driven Start: a helper
    # restart adopts a model server that is already serving (helper 1.3.0), and without this
    # the adopted server ran with no governor at all (seen live 2026-09-07 18:06: status stuck
    # at zeros after `systemctl --user restart freetoken-settings`). The loop idles while no
    # server is serving and picks the adopted one up on its next tick; a page Start re-reads
    # the cushions and replaces it.
    process_manager.start_governor()
    # Same reasoning for the crash watchdog: it must outlive page Starts and adopt a server
    # that is already serving, so it starts with the helper (2026-09-07: port 2020 sat dead
    # from 23:38 to 04:20 after a scheduler crash because nothing watched it).
    process_manager.start_watchdog()
    try:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
    finally:
        panel.stop_hold_watcher()
        process_manager.stop_watchdog()
        process_manager.stop_governor()
        process_manager.close()
    return 0


__all__ = ["main", "start_panel"]

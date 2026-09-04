"""FastAPI application for the loopback-only Windows settings helper."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .boot_parser import BootFile, BootParseError, BootValidationError
from .dials import DIALS, EXTENSION_DIALS, ENV_DIALS, dial_value_for_display, validate_settings
from .process_manager import LifecycleError, ProcessManager
from .profiles_manager import ProfileValidationError, ProfilesManager

HELPER_VERSION = "1.0.0"


class SettingsBody(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)


class ProfileBody(BaseModel):
    name: str
    description: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)


class ServerActionBody(BaseModel):
    force: bool = False


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_paths() -> dict[str, Path]:
    root = repository_root()
    return {
        "boot": Path(os.environ.get("FREETOKEN_SETTINGS_BOOT_FILE", root / "boot-2020.ps1")),
        "stop": root / "scripts" / "stop-qwen38-flash-next-windows.ps1",
        "log": root / "logs" / "server-2020.log",
        "profiles": root / "boot-profiles.json",
        "lock": root / "prompts" / "settings-page" / "gpu.lock",
        "static": Path(__file__).with_name("static") / "index.html",
    }


def _validation_response(errors: list[dict[str, str]]) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": errors})


def _boot_http_error(exc: BootParseError) -> HTTPException:
    return HTTPException(status_code=500, detail=f"BootParseError: {exc}")


def _active_profile(settings: dict[str, Any]) -> str:
    if (
        settings.get("KVDtype") == "bf16"
        and settings.get("MoECacheSize") == 4188
        and settings.get("KVPark", "off") == "off"
    ):
        return "A"
    if (
        settings.get("KVDtype") == "fp8"
        and settings.get("MoECacheSize") == 5332
        and settings.get("KVPark", "off") == "off"
    ):
        return "B"
    return "custom"


def _settings_payload(boot: BootFile) -> dict[str, Any]:
    settings = boot.load()
    primary = {
        dial.name: settings.get(dial.name, dial.default)
        for dial in DIALS
        if dial.name not in EXTENSION_DIALS
    }
    dials = []
    for dial in DIALS:
        value = settings.get(dial.name, dial.default)
        dials.append(dial.as_dict(dial_value_for_display(dial, value)))
    return {
        "bootFilePath": str(boot.path),
        "activeProfile": _active_profile(settings),
        "settings": primary,
        "dials": dials,
    }


def create_app(
    *,
    boot_file: str | os.PathLike[str] | None = None,
    process_manager: ProcessManager | None = None,
    profiles: ProfilesManager | None = None,
    log_path: str | os.PathLike[str] | None = None,
    static_path: str | os.PathLike[str] | None = None,
    version: str = HELPER_VERSION,
    wall_now=time.time,
) -> FastAPI:
    """Build an app with injectable file/process pieces so routes are testable without a GPU."""
    paths = default_paths()
    boot = BootFile(boot_file or paths["boot"])
    if process_manager is None:
        process_manager = ProcessManager(
            boot_file=boot.path,
            stop_script=paths["stop"],
            log_path=log_path or paths["log"],
            lock_path=paths["lock"],
            port=2020,
        )
    if profiles is None:
        profiles = ProfilesManager(profiles_path := paths["profiles"], boot_file=boot.path)
    elif profiles.boot_file is None:
        profiles.boot_file = boot.path
    log = Path(log_path or getattr(process_manager, "log_path", paths["log"]))
    static = Path(static_path or paths["static"])
    started = time.monotonic()
    app = FastAPI(title="FreeToken Settings Helper", version=version)
    app.state.boot_file = boot
    app.state.process_manager = process_manager
    app.state.profiles = profiles
    app.state.log_path = log
    app.state.started_monotonic = started

    @app.get("/")
    async def root():
        if static.is_file():
            return FileResponse(static, media_type="text/html")
        return PlainTextResponse("Settings page is not available yet\n", status_code=404)

    @app.get("/api/settings")
    async def get_settings():
        try:
            return _settings_payload(boot)
        except BootParseError as exc:
            raise _boot_http_error(exc) from exc

    @app.put("/api/settings")
    async def put_settings(body: SettingsBody):
        errors = validate_settings(body.settings)
        if errors:
            return _validation_response(errors)
        try:
            saved = boot.save(body.settings)
        except BootValidationError as exc:
            return _validation_response(exc.errors)
        except BootParseError as exc:
            raise _boot_http_error(exc) from exc
        return {
            "status": "saved",
            "backupPath": str(boot.backup_path),
            "settings": saved,
        }

    @app.get("/api/profiles")
    async def get_profiles():
        return {"profiles": profiles.list()}

    @app.post("/api/profiles", status_code=201)
    async def post_profile(body: ProfileBody):
        try:
            return profiles.create(body.name, body.description, body.settings)
        except ProfileValidationError as exc:
            return _validation_response(exc.errors)

    @app.delete("/api/profiles/{profile_id}")
    async def delete_profile(profile_id: str):
        return profiles.delete(profile_id)

    @app.post("/api/profiles/{profile_id}/apply")
    async def apply_profile(profile_id: str):
        try:
            return profiles.apply(profile_id, boot.path)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"profile {profile_id} not found")
        except BootValidationError as exc:
            return _validation_response(exc.errors)
        except BootParseError as exc:
            raise _boot_http_error(exc) from exc

    @app.get("/api/status")
    async def status():
        server = process_manager.server_status()
        geometry = server.get("geometry") or {}
        parking = server.get("parking") or {}
        return {
            "helper": {
                "status": "up",
                "uptimeS": max(0, int(time.monotonic() - started)),
                "version": version,
            },
            "server": {
                "reachable": bool(server.get("reachable")),
                "state": server.get("state", "unreachable"),
                "port": process_manager.port,
                "activeRequests": int(server.get("activeRequests", 0) or 0),
                "uptimeS": int(server.get("uptimeS", 0) or 0),
                "geometry": {
                    "numPages": int(geometry.get("numPages", 0) or 0),
                    "pageSize": int(geometry.get("pageSize", 0) or 0),
                    "moeCacheSize": int(geometry.get("moeCacheSize", 0) or 0),
                    "numMambaSlots": int(geometry.get("numMambaSlots", 0) or 0),
                    "gpuOwnedLayers": list(geometry.get("gpuOwnedLayers") or []),
                    "gpuOwnedReservedBytes": int(geometry.get("gpuOwnedReservedBytes", 0) or 0),
                },
                "parking": {
                    "mode": parking.get("mode", "off"),
                    "parkedCount": int(parking.get("parkedCount", 0) or 0),
                    "parkedBytes": int(parking.get("parkedBytes", 0) or 0),
                    "hits": int(parking.get("hits", 0) or 0),
                    "misses": int(parking.get("misses", 0) or 0),
                    "lastRestoreMs": float(parking.get("lastRestoreMs", 0.0) or 0.0),
                    "disabled": bool(parking.get("disabled", False)),
                    "lastError": parking.get("lastError"),
                },
            },
            "gpu": {
                "vramUsedMb": int(server.get("vramUsedMb", 0) or 0),
                "vramTotalMb": int(server.get("vramTotalMb", 0) or 0),
                "gpuUtilPercent": int(server.get("gpuUtilPercent", 0) or 0),
            },
            "currentJob": process_manager.current_job(),
        }

    @app.get("/api/logs")
    async def logs(limit: int = Query(default=100, ge=1, le=2000)):
        return {
            "logPath": str(log),
            "totalLines": process_manager.log_total_lines(),
            "lines": process_manager.tail_log(limit),
        }

    @app.post("/api/server/{action}", status_code=202)
    async def server_action(action: str, body: ServerActionBody | None = None):
        if action not in {"start", "stop", "restart"}:
            raise HTTPException(status_code=422, detail="action must be start, stop, or restart")
        try:
            job_id = process_manager.start(action)
        except LifecycleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        job = process_manager.job(job_id) or {}
        return {
            "jobId": job_id,
            "action": action,
            "status": job.get("stage", "stopping"),
            "message": f"Initiated {action} sequence for port {process_manager.port}",
        }

    @app.get("/api/server/jobs/{job_id}")
    async def server_job(job_id: str):
        job = process_manager.job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id} not found")
        return job

    return app


# A spelling used by a few small integrations.
build_app = create_app


__all__ = ["HELPER_VERSION", "build_app", "create_app", "default_paths", "repository_root"]

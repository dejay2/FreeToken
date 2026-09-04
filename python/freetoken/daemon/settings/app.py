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
from .browse import BROWSE_KINDS, list_directory
from .dials import (
    DIAL_BY_NAME,
    DIALS,
    EXTENSION_DIALS,
    ENV_DIALS,
    GROUP_INFO,
    adapt_dial,
    canonical_value,
    dial_value_for_display,
    validate_settings,
)
from .download import DownloadManager, create_download_router
from .model_info import ModelInfo, read_model
from .process_manager import LifecycleError, ProcessManager
from .profiles_manager import ProfileError, ProfileValidationError, ProfilesManager

HELPER_VERSION = "1.2.0"


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


def _models_dir_from_boot(boot: BootFile, supplied: str | os.PathLike[str] | None) -> Path:
    """Pick the download root beside the saved model, or a safe repository fallback."""
    if supplied is not None and str(supplied).strip():
        return Path(supplied)
    try:
        settings = boot.load()
    except Exception:
        settings = {}
    model_path = settings.get("ModelPath") if isinstance(settings, dict) else None
    if isinstance(model_path, str) and model_path.strip() and not model_path.lstrip().startswith(("$", "(")):
        folder = Path(os.path.expandvars(os.path.expanduser(model_path.strip())))
        if folder.name:
            return folder.parent
    return repository_root() / "models"


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


def _model_for(settings: dict[str, Any], override: str | None = None) -> ModelInfo:
    """The model the dials are shaped for: an explicit folder (the page previewing a folder
    it has not saved yet) or the saved ModelPath."""
    path = override if override is not None and override.strip() else settings.get("ModelPath", "")
    return read_model(path if isinstance(path, str) else "")


def _settings_payload(
    boot: BootFile, model_path: str | None = None, active_profile: str | None = None
) -> dict[str, Any]:
    settings = boot.load()
    model = _model_for(settings, model_path)
    primary = {
        dial.name: settings.get(dial.name, dial.default)
        for dial in DIALS
        if dial.name not in EXTENSION_DIALS
    }
    dials = []
    for dial in DIALS:
        value = settings.get(dial.name, dial.default)
        dials.append(dial.as_dict(dial_value_for_display(dial, value), model))
    groups = [
        {"name": name, "plain": info.get("plain", name), "info": info.get("info", "")}
        for name, info in GROUP_INFO.items()
    ]
    return {
        "bootFilePath": str(boot.path),
        "activeProfile": _active_profile(settings) if active_profile is None else active_profile,
        "settings": primary,
        "dials": dials,
        "groups": groups,
        "model": model.as_dict(),
    }


def create_app(
    *,
    boot_file: str | os.PathLike[str] | None = None,
    process_manager: ProcessManager | None = None,
    profiles: ProfilesManager | None = None,
    log_path: str | os.PathLike[str] | None = None,
    static_path: str | os.PathLike[str] | None = None,
    downloads_dir: str | os.PathLike[str] | None = None,
    models_dir: str | os.PathLike[str] | None = None,
    version: str = HELPER_VERSION,
    wall_now=time.time,
) -> FastAPI:
    """Build an app with injectable file/process pieces so routes are testable without a GPU."""
    paths = default_paths()
    default_boot = Path(boot_file or paths["boot"])
    if profiles is None:
        profiles = ProfilesManager(paths["profiles"], boot_file=default_boot)
    else:
        profiles.configure_boot_file(default_boot)
    active_boot = profiles.boot_file or default_boot
    boot = BootFile(active_boot)
    if process_manager is None:
        process_manager = ProcessManager(
            boot_file=boot.path,
            stop_script=paths["stop"],
            log_path=log_path or paths["log"],
            lock_path=paths["lock"],
            port=2020,
        )
    else:
        # The helper, the process controller and the profile store must all name the same file.
        process_manager.boot_file = boot.path
    log = Path(log_path or getattr(process_manager, "log_path", paths["log"]))
    static = Path(static_path or paths["static"])
    model_root = _models_dir_from_boot(
        boot,
        models_dir if models_dir is not None else downloads_dir,
    )
    download_root = Path(downloads_dir) if downloads_dir is not None and str(downloads_dir).strip() else model_root

    def card_memory_bytes() -> int:
        try:
            status = process_manager.server_status()
            nested = status.get("gpu") if isinstance(status, dict) else None
            total_mb = status.get("vramTotalMb", 0) if isinstance(status, dict) else 0
            if not total_mb and isinstance(nested, dict):
                total_mb = nested.get("vramTotalMb", 0)
            return max(0, int(total_mb or 0)) * 1024 * 1024
        except Exception:
            return 0

    download_manager = DownloadManager(
        model_root,
        download_root,
        card_memory=card_memory_bytes,
    )
    started = time.monotonic()
    app = FastAPI(title="FreeToken Settings Helper", version=version)
    app.state.boot_file = boot
    app.state.default_boot_file = default_boot
    app.state.process_manager = process_manager
    app.state.profiles = profiles
    app.state.log_path = log
    app.state.models_dir = model_root
    app.state.downloads_dir = download_root
    app.state.download_manager = download_manager
    app.state.started_monotonic = started
    app.include_router(create_download_router(models_dir=model_root, manager=download_manager))

    def set_active_boot(path: str | os.PathLike[str]) -> None:
        nonlocal boot
        boot = BootFile(path)
        profiles.boot_file = boot.path
        process_manager.boot_file = boot.path
        app.state.boot_file = boot

    @app.get("/")
    async def root():
        if static.is_file():
            return FileResponse(static, media_type="text/html")
        return PlainTextResponse("Settings page is not available yet\n", status_code=404)

    @app.get("/api/settings")
    async def get_settings(model: str | None = Query(default=None)):
        """``?model=<folder>`` shapes the dials for a folder the page is previewing but has
        not saved yet; without it the saved ModelPath is used."""
        try:
            return _settings_payload(boot, model, profiles.active_profile_id)
        except BootParseError as exc:
            raise _boot_http_error(exc) from exc

    @app.get("/api/model")
    async def get_model(path: str = Query(default="")):
        """Describe a model folder from its config.json alone (no weight files are opened)."""
        return read_model(path).as_dict()

    @app.put("/api/settings")
    async def put_settings(body: SettingsBody):
        try:
            saved_settings = boot.load()
        except BootParseError as exc:
            raise _boot_http_error(exc) from exc
        model_path = body.settings.get("ModelPath")
        model = _model_for(saved_settings, model_path if isinstance(model_path, str) else None)
        errors = validate_settings(body.settings, model)
        if not errors and isinstance(model_path, str) and model_path != saved_settings.get("ModelPath", ""):
            # A new model must also fit what the file already holds: a 32k model saved next to
            # a 262,144-token chat would reserve memory for a length it cannot produce.
            merged = {**saved_settings, **body.settings}
            errors = validate_settings(merged, model)
        if errors:
            return _validation_response(errors)
        # Text-stored counts take the model's spelling ("auto:3" on the measured model, "3"
        # elsewhere) whether the caller sent a number or text.
        changes = dict(body.settings)
        for name, value in body.settings.items():
            dial = DIAL_BY_NAME.get(name)
            if dial is not None and dial.stored_as:
                changes[name] = canonical_value(dial, value, adapt_dial(dial, model).get("storedAs"))
        try:
            saved = boot.save(changes)
            profiles.sync_active(saved)
        except BootValidationError as exc:
            return _validation_response(exc.errors)
        except BootParseError as exc:
            raise _boot_http_error(exc) from exc
        return {
            "status": "saved",
            "backupPath": str(boot.backup_path),
            "settings": saved,
        }

    @app.get("/api/browse")
    async def browse(path: str = Query(default=""), kind: str = Query(default="folder")):
        if kind not in BROWSE_KINDS:
            raise HTTPException(status_code=422, detail=f"kind must be one of {', '.join(BROWSE_KINDS)}")
        return list_directory(path, kind)

    @app.get("/api/profiles")
    async def get_profiles():
        return {"profiles": profiles.list()}

    @app.post("/api/profiles", status_code=201)
    async def post_profile(body: ProfileBody):
        try:
            return profiles.create(body.name, body.description, body.settings)
        except ProfileValidationError as exc:
            return _validation_response(exc.errors)

    @app.put("/api/profiles/{profile_id}")
    async def put_profile(profile_id: str, body: ProfileBody):
        try:
            return profiles.update(
                profile_id,
                name=body.name,
                description=body.description,
                settings=body.settings,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"profile {profile_id} not found")
        except ProfileValidationError as exc:
            return _validation_response(exc.errors)
        except ProfileError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/profiles/{profile_id}")
    async def delete_profile(profile_id: str):
        try:
            result = profiles.delete(profile_id)
        except ProfileError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if result.get("bootFilePath") and result.get("activeProfileId"):
            set_active_boot(result["bootFilePath"])
        return result

    @app.post("/api/profiles/{profile_id}/activate")
    async def activate_profile(profile_id: str):
        try:
            result = profiles.activate(profile_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"profile {profile_id} not found")
        except ProfileError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        set_active_boot(result["bootFilePath"])
        return result

    @app.post("/api/profiles/{profile_id}/apply")
    async def apply_profile(profile_id: str):
        # A profile is checked against the model it will run with (the profile's own
        # ModelPath if it names one, else the saved one) before the file is rewritten.
        profile = next((item for item in profiles.list() if item.get("id") == profile_id), None)
        if profile is not None:
            try:
                saved_settings = boot.load()
            except BootParseError as exc:
                raise _boot_http_error(exc) from exc
            merged = {**saved_settings, **(profile.get("settings") or {})}
            errors = validate_settings(merged, _model_for(merged))
            if errors:
                return _validation_response(errors)
        try:
            result = profiles.apply(profile_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"profile {profile_id} not found")
        except ProfileError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BootValidationError as exc:
            return _validation_response(exc.errors)
        except BootParseError as exc:
            raise _boot_http_error(exc) from exc
        if result.get("activated") and result.get("bootFilePath"):
            set_active_boot(result["bootFilePath"])
        else:
            profiles.sync_active(result.get("settings") or {})
        return result

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

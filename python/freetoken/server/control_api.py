"""Read-only control-plane endpoints consumed by the desktop app: /health (lifecycle),
/ready (status-code readiness for process managers), /v1/stats (runtime metrics, Task 6), /v1/requests (request log ring, Task 5).

All handlers read a shared FrontendManager snapshot via ``get_state``; nothing here touches
the scheduler or blocks. Registered on the app alongside the OpenAI/Anthropic/Responses routes.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def build_health(state: Any, version: str) -> dict:
    """Full-lifecycle health doc: loading -> ok -> error."""
    instance_id = getattr(state, "instance_id", None)
    # Applying the maintenance deadline here means a /health poller alone is enough to move
    # a stuck "rebuilding" to error; it used to answer ok for as long as the gate stayed shut.
    check = getattr(state, "check_maintenance", None)
    maintenance = check() if callable(check) else None
    check_inference = getattr(state, "check_inference", None)
    inference = check_inference() if callable(check_inference) else None
    fatal = getattr(state, "fatal_error", None)
    if fatal:
        return {"status": "error", "message": fatal, "instance_id": instance_id, "inference": inference}

    mstate = getattr(state, "maintenance_state", "serving")
    config = getattr(state, "config", None)
    model = getattr(config, "served_model_name", None)

    if mstate == "loading":
        lp = getattr(state, "load_progress", None)
        return {
            "status": "loading",
            "phase": lp.phase if lp is not None else "other",
            "progress": {
                "done_bytes": lp.done_bytes if lp is not None else 0,
                "total_bytes": lp.total_bytes if lp is not None else 0,
            },
            "model": model,
            "instance_id": instance_id,
        }

    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    doc = {
        "status": "ok",
        "model": model,
        "instance_id": instance_id,
        "uptime_s": uptime_s,
        "maintenance": mstate,
        "version": version,
        "inference": inference,
    }
    if isinstance(maintenance, dict) and maintenance.get("age_s") is not None:
        doc["maintenance_age_s"] = maintenance["age_s"]
        doc["maintenance_phase"] = maintenance.get("phase")
        doc["maintenance_progress_idle_s"] = maintenance.get("progress_idle_s")
    return doc


# Maintenance states in which a chat request is answered: a runtime cache rebuild is waited
# out by the chat routes' gate (a few seconds of slowness), so it still counts as ready.
_READY_MAINTENANCE = ("serving", "rebuilding")


def is_ready(health: dict) -> bool:
    """True when chat routes will answer instead of returning 503."""
    return health.get("status") == "ok" and health.get("maintenance") in _READY_MAINTENANCE


def register_control_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict] | None = None,
) -> None:
    @app.get("/health")
    async def health():
        return build_health(get_state(), app.version)

    # /health must stay 200 while loading (the desktop app renders its progress body), so
    # process managers that only look at the status code -- llama-swap's checkEndpoint --
    # would forward the first chat into a 503. /ready answers 503 until the model serves.
    @app.get("/ready")
    async def ready():
        doc = build_health(get_state(), app.version)
        if is_ready(doc):
            return {"status": "ready", "model": doc.get("model")}
        return JSONResponse(
            {"status": "not_ready", "health": doc.get("status"), "maintenance": doc.get("maintenance")},
            status_code=503,
        )

    from . import request_ring

    @app.get("/v1/requests")
    async def list_requests(since: int = 0, limit: int = 100):
        limit = max(1, min(limit, 512))
        entries, next_cursor = request_ring.requests_since(since, limit)
        return {"entries": entries, "next_cursor": next_cursor}

    from .stats import build_stats

    @app.get("/v1/stats")
    async def stats():
        doc = build_stats(
            get_state(), request_ring.requests_p95_ms(), request_ring.requests_ttft_mean_ms()
        )
        # Surface the model's recommended sampling (from its generation_config.json / GGUF
        # metadata) so clients can seed their sampling controls per-model instead of guessing.
        if get_model_sampling is not None:
            doc["model"]["sampling"] = get_model_sampling() or {}
        return doc

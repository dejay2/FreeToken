"""Small, torch-free bridge from the settings page to its local serving process.

Only named-cache controls, cache status, models and a bounded generation test are
exposed. Keep rendering and cache validation authoritative in the serving engine.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from fastapi import APIRouter, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictBool
from starlette.concurrency import run_in_threadpool

NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
MAX_BODY_BYTES = 16 * 1024 * 1024


class CacheRequest(BaseModel):
    format: Literal["openai", "anthropic"]
    request: dict[str, Any]


class Registration(CacheRequest):
    name: str = Field(pattern=NAME_PATTERN)
    prefix_tokens: int | None = Field(default=None, ge=0)
    ttl_seconds: float = Field(default=300, ge=0, le=86400, allow_inf_nan=False)


class Retention(BaseModel):
    max_retained_bytes: int = Field(ge=0)


class RecentCapture(BaseModel):
    enabled: StrictBool


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _error(code: int, message: str) -> JSONResponse:
    return JSONResponse({"status": "failed", "result": {}, "error": message}, status_code=code)


def _forward(port: int, method: str, path: str, payload: dict | None, timeout: float) -> JSONResponse:
    data = json.dumps(payload).encode() if payload is not None else None
    if data is not None and len(data) > MAX_BODY_BYTES:
        return _error(413, "The request exceeds the 16 MiB prompt-cache limit.")
    request = Request(
        f"http://127.0.0.1:{int(port)}{path}", data=data, method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    # Never use ambient HTTP proxies or follow a redirect away from the local API.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as exc:
            response = exc  # Preserve structured engine errors and their HTTP status.
        with response:
            code = response.code
            raw = response.read(MAX_BODY_BYTES + 1)
        if 300 <= code < 400 or len(raw) > MAX_BODY_BYTES:
            return _error(502, "The local server returned an unexpected response.")
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("expected an object")
        except (ValueError, UnicodeDecodeError):
            return _error(502, "The local server did not return a JSON response. Check its version and logs.")
        return JSONResponse(body, status_code=code, headers={"Cache-Control": "no-store"})
    except (TimeoutError, socket.timeout):
        return _error(504, "The local server timed out. Work may still be running; refresh before trying again.")
    except (URLError, OSError):
        return _error(503, "Cannot reach the local model server. Start it and wait until it is ready, then refresh.")


def create_prompt_cache_router(server_port: Callable[[], int]) -> APIRouter:
    router = APIRouter(prefix="/api/prompt-cache", tags=["prompt cache"])

    async def forward(method: str, path: str, payload: dict | None = None, timeout: float = 40):
        return await run_in_threadpool(_forward, server_port(), method, path, payload, timeout)

    @router.get("/prefixes")
    async def list_prefixes():
        return await forward("GET", "/v1/cache/prefixes")

    @router.get("/models")
    async def models():
        return await forward("GET", "/v1/models", timeout=5)

    @router.get("/recent")
    async def recent():
        return await forward("GET", "/v1/cache/recent", timeout=5)

    @router.delete("/recent")
    async def clear_recent():
        return await forward("DELETE", "/v1/cache/recent", timeout=5)

    @router.put("/recent/settings")
    async def configure_recent(body: RecentCapture):
        return await forward("PUT", "/v1/cache/recent/settings", body.model_dump(), timeout=5)

    @router.get("/recent/{id}")
    async def recent_prompt(id: str = Path(pattern=r"^[a-f0-9]{32}$")):
        return await forward("GET", f"/v1/cache/recent/{id}", timeout=5)

    @router.get("/status")
    async def cache_status():
        return await forward("GET", "/v1/cache/status", timeout=5)

    @router.post("/prefixes")
    async def register(body: Registration):
        return await forward("POST", "/v1/cache/prefixes", body.model_dump(exclude_none=True))

    @router.put("/settings")
    async def retention(body: Retention):
        return await forward("PUT", "/v1/cache/prefixes/settings", body.model_dump())

    @router.post("/prefixes/{name}/warm")
    async def warm(name: str = Path(pattern=NAME_PATTERN)):
        return await forward("POST", f"/v1/cache/prefixes/{name}/warm")

    @router.delete("/prefixes/{name}")
    async def delete(name: str = Path(pattern=NAME_PATTERN)):
        return await forward("DELETE", f"/v1/cache/prefixes/{name}")

    @router.post("/test")
    async def test(body: CacheRequest):
        request = dict(body.request)
        # Keep template-affecting inputs intact. Only bound the generated output.
        request.update(stream=False, max_tokens=64, n=1)
        request.pop("stream_options", None)
        if "max_completion_tokens" in request:
            request["max_completion_tokens"] = 64
        path = "/v1/chat/completions" if body.format == "openai" else "/v1/messages"
        return await forward("POST", path, request, timeout=180)

    return router

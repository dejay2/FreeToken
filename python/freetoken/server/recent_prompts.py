"""Bounded, process-local request snapshots for the prompt-cache picker.

Snapshots preserve the incoming JSON, including tools/template options, independently
of request validation and normalization. Nothing is written to disk or tokenized here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictBool

logger = logging.getLogger(__name__)
FORMATS = {"/v1/chat/completions": "openai", "/v1/messages": "anthropic"}


def _has_image(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(part, dict) and (
            part.get("type") in {"image", "image_url"} or "image" in part or "image_url" in part
            or _has_image(part.get("content"))
        ) for part in content
    )


def _text(content) -> str:
    if isinstance(content, str):
        return content[:240]
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or _text(part.get("content"))
                if isinstance(text, str) and text:
                    return text[:240]
    return ""


@dataclass(frozen=True)
class _Entry:
    metadata: dict
    raw: bytes
    expires_at: float


class RecentPrompts:
    def __init__(self, *, capacity: int = 50, max_bytes: int = 16 * 1024 * 1024,
                 max_request_bytes: int = 4 * 1024 * 1024, ttl_seconds: float = 3600,
                 clock: Callable[[], float] = time.monotonic):
        self.capacity = capacity
        self.max_bytes = max_bytes
        self.max_request_bytes = min(max_request_bytes, max_bytes)
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.enabled = True
        self.epoch = 0
        self.skipped_count = 0
        self._bytes = 0
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def _evict(self):
        _, entry = self._entries.popitem(last=False)
        self._bytes -= len(entry.raw)

    def _expire(self):
        now = self.clock()
        while self._entries and next(iter(self._entries.values())).expires_at <= now:
            self._evict()

    def add(self, format: str, raw: bytes, *, epoch: int) -> str | None:
        if not self.enabled or epoch != self.epoch or format not in FORMATS.values():
            return None
        self._expire()
        if len(raw) > self.max_request_bytes:
            self.skipped_count += 1
            return None
        body = json.loads(raw)
        if not isinstance(body, dict) or body.get("cache_private") or body.get("private"):
            return None
        messages = body.get("messages")
        if not isinstance(messages, list) or _has_image(body.get("system")) or any(
            isinstance(message, dict) and _has_image(message.get("content")) for message in messages
        ):
            return None
        preview = next((text for message in reversed(messages) if isinstance(message, dict)
                        and message.get("role") == "user" and (text := _text(message.get("content")))), "")
        if not preview:
            preview = _text(body.get("system")) or next((text for message in messages
                if isinstance(message, dict) and (text := _text(message.get("content")))), "")
        id = uuid.uuid4().hex
        metadata = {"id": id, "format": format, "model": str(body.get("model") or "")[:160],
                    "received_at": datetime.now(timezone.utc).isoformat(),
                    "message_count": len(messages), "preview": preview or "Text request",
                    "size_bytes": len(raw)}
        while self._entries and (len(self._entries) >= self.capacity or self._bytes + len(raw) > self.max_bytes):
            self._evict()
        self._entries[id] = _Entry(metadata, raw, self.clock() + self.ttl_seconds)
        self._bytes += len(raw)
        return id

    def list(self) -> dict:
        self._expire()
        return {"prompts": [dict(entry.metadata) for entry in reversed(self._entries.values())],
                "enabled": self.enabled, "capacity": self.capacity, "stored_bytes": self._bytes,
                "max_bytes": self.max_bytes, "max_request_bytes": self.max_request_bytes,
                "ttl_seconds": self.ttl_seconds, "skipped_count": self.skipped_count}

    def get(self, id: str) -> dict | None:
        self._expire()
        entry = self._entries.get(id)
        return entry.metadata | {"request": json.loads(entry.raw)} if entry else None

    def clear(self):
        self.epoch += 1
        self._entries.clear()
        self._bytes = 0
        self.skipped_count = 0

    def set_enabled(self, enabled: bool):
        if enabled != self.enabled:
            self.epoch += 1
            self.enabled = enabled


class CaptureSettings(BaseModel):
    enabled: StrictBool


def register_recent_prompt_routes(app: FastAPI) -> RecentPrompts:
    store = RecentPrompts()
    app.state.recent_prompts = store
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with previous_lifespan(application) as state:
            async def expire_idle():
                while True:
                    await asyncio.sleep(60)
                    store._expire()
            cleanup = asyncio.create_task(expire_idle())
            try:
                yield state
            finally:
                cleanup.cancel()
                with suppress(asyncio.CancelledError):
                    await cleanup
                store.clear()

    app.router.lifespan_context = lifespan

    @app.middleware("http")
    async def capture(request: Request, call_next):
        format = FORMATS.get(request.url.path)
        if not format or request.method != "POST" or not store.enabled:
            return await call_next(request)
        epoch = store.epoch
        # Starlette replays the cached body to the downstream handler unchanged.
        raw = await request.body()
        response = await call_next(request)
        if 200 <= response.status_code < 300:
            try:
                store.add(format, raw, epoch=epoch)
            except Exception:  # Request history must never break generation.
                logger.warning("Could not retain a recent prompt", exc_info=False)
        return response

    def reply(result):
        return JSONResponse({"status": "ok", "result": result, "error": None},
                            headers={"Cache-Control": "no-store"})

    @app.get("/v1/cache/recent")
    async def recent():
        return reply(store.list())

    @app.delete("/v1/cache/recent")
    async def clear():
        store.clear()
        return reply(store.list())

    @app.put("/v1/cache/recent/settings")
    async def configure(settings: CaptureSettings):
        store.set_enabled(settings.enabled)
        return reply(store.list())

    @app.get("/v1/cache/recent/{id}")
    async def selected(id: str):
        result = store.get(id)
        if result is None:
            return JSONResponse({"status": "not_found", "result": {},
                                 "error": "This request has expired or was cleared. Refresh the recent list."}, status_code=404)
        return reply(result)

    return store

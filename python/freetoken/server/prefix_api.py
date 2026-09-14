"""Named prefix-preset management API.

The frontend converts an existing OpenAI or Anthropic request into the same
``GenSpec`` used by generation. The tokenizer worker performs the actual render
and tokenization; scheduler-owned code remains authoritative for registration,
alignment, support checks and retention.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from freetoken.message import PrefixCacheMsg
from pydantic import BaseModel, Field, ValidationError

from .anthropic_api import convert_anthropic_to_genspec
from .anthropic_models import AnthropicMessagesRequest
from .api_models import ChatCompletionRequest
from .openai_api import _maintenance_gate, chat_request_to_genspec


PREFIX_CONTROL_TIMEOUT_S = 30.0


class PrefixRegistrationRequest(BaseModel):
    name: str
    format: Literal["openai", "anthropic"]
    request: dict[str, Any]
    prefix_tokens: int | None = Field(default=None, ge=0)
    ttl_seconds: float = Field(default=300.0, ge=0)


class PrefixSettingsRequest(BaseModel):
    max_retained_bytes: int = Field(ge=0)


def _request_has_images(request: dict[str, Any]) -> bool:
    """Inspect only prompt content, never tool schemas that may mention images."""

    def content_has_image(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        return any(
            isinstance(part, dict)
            and (
                part.get("type") in {"image", "image_url"}
                or "image" in part
                or "image_url" in part
                or content_has_image(part.get("content"))
            )
            for part in content
        )

    if content_has_image(request.get("system")):
        return True
    messages = request.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(item, dict) and content_has_image(item.get("content"))
        for item in messages
    )


def _registration_spec(
    req: PrefixRegistrationRequest,
    state: Any,
    model_sampling: dict[str, Any],
):
    raw = req.request
    if raw.get("cache_private") or raw.get("private"):
        raise ValueError("private requests cannot be registered as shared prefixes")
    if _request_has_images(raw):
        raise ValueError("image inputs cannot be registered as shared prefixes")
    try:
        if req.format == "openai":
            parsed = ChatCompletionRequest.model_validate(raw)
            return chat_request_to_genspec(parsed, model_sampling)
        parsed = AnthropicMessagesRequest.model_validate(raw)
        return convert_anthropic_to_genspec(
            parsed,
            model_sampling,
            reasoning_parser=getattr(state.config, "reasoning_parser", None),
        )
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def _response_status(status: str) -> int:
    return {
        "invalid": 400,
        "not_found": 404,
        "unsupported": 409,
        "busy": 409,
        "failed": 500,
    }.get(status, 200)


def _response(result: dict[str, Any], *, success_status: int | None = None) -> JSONResponse:
    body = {
        "status": result.get("status", "failed"),
        "result": result.get("result") or {},
        "error": result.get("error"),
    }
    status_code = _response_status(body["status"])
    if success_status is not None and status_code == 200:
        status_code = success_status
    return JSONResponse(body, status_code=status_code)


async def _dispatch(state: Any, msg: PrefixCacheMsg, *, success_status: int | None = None):
    try:
        result = await state.dispatch_prefix(msg, timeout=PREFIX_CONTROL_TIMEOUT_S)
    except asyncio.TimeoutError:
        return JSONResponse(
            {
                "status": "failed",
                "result": {},
                "error": "prefix control request timed out",
            },
            status_code=504,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 -- queue/backend failure is an HTTP failure
        return JSONResponse(
            {
                "status": "failed",
                "result": {},
                "error": f"failed to dispatch prefix control request: {exc}",
            },
            status_code=503,
        )
    return _response(result, success_status=success_status)


def register_prefix_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]],
) -> None:
    async def ready_state():
        state = get_state()
        gate = await _maintenance_gate(state)
        return state, gate

    @app.post("/v1/cache/prefixes")
    async def register_prefix(req: PrefixRegistrationRequest):
        state, gate = await ready_state()
        if gate is not None:
            return gate
        try:
            spec = _registration_spec(req, state, get_model_sampling())
        except ValueError as exc:
            return _response({"status": "invalid", "result": {}, "error": str(exc)})
        return await _dispatch(
            state,
            PrefixCacheMsg(
                request_id=str(uuid.uuid4()),
                action="register",
                name=req.name,
                text=spec.messages,
                tools=spec.template_tools,
                chat_template_kwargs=spec.chat_template_kwargs,
                preserve_system_order=spec.preserve_system_order,
                prefix_tokens=req.prefix_tokens,
                ttl_seconds=req.ttl_seconds,
            ),
        )

    @app.get("/v1/cache/prefixes")
    async def list_prefixes():
        state, gate = await ready_state()
        if gate is not None:
            return gate
        return await _dispatch(
            state, PrefixCacheMsg(request_id=str(uuid.uuid4()), action="list")
        )

    @app.put("/v1/cache/prefixes/settings")
    async def configure_prefixes(req: PrefixSettingsRequest):
        state, gate = await ready_state()
        if gate is not None:
            return gate
        return await _dispatch(
            state,
            PrefixCacheMsg(
                request_id=str(uuid.uuid4()),
                action="configure",
                max_retained_bytes=req.max_retained_bytes,
            ),
        )

    @app.get("/v1/cache/prefixes/{name}")
    async def get_prefix(name: str):
        state, gate = await ready_state()
        if gate is not None:
            return gate
        return await _dispatch(
            state,
            PrefixCacheMsg(request_id=str(uuid.uuid4()), action="get", name=name),
        )

    @app.post("/v1/cache/prefixes/{name}/warm")
    async def warm_prefix(name: str):
        state, gate = await ready_state()
        if gate is not None:
            return gate
        return await _dispatch(
            state,
            PrefixCacheMsg(request_id=str(uuid.uuid4()), action="warm", name=name),
            success_status=202,
        )

    @app.delete("/v1/cache/prefixes/{name}")
    async def delete_prefix(name: str):
        state, gate = await ready_state()
        if gate is not None:
            return gate
        return await _dispatch(
            state,
            PrefixCacheMsg(request_id=str(uuid.uuid4()), action="delete", name=name),
        )

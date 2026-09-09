from __future__ import annotations

import base64
import io
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any, List
from urllib.error import HTTPError, URLError
from urllib.parse import unquote_to_bytes, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, url2pathname

import torch
from freetoken.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    CacheParkStatusMsg,
    CacheParkStatusReply,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
    CacheResidencyBackendMsg,
    CacheResidencyMsg,
    CacheResidencyReply,
    CacheResidencyResultMsg,
    CacheStepBackendMsg,
    CacheStepMsg,
    CacheStepReply,
    CacheStepResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    PromptAdmittedMsg,
    RoutingStatsBackendMsg,
    RoutingStatsMsg,
    RoutingStatsReply,
    RoutingStatsResultMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from freetoken.utils import (
    ZmqPullQueue,
    ZmqPushQueue,
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
)


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


def _prompt_admitted_reply(msg: PromptAdmittedMsg) -> UserReply:
    """Translate the scheduler's admission signal onto the existing frontend usage path."""
    return UserReply(
        uid=msg.uid,
        incremental_output="",
        finished=False,
        prompt_tokens_delta=msg.prompt_tokens,
        cached_tokens=msg.cached_tokens,
    )


def _sampled_reply(msg: DetokenizeMsg, incremental_output: str) -> UserReply:
    """Frontend reply for one step's sampled run. Usage bills every token in the run."""
    return UserReply(
        uid=msg.uid,
        incremental_output=incremental_output,
        finished=msg.finished,
        finish_reason=msg.finish_reason,
        matched_stop=msg.matched_stop,
        completion_tokens_delta=len(msg.next_tokens),
        kv_used_pages=msg.kv_used_pages,
        kv_total_pages=msg.kv_total_pages,
        mamba_used_slots=msg.mamba_used_slots,
        mamba_total_slots=msg.mamba_total_slots,
        swa_used_tokens=msg.swa_used_tokens,
        swa_total_tokens=msg.swa_total_tokens,
        gpu_mem_bytes=msg.gpu_mem_bytes,
    )


def _error_reply(msg: ErrorReplyMsg) -> UserReply:
    return UserReply(
        uid=msg.uid, incremental_output="", finished=True, error=msg.error, error_code=msg.code,
    )


def _put_user_replies(send_frontend: Any, replies: List[UserReply]) -> None:
    if replies:
        send_frontend.put(
            replies[0] if len(replies) == 1 else BatchFrontendMsg(data=replies)
        )


def _send_generation_replies(
    send_frontend: Any,
    admitted: List[UserReply],
    sampled: List[UserReply],
    terminal_errors: List[UserReply],
) -> None:
    """Preserve the accounting barrier within one tokenizer queue drain.

    Scheduler abort acknowledgements are terminal: every already-sampled DetokenizeMsg
    drained alongside them must reach FrontendManager first. Admission messages remain
    first so per-request usage precedes that request's sampled completion.
    """
    _put_user_replies(send_frontend, admitted)
    _put_user_replies(send_frontend, sampled)
    _put_user_replies(send_frontend, terminal_errors)


def _tokenize_requests(
    tokenize_manager: Any,
    multimodal_processor: Any,
    messages: List[TokenizeMsg],
    logger: Any,
) -> tuple[
    List[TokenizeMsg],
    List[torch.Tensor],
    List[dict[str, torch.Tensor] | None],
    List[UserReply],
]:
    """Tokenize independently, returning backend work plus terminal frontend errors.

    Successful tokenization deliberately emits no prompt-token reply: accounting starts
    only when the scheduler later confirms first-prefill admission.
    """
    ok_msgs: List[TokenizeMsg] = []
    ok_tensors: List[torch.Tensor] = []
    ok_multimodal: List[dict[str, torch.Tensor] | None] = []
    errors: List[UserReply] = []
    for msg in messages:
        try:
            tokens, multimodal = multimodal_processor.encode(msg, tokenize_manager)
        except Exception as exc:  # noqa: BLE001 — isolate, never crash the worker
            logger.warning(f"tokenization failed for request {msg.uid}: {exc!r}")
            errors.append(
                UserReply(
                    uid=msg.uid,
                    incremental_output="",
                    finished=True,
                    error=f"could not encode request: {exc}",
                )
            )
            continue
        # A zero-token prompt would trip the scheduler's input_len > 0 invariant and
        # crash the worker; reject it here as a terminal error instead.
        if tokens.numel() == 0:
            errors.append(
                UserReply(
                    uid=msg.uid,
                    incremental_output="",
                    finished=True,
                    error="prompt must contain at least one token",
                )
            )
            continue
        ok_msgs.append(msg)
        ok_tensors.append(tokens)
        ok_multimodal.append(multimodal)
    return ok_msgs, ok_tensors, ok_multimodal, errors


_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_IMAGE_TIMEOUT_SECONDS = 30
_MAX_IMAGE_REDIRECTS = 5
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class _BoundedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, max_redirects: int):
        super().__init__()
        self.max_redirects = max_redirects

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirects = int(getattr(req, "_freetoken_redirects", 0)) + 1
        if redirects > self.max_redirects:
            raise HTTPError(
                req.full_url,
                code,
                f"image redirect limit exceeded ({self.max_redirects})",
                headers,
                fp,
            )
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected._freetoken_redirects = redirects
        return redirected


def _image_source(part: dict[str, Any]) -> str:
    value = part.get("image_url", part.get("image"))
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str) or not value:
        raise ValueError("picture content part needs a non-empty source")
    return value


def _message_image_sources(text: str | List[dict[str, Any]]) -> list[str]:
    if not isinstance(text, list):
        return []
    sources: list[str] = []
    for message in text:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"image", "image_url"} or "image" in part or "image_url" in part:
                sources.append(_image_source(part))
    return sources


def _read_limited(stream, *, expected_size: int | None = None) -> bytes:
    if expected_size is not None and expected_size > _MAX_IMAGE_BYTES:
        raise ValueError("picture exceeds the 64 MiB input limit")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(1 << 20, _MAX_IMAGE_BYTES + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_IMAGE_BYTES:
            raise ValueError("picture exceeds the 64 MiB input limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _file_url_path(source: str) -> Path:
    parsed = urlsplit(source)
    path = url2pathname(parsed.path)
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        path = f"//{parsed.netloc}{path}"
    return Path(path)


def _read_image_source(source: str) -> bytes:
    """Read one approved local-only picture source under strict byte/network bounds."""
    if not source:
        raise ValueError("picture source must not be empty")
    if source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if not separator:
            raise ValueError("invalid picture data URL")
        try:
            data = (
                base64.b64decode(payload, validate=True)
                if ";base64" in header.lower()
                else unquote_to_bytes(payload)
            )
        except Exception as exc:
            raise ValueError("invalid picture data URL") from exc
        if len(data) > _MAX_IMAGE_BYTES:
            raise ValueError("picture exceeds the 64 MiB input limit")
        return data

    parsed = urlsplit(source)
    if parsed.scheme.lower() in {"http", "https"}:
        request = Request(source, headers={"User-Agent": "FreeToken/vision"})
        opener = build_opener(_BoundedRedirectHandler(_MAX_IMAGE_REDIRECTS))
        try:
            with opener.open(request, timeout=_IMAGE_TIMEOUT_SECONDS) as response:  # noqa: S310
                length = response.headers.get("Content-Length")
                expected = int(length) if length and length.isdigit() else None
                return _read_limited(response, expected_size=expected)
        except HTTPError as exc:
            if "redirect limit" in str(exc):
                raise ValueError(str(exc)) from exc
            raise ValueError(f"could not download picture: HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ValueError(f"could not download picture: {exc}") from exc

    if parsed.scheme.lower() == "file":
        path = _file_url_path(source)
    elif _WINDOWS_DRIVE_PATH.match(source) or source.startswith(("\\\\", "//")):
        path = Path(source)
    elif not parsed.scheme:
        path = Path(source)
    else:
        raise ValueError("picture source must use data, http, https, file, or a local path")

    try:
        size = path.stat().st_size
        if size > _MAX_IMAGE_BYTES:
            raise ValueError("picture exceeds the 64 MiB input limit")
        with path.open("rb") as stream:
            return _read_limited(stream, expected_size=size)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"could not read picture file {path}: {exc}") from exc


def _load_rgb_image(data: bytes):
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.seek(0)
            image = opened.convert("RGB")
            image.load()
            return image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"invalid picture: {exc}") from exc


class _MultimodalProcessor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.processor = None

    def encode(
        self, msg: TokenizeMsg, tokenize_manager: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        sources = _message_image_sources(msg.text)
        if not sources:
            return tokenize_manager.tokenize([msg])[0], None

        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_path)
        prompt = tokenize_manager.render_prompt(msg)
        images = []
        try:
            images = [_load_rgb_image(_read_image_source(source)) for source in sources]
            encoded = self.processor(text=[prompt], images=images, return_tensors="pt")
        finally:
            for image in images:
                image.close()

        required = {"input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids"}
        missing = sorted(required.difference(encoded))
        if missing:
            raise ValueError(f"model picture processor did not return: {', '.join(missing)}")
        input_ids = encoded["input_ids"]
        pixels = encoded["pixel_values"]
        grid = encoded["image_grid_thw"]
        token_types = encoded["mm_token_type_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"picture processor input_ids must be [1,tokens], got {input_ids.shape}")
        if pixels.ndim != 2:
            raise ValueError(f"picture processor pixel_values must be [patches,width], got {pixels.shape}")
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError(f"picture processor image_grid_thw must be [pictures,3], got {grid.shape}")
        if token_types.ndim == 2 and token_types.shape[0] == 1:
            token_types = token_types[0]
        if token_types.ndim != 1 or token_types.numel() != input_ids.shape[1]:
            raise ValueError("picture processor token markers must match input_ids")
        return input_ids[0].to(device="cpu", dtype=torch.int32), {
            "pixel_values": pixels.to(device="cpu", dtype=torch.bfloat16),
            "image_grid_thw": grid.to(device="cpu", dtype=torch.int64),
            "mm_token_type_ids": token_types.to(device="cpu", dtype=torch.int32),
        }


# Every control type the worker forwards or absorbs; a message outside this tuple (and outside
# the tokenize/detokenize/abort sets) trips the accounting assert in the loop below, which is
# how a missing passthrough shows up: the worker exits instead of silently dropping the request.
_CONTROL_MSG_TYPES = (
    CacheParkStatusMsg,
    CacheRebuildMsg,
    CacheRebuildResultMsg,
    CacheResidencyMsg,
    CacheResidencyResultMsg,
    CacheStepMsg,
    CacheStepResultMsg,
    ErrorReplyMsg,
    PromptAdmittedMsg,
    RoutingStatsMsg,
    RoutingStatsResultMsg,
)


def _forward_control_msg(m, send_backend, send_frontend) -> bool:
    """Forward one control message api -> scheduler or scheduler -> api. Returns True if forwarded.

    Field by field, no ``**vars``: the tokenizer and backend/frontend shapes are deliberately
    separate dataclasses (see freetoken/message), so a new field must be threaded here.
    """
    if isinstance(m, CacheParkStatusMsg):
        send_frontend.put(CacheParkStatusReply(status=m.status))
    elif isinstance(m, CacheRebuildMsg):
        send_backend.put(
            CacheRebuildBackendMsg(
                request_id=m.request_id,
                moe_cache_size=m.moe_cache_size,
                num_pages=m.num_pages,
                num_mamba_slots=m.num_mamba_slots,
                num_swa_pages=m.num_swa_pages,
                mode=m.mode,
                layer_moves=m.layer_moves,
            )
        )
    elif isinstance(m, CacheStepMsg):
        send_backend.put(
            CacheStepBackendMsg(
                request_id=m.request_id,
                axis=m.axis,
                direction=m.direction,
                ram_tight=m.ram_tight,
            )
        )
    elif isinstance(m, CacheResidencyMsg):
        send_backend.put(CacheResidencyBackendMsg(request_id=m.request_id))
    elif isinstance(m, RoutingStatsMsg):
        send_backend.put(RoutingStatsBackendMsg(request_id=m.request_id, reset=m.reset))
    elif isinstance(m, RoutingStatsResultMsg):
        send_frontend.put(RoutingStatsReply(request_id=m.request_id, stats=m.stats, error=m.error))
    elif isinstance(m, CacheRebuildResultMsg):
        send_frontend.put(
            CacheRebuildReply(
                request_id=m.request_id,
                status=m.status,
                moe_cache_size=m.moe_cache_size,
                num_pages=m.num_pages,
                mamba_slots=m.mamba_slots,
                num_swa_pages=m.num_swa_pages,
                error=m.error,
            )
        )
    elif isinstance(m, CacheStepResultMsg):
        send_frontend.put(
            CacheStepReply(
                request_id=m.request_id,
                status=m.status,
                applied=m.applied,
                layer=m.layer,
                at_floor=m.at_floor,
                moe_cache_size=m.moe_cache_size,
                layers=m.layers,
                vram_free_bytes=m.vram_free_bytes,
                error=m.error,
                exhausted=bool(getattr(m, "exhausted", False)),
            )
        )
    elif isinstance(m, CacheResidencyResultMsg):
        send_frontend.put(
            CacheResidencyReply(
                request_id=m.request_id,
                status=m.status,
                residency=m.residency,
                error=m.error,
            )
        )
    else:
        return False
    return True

@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    multimodal_processor = _MultimodalProcessor(tokenizer_path)
    detokenize_manager = DetokenizeManager(
        tokenizer, load_eos_token_ids(tokenizer_path, tokenizer)
    )

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            prompt_admitted_msg = [m for m in pending_msg if isinstance(m, PromptAdmittedMsg)]
            error_reply_msg = [m for m in pending_msg if isinstance(m, ErrorReplyMsg)]
            # Control messages are pure passthrough (no tokenization): CacheRebuildMsg /
            # CacheStepMsg / CacheResidencyMsg / RoutingStatsMsg (api -> scheduler) and
            # status/result replies (scheduler -> api).
            for m in pending_msg:
                _forward_control_msg(m, send_backend, send_frontend)
            n_control = sum(isinstance(m, _CONTROL_MSG_TYPES) for m in pending_msg)
            assert (
                len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) + n_control
                == len(pending_msg)
            )
            sampled_replies: List[UserReply] = []
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                sampled_replies = [
                    _sampled_reply(msg, reply)
                    for msg, reply in zip(detokenize_msg, replies, strict=True)
                ]

            # An error reply and a client abort are both terminal for their uid, and neither
            # produces the finished DetokenizeMsg that would release the decode state.
            for msg in error_reply_msg:
                detokenize_manager.discard(msg.uid)
            for msg in abort_msg:
                detokenize_manager.discard(msg.uid)

            _send_generation_replies(
                send_frontend,
                [_prompt_admitted_reply(msg) for msg in prompt_admitted_msg],
                sampled_replies,
                [_error_reply(msg) for msg in error_reply_msg],
            )

            if len(tokenize_msg) > 0:
                # Tokenize per-message so a single un-renderable request (e.g. a chat template
                # that rejects the message layout) becomes a terminal error reply for THAT uid
                # instead of an uncaught exception that kills the worker and bricks the server.
                ok_msgs, ok_tensors, ok_multimodal, errors = _tokenize_requests(
                    tokenize_manager, multimodal_processor, tokenize_msg, logger
                )
                if errors:
                    send_frontend.put(
                        errors[0] if len(errors) == 1 else BatchFrontendMsg(data=errors)
                    )
                if ok_msgs:
                    backend = []
                    for msg, tokens, mm in zip(
                        ok_msgs, ok_tensors, ok_multimodal, strict=True
                    ):
                        backend.append(
                            UserMsg(
                                uid=msg.uid,
                                input_ids=tokens,
                                sampling_params=msg.sampling_params,
                                mm_pixel_values=(mm or {}).get("pixel_values"),
                                mm_image_grid_thw=(mm or {}).get("image_grid_thw"),
                                mm_token_type_ids=(mm or {}).get("mm_token_type_ids"),
                            )
                        )
                    send_backend.put(
                        backend[0] if len(backend) == 1 else BatchBackendMsg(data=backend)
                    )
            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass

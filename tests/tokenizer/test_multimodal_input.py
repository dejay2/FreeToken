from __future__ import annotations

import base64
import contextlib
import io
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer import server as tokenizer_server
from freetoken.tokenizer.server import (
    _MultimodalProcessor,
    _load_rgb_image,
    _message_image_sources,
    _read_image_source,
)


def _png_bytes(*, color=(220, 30, 40)) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (16, 12), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    return buffer.getvalue()


@contextlib.contextmanager
def _http_fixture(payload: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == "/picture.png":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path.startswith("/redirect/"):
                count = int(self.path.rsplit("/", 1)[-1])
                self.send_response(302)
                self.send_header(
                    "Location", "/picture.png" if count == 0 else f"/redirect/{count - 1}"
                )
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _picture_message(source: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": source}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]


def test_picture_sources_are_extracted_in_message_order():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "image", "image": "C:\\pictures\\second.png"},
                {"type": "text", "text": "compare"},
            ],
        }
    ]

    assert _message_image_sources(messages) == [
        "data:image/png;base64,AA==",
        "C:\\pictures\\second.png",
    ]


def test_data_url_base64_and_percent_encoding_are_read():
    payload = b"picture-bytes\x00\xff"
    encoded = base64.b64encode(payload).decode("ascii")

    assert _read_image_source(f"data:image/png;base64,{encoded}") == payload
    assert _read_image_source("data:image/png,picture%2Dbytes") == b"picture-bytes"


@pytest.mark.parametrize(
    "source",
    [
        "data:image/png;base64,***not-base64***",
        "ftp://example.test/picture.png",
        "",
    ],
)
def test_invalid_sources_are_rejected(source: str):
    with pytest.raises(ValueError):
        _read_image_source(source)


def test_direct_windows_path_and_file_url_are_read(tmp_path: Path):
    path = tmp_path / "picture.png"
    payload = _png_bytes()
    path.write_bytes(payload)

    assert _read_image_source(str(path)) == payload
    assert _read_image_source(path.as_uri()) == payload


def test_http_source_and_bounded_redirects_are_read():
    payload = _png_bytes()
    with _http_fixture(payload) as base:
        assert _read_image_source(f"{base}/picture.png") == payload
        assert _read_image_source(f"{base}/redirect/4") == payload
        with pytest.raises(ValueError, match="redirect"):
            _read_image_source(f"{base}/redirect/5")


def test_source_limit_applies_to_data_local_and_http(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tokenizer_server, "_MAX_IMAGE_BYTES", 32)
    payload = b"x" * 33
    encoded = base64.b64encode(payload).decode("ascii")
    path = tmp_path / "too-large.bin"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="64 MiB|input limit"):
        _read_image_source(f"data:image/png;base64,{encoded}")
    with pytest.raises(ValueError, match="64 MiB|input limit"):
        _read_image_source(str(path))
    with _http_fixture(payload) as base:
        with pytest.raises(ValueError, match="64 MiB|input limit"):
            _read_image_source(f"{base}/picture.png")


def test_picture_decode_converts_to_rgb_and_rejects_invalid_bytes():
    image = _load_rgb_image(_png_bytes())
    try:
        assert image.mode == "RGB"
        assert image.size == (16, 12)
    finally:
        image.close()

    with pytest.raises(ValueError, match="invalid|decode|picture"):
        _load_rgb_image(b"not a picture")


def test_processor_output_contract_uses_transport_dtypes():
    payload = _png_bytes()
    source = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")

    class FakeProcessor:
        def __call__(self, *, text, images, return_tensors):
            assert text == ["rendered prompt"]
            assert len(images) == 1 and images[0].mode == "RGB"
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.int64),
                "pixel_values": torch.arange(24, dtype=torch.float32).reshape(3, 8),
                "image_grid_thw": torch.tensor([[1, 2, 4]], dtype=torch.int64),
                "mm_token_type_ids": torch.tensor([[0, 1, 1]], dtype=torch.int64),
            }

    class FakeTokenizeManager:
        def render_prompt(self, msg):
            return "rendered prompt"

    processor = _MultimodalProcessor("unused")
    processor.processor = FakeProcessor()
    msg = TokenizeMsg(
        uid=1,
        text=_picture_message(source),
        sampling_params=SamplingParams(max_tokens=4),
    )

    input_ids, mm = processor.encode(msg, FakeTokenizeManager())

    assert input_ids.dtype == torch.int32 and input_ids.shape == (3,)
    assert mm is not None
    assert mm["pixel_values"].dtype == torch.bfloat16
    assert mm["pixel_values"].shape == (3, 8)
    assert mm["image_grid_thw"].dtype == torch.int64
    assert mm["image_grid_thw"].shape == (1, 3)
    assert mm["mm_token_type_ids"].dtype == torch.int32
    assert mm["mm_token_type_ids"].shape == (3,)


def test_text_only_processor_path_does_not_load_picture_packages():
    class FakeTokenizeManager:
        def tokenize(self, messages):
            return [torch.tensor([4, 5], dtype=torch.int32)]

    processor = _MultimodalProcessor("unused")
    msg = TokenizeMsg(
        uid=2,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(max_tokens=4),
    )

    input_ids, mm = processor.encode(msg, FakeTokenizeManager())

    assert torch.equal(input_ids, torch.tensor([4, 5], dtype=torch.int32))
    assert mm is None
    assert processor.processor is None


@pytest.mark.needs_weights
def test_real_qwen_processor_contract_when_model_is_configured():
    model_path = os.getenv("FREETOKEN_TEST_VISION_PROCESSOR")
    if not model_path:
        pytest.skip("set FREETOKEN_TEST_VISION_PROCESSOR to the local Qwen checkpoint")

    from freetoken.tokenizer.tokenize import TokenizeManager
    from freetoken.utils import load_tokenizer

    payload = _png_bytes(color=(20, 120, 230))
    source = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    manager = TokenizeManager(load_tokenizer(model_path))
    processor = _MultimodalProcessor(model_path)
    msg = TokenizeMsg(
        uid=3,
        text=_picture_message(source),
        sampling_params=SamplingParams(max_tokens=4),
    )

    input_ids, mm = processor.encode(msg, manager)

    assert input_ids.ndim == 1 and input_ids.dtype == torch.int32
    assert mm is not None
    assert mm["pixel_values"].ndim == 2 and mm["pixel_values"].dtype == torch.bfloat16
    assert mm["image_grid_thw"].ndim == 2 and mm["image_grid_thw"].shape[1] == 3
    assert mm["mm_token_type_ids"].shape == input_ids.shape

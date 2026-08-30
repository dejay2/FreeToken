"""Private end-to-end acceptance for Qwen3.8 still-picture input on Windows."""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PICTURE_SOURCE_LIMIT = 64 << 20


def get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(
    url: str, payload: dict[str, Any], timeout: int = 900
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _answer(body: dict[str, Any]) -> str:
    message = body["choices"][0]["message"]
    return " ".join(
        value for value in (message.get("reasoning_content"), message.get("content")) if value
    )


def _picture_payload(model: str, source: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": source}},
                ],
            }
        ],
        "max_tokens": 320,
        "reasoning_effort": "low",
        "temperature": 0,
    }


def _make_fixture(path: Path, *, code: str, primary: str, secondary: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (800, 520), "white")
    draw = ImageDraw.Draw(image)
    if primary == "blue":
        draw.rectangle((55, 55, 300, 300), fill=(20, 90, 230), outline="black", width=10)
        first_label = "BLUE SQUARE"
    else:
        draw.polygon(
            [(55, 300), (180, 55), (305, 300)],
            fill=(20, 180, 70),
            outline="black",
            width=10,
        )
        first_label = "GREEN TRIANGLE"
    if secondary == "red":
        draw.ellipse((475, 65, 720, 310), fill=(235, 45, 55), outline="black", width=10)
        second_label = "RED CIRCLE"
    else:
        draw.rectangle((475, 65, 720, 310), fill=(245, 205, 20), outline="black", width=10)
        second_label = "YELLOW RECTANGLE"
    draw.text((70, 335), first_label, fill="black", font=ImageFont.truetype("arialbd.ttf", 30))
    draw.text((455, 335), second_label, fill="black", font=ImageFont.truetype("arialbd.ttf", 27))
    draw.text((205, 410), f"CODE {code}", fill="black", font=ImageFont.truetype("arialbd.ttf", 58))
    image.save(path, optimize=True)


class _FixtureHandler(BaseHTTPRequestHandler):
    picture: bytes = b""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path == "/picture.png":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(self.picture)))
            self.end_headers()
            self.wfile.write(self.picture)
            return
        if self.path.startswith("/redirect/"):
            step = int(self.path.rsplit("/", 1)[1])
            self.send_response(302)
            self.send_header("Location", f"/redirect/{step + 1}")
            self.end_headers()
            return
        if self.path == "/oversized.bin":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(PICTURE_SOURCE_LIMIT + 1))
            self.end_headers()
            chunk = b"x" * (1 << 20)
            try:
                for _ in range(65):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            return
        self.send_error(404)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


@contextlib.contextmanager
def fixture_server(picture: bytes):
    _FixtureHandler.picture = picture
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _system_used_bytes() -> int:
    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return int(status.ullTotalPhys - status.ullAvailPhys)


def _gpu_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return int(output.splitlines()[0].strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class MemorySampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=15)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self.samples.append(
                {
                    "time": time.time(),
                    "gpu_used_mib": _gpu_used_mib(),
                    "system_used_bytes": _system_used_bytes(),
                }
            )


def _assert_picture(
    chat_url: str,
    model: str,
    source: str,
    code: str,
    *,
    prompt: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    status, body = post_json(chat_url, _picture_payload(model, source, prompt))
    elapsed = time.perf_counter() - started
    assert status == 200, body
    answer = _answer(body)
    assert code in answer, answer
    usage = body.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get(
        "cached_tokens", usage.get("cached_tokens", 0)
    )
    assert cached in (None, 0), usage
    return {
        "seconds": elapsed,
        "answer": body["choices"][0]["message"].get("content") or answer,
        "usage": usage,
        "cached_tokens": cached or 0,
    }


def _assert_error(chat_url: str, model: str, source: str, needle: str) -> dict[str, Any]:
    status, body = post_json(
        chat_url,
        _picture_payload(model, source, "Describe this picture."),
        timeout=120,
    )
    encoded = json.dumps(body).lower()
    assert status == 400 and needle.lower() in encoded, (status, body)
    return {"status": status, "error": body}


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"
    fixture_dir = args.fixture_dir.resolve()
    first = fixture_dir / "pi-picture.png"
    second = fixture_dir / "second-picture.png"
    _make_fixture(first, code="424242", primary="blue", secondary="red")
    _make_fixture(second, code="777777", primary="green", secondary="yellow")
    first_bytes = first.read_bytes()
    data_url = "data:image/png;base64," + base64.b64encode(first_bytes).decode("ascii")

    evidence: dict[str, Any] = {
        "started_at_unix": time.time(),
        "base_url": base_url,
        "model": args.model,
        "fixtures": {"first": str(first), "second": str(second)},
        "sources": {},
        "errors": {},
    }
    model_body = get_json(f"{base_url}/models")
    assert any(item.get("id") == args.model for item in model_body.get("data", [])), model_body
    status_body = get_json(f"{base_url}/cache/status")
    geometry = status_body["geometry"]
    usable = (geometry["num_pages"] - 1) * geometry["page_size"]
    assert geometry["num_pages"] == 4097 and usable == 262144, geometry
    evidence["cache_geometry"] = geometry

    prompt = "Read the large numeric code in this picture. Reply with only the code."
    with fixture_server(first_bytes) as fixture_url, MemorySampler() as memory:
        sources = {
            "data": data_url,
            "direct_path": first.as_posix(),
            "file_url": first.as_uri(),
            "loopback_http": f"{fixture_url}/picture.png",
        }
        for name, source in sources.items():
            evidence["sources"][name] = _assert_picture(
                chat_url, args.model, source, "424242", prompt=prompt
            )
            print(f"PASS picture source {name}")

        first_result = _assert_picture(chat_url, args.model, first.as_posix(), "424242", prompt=prompt)
        second_result = _assert_picture(
            chat_url, args.model, second.as_posix(), "777777", prompt=prompt
        )
        evidence["cross_picture"] = {"first": first_result, "second": second_result}
        print("PASS cache-private cross-picture requests")

        two_payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Read both pictures in order. Report both numeric codes.",
                        },
                        {"type": "image_url", "image_url": {"url": first.as_posix()}},
                        {"type": "image_url", "image_url": {"url": second.as_posix()}},
                    ],
                }
            ],
            "max_tokens": 420,
            "reasoning_effort": "low",
            "temperature": 0,
        }
        code, body = post_json(chat_url, two_payload)
        answer = _answer(body)
        assert code == 200 and "424242" in answer and "777777" in answer, (code, body)
        evidence["two_picture"] = {"answer": answer, "usage": body.get("usage")}
        print("PASS two-picture order")

        evidence["errors"]["malformed_base64"] = _assert_error(
            chat_url, args.model, "data:image/png;base64,%%%", "invalid picture data url"
        )
        evidence["errors"]["invalid_bytes"] = _assert_error(
            chat_url,
            args.model,
            "data:image/png;base64," + base64.b64encode(b"not a picture").decode(),
            "invalid picture",
        )
        evidence["errors"]["oversized"] = _assert_error(
            chat_url, args.model, f"{fixture_url}/oversized.bin", "64 mib"
        )
        evidence["errors"]["redirect_overflow"] = _assert_error(
            chat_url, args.model, f"{fixture_url}/redirect/0", "redirect"
        )
        evidence["errors"]["unreadable_file"] = _assert_error(
            chat_url,
            args.model,
            (fixture_dir / "does-not-exist.png").as_posix(),
            "could not read picture",
        )
        print("PASS bounded picture errors")

        text_payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly HEALTHY."}],
            "max_tokens": 64,
            "reasoning_effort": "low",
            "temperature": 0,
        }
        started = time.perf_counter()
        code, body = post_json(chat_url, text_payload)
        elapsed = time.perf_counter() - started
        assert code == 200 and "HEALTHY" in _answer(body), body
        completion = (body.get("usage") or {}).get("completion_tokens", 0)
        evidence["short_text"] = {
            "seconds": elapsed,
            "usage": body.get("usage"),
            "tokens_per_second": completion / elapsed if elapsed else None,
        }
        print("PASS text after picture failures")

        if not args.skip_long_text:
            long_prompt = (
                "This is an ordinary text-only chunking test. Read every repeated word, then "
                "reply with exactly LONG_TEXT_OK. "
                + " token" * 32_768
            )
            long_payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": long_prompt}],
                "max_tokens": 96,
                "reasoning_effort": "low",
                "temperature": 0,
            }
            started = time.perf_counter()
            code, body = post_json(chat_url, long_payload, timeout=1800)
            elapsed = time.perf_counter() - started
            usage = body.get("usage") or {}
            assert code == 200 and usage.get("prompt_tokens", 0) >= 32_768, (code, body)
            evidence["long_text"] = {
                "seconds": elapsed,
                "answer": _answer(body),
                "usage": usage,
            }
            print("PASS 32K text-only chunking")

    samples = memory.samples
    evidence["memory_samples"] = samples
    evidence["memory_summary"] = {
        "peak_gpu_used_mib": max(
            (sample["gpu_used_mib"] for sample in samples if sample["gpu_used_mib"] is not None),
            default=None,
        ),
        "peak_system_used_bytes": max(
            (sample["system_used_bytes"] for sample in samples), default=None
        ),
    }
    get_json(f"{base_url}/models")
    evidence["finished_at_unix"] = time.time()
    evidence["passed"] = True
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:2020/v1")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".local" / "vision-fixtures",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".local"
        / "qwen38-vision-acceptance.json",
    )
    parser.add_argument("--skip-long-text", action="store_true")
    args = parser.parse_args()
    try:
        evidence = run(args)
    except Exception as exc:  # noqa: BLE001 - CLI must write a concise terminal failure
        print(f"FAIL {exc}")
        return 1
    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"PASS evidence written to {args.evidence_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

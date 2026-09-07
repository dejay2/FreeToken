#!/usr/bin/env python3
"""Local OpenAI-compatible stub server for FreeToken squeeze tests.

Provides a minimal HTTP server answering /v1/chat/completions and /health
so hammer.py and bench.py can be verified locally on a devbox without hitting
the live serving box (which may be in use).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StubRequestHandler(BaseHTTPRequestHandler):
    """Subclass of BaseHTTPRequestHandler answering OpenAI-shaped chat completions."""

    server_version = "FreeTokenStub/0.1"

    def log_message(self, format: str, *args) -> None:
        # Suppress verbose access logs unless explicitly requested
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    def do_GET(self) -> None:
        if self.path in ("/health", "/"):
            body = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if not self.path.startswith("/v1/chat/completions"):
            self.send_error(404, "Not Found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        try:
            req_body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            req_body = {}

        delay = getattr(self.server, "delay", 0.0)
        if delay > 0:
            time.sleep(delay)

        max_tokens = int(req_body.get("max_tokens", 64))
        model = str(req_body.get("model", "stub-model"))
        stream = bool(req_body.get("stream", False))

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # First chunk: role
            chunk1 = {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk1)}\n\n".encode("utf-8"))
            self.wfile.flush()

            # Second chunk: content
            chunk2 = {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": "Stub completion response."}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk2)}\n\n".encode("utf-8"))
            self.wfile.flush()

            # Third chunk: finish and usage
            chunk3 = {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 16,
                    "completion_tokens": max_tokens,
                    "total_tokens": 16 + max_tokens,
                },
            }
            self.wfile.write(f"data: {json.dumps(chunk3)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            resp = {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Stub response: The lighthouse keeper watched the beacon rotate through the storm.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 16,
                    "completion_tokens": max_tokens,
                    "total_tokens": 16 + max_tokens,
                },
            }
            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def run_server(host: str = "127.0.0.1", port: int = 12020, delay: float = 0.0, verbose: bool = False) -> None:
    server = ThreadingHTTPServer((host, port), StubRequestHandler)
    server.delay = delay  # type: ignore[attr-defined]
    server.verbose = verbose  # type: ignore[attr-defined]
    print(f"Stub server listening on http://{host}:{port} (delay={delay}s)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStub server stopped.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local stub OpenAI HTTP server for testing.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=12020, help="Port to listen on (default: 12020)")
    parser.add_argument("--delay", type=float, default=0.0, help="Simulated latency in seconds (default: 0.0)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose HTTP access logs")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, delay=args.delay, verbose=args.verbose)


if __name__ == "__main__":
    main()

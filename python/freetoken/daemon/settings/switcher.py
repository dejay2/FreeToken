"""A small client for the model switcher (llama-swap, 127.0.0.1:2040). urllib only, torch-free."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

DEFAULT_URL = "http://127.0.0.1:2040"
LOADED_STATES = frozenset({"starting", "ready"})


class SwitcherError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _json(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8") or "null")
    except (UnicodeDecodeError, ValueError):
        return data.decode("utf-8", "replace")


class SwitcherClient:
    def __init__(self, base_url: str | None = None, *, timeout: float = 5.0,
                 urlopen: Callable[..., Any] = urllib.request.urlopen) -> None:
        self.base_url = (base_url or os.environ.get("FREETOKEN_SWITCHER_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self._urlopen = urlopen

    def _request(self, method: str, path: str, *, timeout: float | None = None) -> tuple[int, Any]:
        request = urllib.request.Request(self.base_url + path, method=method, data=b"" if method == "POST" else None)
        try:
            with self._urlopen(request, timeout=timeout or self.timeout) as response:
                return response.status, _json(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, _json(exc.read())

    @staticmethod
    def _quote(model_id: str) -> str:
        return urllib.parse.quote(model_id, safe="")

    def running(self) -> dict[str, str] | None:
        try:
            status, body = self._request("GET", "/running")
        except OSError:
            return None
        if status != 200 or not isinstance(body, dict):
            return None
        return {str(row.get("model")): str(row.get("state"))
                for row in body.get("running") or [] if isinstance(row, dict)}

    def config_hash(self) -> str | None:
        try:
            status, body = self._request("GET", "/api/config/hash")
        except OSError:
            return None
        return body.get("sha256") if status == 200 and isinstance(body, dict) else None

    def unload(self, model_id: str) -> bool:
        try:
            status, _ = self._request("POST", f"/api/models/unload/{self._quote(model_id)}", timeout=240.0)
        except OSError:
            return False
        return status == 200

    def load(self, model_id: str, *, timeout: float = 900.0) -> None:
        """Blocks until the model is ready. OSError when the switcher is not running."""
        status, body = self._request("POST", f"/api/models/load/{self._quote(model_id)}", timeout=timeout)
        if status == 200:
            return
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            raise SwitcherError(status, str(error.get("code") or "load_failed"), str(error.get("message") or ""))
        raise SwitcherError(status, "load_failed", str(body or f"HTTP {status}"))


__all__ = ["DEFAULT_URL", "LOADED_STATES", "SwitcherClient", "SwitcherError"]

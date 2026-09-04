"""Hugging Face model previews and resumable local downloads for the settings helper.

The helper deliberately does not import torch or any model package.  Hugging Face is imported only
when a preview or a download actually needs it, so the page can still start on a machine without
CUDA.  Download progress is measured from the target folder itself; the Hub downloader's progress
callbacks do not reliably cover local cache files or partial files.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .model_info import (
    GIB,
    ModelInfo,
    describe_config,
    ple_bytes_from_header,
    read_model,
    safetensors_ple_bytes,
)


_REPO_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
# The Hub repositories used by the engine can contain source checkpoints under ``original/`` or
# ``metal/``.  Keep this list in sync with ``_is_downloadable_file``: preview totals and the actual
# snapshot calls must describe and fetch the same small, top-level set.
_DOWNLOAD_ALLOW_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "chat_template*",
    "chat-template*",
)
_DOWNLOAD_IGNORE_PATTERNS = ("*/*",)
_WEIGHT_SUFFIXES = (".safetensors",)
_TERMINAL_STAGES = frozenset({"done", "failed", "cancelled"})


class InvalidRepository(ValueError):
    """The submitted value is not a Hugging Face ``owner/name`` repository."""


class DownloadConflict(RuntimeError):
    """A target folder or another active download already owns the requested work."""


class DownloadBody(BaseModel):
    repo: str = Field(min_length=1)


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int | None = None


@dataclass
class DownloadJob:
    job_id: str
    repo: str
    target_folder: Path
    stage: str = "queued"
    received_bytes: int = 0
    total_bytes: int = 0
    percent: float = 0.0
    files: list[str] = field(default_factory=list)
    error: str | None = None
    cancel_requested: bool = False
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.job_id,
            "repo": self.repo,
            "targetFolder": str(self.target_folder),
            "stage": self.stage,
            "receivedBytes": int(self.received_bytes),
            "totalBytes": int(self.total_bytes),
            "percent": float(self.percent),
            "files": list(self.files),
            "error": self.error,
        }


def parse_repo(value: str) -> str:
    """Return a canonical ``owner/name`` from a Hub link or repository name.

    Accepted links intentionally cover only the two forms the page advertises.  In particular,
    arbitrary URLs are not converted into filesystem names, which keeps a pasted link from
    escaping the configured models folder.
    """
    text = str(value or "").strip()
    if not text:
        raise InvalidRepository("Paste a Hugging Face model link or an owner/name.")

    if "://" in text:
        try:
            parsed = urlsplit(text)
            hostname = parsed.hostname
        except ValueError as exc:
            raise InvalidRepository("That is not a valid Hugging Face model link.") from exc
        if parsed.scheme.lower() != "https" or hostname is None or hostname.lower() != "huggingface.co":
            raise InvalidRepository("Use an https://huggingface.co/owner/name link.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise InvalidRepository("The model link must not contain a user, query, or fragment.")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 4 and parts[2] == "tree" and parts[3] == "main":
            parts = parts[:2]
        if len(parts) != 2:
            raise InvalidRepository("Use a model link ending in /owner/name or /owner/name/tree/main.")
    else:
        parts = text.rstrip("/").split("/")
        if len(parts) != 2:
            raise InvalidRepository("Use a Hugging Face model link or an owner/name.")

    owner, name = parts
    if not _REPO_PART.fullmatch(owner) or not _REPO_PART.fullmatch(name):
        raise InvalidRepository("The owner and model name contain an unsupported character.")
    return f"{owner}/{name}"


# Friendly aliases for small integrations and tests that describe this as Hub parsing.
parse_huggingface_repo = parse_repo
parse_hf_repo = parse_repo


def physical_memory_bytes() -> int:
    """Return total physical PC memory without importing a GPU library."""
    if os.name == "nt":
        class _MemoryStatusEx(ctypes.Structure):
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

        # Keeping the structure local avoids importing any platform package in the torch-free
        # helper while still matching the Windows SDK's MEMORYSTATUSEX layout.
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            pass
        return 0

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 0
    return max(0, pages * page_size)


# A shorter name is useful to callers that only need the measurement.
pc_memory_bytes = physical_memory_bytes


def disk_free_bytes(path: str | os.PathLike[str]) -> int:
    """Return free bytes on the drive containing ``path`` (even before the folder exists)."""
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        return max(0, int(shutil.disk_usage(candidate).free))
    except OSError:
        return 0


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _size(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _remote_files(info: Any) -> list[RemoteFile]:
    siblings = _field(info, "siblings", []) or []
    if isinstance(siblings, dict):
        siblings = siblings.values()
    result: list[RemoteFile] = []
    seen: set[str] = set()
    for sibling in siblings:
        name = _field(sibling, "rfilename") or _field(sibling, "path") or _field(sibling, "name")
        if not isinstance(name, str) or not name or name in seen:
            continue
        lfs = _field(sibling, "lfs")
        size = _size(_field(sibling, "size"))
        if size is None and lfs is not None:
            size = _size(_field(lfs, "size"))
        result.append(RemoteFile(name=name, size=size))
        seen.add(name)
    return result


def _is_top_level_file(name: str) -> bool:
    """Return whether a Hub path is a file directly in the repository root."""
    return bool(name) and "/" not in name and "\\" not in name


def _is_weight_file(name: str) -> bool:
    return _is_top_level_file(name) and name.lower().endswith(_WEIGHT_SUFFIXES)


def _is_downloadable_file(name: str) -> bool:
    """Keep only root weights and the small files needed to load their tokenizer and config."""
    if not _is_top_level_file(name):
        return False
    lower = name.lower()
    return (
        lower.endswith(".safetensors")
        or lower.endswith(".safetensors.index.json")
        or lower
        in {
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "spiece.model",
        }
        or lower.startswith("tokenizer")
        or lower.startswith("chat_template")
        or lower.startswith("chat-template")
    )


def _downloadable_files(files: list[RemoteFile]) -> list[RemoteFile]:
    return [item for item in files if _is_downloadable_file(item.name)]


def _default_hf_api() -> Any:
    # Lazy import is intentional: the daemon import-safety test blocks torch and its neighbours.
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    return HfApi(token=token) if token else HfApi()


def _default_snapshot_download(repo: str, *, local_dir: str, **kwargs: Any) -> Any:
    # Lazy import keeps the settings page usable when only the daemon dependencies are installed.
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, local_dir=local_dir, **kwargs)


def _default_config_fetcher(repo: str, downloads_dir: Path) -> dict[str, Any]:
    """Fetch only config.json into a temporary folder; no model weight is downloaded."""
    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory(prefix="freetoken-config-") as temporary:
        kwargs: dict[str, Any] = {
            "repo_id": repo,
            "filename": "config.json",
            "local_dir": temporary,
        }
        token = os.environ.get("HF_TOKEN")
        if token:
            kwargs["token"] = token
        path = hf_hub_download(**kwargs)
        with Path(path).open("r", encoding="utf-8") as fh:
            return json.load(fh)


_MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024


def _remote_range(url: str, start: int, end: int, token: str | None) -> bytes:
    """Read one bounded byte range without ever buffering a full weight shard."""
    expected = end - start + 1
    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(request, timeout=30) as response:
            status = getattr(response, "status", None) or response.getcode()
            content_range = response.headers.get("Content-Range")
            data = response.read(expected + 1)
    except (OSError, ValueError):
        return b""
    # A server that ignores Range would otherwise hand us the entire weight file. Reject it
    # after reading only one extra byte, and require a partial response for non-empty ranges.
    if len(data) != expected or (status == 200 and not content_range):
        return b""
    return data


def _default_safetensors_header_fetcher(repo: str, filename: str) -> bytes:
    """Fetch only a remote safetensors header, never its tensor payload."""
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=repo, filename=filename)
    token = os.environ.get("HF_TOKEN")
    prefix = _remote_range(url, 0, 7, token)
    if len(prefix) != 8:
        return b""
    header_length = int.from_bytes(prefix, "little", signed=False)
    if not 0 < header_length <= _MAX_SAFETENSORS_HEADER_BYTES:
        return b""
    return _remote_range(url, 8, 7 + header_length, token)


def _header_ple_bytes(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ple_bytes_from_header(value)
    if isinstance(value, (str, os.PathLike)):
        return safetensors_ple_bytes(value)
    return 0


def _decode_config(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, (str, os.PathLike)):
        candidate = Path(value)
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as fh:
                value = json.load(fh)
        else:
            value = json.loads(str(value))
    if not isinstance(value, dict):
        raise ValueError("Hub config.json is not a settings object.")
    return value


def _folder_size(folder: Path) -> int:
    """Count every regular file below a target, including Hub's ``.cache`` partials."""
    if not folder.exists():
        return 0
    if folder.is_file():
        try:
            return max(0, int(folder.stat().st_size))
        except OSError:
            return 0
    total = 0
    try:
        for root, dirs, files in os.walk(folder, followlinks=False):
            # A symlinked directory can point outside the target; do not count or traverse it.
            dirs[:] = [name for name in dirs if not (Path(root) / name).is_symlink()]
            for name in files:
                path = Path(root) / name
                try:
                    if path.is_symlink():
                        continue
                    total += max(0, int(path.stat().st_size))
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _as_bytes(value: int | Callable[[], int] | None, default: int = 0) -> int:
    try:
        result = value() if callable(value) else (default if value is None else value)
        return max(0, int(result))
    except (TypeError, ValueError, OverflowError, OSError):
        return 0


def _fit(
    model: ModelInfo,
    download_bytes: int,
    *,
    pc_bytes: int,
    card_bytes: int,
) -> dict[str, Any]:
    expert_bytes = max(0, int(model.total_expert_bytes or 0))
    download_bytes = max(0, int(download_bytes))
    if model.is_moe:
        # Fit is about resident model tensors, not tokenizer/config files or the demand-paged
        # PLE table. Prefer the known top-level weight total; Hub metadata includes both kinds.
        weight_bytes = max(0, int(model.weight_bytes or 0))
        source_bytes = weight_bytes or download_bytes
        ple_bytes = max(0, int(model.ple_bytes or 0))
        dense_bytes = max(0, source_bytes - expert_bytes - ple_bytes)
        needs_bytes = expert_bytes + dense_bytes
        budget_bytes = max(0, pc_bytes - 8 * GIB) + min(int(card_bytes * 0.75), 24 * GIB)
    else:
        dense_bytes = download_bytes
        needs_bytes = dense_bytes
        budget_bytes = card_bytes

    fits = bool(budget_bytes and needs_bytes <= budget_bytes)
    shortfall = max(0, needs_bytes - budget_bytes)
    headroom = max(0, budget_bytes - needs_bytes)
    # A fit with less than ten percent or 8 GiB spare deserves an amber warning in the page.
    tight = fits and headroom < max(8 * GIB, int(budget_bytes * 0.10))
    if not fits:
        if not budget_bytes:
            verdict = "The available memory could not be measured, so this model is not cleared to start."
        else:
            verdict = (
                f"This model needs {shortfall / GIB:.1f} GiB more memory than this PC can provide, "
                "so it will not start here."
            )
    elif tight:
        verdict = f"This model fits, but leaves only {headroom / GIB:.1f} GiB of memory spare."
    else:
        verdict = "This model fits within this PC's estimated memory budget."
    return {
        "pcMemoryBytes": int(pc_bytes),
        "cardMemoryBytes": int(card_bytes),
        "expertBytes": expert_bytes,
        "denseBytes": dense_bytes,
        "needsBytes": int(needs_bytes),
        "budgetBytes": int(budget_bytes),
        "fits": fits,
        "tight": tight,
        "shortfallBytes": int(shortfall),
        "headroomBytes": int(headroom),
        "verdict": verdict,
    }


class DownloadManager:
    """Own preview data and one background Hub download for a helper process."""

    def __init__(
        self,
        models_dir: str | os.PathLike[str],
        downloads_dir: str | os.PathLike[str] | None = None,
        *,
        api_factory: Callable[[], Any] | Any | None = None,
        config_fetcher: Callable[..., Any] | None = None,
        snapshot_downloader: Callable[..., Any] | None = None,
        header_fetcher: Callable[..., Any] | None = None,
        pc_memory: int | Callable[[], int] | None = None,
        card_memory: int | Callable[[], int] | None = None,
        disk_free: Callable[[str | os.PathLike[str]], int] | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.downloads_dir = Path(downloads_dir) if downloads_dir is not None else self.models_dir
        self._api_factory = api_factory
        self._config_fetcher = config_fetcher
        self._snapshot_downloader = snapshot_downloader or _default_snapshot_download
        # Test doubles do not have to make network calls for header metadata. The live helper,
        # which uses the default Hub API, gets a bounded Range reader unless a seam is supplied.
        self._header_fetcher = (
            header_fetcher
            if header_fetcher is not None
            else (_default_safetensors_header_fetcher if api_factory is None else None)
        )
        self._pc_memory = pc_memory if pc_memory is not None else physical_memory_bytes
        self._card_memory = card_memory
        self._disk_free = disk_free or disk_free_bytes
        self._jobs: dict[str, DownloadJob] = {}
        self._active_id: str | None = None
        self._partial_targets: set[str] = set()
        self._lock = threading.RLock()

    # ---- preview and model catalogue -------------------------------------

    def _api(self) -> Any:
        factory = self._api_factory
        if factory is None:
            return _default_hf_api()
        if hasattr(factory, "model_info"):
            return factory
        token = os.environ.get("HF_TOKEN")
        try:
            return factory(token=token) if token else factory()
        except TypeError:
            return factory()

    def _hub_files(self, repo: str) -> list[RemoteFile]:
        api = self._api()
        try:
            info = api.model_info(repo, files_metadata=True)
        except TypeError:
            # Tiny test doubles and older Hub clients do not have the metadata keyword.
            info = api.model_info(repo)
        return _remote_files(info)

    def _config(self, repo: str) -> dict[str, Any]:
        if self._config_fetcher is None:
            return _default_config_fetcher(repo, self.downloads_dir)
        fetcher = self._config_fetcher
        try:
            return _decode_config(fetcher(repo))
        except TypeError:
            # Keep the seam friendly to a fetcher that also wants the configured scratch directory.
            return _decode_config(fetcher(repo, self.downloads_dir))

    def _ple_bytes(self, repo: str, files: list[RemoteFile]) -> int:
        fetcher = self._header_fetcher
        if fetcher is None:
            return 0
        total = 0
        for item in files:
            try:
                try:
                    header = fetcher(repo, item.name)
                except TypeError:
                    header = fetcher(repo=repo, filename=item.name)
                total += _header_ple_bytes(header)
            except Exception:  # noqa: BLE001 - an optional metadata probe cannot block preview
                continue
        return total

    def _measure_card(self) -> int:
        # An unknown card is represented by zero; unlike an estimate, that cannot falsely clear a
        # model for download.  The app supplies a callback backed by ProcessManager.server_status.
        return _as_bytes(self._card_memory)

    def preview(self, value: str) -> dict[str, Any]:
        repo = parse_repo(value)
        _, name = repo.split("/", 1)
        files = _downloadable_files(self._hub_files(repo))
        config = self._config(repo)
        target = self.models_dir / name
        model = describe_config(config, name, path=str(target))
        weight_files = [item for item in files if _is_weight_file(item.name)]
        download_bytes = sum(item.size or 0 for item in files)
        model.weight_files = len(weight_files)
        model.weight_bytes = sum(item.size or 0 for item in weight_files)
        if model.has_ple:
            model.ple_bytes = self._ple_bytes(repo, weight_files)
        pc_bytes = _as_bytes(self._pc_memory)
        card_bytes = self._measure_card()
        fit = _fit(model, download_bytes, pc_bytes=pc_bytes, card_bytes=card_bytes)
        try:
            free_bytes = max(0, int(self._disk_free(target)))
        except (TypeError, ValueError, OSError):
            free_bytes = 0
        disk_fits = bool(download_bytes and free_bytes >= download_bytes)
        result = model.as_dict()
        result.update(
            {
                "downloadBytes": int(download_bytes),
                "weightFiles": len(weight_files),
                "targetFolder": str(target),
                "exists": target.exists(),
                "fit": fit,
                "diskFreeBytes": free_bytes,
                "diskFits": disk_fits,
            }
        )
        return result

    def models(self) -> list[dict[str, Any]]:
        try:
            folders = sorted((item for item in self.models_dir.iterdir() if item.is_dir()), key=lambda p: p.name.lower())
        except (FileNotFoundError, NotADirectoryError, OSError):
            return []
        pc_bytes = _as_bytes(self._pc_memory)
        card_bytes = self._measure_card()
        result: list[dict[str, Any]] = []
        for folder in folders:
            model = read_model(folder)
            download_bytes = model.weight_bytes
            item = model.as_dict()
            item["fit"] = _fit(model, download_bytes, pc_bytes=pc_bytes, card_bytes=card_bytes)
            item["fitVerdict"] = item["fit"]["verdict"]
            result.append(item)
        return result

    # ---- background job API ----------------------------------------------

    @staticmethod
    def _target_key(path: Path) -> str:
        return os.path.normcase(os.path.abspath(str(path)))

    def start(self, value: str) -> DownloadJob:
        repo = parse_repo(value)
        _, name = repo.split("/", 1)
        target = self.models_dir / name
        target_key = self._target_key(target)
        with self._lock:
            if self._active_id is not None:
                active = self._jobs.get(self._active_id)
                stage = active.stage if active is not None else "downloading"
                raise DownloadConflict(f"Another model download is already {stage}.")
            if target.exists() and target_key not in self._partial_targets:
                raise DownloadConflict("That model folder already exists; it will not be overwritten.")
            job = DownloadJob(
                job_id=f"download-{uuid.uuid4().hex[:12]}",
                repo=repo,
                target_folder=target,
            )
            self._jobs[job.job_id] = job
            self._active_id = job.job_id
            self._partial_targets.discard(target_key)
            thread = threading.Thread(
                target=self._run,
                args=(job.job_id,),
                name=f"settings-download-{job.job_id[-6:]}",
                daemon=True,
            )
            thread.start()
            return job

    def get(self, job_id: str) -> DownloadJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            self._refresh(job)
            return job

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
            for job in jobs:
                self._refresh(job)
            return [job.as_dict() for job in reversed(jobs)]

    def cancel(self, job_id: str) -> DownloadJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.stage not in _TERMINAL_STAGES:
                # snapshot_download cannot be interrupted mid-file.  The worker sees this flag
                # before the next file, leaving the partial target for a later resume.
                job.cancel_requested = True
            self._refresh(job)
            return job

    def _refresh(self, job: DownloadJob) -> None:
        job.received_bytes = _folder_size(job.target_folder)
        if job.total_bytes:
            job.percent = min(100.0, job.received_bytes * 100.0 / job.total_bytes)
        elif job.stage == "done":
            job.percent = 100.0
        else:
            job.percent = 0.0

    def _cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs[job_id]
            return job.cancel_requested

    def _set_stage(self, job_id: str, stage: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.stage = stage

    def _finish(self, job_id: str, stage: str, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.stage = stage
            job.error = error
            job.completed_at = time.time()
            self._refresh(job)
            key = self._target_key(job.target_folder)
            if stage in {"cancelled", "failed"} and job.target_folder.exists():
                self._partial_targets.add(key)
            elif stage == "done":
                self._partial_targets.discard(key)
            if self._active_id == job_id:
                self._active_id = None

    def _run(self, job_id: str) -> None:
        self._set_stage(job_id, "downloading")
        try:
            with self._lock:
                job = self._jobs[job_id]
            try:
                files = _downloadable_files(self._hub_files(job.repo))
            except Exception:
                # snapshot_download can discover the repository itself.  A metadata outage should
                # not turn a valid request into a false preflight failure, but the fallback still
                # carries the same top-level allowlist and never downloads a whole repository.
                files = []
            with self._lock:
                job.total_bytes = sum(item.size or 0 for item in files)
            if self._cancelled(job_id):
                self._finish(job_id, "cancelled")
                return

            job.target_folder.mkdir(parents=True, exist_ok=True)
            if files:
                for item in files:
                    if self._cancelled(job_id):
                        self._finish(job_id, "cancelled")
                        return
                    self._download_one(job.repo, job.target_folder, item.name)
                    with self._lock:
                        if item.name not in job.files:
                            job.files.append(item.name)
                if self._cancelled(job_id):
                    self._finish(job_id, "cancelled")
                    return
            else:
                # No metadata is still useful for private repos and small test doubles.  The Hub
                # call remains one snapshot operation and the route reports bytes from disk.
                self._download_one(job.repo, job.target_folder, None)

            with self._lock:
                if not job.total_bytes:
                    job.total_bytes = _folder_size(job.target_folder)
            self._finish(job_id, "done")
        except Exception as exc:  # noqa: BLE001 - a failed download must not kill the helper
            self._finish(job_id, "failed", error=str(exc))

    def _download_one(self, repo: str, target: Path, filename: str | None) -> Any:
        # Exact names are used after a manifest is available.  The pattern fallback keeps the same
        # categories when metadata is unavailable; ``*/*`` matters because Hub globbing lets ``*``
        # cross directory boundaries.
        kwargs: dict[str, Any] = {
            "local_dir": str(target),
            "allow_patterns": [filename] if filename else list(_DOWNLOAD_ALLOW_PATTERNS),
            "ignore_patterns": list(_DOWNLOAD_IGNORE_PATTERNS),
        }
        token = os.environ.get("HF_TOKEN")
        if token:
            kwargs["token"] = token
        downloader = self._snapshot_downloader
        try:
            return downloader(repo, **kwargs)
        except TypeError as first_error:
            # Keep every filtering argument on compatibility retries.  Dropping an allowlist here
            # would silently turn a small model download into a whole-repository download.
            without_token = dict(kwargs)
            without_token.pop("token", None)
            try:
                return downloader(repo, **without_token)
            except TypeError:
                try:
                    return downloader(repo_id=repo, **without_token)
                except TypeError:
                    raise first_error


# ---- FastAPI routers ------------------------------------------------------


def create_router(
    *,
    models_dir: str | os.PathLike[str] | None = None,
    downloads_dir: str | os.PathLike[str] | None = None,
    manager: DownloadManager | None = None,
    api_factory: Callable[[], Any] | Any | None = None,
    config_fetcher: Callable[..., Any] | None = None,
    snapshot_downloader: Callable[..., Any] | None = None,
    header_fetcher: Callable[..., Any] | None = None,
    pc_memory: int | Callable[[], int] | None = None,
    card_memory: int | Callable[[], int] | None = None,
    disk_free: Callable[[str | os.PathLike[str]], int] | None = None,
) -> APIRouter:
    """Build the downloads and model-list routes for one app instance.

    The nested downloads router has the required ``/api/downloads`` prefix.  It is mounted inside
    one small root router so the companion ``/api/models`` catalogue can be included by app.py in
    the same single ``app.include_router(...)`` statement.
    """
    if manager is None:
        if models_dir is None:
            raise ValueError("models_dir is required when no DownloadManager is supplied")
        manager = DownloadManager(
            models_dir,
            downloads_dir,
            api_factory=api_factory,
            config_fetcher=config_fetcher,
            snapshot_downloader=snapshot_downloader,
            header_fetcher=header_fetcher,
            pc_memory=pc_memory,
            card_memory=card_memory,
            disk_free=disk_free,
        )
    downloads = APIRouter(prefix="/api/downloads", tags=["downloads"])
    models = APIRouter(tags=["models"])

    @downloads.get("/preview")
    def preview(repo: str = Query(..., min_length=1)):
        try:
            return manager.preview(repo)
        except InvalidRepository as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - turn Hub errors into a readable HTTP response
            raise HTTPException(status_code=502, detail=f"Could not preview that model: {exc}") from exc

    @downloads.post("", status_code=202)
    def start(body: DownloadBody):
        try:
            job = manager.start(body.repo)
        except InvalidRepository as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except DownloadConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"id": job.job_id}

    @downloads.get("/{job_id}")
    def progress(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"download {job_id} not found")
        return job.as_dict()

    @downloads.post("/{job_id}/cancel", status_code=202)
    def cancel(job_id: str):
        job = manager.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"download {job_id} not found")
        return job.as_dict()

    @downloads.get("")
    def list_downloads():
        return {"downloads": manager.list()}

    @models.get("/api/models")
    def list_models():
        return {"models": manager.models()}

    root = APIRouter()
    root.include_router(downloads)
    root.include_router(models)
    return root


create_download_router = create_router


__all__ = [
    "DownloadBody",
    "DownloadConflict",
    "DownloadJob",
    "DownloadManager",
    "InvalidRepository",
    "RemoteFile",
    "create_download_router",
    "create_router",
    "disk_free_bytes",
    "parse_hf_repo",
    "parse_huggingface_repo",
    "parse_repo",
    "pc_memory_bytes",
    "physical_memory_bytes",
]

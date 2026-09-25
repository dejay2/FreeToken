"""Hugging Face model previews and resumable local downloads for the settings helper.

The helper deliberately does not import torch or any model package.  Hugging Face is imported only
when a preview or a download actually needs it, so the page can still start on a machine without
CUDA.  Download progress is measured from the target folder itself; the Hub downloader's progress
callbacks do not reliably cover local cache files or partial files.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .memory_reclaim import release_completed_file
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

# The Add model wizard (control panel Stage B). A repo's SHA256SUMS wins over the Hub's
# per-file LFS sha256; files with neither (small git files) are not checked.
_SUMS_NAMES = ("SHA256SUMS", "SHA256SUMS.txt")
_NINFER_ENTRY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.ninfer$")
_NINFER_PART = re.compile(r"^(?P<entry>.+\.ninfer)\.part-\d{4}$")  # tools/artifact/writer.py naming
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SUMS_LINE = re.compile(r"^\s*([0-9A-Fa-f]{64})\s+\*?(\S.*?)\s*$")
_HASH_CHUNK = 8 * 1024 * 1024
# Wizard staging folders: ``<root>/.incoming-download-<hex>``. The dot keeps them out of the
# page's Browse list; the fixed prefix lets a later helper sweep the ones a crash left behind.
_STAGING_PREFIX = ".incoming-"
_STAGING_GLOB = _STAGING_PREFIX + "download-"
# 4 MiB per read keeps a cancel within a fraction of a second even at 100 MB/s while staying
# well under the 8 MiB hash chunk in per-call overhead.
_STREAM_CHUNK = 4 * 1024 * 1024
_STREAM_TIMEOUT_S = 60
# renameat2(2) constants (linux/fs.h, fcntl.h); glibc has wrapped the call since 2.28.
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


class InvalidRepository(ValueError):
    """The submitted value is not a Hugging Face ``owner/name`` repository."""


class DownloadConflict(RuntimeError):
    """A target folder or another active download already owns the requested work."""


class AddUnsupported(ValueError):
    """The repo holds nothing the engines can run."""


class ChecksumMismatch(RuntimeError):
    """A downloaded file does not match the checksum the repo publishes."""


class DownloadCancelled(Exception):
    """Raised inside the worker when the page's Cancel arrives mid-file, mid-hash or before moving."""


logger = logging.getLogger("freetoken.daemon.settings.download")

_OWN_WORDS = (InvalidRepository, DownloadConflict, AddUnsupported, ChecksumMismatch)


def _network_like(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return any((klass.__module__ or "").split(".")[0] in ("huggingface_hub", "requests", "urllib3", "urllib", "http", "socket", "ssl")
               for klass in type(exc).__mro__)


def _plain_job_error(exc: BaseException) -> str:
    """The failed line the wizard shows. This module's own exceptions (and the bare RuntimeErrors
    it raises about a file's size or arrival) already read as plain sentences; anything else,
    a Hub HTTP error with its URL and request id above all, is logged and replaced with a
    short reason (review item 11)."""
    if isinstance(exc, _OWN_WORDS) or (type(exc) is RuntimeError):
        return str(exc)
    logger.warning("a model download failed: %s: %s", type(exc).__name__, exc)
    if _network_like(exc):
        return "The connection to Hugging Face was lost, so the download stopped."
    if isinstance(exc, OSError):
        return f"The download stopped: {exc.strerror or exc}."
    return "Something went wrong during the download. Check the helper log for the details."


class DownloadBody(BaseModel):
    repo: str = Field(min_length=1)


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int | None = None
    sha256: str | None = None


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
    # Add model wizard jobs (kind "add"): staging folder, final target, what to fetch.
    kind: str = "folder"
    engine: str | None = None
    staging: Path | None = None
    target: Path | None = None
    fetch: list[RemoteFile] = field(default_factory=list)
    finals: list[str] = field(default_factory=list)
    sums_name: str | None = None
    verified: list[str] = field(default_factory=list)
    # SHA256SUMS rows read at plan time (name -> hex), and what actually checked each file:
    # "SHA256SUMS", "published" (the Hub's LFS digest) or None (nothing could).
    sums: dict[str, str] = field(default_factory=dict)
    checks: dict[str, str | None] = field(default_factory=dict)

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
            "kind": self.kind,
            "engine": self.engine,
            "verified": list(self.verified),
            "checks": dict(self.checks),
            "resultPath": str(self.target) if self.kind == "add" and self.stage == "done" and self.target else None,
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
        digest = _field(lfs, "sha256") if lfs is not None else None
        digest = digest.lower() if isinstance(digest, str) and _SHA_RE.fullmatch(digest.lower()) else None
        result.append(RemoteFile(name=name, size=size, sha256=digest))
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


def parse_sha256sums(text: str) -> dict[str, str]:
    """``<hex>  <name>`` or ``<hex> *<name>`` lines (sha256sum's text and binary forms)."""
    sums: dict[str, str] = {}
    for line in text.splitlines():
        match = _SUMS_LINE.match(line)
        if match:
            name = match.group(2)
            sums[name[2:] if name.startswith("./") else name] = match.group(1).lower()
    return sums


def sha256_file(path: str | os.PathLike[str], cancelled: Callable[[], bool] | None = None) -> str:
    """Hash a file; ``cancelled`` is polled once per 8 MiB so a 19 GB hash can be stopped too."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(_HASH_CHUNK):
            if cancelled is not None and cancelled():
                raise DownloadCancelled(str(path))
            digest.update(block)
    return digest.hexdigest()


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Rename ``source`` to ``destination``, refusing (FileExistsError) when the name is taken.

    ``os.rename`` onto an *empty* directory silently replaces it on Linux, so an exists() check
    followed by a rename leaves a window in which a folder created by the user is lost. Linux
    has renameat2(RENAME_NOREPLACE), which makes the refusal atomic; it is reached through
    ctypes because ``os`` has no wrapper. Windows ``os.rename`` refuses an existing target
    natively. Anywhere else without the flag (EINVAL on some FUSE and 9p mounts, ENOSYS on old
    kernels, other POSIX systems) a folder is placed by ``_move_into_new_folder``: an atomic
    ``os.mkdir`` claims the name, then each file goes in without overwriting. A check-then-
    rename was used here before, and a folder made in between was replaced (review, PR #17).
    """
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = libc.renameat2
        except (OSError, AttributeError):
            renameat2 = None
        if renameat2 is not None:
            renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
            renameat2.restype = ctypes.c_int
            result = renameat2(_AT_FDCWD, os.fsencode(source), _AT_FDCWD, os.fsencode(destination), _RENAME_NOREPLACE)
            if result == 0:
                return
            code = ctypes.get_errno()
            if code in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(code, os.strerror(code), str(destination))
            if code not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise OSError(code, os.strerror(code), str(source), None, str(destination))
    if sys.platform == "win32":
        os.rename(source, destination)  # refuses an existing name (FileExistsError) by itself
        return
    if source.is_dir() and not source.is_symlink():
        _move_into_new_folder(source, destination)
        return
    _claim_file(source, destination)
    source.unlink(missing_ok=True)


def _claim_file(source: Path, destination: Path) -> None:
    """Put one file at ``destination`` without ever replacing what is there.

    ``os.link`` refuses an existing name (a plain rename would silently replace it). A
    filesystem that cannot hard-link (EPERM on WSL's drvfs mounts, EXDEV across volumes)
    gets the same guarantee from an exclusive create of the name followed by a replace of
    that placeholder, which is ours. With a hard link, ``source`` is left for the caller.
    """
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)  # FileExistsError if taken
        os.close(fd)
        try:
            os.replace(source, destination)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


def _move_into_new_folder(source: Path, destination: Path) -> None:
    """Move folder ``source`` to the new name ``destination`` with no overwrite-capable rename:
    ``os.mkdir`` claims the name atomically (FileExistsError if it is taken), each file is
    linked or exclusively created inside it, and anything that clashes puts everything back."""
    os.mkdir(destination)
    placed: list[tuple[Path, Path]] = []  # (source file, placed file)
    folders: list[Path] = [destination]
    try:
        for root, dirs, names in os.walk(source):
            here = destination / Path(root).relative_to(source)
            for name in sorted(dirs):
                if (Path(root) / name).is_symlink():
                    names.append(name)  # a link to a folder is moved as a link, never walked
                    continue
                os.mkdir(here / name)
                folders.append(here / name)
            dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
            for name in sorted(names):
                src, dst = Path(root) / name, here / name
                if src.is_symlink():
                    os.symlink(os.readlink(src), dst)  # FileExistsError if the name was taken
                else:
                    _claim_file(src, dst)
                placed.append((src, dst))
    except BaseException:
        for src, dst in reversed(placed):
            try:
                if src.exists() or src.is_symlink():
                    dst.unlink()
                else:
                    os.replace(dst, src)
            except OSError:
                pass
        for folder in reversed(folders):
            try:
                folder.rmdir()  # only when empty: whatever someone else put there stays
            except OSError:
                pass
        raise
    shutil.rmtree(source, ignore_errors=True)


class _DropTokenAcrossHosts(HTTPRedirectHandler):
    """Forward a Hub redirect (to its CDN) without the bearer token, as huggingface_hub does."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - stdlib signature
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlsplit(newurl).hostname != urlsplit(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


def _hub_file_url(repo: str, filename: str) -> str:
    try:
        from huggingface_hub import hf_hub_url  # honours HF_ENDPOINT; lazy like the other Hub imports
    except ImportError:
        return f"https://huggingface.co/{repo}/resolve/main/{quote(filename)}"
    return hf_hub_url(repo_id=repo, filename=filename)


def _copy_stream(response: Any, destination: Path, cancelled: Callable[[], bool]) -> None:
    """Copy an open response to ``destination``, polling ``cancelled`` before every chunk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as fh:
        while True:
            if cancelled():
                raise DownloadCancelled(destination.name)
            block = response.read(_STREAM_CHUNK)
            if not block:
                return
            fh.write(block)


def _stream_hub_file(
    repo: str, filename: str, destination: str | os.PathLike[str], cancelled: Callable[[], bool], *,
    opener: Callable[..., Any] | None = None,
) -> None:
    """The wizard's file fetch: one plain HTTP GET of the Hub's resolve URL, written in chunks.

    Why not snapshot_download: it cannot be interrupted, so a single 19 GB .ninfer could only
    be cancelled once it had finished (and a cancel that landed after the last file ended
    "done"). Running it in a subprocess and killing it would interrupt it, but could not carry
    the test doubles and would leave the Hub cache's partial files to clean. A streamed copy is
    forty lines, polls the cancel flag per 4 MiB, and the wizard already checks size and sha256
    itself and deletes the staging folder on any failure, so the Hub client's resume and retry
    logic adds nothing here.
    """
    request = Request(_hub_file_url(repo, filename), headers={"User-Agent": "freetoken-settings"})
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    open_url = opener if opener is not None else build_opener(_DropTokenAcrossHosts()).open
    with open_url(request, timeout=_STREAM_TIMEOUT_S) as response:
        _copy_stream(response, Path(destination), cancelled)


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
        file_fetcher: Callable[..., Any] | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.downloads_dir = Path(downloads_dir) if downloads_dir is not None else self.models_dir
        self._api_factory = api_factory
        self._config_fetcher = config_fetcher
        self._snapshot_downloader = snapshot_downloader or _default_snapshot_download
        # The Add model wizard fetches one file at a time through this seam:
        # ``fetcher(repo, filename, destination, cancelled)``; see _stream_hub_file.
        self._file_fetcher = file_fetcher or _stream_hub_file
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
        # A helper restart mid-download leaves the worker's staging folder behind (its cleanup
        # runs in the worker thread). No job exists yet, so every staging folder is an orphan.
        self.sweep_staging(self.models_dir, self.downloads_dir)

    def sweep_staging(self, *roots: str | os.PathLike[str]) -> list[Path]:
        """Delete every ``.incoming-download-*`` folder under ``roots`` that no live job owns."""
        with self._lock:
            owned = {
                self._target_key(job.staging)
                for job in self._jobs.values()
                if job.staging is not None and job.stage not in _TERMINAL_STAGES
            }
        removed: list[Path] = []
        for root in {Path(root) for root in roots}:
            try:
                entries = list(root.iterdir())
            except OSError:
                continue
            for entry in entries:
                if not entry.name.startswith(_STAGING_GLOB) or entry.is_symlink() or not entry.is_dir():
                    continue
                if self._target_key(entry) in owned:
                    continue
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(entry)
        return removed

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
            # Dot-named folders are the wizard's staging (and any other hidden folder), not models.
            folders = sorted(
                (item for item in self.models_dir.iterdir() if item.is_dir() and not item.name.startswith(".")),
                key=lambda p: p.name.lower(),
            )
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
                # Folder jobs: snapshot_download cannot be interrupted mid-file, so the worker
                # sees this flag before the next file and leaves the partial target for a later
                # resume.  Wizard jobs stream each file themselves and poll the flag per chunk,
                # per hash chunk and once more before moving (see _run_add).
                job.cancel_requested = True
            self._refresh(job)
            return job

    def _refresh(self, job: DownloadJob) -> None:
        if job.kind == "add":
            # Staging is gone once the files are placed (or deleted); count from the plan then.
            if job.stage in ("verifying", "moving", "done"):
                job.received_bytes = job.total_bytes
            else:
                received = _folder_size(job.staging) if job.staging is not None else 0
                job.received_bytes = min(received, job.total_bytes) if job.total_bytes else received
        else:
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
            if job.kind != "add":
                # Wizard jobs never leave a resumable partial target: their staging folder
                # is deleted instead (see _abandon).
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
                    if item.name.endswith('.safetensors'):
                        release_completed_file(job.target_folder / item.name)
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
                for path in job.target_folder.glob('*.safetensors'):
                    release_completed_file(path)

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

    # ---- Add model wizard (control panel Stage B) -------------------------

    def _fetch_file(self, repo: str, filename: str, destination: Path, cancelled: Callable[[], bool]) -> None:
        self._file_fetcher(repo, filename, destination, cancelled)

    def _read_sums(self, repo: str, sums: RemoteFile | None) -> dict[str, str]:
        """The repo's SHA256SUMS rows, read at plan time so each file's label says what will
        actually check it (a file the list leaves out falls back to the Hub digest or nothing)."""
        if sums is None:
            return {}
        with tempfile.TemporaryDirectory(prefix="freetoken-sums-") as temporary:
            path = Path(temporary) / sums.name
            self._fetch_file(repo, sums.name, path, lambda: False)
            return parse_sha256sums(path.read_text(encoding="utf-8", errors="replace"))

    @staticmethod
    def _check_label(item: RemoteFile, sums: dict[str, str]) -> str | None:
        if item.name in sums:
            return "SHA256SUMS"
        return "published" if item.sha256 else None

    def _plan_add(
        self, value: str, entry: str | None, folder_root: str | os.PathLike[str], ninfer_root: str | os.PathLike[str]
    ) -> tuple[dict[str, Any], list[RemoteFile], RemoteFile | None, dict[str, str]]:
        repo = parse_repo(value)
        _, name = repo.split("/", 1)
        listing = [item for item in self._hub_files(repo) if _is_top_level_file(item.name)]
        by_name = {item.name: item for item in listing}
        sums = next((by_name[n] for n in _SUMS_NAMES if n in by_name), None)
        entries = sorted(n for n in by_name if _NINFER_ENTRY.fullmatch(n))
        architecture = None
        if entries:
            if entry is not None and entry not in entries:
                raise InvalidRepository(f"{entry} is not in that repo.")
            engine, root = "ninfer", Path(ninfer_root)
            chosen = entry or (entries[0] if len(entries) == 1 else None)
            parts = sorted(n for n in by_name if (m := _NINFER_PART.match(n)) and m.group("entry") == chosen)
            fetch = [by_name[chosen], *(by_name[n] for n in parts)] if chosen else []
            finals = [chosen, *parts] if chosen else []
            target = root / chosen if chosen else None
        elif "config.json" in by_name:
            info = describe_config(self._config(repo), name)
            if not info.supported:
                raise AddUnsupported(
                    f"This model's design ({info.architecture or 'not named in its config.json'}) "
                    "is not supported by your engines."
                )
            engine, root, chosen, architecture = "freetoken", Path(folder_root), None, info.architecture
            fetch, finals, target = _downloadable_files(listing), [name], Path(folder_root) / name
        else:
            raise AddUnsupported(
                "This repo has no NInfer file (.ninfer) and no model folder (config.json), "
                "so it is not supported by your engines."
            )
        rows = self._read_sums(repo, sums) if fetch else {}
        total = sum(item.size or 0 for item in fetch)
        try:
            free = max(0, int(self._disk_free(root)))
        except (TypeError, ValueError, OSError):
            free = 0
        plan = {
            "repo": repo,
            "name": name,
            "engine": engine,
            "kind": "ninfer" if engine == "ninfer" else "folder",
            "entries": entries,
            "entry": chosen,
            "architecture": architecture,
            "files": [
                {
                    "name": item.name,
                    "bytes": int(item.size or 0),
                    "check": self._check_label(item, rows),
                }
                for item in fetch
            ],
            "sumsFile": sums.name if sums is not None else None,
            "totalBytes": int(total),
            "root": str(root),
            "target": str(target) if target else None,
            "finals": finals,
            "exists": any((root / final).exists() for final in finals),
            "diskFreeBytes": free,
            "diskFits": free >= total,
        }
        return plan, fetch, sums, rows

    def plan_add(
        self, value: str, entry: str | None = None, *, folder_root: str | os.PathLike[str],
        ninfer_root: str | os.PathLike[str],
    ) -> dict[str, Any]:
        """Describe what an Add-model download would fetch and where it would land."""
        return self._plan_add(value, entry, folder_root, ninfer_root)[0]

    def start_add(
        self, value: str, entry: str | None = None, *, folder_root: str | os.PathLike[str],
        ninfer_root: str | os.PathLike[str],
    ) -> DownloadJob:
        plan, fetch, sums, rows = self._plan_add(value, entry, folder_root, ninfer_root)
        if not fetch:
            raise InvalidRepository("Pick which NInfer file to download.")
        if plan["exists"]:
            raise DownloadConflict("It is already on this PC, so it was not downloaded again. Add it from “On this PC”.")
        if not plan["diskFits"]:
            raise DownloadConflict(
                f"Not enough drive space: it needs {plan['totalBytes'] / GIB:.1f} GB and "
                f"{plan['diskFreeBytes'] / GIB:.1f} GB is free."
            )
        with self._lock:
            if self._active_id is not None:
                active = self._jobs.get(self._active_id)
                raise DownloadConflict(f"Another model download is already {active.stage if active else 'running'}.")
            # Orphans from a helper restart go now, while it is certain no job owns them.
            self.sweep_staging(folder_root, ninfer_root)
            job_id = f"download-{uuid.uuid4().hex[:12]}"
            # The dot keeps the staging folder out of the page's Browse list.
            staging = Path(plan["root"]) / f"{_STAGING_PREFIX}{job_id}"
            job = DownloadJob(
                job_id=job_id,
                repo=plan["repo"],
                target_folder=staging,
                total_bytes=plan["totalBytes"],
                kind="add",
                engine=plan["engine"],
                staging=staging,
                target=Path(plan["target"]),
                fetch=list(fetch),
                finals=plan["finals"],
                sums_name=sums.name if sums is not None else None,
                sums=dict(rows),
            )
            self._jobs[job_id] = job
            self._active_id = job_id
            threading.Thread(
                target=self._run_add, args=(job_id,), name=f"settings-add-{job_id[-6:]}", daemon=True
            ).start()
            return job

    def latest_add(self) -> dict[str, Any] | None:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.kind == "add"]
            if not jobs:
                return None
            self._refresh(jobs[-1])
            return jobs[-1].as_dict()

    def _abandon(self, job_id: str, stage: str, error: str | None = None) -> None:
        """Spec error table: a failed or cancelled download leaves no partial files."""
        job = self._jobs[job_id]
        if job.staging is not None:
            shutil.rmtree(job.staging, ignore_errors=True)
        self._finish(job_id, stage, error)

    def _run_add(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
        self._set_stage(job_id, "downloading")

        def cancelled() -> bool:
            return self._cancelled(job_id)

        try:
            job.staging.mkdir(parents=True)
            for item in job.fetch:
                if cancelled():
                    raise DownloadCancelled(item.name)
                path = job.staging / item.name
                self._fetch_file(job.repo, item.name, path, cancelled)
                if not path.is_file():
                    raise RuntimeError(f"{item.name} did not arrive from Hugging Face.")
                if item.size is not None and path.stat().st_size != item.size:
                    raise RuntimeError(
                        f"{item.name} arrived with the wrong size ({path.stat().st_size:,} bytes, "
                        f"expected {item.size:,})."
                    )
                if item.name.endswith((".safetensors", ".ninfer")) or _NINFER_PART.match(item.name):
                    release_completed_file(path)
                with self._lock:
                    job.files.append(item.name)
            self._set_stage(job_id, "verifying")
            # The repo's own list wins over the Hub's published LFS digests; a file in neither is
            # labelled so (None) rather than claimed as checked.
            expected: dict[str, tuple[str, str]] = {item.name: ("published", item.sha256) for item in job.fetch if item.sha256}
            expected.update({name: ("SHA256SUMS", digest) for name, digest in job.sums.items()})
            for item in job.fetch:
                source, want = expected.get(item.name, (None, None))
                with self._lock:
                    job.checks[item.name] = source
                if want is None:
                    continue
                if sha256_file(job.staging / item.name, cancelled) != want:
                    raise ChecksumMismatch(
                        f"{item.name} does not match the checksum the repo publishes, so the download was deleted."
                    )
                with self._lock:
                    job.verified.append(item.name)
            # A cancel that landed after the last chunk (or during a hash-free verify) must not
            # end "done": this is the last look before anything leaves the staging folder.
            if cancelled():
                raise DownloadCancelled("moving")
            self._set_stage(job_id, "moving")
            self._place(job)
            self._finish(job_id, "done")
        except DownloadCancelled:
            self._abandon(job_id, "cancelled")
        except Exception as exc:  # noqa: BLE001 - a failed download must not kill the helper
            self._abandon(job_id, "failed", _plain_job_error(exc))

    @staticmethod
    def _claim_and_move(source: Path, destination: Path) -> None:
        """Put one file at ``destination`` without ever replacing what is there (_claim_file)."""
        _claim_file(source, destination)

    @classmethod
    def _place(cls, job: DownloadJob) -> None:
        """Move the checked files into place without ever overwriting: a folder is renamed only
        when its name is free; a file is linked, which fails when the name exists. Anything
        already placed is taken back if a later file cannot be."""
        staging, target = job.staging, job.target
        if job.engine == "freetoken":
            shutil.rmtree(staging / ".cache", ignore_errors=True)
            try:
                _rename_noreplace(staging, target)
            except FileExistsError as exc:
                raise DownloadConflict(
                    f"{target.name} appeared in {target.parent} while downloading; nothing was overwritten."
                ) from exc
            return
        placed: list[Path] = []
        try:
            for name in job.finals:
                destination = target.parent / name
                try:
                    cls._claim_and_move(staging / name, destination)
                except FileExistsError as exc:
                    raise DownloadConflict(
                        f"{name} appeared in {destination.parent} while downloading; nothing was overwritten."
                    ) from exc
                placed.append(destination)
        except BaseException:
            for path in placed:
                path.unlink(missing_ok=True)
            raise
        shutil.rmtree(staging, ignore_errors=True)


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
    file_fetcher: Callable[..., Any] | None = None,
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
            file_fetcher=file_fetcher,
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
    "AddUnsupported",
    "ChecksumMismatch",
    "DownloadBody",
    "DownloadCancelled",
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
    "parse_sha256sums",
    "pc_memory_bytes",
    "physical_memory_bytes",
    "sha256_file",
]

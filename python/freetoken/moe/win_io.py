"""Unbuffered (cache-bypassing) file reads on Windows -- the ``O_DIRECT`` stand-in.

The expert-bank load reads the whole checkpoint (tens of GiB) once, straight into
pinned host banks that then live for the process lifetime. On POSIX that read uses
``O_DIRECT`` (plus ``posix_fadvise(POSIX_FADV_DONTNEED)`` around each shard), so the
page cache never grows and the banks are the only resident copy. Windows has neither
call: the cache manager parks every shard page on the standby list, so a 63 GiB
checkpoint pushes whole-system RAM to the edge exactly while 63 GiB of banks are being
pinned.

There is no reliable *retroactive* purge on Windows -- ``FILE_FLAG_NO_BUFFERING`` is an
**open-mode** flag -- so the fix is "never cache" rather than "evict after the fact":
open with ``CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN)`` and read
at absolute offsets with ``ReadFile`` plus an ``OVERLAPPED`` offset.

Alignment is the whole difficulty. ``FILE_FLAG_NO_BUFFERING`` requires the file offset,
the byte count **and** the destination address to each be a multiple of the volume's
sector size. A safetensors data offset is arbitrary, a bank row's address is arbitrary,
and the final chunk of a file ends mid-sector, so anything that does not line up is
routed through a page-aligned bounce buffer and copied into place -- the same shape as
the POSIX :func:`freetoken.moe.host_banks.read_range_into`.

``FREETOKEN_WIN_UNBUFFERED_IO=0`` turns the whole thing off and restores the buffered
(cached) reads. Callers are expected to fall back to buffered IO if a handle cannot be
opened unbuffered at all (network shares and some virtual filesystems refuse it).
"""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor

_BLK = 4096  # the alignment every caller already guarantees (page size)
_DEFAULT_CHUNK = 8 << 20
_MAX_READ = 1 << 30  # ReadFile's byte count is a DWORD; stay well inside it
_ENV = "FREETOKEN_WIN_UNBUFFERED_IO"

_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3
_FILE_FLAG_NO_BUFFERING = 0x20000000
_FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
_ERROR_HANDLE_EOF = 38
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


if os.name == "nt":
    from ctypes import wintypes

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.ReadFile.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_OVERLAPPED),
    ]
    _k32.ReadFile.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
    _k32.GetDiskFreeSpaceW.argtypes = [wintypes.LPCWSTR] + [ctypes.POINTER(wintypes.DWORD)] * 4
    _k32.GetDiskFreeSpaceW.restype = wintypes.BOOL
else:  # importable everywhere; every entry point below is inert off Windows
    _OVERLAPPED = None
    _k32 = None


def supported() -> bool:
    """True where the unbuffered reader can run at all (Windows with kernel32 reachable)."""
    return _k32 is not None


def enabled() -> bool:
    """:func:`supported` and not disabled by ``FREETOKEN_WIN_UNBUFFERED_IO=0``.

    The escape hatch restores the pre-fix behaviour (cached reads via mmap/safetensors)
    for debugging or for a volume where unbuffered IO misbehaves."""
    if not supported():
        return False
    return os.environ.get(_ENV, "").strip().lower() not in ("0", "false", "no", "off")


_sector_cache: dict[str, int] = {}


def sector_size(path: str) -> int:
    """Volume sector size for ``path`` -- the alignment ``FILE_FLAG_NO_BUFFERING`` demands.

    Falls back to :data:`_BLK` when the volume cannot be queried (UNC paths, substituted
    drives); callers only ever align to :data:`_BLK`, and :class:`UnbufferedReader`
    refuses to run unbuffered if the reported sector does not divide it."""
    if _k32 is None:
        return _BLK
    root = os.path.splitdrive(os.path.abspath(path))[0]
    if not root:
        return _BLK
    root += os.sep
    cached = _sector_cache.get(root)
    if cached is not None:
        return cached
    spc = wintypes.DWORD(0)
    bps = wintypes.DWORD(0)
    free = wintypes.DWORD(0)
    total = wintypes.DWORD(0)
    ok = _k32.GetDiskFreeSpaceW(root, ctypes.byref(spc), ctypes.byref(bps),
                                ctypes.byref(free), ctypes.byref(total))
    sec = bps.value if ok and bps.value > 0 else _BLK
    _sector_cache[root] = sec
    return sec


def _addr(mv: memoryview) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


def _oserror(what: str) -> OSError:
    err = ctypes.get_last_error()
    return OSError(0, f"{what}: {ctypes.FormatError(err).strip()}", None, err)


class UnbufferedReader:
    """A file opened ``FILE_FLAG_NO_BUFFERING``, read at absolute offsets.

    One handle **per thread** (created lazily): a handle opened without
    ``FILE_FLAG_OVERLAPPED`` serializes its IO, so concurrent readers each need their
    own -- which also keeps every ``ReadFile`` synchronous and removes the whole
    IO-pending/event dance. The offset still travels in an ``OVERLAPPED``, so nothing
    depends on a shared file pointer.

    Use it as a context manager (or call :meth:`close`) -- the handles are process
    resources, not garbage collected promptly.
    """

    __slots__ = ("path", "size", "sector", "_local", "_handles", "_lock", "_closed")

    def __init__(self, path: str, *, sector: int | None = None) -> None:
        if _k32 is None:
            raise OSError("unbuffered reads need Windows")
        self.path = os.fspath(path)
        self.size = os.path.getsize(self.path)
        self.sector = sector or sector_size(self.path)
        if self.sector <= 0 or _BLK % self.sector:
            raise OSError(
                f"volume sector size {self.sector} does not divide the {_BLK}-byte "
                f"alignment the bank buffers guarantee ({self.path})"
            )
        self._local = threading.local()
        self._handles: list[int] = []
        self._lock = threading.Lock()
        self._closed = False
        self._handle()  # open once up front so an unsupported volume fails here, not mid-read

    # -- handles ------------------------------------------------------------------

    def _handle(self) -> int:
        h = getattr(self._local, "handle", None)
        if h is not None:
            return h
        if self._closed:
            raise ValueError("reader is closed")
        h = _k32.CreateFileW(
            self.path, _GENERIC_READ, _FILE_SHARE_READ | _FILE_SHARE_WRITE, None,
            _OPEN_EXISTING, _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_SEQUENTIAL_SCAN, None,
        )
        if h is None or h == _INVALID_HANDLE_VALUE:
            raise _oserror(f"CreateFileW(FILE_FLAG_NO_BUFFERING) {self.path}")
        self._local.handle = h
        with self._lock:
            self._handles.append(h)
        return h

    def close(self) -> None:
        with self._lock:
            handles, self._handles = self._handles, []
            self._closed = True
        for h in handles:
            _k32.CloseHandle(h)
        self._local = threading.local()

    def __enter__(self) -> "UnbufferedReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- raw reads ----------------------------------------------------------------

    def _read_at(self, addr: int, nbytes: int, offset: int) -> int:
        """One ``ReadFile`` at an absolute offset; returns the byte count (short at EOF)."""
        ov = _OVERLAPPED()
        ov.Offset = offset & 0xFFFFFFFF
        ov.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
        got = wintypes.DWORD(0)
        ok = _k32.ReadFile(self._handle(), ctypes.c_void_p(addr), nbytes,
                           ctypes.byref(got), ctypes.byref(ov))
        if not ok:
            if ctypes.get_last_error() == _ERROR_HANDLE_EOF:
                return 0
            raise _oserror(f"ReadFile({nbytes} @ {offset}) {self.path}")
        return got.value

    def _read_span(self, addr: int, nbytes: int, offset: int) -> int:
        """Fill ``nbytes`` (sector-aligned offset/count/address) or stop at EOF.

        Returns the bytes actually read. A short count means EOF -- the last sector of a
        file is allowed to come back partially filled even though the request had to be a
        whole sector -- and a continuation read could not stay aligned anyway, so the loop
        stops there and the caller decides whether it got enough."""
        done = 0
        while done < nbytes:
            want = min(nbytes - done, _MAX_READ)  # ReadFile's count is a DWORD
            n = self._read_at(addr + done, want, offset + done)
            done += n
            if n < want:
                break  # EOF (or a driver that cannot continue); the caller checks the total
        return done

    def _bounce(self, span: int) -> memoryview:
        """Per-thread page-aligned scratch (anonymous mmaps are page aligned)."""
        buf = getattr(self._local, "bounce", None)
        if buf is None or len(buf) < span:
            buf = mmap.mmap(-1, max(span, self.sector))
            self._local.bounce = buf
            self._local.bounce_mv = memoryview(buf)
        return self._local.bounce_mv

    # -- public API ---------------------------------------------------------------

    def read_into(self, dst: memoryview | mmap.mmap | bytearray, file_offset: int,
                  nbytes: int) -> int:
        """Read ``self.path[file_offset : file_offset + nbytes]`` into the front of ``dst``.

        Whatever is already sector-aligned (offset, count and destination address) is
        DMA'd straight into ``dst``; the unaligned head/tail -- and the whole read when
        the destination address itself is skewed -- goes through the page-aligned bounce
        and is copied into place. Bytes past ``nbytes`` in ``dst`` are never touched."""
        if nbytes == 0:
            return 0
        mv = (dst if isinstance(dst, memoryview) else memoryview(dst)).cast("B")
        if len(mv) < nbytes:
            raise ValueError(f"destination holds {len(mv)} bytes, need {nbytes}")
        if file_offset < 0 or file_offset + nbytes > self.size:
            raise OSError(
                f"read of {nbytes} bytes at {file_offset} runs past the end of "
                f"{self.path} ({self.size} bytes)"
            )
        sec = self.sector
        if file_offset % sec == 0 and _addr(mv) % sec == 0:
            whole = (nbytes // sec) * sec
            if whole and self._read_span(_addr(mv), whole, file_offset) != whole:
                raise OSError(f"short unbuffered read at {file_offset} in {self.path}")
            rest = nbytes - whole
            if rest:  # the trailing partial sector: read a full sector into the bounce
                bmv = self._bounce(sec)
                got = self._read_span(_addr(bmv[:sec]), sec, file_offset + whole)
                if got < rest:
                    raise OSError(f"short unbuffered read at {file_offset} in {self.path}")
                mv[whole:nbytes] = bmv[:rest]
            return nbytes
        head = file_offset % sec
        span = ((head + nbytes + sec - 1) // sec) * sec
        bmv = self._bounce(span)
        got = self._read_span(_addr(bmv[:span]), span, file_offset - head)
        if got < head + nbytes:
            raise OSError(
                f"short unbuffered read: {got} of {head + nbytes} bytes at "
                f"{file_offset - head} in {self.path}"
            )
        mv[:nbytes] = bmv[head:head + nbytes]
        return nbytes

    def read(self, file_offset: int, nbytes: int) -> bytes:
        """Convenience: the range as ``bytes`` (for headers and other small reads)."""
        if nbytes == 0:
            return b""
        buf = bytearray(nbytes)
        self.read_into(memoryview(buf), file_offset, nbytes)
        return bytes(buf)


def read_range_into(buf: memoryview | mmap.mmap, path: str, *, file_offset: int, nbytes: int,
                    dest_offset: int = 0, workers: int = 8,
                    chunk: int = _DEFAULT_CHUNK) -> int:
    """Chunked, multi-threaded unbuffered read of ``path[file_offset:+nbytes]`` into
    ``buf`` at ``dest_offset``. Windows counterpart of
    :func:`freetoken.moe.host_banks.read_range_into`; returns ``nbytes``."""
    mv = (buf if isinstance(buf, memoryview) else memoryview(buf)).cast("B")
    if dest_offset + nbytes > len(mv):
        raise ValueError(f"destination holds {len(mv)} bytes, need {dest_offset + nbytes}")
    if nbytes == 0:
        return 0
    with UnbufferedReader(path) as rd:
        offs = list(range(0, nbytes, chunk))

        def one(i: int) -> None:
            n = min(chunk, nbytes - i)
            rd.read_into(mv[dest_offset + i:dest_offset + i + n], file_offset + i, n)

        if len(offs) <= 1 or workers <= 1:
            for o in offs:
                one(o)
        else:
            with ThreadPoolExecutor(min(workers, len(offs))) as ex:
                list(ex.map(one, offs))
    return nbytes


def read_file_into(buf: memoryview | mmap.mmap, path: str, *, workers: int = 8,
                   chunk: int = _DEFAULT_CHUNK) -> int:
    """Whole-file counterpart of :func:`read_range_into`; returns the file size."""
    size = os.path.getsize(path)
    read_range_into(buf, path, file_offset=0, nbytes=size, workers=workers, chunk=chunk)
    return size


__all__ = [
    "UnbufferedReader",
    "enabled",
    "read_file_into",
    "read_range_into",
    "sector_size",
    "supported",
]

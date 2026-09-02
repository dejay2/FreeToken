"""Windows unbuffered (cache-bypassing) reader: alignment/bounce arithmetic and wiring.

The boot-time expert load reads tens of GiB of shards into pinned host banks. On POSIX
those reads use ``O_DIRECT`` so nothing lands in the page cache; on Windows the same
reads used to go through the cache manager and parked ~63 GiB on the standby list.
:mod:`freetoken.moe.win_io` is the Windows counterpart -- these tests pin down the part
that is easy to get wrong: the sector alignment of the file offset, the byte count and
the destination address, and the bounce buffer that covers every combination that is
not aligned (including the last partial sector at EOF).
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import struct

import pytest

from freetoken.moe import win_io

_NT = os.name == "nt"
nt_only = pytest.mark.skipif(not _NT, reason="Windows-only unbuffered reader")

_SEED = b"".join(bytes([(i * 37 + (i >> 8) * 11) & 0xFF]) for i in range(256))


def _make_file(tmp_path, size: int, name: str = "blob.bin") -> str:
    path = os.fspath(tmp_path / name)
    with open(path, "wb") as f:
        while size > 0:
            n = min(size, len(_SEED))
            f.write(_SEED[:n])
            size -= n
    return path


def _expected(path: str, offset: int, nbytes: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(nbytes)


def _addr(mv: memoryview) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


# --------------------------------------------------------------------------------------
# platform probes (run everywhere)
# --------------------------------------------------------------------------------------


def test_supported_matches_platform():
    assert win_io.supported() is _NT


def test_enabled_follows_env(monkeypatch):
    monkeypatch.setenv("FREETOKEN_WIN_UNBUFFERED_IO", "0")
    assert win_io.enabled() is False
    monkeypatch.setenv("FREETOKEN_WIN_UNBUFFERED_IO", "1")
    assert win_io.enabled() is _NT
    monkeypatch.delenv("FREETOKEN_WIN_UNBUFFERED_IO")
    assert win_io.enabled() is _NT  # on by default where it works


@nt_only
def test_sector_size_divides_the_block(tmp_path):
    sec = win_io.sector_size(_make_file(tmp_path, 4096))
    assert sec > 0
    assert win_io._BLK % sec == 0, f"sector {sec} does not divide the {win_io._BLK} block"


# --------------------------------------------------------------------------------------
# alignment / bounce arithmetic
# --------------------------------------------------------------------------------------

_SIZE = 4096 * 5 + 137  # neither a sector nor a page multiple: the tail always bounces

# (file_offset, nbytes) -- aligned, unaligned at one/both ends, sub-sector, EOF-spanning
_RANGES = [
    (0, 4096),                 # exactly aligned, one sector
    (0, 4096 * 3),             # exactly aligned, several sectors
    (4096, 4096),              # aligned offset and count, not at 0
    (1, 4096),                 # unaligned head, aligned length
    (0, 4095),                 # aligned head, unaligned tail
    (7, 9),                    # shorter than one sector, unaligned at both ends
    (4095, 2),                 # straddles a sector boundary
    (4096 * 2 + 3, 4096 * 2 + 11),  # unaligned at both ends, multi-sector
    (0, _SIZE),                # the whole file (last sector is partial)
    (_SIZE - 1, 1),            # the final byte
    (_SIZE - 4096, 4096),      # the last 4096 bytes, running to EOF
    (4096 * 5, _SIZE - 4096 * 5),   # aligned start, partial final sector
    (4096 * 5 + 1, _SIZE - 4096 * 5 - 1),  # unaligned start into the partial final sector
]


@nt_only
@pytest.mark.parametrize("offset,nbytes", _RANGES)
@pytest.mark.parametrize("dest_skew", [0, 1, 512, 4095])
def test_read_into_matches_a_normal_read(tmp_path, offset, nbytes, dest_skew):
    """Byte-exact against ``open().seek().read()`` for every alignment combination,
    with the destination address itself deliberately skewed off the sector."""
    path = _make_file(tmp_path, _SIZE)
    backing = mmap.mmap(-1, dest_skew + nbytes + 4096)
    dst = memoryview(backing)[dest_skew:dest_skew + nbytes]
    assert _addr(memoryview(backing)) % 4096 == 0  # anonymous mmaps are page aligned
    with win_io.UnbufferedReader(path) as rd:
        assert rd.size == _SIZE
        rd.read_into(dst, offset, nbytes)
    assert bytes(dst) == _expected(path, offset, nbytes)


@nt_only
@pytest.mark.parametrize("offset,nbytes", _RANGES)
def test_read_returns_bytes(tmp_path, offset, nbytes):
    path = _make_file(tmp_path, _SIZE)
    with win_io.UnbufferedReader(path) as rd:
        assert rd.read(offset, nbytes) == _expected(path, offset, nbytes)


@nt_only
def test_read_into_does_not_touch_bytes_past_the_range(tmp_path):
    """The bounce must never spill the rounded-up sector tail into the destination."""
    path = _make_file(tmp_path, _SIZE)
    backing = mmap.mmap(-1, 8192)
    mv = memoryview(backing)
    mv[:] = b"\xAA" * 8192
    with win_io.UnbufferedReader(path) as rd:
        rd.read_into(mv[100:100 + 33], 4090, 33)
    assert bytes(mv[100:133]) == _expected(path, 4090, 33)
    assert bytes(mv[:100]) == b"\xAA" * 100
    assert bytes(mv[133:8192]) == b"\xAA" * (8192 - 133)


@nt_only
def test_read_past_eof_raises(tmp_path):
    path = _make_file(tmp_path, _SIZE)
    dst = memoryview(bytearray(64))
    with win_io.UnbufferedReader(path) as rd:
        with pytest.raises(OSError):
            rd.read_into(dst, _SIZE - 8, 64)


@nt_only
@pytest.mark.parametrize("chunk", [4096, 1 << 16, 1 << 20])
@pytest.mark.parametrize("dest_offset", [0, 1, 4096, 4097])
def test_read_range_into_chunked_and_threaded(tmp_path, chunk, dest_offset):
    size = (1 << 20) + 777
    path = _make_file(tmp_path, size, "big.bin")
    offset, nbytes = 3, size - 5
    backing = mmap.mmap(-1, dest_offset + nbytes + 4096)
    got = win_io.read_range_into(
        backing, path, file_offset=offset, nbytes=nbytes, dest_offset=dest_offset,
        workers=4, chunk=chunk,
    )
    assert got == nbytes
    mv = memoryview(backing)
    assert bytes(mv[dest_offset:dest_offset + nbytes]) == _expected(path, offset, nbytes)


@nt_only
def test_read_range_into_rejects_a_short_destination(tmp_path):
    path = _make_file(tmp_path, _SIZE)
    with pytest.raises(ValueError):
        win_io.read_range_into(bytearray(16), path, file_offset=0, nbytes=32)


@nt_only
@pytest.mark.parametrize("size", [1, 4095, 4096, 4097, (1 << 20) + 13])
def test_read_file_into_whole_file(tmp_path, size):
    path = _make_file(tmp_path, size, f"whole-{size}.bin")
    asize = ((size + 4095) // 4096) * 4096
    backing = mmap.mmap(-1, asize)
    assert win_io.read_file_into(backing, path, workers=4, chunk=1 << 16) == size
    assert bytes(memoryview(backing)[:size]) == _expected(path, 0, size)


@nt_only
def test_reader_is_reusable_across_threads(tmp_path):
    """One reader serves many worker threads (a handle per thread), byte-exact."""
    from concurrent.futures import ThreadPoolExecutor

    size = (1 << 20) + 9
    path = _make_file(tmp_path, size, "threads.bin")
    backing = mmap.mmap(-1, size + 4096)
    mv = memoryview(backing)
    step = 4093  # deliberately not a sector multiple: every chunk bounces
    with win_io.UnbufferedReader(path) as rd:
        def one(o):
            n = min(step, size - o)
            rd.read_into(mv[o:o + n], o, n)

        with ThreadPoolExecutor(8) as ex:
            list(ex.map(one, range(0, size, step)))
    assert bytes(mv[:size]) == _expected(path, 0, size)


# --------------------------------------------------------------------------------------
# safetensors shard reader built on it
# --------------------------------------------------------------------------------------


def _write_safetensors(path: str, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, blob, off = {}, bytearray(), 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        blob += data
        off += len(data)
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(bytes(blob))


@pytest.mark.parametrize("whole", [True, False])
def test_direct_shard_matches_safetensors(tmp_path, whole):
    """``DirectShard`` returns the same bytes/dtype/shape as safetensors' own reader.

    Runs on every platform: the reader falls back to O_DIRECT (POSIX) or a plain
    buffered read when the unbuffered Windows path is unavailable."""
    import safetensors
    import torch
    from freetoken.models.weight import DirectShard

    path = os.fspath(tmp_path / "shard.safetensors")
    payload = {
        "a.weight": ("U8", [3, 7], bytes(range(21))),
        "b.scale": ("F32", [2], struct.pack("<2f", 1.5, -2.25)),
        "c.big": ("U8", [9000], bytes(i & 0xFF for i in range(9000))),
        "d.scalar": ("F32", [], struct.pack("<f", 7.0)),
    }
    _write_safetensors(path, payload)
    with safetensors.safe_open(path, framework="pt", device="cpu") as ref, \
            DirectShard(path, whole=whole) as shard:
        assert set(shard.keys()) == set(payload)
        for name, (_dtype, shape, _data) in payload.items():
            want, got = ref.get_tensor(name), shard.get_tensor(name)
            assert got.dtype == want.dtype, name
            assert list(got.shape) == shape == list(want.shape), name
            assert torch.equal(got, want), name


# --------------------------------------------------------------------------------------
# real checkpoint (Windows, opt-in)
# --------------------------------------------------------------------------------------


def _real_shard() -> str | None:
    """Smallest expert ``*.safetensors`` under ``FREETOKEN_TEST_WIN_IO_MODEL`` (read-only).

    Expert shards are what the boot path actually streams; any shard will do otherwise."""
    import glob

    folder = os.environ.get("FREETOKEN_TEST_WIN_IO_MODEL", "").strip()
    if not folder or not os.path.isdir(folder):
        return None
    shards = glob.glob(os.path.join(folder, "*.safetensors"))
    experts = [p for p in shards if "expert" in os.path.basename(p)]
    pool = experts or shards
    return min(pool, key=os.path.getsize) if pool else None


@nt_only
@pytest.mark.slow
@pytest.mark.needs_weights
def test_real_shard_ranges_match_a_buffered_read():
    path = _real_shard()
    if path is None:
        pytest.skip("set FREETOKEN_TEST_WIN_IO_MODEL to a checkpoint directory")
    size = os.path.getsize(path)
    spots = [0, 1, 4095, 4096, 1 << 20, size // 2, size // 2 + 3, size - 8193, size - 1]
    with win_io.UnbufferedReader(path) as rd:
        for off in spots:
            off = max(0, min(off, size - 1))
            n = min(65537, size - off)
            assert rd.read(off, n) == _expected(path, off, n), off


@nt_only
@pytest.mark.slow
@pytest.mark.needs_weights
def test_real_shard_whole_read_matches_and_stays_out_of_the_cache():
    """Read a full shard unbuffered and check it byte-for-byte against a buffered read.

    With ``FREETOKEN_TEST_WIN_CACHE=1`` also assert the system standby cache barely
    moved (a buffered read of the same shard would add ~its whole size). Off by
    default: the counter is system-wide, so anything else doing IO makes it flaky.
    """
    path = _real_shard()
    if path is None:
        pytest.skip("set FREETOKEN_TEST_WIN_IO_MODEL to a checkpoint directory")
    size = os.path.getsize(path)
    asize = ((size + 4095) // 4096) * 4096
    before = _standby_bytes()
    backing = mmap.mmap(-1, asize)
    assert win_io.read_file_into(backing, path, workers=8) == size
    after = _standby_bytes()
    mv = memoryview(backing)
    with open(path, "rb") as f:
        pos = 0
        while pos < size:
            block = f.read(1 << 24)
            assert bytes(mv[pos:pos + len(block)]) == block, pos
            pos += len(block)
    if os.environ.get("FREETOKEN_TEST_WIN_CACHE", "").strip() == "1":
        assert before is not None and after is not None, "standby counter unavailable"
        assert after - before < size // 4, (
            f"standby cache grew {(after - before) / 2**20:.0f} MiB reading a "
            f"{size / 2**20:.0f} MiB shard unbuffered"
        )


@nt_only
@pytest.mark.slow
@pytest.mark.needs_weights
@pytest.mark.parametrize("whole", [True, False])
def test_real_shard_through_direct_shard_matches_safetensors(whole):
    """The actual boot path: every tensor of a real shard, read the way the nvfp4 loader
    now reads it, against safetensors' own mmap reader."""
    import safetensors
    import torch
    from freetoken.models.weight import DirectShard

    path = _real_shard()
    if path is None:
        pytest.skip("set FREETOKEN_TEST_WIN_IO_MODEL to a checkpoint directory")
    with safetensors.safe_open(path, framework="pt", device="cpu") as ref, \
            DirectShard(path, whole=whole) as shard:
        names = sorted(shard.keys())
        assert names == sorted(k for k in ref.keys())
        if not whole:  # range mode is per-tensor: a sample is enough, the shard is 300 MiB
            names = names[:24] + names[len(names) // 2:len(names) // 2 + 24] + names[-24:]
        for name in names:
            want, got = ref.get_tensor(name), shard.get_tensor(name)
            assert got.dtype == want.dtype and got.shape == want.shape, name
            assert torch.equal(got.view(torch.uint8) if got.dtype.itemsize == 1 else got,
                               want.view(torch.uint8) if want.dtype.itemsize == 1 else want), name


@nt_only
@pytest.mark.slow
@pytest.mark.needs_weights
@pytest.mark.parametrize("parallel", [False, True])
def test_nvfp4_layer0_banks_match_the_checkpoint(monkeypatch, parallel):
    """End-to-end on the real boot path: build layer 0's NVFP4 expert banks through the
    (now unbuffered) serial and parallel loaders and check sampled experts against
    safetensors. No GPU -- ``FREETOKEN_SKIP_BANK_PIN`` turns the pin-after-fill step off."""
    import dataclasses
    import random
    from types import SimpleNamespace

    import safetensors
    import torch
    from freetoken.models.nvfp4_banks import (
        load_nvfp4_expert_source_banks,
        load_nvfp4_expert_source_banks_parallel,
    )
    from freetoken.models.qwen4_exp.weight import _NVFP4_SOURCE_SPEC

    folder = os.environ.get("FREETOKEN_TEST_WIN_IO_MODEL", "").strip()
    if not folder or not os.path.isdir(os.path.join(folder)):
        pytest.skip("set FREETOKEN_TEST_WIN_IO_MODEL to the qwen4_exp NVFP4 checkpoint")
    if not os.path.isfile(os.path.join(folder, "model.safetensors.index.json")):
        pytest.skip("checkpoint has no safetensors index")
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")

    E, H, I = 512, 2560, 640
    spec = dataclasses.replace(
        _NVFP4_SOURCE_SPEC, layer_to_bank=lambda layer, config: 0 if layer == 0 else None
    )
    config = SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I,
                             num_moe_layers=1)
    load = load_nvfp4_expert_source_banks_parallel if parallel else load_nvfp4_expert_source_banks
    banks = load(folder, config, spec, drop_page_cache=lambda path: None, primary=False)

    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]

    def ref(key: str) -> "torch.Tensor":
        with safetensors.safe_open(os.path.join(folder, weight_map[key]),
                                   framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    lm = "model.language_model"
    for expert in random.Random(1).sample(range(E), 6):
        base = f"{lm}.layers.0.mlp.experts.{expert}"
        assert torch.equal(banks["gate_up_packed"][0][expert, :I], ref(f"{base}.gate_proj.weight"))
        assert torch.equal(banks["gate_up_packed"][0][expert, I:], ref(f"{base}.up_proj.weight"))
        assert torch.equal(banks["down_packed"][0][expert], ref(f"{base}.down_proj.weight"))
        for proj, bank, rows in (("gate_proj", "gate_up_scale", slice(0, I)),
                                 ("up_proj", "gate_up_scale", slice(I, 2 * I)),
                                 ("down_proj", "down_scale", slice(None))):
            scale = ref(f"{base}.{proj}.weight_scale")
            assert torch.equal(banks[bank][0][expert][rows].reshape(-1).view(torch.uint8),
                               scale.reshape(-1).view(torch.uint8)), (expert, proj)
        gate_g = ref(f"{base}.gate_proj.weight_scale_2").to(torch.float16)
        assert torch.equal(banks["gate_up_global"][0][expert, :I], gate_g.reshape(1).expand(I))


def _standby_bytes() -> int | None:
    """``\\Memory\\Standby Cache Normal Priority Bytes``, or None if unreadable."""
    if not _NT:
        return None
    import subprocess

    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Counter '\\Memory\\Standby Cache Normal Priority Bytes')"
             ".CounterSamples[0].CookedValue"],
            capture_output=True, text=True, timeout=60,
        )
        return int(float(out.stdout.strip()))
    except Exception:
        return None

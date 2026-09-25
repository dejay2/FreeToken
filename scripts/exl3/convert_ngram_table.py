"""One-time conversion of an EXL3 ``ngram_embedding.safetensors`` into FreeToken's table.

Usage (serving box): ``.venv/bin/python scripts/exl3/convert_ngram_table.py <model-dir>``
[--device cuda] [--chunk-rows 32768] [--dtype fp8|auto]

Writes ``freetoken-ple-000NN-of-000MM.safetensors`` (8 table shards per file) plus
``freetoken-ple.index.json`` into <model-dir>. turboderp's files are left untouched.
Never holds more than one table shard in host memory. Spec 2026-09-25 section 4.

Ruling R1 (2026-09-25 plan review): the converter must not squeeze the running box's memory.
The source file is opened as one whole-file mmap that stays open across all three passes
(opened once in ``convert_table``, closed only at the end), so a bare
``drop_page_cache(src)`` -- ``posix_fadvise(DONTNEED)`` on a *separate* fd -- is close to a
no-op on its own: Linux's ``invalidate_mapping_pages`` skips pages currently mapped into a
process's page tables, so a decoded shard's bytes stay resident until the mmap closes. After
every decode of a source table shard, :func:`_release_shard_pages` first
``mm.madvise(MADV_DONTNEED, ...)``s that shard's page-aligned byte range on the *live* mapping
to drop this process's page-table entries, and only then is ``drop_page_cache(src)`` able to
actually reclaim the underlying page cache. Output files get a plain ``drop_page_cache`` after
each write (no live mapping there to fight). fix round 1 (review finding, binding): the
original version only called ``drop_page_cache(src)`` per shard and never released the mmap's
own hold on the pages, so it could keep close to the full source table resident for the whole
run despite the per-shard docstring claim below.

The same ruling says: if the precision gate below would pick bf16 (median per-row relative RMS
error > ``_GATE``), stop after the report pass and do not write a bf16 table -- FreeToken's
loader only accepts F8_E4M3 (``weight.py:717``) and adding bf16 support to it is out of scope
for this task. :class:`Fp8GateExceeded` carries the finished report so the caller can inspect
the numbers without a stack trace.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys

import numpy as np
import safetensors.torch
import torch

from freetoken.models.loader import drop_page_cache
from freetoken.models.qwen4_exp.exl3_ngram import (
    dequant_rows, head_of_rows, mul1_codebook, words_per_row,
)

_FP8_MAX = 448.0
_SHARDS_PER_FILE = 8
_GATE = 0.05  # median per-row relative RMS error allowed for the fp8 table
_SAMPLE_ROWS = 1_000_000
_PAGE_SIZE = mmap.PAGESIZE


class Fp8GateExceeded(RuntimeError):
    """Raised when the precision gate would pick bf16 under an ``auto`` dtype choice.

    Ruling R1: FreeToken's PLE loader only serves F8_E4M3 tables (``weight.py:717``), so the
    converter refuses to write a bf16 table on its own; ``.report`` carries the finished
    report-pass numbers (median/p99/max relative RMS) for the caller to inspect and decide.
    """

    def __init__(self, report: dict):
        super().__init__(
            f"gate picked bf16 (median_rel_rms={report['median_rel_rms']:.4f} > {_GATE}); "
            f"refusing to write a bf16 PLE table -- FreeToken's loader only serves F8_E4M3. "
            f"Report: {json.dumps(report)}"
        )
        self.report = report


def _header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _small(path, header, base, key, dtype):
    meta = header[key]
    begin, end = meta["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(base + begin)
        raw = fh.read(end - begin)
    return torch.frombuffer(bytearray(raw), dtype=dtype).view(*meta["shape"]).clone()


def _find(header, suffix):
    keys = [k for k in header if k.endswith(suffix)]
    if len(keys) != 1:
        raise ValueError(f"expected one *{suffix} tensor, found {len(keys)}")
    return keys[0]


def _release_shard_pages(mm: mmap.mmap, base: int, meta: dict) -> None:
    """Drop a decoded shard's pages from the *live* source mmap (fix round 1, Ruling R1).

    ``mm`` stays open across all three passes, so its mapped pages pin the shard resident even
    after ``drop_page_cache(src)`` runs against a separate fd. ``madvise`` requires a
    page-aligned ``(start, length)``, so the shard's byte range is rounded out to page
    boundaries before the call; rounding out (never in) means this can only touch a
    neighbouring shard's not-yet-decoded boundary page, which is harmless -- it will simply be
    re-faulted in fresh when that shard's turn comes. Best-effort: ``madvise`` is unsupported on
    Windows, so any failure here is swallowed rather than failing the conversion over a
    memory-management hint.
    """
    begin, end = meta["data_offsets"]
    start = base + begin
    stop = base + end
    aligned_start = (start // _PAGE_SIZE) * _PAGE_SIZE
    aligned_len = ((stop - aligned_start + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE
    try:
        mm.madvise(mmap.MADV_DONTNEED, aligned_start, aligned_len)
    except (AttributeError, OSError, ValueError):
        pass


def _decode_shard(mm, base, meta, k, rows_start, head_offsets, head_bias, codebook, device, chunk):
    rows, words = meta["shape"]
    begin = base + meta["data_offsets"][0]
    packed_all = np.frombuffer(mm, dtype=np.int16, count=rows * words, offset=begin).reshape(rows, words)
    out = torch.empty((rows, 160), dtype=torch.float32)
    for lo in range(0, rows, chunk):
        hi = min(rows, lo + chunk)
        packed = torch.from_numpy(packed_all[lo:hi].copy()).to(device)
        ids = torch.arange(rows_start + lo, rows_start + hi, device=device)
        bias = head_bias.to(device)[head_of_rows(ids, head_offsets.to(device))]
        out[lo:hi] = dequant_rows(packed, k, codebook, bias).cpu()
    return out


def convert_table(model_dir: str, *, device: str = "cuda", chunk_rows: int = 32768,
                  out_dtype: str = "auto") -> dict:
    src = os.path.join(model_dir, "ngram_embedding.safetensors")
    header, base = _header(src)
    meta = header.pop("__metadata__", {})
    if meta.get("format") != "exl3_ngram_trellis" or meta.get("version") != "1":
        raise ValueError(f"{src} is not an exl3_ngram_trellis v1 table: {meta}")
    k = int(meta["K"])
    offsets_key = _find(header, "ngram_embedding.head_offsets")
    prefix = offsets_key[: -len(".head_offsets")]
    head_offsets = _small(src, header, base, offsets_key, torch.int64)
    head_vocab = _small(src, header, base, _find(header, "ngram_embedding.head_vocab_sizes"), torch.int64)
    multipliers = _small(src, header, base, _find(header, "ngram_embedding.layer_multipliers"), torch.int64)
    head_bias = _small(src, header, base, _find(header, "ngram_embedding.head_bias"), torch.float16)
    expect_offsets = torch.cat((torch.zeros(1, dtype=torch.int64), head_vocab.cumsum(0)[:-1]))
    if not torch.equal(head_offsets, expect_offsets):
        raise ValueError(f"head_offsets {head_offsets.tolist()} disagree with head_vocab_sizes")
    shard_keys = sorted((key for key in header if ".shard_" in key and key.endswith(".trellis")),
                        key=lambda key: int(key.rsplit(".shard_", 1)[1].split(".")[0]))
    if not shard_keys:
        raise ValueError(f"{src} has no shard_N.trellis tensors")
    shapes = {tuple(header[key]["shape"]) for key in shard_keys}
    if len(shapes) != 1 or next(iter(shapes))[1] != words_per_row(k):
        raise ValueError(f"table shards have shapes {shapes}; K={k} needs {words_per_row(k)} words")
    rows_per_shard = next(iter(shapes))[0]
    codebook = mul1_codebook(device)

    fd = os.open(src, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    try:
        # Pass 1: global absmax (for the one per-tensor fp8 scale FreeToken's loader expects,
        # weight.py _PLE_SCALE_SUFFIX) and the precision sample.
        absmax = 0.0
        for i, key in enumerate(shard_keys):
            dec = _decode_shard(mm, base, header[key], k, i * rows_per_shard, head_offsets,
                                head_bias, codebook, device, chunk_rows)
            absmax = max(absmax, float(dec.abs().max()))
            _release_shard_pages(mm, base, header[key])  # R1 fix round 1: unmap before fadvise
            drop_page_cache(src)  # R1: never let the checkpoint linger in page cache mid-run
        scale = absmax / _FP8_MAX if absmax > 0 else 1.0
        sample_every = max(1, (rows_per_shard * len(shard_keys)) // _SAMPLE_ROWS)
        rels = []
        for i, key in enumerate(shard_keys):
            dec = _decode_shard(mm, base, header[key], k, i * rows_per_shard, head_offsets,
                                head_bias, codebook, device, chunk_rows)[::sample_every]
            _release_shard_pages(mm, base, header[key])  # R1 fix round 1: unmap before fadvise
            drop_page_cache(src)  # R1
            fp8 = (dec / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).float() * scale
            denom = dec.pow(2).mean(1).sqrt().clamp_min(1e-12)
            rels.append((fp8 - dec).pow(2).mean(1).sqrt() / denom)
        rel = torch.cat(rels)
        median = float(rel.median())
        dtype = out_dtype if out_dtype != "auto" else ("fp8" if median <= _GATE else "bf16")

        report = {
            "dtype": "F8_E4M3" if dtype == "fp8" else "BF16",
            "scale": scale,
            "median_rel_rms": median,
            "p99_rel_rms": float(rel.quantile(0.99)) if rel.numel() > 1 else median,
            "max_rel_rms": float(rel.max()),
            "shards": len(shard_keys),
            "rows_per_shard": rows_per_shard,
        }
        # R1: an auto choice never silently writes an unserved bf16 table; report and stop.
        if out_dtype == "auto" and dtype == "bf16":
            raise Fp8GateExceeded(report)

        # Pass 2: write files, _SHARDS_PER_FILE table shards each.
        weight_map: dict[str, str] = {}
        n_files = (len(shard_keys) + _SHARDS_PER_FILE - 1) // _SHARDS_PER_FILE
        for f in range(n_files):
            name = f"freetoken-ple-{f + 1:05d}-of-{n_files:05d}.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            for i in range(f * _SHARDS_PER_FILE, min(len(shard_keys), (f + 1) * _SHARDS_PER_FILE)):
                dec = _decode_shard(mm, base, header[shard_keys[i]], k, i * rows_per_shard,
                                    head_offsets, head_bias, codebook, device, chunk_rows)
                _release_shard_pages(mm, base, header[shard_keys[i]])  # R1 fix round 1
                drop_page_cache(src)  # R1
                out_key = f"{prefix}.shard_{i}.weight"
                if dtype == "fp8":
                    tensors[out_key] = (dec / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
                else:
                    tensors[out_key] = dec.to(torch.bfloat16)
                weight_map[out_key] = name
            if f == 0:
                ple_base = prefix.rsplit(".ngram_embedding", 1)[0]
                tensors[f"{prefix}.weight_scale"] = torch.tensor(
                    scale if dtype == "fp8" else 1.0, dtype=torch.bfloat16)
                tensors[f"{ple_base}.layer_multipliers"] = multipliers
                tensors[f"{ple_base}.ngram_heads_vocab_sizes"] = head_vocab
                tensors[f"{ple_base}.ngram_heads_offsets"] = head_offsets
                for key in (f"{prefix}.weight_scale", f"{ple_base}.layer_multipliers",
                            f"{ple_base}.ngram_heads_vocab_sizes", f"{ple_base}.ngram_heads_offsets"):
                    weight_map[key] = name
            out_path = os.path.join(model_dir, name)
            safetensors.torch.save_file(tensors, out_path)
            drop_page_cache(out_path)  # R1: don't leave the freshly written shard in page cache
            del tensors
    finally:
        mm.close()
        os.close(fd)
    with open(os.path.join(model_dir, "freetoken-ple.index.json"), "w", encoding="utf-8") as fh:
        json.dump({"weight_map": weight_map}, fh, indent=1, sort_keys=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-rows", type=int, default=32768)
    # no bf16: the loader serves only F8_E4M3 tables (R1), so a bf16 run would write ~100 GB for nothing
    parser.add_argument("--dtype", choices=("fp8", "auto"), default="auto")
    args = parser.parse_args(argv)
    try:
        report = convert_table(args.model_dir, device=args.device,
                               chunk_rows=args.chunk_rows, out_dtype=args.dtype)
    except Fp8GateExceeded as exc:
        print(json.dumps({**exc.report, "written": False}, indent=1))
        return 3
    print(json.dumps({**report, "written": True}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

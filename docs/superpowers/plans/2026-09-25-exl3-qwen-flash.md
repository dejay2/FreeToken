# EXL3 Qwen3.8-Flash-Next Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FreeToken serves turboderp's 3.05bpw EXL3 Qwen3.8-Flash-Next with every linear kept EXL3 at runtime, pictures and MTP on, picked from the control panel, while today's NVFP4 Flash is unchanged.

**Architecture:** A new dense op (`Exl3Linear` and friends) calls the ExLlamaV3 wheel's `exl3_gemm` for small row counts and reconstruct+GEMM for prompts; the Qwen modules pick it when `ModelConfig.linear_storage == "exl3"`. The GLM EXL3 routed-expert path is generalised from fixed K=2 to "K from the checkpoint". The n-gram table is converted once, offline, to FreeToken's existing FP8 table layout.

**Tech Stack:** Python 3.12, torch 2.11 / CUDA 13.2, ExLlamaV3 `exllamav3_ext` wheel (1.4.6 on the box, MIT), pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-exl3-qwen-flash-design.md` (commit a9ff859). Read it first.

## Global Constraints

- Branch `feat/exl3-qwen-flash`; commits `type: subject` (`feat|fix|perf|refactor|build|ci|docs|test|chore`); never force-push; push target `origin` only.
- Every new behaviour is keyed on `quant_method == "exl3"`; the NVFP4 path must stay byte-identical (existing tests unchanged and passing).
- Refuse to boot with the tensor name and reason on any EXL3 mismatch — never a silent fallback to bf16.
- EXL3 codebook: `mul1` only. `mcg` is refused.
- Dense EXL3 small-row path: rows ≤ 144 use `exl3_gemm`; rows > 144 reconstruct (exllamav3 `modules/quant/exl3.py:10`). `lm_head` never reconstructs (tiles by 144 rows).
- The dense EXL3 workspace is allocated once after weight load and never reallocated (captured CUDA graphs hold its addresses).
- Converted word table: F8_E4M3 `[2500012, 160]` × 128 shards + one bf16 `weight_scale`; precision gate median per-row relative RMS ≤ 5 %, else bf16 table.
- Devbox has no GPU: CPU tests run there with `PYTHONPATH=python .venv/bin/python -m pytest <paths> -q`; GPU tests skip there and run on the 5090 in WSL `vllm` (`~/FreeToken`, `.venv`).
- Live boots on the 5090 need Jay's explicit OK each time; one server at a time; stop the daily server from the control panel first; never kill non-python processes.
- Comments carry measurements and reasons (CLAUDE.md "Conventions"). No `.local/` paths in tracked files.
- Helpers/subagents never run `git reset`, `git clean`, `git checkout -- .` or any destructive git command in the main checkout.

## Review Focus

1. **A checkpoint tensor with a K different from the hint** (e.g. `index_qk_proj` K=3 next to attention K=5, MTP `fc_*` K=4) — must load and run correctly; covered in Task 3 (`test_load_adopts_checkpoint_k`) and Task 4 (synthetic checkpoint mixes K).
2. **Prompt rows crossing 144** (145, 4096 rows, and a prompt chunk that is not a multiple of 144 through `lm_head`) — reconstruct path and `lm_head` tiling must match the small-row path; Task 3 tests at T=1, 2, 144, 145, 300.
3. **CUDA-graph replay after a second Exl3Linear call reused the shared fp16 output buffer** — a later layer must not see an earlier layer's output overwritten; Task 3 GPU test `test_graph_replay_two_linears`.
4. **The NVFP4 checkpoint after these changes** — must parse to exactly the old flags and load the same state keys; Task 2 `test_nvfp4_flags_unchanged`, Task 4 `test_nvfp4_iter_weights_unchanged`.
5. **Word table rows at head boundaries and the last row** — head bias must come from the right head (`searchsorted(right=True) - 1`); Task 1 `test_head_boundary_rows`.

---

## File map

| File | Responsibility |
|---|---|
| Create `python/freetoken/kernel/exl3_linear.py` | `Exl3Linear`, `Exl3ColMerged`, `Exl3LMHead`, dense workspace, wheel `exl3_gemm` adapter |
| Create `python/freetoken/models/qwen4_exp/exl3_ngram.py` | pure-torch decoder for `exl3_ngram_trellis` rows (port of exllamav3 `ngram_codec.py`, MIT) |
| Create `scripts/exl3/convert_ngram_table.py` | one-time table converter (CLI) |
| Create `scripts/exl3/probe_wheel.py` | box check of wheel routines, signatures and shapes |
| Modify `python/freetoken/models/config.py` | `ModelConfig.linear_storage`, `exl3_expert_k` fields |
| Modify `python/freetoken/models/qwen4_exp/config.py` | exl3 detection; vision `exl3` flag |
| Modify `python/freetoken/models/qwen4_exp/{attention,gdn,model,weight,vision,mtp_spike}.py`, `models/qwen3_5_moe/moe.py` | pick EXL3 ops; loader rename/fusions; sidecar table index |
| Modify `python/freetoken/models/vision_weight.py` | allow EXL3 components for named vision linears |
| Modify `python/freetoken/models/exl3_banks.py`, `moe/fused_exl3.py`, `kernel/exl3_mgemm.py`, `moe/offload_cache.py`, `kernel/aot_models.py`, `moe/expert_banks.py`, `engine/memory_plan.py` | K from checkpoint; K-aware byte formulas; top_k 10 graph rule |
| Modify `python/freetoken/engine/engine.py`, `engine/spec_draft.py`, `engine/spec_lmhead.py` | workspace prep; exl3 draft experts; exl3 head is a quantized head |
| Modify `python/freetoken/daemon/settings/{model_info,model_detect}.py` | exl3 format, K-aware sizes, missing-table warning |
| Tests | `tests/kernels/test_exl3_linear.py`, `tests/models/qwen4_exp/test_exl3_{ngram,weight,vision,mtp}.py`, additions to `tests/models/qwen4_exp/test_config.py`, `tests/models/test_exl3_banks.py`, `tests/moe/test_fused_exl3.py`, `tests/settings/test_model_info.py` |
| Create `docs/research/exl3-qwen-flash-2026-09-XX.md` | measured numbers (Task 10) |

---

### Task 0: Download the checkpoint and probe the wheel on the 5090

Operational, on the box (WSL `vllm`), no product code except the probe script. Does not need a server boot, so no card takeover; it may run while the daily server serves.

**Files:**
- Create: `scripts/exl3/probe_wheel.py`

**Interfaces:**
- Produces: `~/models/Qwen3.8-Flash-Next-exl3-3.05bpw/` (full branch `3.05bpw_h5_ng5`); a probe report pasted into the task report (signatures, `hasattr` results, shape table) that Tasks 3 and 5 rely on.

- [ ] **Step 1: Write the probe script**

```python
"""Report what the installed ExLlamaV3 wheel offers for the EXL3 Qwen Flash path.

Run on the serving box: ``.venv/bin/python scripts/exl3/probe_wheel.py``. Prints one line per
check; exits 1 if a routine the plan needs is missing or a Flash shape is unsupported.
"""

from __future__ import annotations

import sys

import torch  # noqa: F401  (loads libc10 before the extension)
import exllamav3_ext as ext

NEEDED = ("exl3_gemm", "exl3_mgemm", "reconstruct_had_slice",
          "exl3_gemm_shape_compat", "exl3_gemm_num_kernel_shapes")
OPTIONAL = ("ngram_dequant", "exl3_gemv")
# (in_features, out_features, K) of every Flash EXL3 linear, from the 3.05bpw_h5_ng5 headers.
SHAPES = [
    (2560, 640, 3), (640, 2560, 3),                   # routed experts gate/up, down
    (2560, 640, 3),                                   # indexer index_qk_proj
    (2560, 12288, 5), (2560, 512, 5), (6144, 2560, 5),  # q, k|v, o
    (2560, 10240, 5), (2560, 6144, 5), (6144, 2560, 5),  # GDN qkv, z, out
    (2560, 640, 5), (640, 2560, 5),                   # shared expert gate/up, down
    (2560, 248320, 5),                                # lm_head
    (1152, 1152, 5), (1152, 4352, 5), (4352, 1152, 5),  # vision attn.proj, fc1, fc2
    (4608, 4608, 5), (4608, 2560, 5),                 # vision merger fc1, fc2
    (2560, 2560, 4),                                  # MTP fc_embedding / fc_hidden
]


def main() -> int:
    bad = 0
    for name in NEEDED + OPTIONAL:
        present = hasattr(ext, name)
        print(f"{'OK ' if present else 'MISSING'} {name}")
        if not present and name in NEEDED:
            bad += 1
    print("exl3_gemm doc:", getattr(ext.exl3_gemm, "__doc__", "")[:400])
    shapes = int(ext.exl3_gemm_num_kernel_shapes())
    for fin, fout, k in SHAPES:
        ok = any(bool(ext.exl3_gemm_shape_compat(s, 1, fin, fout, k)) for s in range(1, shapes + 1))
        print(f"{'OK ' if ok else 'NO '} shape in={fin} out={fout} K={k}")
        bad += 0 if ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Commit the script on the devbox and push the branch**

```bash
git add scripts/exl3/probe_wheel.py
git commit -m "chore: probe script for the EXL3 wheel routines the Qwen Flash path needs"
git push origin feat/exl3-qwen-flash
```

- [ ] **Step 3: On the box, check out the branch in a separate worktree and run the probe**

Use the SSH route (`cat script | ssh -o BatchMode=yes 5090 'wsl -d vllm -e bash -l'`). Never switch the serving checkout `~/FreeToken` off `mtp-upstream-merge`.

```bash
cd ~/FreeToken && git fetch origin feat/exl3-qwen-flash
git worktree add ~/FreeToken-exl3 origin/feat/exl3-qwen-flash
cd ~/FreeToken-exl3 && ~/FreeToken/.venv/bin/python scripts/exl3/probe_wheel.py
~/FreeToken/.venv/bin/python -c "import exllamav3_ext as e, inspect; print(e.exl3_gemm.__doc__)"
```
Expected: every NEEDED line `OK`, every shape `OK`. Record the exact `exl3_gemm` signature string.

If a NEEDED routine or shape is missing: build exllamav3 1.5.1 from source for the venv (`git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3 && git checkout 6b84a21 && TORCH_CUDA_ARCH_LIST=12.0 ~/FreeToken/.venv/bin/pip install --no-build-isolation .`), keep the old wheel's dist-info name in the report, re-run the probe, then re-run the GLM EXL3 tests (`tests/kernels/test_exl3_mgemm.py`) to confirm nothing regressed. Report back before continuing if the build fails.

- [ ] **Step 4: Download the checkpoint (about 85 GB, `/` has 344 GB free)**

```bash
cd ~/models && ~/FreeToken/.venv/bin/huggingface-cli download turboderp/Qwen3.8-Flash-Next-exl3 \
  --revision 3.05bpw_h5_ng5 --local-dir Qwen3.8-Flash-Next-exl3-3.05bpw
du -sh Qwen3.8-Flash-Next-exl3-3.05bpw && ls Qwen3.8-Flash-Next-exl3-3.05bpw
```
Expected: 7 `model-0000N-of-00007.safetensors`, `ngram_embedding.safetensors` (32.64 GB), `config.json` with `quant_method: exl3`, `preprocessor_config.json`. Total ≈ 85 GB.

- [ ] **Step 5: Report** — paste the probe output, the signature, and `du -sh` into the task report. No commit.

---

### Task 1: Word-table decoder and one-time converter

**Files:**
- Create: `python/freetoken/models/qwen4_exp/exl3_ngram.py`
- Create: `scripts/exl3/convert_ngram_table.py`
- Modify: `python/freetoken/models/qwen4_exp/weight.py:726-734` (`_ple_table_files` reads the sidecar index first)
- Test: `tests/models/qwen4_exp/test_exl3_ngram.py`

**Interfaces:**
- Produces: `mul1_codebook(device) -> Tensor[65536] fp16`; `dequant_rows(packed: Tensor[N, 1+10K] int16, k: int, codebook, bias: Tensor[N,160] | None) -> Tensor[N,160] fp32`; `head_of_rows(rows: Tensor[N] int64, head_offsets: Tensor[H] int64) -> Tensor[N] int64`; `convert_table(model_dir: str, *, device: str, chunk_rows: int, out_dtype: str) -> dict` (report); sidecar file `freetoken-ple.index.json` (`{"weight_map": {name: file}}`) and table files `freetoken-ple-000NN-of-000MM.safetensors` inside the model dir.

- [ ] **Step 1: Write failing tests**

```python
"""EXL3 n-gram table decode + conversion (spec 2026-09-25 section 4)."""

from __future__ import annotations

import json
import struct

import pytest
import safetensors.torch
import torch

from freetoken.models.qwen4_exp.exl3_ngram import (
    ROW_DIM, dequant_rows, head_of_rows, mul1_codebook, pack_rows, words_per_row,
)


def _rows(n: int, k: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    states = torch.randint(0, 1 << 16, (n, ROW_DIM), generator=g, dtype=torch.int64)
    # A tail-biting ring constrains neighbouring states; build legal states by packing then
    # unpacking, so the reference below compares like with like.
    scales = (torch.rand(n, generator=g) + 0.5).to(torch.float16)
    return pack_rows(states, scales, k)


def test_words_per_row_matches_checkpoint():
    assert words_per_row(5) == 51  # 3.05bpw_h5_ng5 header: shard trellis [2500012, 51]


def test_codebook_is_bit_exact_fp16():
    cb = mul1_codebook("cpu")
    assert cb.dtype == torch.float16 and cb.shape == (65536,)
    # decode_3inst<2> of state 0: bsum 0 -> (1024 * 0x1eee) + 0xc931 in fp16
    k_inv = torch.tensor([0x1EEE], dtype=torch.uint16).view(torch.float16).float()
    k_bias = torch.tensor([0xC931], dtype=torch.uint16).view(torch.float16).float()
    assert cb[0].float() == (1024.0 * k_inv + k_bias).half().float()


def test_pack_unpack_round_trip_values():
    packed = _rows(64, 5)
    cb = mul1_codebook("cpu")
    once = dequant_rows(packed, 5, cb)
    again = dequant_rows(pack_rows(*_unpack(packed, 5), 5), 5, cb)
    assert torch.equal(once, again)


def _unpack(packed, k):
    from freetoken.models.qwen4_exp.exl3_ngram import unpack_rows
    return unpack_rows(packed, k)


def test_head_boundary_rows():
    offsets = torch.tensor([0, 10, 25, 40])
    rows = torch.tensor([0, 9, 10, 24, 25, 39, 40, 41])
    assert head_of_rows(rows, offsets).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def _write_exl3_table(path, *, shards=4, rows_per_shard=8, k=5, heads=2, layer=1):
    prefix = f"model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding"
    tensors = {}
    for s in range(shards):
        tensors[f"{prefix}.shard_{s}.trellis"] = _rows(rows_per_shard, k, seed=s)
    total = shards * rows_per_shard
    tensors[f"{prefix}.head_bias"] = torch.randn(heads, ROW_DIM).half()
    tensors[f"{prefix}.head_offsets"] = torch.tensor([0, total // 2])
    tensors[f"{prefix}.head_vocab_sizes"] = torch.tensor([total // 2, total - total // 2])
    tensors[f"{prefix}.layer_multipliers"] = torch.tensor([1, 3, 5])
    meta = {"format": "exl3_ngram_trellis", "version": "1", "K": str(k), "codebook": "mul1",
            "row_dim": str(ROW_DIM), "rows": str(total), "shard_rows": str(rows_per_shard)}
    safetensors.torch.save_file(tensors, str(path), metadata=meta)
    return tensors, prefix


def test_convert_writes_freetoken_layout(tmp_path):
    from scripts.exl3.convert_ngram_table import convert_table  # noqa: E402

    tensors, prefix = _write_exl3_table(tmp_path / "ngram_embedding.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "exl3"}}))
    report = convert_table(str(tmp_path), device="cpu", chunk_rows=5, out_dtype="fp8")
    index = json.loads((tmp_path / "freetoken-ple.index.json").read_text())["weight_map"]
    shard_names = sorted(n for n in index if ".ngram_embedding.shard_" in n)
    assert len(shard_names) == 4 and all(n.endswith(".weight") for n in shard_names)
    assert f"{prefix}.weight_scale" in index
    # hash constants are renamed to FreeToken's NGramEmbedding state keys
    base = prefix.rsplit(".ngram_embedding", 1)[0]
    for key in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
        assert f"{base}.{key}" in index
    # decoded values agree with the fp32 reference within fp8 resolution
    cb = mul1_codebook("cpu")
    got_file = tmp_path / index[f"{prefix}.shard_1.weight"]
    got = safetensors.torch.load_file(str(got_file))
    scale = got[f"{prefix}.weight_scale"].float() if f"{prefix}.weight_scale" in got else None
    if scale is None:
        scale = safetensors.torch.load_file(str(tmp_path / index[f"{prefix}.weight_scale"]))[
            f"{prefix}.weight_scale"].float()
    rows = torch.arange(8, 16)
    heads = head_of_rows(rows, tensors[f"{prefix}.head_offsets"])
    ref = dequant_rows(tensors[f"{prefix}.shard_1.trellis"], 5, cb,
                       tensors[f"{prefix}.head_bias"][heads])
    approx = got[f"{prefix}.shard_1.weight"].float() * scale
    rel = ((approx - ref).pow(2).mean(1).sqrt() / ref.pow(2).mean(1).sqrt())
    assert rel.median() < 0.05
    assert report["median_rel_rms"] < 0.05 and report["dtype"] == "F8_E4M3"


def test_convert_refuses_mismatched_hash_constants(tmp_path):
    from scripts.exl3.convert_ngram_table import convert_table

    tensors, prefix = _write_exl3_table(tmp_path / "ngram_embedding.safetensors")
    tensors[f"{prefix}.head_offsets"] = torch.tensor([0, 3])  # not what head_vocab_sizes imply
    safetensors.torch.save_file(tensors, str(tmp_path / "ngram_embedding.safetensors"),
                                metadata={"format": "exl3_ngram_trellis", "version": "1", "K": "5",
                                          "row_dim": "160", "rows": "32", "shard_rows": "8"})
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="head_offsets"):
        convert_table(str(tmp_path), device="cpu", chunk_rows=5, out_dtype="fp8")
```

Add an empty `scripts/__init__.py` and `scripts/exl3/__init__.py` only if `scripts` is not already importable in tests (check `ls scripts/__init__.py`); otherwise import via `importlib.util.spec_from_file_location` in a small helper at the top of the test.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=python:. .venv/bin/python -m pytest tests/models/qwen4_exp/test_exl3_ngram.py -q`
Expected: FAIL with `ModuleNotFoundError: freetoken.models.qwen4_exp.exl3_ngram`.

- [ ] **Step 3: Implement the decoder**

```python
"""Decoder for ExLlamaV3's ``exl3_ngram_trellis`` n-gram tables (format version 1).

Port of exllamav3 ``modules/quant/exl3_lib/ngram_codec.py`` at 6b84a21 (MIT License,
Copyright (c) 2025 Turboderp; full notice in ``freetoken/kernel/exl3.py``). A packed row is
``1 + 10*K`` int16 words: word 0 is the fp16 row scale's bit pattern, words 1.. hold a
160*K-bit tail-biting ring where stream bits [i*K, (i+1)*K) are the low K bits of state i.

    row[i] = mul1(state_i) * scale + head_bias[head(row)]

FreeToken serves a converted FP8 table (spec 2026-09-25 section 4); this module is the
converter's decoder and its test oracle, never a serve-time path.
"""

from __future__ import annotations

import torch

ROW_DIM = 160
_MUL1 = 0x83DCD12D


def words_per_row(k: int) -> int:
    return 1 + ROW_DIM * k // 16


def mul1_codebook(device) -> torch.Tensor:
    s = torch.arange(65536, dtype=torch.int64, device=device)
    prod = (s * _MUL1) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    h = (1024 + bsum).float()
    k_inv = torch.tensor([0x1EEE], dtype=torch.uint16).view(torch.float16).float().item()
    k_bias = torch.tensor([0xC931], dtype=torch.uint16).view(torch.float16).float().item()
    return (h * k_inv + k_bias).to(torch.float16)


def pack_rows(states: torch.Tensor, scales_f16: torch.Tensor, k: int) -> torch.Tensor:
    n = states.shape[0]
    dev = states.device
    new_bits = states.to(torch.int64) & ((1 << k) - 1)
    bits = (new_bits.unsqueeze(-1) >> torch.arange(k, device=dev)) & 1
    bits = bits.reshape(n, ROW_DIM * k // 16, 16)
    words = (bits << torch.arange(16, device=dev)).sum(dim=-1)
    words = (words & 0xFFFF).to(torch.uint16).view(torch.int16)
    scale_words = scales_f16.to(torch.float16).view(torch.int16).unsqueeze(1)
    return torch.cat((scale_words, words), dim=1).contiguous()


def unpack_rows(packed: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    dev = packed.device
    scales = packed[:, 0].contiguous().view(torch.float16)
    words = packed[:, 1:].contiguous().view(torch.uint16).to(torch.int64)
    stream = ((words.unsqueeze(-1) >> torch.arange(16, device=dev)) & 1).reshape(
        packed.shape[0], ROW_DIM * k
    )
    i = torch.arange(ROW_DIM, device=dev).unsqueeze(1)
    m = torch.arange(16, device=dev).unsqueeze(0)
    src = ((i - m // k) % ROW_DIM) * k + m % k
    states = (stream[:, src] << m).sum(dim=-1)
    return states, scales


def dequant_rows(packed, k: int, codebook: torch.Tensor, bias=None) -> torch.Tensor:
    if packed.shape[1] != words_per_row(k):
        raise ValueError(f"packed rows have {packed.shape[1]} words, K={k} needs {words_per_row(k)}")
    states, scales = unpack_rows(packed, k)
    out = codebook[states].float() * scales.float().unsqueeze(1)
    if bias is not None:
        out = out + bias.float()
    return out


def head_of_rows(rows: torch.Tensor, head_offsets: torch.Tensor) -> torch.Tensor:
    heads = torch.searchsorted(head_offsets.to(rows.device), rows, right=True) - 1
    return heads.clamp(0, head_offsets.shape[0] - 1)


__all__ = ["ROW_DIM", "dequant_rows", "head_of_rows", "mul1_codebook", "pack_rows",
           "unpack_rows", "words_per_row"]
```

- [ ] **Step 4: Implement the converter**

`scripts/exl3/convert_ngram_table.py`:

```python
"""One-time conversion of an EXL3 ``ngram_embedding.safetensors`` into FreeToken's table.

Usage (serving box): ``.venv/bin/python scripts/exl3/convert_ngram_table.py <model-dir>``
[--device cuda] [--chunk-rows 32768] [--dtype fp8|bf16|auto]

Writes ``freetoken-ple-000NN-of-000MM.safetensors`` (8 table shards per file) plus
``freetoken-ple.index.json`` into <model-dir>. turboderp's files are left untouched.
Never holds more than one table shard in host memory. Spec 2026-09-25 section 4.
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

from freetoken.models.qwen4_exp.exl3_ngram import (
    dequant_rows, head_of_rows, mul1_codebook, words_per_row,
)

_FP8_MAX = 448.0
_SHARDS_PER_FILE = 8
_GATE = 0.05  # median per-row relative RMS error allowed for the fp8 table
_SAMPLE_ROWS = 1_000_000


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
        scale = absmax / _FP8_MAX if absmax > 0 else 1.0
        sample_every = max(1, (rows_per_shard * len(shard_keys)) // _SAMPLE_ROWS)
        rels = []
        for i, key in enumerate(shard_keys):
            dec = _decode_shard(mm, base, header[key], k, i * rows_per_shard, head_offsets,
                                head_bias, codebook, device, chunk_rows)[::sample_every]
            fp8 = (dec / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).float() * scale
            denom = dec.pow(2).mean(1).sqrt().clamp_min(1e-12)
            rels.append((fp8 - dec).pow(2).mean(1).sqrt() / denom)
        rel = torch.cat(rels)
        median = float(rel.median())
        dtype = out_dtype if out_dtype != "auto" else ("fp8" if median <= _GATE else "bf16")

        # Pass 2: write files, _SHARDS_PER_FILE table shards each.
        weight_map: dict[str, str] = {}
        n_files = (len(shard_keys) + _SHARDS_PER_FILE - 1) // _SHARDS_PER_FILE
        for f in range(n_files):
            name = f"freetoken-ple-{f + 1:05d}-of-{n_files:05d}.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            for i in range(f * _SHARDS_PER_FILE, min(len(shard_keys), (f + 1) * _SHARDS_PER_FILE)):
                dec = _decode_shard(mm, base, header[shard_keys[i]], k, i * rows_per_shard,
                                    head_offsets, head_bias, codebook, device, chunk_rows)
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
            safetensors.torch.save_file(tensors, os.path.join(model_dir, name))
            del tensors
    finally:
        mm.close()
        os.close(fd)
    with open(os.path.join(model_dir, "freetoken-ple.index.json"), "w", encoding="utf-8") as fh:
        json.dump({"weight_map": weight_map}, fh, indent=1, sort_keys=True)
    report = {
        "dtype": "F8_E4M3" if dtype == "fp8" else "BF16",
        "scale": scale,
        "median_rel_rms": median,
        "p99_rel_rms": float(rel.quantile(0.99)) if rel.numel() > 1 else median,
        "max_rel_rms": float(rel.max()),
        "shards": len(shard_keys),
        "rows_per_shard": rows_per_shard,
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-rows", type=int, default=32768)
    parser.add_argument("--dtype", choices=("fp8", "bf16", "auto"), default="auto")
    args = parser.parse_args(argv)
    print(json.dumps(convert_table(args.model_dir, device=args.device,
                                   chunk_rows=args.chunk_rows, out_dtype=args.dtype), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note the bf16 branch writes `BF16` shards; FreeToken's loader only accepts `F8_E4M3` (`weight.py:717`) — if the gate ever picks bf16, stop and report (spec backup plan) rather than editing the loader in this task.

- [ ] **Step 5: Teach `_ple_table_files` the sidecar**

In `python/freetoken/models/qwen4_exp/weight.py`, at the top of `_ple_table_files`:

```python
    # EXL3 checkpoints carry a trellis-packed table FreeToken does not serve; the one-time
    # converter (scripts/exl3/convert_ngram_table.py) writes the FP8 table beside it and lists
    # it here, leaving the checkpoint's own index untouched.
    sidecar = os.path.join(folder, "freetoken-ple.index.json")
    if os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
        return sorted(os.path.join(folder, shard) for shard in files)
```

Add to the test file:

```python
def test_ple_table_files_prefers_sidecar(tmp_path):
    from freetoken.models.qwen4_exp.weight import _ple_table_files

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (tmp_path / "freetoken-ple.index.json").write_text(json.dumps({"weight_map": {
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": "a.safetensors"}}))
    assert _ple_table_files(str(tmp_path)) == [str(tmp_path / "a.safetensors")]
```

Also make `_rename` skip turboderp's table file keys (they all contain `_PLE_TABLE_INFIX` already — confirm with a test that `_rename("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.head_bias")` is `None`).

And refuse to boot an unconverted EXL3 folder with the fix in the message. At the top of `_ple_layout`, after `folder = download_hf_weight(model_path)`:

```python
    if (os.path.exists(os.path.join(folder, "ngram_embedding.safetensors"))
            and not os.path.exists(os.path.join(folder, "freetoken-ple.index.json"))):
        raise ValueError(
            f"{folder} holds an EXL3 trellis n-gram table FreeToken does not serve; convert it once: "
            f"python scripts/exl3/convert_ngram_table.py {folder}")
```

Test:

```python
def test_unconverted_exl3_table_names_the_converter(tmp_path):
    from types import SimpleNamespace
    from freetoken.models.qwen4_exp.weight import _ple_layout

    (tmp_path / "ngram_embedding.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="convert_ngram_table.py"):
        _ple_layout(str(tmp_path), SimpleNamespace(split_ngram_parts=128, ngram_head_dim=160))
```

- [ ] **Step 6: Run tests** — `PYTHONPATH=python:. .venv/bin/python -m pytest tests/models/qwen4_exp/test_exl3_ngram.py tests/models/qwen4_exp/test_weight.py tests/models/qwen4_exp/test_ple_disk.py -q` → PASS.

- [ ] **Step 7: Commit**

```bash
git add python/freetoken/models/qwen4_exp/exl3_ngram.py scripts/exl3/convert_ngram_table.py \
  python/freetoken/models/qwen4_exp/weight.py tests/models/qwen4_exp/test_exl3_ngram.py
git commit -m "feat: convert the EXL3 n-gram table once into FreeToken's FP8 table layout"
```

- [ ] **Step 8: Box run (no server boot needed)** — in `~/FreeToken-exl3` after `git pull`:
`~/FreeToken/.venv/bin/python scripts/exl3/convert_ngram_table.py ~/models/Qwen3.8-Flash-Next-exl3-3.05bpw --device cuda`. The GPU must have ~2 GB free (the daily server leaves ~1.5-2.5 GB; if the probe `nvidia-smi` shows less, run with `--device cpu --chunk-rows 4096` and expect hours). Paste the JSON report (dtype, median, p99, max) into the task report; it goes into Task 10's note. If `hasattr(exllamav3_ext, "ngram_dequant")`, also compare 10,000 random rows against `ext.ngram_dequant` and report the max abs difference.

---

### Task 2: Detect EXL3 in the Qwen config

**Files:**
- Modify: `python/freetoken/models/config.py` (ModelConfig fields near :332-353)
- Modify: `python/freetoken/models/qwen4_exp/config.py:224-265` and `Qwen4VisionConfig` (:20-35), `_parse_vision_config` (:158)
- Test: `tests/models/qwen4_exp/test_config.py` (append)

**Interfaces:**
- Produces: `ModelConfig.linear_storage: str = "bf16"` (`"bf16" | "exl3"`); `ModelConfig.exl3_expert_k: int = 2`; `Qwen4VisionConfig.exl3: bool = False`. For an exl3 checkpoint: `expert_quant="exl3"`, `lm_head_quant="exl3"`, `linear_storage="exl3"`, `attn_quant="none"`, `dense_quant="none"` (so `FREETOKEN_DENSE_QUANT=int8` can still only touch bf16 projections such as HC), `exl3_expert_k=int(bits)`.

- [ ] **Step 1: Failing tests (append to test_config.py)**

```python
_EXL3_QUANT = {"quant_method": "exl3", "version": "1.4.4", "bits": 3.05, "head_bits": 5,
               "codebook": "mul1", "out_scales": "always", "vision_bits": 5, "mtp_bits": 3}


def test_exl3_checkpoint_sets_exl3_flags(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DENSE_QUANT", raising=False)
    hf = _hf_config()
    hf.quantization_config = dict(_EXL3_QUANT)
    cfg = parse_config(hf)
    assert (cfg.expert_quant, cfg.lm_head_quant, cfg.linear_storage) == ("exl3", "exl3", "exl3")
    assert (cfg.attn_quant, cfg.dense_quant) == ("none", "none")
    assert cfg.exl3_expert_k == 3


def test_exl3_dense_override_only_touches_bf16_projections(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DENSE_QUANT", "int8")
    hf = _hf_config()
    hf.quantization_config = dict(_EXL3_QUANT)
    cfg = parse_config(hf)
    assert cfg.lm_head_quant == "exl3" and cfg.linear_storage == "exl3"
    assert cfg.dense_quant == "int8"  # HC projections, bf16 in the EXL3 checkpoint


def test_exl3_refuses_mcg_codebook():
    hf = _hf_config()
    hf.quantization_config = dict(_EXL3_QUANT, codebook="mcg")
    with pytest.raises(ValueError, match="mul1"):
        parse_config(hf)


def test_nvfp4_flags_unchanged(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DENSE_QUANT", raising=False)
    cfg = parse_config(_hf_config())
    assert cfg.linear_storage == "bf16"
    assert (cfg.expert_quant, cfg.attn_quant, cfg.dense_quant, cfg.lm_head_quant) == (
        "nvfp4", "none", "none", "none")
```

(If `_hf_config()` in this file does not carry the NVFP4 ignore list that yields those four flags, copy the expected tuple from the existing `test_moe_and_quant_flags` instead.)

- [ ] **Step 2: Run** `PYTHONPATH=python .venv/bin/python -m pytest tests/models/qwen4_exp/test_config.py -q` → the four new tests FAIL (`AttributeError: linear_storage`).

- [ ] **Step 3: Implement**

`models/config.py`, beside `lm_head_quant` in ModelConfig:

```python
    # Storage format of the non-expert linears in the checkpoint. "exl3" (turboderp's EXL3
    # builds) keeps every quantized linear packed at runtime (kernel/exl3_linear.py); "bf16"
    # is every other checkpoint, where dense_quant/attn_quant pick load-time conversions.
    linear_storage: str = "bf16"
    # K (bits per weight) of the routed EXL3 experts; one K for every routed expert, checked
    # against the trellis headers at load (models/exl3_banks.py). GLM-5.3 2.05bpw is 2.
    exl3_expert_k: int = 2
```

`qwen4_exp/config.py`, in the `else:` of `get is None`, before the fp8 check:

```python
        if algo == "exl3":
            # turboderp EXL3: every linear except routers, gates, GDN a/b, HC and PLE
            # projections is trellis-packed (3.05bpw_h5_ng5 headers, 2026-09-25): experts K=3,
            # attention/GDN/shared/lm_head K=5. `bits` is the average; the expert K is its floor.
            codebook = str(get("codebook") or "mul1").lower()
            if codebook != "mul1":
                raise ValueError(f"EXL3 codebook {codebook!r} is unsupported; only mul1 is")
            expert_quant = lm_head_quant = "exl3"
            attn_quant = dense_quant = "none"
            linear_storage = "exl3"
            exl3_expert_k = int(float(get("bits")))
            vision_exl3 = True
```

Initialise `linear_storage = "bf16"`, `exl3_expert_k = 2`, `vision_exl3 = False` before the `if get is None` and convert the existing `if algo == "fp8" and block:` into `elif`. The dense override block below is unchanged (it only upgrades `"none"`). Pass `linear_storage=linear_storage, exl3_expert_k=exl3_expert_k` into the `ModelConfig(...)` at :381-384, and `exl3=vision_exl3` into `Qwen4VisionConfig` (add the field `exl3: bool = False`; `_parse_vision_config` gains an `exl3: bool = False` parameter).

- [ ] **Step 4: Run** the whole `tests/models/qwen4_exp/test_config.py` and `tests/models/test_config*.py` → PASS.

- [ ] **Step 5: Commit** `git commit -am "feat: recognise EXL3 Qwen Flash checkpoints instead of loading them as bf16"`

---

### Task 3: The dense EXL3 op

**Files:**
- Create: `python/freetoken/kernel/exl3_linear.py`
- Test: `tests/kernels/test_exl3_linear.py`

**Interfaces:**
- Consumes: `freetoken.kernel.exl3.reconstruct(trellis, suh, svh, *, k, codebook, out, work)` and `reconstruct_reference(...)` (existing).
- Produces:
  - `class Exl3Linear(BaseOP)`: `__init__(in_features: int, out_features: int, has_bias: bool = False, *, k_hint: int = 5, allow_reconstruct: bool = True)`; attributes `trellis, suh, svh, mul1, bias`, `k: int`; `forward(x: Tensor[..., in]) -> Tensor[..., out] bf16`; `resident_bytes: int`.
  - `class Exl3ColMerged(BaseOP)`: `__init__(in_features: int, parts: list[tuple[str, int]], *, k_hint: int = 5)`; sub-ops are attributes named by `parts[i][0]`; `forward(x) -> Tensor[..., sum(out)]` in `parts` order.
  - `class Exl3LMHead(Exl3Linear)`: `__init__(num_embeddings: int, embedding_dim: int)`; `forward(x)` (prefill last-row slice like `ParallelLMHead`), `forward_all(x)`; attrs `num_embeddings, embedding_dim, num_embeddings_tp, vocab_range, tp_size=1`.
  - `iter_exl3_linears(root: BaseOP) -> Iterator[Exl3Linear]`
  - `prepare_exl3_dense_workspace(root: BaseOP, device) -> Exl3DenseWorkspace | None` (no-op returning None when the tree holds no Exl3Linear; raises if called twice with a larger need).
  - `require_exl3_dense_workspace_fits(root: BaseOP, device) -> None`

- [ ] **Step 1: Failing CPU tests**

```python
"""Dense EXL3 linear (spec 2026-09-25 section 2). CPU tests stub the wheel with the
pure-torch reconstruction oracle; the GPU tests at the bottom run on the 5090."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel import exl3 as exl3_kernel
from freetoken.kernel import exl3_linear as el


def _parts(fin, fout, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    trellis = torch.randint(-32768, 32767, (fin // 16, fout // 16, 16 * k), generator=g,
                            dtype=torch.int32).to(torch.int16)
    suh = (torch.randint(0, 2, (fin,), generator=g) * 2 - 1).half()
    svh = (torch.rand(fout, generator=g) * 0.02 + 0.01).half()
    return {"trellis": trellis, "suh": suh, "svh": svh, "mul1": torch.tensor(0, dtype=torch.int32)}


def _ref(x, p, k):
    w = exl3_kernel.reconstruct_reference(p["trellis"], p["suh"], p["svh"], k=k, codebook="mul1")
    return F.linear(x.float(), w.float())


@pytest.fixture
def cpu_wheel(monkeypatch):
    """Route the op's two wheel seams to the CPU oracle."""
    def fake_gemm(x16, trellis, y16, suh, xh, svh):
        k = trellis.shape[-1] // 16
        w = exl3_kernel.reconstruct_reference(trellis, suh, svh, k=k, codebook="mul1")
        y16.copy_(F.linear(x16.float(), w.float()).half())

    def fake_reconstruct(op, out, work):
        out.copy_(exl3_kernel.reconstruct_reference(op.trellis, op.suh, op.svh, k=op.k, codebook="mul1"))
        return out

    monkeypatch.setattr(el, "_exl3_gemm", fake_gemm)
    monkeypatch.setattr(el, "_reconstruct_into", fake_reconstruct)


def _loaded(fin, fout, k, *, bias=False, k_hint=5, **kw):
    op = el.Exl3Linear(fin, fout, has_bias=bias, k_hint=k_hint, **kw)
    state = {f"p.{n}": t for n, t in _parts(fin, fout, k).items()}
    if bias:
        state["p.bias"] = torch.randn(fout).bfloat16()
    op.load_state_dict(state, prefix="p")
    return op, state


def test_load_adopts_checkpoint_k():
    op, _ = _loaded(128, 256, 3, k_hint=5)
    assert op.k == 3 and op.trellis.shape == (8, 16, 48)


@pytest.mark.parametrize("bad,match", [
    (lambda s: s.update({"p.trellis": s["p.trellis"][:, :, :40]}), "K"),
    (lambda s: s.update({"p.suh": s["p.suh"].float()}), "suh"),
    (lambda s: s.update({"p.svh": s["p.svh"][:-16]}), "svh"),
    (lambda s: s.pop("p.mul1"), "mul1"),
])
def test_load_rejects_malformed_parts(bad, match):
    op = el.Exl3Linear(128, 256)
    state = {f"p.{n}": t for n, t in _parts(128, 256, 3).items()}
    bad(state)
    with pytest.raises((ValueError, KeyError), match=match):
        op.load_state_dict(state, prefix="p")


@pytest.mark.parametrize("rows", [1, 2, 144, 145, 300])
def test_forward_matches_reference(cpu_wheel, rows):
    op, state = _loaded(128, 256, 3, bias=True)
    el.prepare_exl3_dense_workspace(op, torch.device("cpu"), _reset=True)
    x = torch.randn(rows, 128).bfloat16()
    p = {n: state[f"p.{n}"] for n in ("trellis", "suh", "svh")}
    want = _ref(x, p, 3) + state["p.bias"].float()
    got = op.forward(x).float()
    assert got.shape == (rows, 256)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


def test_forward_keeps_leading_dims(cpu_wheel):
    op, _ = _loaded(128, 256, 3)
    el.prepare_exl3_dense_workspace(op, torch.device("cpu"), _reset=True)
    assert op.forward(torch.randn(2, 4, 128).bfloat16()).shape == (2, 4, 256)


def test_col_merged_concatenates_in_part_order(cpu_wheel):
    op = el.Exl3ColMerged(128, [("q_proj", 256), ("k_proj", 128)])
    state = {}
    for name, fout in (("q_proj", 256), ("k_proj", 128)):
        state.update({f"m.{name}.{n}": t for n, t in _parts(128, fout, 3, seed=fout).items()})
    op.load_state_dict(dict(state), prefix="m")
    el.prepare_exl3_dense_workspace(op, torch.device("cpu"), _reset=True)
    x = torch.randn(3, 128).bfloat16()
    q = _ref(x, {n: state[f"m.q_proj.{n}"] for n in ("trellis", "suh", "svh")}, 3)
    k = _ref(x, {n: state[f"m.k_proj.{n}"] for n in ("trellis", "suh", "svh")}, 3)
    torch.testing.assert_close(op.forward(x).float(), torch.cat([q, k], -1), rtol=2e-2, atol=2e-2)


def test_lm_head_never_reconstructs(cpu_wheel, monkeypatch):
    head = el.Exl3LMHead(num_embeddings=256, embedding_dim=128)
    head.load_state_dict({f"lm_head.{n}": t for n, t in _parts(128, 256, 5).items()}, prefix="lm_head")
    el.prepare_exl3_dense_workspace(head, torch.device("cpu"), _reset=True)
    monkeypatch.setattr(el, "_reconstruct_into", lambda *a, **k: pytest.fail("lm_head reconstructed"))
    assert head.forward_all(torch.randn(300, 128).bfloat16()).shape == (300, 256)


def test_forward_without_workspace_is_a_clear_error(cpu_wheel):
    op, _ = _loaded(128, 256, 3)
    el._WORKSPACES.clear()
    with pytest.raises(RuntimeError, match="prepare_exl3_dense_workspace"):
        op.forward(torch.randn(1, 128).bfloat16())


def test_workspace_never_grows_after_prepare(cpu_wheel):
    small, _ = _loaded(128, 256, 3)
    el.prepare_exl3_dense_workspace(small, torch.device("cpu"), _reset=True)
    big, _ = _loaded(256, 512, 3)
    with pytest.raises(RuntimeError, match="workspace"):
        el.require_exl3_dense_workspace_fits(big, torch.device("cpu"))


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the 5090")


@cuda
@pytest.mark.parametrize("fin,fout,k", [(2560, 12288, 5), (6144, 2560, 5), (2560, 640, 3),
                                         (2560, 2560, 4), (1152, 4352, 5)])
@pytest.mark.parametrize("rows", [1, 2, 8, 144, 145, 4096])
def test_gpu_matches_card_reconstruction(fin, fout, k, rows):
    dev = torch.device("cuda")
    op = el.Exl3Linear(fin, fout, k_hint=k)
    op.load_state_dict({f"p.{n}": t.to(dev) for n, t in _parts(fin, fout, k).items()}, prefix="p")
    el.prepare_exl3_dense_workspace(op, dev, _reset=True)
    x = torch.randn(rows, fin, device=dev).bfloat16()
    w = exl3_kernel.reconstruct(op.trellis, op.suh, op.svh, k=k, codebook="mul1")
    want = F.linear(x.float(), w.float())
    got = op.forward(x).float()
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-2, rel


@cuda
def test_graph_replay_two_linears():
    dev = torch.device("cuda")
    a = el.Exl3Linear(2560, 640, k_hint=5)
    b = el.Exl3Linear(640, 2560, k_hint=5)
    a.load_state_dict({f"a.{n}": t.to(dev) for n, t in _parts(2560, 640, 5, 1).items()}, prefix="a")
    b.load_state_dict({f"b.{n}": t.to(dev) for n, t in _parts(640, 2560, 5, 2).items()}, prefix="b")

    class Both(el.BaseOP):
        def __init__(self):
            self.a, self.b = a, b

    el.prepare_exl3_dense_workspace(Both(), dev, _reset=True)
    x = torch.randn(1, 2560, device=dev).bfloat16()
    eager_a = a.forward(x); eager = b.forward(eager_a) + eager_a.sum()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_a = a.forward(x)
        out = b.forward(out_a) + out_a.sum()
    g.replay(); torch.cuda.synchronize()
    torch.testing.assert_close(out, eager)
```

- [ ] **Step 2: Run** `PYTHONPATH=python .venv/bin/python -m pytest tests/kernels/test_exl3_linear.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `kernel/exl3_linear.py`**

```python
"""Dense EXL3 linears kept packed at runtime (spec 2026-09-25 section 2).

Small row counts run ExLlamaV3's ``exl3_gemm`` straight off the trellis; prompts above
GEMM_MAX_ROWS reconstruct the weight into a shared bf16 scratch and run a normal GEMM, the
same split ExLlamaV3 makes (modules/quant/exl3.py:10, AUTO_RECONSTRUCT_THRESHOLD = 144).
The wheel's own C++ wrapper allocates its Hadamard buffer per call when rows > 1
(exllamav3_ext/libtorch/linear.cpp:41-43), which a CUDA graph cannot hold, so this module
owns fixed fp16 input / Hadamard / output buffers, allocated once after weight load.
Known limit: the wheel's one-token GEMV only covers K 2-4 (exl3_gemv.cu:115-122); the K=5
dense layers of the 3.05bpw build take the general GEMM kernel.

The wheel routines are reached through two seams (_exl3_gemm, _reconstruct_into) so CPU tests
can substitute the pure-torch oracle in kernel/exl3.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F

from freetoken.kernel import exl3 as _exl3
from freetoken.layers.base import BaseOP, OPList, _concat_prefix

GEMM_MAX_ROWS = 144
_COMPONENTS = ("trellis", "suh", "svh", "mul1")


def _load_ext():
    try:
        import exllamav3_ext
    except ImportError as exc:  # pragma: no cover - depends on the optional wheel
        raise RuntimeError("dense EXL3 linears need the ExLlamaV3 exllamav3_ext wheel") from exc
    return exllamav3_ext


def _exl3_gemm(x16, trellis, y16, suh, xh, svh) -> None:
    # Signature verified on the 5090's 1.4.6 wheel in Task 0: exl3_gemm(A, B, C, suh, A_had,
    # svh, force_shape_idx, mcg, mul1, force_num_sms). Adapt here only if Task 0 differs.
    _load_ext().exl3_gemm(x16, trellis, y16, suh, xh, svh, -1, False, True, 0)


def _reconstruct_into(op: "Exl3Linear", out: torch.Tensor, work: torch.Tensor) -> torch.Tensor:
    return _exl3.reconstruct(op.trellis, op.suh, op.svh, k=op.k, codebook="mul1", out=out, work=work)


@dataclass
class Exl3DenseWorkspace:
    device: torch.device
    max_rows: int
    max_in: int
    max_out: int
    recon_elems: int
    x16: torch.Tensor
    xh: torch.Tensor
    y16: torch.Tensor
    recon_work: torch.Tensor
    recon_out: torch.Tensor


_WORKSPACES: dict[torch.device, Exl3DenseWorkspace] = {}


def _dev_key(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def validate_exl3_parts(name, trellis, suh, svh, mul1, in_features, out_features) -> int:
    if trellis.dtype != torch.int16 or trellis.dim() != 3:
        raise ValueError(f"{name}.trellis must be rank-3 int16, got {trellis.dtype} {tuple(trellis.shape)}")
    if trellis.shape[0] * 16 != in_features or trellis.shape[1] * 16 != out_features:
        raise ValueError(f"{name}.trellis {tuple(trellis.shape)} does not match [{in_features}, {out_features}]")
    last = int(trellis.shape[2])
    if last % 16 or not 1 <= last // 16 <= 8:
        raise ValueError(f"{name}.trellis last dim {last} is not 16*K for an integer K in 1..8")
    if suh.dtype != torch.float16 or tuple(suh.shape) != (in_features,):
        raise ValueError(f"{name}.suh must be fp16 [{in_features}], got {suh.dtype} {tuple(suh.shape)}")
    if svh.dtype != torch.float16 or tuple(svh.shape) != (out_features,):
        raise ValueError(f"{name}.svh must be fp16 [{out_features}], got {svh.dtype} {tuple(svh.shape)}")
    if mul1.dtype != torch.int32 or mul1.dim() != 0:
        raise ValueError(f"{name}.mul1 must be a scalar int32 marker, got {mul1.dtype} {tuple(mul1.shape)}")
    return last // 16


class Exl3Linear(BaseOP):
    def __init__(self, in_features: int, out_features: int, has_bias: bool = False, *,
                 k_hint: int = 5, allow_reconstruct: bool = True):
        if in_features % 16 or out_features % 128:
            raise ValueError(f"EXL3 linear needs in%16 == 0 and out%128 == 0, got [{in_features}, {out_features}]")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.k = int(k_hint)
        self.allow_reconstruct = bool(allow_reconstruct)
        self.trellis = torch.empty(in_features // 16, out_features // 16, 16 * self.k, dtype=torch.int16)
        self.suh = torch.empty(in_features, dtype=torch.float16)
        self.svh = torch.empty(out_features, dtype=torch.float16)
        self.mul1 = torch.empty((), dtype=torch.int32)
        self.bias = torch.empty(out_features) if has_bias else None

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        parts = {}
        for comp in _COMPONENTS:
            key = _concat_prefix(prefix, comp)
            if key not in state_dict:
                raise KeyError(f"EXL3 linear {prefix!r} is missing {comp} ({key})")
            parts[comp] = state_dict.pop(key)
        self.k = validate_exl3_parts(prefix, parts["trellis"], parts["suh"], parts["svh"],
                                     parts["mul1"], self.in_features, self.out_features)
        self.trellis = parts["trellis"].contiguous()
        self.suh = parts["suh"].contiguous()
        self.svh = parts["svh"].contiguous()
        self.mul1 = parts["mul1"]
        if self.bias is not None:
            self.bias = state_dict.pop(_concat_prefix(prefix, "bias"))
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    @property
    def resident_bytes(self) -> int:
        tensors = [self.trellis, self.suh, self.svh] + ([self.bias] if self.bias is not None else [])
        return sum(t.numel() * t.element_size() for t in tensors)

    def _workspace(self, device) -> Exl3DenseWorkspace:
        ws = _WORKSPACES.get(_dev_key(device))
        if ws is None:
            raise RuntimeError("dense EXL3 workspace missing: call prepare_exl3_dense_workspace after weight load")
        return ws

    def _gemm(self, x2: torch.Tensor) -> torch.Tensor:
        ws = self._workspace(x2.device)
        rows = x2.shape[0]
        x16 = ws.x16[: rows * self.in_features].view(rows, self.in_features)
        xh = ws.xh[: rows * self.in_features].view(rows, self.in_features)
        y16 = ws.y16[: rows * self.out_features].view(rows, self.out_features)
        x16.copy_(x2)
        _exl3_gemm(x16, self.trellis, y16, self.suh, xh, self.svh)
        # A fresh bf16 tensor: y16 is shared by every EXL3 linear on this card.
        return y16.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        rows = x2.shape[0]
        if rows > GEMM_MAX_ROWS and self.allow_reconstruct:
            ws = self._workspace(x2.device)
            n = self.in_features * self.out_features
            w = _reconstruct_into(
                self,
                ws.recon_out[:n].view(self.out_features, self.in_features),
                ws.recon_work[:n].view(self.in_features, self.out_features),
            )
            y = F.linear(x2.to(torch.bfloat16), w, self.bias)
        else:
            if rows <= GEMM_MAX_ROWS:
                y = self._gemm(x2)
            else:
                y = torch.cat([self._gemm(x2[i : i + GEMM_MAX_ROWS])
                               for i in range(0, rows, GEMM_MAX_ROWS)])
            if self.bias is not None:
                y = y + self.bias
        return y.view(*lead, self.out_features)


class Exl3ColMerged(BaseOP):
    """Several EXL3 linears on one input, outputs concatenated in ``parts`` order. Trellis
    tensors cannot be concatenated like bf16 weights, so a fused bf16 projection (q|k|v,
    GDN qkv|z, shared gate|up) becomes one GEMM per part."""

    def __init__(self, in_features: int, parts: list[tuple[str, int]], *, k_hint: int = 5):
        self._names = [name for name, _ in parts]
        self.out_features = sum(size for _, size in parts)
        for name, size in parts:
            setattr(self, name, Exl3Linear(in_features, size, k_hint=k_hint))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([getattr(self, name).forward(x) for name in self._names], dim=-1)


class Exl3LMHead(Exl3Linear):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(embedding_dim, num_embeddings, has_bias=False, k_hint=5,
                         allow_reconstruct=False)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_tp = num_embeddings
        self.vocab_range = (0, num_embeddings)
        self.tp_size = 1

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return super().forward(x)


def iter_exl3_linears(root) -> Iterator[Exl3Linear]:
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Exl3Linear):
            yield node
        if isinstance(node, OPList):
            stack.extend(node.op_list)
        if isinstance(node, BaseOP):
            for value in vars(node).values():
                if isinstance(value, BaseOP):
                    stack.append(value)
                elif isinstance(value, (list, tuple)):
                    stack.extend(v for v in value if isinstance(v, BaseOP))


def _need(root) -> tuple[int, int, int] | None:
    ops = list(iter_exl3_linears(root))
    if not ops:
        return None
    max_in = max(op.in_features for op in ops)
    max_out = max(op.out_features for op in ops)
    recon = max((op.in_features * op.out_features for op in ops if op.allow_reconstruct), default=0)
    return max_in, max_out, recon


def prepare_exl3_dense_workspace(root, device, *, _reset: bool = False):
    need = _need(root)
    if need is None:
        return None
    key = _dev_key(device)
    if _reset:
        _WORKSPACES.pop(key, None)
    if key in _WORKSPACES:
        require_exl3_dense_workspace_fits(root, device)
        return _WORKSPACES[key]
    max_in, max_out, recon = need
    rows = GEMM_MAX_ROWS
    ws = Exl3DenseWorkspace(
        device=key, max_rows=rows, max_in=max_in, max_out=max_out, recon_elems=recon,
        x16=torch.empty(rows * max_in, dtype=torch.float16, device=key),
        xh=torch.empty(rows * max_in, dtype=torch.float16, device=key),
        y16=torch.empty(rows * max_out, dtype=torch.float16, device=key),
        recon_work=torch.empty(recon, dtype=torch.float16, device=key),
        recon_out=torch.empty(recon, dtype=torch.bfloat16, device=key),
    )
    _WORKSPACES[key] = ws
    return ws


def require_exl3_dense_workspace_fits(root, device) -> None:
    need = _need(root)
    if need is None:
        return
    ws = _WORKSPACES.get(_dev_key(device))
    if ws is None:
        raise RuntimeError("dense EXL3 workspace missing: call prepare_exl3_dense_workspace after weight load")
    max_in, max_out, recon = need
    if max_in > ws.max_in or max_out > ws.max_out or recon > ws.recon_elems:
        raise RuntimeError(
            f"dense EXL3 workspace is too small (in {ws.max_in}/{max_in}, out {ws.max_out}/{max_out}, "
            f"reconstruct {ws.recon_elems}/{recon}); it cannot grow after CUDA graphs are captured"
        )


__all__ = ["Exl3ColMerged", "Exl3DenseWorkspace", "Exl3LMHead", "Exl3Linear", "GEMM_MAX_ROWS",
           "iter_exl3_linears", "prepare_exl3_dense_workspace", "require_exl3_dense_workspace_fits",
           "validate_exl3_parts"]
```

Check `from freetoken.core import get_global_ctx` is the import `layers/embedding.py` uses (copy its import line exactly).

- [ ] **Step 4: Run CPU tests** → PASS. Then on the box: `cd ~/FreeToken-exl3 && git pull && ~/FreeToken/.venv/bin/python -m pytest tests/kernels/test_exl3_linear.py -q` (no server boot needed; the tests use < 1 GB of card memory — if `nvidia-smi` shows less than 2 GB free, stop and ask Jay before running). Expected: all PASS; paste the output.

- [ ] **Step 5: Commit** `git add ... && git commit -m "feat: dense EXL3 linear that keeps the trellis packed on the card"`

---

### Task 4: Wire EXL3 into the Qwen model and loader

**Files:**
- Modify: `python/freetoken/models/qwen4_exp/attention.py:75-137`
- Modify: `python/freetoken/models/qwen4_exp/gdn.py:60-117,175-187` and `model.py:35-55` (pass `linear_storage`)
- Modify: `python/freetoken/models/qwen3_5_moe/moe.py:22-53`
- Modify: `python/freetoken/models/qwen4_exp/model.py:223-243` (lm_head)
- Modify: `python/freetoken/engine/spec_lmhead.py:67` (`_QUANTIZED_HEADS`)
- Modify: `python/freetoken/models/qwen4_exp/weight.py:118-213` (rename + fusions)
- Modify: `python/freetoken/engine/engine.py:725` (workspace prep)
- Test: `tests/models/qwen4_exp/test_exl3_weight.py`

**Interfaces:**
- Consumes: Task 2 `config.linear_storage`; Task 3 ops.
- Produces: `weight.is_exl3_checkpoint(model_path: str) -> bool`; `weight._exl3_rename(name: str) -> str`; state keys for exl3 models: `model.layers.N.self_attn.qkv_proj.{q,k,v}_proj.{trellis,suh,svh,mul1}`, `...linear_attn.in_proj_qkvz.{in_proj_qkv,in_proj_z}.*`, `...linear_attn.in_proj_ba.weight`, `...mlp.shared_expert.gate_up_proj.{gate_proj,up_proj}.*`, `lm_head.*`.

- [ ] **Step 1: Failing tests**

```python
"""EXL3 Qwen Flash dense wiring and loader (spec 2026-09-25 section 2)."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.kernel.exl3_linear import Exl3ColMerged, Exl3LMHead, Exl3Linear
from freetoken.models.qwen4_exp import weight as W


def test_exl3_rename_nests_fused_parts():
    assert W._exl3_rename("model.layers.3.self_attn.q_proj.trellis") == \
        "model.layers.3.self_attn.qkv_proj.q_proj.trellis"
    assert W._exl3_rename("model.layers.0.linear_attn.in_proj_z.svh") == \
        "model.layers.0.linear_attn.in_proj_qkvz.in_proj_z.svh"
    assert W._exl3_rename("model.layers.0.mlp.shared_expert.up_proj.mul1") == \
        "model.layers.0.mlp.shared_expert.gate_up_proj.up_proj.mul1"
    # not an EXL3 component or not a fused part: unchanged
    assert W._exl3_rename("model.layers.0.linear_attn.in_proj_b.weight") == \
        "model.layers.0.linear_attn.in_proj_b.weight"
    assert W._exl3_rename("model.layers.3.self_attn.o_proj.trellis") == \
        "model.layers.3.self_attn.o_proj.trellis"


def test_exl3_modules_are_built(exl3_model):
    layer_attn = exl3_model.model.layers.op_list[3].self_attn
    assert isinstance(layer_attn.qkv_proj, Exl3ColMerged)
    assert isinstance(layer_attn.o_proj, Exl3Linear)
    assert isinstance(layer_attn.indexer.index_qk_proj, Exl3Linear)
    gdn = exl3_model.model.layers.op_list[0].linear_attn
    assert isinstance(gdn.in_proj_qkvz, Exl3ColMerged) and isinstance(gdn.out_proj, Exl3Linear)
    assert isinstance(exl3_model.model.layers.op_list[0].mlp.shared_expert.gate_up_proj, Exl3ColMerged)
    assert isinstance(exl3_model.lm_head, Exl3LMHead)


def test_iter_weights_matches_model_state(exl3_checkpoint, exl3_model):
    got = {name for name, _ in W.iter_weights(exl3_checkpoint, torch.device("cpu"),
                                              include_moe_experts=False, include_non_moe=True,
                                              include_vision=False)}
    want = {k for k in exl3_model.state_dict() if ".ple_embedding.ngram" not in k}
    assert got == want


def test_nvfp4_iter_weights_unchanged(nvfp4_checkpoint_names):
    # the NVFP4 key set and fusions are exactly the pre-change ones
    before, after = nvfp4_checkpoint_names
    assert before == after
```

Fixtures (put in the same file): `exl3_model` builds `Qwen4ExpForCausalLM` from `toy_hf_config(num_layers=4)` with `quantization_config=_EXL3_QUANT` (from Task 2's test) under the model-build context the existing `test_skeleton.py` uses (copy its setup lines); `exl3_checkpoint` walks `exl3_model.state_dict()` and writes a tiny checkpoint with the **checkpoint** names — reverse `_exl3_rename`, prefix `model.language_model.`, give the indexer K=3 and everything else K=5 so the loader has to adopt mixed K (Review Focus 1); split `in_proj_ba.weight` back into `in_proj_b.weight`/`in_proj_a.weight`; plus `config.json` with the exl3 quant block and an index. `nvfp4_checkpoint_names` runs `iter_weights` over the existing NVFP4 toy fixture used by `test_weight_ckpt.py` (reuse its builder) and compares with a key list captured from `git stash`-free logic: build the list from `_FUSIONS` expectations exactly as `test_weight.py` already asserts — if `test_weight.py` already pins the NVFP4 key set, drop this test and rely on it.

- [ ] **Step 2: Run** `PYTHONPATH=python .venv/bin/python -m pytest tests/models/qwen4_exp/test_exl3_weight.py -q` → FAIL.

- [ ] **Step 3: Build the EXL3 ops in the modules**

`attention.py` — indexer:

```python
        if getattr(config, "linear_storage", "bf16") == "exl3":
            # 3.05bpw_h5_ng5 stores index_qk_proj at K=3 (the rest of attention is K=5).
            self.index_qk_proj = Exl3Linear(args.hidden_size, sum(self._split), k_hint=3)
        else:
            self.index_qk_proj = make_dense_replicated(...)  # existing call unchanged
```

attention — qkv/o:

```python
        if getattr(config, "linear_storage", "bf16") == "exl3":
            # EXL3 ships q/k/v separately packed; q carries the output gate (2*qo rows).
            self.qkv_proj = Exl3ColMerged(config.hidden_size, [
                ("q_proj", self.qo_attn_dim * 2), ("k_proj", self.kv_attn_dim), ("v_proj", self.kv_attn_dim)])
            self.o_proj = Exl3Linear(self.qo_attn_dim, config.hidden_size)
        else:
            ...existing two calls...
```

`gdn.py` — add constructor parameter `linear_storage: str = "bf16"`; `self._exl3 = linear_storage == "exl3"`; new first branch:

```python
        if self._exl3:
            # EXL3 packs qkv and z (K=5) but ships b and a as plain fp16 [48, 2560]; mirror the
            # fp8 split: two packed GEMMs for qkv|z plus one bf16 GEMM for b|a.
            self.in_proj_qkvz = Exl3ColMerged(
                hidden_size, [("in_proj_qkv", self.conv_dim), ("in_proj_z", self.value_dim)])
            self.in_proj_ba = LinearColParallelMerged(hidden_size, [num_v_heads, num_v_heads], has_bias=False)
        elif self._fp8:
```
and `out_proj`: `Exl3Linear(self.value_dim, hidden_size) if self._exl3 else make_replicated_quant(...)`. In `forward`, change `if self._fp8:` to `if self._fp8 or self._exl3:`. `model.py:build_linear_mixer` passes `linear_storage=getattr(config, "linear_storage", "bf16")`.

`qwen3_5_moe/moe.py` `_SharedExpert.__init__` — new first branch:

```python
        if getattr(config, "linear_storage", "bf16") == "exl3":
            from freetoken.kernel.exl3_linear import Exl3ColMerged, Exl3Linear

            self.gate_up_proj = Exl3ColMerged(
                hidden_size, [("gate_proj", intermediate_size), ("up_proj", intermediate_size)])
            self.down_proj = Exl3Linear(intermediate_size, hidden_size)
        elif getattr(config, "expert_quant", "none") == "fp8_block":
```

`model.py` lm_head — new first branch:

```python
        if getattr(config, "lm_head_quant", "none") == "exl3":
            from freetoken.kernel.exl3_linear import Exl3LMHead

            assert not config.tie_word_embeddings, "EXL3 lm_head assumes untied embeddings"
            self.lm_head = Exl3LMHead(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        elif ... nvfp4 (existing)
```

`spec_lmhead.py:67`: `_QUANTIZED_HEADS = ("Nvfp4LMHead", "Int8LMHead", "Exl3LMHead")`.

- [ ] **Step 4: Loader**

In `qwen4_exp/weight.py`:

```python
_EXL3_COMPONENT_SUFFIXES = (".trellis", ".suh", ".svh", ".mul1")
# checkpoint part -> nested module path inside the Exl3ColMerged that replaces a bf16 fusion
_EXL3_NESTED = {
    ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
    ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
    ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
    ".linear_attn.in_proj_qkv.": ".linear_attn.in_proj_qkvz.in_proj_qkv.",
    ".linear_attn.in_proj_z.": ".linear_attn.in_proj_qkvz.in_proj_z.",
    ".mlp.shared_expert.gate_proj.": ".mlp.shared_expert.gate_up_proj.gate_proj.",
    ".mlp.shared_expert.up_proj.": ".mlp.shared_expert.gate_up_proj.up_proj.",
}
# bf16 fusions that still apply to an EXL3 checkpoint: HC as today, and GDN b|a (fp16 in the
# checkpoint) now that qkv|z are packed separately.
_EXL3_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    key: value for key, value in _FUSIONS.items() if "hyper_connection" in key
}
_EXL3_FUSIONS[".linear_attn.in_proj_ba.weight"] = (
    (".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight"), 0)


def is_exl3_checkpoint(model_path: str) -> bool:
    folder = download_hf_weight(model_path)
    try:
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as fh:
            quant = json.load(fh).get("quantization_config") or {}
    except FileNotFoundError:
        return False
    return str(quant.get("quant_method") or "").lower() == "exl3"


def _exl3_rename(name: str) -> str:
    if not name.endswith(_EXL3_COMPONENT_SUFFIXES):
        return name
    for part, nested in _EXL3_NESTED.items():
        if part in name:
            return name.replace(part, nested, 1)
    return name
```

`_try_fuse` gains a `fusions` parameter (default `_FUSIONS`); `iter_weights` computes `exl3 = is_exl3_checkpoint(model_path)` once, uses `fusions = _EXL3_FUSIONS if exl3 else _FUSIONS`, and after `name = _rename(...)` does `if exl3: name = _exl3_rename(name)`. For exl3, also force `mapped_vision = None` (the mmap view assumes one bf16 extent, `weight.py:72-76`) and log once: `logger.info("EXL3 checkpoint: picture weights served from RAM (mmap needs one bf16 extent)")`.

- [ ] **Step 5: Engine workspace**

`engine/engine.py`, directly after `self.model.load_state_dict(self._load_weight_state_dict(config))` (:725):

```python
        # Dense EXL3 linears share one fixed fp16 workspace; allocate it before the KV budget
        # and before any CUDA graph captures its addresses. No-op for every other checkpoint.
        from freetoken.kernel.exl3_linear import prepare_exl3_dense_workspace

        prepare_exl3_dense_workspace(self.model, self.device)
```

Confirm by reading the surrounding code that the cache budget (`cache_budget.py` callers) runs after this line; if it runs before, move the call to just after the model is materialised and before the budget.

- [ ] **Step 6: Run** `PYTHONPATH=python .venv/bin/python -m pytest tests/models/qwen4_exp -q` → all PASS (new and old).

- [ ] **Step 7: Commit** `git commit -am "feat: build Qwen Flash attention, GDN, shared expert and lm_head from EXL3 weights"` (add the new test file first).

---

### Task 5: Routed experts at any K, and graphs at top_k 10

**Files:**
- Modify: `python/freetoken/models/exl3_banks.py` (:46, :199-205, :216-240, :297-345, :402-425, :517-575)
- Modify: `python/freetoken/moe/fused_exl3.py` (:24-26, :75-125, :285-305, :455-535, :938-965)
- Modify: `python/freetoken/kernel/exl3_mgemm.py` (:44, :204-256)
- Modify: `python/freetoken/moe/offload_cache.py:101-120`, `kernel/aot_models.py:105-125`, `moe/expert_banks.py:445-460`, `engine/memory_plan.py:268-300`, `engine/engine.py:1220-1250`
- Test: `tests/models/test_exl3_banks.py`, `tests/moe/test_fused_exl3.py`, `tests/engine/test_exl3_backend.py`

**Interfaces:**
- Consumes: `ModelConfig.exl3_expert_k` (Task 2).
- Produces: `exl3_banks._expected_shape(proj, kind, hidden, intermediate, k)`, `_bank_specs(experts, hidden, intermediate, k)`; `offload_cache.bank_bytes_per_expert(fmt: str, hidden: int, intermediate: int, model_config) -> int` (the only way callers size a bank row); `fused_exl3.prepare_exl3_scratch(..., k: int)` and `Exl3Scratch.k`; `Exl3MgemmBanks.from_banks(banks)` accepts any K 1..8 and records it; `decode_is_graph_safe(config)` accepts top_k ≤ 128 when `config.exl3_expert_op == "mgemm"`.

- [ ] **Step 1: Failing tests**

In `tests/models/test_exl3_banks.py` parametrize the writer over K: change `_tensor`/`_shape` calls in `_write_checkpoint` to take `k` (default 2) and add:

```python
def test_loads_k3_banks(tmp_path):
    cfg = SimpleNamespace(**vars(_CONFIG), exl3_expert_k=3)
    folder = _write_checkpoint(tmp_path, k=3)
    sources = load_exl3_expert_sources(folder, cfg)
    assert sources["gate_trellis"][0].shape[-1] == 48


def test_refuses_k_other_than_config(tmp_path):
    cfg = SimpleNamespace(**vars(_CONFIG), exl3_expert_k=2)
    folder = _write_checkpoint(tmp_path, k=3)
    with pytest.raises(ValueError, match="K=3"):
        load_exl3_expert_sources(folder, cfg)


def test_refuses_mixed_expert_k(tmp_path):
    cfg = SimpleNamespace(**vars(_CONFIG), exl3_expert_k=3)
    def mutate(name, value):
        if ".experts.1.down_proj.trellis" in name:
            return name, value[..., :32]  # K=2 among K=3
        return name, value
    folder = _write_checkpoint(tmp_path, mutate, k=3)
    with pytest.raises(ValueError, match="K="):
        load_exl3_expert_sources(folder, cfg)
```

In `tests/moe/test_fused_exl3.py`:

```python
def test_graph_rule_accepts_top10_on_mgemm():
    cfg = SimpleNamespace(max_running_req=1, cuda_graph_bs=None, cuda_graph_max_bs=1,
                          exl3_expert_op="mgemm",
                          model_config=SimpleNamespace(num_experts_per_tok=10))
    assert decode_is_graph_safe(cfg)


def test_graph_rule_keeps_reconstruct_limit():
    cfg = SimpleNamespace(max_running_req=1, cuda_graph_bs=None, cuda_graph_max_bs=1,
                          exl3_expert_op="reconstruct",
                          model_config=SimpleNamespace(num_experts_per_tok=10))
    assert not decode_is_graph_safe(cfg)
```

and a byte-formula test (new, in `tests/moe/test_exl3_cache.py`):

```python
def test_bank_bytes_follow_k():
    from freetoken.moe.offload_cache import bank_bytes_per_expert
    glm = bank_bytes_per_expert("exl3", 4096, 2048, SimpleNamespace(exl3_expert_k=2))
    assert glm == 6_328_320  # the GLM-5.3 2.05bpw figure measured 2026-09-04
    qwen = bank_bytes_per_expert("exl3", 2560, 640, SimpleNamespace(exl3_expert_k=3))
    assert qwen == 3 * 160 * 40 * 48 * 2 + 2 * (2560 + 640) * 2 + (640 + 2560) * 2
    assert bank_bytes_per_expert("nvfp4", 2560, 640, SimpleNamespace()) == 2_772_480
```

- [ ] **Step 2: Run** those files → new tests FAIL.

- [ ] **Step 3: Implement**

- `exl3_banks.py`: delete `_EXL3_K`; `_expected_shape(..., k)` uses `16 * k`; `_collect_records` returns geometry plus `k = int(getattr(config, "exl3_expert_k", 2))` as a sixth element; `_validate_headers` compares each trellis K with that `k` and raises `f"EXL3 {record.name!r} has K={found}; the config expects K={k} for every routed expert"`; `_bank_specs(experts, hidden, intermediate, k)`; `_alloc_banks`, `_validate_loaded_tensor`, `dummy_exl3_expert_sources` pass `k`. Update the module docstring's "fixed K=2/mul1 GLM" wording to "one K per checkpoint (GLM 2.05bpw K=2, Qwen Flash 3.05bpw K=3)" and the "71 GiB" comments to "the multi-GiB bank set".
- `fused_exl3.py`: remove `_EXL3_K`; add `k: int = 2` to `prepare_exl3_scratch` and a `k` field to `Exl3Scratch`; replace every `16 * _EXL3_K` / `k=_EXL3_K` with `scratch.k` (or the bank's `trellis.shape[-1] // 16` where no scratch is at hand, validated equal to `scratch.k`). `decode_is_graph_safe`:

```python
    op = getattr(config, "exl3_expert_op", None) or DEFAULT_EXL3_EXPERT_OP
    # The packed mgemm route list holds EXL3_MGEMM_MAX_INDICES (128) entries, so one decode
    # row fits any top_k up to 128 (Qwen Flash routes 10). The reconstruct arena stays 8 wide.
    limit = EXL3_MGEMM_MAX_INDICES if op == "mgemm" else _MAX_RECONSTRUCT_EXPERTS
    if not 1 <= top_k <= limit:
        return False
```

  Also, in the decode branch of `fused_experts_exl3`, before falling back to reconstruct-first, add:

```python
        if torch.cuda.is_current_stream_capturing() and top_k > _MAX_RECONSTRUCT_EXPERTS:
            raise RuntimeError(
                f"EXL3 packed mgemm is unavailable and reconstruct-first cannot capture top_k={top_k}; "
                "boot with --cuda-graph-max-bs 0")
```
- `exl3_mgemm.py`: remove the fixed-K check in `from_banks`; require all three trellis banks to share one K in 1..8 and pass it to `cls(banks, ptrs, k=k)`; `exl3_mgemm_shape_supported(..., k)` callers pass `tables.k`. Keep `EXL3_MGEMM_K = 2` only as the default for `exl3_mgemm_shape_supported` so GLM callers are unchanged.
- `offload_cache.py`: replace the `"exl3"` lambda with a K-aware function and add the helper:

```python
def _exl3_bytes(H: int, I: int, k: int = 2) -> int:
    # 16*K uint16 per 16x16 tile for gate, up and down, plus fp16 suh/svh per projection.
    # GLM-5.3 2.05bpw (K=2, H=4096, I=2048): 6,328,320 B; Qwen Flash 3.05bpw (K=3, H=2560,
    # I=640): 1,862,400 B.
    tiles = (H // 16) * (I // 16)
    return 3 * tiles * 16 * k * 2 + 2 * (H + I) * 2 + (I + H) * 2


def bank_bytes_per_expert(fmt: str, hidden: int, intermediate: int, model_config) -> int:
    if fmt == "exl3":
        return _exl3_bytes(hidden, intermediate, int(getattr(model_config, "exl3_expert_k", 2)))
    return _BANK_BYTES_PER_EXPERT[fmt](hidden, intermediate)
```
  keep `_BANK_BYTES_PER_EXPERT["exl3"] = _exl3_bytes` (K=2 default) for any caller not yet migrated; migrate `expert_banks.py:452`, `memory_plan.py:279` and the formula call below it to `bank_bytes_per_expert(...)`.
- `aot_models.py:111-125`: take `k` (default 2) and use `(H // 16) * (I // 16) * 16 * k * 2` for the three trellis entries; pass `getattr(model_config, "exl3_expert_k", 2)` from its caller (read the caller at :305 to find the config in scope).
- `engine.py:1227`: `prepare_exl3_scratch(..., k=int(getattr(config.model_config, "exl3_expert_k", 2)))`.

- [ ] **Step 4: Run** `PYTHONPATH=python .venv/bin/python -m pytest tests/models/test_exl3_banks.py tests/moe/test_fused_exl3.py tests/moe/test_exl3_cache.py tests/engine/test_exl3_backend.py tests/kernels/test_exl3.py -q` → PASS on the devbox; on the box also `tests/kernels/test_exl3_mgemm.py` (GLM real-bank test needs `FREETOKEN_GLM53_EXL3_MODEL=~/models/GLM-5.3-Flash-exl3-2.05bpw`) and a new GPU test in `tests/kernels/test_exl3_mgemm.py`:

```python
@cuda
def test_mgemm_k3_qwen_shape_matches_reconstruct():
    from freetoken.kernel.exl3 import reconstruct
    from freetoken.kernel.exl3_mgemm import Exl3MgemmBanks, fused_experts_exl3_mgemm

    dev, H, I, k, E = torch.device("cuda"), 2560, 640, 3, 4
    g = torch.Generator().manual_seed(0)

    def parts(fin, fout):
        trellis = torch.randint(-32768, 32767, (fin // 16, fout // 16, 16 * k), generator=g,
                                dtype=torch.int32).to(torch.int16)
        suh = (torch.randint(0, 2, (fin,), generator=g) * 2 - 1).half()
        svh = (torch.rand(fout, generator=g) * 0.02 + 0.01).half()
        return trellis, suh, svh

    experts = [(parts(H, I), parts(H, I), parts(I, H)) for _ in range(E)]
    banks = []
    for proj in range(3):
        for comp in range(3):
            banks.append(torch.stack([experts[e][proj][comp] for e in range(E)]).to(dev).contiguous())
    tables = Exl3MgemmBanks.from_banks(banks)
    x = torch.randn(1, H, device=dev).bfloat16()
    ids = torch.tensor([[0, 1, 2, 3]], device=dev, dtype=torch.int32)
    weights = torch.full((1, 4), 0.25, device=dev)
    got = fused_experts_exl3_mgemm(x, tables, weights, ids, activation="silu", swiglu_limit=None).float()

    want = torch.zeros(1, H, device=dev)
    for e in range(E):
        (gt, gs, gv), (ut, us, uv), (dt, ds, dv) = [[t.to(dev) for t in p] for p in experts[e]]
        gw = reconstruct(gt, gs, gv, k=k, codebook="mul1").float()
        uw = reconstruct(ut, us, uv, k=k, codebook="mul1").float()
        dw = reconstruct(dt, ds, dv, k=k, codebook="mul1").float()
        act = torch.nn.functional.silu(x.float() @ gw.T) * (x.float() @ uw.T)
        want += 0.25 * (act @ dw.T)
    assert (got - want).norm() / want.norm() < 1e-2
```
If `fused_experts_exl3_mgemm` names its keyword arguments differently from the signature quoted in Task 5's interface (read `kernel/exl3_mgemm.py:578-590`), follow the file.

- [ ] **Step 5: Commit** `git commit -am "feat: EXL3 routed experts at any K, and decode graphs for top-10 routing on the packed path"`

---

### Task 6: First boot on the 5090 (vision and MTP off, 1 chat)

Operational. **Ask Jay for the OK before starting** — the boot takes over the card and stops the daily server.

**Files:** none tracked. Boot script lives in the session scratchpad on the box.

- [ ] **Step 1: Pre-flight** — on the box: `cd ~/FreeToken-exl3 && git pull`; confirm `freetoken-ple.index.json` exists in the model dir (Task 1 step 8); `free -g`; `nvidia-smi`.
- [ ] **Step 2: Stop the daily server** through the page: `curl -X POST http://127.0.0.1:2031/api/server/stop`; wait until `nvidia-smi` shows < 3 GB used.
- [ ] **Step 3: Boot** from the worktree on a side port (never 2020):

```bash
cd ~/FreeToken-exl3 && PYTHONPATH=python sudo prlimit --memlock=unlimited:unlimited --nofile=1048576:1048576 \
  setpriv --reuid=jay --regid=jay --init-groups env HOME=/home/jay FREETOKEN_BANK_CUDA_ALLOC=1 \
  FREETOKEN_PIN_BUDGET_GB=73 ~/FreeToken/.venv/bin/python -m freetoken.cli serve \
  --model-path ~/models/Qwen3.8-Flash-Next-exl3-3.05bpw --port 2036 --moe-backend offload \
  --max-running-requests 1 --cuda-graph-max-bs 1 --ple-backend disk --num-tokens 65536 \
  > ~/exl3-boot-$(date +%H%M).log 2>&1 &
```
(Copy any other flags the daily `boot-2020.ps1` profile uses that apply, from `~/FreeToken/boot-2020.ps1`; leave vision and MTP off.)
- [ ] **Step 4: Check** `curl 127.0.0.1:2036/health` until serving; then 8 fixed prompts with `chat_template_kwargs.enable_thinking=false`, `temperature 0`, `max_tokens 128`: "What is 17*23?", "Name the capital of Australia.", "Write a Python function that reverses a string.", "Translate 'good morning' into French.", "List three primary colours.", "What year did the Berlin Wall fall?", "Summarise photosynthesis in one sentence.", "Continue: The quick brown fox". Pass bar: every answer correct and coherent. Also send the same 8 to the daily NVFP4 server afterwards (Step 6) and note agreement.
- [ ] **Step 5: Record** boot time, `nvidia-smi` used, `free -g`, log lines "EXL3 routed experts: packed mgemm path selected", tokens/s of the prompts.
- [ ] **Step 6: Stop the side server, restart the daily one** (`curl -X POST http://127.0.0.1:2031/api/server/start`, wait ~100 s, `curl 127.0.0.1:2020/health`). Report.

If the boot or answers fail: use superpowers:systematic-debugging; per-layer agreement (Task 3 GPU tests) first, then compare one layer's output against the fallback reconstruct path.

---

### Task 7: Pictures

**Files:**
- Modify: `python/freetoken/models/qwen4_exp/vision.py:102-190`
- Modify: `python/freetoken/models/vision_weight.py`
- Modify: `python/freetoken/models/qwen4_exp/weight.py` (`_rename` for exl3 vision q/k/v)
- Test: `tests/models/qwen4_exp/test_exl3_vision.py`

**Interfaces:**
- Consumes: `Qwen4VisionConfig.exl3` (Task 2), `Exl3Linear` (Task 3).
- Produces: vision state keys `visual.blocks.N.attn.proj.{trellis,suh,svh,mul1,bias}`, `visual.blocks.N.mlp.linear_fc{1,2}.*`, `visual.merger.linear_fc{1,2}.*`; `attn.qkv.{weight,bias}` stay bf16.

- [ ] **Step 1: Failing tests**

```python
"""EXL3 picture tower (spec 2026-09-25 section 5)."""

import pytest
import torch

from freetoken.kernel.exl3_linear import Exl3Linear
from freetoken.models.qwen4_exp import weight as W
from freetoken.models.vision_weight import require_dense_vision_weight


def test_exl3_vision_skips_packed_qkv_parts():
    for part in ("q_proj", "k_proj", "v_proj"):
        for comp in ("trellis", "suh", "svh", "mul1", "bias"):
            name = f"model.visual.blocks.0.attn.{part}.{comp}"
            assert W._rename(name, include_vision=True, exl3=True) is None


def test_exl3_vision_keeps_bf16_qkv_and_packed_proj():
    assert W._rename("model.visual.blocks.0.attn.qkv.weight", include_vision=True, exl3=True) == \
        "visual.blocks.0.attn.qkv.weight"
    assert W._rename("model.visual.blocks.0.attn.proj.trellis", include_vision=True, exl3=True) == \
        "visual.blocks.0.attn.proj.trellis"


def test_vision_weight_check_allows_named_exl3_linears():
    t = torch.zeros(4, dtype=torch.int16)
    require_dense_vision_weight("visual.blocks.0.mlp.linear_fc1.trellis", t, exl3=True)
    with pytest.raises(NotImplementedError):
        require_dense_vision_weight("visual.patch_embed.proj.trellis", t, exl3=True)
    with pytest.raises(NotImplementedError):
        require_dense_vision_weight("visual.blocks.0.mlp.linear_fc1.trellis", t)  # not exl3


def test_exl3_vision_modules(exl3_vision_config):
    from freetoken.models.qwen4_exp.vision import Qwen4VisionBlock, Qwen4VisionPatchMerger
    block = Qwen4VisionBlock(exl3_vision_config)
    assert isinstance(block.attn.proj, Exl3Linear)
    assert isinstance(block.mlp.linear_fc1, Exl3Linear) and isinstance(block.mlp.linear_fc2, Exl3Linear)
    assert not isinstance(block.attn.qkv, Exl3Linear)
    merger = Qwen4VisionPatchMerger(exl3_vision_config)
    assert isinstance(merger.linear_fc1, Exl3Linear)
```

(`exl3_vision_config` fixture: `dataclasses.replace` of the vision config the existing `test_vision.py` builds, with `exl3=True`; dimensions must be multiples of 128 for out features — use hidden 128, intermediate 256.)

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

- `vision.py`: a helper at module top:

```python
def _vision_linear(config, fin, fout):
    if getattr(config, "exl3", False):
        from freetoken.kernel.exl3_linear import Exl3Linear
        # 3.05bpw_h5_ng5 vision linears are K=5 (vision_bits); layer-stream copies need the
        # workspace block's shapes to equal the loaded ones, so the hint must be exact.
        return Exl3Linear(fin, fout, has_bias=True, k_hint=int(getattr(config, "exl3_k", 5)))
    return LinearReplicated(fin, fout, has_bias=True)
```
  use it for `attn.proj`, `mlp.linear_fc1/fc2`, merger `linear_fc1/fc2`; keep `attn.qkv` as `LinearReplicated` (the checkpoint ships a bf16 `qkv.weight` next to the packed q/k/v). Add `exl3_k: int = 5` to `Qwen4VisionConfig` and set it from `vision_bits` in Task 2's parse (`int(get("vision_bits") or 5)`).
- `weight.py`: `_rename(raw_name, *, include_vision=False, exl3=False)`; when `exl3` and the name matches `r"^(model\.)?visual\.blocks\.\d+\.attn\.[qkv]_proj\."`, return `None`. `iter_weights` passes `exl3=exl3`.
- `vision_weight.py`: `require_dense_vision_weight(name, tensor, *, exl3=False)`; when `exl3` and the name ends with an EXL3 component and its module is one of `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`, `merger.linear_fc1`, `merger.linear_fc2`, return the tensor without the dtype check. Every caller passes `exl3=` (the qwen4 `iter_weights` and the FTW path in `models/weight.py` — FTW passes `False`).

- [ ] **Step 4: Run** `tests/models/qwen4_exp -q` → PASS. Then on the box (GPU, no server): extend `tests/models/qwen4_exp/test_vision.py`'s layer-stream test with an exl3 variant if it has one that runs on CUDA; otherwise add `test_exl3_layer_stream_block_copy` building two blocks (CPU-held + GPU workspace) with `exl3=True`, loading random K=5 parts into the CPU one, calling the module's `_copy_component_state_` and checking `trellis` equality on the GPU side.
- [ ] **Step 5: Commit** `git commit -am "feat: EXL3 picture tower with packed projections and the shipped bf16 qkv"`
- [ ] **Step 6: Box boot with pictures** (Jay's OK) — Task 6's boot with `FREETOKEN_LOAD_VISION=1 FREETOKEN_VISION_EXECUTION=layer-stream` (use the exact env names from `models/config.py` `vision_load_enabled` / `vision_execution_mode`). Send the red-circle probe the earlier sessions used (`reasoning_effort none`); expect "a red circle". Stop, restore daily server.

---

### Task 8: Guess-ahead helper (MTP) from the EXL3 checkpoint

**Files:**
- Modify: `python/freetoken/models/qwen4_exp/mtp_spike.py` (:97-125 derive config, :1198-1225 fc layers, :1716-1760 plan builder, new `MTPExl3ExpertBanks` + `MTPExl3GPUExpertRunner`)
- Modify: `python/freetoken/engine/spec_draft.py:112-178` (placement `exl3`)
- Modify: `python/freetoken/models/exl3_banks.py` (stack helper for a key prefix)
- Test: `tests/models/qwen4_exp/test_exl3_mtp.py`

**Interfaces:**
- Consumes: Tasks 3-5.
- Produces: `exl3_banks.stack_exl3_experts(tensor_of: Callable[[str], Tensor], *, prefix: str, experts: int, hidden: int, intermediate: int, k: int) -> dict[str, Tensor]` (nine CPU banks keyed by `EXL3_BANK_NAMES`); `MTPExl3ExpertBanks(banks: dict, k: int)` with `quant_format = "exl3"`, `num_experts`, `hidden_size`, `intermediate_size`, `bytes_per_expert`, `from_store(store, config)`; `MTPExl3GPUExpertRunner(banks, *, top_k, activation, renormalize, max_tokens, num_threads, device, max_gather_rows=None)` with the same `route`/`run_routed`/`forward`/`raise_if_unhealthy`/`close` contract as `MTPGPUExpertRunner`; `resolve_spec_expert_placement(environ=None, *, model_config=None) -> ("exl3", None)` for exl3 checkpoints.

- [ ] **Step 1: Failing tests**

```python
"""EXL3 MTP head (spec 2026-09-25 section 6)."""

import pytest
import torch

from freetoken.models.qwen4_exp import mtp_spike as M


def _raw_exl3_mtp_names(experts=4):
    names = []
    for mod in ("fc_embedding", "fc_hidden"):
        names += [f"mtp.{mod}.{c}" for c in ("trellis", "suh", "svh", "mul1")]
    for part in ("q_proj", "k_proj", "v_proj", "o_proj"):
        names += [f"mtp.layers.0.self_attn.{part}.{c}" for c in ("trellis", "suh", "svh", "mul1")]
    for part in ("gate_proj", "up_proj", "down_proj"):
        names += [f"mtp.layers.0.mlp.shared_expert.{part}.{c}" for c in ("trellis", "suh", "svh", "mul1")]
        for e in range(experts):
            names += [f"mtp.layers.0.mlp.experts.{e}.{part}.{c}" for c in ("trellis", "suh", "svh", "mul1")]
    return names


def test_plan_maps_exl3_dense_and_leaves_experts_to_banks():
    raw = _raw_exl3_mtp_names()
    model_names = [
        "fc_embedding.trellis", "fc_embedding.suh", "fc_embedding.svh", "fc_embedding.mul1",
        "layers.0.self_attn.qkv_proj.q_proj.trellis", "layers.0.self_attn.o_proj.mul1",
        "layers.0.mlp.shared_expert.gate_up_proj.up_proj.svh",
    ]
    plan = M.build_mtp_weight_plan(raw, model_names, exl3=True, strict=False)
    by_model = {e.model_name: e.raw_names for e in plan.entries}
    assert by_model["layers.0.self_attn.qkv_proj.q_proj.trellis"] == ("mtp.layers.0.self_attn.q_proj.trellis",)
    assert by_model["layers.0.mlp.shared_expert.gate_up_proj.up_proj.svh"] == (
        "mtp.layers.0.mlp.shared_expert.up_proj.svh",)


def test_plan_rejects_unknown_raw_names_in_exl3_mode():
    raw = _raw_exl3_mtp_names() + ["mtp.surprise.weight"]
    with pytest.raises(ValueError, match="unexpected MTP source"):
        M.build_mtp_weight_plan(raw, [], exl3=True)


def test_stack_exl3_experts_orders_banks():
    from freetoken.models.exl3_banks import EXL3_BANK_NAMES, stack_exl3_experts
    H, I, k, E = 128, 256, 3, 4
    def tensor_of(name):
        e = int(name.split(".experts.")[1].split(".")[0])
        if name.endswith(".trellis"):
            shape = (H // 16, I // 16, 16 * k) if "down" not in name else (I // 16, H // 16, 16 * k)
            return torch.full(shape, e, dtype=torch.int16)
        if name.endswith(".mul1"):
            return torch.tensor(0, dtype=torch.int32)
        n = (H if name.endswith("suh") else I) if "down" not in name else (I if name.endswith("suh") else H)
        return torch.full((n,), float(e), dtype=torch.float16)
    banks = stack_exl3_experts(tensor_of, prefix="mtp.layers.0.mlp.experts", experts=E,
                               hidden=H, intermediate=I, k=k)
    assert list(banks) == list(EXL3_BANK_NAMES)
    assert banks["gate_trellis"].shape == (E, H // 16, I // 16, 48)
    assert int(banks["down_trellis"][2].flatten()[0]) == 2


def test_placement_is_exl3_for_exl3_checkpoints():
    from types import SimpleNamespace
    from freetoken.engine.spec_draft import resolve_spec_expert_placement
    cfg = SimpleNamespace(expert_quant="exl3", linear_storage="exl3")
    assert resolve_spec_expert_placement({}, model_config=cfg) == ("exl3", None)
    with pytest.raises(ValueError, match="exl3"):
        resolve_spec_expert_placement({"FREETOKEN_MTP_SPEC_EXPERT_FORMAT": "nvfp4"}, model_config=cfg)
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

- `derive_mtp_model_config`: keep `linear_storage` and `exl3_expert_k` from `base` (they are carried by `replace` automatically — add a comment that they deliberately survive, and set `lm_head_quant` untouched).
- `Qwen4ExpMTPModel.__init__`: `fc_embedding`/`fc_hidden` become `Exl3Linear(config.hidden_size, config.hidden_size, k_hint=4)` when `config.linear_storage == "exl3"` (3.05bpw_h5_ng5 headers: `mtp.fc_*` K=4). `Exl3Linear.forward` already flattens leading dims, so `fc_hidden` on `[T, hc, H]` works.
- `build_mtp_weight_plan(raw_names, expected_model_names, *, exl3: bool = False, strict: bool = True)`: when `exl3`, the source for a model name ending in an EXL3 component is `"mtp." + _exl3_unnest(model_name)`, where `_exl3_unnest` reverses `weight._EXL3_NESTED` (import it); raw names matching `mtp.layers.0.mlp.experts.\d+.` are removed from the "unexpected" check (the expert bank loader consumes them); `_MTP_FUSIONS` entries for qkv and shared gate_up are skipped, the HC fusions kept. `strict=False` (test only) skips the "every raw name used" check for dense names so the test can pass a partial model list; production calls keep `strict=True`.
- `exl3_banks.stack_exl3_experts`: build `_bank_specs(experts, hidden, intermediate, k)` tensors on CPU, fill row `e` of each bank from `tensor_of(f"{prefix}.{e}.{proj}.{kind}")` with `_validate_loaded_tensor`-style checks (shape via `_expected_shape(..., k)`, dtype), return a dict in `EXL3_BANK_NAMES` order.
- `MTPExl3ExpertBanks.from_store(store, config)`: `stack_exl3_experts(store.tensor, prefix="mtp.layers.0.mlp.experts", experts=config.num_experts, hidden=config.hidden_size, intermediate=config.moe_intermediate_size, k=config.exl3_expert_k)`; `bytes_per_expert = bank_bytes_per_expert("exl3", H, I, config)`.
- `MTPExl3GPUExpertRunner`: copy `MTPGPUExpertRunner.__init__` validation; move the nine banks to the device once; `self.tables = Exl3MgemmBanks.from_banks([self.banks9[n] for n in EXL3_BANK_NAMES])`; `self.scratch = prepare_exl3_mgemm_scratch(device=device, max_rows=EXL3_MGEMM_MAX_INDICES, max_features=max(H, I), preallocate_fused=True)`; `run_routed` validates like the bf16 runner, then tiles rows by `EXL3_MGEMM_MAX_INDICES // top_k` (12 rows at top_k 10) and calls `fused_experts_exl3_mgemm(hidden, self.tables, topk_weights, topk_ids, activation="silu", swiglu_limit=None, scratch=self.scratch)` per tile, concatenating. `resident_bytes` = sum of the nine banks. Report `stats` like the bf16 runner.
- `spec_draft.py`: `resolve_spec_expert_placement(environ=None, *, model_config=None)`: if `getattr(model_config, "expert_quant", None) == "exl3"`: return `("exl3", None)` unless the env names another format, which raises `ValueError("this EXL3 checkpoint's MTP experts are EXL3; unset FREETOKEN_MTP_SPEC_EXPERT_FORMAT")`. `spec_expert_runner_type("exl3")` → `MTPExl3GPUExpertRunner`; `load_spec_expert_banks("exl3", None, store)` → `MTPExl3ExpertBanks.from_store(store, <model config>)` (thread the model config through from the caller at :368 — read it to find the in-scope config); the weight-plan call passes `exl3=(model_config.linear_storage == "exl3")`. After building the draft model, call `require_exl3_dense_workspace_fits(draft_model, device)`.

- [ ] **Step 4: Run** `tests/models/qwen4_exp/test_exl3_mtp.py tests/models/qwen4_exp/test_mtp_spike.py -q` and `tests/engine` → PASS. GPU test on the box: add to `test_exl3_mtp.py` a `@cuda` test that runs `MTPExl3GPUExpertRunner.run_routed` on 4 synthetic experts (H=2560, I=640, K=3) for 1 and 30 rows and compares with a reconstruct-based reference (relative error < 1e-2).
- [ ] **Step 5: Commit** `git commit -am "feat: MTP draft head straight from the EXL3 checkpoint, experts packed on the card"`
- [ ] **Step 6: Box boot with MTP** (Jay's OK): Task 7's boot plus `FREETOKEN_MTP_SPECULATE=1 FREETOKEN_MTP_SPEC_GRAPH=1` and the other MTP knobs the daily profile sets (copy from `~/FreeToken/boot-2020.ps1` / the settings page boot file; `--ple-backend` must not be `disk` with MTP — use `mmap`). Record acceptance rate and tokens/s on the 8 prompts. Stop, restore.

---

### Task 9: Control panel and switcher

**Files:**
- Modify: `python/freetoken/daemon/settings/model_info.py:110-230`
- Modify: `python/freetoken/daemon/settings/model_detect.py:233-260`
- Modify: the registry entry on the box (`~/.config/freetoken/registry.json`) through the page's Add-model flow — no hand edit
- Test: `tests/settings/test_model_info.py`, `tests/settings/test_model_detect.py`

**Interfaces:**
- Produces: `expert_format(config) == "exl3"` for `quant_method: exl3`; `BYTES_PER_EXPERT["exl3"]` K-aware via a new `expert_bytes(fmt, H, I, *, k=2)`; `ModelInfo.extra["exl3_expert_k"]`; `model_detect` warning text "EXL3 word table not converted yet: run scripts/exl3/convert_ngram_table.py <folder>".

- [ ] **Step 1: Failing tests**

```python
def test_exl3_expert_format_and_size(tmp_path):
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"],
           "quantization_config": {"quant_method": "exl3", "bits": 3.05, "head_bits": 5},
           "text_config": {"hidden_size": 2560, "moe_intermediate_size": 640, "num_experts": 512,
                           "num_hidden_layers": 48, "num_experts_per_tok": 10}}
    assert model_info.expert_format(cfg) == "exl3"
    info = model_info.describe_config(cfg, "q")
    assert info.expert_format_label == "EXL3 (3-bit experts)"
    assert info.bytes_per_expert == 3 * 160 * 40 * 48 * 2 + 2 * (2560 + 640) * 2 + (640 + 2560) * 2


def test_exl3_bytes_match_engine_formula():
    from types import SimpleNamespace
    from freetoken.moe.offload_cache import bank_bytes_per_expert  # torch import is fine in tests
    assert model_info.expert_bytes("exl3", 2560, 640, k=3) == bank_bytes_per_expert(
        "exl3", 2560, 640, SimpleNamespace(exl3_expert_k=3))
```

and in `test_model_detect.py`: a folder with the exl3 `config.json`, one `.safetensors` file and `ngram_embedding.safetensors` but no `freetoken-ple.index.json` → detection result carries the warning; with the sidecar present → no warning. (Follow the existing folder-fixture helper in that file.)

- [ ] **Step 2: Run** `PYTHONPATH=python .venv/bin/python -m pytest tests/settings -q` → new tests FAIL.
- [ ] **Step 3: Implement** — `expert_format`: `if method == "exl3": return "exl3"` before the final `return ""`. Add:

```python
def expert_bytes(fmt: str, H: int, I: int, *, k: int = 2) -> int:
    if fmt == "exl3":
        # Mirrors freetoken.moe.offload_cache._exl3_bytes (torch-free copy; keep in step).
        tiles = (H // 16) * (I // 16)
        return 3 * tiles * 16 * k * 2 + 2 * (H + I) * 2 + (I + H) * 2
    return BYTES_PER_EXPERT[fmt](H, I)
```
  and route `describe_config`'s bytes computation through it with `k = int(float(quant.get("bits", 2)))` for exl3. `FORMAT_LABELS["exl3"] = "EXL3 (3-bit experts)"` — make the label use the actual K: build it as `f"EXL3 ({k}-bit experts)"` where the label is set. PLE bytes: `ple_bytes` must count the converted table (`freetoken-ple-*.safetensors`), not turboderp's `ngram_embedding.safetensors`; extend the header-marker scan in `model_info.py:64-91` to skip files whose `__metadata__.format == "exl3_ngram_trellis"`.
  `model_detect.detect`: when the config is exl3 and `freetoken-ple.index.json` is missing, attach the warning (same field the detector already uses for warnings — read `detect` to find it).
- [ ] **Step 4: Run** `tests/settings tests/daemon -q` → PASS (these must stay torch-free: `tests/settings/test_settings_import_safety.py`).
- [ ] **Step 5: Commit** `git commit -am "feat: control panel recognises EXL3 Qwen Flash and sizes it from the packed experts"`
- [ ] **Step 6: Box** (Jay's OK, after merge to the serving branch or from the worktree helper on port 2032 as in earlier batches): Add the model through the page (`qwen3.8-flash-exl3`), pin `--max-running-requests 1`, `--cuda-graph-max-bs 1`, vision picture weights `ram`, PLE backend `mmap` (MTP on). Run the page's fit check; then Start; compare the estimate with the real boot's host/card numbers (must be within 5 %). Start/stop through the switcher once (`curl 127.0.0.1:2040/v1/chat/completions` with `model: qwen3.8-flash-exl3`). Restore the daily model.

---

### Task 10: Measurements and research note

Operational + one doc. Jay's OK for the boots.

**Files:**
- Create: `docs/research/exl3-qwen-flash-2026-09-XX.md` (real date)

- [ ] **Step 1:** Same-day A/B, one server at a time, same flags except model: NVFP4 daily profile vs EXL3 profile (1 chat). Workloads: the `ab_send` set (numbers, code, essay, 8k-chat, 8k-greedy, cold-7k TTFT, warm-turn TTFT) — find `ab_send.py` in the earlier scratch dirs listed in memory (`project-mtp-spike-env`) or the box's `~/FreeToken/prompts` tree; copy it into the session scratchpad, not the repo. Record decode tok/s, TTFT, PC free memory while serving (`free -g` in WSL and Windows `Get-Counter '\Memory\Available MBytes'`), card used, boot time, MTP acceptance.
- [ ] **Step 2:** Write the note: what was measured, table of both models, the converter's precision report (Task 1), the speed gate verdict (below), what was not measured. Follow the style of `docs/research/memory-audit-qwen38-rtx5090.md`.
- [ ] **Step 3:** Speed gate: if EXL3 8k-chat decode is below 60 % of NVFP4's, run Task 11 and re-measure; record both.
- [ ] **Step 4:** Commit `git add docs/research/exl3-qwen-flash-*.md && git commit -m "docs(research): measure EXL3 Qwen Flash against NVFP4 on the 5090"`.

---

### Task 11 (conditional — only if Task 10's speed gate trips): int8 fallback for dense EXL3

**Files:**
- Modify: `python/freetoken/kernel/exl3_linear.py`
- Test: `tests/kernels/test_exl3_linear.py`

**Interfaces:**
- Produces: env `FREETOKEN_EXL3_DENSE=int8` makes every `Exl3Linear` reconstruct its weight at load (card) and serve it through `kernel/triton/int8_linear.int8_linear`; `Exl3LMHead` included; experts untouched.

- [ ] **Step 1: Failing test**

```python
def test_int8_fallback_matches_reference(cpu_wheel, monkeypatch):
    monkeypatch.setenv("FREETOKEN_EXL3_DENSE", "int8")
    monkeypatch.setattr(el, "_int8_linear", lambda x, w, s, b: F.linear(x.float(), w.float() * s[:, None].float()).bfloat16() + (0 if b is None else b))
    op, state = _loaded(128, 256, 3)
    assert op._int8_weight is not None and op.trellis.numel() == 0  # packed copy freed
    x = torch.randn(3, 128).bfloat16()
    want = _ref(x, {n: state[f"p.{n}"] for n in ("trellis", "suh", "svh")}, 3)
    torch.testing.assert_close(op.forward(x).float(), want, rtol=3e-2, atol=3e-2)
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — in `Exl3Linear.load_state_dict`, after validation: if `os.environ.get("FREETOKEN_EXL3_DENSE", "").lower() == "int8"`: `w = _reconstruct_into(self, torch.empty(out, in, bf16, device), torch.empty(in, out, fp16, device))`; `self._int8_weight, self._int8_scale = quantize_int8_rows(w)` (from `kernel/triton/int8_linear.py`); replace `trellis/suh/svh` with empty tensors of the same dtype (`torch.empty(0, ...)`) so memory is released; `forward` starts with `if self._int8_weight is not None: return _int8_linear(x2, self._int8_weight, self._int8_scale, self.bias).view(...)`. `_int8_linear` is a module seam bound to `int8_linear`. `_need()` skips int8-mode ops. Document the measured trigger in the module docstring.
- [ ] **Step 4: Run tests; box re-measure** (Task 10 step 1 with the env set).
- [ ] **Step 5: Commit** `git commit -am "perf: optional int8 serving of dense EXL3 layers when the packed GEMM is too slow"`

---

### Task 12 (conditional — only if Jay wants 2 chats on the EXL3 model): graphs at batch 2

Measure first: on the box, boot with `--max-running-requests 2 --cuda-graph-max-bs 0` and record two-chat throughput. Then extend `decode_is_graph_safe` to accept `cuda_graph_bs` `[1, 2]` / `max_running_req` 2 on the mgemm path (route list 20 ≤ 128), size `prepare_exl3_scratch(decode_max_tokens=2)`, add a `@cuda` capture/replay test at bs 2 in `tests/moe/test_fused_exl3.py` mirroring the bs 1 one, boot with graphs at 2 and compare. Keep the change only if two-chat aggregate tok/s improves by ≥ 10 % over graphs-off; write the numbers into the Task 10 note. Commit `perf: two-chat decode graphs for EXL3 routed experts`.

---

### Task 13: Whole-branch check and PR

- [ ] Run on the devbox: `PYTHONPATH=python .venv/bin/python -m pytest tests/settings tests/daemon tests/models tests/moe tests/kernels tests/engine -q -m "not slow"`; compare the failure list with `mtp-upstream-merge` (run the same command in a clean worktree of `mtp-upstream-merge`) — no new failures allowed.
- [ ] Run the GPU-marked EXL3 tests on the box once more from `~/FreeToken-exl3`.
- [ ] Final NVFP4 boot on the box from `~/FreeToken-exl3` with the daily profile (Jay's OK): same 8 prompts, same speed within noise (±3 %) — the unchanged-default check.
- [ ] Push and open the PR (`gh-axi pr create`), title `feat: run the EXL3 Qwen3.8 Flash copy (packed on the card) beside NVFP4 Flash`, body: what, measured numbers table from the note, what was not measured, test commands, attribution footer. Remove the `~/FreeToken-exl3` worktree after merge.

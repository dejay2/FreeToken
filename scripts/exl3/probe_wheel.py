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

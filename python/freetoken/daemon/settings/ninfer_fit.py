"""Graphics-card estimate for NInfer models (control panel fit check).

need = model file + chat memory (KV) + fixed costs + the Windows desktop's share of the card.

Calibrated on the measured QUASAR boot on the RTX 5090 box (2026-09-24): 29.7 GB of the card in
use with quasar-27b loaded at --kv-capacity 200000, --kv-dtype int8, --spec dflash2, --vision,
--max-concurrency 4; artifact 19,782,132,224 bytes. GB here is 1024^3 bytes (nvidia-smi MiB /
1024). With nothing loaded the card shows 1.8-2.1 GB in use (own-switcher acceptance,
docs/research/own-switcher-acceptance-2026-09-24.md), hence the 1.9 GB desktop share.

The KV rate is NInfer's paged-KV layout (src/core/paged_kv_storage.h, head dim 256) for the
Qwen3.8 27B family: 16 full-attention layers x 4 KV heads (docs/maintainer/qwen3.6-27b-model.md).
Per token and head: bf16 1024 B, int8 528 B, fp8 516 B, nvfp4 288 B, k8v4 402 B. Other model
families need their own geometry; Stage B reads it from the artifact header.

The fixed costs split the remaining 3.1 GB of the measurement. The split is a judgement; only
their sum is measured. Task 13 re-measures and records any recalibration here.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import ninfer_dials

GIB = 1024 ** 3
KV_BYTES_PER_TOKEN_HEAD = {"bf16": 1024, "int8": 528, "fp8": 516, "nvfp4": 288, "k8v4": 402}
KV_LAYERS = 16
KV_HEADS = 4
DESKTOP_RESERVE_BYTES = int(1.9 * GIB)
RUNTIME_BASE_BYTES = int(1.6 * GIB)  # CUDA context, workspaces, 1,024-token prefill buffers, lane state
CUDA_GRAPH_BYTES = int(0.3 * GIB)
VISION_BYTES = int(0.6 * GIB)
SPEC_BYTES = {"off": 0, "mtp": int(0.3 * GIB), "dflash": int(0.6 * GIB), "dflash2": int(0.6 * GIB)}
AUTO_HEADROOM_BYTES = GIB  # --kv-capacity auto keeps 1 GiB spare (include/ninfer/types.h)
TIGHT_MARGIN_BYTES = int(1.5 * GIB)


def kv_bytes_per_token(dtype: str) -> int:
    return KV_LAYERS * KV_HEADS * KV_BYTES_PER_TOKEN_HEAD[dtype]


def estimate(settings: Mapping[str, Any], artifact_bytes: int, *, card_total_bytes: int | None = None) -> dict[str, Any]:
    full = ninfer_dials.normalized(settings)
    dtype = full["kv-dtype"]
    fixed = [("Model file", int(artifact_bytes)), ("Engine working memory", RUNTIME_BASE_BYTES)]
    if not full["no-cuda-graph"]:
        fixed.append(("Speed recordings", CUDA_GRAPH_BYTES))
    if full["vision"]:
        fixed.append(("Pictures", VISION_BYTES))
    if SPEC_BYTES.get(full["spec"]):
        fixed.append(("Guess-ahead helper", SPEC_BYTES[full["spec"]]))
    fixed.append(("Windows desktop", DESKTOP_RESERVE_BYTES))
    notes: list[str] = []
    capacity = full["kv-capacity"]
    if capacity == 0:
        room = (card_total_bytes or 0) - sum(size for _, size in fixed) - AUTO_HEADROOM_BYTES
        kv = max(0, room)
        tokens = kv // kv_bytes_per_token(dtype)
        notes.append("'Fill the card' uses whatever card memory is left, keeping 1 GB spare.")
    else:
        tokens = capacity if capacity != "" else full["max-context"]
        kv = tokens * kv_bytes_per_token(dtype)
    components = [{"label": label, "bytes": int(size)} for label, size in fixed]
    components.append({"label": f"Chat memory ({tokens:,} tokens, {dtype})", "bytes": int(kv)})
    return {"needBytes": sum(item["bytes"] for item in components), "components": components,
            "notes": notes, "kvTokens": int(tokens)}


def verdict(need_bytes: int | None, total_bytes: int | None) -> str:
    if not total_bytes or need_bytes is None:
        return "unknown"
    if need_bytes > total_bytes:
        return "wont_fit"
    if total_bytes - need_bytes < TIGHT_MARGIN_BYTES:
        return "tight"
    return "fits"


__all__ = [name for name in dir() if not name.startswith("_")]

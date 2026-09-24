"""Graphics-card estimate for NInfer models (control panel fit check).

need = model file + chat memory (KV) + fixed costs + the Windows desktop's share of the card.

Calibrated on the measured QUASAR boot on the RTX 5090 box (2026-09-24): 29.7 GB of the card in
use with quasar-27b loaded at --kv-capacity 200000, --kv-dtype int8, --spec dflash2, --vision,
--max-concurrency 4; artifact 19,782,132,224 bytes. GB here is 1024^3 bytes (nvidia-smi MiB /
1024). With nothing loaded the card shows 1.8-2.1 GB in use (own-switcher acceptance,
docs/research/own-switcher-acceptance-2026-09-24.md), hence the 1.9 GB desktop share.

The KV rate is NInfer's paged-KV layout (src/core/paged_kv_storage.h, head dim 256) for the
Qwen3.8 27B family: 16 full-attention layers x 4 KV heads
(engines/ninfer/docs/maintainer/qwen3.6-27b-model.md section 2.2 "Full attention"; QUASAR is the
Qwen3.8-27B artifact and engines/ninfer/docs/maintainer/qwen3.8-27b-artifact.md section "1" names
that same qwen3.6-27b-model.md as the model-math/state-behavior doc it shares). Per token and
head: bf16 1024 B, int8 528 B, fp8 516 B, nvfp4 288 B, k8v4 402 B. Other model families need their
own geometry; Stage B reads it from the artifact header.

Per-lane chat state (fix round 1, 2026-09-24): the GDN (Gated DeltaNet) linear-attention layers
keep a fixed-size recurrent-state + causal-conv-history "StateImage" per active lane, sized by
`max-concurrency + device-state-slots` regardless of context length
(src/runtime/engine/engine.cpp:63 `device_state_slots.value_or(concurrency)` when unset,
:75-76 `total_device_state_slots = concurrency + *cache.device_state_slots`; :46-54 forces it to 0
when `--no-prefix-reuse` disables the context cache). Per-slot geometry -- 48 GDN layers x 10240
conv channels x 3-wide BF16 history, and 48 layers x 48 heads x 128 x 128 FP32 recurrent matrices
-- is exactly the shape math in src/core/linear_attention_state.{h,cpp} (`LinearAttentionStatePoolSpec`,
`plan_linear_attention_state_pool`) and is also tabulated directly in
engines/ninfer/docs/maintainer/qwen3.6-27b-model.md section 13 "State inventory". At the QUASAR
calibration point (max-concurrency 4, device-state-slots left at its default) that is 8 slots.

The fixed costs split the remaining 3.1 GB of the measurement minus the per-lane state share at
the calibration point (8 slots). The split is a judgement; only their sum is measured. Task 13
re-measures and records any recalibration here.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import ninfer_dials

GIB = 1024 ** 3
KV_BYTES_PER_TOKEN_HEAD = {"bf16": 1024, "int8": 528, "fp8": 516, "nvfp4": 288, "k8v4": 402}
KV_LAYERS = 16
KV_HEADS = 4
DESKTOP_RESERVE_BYTES = int(1.9 * GIB)

# Per-lane GDN StateImage geometry (src/core/linear_attention_state.{h,cpp}; shapes also
# tabulated in engines/ninfer/docs/maintainer/qwen3.6-27b-model.md section 13 "State inventory"):
# conv history is [conv_channels, conv_width, slot_count] BF16, recurrent state is
# [key_head_dim, value_head_dim, value_heads, slot_count] FP32, one of each per GDN layer.
GDN_LAYERS = 48  # qwen3.6-27b-model.md section 2.1: 64 Text layers, 16 full-attention, 48 GDN
GDN_CONV_CHANNELS = 10240  # Q+K+V width 2048+2048+6144 (section 2.3 "Gated DeltaNet")
GDN_CONV_WIDTH = 3  # causal convolution width 4 needs 3 history columns (section 2.3)
GDN_VALUE_HEADS = 48
GDN_KEY_HEAD_DIM = 128
GDN_VALUE_HEAD_DIM = 128
DEVICE_STATE_SLOT_BYTES = GDN_LAYERS * (
    GDN_CONV_CHANNELS * GDN_CONV_WIDTH * 2  # BF16 conv history
    + GDN_VALUE_HEADS * GDN_KEY_HEAD_DIM * GDN_VALUE_HEAD_DIM * 4  # FP32 recurrent matrices
)  # 153,944,064 B/slot (~146.8 MiB); continuation-hidden plane bytes are negligible, left out
# QUASAR calibration: max-concurrency 4, device-state-slots left empty -> engine.cpp:63 defaults
# it to concurrency (4), so total slots = concurrency + device_state_slots = 8
# (engine.cpp:75-76). That per-lane share (8 * DEVICE_STATE_SLOT_BYTES) is carved out of the
# judgement-split 1.6 GiB below so the calibrated total (29.72 vs measured 29.7 GB) is unchanged.
RUNTIME_BASE_BYTES = int(1.6 * GIB) - 8 * DEVICE_STATE_SLOT_BYTES  # CUDA context, workspaces, 1,024-token prefill buffers
CUDA_GRAPH_BYTES = int(0.3 * GIB)
VISION_BYTES = int(0.6 * GIB)
SPEC_BYTES = {"off": 0, "mtp": int(0.3 * GIB), "dflash": int(0.6 * GIB), "dflash2": int(0.6 * GIB)}
AUTO_HEADROOM_BYTES = GIB  # --kv-capacity auto keeps 1 GiB spare (include/ninfer/types.h)
TIGHT_MARGIN_BYTES = int(1.5 * GIB)


def kv_bytes_per_token(dtype: str) -> int:
    try:
        return KV_LAYERS * KV_HEADS * KV_BYTES_PER_TOKEN_HEAD[dtype]
    except KeyError:
        raise ValueError(
            f"unknown chat memory precision {dtype!r}; must be one of "
            + ", ".join(KV_BYTES_PER_TOKEN_HEAD)
        ) from None


def estimate(settings: Mapping[str, Any], artifact_bytes: int, *, card_total_bytes: int | None = None) -> dict[str, Any]:
    full = ninfer_dials.normalized(settings)
    dtype = full["kv-dtype"]
    concurrency = full["max-concurrency"]
    device_state_slots = full["device-state-slots"]
    if device_state_slots == "":
        # engine.cpp:63 defaults an unset --device-state-slots to --max-concurrency, but only
        # when the context cache is enabled; engine.cpp:46-54 forces it to 0 when
        # --no-prefix-reuse disables the cache (normalized() already resets an override back to
        # "" in that case, via ninfer_dials.CONTEXT_CACHE).
        device_state_slots = 0 if full["no-prefix-reuse"] else concurrency
    total_device_state_slots = concurrency + device_state_slots
    fixed = [
        ("Model file", int(artifact_bytes)),
        ("Engine working memory", RUNTIME_BASE_BYTES),
        ("Chat lane state", total_device_state_slots * DEVICE_STATE_SLOT_BYTES),
    ]
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

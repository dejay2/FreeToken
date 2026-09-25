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

Startup refusal check (fit round 2, 2026-09-25)
-----------------------------------------------
The used-memory estimate above matches what the card shows once NInfer is up (mc4: 31.91 GB
estimated vs 31.97 GB measured, decimal), but NInfer refuses to start on a different number: it
plans an up-front *runtime reservation* and throws if that does not fit in what is free after the
weights. Live on the RTX 5090 (cudaMemGetInfo total 34,190,917,632 B, WSL), QUASAR at kv-capacity
200000 int8, dflash2 draft-tokens 7, lm-head-draft, vision, max-context 150000 started at
max-concurrency 4 and 5 and refused at 6:

    FATAL server failed during startup | requested Engine runtime reservation requires
    13177821184 bytes, but only 13111561216 bytes are available for runtime capacity

`capacity |` startup lines (src/serve/operational_log.cpp:455), all kv 200000:
    mc4 int8 ctx150000  pages 3,125/9,376   runtime 10.6 GiB  free 2.30 GiB  card 30484 MiB
    mc5 int8 ctx150000  pages 3,125/11,720  runtime 11.4 GiB  free 1.91 GiB  card 30892 MiB
    mc6 int8 ctx150000  refused: needs 13,177,821,184 B, 13,111,561,216 B available
    mc4 int8 ctx100000  pages 3,125/6,252   runtime 10.6 GiB  free 2.30 GiB
    mc5 int8 ctx100000  runtime 11.4 GiB    free 1.89 GiB
    mc4 fp8  ctx150000  runtime 10.4 GiB    free 2.45 GiB  card 30354 MiB
    mc3 int8            card 30077 MiB;  idle card (desktop only) 2614-2688 MiB

Where the numbers come from in the frozen runtime (engines/ninfer):
* The check is `resolve_kv_capacity` (src/runtime/engine/kv_capacity.cpp:76-127): explicit
  kv-capacity -> pages = ceil(tokens / 64); reservation = curve.reservation_bytes(pages); refuse
  when reservation > available. Automatic (kv-capacity 0) first takes 1 GiB of headroom off
  `available` and grows pages until the budget is spent.
* `available` is cudaMemGetInfo free minus the device weights (src/targets/registry.cpp:67-78,
  :106-121: once as a preflight on the planned weights, again after they are loaded).
* The reservation is `persistent.bytes + workspace.capacity + graph_allowance_bytes`
  (src/targets/qwen3_6/impl/runtime/layouts_impl.h:709-779). The parts that scale are modelled
  exactly from that file: the Main KV page pool (pages x 64 tokens x 16 layers x 4 heads x the
  per-token-head bytes above; MTP adds its one layer and mc x ceil((K-1)/64) extra pages,
  :104-113); StateImage slots = max-concurrency + device-state-slots, each 48 GDN layers of conv
  history + FP32 recurrent state, the BF16 continuation hidden, and for DFlash the 5-layer x
  2048-token x 8-head x 128 BF16 local K/V (state_image.cpp:148-165, cyclic_kv_cache.cpp:50-70);
  GDN replay records for speculation, max-concurrency x 48 layers x (K+1) columns of conv/key/
  value/gate (core/gdn_replay_records.cpp:96-112); DFlash pending features 25600 x (K+1) x mc
  BF16 and the sampling token counts 248077 x mc I32 (layouts_impl.h:205-240); and the CUDA
  graph allowance (layouts_impl.h:736-773 with qwen3_6_27b/impl/variant.cpp:118-164): 12 MiB x mc
  plain, per-batch max(12 or 82 MiB) x mc for MTP, and for DFlash2 the sum over batch sizes
  1..mc of 64 MiB per attention tier whose final window is <= 4096 tokens and 96 MiB per tier
  above (tiers end at 96/511/2047/8191/32767/max-context-1, so 3 x 64 + 3 x 96 = 480 MiB per
  lane once max-context exceeds 32768 -- which is why 100000 and 150000 plan the same runtime).
* What is left -- the workspace (a peak over prefill-chunk / verify / vision plans) and small
  tensors -- is calibrated as one constant from the mc6 refusal: 954,523,336 B (0.89 GiB) for
  QUASAR dflash2 + vision, prefill-chunk 1024. Other feature mixes reuse it; it is the least
  certain term.
* Weights as NInfer sees them: at mc6, total - available = 21,079,356,416 B were in use before
  planning. With the 2.6 GiB desktop measured that day and the 19,782,132,224 B artifact, the
  device weights plus CUDA context come to the artifact minus 1,494,504,550 B (not every artifact
  object is materialized on the card: src/artifact/binder.cpp:85-116 host/validate-only objects).
  The startup check therefore uses a 2.6 GiB desktop, the value that reproduces the room NInfer
  reported on the refusal day (idle card 2614-2688 MiB then); the used-memory components keep the
  typical 1.9 GiB. When the switcher is known to have nothing loaded, the panel passes the live
  card reading and the check uses whichever is larger (fix round 1 ruling, 2026-09-25).
* The workspace constant is scaled by max(1, prefill-chunk / 1024): the prefill buffers
  (layouts_impl.h:200-221 prefill features/positions/hidden sized by effective_prefill_chunk, and
  the text_prefill workspace plan, :247-270 and :394) grow with the chunk. Scaling the whole
  constant over-predicts, which is the safe side for a refusal check; smaller chunks keep it.

Calibration (predicted vs logged): runtime mc4 int8 10.58 GiB (10.6), mc5 11.42 (11.4), mc6
13,177,821,184 B exactly, mc4 fp8 10.43 (10.4); room 13,111,561,216 B at 2.6 GiB desktop on the
34,190,917,632 B card -> mc4 fits with 1.7 GiB spare, mc5 is tight (0.8 GiB), mc6 refuses.

Upstream runtime (Fable, Twin) calibration (track 5, 2026-09-25)
----------------------------------------------------------------
Fable and Twin run engines/ninfer-upstream (Qwen3_5ForCausalLM artifacts, v3). Its planner
(src/models/qwen3_5/program/planning/startup.cpp:101-264 persistent layout, :827-870 graph
allowance and reservation = persistent + workspace + graphs; runtime/engine/kv_capacity.cpp:
76-141 the same refusal) has the same shape as the frozen runtime, so the KV, StateImage, replay
and graph terms above carry over unchanged. Measured live on the RTX 5090 (WSL), ninfer-serve run
by hand with the config.yaml command line (kv-capacity 200000, fp8, mtp draft-tokens 4,
lm-head-draft, vision, max-context 150000, prefill-chunk default) plus --request-log-jsonl, whose
startup record carries every byte (src/serve/request_log.cpp:500-517):

    fable mc4  reservation 9,523,597,057 = sequence 8,313,016,064 + workspace 866,648,065
               + graphs 343,932,928; kv payload 7,018,128,384; available 10,948,182,016
    fable mc6  reservation 10,337,772,033 (sequence 8,955,224,576, graphs 515,899,392), started
    fable mc8  refused: requires 11,151,947,009 bytes, only 10,948,182,016 available
    twin       byte-identical to fable at mc4 and mc8 (same geometry; weights 21,479,648,768 B)
    mc1        refused earlier: kv-capacity 200000 > mc x ceil(150000/64) pages (not a memory limit)

The KV payload matches the formula to the byte and the graph allowance is the MTP 82 MiB x mc.
Against the frozen-runtime constants the reservation was 51-64 MB high (+0.6%) and the room
445 MB high (Fable: 11,393,612,540 predicted vs 10,948,182,016), so mc8 read "tight" instead of
a refusal. Upstream therefore gets its own two constants plus a per-lane remainder:
* workspace: the workspace arena is the Vision encode peak (866,648,065 B, the same at mc4 and
  mc6); the fitted fixed remainder is 877,647,105 B, plus 3,131,724 B per lane (round state /
  sampling tensors the model above leaves out; the mc4/mc6/mc8 points are exactly linear).
* weights delta: -1,041,913,962 B, fitted on Twin's artifact (21,492,920,836 B) so that the
  heavier Fable artifact (21,500,080,900 B, same 21,479,648,768 B on the card) comes out 7 MB
  under the logged room -- the safe side. cudaMemGetInfo reported the same available bytes to
  the byte with the idle card at 1.5 GiB and at 2.1 GiB, so the 2.6 GiB startup desktop stays.
After: mc4/mc6/mc8 reservations exact, room within 7.2 MB (both models), mc8 refuses.
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


MIB = 1024 ** 2
PAGE_TOKENS = 64  # kPagedKVPageSize
# Startup reservation geometry (layouts_impl.h, qwen3_6_27b/impl/config.h).
GDN_KEY_HEADS = 16
TEXT_HIDDEN = 5120
TOKEN_DOMAIN = 248077  # qwen3_6/frontend.h kTokenDomain
DFLASH_LOCAL_LAYERS, DFLASH_LOCAL_CAPACITY, DFLASH_KV_HEADS, DFLASH_HEAD_DIM = 5, 2048, 8, 128
DFLASH_FEATURE_ROWS = 25600
STATE_IMAGE_SLOT_BYTES = DEVICE_STATE_SLOT_BYTES + TEXT_HIDDEN * 2  # + BF16 continuation hidden
DFLASH_LOCAL_SLOT_BYTES = 2 * DFLASH_LOCAL_LAYERS * DFLASH_LOCAL_CAPACITY * DFLASH_KV_HEADS * DFLASH_HEAD_DIM * 2
# Calibrated remainder (workspace peak + small tensors) from the 2026-09-25 mc6 refusal; see the
# module docstring. 13,177,821,184 - KV 6,758,400,000 - 6 lanes x 910,816,308 = 954,523,336.
RUNTIME_WORKSPACE_BYTES = 954_523_336
# Device weights + CUDA context relative to the artifact size, from the same refusal:
# 34,190,917,632 - 13,111,561,216 - 2.6 GiB desktop - 19,782,132,224 artifact.
WEIGHTS_SEEN_DELTA_BYTES = -1_494_504_550
STARTUP_DESKTOP_BYTES = int(2.6 * GIB)  # idle card 2614-2688 MiB on 2026-09-25
MASKED_DRAFT = ("dflash", "dflash2")
# Per-runtime calibrated terms (module docstring). "ninfer" is the frozen QUASAR runtime (mc6
# dflash2 refusal, 2026-09-25); "ninfer-upstream" is Fable/Twin (mtp draft 4, fp8, vision,
# kv 200000: mc4/mc6 started, mc8 refused at 11,151,947,009 vs 10,948,182,016, 2026-09-25).
RUNTIME_CALIBRATION = {
    "ninfer": {"workspace": RUNTIME_WORKSPACE_BYTES, "per_lane": 0,
               "weights_delta": WEIGHTS_SEEN_DELTA_BYTES},
    "ninfer-upstream": {"workspace": 877_647_105, "per_lane": 3_131_724,
                        "weights_delta": -1_041_913_962},
}
# Startup "tight" margin per runtime. The upstream prediction is byte-exact (room within 7.2 MB,
# safe side), and the shipped Fable/Twin mc4 setting starts with 1.33 GiB of slack and has
# served since 2026-09-24, so 1.0 GiB warns only below what has been measured to work.
STARTUP_TIGHT_MARGIN = {"ninfer-upstream": int(1.0 * GIB)}
DEFAULT_RUNTIME = "ninfer"


def _calibration(runtime: str | None) -> Mapping[str, int]:
    return RUNTIME_CALIBRATION.get(runtime or DEFAULT_RUNTIME, RUNTIME_CALIBRATION[DEFAULT_RUNTIME])


def _graph_profiles_through(max_frontier: int, preferred_ends: list[int]) -> list[tuple[int, int]]:
    """variant.cpp:25-38."""
    out, begin = [], 0
    for preferred in preferred_ends:
        if begin > max_frontier:
            break
        end = min(preferred, max_frontier)
        out.append((begin, end))
        if end == max_frontier:
            return out
        begin = end + 1
    if begin <= max_frontier:
        out.append((begin, max_frontier))
    return out


def _mtp_graph_ends(draft: int) -> list[int]:
    """variant.cpp:124-151."""
    ends = [v - 2 * draft for v in (128, 512, 2048, 4096, 8198, 16390, 32768) if v >= 2 * draft]
    extra = {3: (1029,), 4: (128, 512, 1029), 5: (128, 160, 2054, 8198)}.get(draft, ())
    ends += [v - (draft + 1) for v in extra if v >= draft + 1]
    return sorted(set(ends))


def graph_allowance_bytes(spec: str, concurrency: int, capacity: int, draft: int) -> int:
    """layouts_impl.h:736-773: CUDA graph allowance inside the runtime reservation."""
    if spec == "off":
        return 12 * MIB * concurrency
    if spec == "mtp":
        profiles = _graph_profiles_through(capacity - 1, _mtp_graph_ends(draft))
        per_batch = max((12 if min(capacity, hi + 2 * draft) <= 4096 else 82) * MIB for _, hi in profiles)
        return per_batch * concurrency
    # DFlash2: one topology class per tier, the same tiers for every batch size 1..mc.
    tiers = _graph_profiles_through(capacity - 1, [96, 511, 2047, 8191, 32767])
    per_batch = sum((64 if min(capacity, hi + draft + 1) <= 4096 else 96) * MIB for _, hi in tiers)
    return per_batch * concurrency


def runtime_reservation(full: Mapping[str, Any], pages: int, runtime: str | None = None) -> dict[str, int]:
    """NInfer's up-front runtime reservation for `pages` Main KV page groups, split by part."""
    cal = _calibration(runtime)
    concurrency = int(full["max-concurrency"])
    capacity = int(full["max-context"])
    spec = full["spec"]
    draft = int(full["draft-tokens"] or 0) if spec != "off" else 0
    dss = full["device-state-slots"]
    if dss == "":
        dss = 0 if full["no-prefix-reuse"] else concurrency
    slots = concurrency + int(dss)
    token_head = KV_BYTES_PER_TOKEN_HEAD[full["kv-dtype"]]
    kv = pages * PAGE_TOKENS * KV_LAYERS * KV_HEADS * token_head
    if spec == "mtp":  # one MTP attention layer over the pages plus the draft tail pages
        mtp_pages = pages + concurrency * -(-(draft - 1) // PAGE_TOKENS)
        kv += mtp_pages * PAGE_TOKENS * KV_HEADS * token_head
    slot_bytes = STATE_IMAGE_SLOT_BYTES + (DFLASH_LOCAL_SLOT_BYTES if spec in MASKED_DRAFT else 0)
    lanes = concurrency * TOKEN_DOMAIN * 4
    if spec != "off":
        width = draft + 1
        lanes += concurrency * GDN_LAYERS * width * (
            GDN_CONV_CHANNELS * 2 + GDN_KEY_HEAD_DIM * GDN_KEY_HEADS * 2
            + GDN_VALUE_HEAD_DIM * GDN_VALUE_HEADS * 2 + 2 * GDN_VALUE_HEADS * 4)
    if spec in MASKED_DRAFT:
        lanes += DFLASH_FEATURE_ROWS * (draft + 1) * concurrency * 2
    graphs = 0 if full["no-cuda-graph"] else graph_allowance_bytes(spec, concurrency, capacity, draft)
    parts = {"kv": kv, "stateImages": slots * slot_bytes, "lanes": lanes, "graphs": graphs,
             "workspace": int(cal["workspace"] * max(1.0, int(full["prefill-chunk"]) / 1024))
             + cal["per_lane"] * concurrency}
    parts["total"] = sum(parts.values())
    return parts


def _page_bounds(full: Mapping[str, Any]) -> tuple[int, int]:
    """layouts_impl.h:801-808: minimum = max(logical pages, mc), maximum = mc x logical pages."""
    logical = -(-int(full["max-context"]) // PAGE_TOKENS)
    concurrency = int(full["max-concurrency"])
    return max(logical, concurrency), concurrency * logical


def startup_room(artifact_bytes: int, card_total_bytes: int, desktop_bytes: int | None = None,
                 runtime: str | None = None) -> int:
    """What cudaMemGetInfo will call free after the weights (registry.cpp:67-78, :121).

    desktop_bytes is a live card reading taken with nothing loaded; the larger of it and
    STARTUP_DESKTOP_BYTES is used."""
    desktop = max(STARTUP_DESKTOP_BYTES, int(desktop_bytes or 0))
    return card_total_bytes - desktop - (int(artifact_bytes) + _calibration(runtime)["weights_delta"])


def startup_check(full: Mapping[str, Any], artifact_bytes: int, card_total_bytes: int | None,
                  desktop_bytes: int | None = None, runtime: str | None = None) -> dict[str, Any]:
    """Predict kv_capacity.cpp:76-127 for these settings: reservation, room and a verdict."""
    minimum, maximum = _page_bounds(full)
    capacity = full["kv-capacity"]
    room = startup_room(artifact_bytes, card_total_bytes, desktop_bytes, runtime) if card_total_bytes else None
    if capacity == 0:
        if room is None:
            pages = minimum
        else:
            base = runtime_reservation(full, minimum, runtime)["total"]
            stride = runtime_reservation(full, minimum + 1, runtime)["total"] - base if minimum < maximum else 0
            spare = room - AUTO_HEADROOM_BYTES - base
            pages = minimum if spare < 0 or not stride else min(maximum, minimum + spare // stride)
    else:
        tokens = capacity if capacity != "" else full["max-context"]
        pages = max(-(-int(tokens) // PAGE_TOKENS), minimum)
    reservation = runtime_reservation(full, pages, runtime)["total"]
    budget = None if room is None else room - (AUTO_HEADROOM_BYTES if capacity == 0 else 0)
    out = {"reservationBytes": int(reservation), "roomBytes": None if room is None else int(room),
           "verdict": "unknown", "message": ""}
    if budget is None:
        return out
    need_gb, free_gb = reservation / GIB, max(0, budget) / GIB
    if reservation > budget:
        out["verdict"] = "wont_fit"
        out["message"] = (f"NInfer would refuse to start. It needs to set aside {need_gb:.1f} GB for chats "
                          f"but only {free_gb:.1f} GB would be free. Try fewer chats at the same time "
                          "or a smaller chat memory.")
    elif capacity != 0 and budget - reservation < STARTUP_TIGHT_MARGIN.get(runtime or DEFAULT_RUNTIME,
                                                                           TIGHT_MARGIN_BYTES):
        # 'Fill the card' grows the chat memory into whatever is free by design, so only its
        # minimum can fail; tight applies to an explicit size.
        out["verdict"] = "tight"
        out["message"] = (f"Close to the limit: NInfer sets aside {need_gb:.1f} GB for chats and only "
                          f"{free_gb:.1f} GB would be free. A busier Windows desktop could stop it starting.")
    else:
        out["verdict"] = "fits"
    return out


def worst_verdict(*verdicts: str) -> str:
    order = {"wont_fit": 3, "tight": 2, "fits": 1, "unknown": 0}
    known = [v for v in verdicts if v != "unknown"]
    return max(known, key=order.__getitem__) if known else "unknown"


def kv_bytes_per_token(dtype: str) -> int:
    try:
        return KV_LAYERS * KV_HEADS * KV_BYTES_PER_TOKEN_HEAD[dtype]
    except KeyError:
        raise ValueError(
            f"unknown chat memory precision {dtype!r}; must be one of "
            + ", ".join(KV_BYTES_PER_TOKEN_HEAD)
        ) from None


def estimate(settings: Mapping[str, Any], artifact_bytes: int, *, card_total_bytes: int | None = None,
             desktop_bytes: int | None = None, runtime: str | None = None) -> dict[str, Any]:
    """runtime is the registry's NInfer runtime ("ninfer" or "ninfer-upstream"); None = "ninfer"."""
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
    startup = startup_check(full, artifact_bytes, card_total_bytes, desktop_bytes, runtime)
    return {"needBytes": sum(item["bytes"] for item in components), "components": components,
            "notes": notes, "kvTokens": int(tokens),
            "runtimeReservationBytes": startup["reservationBytes"], "runtimeRoomBytes": startup["roomBytes"],
            "startupVerdict": startup["verdict"], "startupMessage": startup["message"]}


def verdict(need_bytes: int | None, total_bytes: int | None) -> str:
    if not total_bytes or need_bytes is None:
        return "unknown"
    if need_bytes > total_bytes:
        return "wont_fit"
    if total_bytes - need_bytes < TIGHT_MARGIN_BYTES:
        return "tight"
    return "fits"


__all__ = [name for name in dir() if not name.startswith("_")]

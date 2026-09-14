"""The torch-free settings catalogue and input checks for the Windows helper.

Every dial carries a plain-language label, a plain-language explanation with the numbers the
research notes measured for it, an effect list, and (for amounts) a slider range in everyday
units. The page renders only what this catalogue sends, so a dial's wording lives here and
never in the HTML. Effects are ``axis:direction`` strings; the axis set is EFFECT_AXES.

Numbers quoted in ``info`` come from ``scripts/start-qwen38-flash-next-mmap-windows.ps1``'s
parameter comments and from ``docs/research/`` (memory audit, the 2026-09-02 slot-cache sweep,
the 2026-09-03 parking bench and live KV measurements, the 2026-09-04 FP8 gate). Keep them in
step with those documents when the measurements change.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .model_info import GIB as _GIB, ModelInfo, gib

EFFECT_AXES = ("speed", "accuracy", "vram", "ram", "ssd", "boot")
EFFECT_DIRECTIONS = ("up", "down", "mixed")

GIB = 1024 ** 3


@dataclass(frozen=True)
class Dial:
    name: str
    control: str
    default: Any
    unit: str
    help: str
    group: str
    options: tuple[str, ...] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    source: str = "launcher"
    engine_mapping: str = ""
    numeric_kind: str = "int"
    # Plain-language presentation. ``plain`` is the row label, ``info`` the text behind the
    # (i) button, ``effects`` the chips shown beside it ("what turning this up or on does").
    plain: str = ""
    info: str = ""
    # One-line text shown directly under the technical label on the settings page.
    blurb: str = ""
    effects: tuple[str, ...] = ()
    # Slider range in display units: (low, high, step). Deliberately narrower than the
    # validation bounds, which are the launcher's technical limits, not sensible values.
    slider: tuple[float, float, float] | None = None
    # Show and edit the value in a friendlier unit: stored = shown * display_factor.
    display_unit: str = ""
    display_factor: int | float = 1
    # A sentinel stored value meaning "let the engine decide"; the page offers it as a switch.
    auto_value: int | None = None
    auto_label: str = ""
    # Browse button kind for path dials: "folder", "model" (folder that must hold a model) or "file".
    browse: str = ""
    # Expert settings are hidden behind the page's "Show expert settings" switch.
    advanced: bool = False
    option_labels: tuple[str, ...] | None = None
    # A number the launcher stores as text: the page shows the count, the file keeps
    # ``stored_as.format(n=count)`` (``stored_zero`` when the count is 0). Used by the
    # layers-on-card dial, whose launcher value is "auto:3" (measured ranking) or "3".
    stored_as: str = ""
    stored_zero: str = ""

    def as_dict(self, value: Any, model: Any = None) -> dict[str, Any]:
        """The page contract. ``model`` (a ModelInfo) reshapes the limits, slider and
        explanation of the model-dependent dials; without it the catalogue's defaults
        (sized for Qwen3.8-Flash-Next) are sent unchanged."""
        over = adapt_dial(self, model) if model is not None else {}
        return {
            "name": self.name,
            "group": self.group,
            "control": self.control,
            "value": value,
            "unit": self.unit,
            "help": self.help,
            "options": list(self.options) if self.options is not None else None,
            "optionLabels": list(self.option_labels) if self.option_labels is not None else None,
            "min": over.get("min", self.minimum),
            "max": over.get("max", self.maximum),
            "plain": self.plain or self.name,
            "blurb": self.blurb,
            "info": over.get("info", self.info),
            "effects": [
                {"axis": axis, "direction": direction}
                for axis, direction in (item.split(":", 1) for item in self.effects)
            ],
            "slider": list(over["slider"]) if "slider" in over else (list(self.slider) if self.slider is not None else None),
            "displayUnit": self.display_unit,
            "displayFactor": self.display_factor,
            "autoValue": self.auto_value,
            "autoLabel": self.auto_label,
            "browse": self.browse,
            "advanced": self.advanced,
            "engine": self.engine_mapping,
            "storedAs": over.get("storedAs", self.stored_as),
            "storedZero": self.stored_zero,
            "modelAware": bool(over) or self.name in MODEL_AWARE_DIALS,
            # The GPU-owned-layer charge, for the live minimum the page prints under the
            # slot field. Present only on the slot dial of a readable MoE model; the page
            # keys off these instead of hard-coding this model's 512 and 1,024.
            "expertsPerLayer": over.get("expertsPerLayer"),
            "streamingFloor": over.get("streamingFloor"),
            "ownedDial": over.get("ownedDial", ""),
        }


GROUP_INFO: dict[str, dict[str, str]] = {
    "Model & chats": {
        "plain": "Model & chats",
        "info": "Model choice, chat length, shared KV memory and parking for inactive chats.",
    },
    "Memory & experts": {
        "plain": "Memory & experts",
        "info": "GPU expert slots, dense weights, embeddings and the memory fit check.",
    },
    "Memory governor": {
        "plain": "Memory governor",
        "info": "Automatic movement of expert layers between GPU memory, RAM and SSD.",
    },
    "MTP": {
        "plain": "MTP",
        "info": "Experimental multi-token prediction settings.",
    },
    "Pictures": {
        "plain": "Pictures",
        "info": "Picture input and the placement of vision weights.",
    },
    "Server & advanced": {
        "plain": "Server & advanced",
        "info": "Expert loading, diagnostics, server access and CUDA graph settings.",
    },
}


# Keep this catalogue independent of the model and CUDA packages. The current model path is read
# from the local boot file at runtime rather than copied into tracked source.
DIALS: tuple[Dial, ...] = (
    Dial(
        "ModelPath", "path", "", "", "Filesystem directory containing the model weights and tokenizer config.",
        "Model & chats", engine_mapping="--model <path>",
        plain="Model path", blurb="Folder containing the model weights and tokenizer.", browse="model",
        info=(
            "The folder holding the model's files. A usable folder contains config.json and the weight "
            "files. Every other setting on this page was tuned for Qwen3.8-Flash-Next on this PC, so "
            "changing the model means re-checking them."
        ),
    ),
    Dial(
        # maximum is a technical ceiling; the real limit is the chosen model's longest chat
        # (config.json max_position_embeddings), applied by adapt_dial / validate_settings.
        "ContextTokens", "number", 262144, "tokens", "Maximum sequence length per request reserved in the KV cache.",
        "Model & chats", minimum=64, maximum=4194304, engine_mapping="--kv-reserve-tokens <N>",
        plain="Context tokens", blurb="Maximum tokens in one request's conversation.", slider=(1024, 262144, 1024), effects=("vram:up",),
        info=(
            "How long one conversation may grow, in tokens. A token is about three quarters of a "
            "word, so 262,144 tokens is roughly 200,000 words. The server sets this much room aside "
            "for a single chat; at 25,344 bytes per token the full 262,144 costs about 6.2 GiB of card "
            "memory at the normal precision, half that at the compact precision."
        ),
    ),
    Dial(
        "KVCacheTokens", "number", 262144, "tokens", "Total capacity of the KV token cache pool (0 selects automatic sizing).",
        "Model & chats", minimum=0, maximum=4194304, engine_mapping="--num-tokens <N>",
        plain="KV cache tokens", blurb="Total KV memory shared by all running conversations.", slider=(0, 524288, 8192), auto_value=0, auto_label="Automatic",
        effects=("vram:up", "speed:up"),
        info=(
            "The total chat memory the card keeps for all chats together. Bigger means more chats can "
            "run at once with long histories before the server has to re-read them. Each 65,536 tokens "
            "costs about 1.55 GiB of card memory (measured 2026-09-02). Automatic (0) does not match "
            "the longest single chat: it grows the pool until the free card memory is used up, which "
            "can leave nothing for the look-ahead trick or the recordings. With the look-ahead trick "
            "on, or a long chat, set this to the longest single chat plus one page of 64 tokens - "
            "262,208 for 262,144. With Dynamic KV memory on, this is the largest size the pool may "
            "grow to; the pool boots at the smallest size and grows in steps."
        ),
    ),
    Dial(
        "KVDynamic", "toggle", True, "switch", "Grow the KV pool on demand by trading MoE expert slots and shrink it back when idle.",
        "Model & chats", engine_mapping="--kv-dynamic",
        plain="Dynamic KV memory", blurb="Start small, grow when a chat needs room, shrink back when quiet.",
        effects=("speed:up",),
        info=(
            "The card's chat memory starts at the smallest size below and the space saved becomes expert "
            "slots, which type faster. When a chat arrives that could outgrow the memory, the server pauses "
            "about a second, hands slots to the memory, and lets it in. After the quiet time below with no "
            "chats, it shrinks back. Nothing is forgotten: parking already keeps each finished chat in PC memory. "
            "Measured 2026-09-12 on this PC: one resize under a second; about 157 slots per 32,768 tokens; "
            "about 6 % typing speed per 1,000 slots."
        ),
    ),
    Dial(
        "KVFloorTokens", "number", 65536, "tokens", "Usable KV tokens the dynamic pool boots with and shrinks back to.",
        "Model & chats", minimum=8192, maximum=4194304, engine_mapping="--kv-floor-tokens <N>",
        plain="Smallest KV memory", blurb="Chat memory when no big chat is around.", slider=(16384, 262144, 8192),
        effects=("speed:up", "vram:down"),
        info="Chats whose prompt plus answer allowance fit in this need no resize. Must be no larger than KV cache tokens.",
    ),
    Dial(
        "KVStepTokens", "number", 32768, "tokens", "Growth rung of the dynamic KV pool.",
        "Model & chats", minimum=8192, maximum=1048576, engine_mapping="--kv-step-tokens <N>",
        plain="KV growth step", blurb="How much the chat memory grows at a time.", slider=(8192, 131072, 8192), advanced=True,
        info="A big chat jumps straight to the rung it needs, so this only sets the smallest change worth a resize. Below 8,192 is refused: a resize costs about a second.",
    ),
    Dial(
        "KVShrinkIdleMin", "number", 10, "min", "Minutes with no request before the dynamic KV pool shrinks to the floor.",
        "Model & chats", minimum=1, maximum=1440, engine_mapping="--kv-shrink-idle-s <N*60>",
        plain="Quiet time before shrinking", blurb="Timer 1: how long the card waits before giving slots back.", slider=(1, 120, 1),
        info="Counted from the last finished chat. The shrink runs only while no chat is active and pauses the server about a second.",
    ),
    Dial(
        "KVParkTTLHours", "number", 5, "h", "Hours a parked chat may sit unused in PC memory or on the SSD before it is dropped.",
        "Model & chats", minimum=0, maximum=168, engine_mapping="--kv-park-ttl-s <N*3600>",
        plain="Parked chat lifetime", blurb="Timer 2: when a quiet chat is dropped from PC memory for good.", slider=(0, 48, 1),
        effects=("ram:down",),
        info="0 keeps parked chats until the space is needed. Dropping frees the pinned RAM; a dropped chat is re-read from scratch if it returns.",
    ),
    Dial(
        "KVDtype", "choice", "bf16", "dtype", "Storage precision for QSA KV cache (bf16 is safe default; fp8 saves ~48% KV VRAM).",
        "Model & chats", options=("bf16", "fp8"), engine_mapping="--kv-dtype fp8 (if fp8)",
        plain="KV cache dtype", blurb="Storage format for the QSA key-value cache.", option_labels=("Normal (bf16)", "Compact (fp8)"),
        effects=("vram:down", "speed:up", "accuracy:mixed"),
        info=(
            "Compact stores the chat memory in half the space: about 3 GiB back at 262,144 tokens. "
            "Measured on this PC 2026-09-03 and 2026-09-04: answers identical to Normal on four prompt "
            "lengths up to the model's end of turn, and 2.9% faster with four chats running. Normal is "
            "the long-standing default; Compact rounds the stored numbers slightly, so keep Normal if "
            "you ever see odd answers on very long chats."
        ),
    ),
    Dial(
        "MaxRunningRequests", "number", 4, "requests", "Maximum number of concurrent inference requests processed in parallel.",
        "Model & chats", minimum=1, maximum=16, engine_mapping="--max-running-requests <N>",
        plain="Max running requests", blurb="Maximum conversations processed at the same time.", slider=(1, 16, 1), effects=("speed:mixed", "vram:up"),
        info=(
            "How many chats the server works on together. Each one answers a little slower, but the "
            "total output goes up: measured 2026-09-03, four chats together produced 15.5 to 49.5 words "
            "per second combined depending on how warm the caches were, against about 73 for one chat "
            "alone. Each extra chat needs its own slice of chat memory."
        ),
    ),
    Dial(
        "KVPark", "choice", "off", "backend", "Offload inactive KV cache prefixes to RAM or SSD between multi-turn chat interactions.",
        "Model & chats", options=("off", "ram", "ssd"), engine_mapping="--kv-park <mode>",
        plain="KV parking", blurb="Where inactive conversation memory is kept.", option_labels=("Off (re-read the chat)", "PC memory", "SSD"),
        effects=("speed:up", "ram:up", "ssd:up"),
        info=(
            "When a chat goes quiet, its memory is moved off the card so other chats can use the space, "
            "and copied back when the chat continues. Without it a long chat must be re-read: a "
            "65,000-token chat takes about 37 seconds to re-read. Measured 2026-09-04: bringing it back "
            "from PC memory takes 0.68 seconds, from the SSD 1.11 seconds, and the answer is identical. "
            "PC memory uses 1.65 GiB per 65,000-token chat and holds it until the chat is dropped; "
            "the SSD option uses drive space instead."
        ),
    ),
    Dial(
        "KVParkIdleMs", "number", 0, "ms", "Idle milliseconds before offloading inactive KV cache prefix to parking store.",
        "Model & chats", minimum=0, maximum=86400000, engine_mapping="--kv-park-idle-ms <N>",
        plain="KV parking idle time", blurb="Wait before moving an inactive conversation off the card.", slider=(0, 600, 5), display_unit="s", display_factor=1000,
        advanced=True, effects=("speed:mixed",),
        info=(
            "How long a chat must sit quiet before it is moved off the card. 0 parks it the moment its "
            "turn ends. A short wait keeps a chat that is about to continue on the card; a long wait "
            "keeps card memory busy for chats that may never return."
        ),
    ),
    Dial(
        "KVParkMinTokens", "number", 8192, "tokens", "Minimum token prefix length required before eligible for KV cache parking.",
        "Model & chats", minimum=64, maximum=4194304, engine_mapping="--kv-park-min-tokens <N>",
        plain="KV parking minimum tokens", blurb="Shortest conversation prefix eligible for parking.", slider=(1024, 131072, 1024), advanced=True, effects=("speed:mixed",),
        info=(
            "Chats shorter than this are simply re-read, because that is nearly as fast as parking them. "
            "Measured 2026-09-03: an 8,192-token chat re-reads in 4.7 seconds and restores in 0.006 "
            "seconds from PC memory, 0.06 from the SSD, so 8,192 is a comfortable line."
        ),
    ),
    Dial(
        "KVParkRAMGiB", "number", 2.0, "GiB", "Maximum pinned host RAM budget allocated for parked KV cache prefixes.",
        "Model & chats", minimum=0.125, maximum=128.0, numeric_kind="float", engine_mapping="--kv-park-ram-gib <N>",
        plain="KV parking RAM", blurb="RAM budget for parked conversation memory.", slider=(0.5, 32, 0.5), effects=("ram:up", "speed:up"),
        info=(
            "Only used when idle chats are kept in PC memory. This memory is locked for the server and "
            "nothing else can use it while it runs. One 65,000-token chat needs 1.65 GiB; a full "
            "262,144-token chat needs 6.3 GiB (measured 2026-09-03). This PC has 96 GiB, of which the "
            "model itself already pins about 65 GiB."
        ),
    ),
    Dial(
        "KVParkSSDDir", "path", "~/.cache/freetoken/kv-park", "path", "Filesystem directory on high-speed SSD used for parked KV cache files.",
        "Model & chats", engine_mapping="--kv-park-ssd-dir <dir>",
        plain="KV parking SSD folder", blurb="SSD folder used for parked conversation files.", browse="folder", effects=("ssd:up",),
        info=(
            "Where parked chats are written when the SSD option is on. Pick a folder on a fast SSD; "
            "the files are read back at about 5.3 GiB per second on this PC's drive. They are deleted "
            "when the chat is dropped."
        ),
    ),
    Dial(
        "KVParkSSDGiB", "number", 32.0, "GiB", "Maximum disk storage budget on SSD allocated for parked KV cache files.",
        "Model & chats", minimum=0.125, maximum=8192.0, numeric_kind="float", engine_mapping="--kv-park-ssd-gib <N>",
        plain="KV parking SSD size", blurb="SSD space reserved for parked conversation files.", slider=(1, 256, 1), effects=("ssd:up", "speed:up"),
        info=(
            "The most drive space parked chats may take. Each parked chat is 1.65 GiB per 65,000 "
            "tokens, so 32 GiB holds about 19 chats that long. When the space is full the oldest "
            "parked chat is dropped and will be re-read if it continues."
        ),
    ),
    Dial(
        "KVParkWindowMiB", "number", 256, "MiB", "Staging window size in host RAM for overlapping SSD disk reads with GPU copies.",
        "Model & chats", minimum=1, maximum=4096, engine_mapping="--kv-park-window-mib <N>",
        plain="KV parking transfer window", blurb="RAM transfer window for SSD parking reads and writes.", slider=(16, 2048, 16), advanced=True, effects=("ram:up",),
        info=(
            "Two buffers of this size sit in locked PC memory to move parked chats between the SSD and "
            "the card. 256 MiB was measured 2026-09-03 at 5.3 GiB per second reads and 1.6 GiB per "
            "second writes; bigger buffers were not faster. Costs twice this amount of PC memory."
        ),
    ),
    Dial(
        "MoECacheSize", "number", 4188, "slots", "Total number of MoE expert slots allocated in GPU VRAM (0 selects automatic sizing).",
        "Memory & experts", minimum=0, maximum=1048576, engine_mapping="--moe-cache-size <N>",
        plain="MoE cache size (slots)", blurb="Number of expert slots kept ready in GPU memory.", slider=(1024, 8192, 64), auto_value=0, auto_label="Automatic",
        effects=("speed:up", "vram:up"),
        info=(
            "The model is made of 24,576 expert pieces (63 GiB) that live in PC memory; the card keeps "
            "this many of them ready (2.77 MB each). This is the main speed dial. Measured 2026-09-02 "
            "on this PC: every 1,000 slots costs 2.58 GiB of card memory and is worth about 5.8 words "
            "per second on an 8,000-token chat (73 words per second at 6,750 slots). Below about 4,750 "
            "the slowdown gets steep. Slots and chat memory share the same card memory, so raising one "
            "leaves less for the other. Every layer kept whole on the card takes all of its experts "
            "out of this same total - 512 slots each on this model - and the layers that still stream "
            "need at least 1,024 slots left over, so the total must be at least 512 x layers on the "
            "card + 1,024. Automatic lets the engine pick the largest count that fits."
        ),
    ),
    Dial(
        # Stored as launcher text: "auto:N" = the N busiest layers of the ranking measured for
        # Qwen3.8 (engine GPU_OWNED_LAYER_RANK), "N" = N layers spread evenly through the model,
        # "" = off. The engine accepts any N up to the model's MoE layer count; the slider top and
        # the storage form come from the chosen model (adapt_dial). Static maximum is a ceiling.
        "GpuOwnedLayers", "number", "auto", "layers", "MoE layers that remain permanently resident in GPU VRAM instead of streaming from host RAM.",
        "Memory & experts", minimum=0, maximum=4096, engine_mapping="--moe-gpu-owned-layers <val>",
        plain="GPU-owned layers", blurb="Number of expert layers kept permanently on the GPU.", slider=(0, 48, 1), stored_as="auto:{n}", stored_zero="",
        effects=("ram:down", "vram:up", "speed:mixed"),
        info=(
            "Keeps every expert of the chosen layers on the card so those layers never need PC memory. "
            "Each layer hands back 1.32 GiB of PC memory and takes 1.32 GiB of card memory, which is "
            "charged against the expert slots above (about 512 slots per layer). The layers are taken "
            "from a busiest-first ranking measured on this PC; six was the measured sweet spot with "
            "4,188 slots, and every extra layer removes about 512 streaming slots. Because the "
            "expert slots above are the total, the layers that still stream need at least 1,024 "
            "slots left over: keep the slot total at or above 512 x layers on the card + 1,024, or "
            "the server refuses to start. 0 turns this off."
        ),
    ),
    Dial(
        "DenseQuant", "choice", "int8", "format", "Weight-only int8 quantization for dense non-MoE layers, saving ~3.9 GiB VRAM.",
        "Memory & experts", options=("", "int8"), engine_mapping="$env:FREETOKEN_DENSE_QUANT",
        plain="Dense quant", blurb="Use int8 weights for dense, always-active layers.", option_labels=("Off (full size)", "On (int8)"),
        effects=("vram:down", "speed:up", "accuracy:mixed"),
        info=(
            "Stores the parts of the model that run on every word at 8 bits instead of 16. Gives 3.9 GiB "
            "of card memory back and roughly doubles those layers' speed: measured 2026-09-02 on this PC, "
            "an 8,000-token chat went from 53 to 72 words per second once the freed memory was spent on "
            "expert slots. The rounding is small and no answer change has been reported on this model."
        ),
    ),
    Dial(
        "MoEVramReserveBytes", "number", -1, "bytes", "VRAM bytes reserved after expert cache sizing for CUDA graphs and draft heads (-1 = auto).",
        "Memory & experts", minimum=-1, maximum=34359738368, engine_mapping="--moe-vram-reserve-bytes <N>",
        plain="MoE VRAM reserve", blurb="GPU memory held back for graphs and draft heads.", slider=(0, 8, 0.25), display_unit="GiB", display_factor=GIB,
        auto_value=-1, auto_label="Automatic", advanced=True, effects=("vram:up",),
        info=(
            "Card memory the expert slots must not use because other things are loaded after them: "
            "the fast-path graphs and, when the guess-ahead trick is on, its 2.25 GiB head. Automatic "
            "reserves 0.75 GiB for graphs plus 2.25 GiB when the trick is on. Too little and the server "
            "fails to start; too much wastes slots."
        ),
    ),
    Dial(
        "MoECacheHeadroomBytes", "number", -1, "bytes", "Free VRAM cushion that expert slot cache must leave untouched (-1 = default 1.5 GiB).",
        "Memory & experts", minimum=-1, maximum=34359738368, engine_mapping="--moe-cache-headroom-bytes <N>",
        plain="MoE cache headroom", blurb="GPU memory the expert cache must leave unused.", slider=(0, 8, 0.25), display_unit="GiB", display_factor=GIB,
        auto_value=-1, auto_label="Automatic (1.5 GiB)", advanced=True, effects=("vram:up", "speed:mixed"),
        info=(
            "Card memory that must stay free after everything is loaded. Automatic keeps 1.5 GiB, the "
            "least any healthy start-up on this PC measured; a run that left only 0.55 GiB free answered "
            "at half speed. If the expert slot count does not leave this much, the server refuses to "
            "start and names the largest count that fits."
        ),
    ),
    Dial(
        "MemoryGovernor", "toggle", True, "boolean",
        "Automatically step expert layers down the VRAM/RAM ladder and shrink pools when free memory drops below cushion, stepping back up when memory returns; the card cushion is also the boot's free-VRAM headroom (--moe-cache-headroom-bytes).",
        "Memory governor",
        plain="Memory governor", blurb="Move expert layers between GPU, RAM, and SSD as memory changes.",
        effects=("speed:down",),
        info=(
            "Steps expert layers between GPU memory, host RAM, and SSD disk storage when other apps or games "
            "use memory, keeping the server alive. When PC memory runs short it first parks layers on the "
            "card (each one costs one layer's worth of shared expert slots, about 3 words a second on "
            "Qwen3.8) and only uses the SSD once the shared slots reach their floor, because a single "
            "SSD layer drops answers to 8-11 words a second. At start-up the card cushion below is also the free "
            "memory the expert slot sizing leaves untouched (the same thing 'Free card memory cushion' "
            "sets by hand; the larger of the two wins)."
        ),
    ),
    Dial(
        "GovernorVRAMFreeGB", "number", 1.5, "GB",
        "Free card memory cushion the governor maintains by stepping layers down or shrinking pools; at boot it is passed as --moe-cache-headroom-bytes (never as the post-cache reserve).",
        "Memory governor", minimum=0.0, maximum=128.0, numeric_kind="float",
        plain="VRAM cushion", blurb="Free GPU memory cushion maintained by the governor.", slider=(0.0, 16.0, 0.25),
        effects=("vram:up", "speed:down"),
        info=(
            "Card memory that must stay free for other programs. Below this cushion, the governor steps expert layers "
            "down or shrinks pools to free VRAM. At start-up the slot sizing also leaves this much free, on top of "
            "what the graphs and the guess-ahead head reserve for themselves."
        ),
    ),
    Dial(
        "GovernorRAMFreeGB", "number", 4.0, "GB",
        "Free main memory cushion the governor maintains by spilling pinned expert layers to SSD disk storage.",
        "Memory governor", minimum=0.0, maximum=1024.0, numeric_kind="float",
        plain="RAM cushion", blurb="Free host RAM cushion maintained by the governor.", slider=(0.0, 64.0, 1.0),
        effects=("ram:up", "speed:down"),
        info=(
            "Host RAM that must stay free for other programs. Below this cushion, the governor spills pinned expert "
            "layers to SSD disk storage to free host memory."
        ),
    ),
    Dial(
        "GovernorUpMarginGB", "number", 0.5, "GB", "Extra headroom before stepping back up.",
        "Memory governor", minimum=0.0, maximum=8.0, numeric_kind="float", source="helper",
        plain="Governor up margin", blurb="Extra free-memory margin required before moving layers up.", slider=(0.0, 8.0, 0.25), advanced=True,
        effects=("vram:up", "ram:up"),
        info=(
            "Extra free memory required before the governor recalls one more expert layer. "
            "A larger cushion makes recalls safer after a busy game or another memory-hungry program, "
            "but keeps more layers parked away from the card."
        ),
    ),
    Dial(
        "GovernorStepIntervalS", "number", 5.0, "s", "Check interval.",
        "Memory governor", minimum=1.0, maximum=60.0, numeric_kind="float", source="helper",
        plain="Governor check interval", blurb="How often the governor checks memory.", slider=(1.0, 60.0, 1.0), advanced=True,
        effects=("speed:mixed",),
        info=(
            "How often the helper checks card and PC memory. Shorter intervals react sooner when a game "
            "starts using memory, while longer intervals make fewer checks. The default five-second check "
            "is the measured balance for the serving box."
        ),
    ),
    Dial(
        "GovernorUpHoldS", "number", 60.0, "s", "Wait before the first step back up.",
        "Memory governor", minimum=0.0, maximum=600.0, numeric_kind="float", source="helper",
        plain="Governor up hold", blurb="Wait above the cushions before the first move up.", slider=(0.0, 600.0, 5.0), advanced=True,
        effects=("speed:down",),
        info=(
            "How long memory must remain comfortably above its cushion before the first layer is recalled. "
            "This wait prevents a brief burst of free memory from immediately bringing a layer back and "
            "then forcing it out again."
        ),
    ),
    Dial(
        "GovernorMaxHoldS", "number", 600.0, "s", "Longest wait when recovery keeps tripping the cushion.",
        "Memory governor", minimum=60.0, maximum=3600.0, numeric_kind="float", source="helper",
        plain="Governor max hold", blurb="Longest wait after repeated memory pressure.", slider=(60.0, 3600.0, 30.0), advanced=True,
        effects=("speed:down",),
        info=(
            "The largest recovery wait the helper may use after free memory repeatedly falls back below "
            "the cushion. Raising it makes repeated recalls less likely, but also delays recovery after "
            "memory pressure has gone away."
        ),
    ),
    Dial(
        "GovernorPostUpGraceS", "number", 10.0, "s", "Grace after a recall before the memory side may step down.",
        "Memory governor", minimum=0.0, maximum=120.0, numeric_kind="float", source="helper",
        plain="Governor post-up grace", blurb="Pause after a recall before reacting to its memory cost.", slider=(0.0, 120.0, 5.0), advanced=True,
        effects=("speed:mixed",),
        info=(
            "After a layer is recalled, this grace period protects the memory side from reacting to the "
            "temporary cost of that move. A longer grace reduces flap between rungs; a shorter one reacts "
            "faster to a genuine new squeeze."
        ),
    ),
    Dial(
        "GovernorRAMRungsBeforeUp", "number", 2, "rungs", "Layers of room needed before a memory recall.",
        "Memory governor", minimum=1, maximum=4, source="helper",
        plain="RAM rungs before up", blurb="Full layer sizes needed before recalling from SSD.", slider=(1, 4, 1), advanced=True,
        effects=("ram:up",),
        info=(
            "How many full expert-layer sizes of free PC memory are required before the helper moves one "
            "layer back from disk. More room protects against the memory cost of a recall, at the cost of "
            "keeping more layers on disk."
        ),
    ),
    Dial(
        "GovernorVRAMRungsBeforeUp", "number", 1, "rungs", "Layers of room needed before a card recall.",
        "Memory governor", minimum=1, maximum=4, source="helper",
        plain="VRAM rungs before up", blurb="Full layer sizes needed before recalling to the GPU.", slider=(1, 4, 1), advanced=True,
        effects=("vram:up",),
        info=(
            "How many full expert-layer sizes of free card memory are required before the helper moves one "
            "layer back to the GPU. More room makes recalls safer, but delays using the card's recovered "
            "memory for speed."
        ),
    ),
    Dial(
        "EmbedHost", "toggle", True, "boolean", "Pin the 1.27 GB token embedding table in host RAM to free GPU VRAM for expert slots.",
        "Memory & experts", engine_mapping="$env:FREETOKEN_EMBED_HOST='1'",
        plain="Embedding on host", blurb="Keep the token embedding table in host RAM.", effects=("vram:down", "ram:up"),
        info=(
            "The table that turns words into numbers is 1.27 GB. On, it lives in locked PC memory and "
            "the card fetches one row per word; that gives 1.27 GB of card memory back for expert slots "
            "at a cost too small to measure. Off, it sits on the card."
        ),
    ),
    Dial(
        "EnableVision", "toggle", True, "boolean", "Enable still-picture vision model weights and multimodal image input endpoints.",
        "Pictures", engine_mapping="$env:FREETOKEN_LOAD_VISION='1'",
        plain="Vision (picture input)", blurb="Allow picture input and load the vision weights.", effects=("ram:up", "boot:up"),
        info=(
            "Loads the picture-reading part of the model (856 MiB) so chats can include images. Where "
            "it waits is set by the two choices below. Off, pictures are refused and that memory is saved."
        ),
    ),
    Dial(
        "VisionPackagesPath", "path", "$visionPackages", "path", "Directory containing local Pillow and TorchVision dependencies for image processing.",
        "Pictures", engine_mapping="Added to $env:PYTHONPATH",
        plain="Vision packages path", blurb="Folder containing Pillow and TorchVision for pictures.", browse="folder", advanced=True,
        info=(
            "The folder holding the extra picture libraries the server needs (Pillow and TorchVision). "
            "Only change it if you installed them somewhere else."
        ),
    ),
    Dial(
        "VisionExecution", "choice", "layer-stream", "mode", "Vision execution strategy: layer-stream stages bounded layers to GPU; gpu keeps all on GPU.",
        "Pictures", options=("layer-stream", "gpu"), engine_mapping="$env:FREETOKEN_VISION_EXECUTION",
        plain="Vision execution", blurb="Process picture weights in pieces or all on the GPU.", option_labels=("Piece by piece (saves card memory)", "All on the card"),
        effects=("vram:down", "speed:mixed"),
        info=(
            "Piece by piece keeps the picture weights in PC memory and moves one block at a time onto "
            "the card through a small reusable workspace, so pictures cost almost no card memory but "
            "take a little longer each. All on the card keeps the whole 856 MiB on the card for the "
            "fastest picture handling."
        ),
    ),
    Dial(
        "VisionWeights", "choice", "mmap", "mode", "Vision weight placement: mmap demand-pages weights only when an image arrives; ram holds them resident.",
        "Pictures", options=("ram", "mmap"), engine_mapping="$env:FREETOKEN_VISION_WEIGHTS",
        plain="Vision weights", blurb="Keep picture weights in RAM or read them on demand.", option_labels=("At start-up, kept in PC memory", "Only when a picture arrives"),
        effects=("ram:down", "speed:mixed"),
        info=(
            "Only when a picture arrives leaves the 856 MiB of picture weights on the drive until the "
            "first picture, handing that PC memory to the model's 47.7 GiB lookup table instead. The "
            "first picture after start-up is slower. At start-up reads them once and keeps them. "
            "Piece-by-piece processing only."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_SPECULATE", "toggle", "0", "boolean", "Enable Multi-Token Prediction speculative decoding draft engine.",
        "MTP", source="env", engine_mapping="$env:FREETOKEN_MTP_SPECULATE",
        plain="MTP speculative decoding", blurb="Turn on draft-and-check speculative decoding.", effects=("speed:mixed", "vram:up"),
        info=(
            "Experimental. A small extra model drafts several words at once and the main model checks "
            "them in one step; when the guesses land, answers come faster. Needs the private MTP files "
            "on this PC, and its head takes 2.25 GiB of card memory when kept resident. The trial's "
            "results are in the research notes; leave it off unless you are testing it."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_RESIDENT", "toggle", "0", "boolean", "Keep MTP draft head model weights permanently resident in GPU VRAM.",
        "MTP", source="env", engine_mapping="$env:FREETOKEN_MTP_RESIDENT",
        plain="MTP resident draft head", blurb="Keep the MTP draft head loaded on the GPU.", advanced=True, effects=("speed:up", "vram:up"),
        info=(
            "Keeps the guess-ahead head on the card all the time (2.25 GiB) instead of loading it when "
            "needed. Faster guesses, less room for expert slots. Only matters when Guess ahead is on."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_SHADOW", "toggle", "0", "boolean", "Run MTP in passive shadow verification mode without returning draft tokens.",
        "MTP", source="env", engine_mapping="$env:FREETOKEN_MTP_SHADOW",
        plain="MTP shadow mode", blurb="Measure draft quality without returning draft tokens.", advanced=True, effects=("speed:down",),
        info=(
            "Runs the guessing head alongside normal answering and only records how often its guesses "
            "would have been right. Answers are unchanged and slightly slower. A testing aid."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_SPEC_DEPTH", "number", 5, "tokens", "Maximum draft chain length per speculation step (1-5).",
        "MTP", minimum=1, maximum=5, source="env", engine_mapping="$env:FREETOKEN_MTP_SPEC_DEPTH",
        plain="MTP speculation depth", blurb="Maximum number of draft tokens per speculation step.", slider=(1, 5, 1), effects=("speed:mixed",),
        info=(
            "How many words the guessing head drafts before the main model checks them. Checking "
            "costs about the same whether it checks one guess or five, so fewer guesses rarely helps: "
            "measured 2026-09-01 on this PC, 1 and 2 lost to guessing off everywhere and 5 was the "
            "best. The engine caps this at 5. Only matters when Guess ahead is on."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_SPEC_GRAPH", "toggle", "1", "boolean", "Capture speculation verification cycles inside CUDA graphs for lower latency.",
        "MTP", source="env", engine_mapping="$env:FREETOKEN_MTP_SPEC_GRAPH",
        plain="MTP verify graph", blurb="Use a captured graph for MTP verification.", effects=("speed:up", "boot:up", "vram:up"),
        info=(
            "Records the guess-checking step once at start-up so it replays with less overhead each "
            "time. Start-up takes about a second longer and the recordings use a little card memory. "
            "Off is never the faster choice: measured 2026-09-05 on this PC, a 9,000-token chat "
            "answered at about 23 words per second with it off. Only matters when Guess ahead is on."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_SPEC_CONF_CUT", "number", 0.8, "probability", "Stop drafting before the first draft token whose top-1 probability is below this (0 disables).",
        "MTP", minimum=0.0, maximum=1.0, numeric_kind="float", source="env",
        engine_mapping="$env:FREETOKEN_MTP_SPEC_CONF_CUT",
        plain="MTP confidence cut", blurb="Stop a draft chain when confidence falls below this value.", slider=(0, 1, 0.05), effects=("speed:up",),
        info=(
            "Before each guess the guessing head says how sure it is, 0 to 1. The chain stops at the "
            "first guess below this number, so a chain of 5 may end after 1 or 2 and the doomed "
            "guesses are never checked. Measured 2026-09-01 over 600 rounds: guesses that landed "
            "averaged 0.91 sureness, guesses that missed 0.46, so 0.8 separates them well. 0 switches "
            "the cut off and every round checks the full chain. Only matters when Guess ahead is on."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_SPEC_COST_AWARE", "toggle", "1", "boolean", "Adapt the speculation bar to measured cycle and plain-step wall time.",
        "MTP", source="env", engine_mapping="$env:FREETOKEN_MTP_SPEC_COST_AWARE",
        plain="MTP cost-aware", blurb="Adjust speculation to measured timing.", effects=("speed:up",),
        info=(
            "The server keeps timing how long a guessing round takes against a plain word and moves "
            "the worth-it bar to match, instead of holding the fixed bar below. Measured 2026-09-01: "
            "with this on an 8,000-token chat ran at 49.5 words per second, the best of any guessing "
            "setup, against 11 to 24 percent slower than guessing off with a fixed bar. Leave it on."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_SPEC_MIN_EMITTED", "number", 2.4, "tokens", "Minimum emitted tokens per speculative cycle before the request falls back to plain decode (0 never falls back).",
        "MTP", minimum=0.0, maximum=6.0, numeric_kind="float", source="env",
        engine_mapping="$env:FREETOKEN_MTP_SPEC_MIN_EMITTED",
        plain="MTP min emitted", blurb="Minimum tokens a speculative cycle must earn.", slider=(0, 6, 0.1), advanced=True, effects=("speed:mixed",),
        info=(
            "A guessing round costs about as much as 2.4 plain words when the sureness cut is on, "
            "3.6 when it is off. A chat whose rounds keep earning less than this bar is switched back "
            "to plain typing, with a fresh try now and then. Measured 2026-09-01: prose lost at a 3.2 "
            "bar with the cut off and recovered as the bar rose to 3.6. If you set the cut to 0, "
            "raise this to 3.6. 0 means never fall back. The engine caps it at guesses plus one, "
            "so a chain of 1 cannot hold a bar above 2. Only matters when Guess ahead is on."
        ),
    ),
    Dial(
        "ExpertLoad", "choice", "parallel", "mode", "Strategy for loading expert banks into RAM (parallel uses unbuffered I/O on Windows).",
        "Server & advanced", options=("auto", "serial", "parallel"), engine_mapping="--expert-load <mode>",
        plain="Expert load", blurb="Choose how expert weight files are read at startup.", option_labels=("Automatic", "One file at a time", "Several files at once"),
        advanced=True, effects=("boot:down", "ram:up"),
        info=(
            "Several files at once reads the 63 GiB of expert pieces with many threads straight from "
            "the drive. Measured 2026-09-02 on this PC: that phase drops from 43 to 37 seconds, and the "
            "whole start-up is a wash (73.1 against 73.2 seconds) because other phases dominate. It "
            "needs about 1.4 GiB more PC memory at its peak; on a PC with less memory choose one file "
            "at a time. Automatic picks several at once here."
        ),
    ),
    Dial(
        "EnableCacheReport", "toggle", True, "boolean", "Enable periodic logging and telemetry reports for KV cache and expert slot utilization.",
        "Server & advanced", engine_mapping="--enable-cache-report",
        plain="Cache report", blurb="Write periodic KV and expert-cache usage reports.", advanced=True,
        info=(
            "Adds a regular line to the boot log saying how full the chat memory and expert slots are. "
            "Handy when tuning; no measurable speed cost. Off keeps the log quieter."
        ),
    ),
    Dial(
        "CollectRoutingStats", "toggle", True, "boolean", "Accumulate decode routing frequency histograms accessible via GET /v1/cache/routing.",
        "Server & advanced", engine_mapping="--moe-collect-decode-freq",
        plain="Routing stats", blurb="Count which experts are selected during decoding.", advanced=True, effects=("speed:down",),
        info=(
            "Keeps a tally of how often each expert piece is chosen, for research. Adds one small extra "
            "step per layer per word. The tally is what picked the six busiest layers for the card."
        ),
    ),
    Dial(
        "Port", "number", 2020, "port", "TCP port the main FreeToken OpenAI-compatible HTTP server listens on.",
        "Server & advanced", minimum=1, maximum=65529, engine_mapping="--port <port>",
        plain="Port", blurb="TCP port used by the main FreeToken server.", advanced=True,
        info=(
            "The number apps use to reach the server on this PC, like a door number. The server also "
            "uses the nine numbers after it for its own helpers, and this settings page lives on 2031. "
            "Change it only if another program already uses 2020."
        ),
    ),
    Dial(
        "DesktopPython", "path", "(Join-Path $env:LOCALAPPDATA 'FreeToken\\venv\\Scripts\\python.exe')", "path", "Path to Python interpreter in the FreeToken Desktop virtual environment.",
        "Server & advanced", engine_mapping="Launcher interpreter",
        plain="Desktop Python", blurb="Python program used to start the server.", browse="file", advanced=True,
        info=(
            "The Python program that runs the server. The default is the one the FreeToken Desktop app "
            "installed, which already has the graphics-card libraries. Leave it unless that install moves."
        ),
    ),
    # Deliberately an env toggle although only the helper reads it: an absent line loads as the
    # default "1" (on), which a launcher switch cannot express (absent = off). The server ignores it.
    Dial(
        "FREETOKEN_AUTO_RESTART", "toggle", "1", "boolean",
        "Settings helper watchdog: start the model server again on its own when it dies while it was meant to be running (rate-limited). Absent from the boot file means on.",
        "Server & advanced", source="env", engine_mapping="$env:FREETOKEN_AUTO_RESTART (read by the helper, not the server)",
        plain="Auto-restart after a crash", blurb="Start the server again by itself if it dies.",
        info=(
            "This page keeps an eye on the server. If it was running and then stops answering without "
            "anyone pressing Stop, the page starts it again with the saved settings, at most three times "
            "an hour. It does not fix the cause; it just gets you serving again within a few minutes. "
            "A crash that keeps coming back is still written to the log."
        ),
    ),
    Dial(
        "FREETOKEN_DIAGNOSTIC_MODE", "toggle", "0", "boolean",
        "Diagnostic boot (Linux/WSL helper only): CUDA_LAUNCH_BLOCKING=1, CUDA graphs and MTP spec graphs off so a card fault is reported at the kernel that caused it.",
        "Server & advanced", source="env", engine_mapping="$env:CUDA_LAUNCH_BLOCKING='1', FREETOKEN_MTP_SPEC_GRAPH=0 and --cuda-graph-max-bs 0 (Linux/WSL launch)",
        plain="Diagnostic mode", blurb="Slower boot that names the routine behind a card fault.", advanced=True,
        effects=("speed:down",),
        info=(
            "Turn this on only while hunting a crash. The server runs each piece of card work one at a "
            "time and skips its recorded fast path, so when the card reports a bad memory access the log "
            "names the exact routine instead of the next thing that happened to wait on it. Answers come "
            "noticeably slower. Turn it off again once the fault has been caught."
        ),
    ),
    Dial(
        "PleBackend", "choice", "auto", "reader",
        "How the PLE n-gram table is read: auto (disk on Linux, mmap when MTP is on), disk (io_uring row store, Linux only, not with MTP), mmap (demand-paged table), pinned (whole table in pinned RAM).",
        "Server & advanced", options=("auto", "disk", "mmap", "pinned"), source="helper", engine_mapping="--ple-backend <reader>",
        plain="PLE table reader", blurb="How the 48 GB n-gram table is read each word.", advanced=True,
        option_labels=("Automatic", "Direct disk reads (Linux, not with MTP)", "Memory-mapped file", "Whole table in PC memory"),
        effects=("speed:mixed", "ram:mixed"),
        info=(
            "Automatic picks the direct disk reader on Linux, or the memory-mapped reader when MTP "
            "speculative decoding is on (the disk reader cannot run under MTP). Memory-mapped reads go "
            "through the PC's file cache; Whole table in PC memory preloads all 48 GB and needs that much "
            "free RAM. Set by hand only to compare readers; a wrong choice is corrected at start with a note."
        ),
    ),
    Dial(
        "CudaGraphMaxBS", "number", 4, "batch size", "Maximum batch size captured into CUDA graphs (-1 disables graph capture).",
        "Server & advanced", minimum=-1, maximum=1024, engine_mapping="--cuda-graph-max-bs <N>",
        plain="CUDA graph max batch", blurb="Largest batch size included in CUDA graph capture.", slider=(1, 16, 1), auto_value=-1, auto_label="Off",
        advanced=True, effects=("speed:up", "vram:up", "boot:up"),
        info=(
            "The server records its per-word work once for each chat count up to this number and "
            "replays the recording, which is faster than rebuilding it every word. Each recording takes "
            "a little card memory (0.75 GiB is reserved for them) and a moment at start-up. Keep it at "
            "least as high as Chats answered at the same time."
        ),
    ),
)

# ---------------------------------------------------------------------------------------
# Model-aware reshaping. The catalogue above is sized for Qwen3.8-Flash-Next; adapt_dial
# re-derives the limits, slider tops, storage form and explanation of these dials from the
# chosen model folder's config.json (model_info.read_model) so another model gets its own
# longest chat, layer count and expert sizes instead of Qwen's.
# ---------------------------------------------------------------------------------------

MODEL_AWARE_DIALS = frozenset({
    "ModelPath",
    "ContextTokens",
    "KVCacheTokens",
    "MoECacheSize",
    "GpuOwnedLayers",
    "EnableVision",
    "FREETOKEN_MTP_SPECULATE",
})

# What the catalogue was measured with; applied when the model folder cannot be read.
REFERENCE_LIMITS: dict[str, dict[str, Any]] = {
    "ContextTokens": {"max": 262144, "slider": (1024, 262144, 1024), "limitNote": " (the built-in limit; the model folder could not be read)"},
    "GpuOwnedLayers": {"max": 48, "slider": (0, 48, 1), "limitNote": " (the built-in limit; the model folder could not be read)"},
}

# Slider top for expert slots on a model we have no sweep for: the slots that fit in this
# much card memory. 24 GiB is what a 32 GB card has left after the always-on weights, the
# chat memory reserve and the headroom on the measured build (memory audit, 2026-09-02).
SLOT_SLIDER_BUDGET_BYTES = 24 * _GIB

# ---------------------------------------------------------------------------------------
# The GPU-owned-layer charge against the expert-slot total.
#
# ``--moe-cache-size`` is the TOTAL expert-slot budget: each GPU-owned MoE layer holds one
# full expert layer and is CHARGED to it, and what is left must still clear the streaming
# floor. Mirrors the engine exactly -- ``engine/cache_budget.py``
# ``lru_slots_after_owned_charge`` (the ``if lru < floor`` refusal, line 69) with the floor
# its callers pass: ``2 * num_experts`` while MoE prefill overlap is on (the engine default,
# ``engine.py:2269`` and ``engine/memory_plan.py:811``; the overlap borrows two full
# expert-layer buffers) and ``num_experts`` without it. ``memory_plan._suggestion.slot_search``
# searches from the same two floors (``num_experts + charge``, ``2 * num_experts + charge``).
# The page cannot turn the overlap off, so it mirrors the on case: the stricter of the two,
# and the one every boot from this page actually gets.
#
# Live failure this guards, 2026-09-07 11:22 BST on the RTX 5090 serving box: the page saved
# MoECacheSize 4288 with GpuOwnedLayers auto:8 and the boot died immediately -- 8 x 512 = 4096
# slots charged, 192 left for the LRU against a floor of 1024. The page had accepted it
# silently, so the arithmetic below now runs before Save/Start instead of 100 s into a boot.
STREAMING_FLOOR_LAYERS = 2
#: The dial whose value is charged against the expert-slot total.
OWNED_LAYERS_DIAL = "GpuOwnedLayers"
#: The dial holding the total.
SLOT_TOTAL_DIAL = "MoECacheSize"


def streaming_floor_slots(experts_per_layer: int) -> int:
    """Slots the still-streaming layers need left over (engine floor, prefill overlap on)."""
    return STREAMING_FLOOR_LAYERS * max(0, int(experts_per_layer))


def minimum_slot_total(owned_layers: int, experts_per_layer: int) -> int:
    """Smallest ``MoECacheSize`` that can carry ``owned_layers`` and still feed the rest."""
    experts = max(0, int(experts_per_layer))
    return max(0, int(owned_layers)) * experts + streaming_floor_slots(experts)


def slot_floor_sentence(experts_per_layer: int) -> str:
    """One plain-language sentence stating the rule with a model's real numbers."""
    experts = max(0, int(experts_per_layer))
    if not experts:
        return ""
    floor = streaming_floor_slots(experts)
    return (
        f"Every layer kept whole on the card takes all of its experts out of this same total "
        f"({experts:,} slots each), and the layers that still stream need at least {floor:,} "
        f"slots left over, so the total must be at least {experts:,} x layers on the card "
        f"+ {floor:,}."
    )


def owned_layer_count(value: Any, num_moe_layers: int | None = None) -> int | None:
    """How many MoE layers a ``GpuOwnedLayers`` value makes the engine own, or None if unknown.

    Mirrors ``engine._parse_gpu_owned_layers_spec`` for every spelling this page accepts:
    ``""`` (off) -> 0, ``"auto"`` -> 6 (the launcher's old spelling of the six-layer default),
    ``"auto:N"`` -> N, a bare count ``"8"``/``8`` -> 8, an explicit id list ``"0,7"`` -> the
    number of DISTINCT ids (the engine parses a list into a set), and the fraction form
    ``"0.125"`` -> ``round(fraction * num_moe_layers)``, which needs the model's layer count
    and is None without it.
    """
    if isinstance(value, str) and "." in value.strip():
        try:
            fraction = float(value.strip())
        except ValueError:
            return None
        if not 0.0 <= fraction <= 1.0 or not num_moe_layers:
            return None
        return round(fraction * int(num_moe_layers))
    if isinstance(value, str) and "," in value:
        ids = set()
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.add(int(part))
            except ValueError:
                return None
        return len(ids)
    try:
        return stored_count(DIAL_BY_NAME[OWNED_LAYERS_DIAL], value)
    except (KeyError, TypeError, ValueError):
        return None


def _words(tokens: int) -> str:
    return f"{int(tokens * 0.75):,}"


def _token_step(top: int) -> int:
    if top >= 65536:
        return 1024
    if top >= 8192:
        return 256
    return 64


def adapt_dial(dial: Dial, model: ModelInfo | None) -> dict[str, Any]:
    """Overrides for one dial given a model: keys ``min``, ``max``, ``slider``, ``info``,
    ``storedAs`` and ``limitNote`` (appended to an out-of-range error). Empty when the
    dial does not depend on the model or the folder could not be read."""
    if dial.name not in MODEL_AWARE_DIALS:
        return {}
    if model is None or not model.found:
        # Unknown folder: hold the limits the catalogue was measured with, so a typo in the
        # model path cannot let a 4-million-token chat through to the launcher.
        return dict(REFERENCE_LIMITS.get(dial.name, {}))
    over: dict[str, Any] = {}
    ref = model.is_reference
    name = dial.name
    label = model.name or "this model"

    if name == "ModelPath":
        parts = [f"The folder now holds {label}"]
        parts.append(
            f"({model.architecture}), a model type this engine knows." if model.supported
            else f"({model.architecture or 'unknown type'}), which this engine does not list, so it may not start."
        )
        if model.is_moe:
            parts.append(
                f"It has {model.num_moe_layers} expert layers of {model.num_experts:,} experts each"
                + (f", {model.experts_per_token} used per word" if model.experts_per_token else "")
                + (f", stored {model.expert_format_label}" if model.expert_format_label else "")
                + "."
            )
        else:
            parts.append("It has no routed experts, so the expert-slot settings do not apply.")
        if model.max_context_tokens:
            parts.append(f"Its longest chat is {model.max_context_tokens:,} tokens (roughly {_words(model.max_context_tokens)} words).")
        parts.append(
            "The speeds quoted on this page were measured on this model." if ref
            else "Speeds quoted on this page were measured on Qwen3.8-Flash-Next and do not carry over; the sizes and limits below are computed for this model."
        )
        over["info"] = " ".join(parts)

    elif name == "ContextTokens" and model.max_context_tokens:
        top = model.max_context_tokens
        step = _token_step(top)
        over["max"] = top
        over["slider"] = (min(1024, top), top, step)
        over["limitNote"] = f" ({label} cannot read a longer chat)"
        text = (
            f"How long one conversation may grow, in tokens. A token is about three quarters of a "
            f"word, so {label}'s limit of {top:,} tokens is roughly {_words(top)} words. The server "
            f"sets this much room aside for a single chat. "
        )
        if ref:
            text += (
                "At 25,344 bytes per token the full 262,144 costs about 6.2 GiB of card memory at "
                "the normal precision, half that at the compact precision."
            )
        else:
            text += "Card memory per token has not been measured for this model; longer costs more."
        over["info"] = text

    elif name == "KVCacheTokens" and model.max_context_tokens:
        top = min(4194304, max(8192, 2 * model.max_context_tokens))
        over["slider"] = (0, top, _token_step(top))
        text = (
            "The total chat memory the card keeps for all chats together. Bigger means more chats "
            "can run at once with long histories before the server has to re-read them. "
        )
        if ref:
            text += "Each 65,536 tokens costs about 1.55 GiB of card memory (measured 2026-09-02). "
        else:
            text += f"The slider runs to twice {label}'s longest chat; the cost per token is not measured for this model. "
        text += (
            "Automatic (0) does not match the longest single chat: it grows the pool until the free "
            "card memory is used up, which can leave nothing for the look-ahead trick or the "
            "recordings. With the look-ahead trick on, or a long chat, set this to the longest "
            f"single chat plus one page of 64 tokens - {model.max_context_tokens + 64:,} for "
            f"{model.max_context_tokens:,}."
        )
        over["info"] = text

    elif name == "MoECacheSize" and model.is_moe:
        total = model.num_moe_layers * model.num_experts
        per = model.bytes_per_expert
        top = total
        if per:
            top = min(total, max(64, (SLOT_SLIDER_BUDGET_BYTES // per) // 64 * 64))
        over["max"] = total
        over["limitNote"] = f" ({label} has {total:,} expert pieces)"
        slider_low = min(1024, top)
        if slider_low >= top:
            slider_low = 0
        over["slider"] = (slider_low, top, 64 if top >= 2048 else 1)
        text = (
            f"{label} is made of {total:,} expert pieces"
            + (f" ({gib(model.total_expert_bytes)})" if model.total_expert_bytes else "")
            + " that live in PC memory; the card keeps this many of them ready"
            + (f" ({per / 1e6:.2f} MB each). Every 1,000 slots costs {gib(per * 1000)} of card memory." if per else ".")
        )
        if ref:
            text += (
                " This is the main speed dial. Measured 2026-09-02 on this PC: every 1,000 slots is "
                "worth about 5.8 words per second on an 8,000-token chat (73 words per second at "
                "6,750 slots); below about 4,750 the slowdown gets steep."
            )
        else:
            text += " More slots means fewer experts fetched from PC memory per word; the speed per slot has not been measured for this model."
        text += " Slots and chat memory share the same card memory."
        text += " " + slot_floor_sentence(model.num_experts or 0)
        text += " Automatic lets the engine pick the largest count that fits."
        over["info"] = text
        # What the page needs to print the live minimum under this field. Sent as metadata so
        # the page never hard-codes this model's 512 experts or its 1,024-slot streaming floor.
        over["expertsPerLayer"] = int(model.num_experts or 0)
        over["streamingFloor"] = streaming_floor_slots(model.num_experts or 0)
        over["ownedDial"] = OWNED_LAYERS_DIAL

    elif name == "GpuOwnedLayers" and model.is_moe:
        layers = model.num_moe_layers
        over["max"] = layers
        over["slider"] = (0, layers, 1)
        over["storedAs"] = "auto:{n}" if ref else "{n}"
        over["limitNote"] = f" ({label} has {layers} expert layers)"
        text = "Keeps every expert of the chosen layers on the card so those layers never need PC memory. "
        if model.bytes_per_layer:
            text += (
                f"On {label} each layer hands back {gib(model.bytes_per_layer)} of PC memory and takes "
                f"{gib(model.bytes_per_layer)} of card memory, charged against the expert slots above "
                f"({model.num_experts:,} slots per layer). All {layers} layers would need "
                f"{gib(model.bytes_per_layer * layers)}. "
            )
        else:
            text += f"{label} has {layers} expert layers of {model.num_experts:,} experts; each kept layer takes that many expert slots. "
        if ref:
            text += (
                "The layers are taken from a busiest-first ranking measured on this PC; six was the "
                "measured sweet spot with 4,188 slots. 0 turns this off."
            )
        else:
            text += (
                "The busiest-first ranking was measured for Qwen3.8 only, so on this model the chosen "
                "number of layers is spread evenly through the model. 0 turns this off."
            )
        text += " " + slot_floor_sentence(model.num_experts or 0)
        over["info"] = text

    elif name == "EnableVision" and not model.has_vision:
        over["info"] = f"{label} has no picture part, so this switch does nothing for it. Leave it off. " + dial.info

    elif name == "FREETOKEN_MTP_SPECULATE" and not model.has_mtp:
        over["info"] = f"{label} ships no guess-ahead head, so this cannot work for it. Leave it off. " + dial.info

    return over


DIAL_BY_NAME = {dial.name: dial for dial in DIALS}
DIAL_CATALOG = DIALS
PRIMARY_DIALS = tuple(dial for dial in DIALS if dial.name not in {
    "KVParkIdleMs",
    "KVParkMinTokens",
    "KVParkRAMGiB",
    "KVParkSSDDir",
    "KVParkSSDGiB",
    "KVParkWindowMiB",
    "MoEVramReserveBytes",
    "MoECacheHeadroomBytes",
})
ENV_DIALS = frozenset(dial.name for dial in DIALS if dial.source == "env")
EXTENSION_DIALS = frozenset(
    {
        "KVParkIdleMs",
        "KVParkMinTokens",
        "KVParkRAMGiB",
        "KVParkSSDDir",
        "KVParkSSDGiB",
        "KVParkWindowMiB",
        "MoEVramReserveBytes",
        "MoECacheHeadroomBytes",
    }
)


def _toggle_value(value: Any, *, allow_text: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if allow_text and isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError("must be a boolean")


_STORED_COUNT_RE = re.compile(r"^(?:(?P<prefix>[A-Za-z]+):)?(?P<n>\d+)$")


def stored_count(dial: Dial, value: Any) -> int:
    """The count behind a text-stored number ("auto:3" -> 3, "auto" -> 6, "" -> 0, 4 -> 4).

    A bare "auto" is the launcher's old spelling of the six-layer default. An explicit id
    list ("3,7,11") counts its entries so the slider still shows how many layers it means.
    """
    if isinstance(value, bool):
        raise ValueError("must be a number")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != int(value):
            raise ValueError("must be a whole number")
        return int(value)
    if not isinstance(value, str):
        raise ValueError("must be a number or text")
    text = value.strip()
    if text == dial.stored_zero or text == "":
        return 0
    if text == "auto":
        return 6
    if "," in text:
        return len([part for part in text.split(",") if part.strip()])
    if "." in text:
        # the engine's fraction form ("0.125" of the layers): kept as typed, and counted as 0
        # here because the layer total is not known at this point
        frac = float(text)
        if not 0.0 <= frac <= 1.0:
            raise ValueError("a fraction must be between 0 and 1")
        return 0
    match = _STORED_COUNT_RE.match(text)
    if match is None:
        raise ValueError("must be a count like 3 or auto:3")
    return int(match.group("n"))


def stored_text(dial: Dial, count: int, stored_as: str | None = None) -> str:
    """Format a count back into launcher text (0 -> stored_zero)."""
    if count <= 0:
        return dial.stored_zero
    return (stored_as if stored_as is not None else dial.stored_as).format(n=count)


def canonical_value(dial: Dial, value: Any, stored_as: str | None = None) -> Any:
    if dial.control == "toggle":
        enabled = _toggle_value(value, allow_text=dial.source == "env")
        return ("1" if enabled else "0") if dial.source == "env" else enabled
    if dial.stored_as:
        if isinstance(value, str) and ("," in value or "." in value.strip()):
            # an explicit id list or fraction typed by hand stays as typed (checked above)
            stored_count(dial, value)
            return value.strip()
        count = stored_count(dial, value)
        keep = stored_as
        if keep is None and isinstance(value, str) and value.strip():
            # a text value keeps its own spelling ("3" stays "3", "auto:3" stays "auto:3")
            keep = "{n}" if _STORED_COUNT_RE.fullmatch(value.strip()) and ":" not in value else None
        return stored_text(dial, count, keep)
    if dial.control == "number":
        if isinstance(value, bool):
            raise ValueError("must be a number")
        if dial.numeric_kind == "float":
            parsed_float = float(value)
            if not math.isfinite(parsed_float):
                raise ValueError("must be finite")
            return parsed_float
        parsed = int(value)
        if isinstance(value, float) and value != parsed:
            raise ValueError("must be a whole number")
        return parsed
    if dial.control in {"choice", "path"}:
        if not isinstance(value, str):
            raise ValueError("must be text")
        return value
    return value


def validate_settings(
    settings: dict[str, Any],
    model: ModelInfo | None = None,
    *,
    ceilings_only: bool = False,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the contract's field/message list without importing any engine package.

    ``model`` (a ModelInfo) applies that model's limits: longest chat, expert-layer count.
    Without one, or with a folder that could not be read, the limits the catalogue was
    measured with apply (REFERENCE_LIMITS). ``ceilings_only`` checks just the technical
    ceilings: the boot-file writer and the profile store use it, since the app has already
    applied the model limits and a stored profile may belong to a different model.

    ``context`` is the settings already saved in the boot file. Only the cross-field check
    below reads it, so a patch that touches one half of a pair is still checked against the
    other half the boot will actually use."""
    errors: list[dict[str, Any]] = []
    for name, value in settings.items():
        dial = DIAL_BY_NAME.get(name)
        if dial is None:
            errors.append({"field": name, "message": f"Unknown setting {name}"})
            continue
        try:
            parsed = canonical_value(dial, value)
        except (TypeError, ValueError, OverflowError):
            errors.append({"field": name, "message": f"Value {value!r} for {name} {dial.control} is invalid"})
            continue
        over = {} if ceilings_only else adapt_dial(dial, model)
        minimum = over.get("min", dial.minimum)
        maximum = over.get("max", dial.maximum)
        limit_note = over.get("limitNote", "")
        compared = stored_count(dial, parsed) if dial.stored_as else parsed
        if dial.options is not None and parsed not in dial.options:
            choices = ", ".join(repr(item) for item in dial.options)
            errors.append({"field": name, "message": f"Value {parsed!r} for {name} must be one of {choices}"})
        if minimum is not None and compared < minimum:
            errors.append({"field": name, "message": f"Value {compared} below minimum {minimum}"})
        if maximum is not None and compared > maximum:
            errors.append({"field": name, "message": f"Value {compared} exceeds maximum {maximum}{limit_note}"})
        if dial.control == "path" and "\x00" in parsed:
            errors.append({"field": name, "message": f"Value for {name} contains a NUL character"})
    if not ceilings_only:
        errors.extend(
            _expert_slot_charge_errors(
                settings, model, context, failed={error["field"] for error in errors}
            )
        )
        errors.extend(_kv_dynamic_pool_errors(settings, model, context))
    return errors


def _read_setting_number(settings: dict[str, Any], context: dict[str, Any] | None, name: str) -> int | float | None:
    """Read ``name`` from ``settings``, falling back to the saved boot-file ``context``.

    Same fallback the expert-slot-charge check uses for the ``MoECacheSize``/``GpuOwnedLayers``
    pair: a patch that only touches one half of a cross-field pair is still checked against
    the value the boot will actually use for the other half.
    """
    inherited = context or {}
    raw = settings.get(name, inherited.get(name))
    if raw is None:
        return None
    try:
        return canonical_value(DIAL_BY_NAME[name], raw)
    except (TypeError, ValueError, OverflowError):
        return None


def _kv_dynamic_enabled(settings: dict[str, Any], context: dict[str, Any] | None) -> bool:
    """Resolve ``KVDynamic`` the same settings-then-context way ``_read_setting_number`` does.

    Falls back to the dial's own default (on) only when the flag is absent from both: a real
    boot file loaded through ``BootFile`` always carries an explicit value for every non-helper
    toggle (``False`` when the boot predates this dial), so this default only applies to a bare
    settings dict checked with no boot context at all.
    """
    inherited = context or {}
    raw = settings.get("KVDynamic", inherited.get("KVDynamic", True))
    try:
        return bool(canonical_value(DIAL_BY_NAME["KVDynamic"], raw))
    except (TypeError, ValueError, OverflowError):
        return False


def _kv_dynamic_pool_errors(
    settings: dict[str, Any], model: ModelInfo | None, context: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Cross-field checks for the dynamic KV pool dials, mirroring the engine's own refusals
    (a floor above the ceiling -- or, with KV cache tokens on automatic, at or above the model's
    own context -- a step under 8192, a floor off the 64-token page grid) so the user is told at
    save time instead of at boot. The engine only refuses these in conjunction
    with ``--kv-dynamic``, so these checks apply only while the resolved ``KVDynamic`` is on.

    Also runs only when the patch touches one of the dials in play, the same restraint
    ``_expert_slot_charge_errors`` uses for the ``MoECacheSize``/``GpuOwnedLayers`` pair: a save
    that changes neither half of a pair must not be refused for a combination the boot file
    already holds.
    """
    errors: list[dict[str, Any]] = []
    if not _kv_dynamic_enabled(settings, context):
        return errors
    # "KVDynamic" itself counts as touching the pair: flipping the pool on activates whatever
    # floor/ceiling the boot file already holds, so a patch of exactly {"KVDynamic": True} must
    # still be checked against them, not just a patch that names KVFloorTokens/KVCacheTokens.
    if settings.keys() & {"KVFloorTokens", "KVCacheTokens", "KVDynamic"}:
        floor = _read_setting_number(settings, context, "KVFloorTokens")
        ceiling = _read_setting_number(settings, context, "KVCacheTokens")
        model_context = int(getattr(model, "max_context_tokens", None) or 0) if model else 0
        if floor is not None and ceiling is not None and ceiling > 0 and floor > ceiling:
            errors.append({
                "field": "KVFloorTokens",
                "message": f"Smallest KV memory {floor} must not exceed KV cache tokens {ceiling}",
            })
        elif floor is not None and not ceiling and model_context and floor >= model_context:
            # KVCacheTokens 0 means "auto": the engine then takes the model's own context as
            # the ceiling and refuses --kv-dynamic outright when the floor reaches it. Say so
            # here instead of letting the boot fail on a default the user never typed.
            errors.append({
                "field": "KVFloorTokens",
                "message": (
                    f"Smallest KV memory {floor} must be below this model's longest chat "
                    f"{model_context} (KV cache tokens is on automatic)"
                ),
            })
    if "KVFloorTokens" in settings:
        floor = _read_setting_number(settings, context, "KVFloorTokens")
        if floor is not None and floor % 64 != 0:
            errors.append({
                "field": "KVFloorTokens",
                "message": f"Smallest KV memory {floor} must be a multiple of 64 (the shipping page size)",
            })
    if "KVStepTokens" in settings:
        step = _read_setting_number(settings, context, "KVStepTokens")
        if step is not None and step < 8192:
            errors.append({"field": "KVStepTokens", "message": "KV growth step must be at least 8192 tokens"})
    return errors


def _expert_slot_charge_errors(
    settings: dict[str, Any],
    model: ModelInfo | None,
    context: dict[str, Any] | None,
    *,
    failed: set,
) -> list[dict[str, Any]]:
    """Refuse an expert-slot total the GPU-owned layers would eat (see STREAMING_FLOOR_LAYERS).

    Runs only when the caller touched one of the two dials: the pair is a property of the
    boot as a whole, but a save that changes neither must not be refused for a total the
    boot file already holds. The message quotes the same arithmetic the engine's refusal
    does, in the page's words, and carries ``minimum`` so the page can offer that value.
    """
    if not settings.keys() & {SLOT_TOTAL_DIAL, OWNED_LAYERS_DIAL}:
        return []
    if failed & {SLOT_TOTAL_DIAL, OWNED_LAYERS_DIAL}:
        # the value is already refused on its own terms; a second message about the pair
        # would only bury the first
        return []
    if model is None or not model.is_moe or not model.num_experts:
        # no readable MoE geometry: the charge per layer is unknown, so there is nothing to
        # check (the engine will still refuse, with the numbers it can see)
        return []
    inherited = context or {}
    raw_total = settings.get(SLOT_TOTAL_DIAL, inherited.get(SLOT_TOTAL_DIAL))
    raw_owned = settings.get(OWNED_LAYERS_DIAL, inherited.get(OWNED_LAYERS_DIAL))
    if raw_total is None or raw_owned is None:
        return []
    try:
        total = int(canonical_value(DIAL_BY_NAME[SLOT_TOTAL_DIAL], raw_total))
    except (TypeError, ValueError, OverflowError):
        return []
    if total <= 0:
        # 0 is "Automatic": --moe-cache-auto charges the owned layers through the budget
        # before the split, so there is no total to run short (cache_budget.py:64-65)
        return []
    layers = owned_layer_count(raw_owned, model.num_moe_layers)
    if not layers or layers < 0:
        # unparsable, or nothing owned: with no charge the engine applies no LRU floor to an
        # explicit total either (lru_slots_after_owned_charge only runs for a non-empty owned
        # set), so a page floor here would refuse boots the engine accepts
        return []
    experts = int(model.num_experts)
    charge = layers * experts
    minimum = minimum_slot_total(layers, experts)
    if total >= minimum:
        return []
    left = total - charge
    floor = streaming_floor_slots(experts)
    layer_word = "layer" if layers == 1 else "layers"
    takes = "takes" if layers == 1 else "take"
    return [
        {
            "field": SLOT_TOTAL_DIAL,
            "message": (
                f"{layers} {layer_word} on the card {takes} {charge:,} of the {total:,} expert "
                f"slots, leaving {left:,}; the layers that still stream need at least "
                f"{floor:,}. Set expert slots to {minimum:,} or more, or keep fewer layers "
                f"on the card."
            ),
            "minimum": minimum,
        }
    ]


def normalise_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize values after validation for the parser and profile store."""
    return {name: canonical_value(DIAL_BY_NAME[name], value) for name, value in settings.items()}


def normalize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return normalise_settings(settings)


def dial_value_for_display(dial: Dial, value: Any) -> Any:
    if dial.source == "env" and dial.control == "toggle":
        return _toggle_value(value, allow_text=True)
    return value


__all__ = [
    "DIALS",
    "DIAL_BY_NAME",
    "DIAL_CATALOG",
    "EFFECT_AXES",
    "EFFECT_DIRECTIONS",
    "GROUP_INFO",
    "PRIMARY_DIALS",
    "ENV_DIALS",
    "EXTENSION_DIALS",
    "MODEL_AWARE_DIALS",
    "OWNED_LAYERS_DIAL",
    "SLOT_TOTAL_DIAL",
    "STREAMING_FLOOR_LAYERS",
    "Dial",
    "adapt_dial",
    "canonical_value",
    "minimum_slot_total",
    "owned_layer_count",
    "slot_floor_sentence",
    "streaming_floor_slots",
    "stored_count",
    "stored_text",
    "dial_value_for_display",
    "normalise_settings",
    "normalize_settings",
    "validate_settings",
]

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
        }


GROUP_INFO: dict[str, dict[str, str]] = {
    "Model and context": {
        "plain": "Model and chat length",
        "info": "Which model runs, how long one chat may get, and how its running memory is stored on the card.",
    },
    "Chats": {
        "plain": "Chats at once",
        "info": "How many people or apps the server answers at the same time.",
    },
    "KV notes and parking": {
        "plain": "Remembering idle chats",
        "info": "Move an idle chat's memory off the card and bring it back when the chat continues, instead of re-reading the whole conversation.",
    },
    "Expert slots and card memory": {
        "plain": "Card memory",
        "info": "How the graphics card's 32 GB is split between ready-to-use expert pieces, chat memory and a safety cushion. The main speed dial lives here.",
    },
    "Picture input": {
        "plain": "Pictures",
        "info": "Let chats include pictures, and choose where the picture weights wait.",
    },
    "Look-ahead speed trick (MTP)": {
        "plain": "Guess-ahead speed trick (experimental)",
        "info": "A trial feature that drafts several words at once and checks them in one go. Needs the private MTP files on this PC.",
    },
    "Loading and diagnostics": {
        "plain": "Start-up and health checks",
        "info": "How the model is read from the drive at start-up and what gets written to the log.",
    },
    "Advanced": {
        "plain": "Other settings",
        "info": "Rarely changed. Wrong values here stop the server from starting.",
    },
}


# Keep this catalogue independent of the model and CUDA packages. The current model path is read
# from the local boot file at runtime rather than copied into tracked source.
DIALS: tuple[Dial, ...] = (
    Dial(
        "ModelPath", "path", "", "", "Filesystem directory containing the model weights and tokenizer config.",
        "Model and context", engine_mapping="--model <path>",
        plain="Model folder", browse="model",
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
        "Model and context", minimum=64, maximum=4194304, engine_mapping="--kv-reserve-tokens <N>",
        plain="Longest single chat", slider=(1024, 262144, 1024), effects=("vram:up",),
        info=(
            "How long one conversation may grow, in tokens. A token is about three quarters of a "
            "word, so 262,144 tokens is roughly 200,000 words. The server sets this much room aside "
            "for a single chat; at 25,344 bytes per token the full 262,144 costs about 6.2 GiB of card "
            "memory at the normal precision, half that at the compact precision."
        ),
    ),
    Dial(
        "KVCacheTokens", "number", 262144, "tokens", "Total capacity of the KV token cache pool (0 selects automatic sizing).",
        "Model and context", minimum=0, maximum=4194304, engine_mapping="--num-tokens <N>",
        plain="Chat memory pool", slider=(0, 524288, 8192), auto_value=0, auto_label="Automatic",
        effects=("vram:up", "speed:up"),
        info=(
            "The total chat memory the card keeps for all chats together. Bigger means more chats can "
            "run at once with long histories before the server has to re-read them. Each 65,536 tokens "
            "costs about 1.55 GiB of card memory (measured 2026-09-02). Automatic lets the engine fit "
            "it to whatever card memory is left after the expert slots."
        ),
    ),
    Dial(
        "KVDtype", "choice", "bf16", "dtype", "Storage precision for QSA KV cache (bf16 is safe default; fp8 saves ~48% KV VRAM).",
        "Model and context", options=("bf16", "fp8"), engine_mapping="--kv-dtype fp8 (if fp8)",
        plain="Chat memory precision", option_labels=("Normal (bf16)", "Compact (fp8)"),
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
        "Chats", minimum=1, maximum=16, engine_mapping="--max-running-requests <N>",
        plain="Chats answered at the same time", slider=(1, 16, 1), effects=("speed:mixed", "vram:up"),
        info=(
            "How many chats the server works on together. Each one answers a little slower, but the "
            "total output goes up: measured 2026-09-03, four chats together produced 15.5 to 49.5 words "
            "per second combined depending on how warm the caches were, against about 73 for one chat "
            "alone. Each extra chat needs its own slice of chat memory."
        ),
    ),
    Dial(
        "KVPark", "choice", "off", "backend", "Offload inactive KV cache prefixes to RAM or SSD between multi-turn chat interactions.",
        "KV notes and parking", options=("off", "ram", "ssd"), engine_mapping="--kv-park <mode>",
        plain="Where idle chats are kept", option_labels=("Off (re-read the chat)", "PC memory", "SSD"),
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
        "KV notes and parking", minimum=0, maximum=86400000, engine_mapping="--kv-park-idle-ms <N>",
        plain="Quiet time before a chat is parked", slider=(0, 600, 5), display_unit="s", display_factor=1000,
        advanced=True, effects=("speed:mixed",),
        info=(
            "How long a chat must sit quiet before it is moved off the card. 0 parks it the moment its "
            "turn ends. A short wait keeps a chat that is about to continue on the card; a long wait "
            "keeps card memory busy for chats that may never return."
        ),
    ),
    Dial(
        "KVParkMinTokens", "number", 8192, "tokens", "Minimum token prefix length required before eligible for KV cache parking.",
        "KV notes and parking", minimum=64, maximum=4194304, engine_mapping="--kv-park-min-tokens <N>",
        plain="Smallest chat worth parking", slider=(1024, 131072, 1024), advanced=True, effects=("speed:mixed",),
        info=(
            "Chats shorter than this are simply re-read, because that is nearly as fast as parking them. "
            "Measured 2026-09-03: an 8,192-token chat re-reads in 4.7 seconds and restores in 0.006 "
            "seconds from PC memory, 0.06 from the SSD, so 8,192 is a comfortable line."
        ),
    ),
    Dial(
        "KVParkRAMGiB", "number", 2.0, "GiB", "Maximum pinned host RAM budget allocated for parked KV cache prefixes.",
        "KV notes and parking", minimum=0.125, maximum=128.0, numeric_kind="float", engine_mapping="--kv-park-ram-gib <N>",
        plain="PC memory set aside for parked chats", slider=(0.5, 32, 0.5), effects=("ram:up", "speed:up"),
        info=(
            "Only used when idle chats are kept in PC memory. This memory is locked for the server and "
            "nothing else can use it while it runs. One 65,000-token chat needs 1.65 GiB; a full "
            "262,144-token chat needs 6.3 GiB (measured 2026-09-03). This PC has 96 GiB, of which the "
            "model itself already pins about 65 GiB."
        ),
    ),
    Dial(
        "KVParkSSDDir", "path", "~/.cache/freetoken/kv-park", "path", "Filesystem directory on high-speed SSD used for parked KV cache files.",
        "KV notes and parking", engine_mapping="--kv-park-ssd-dir <dir>",
        plain="Parking folder on the SSD", browse="folder", effects=("ssd:up",),
        info=(
            "Where parked chats are written when the SSD option is on. Pick a folder on a fast SSD; "
            "the files are read back at about 5.3 GiB per second on this PC's drive. They are deleted "
            "when the chat is dropped."
        ),
    ),
    Dial(
        "KVParkSSDGiB", "number", 32.0, "GiB", "Maximum disk storage budget on SSD allocated for parked KV cache files.",
        "KV notes and parking", minimum=0.125, maximum=8192.0, numeric_kind="float", engine_mapping="--kv-park-ssd-gib <N>",
        plain="Drive space for parked chats", slider=(1, 256, 1), effects=("ssd:up", "speed:up"),
        info=(
            "The most drive space parked chats may take. Each parked chat is 1.65 GiB per 65,000 "
            "tokens, so 32 GiB holds about 19 chats that long. When the space is full the oldest "
            "parked chat is dropped and will be re-read if it continues."
        ),
    ),
    Dial(
        "KVParkWindowMiB", "number", 256, "MiB", "Staging window size in host RAM for overlapping SSD disk reads with GPU copies.",
        "KV notes and parking", minimum=1, maximum=4096, engine_mapping="--kv-park-window-mib <N>",
        plain="SSD transfer buffer", slider=(16, 2048, 16), advanced=True, effects=("ram:up",),
        info=(
            "Two buffers of this size sit in locked PC memory to move parked chats between the SSD and "
            "the card. 256 MiB was measured 2026-09-03 at 5.3 GiB per second reads and 1.6 GiB per "
            "second writes; bigger buffers were not faster. Costs twice this amount of PC memory."
        ),
    ),
    Dial(
        "MoECacheSize", "number", 4188, "slots", "Total number of MoE expert slots allocated in GPU VRAM (0 selects automatic sizing).",
        "Expert slots and card memory", minimum=0, maximum=1048576, engine_mapping="--moe-cache-size <N>",
        plain="Expert slots on the card", slider=(1024, 8192, 64), auto_value=0, auto_label="Automatic",
        effects=("speed:up", "vram:up"),
        info=(
            "The model is made of 24,576 expert pieces (63 GiB) that live in PC memory; the card keeps "
            "this many of them ready (2.77 MB each). This is the main speed dial. Measured 2026-09-02 "
            "on this PC: every 1,000 slots costs 2.58 GiB of card memory and is worth about 5.8 words "
            "per second on an 8,000-token chat (73 words per second at 6,750 slots). Below about 4,750 "
            "the slowdown gets steep. Slots and chat memory share the same card memory, so raising one "
            "leaves less for the other. Automatic lets the engine pick the largest count that fits."
        ),
    ),
    Dial(
        # Stored as launcher text: "auto:N" = the N busiest layers of the ranking measured for
        # Qwen3.8 (engine GPU_OWNED_LAYER_RANK), "N" = N layers spread evenly through the model,
        # "" = off. The engine accepts any N up to the model's MoE layer count; the slider top and
        # the storage form come from the chosen model (adapt_dial). Static maximum is a ceiling.
        "GpuOwnedLayers", "number", "auto", "layers", "MoE layers that remain permanently resident in GPU VRAM instead of streaming from host RAM.",
        "Expert slots and card memory", minimum=0, maximum=4096, engine_mapping="--moe-gpu-owned-layers <val>",
        plain="Layers kept whole on the card", slider=(0, 48, 1), stored_as="auto:{n}", stored_zero="",
        effects=("ram:down", "vram:up", "speed:mixed"),
        info=(
            "Keeps every expert of the chosen layers on the card so those layers never need PC memory. "
            "Each layer hands back 1.32 GiB of PC memory and takes 1.32 GiB of card memory, which is "
            "charged against the expert slots above (about 512 slots per layer). The layers are taken "
            "from a busiest-first ranking measured on this PC; six was the measured sweet spot with "
            "4,188 slots, and every extra layer removes about 512 streaming slots. 0 turns this off."
        ),
    ),
    Dial(
        "DenseQuant", "choice", "int8", "format", "Weight-only int8 quantization for dense non-MoE layers, saving ~3.9 GiB VRAM.",
        "Expert slots and card memory", options=("", "int8"), engine_mapping="$env:FREETOKEN_DENSE_QUANT",
        plain="Shrink the always-on weights", option_labels=("Off (full size)", "On (int8)"),
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
        "Expert slots and card memory", minimum=-1, maximum=34359738368, engine_mapping="--moe-vram-reserve-bytes <N>",
        plain="Card memory held back for later", slider=(0, 8, 0.25), display_unit="GiB", display_factor=GIB,
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
        "Expert slots and card memory", minimum=-1, maximum=34359738368, engine_mapping="--moe-cache-headroom-bytes <N>",
        plain="Free card memory cushion", slider=(0, 8, 0.25), display_unit="GiB", display_factor=GIB,
        auto_value=-1, auto_label="Automatic (1.5 GiB)", advanced=True, effects=("vram:up", "speed:mixed"),
        info=(
            "Card memory that must stay free after everything is loaded. Automatic keeps 1.5 GiB, the "
            "least any healthy start-up on this PC measured; a run that left only 0.55 GiB free answered "
            "at half speed. If the expert slot count does not leave this much, the server refuses to "
            "start and names the largest count that fits."
        ),
    ),
    Dial(
        "EmbedHost", "toggle", True, "boolean", "Pin the 1.27 GB token embedding table in host RAM to free GPU VRAM for expert slots.",
        "Expert slots and card memory", engine_mapping="$env:FREETOKEN_EMBED_HOST='1'",
        plain="Keep the word table in PC memory", effects=("vram:down", "ram:up"),
        info=(
            "The table that turns words into numbers is 1.27 GB. On, it lives in locked PC memory and "
            "the card fetches one row per word; that gives 1.27 GB of card memory back for expert slots "
            "at a cost too small to measure. Off, it sits on the card."
        ),
    ),
    Dial(
        "EnableVision", "toggle", True, "boolean", "Enable still-picture vision model weights and multimodal image input endpoints.",
        "Picture input", engine_mapping="$env:FREETOKEN_LOAD_VISION='1'",
        plain="Allow pictures in chats", effects=("ram:up", "boot:up"),
        info=(
            "Loads the picture-reading part of the model (856 MiB) so chats can include images. Where "
            "it waits is set by the two choices below. Off, pictures are refused and that memory is saved."
        ),
    ),
    Dial(
        "VisionPackagesPath", "path", "$visionPackages", "path", "Directory containing local Pillow and TorchVision dependencies for image processing.",
        "Picture input", engine_mapping="Added to $env:PYTHONPATH",
        plain="Picture software folder", browse="folder", advanced=True,
        info=(
            "The folder holding the extra picture libraries the server needs (Pillow and TorchVision). "
            "Only change it if you installed them somewhere else."
        ),
    ),
    Dial(
        "VisionExecution", "choice", "layer-stream", "mode", "Vision execution strategy: layer-stream stages bounded layers to GPU; gpu keeps all on GPU.",
        "Picture input", options=("layer-stream", "gpu"), engine_mapping="$env:FREETOKEN_VISION_EXECUTION",
        plain="Where pictures are processed", option_labels=("Piece by piece (saves card memory)", "All on the card"),
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
        "Picture input", options=("ram", "mmap"), engine_mapping="$env:FREETOKEN_VISION_WEIGHTS",
        plain="When picture weights are read", option_labels=("At start-up, kept in PC memory", "Only when a picture arrives"),
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
        "Look-ahead speed trick (MTP)", source="env", engine_mapping="$env:FREETOKEN_MTP_SPECULATE",
        plain="Guess ahead", effects=("speed:mixed", "vram:up"),
        info=(
            "Experimental. A small extra model drafts several words at once and the main model checks "
            "them in one step; when the guesses land, answers come faster. Needs the private MTP files "
            "on this PC, and its head takes 2.25 GiB of card memory when kept resident. The trial's "
            "results are in the research notes; leave it off unless you are testing it."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_RESIDENT", "toggle", "0", "boolean", "Keep MTP draft head model weights permanently resident in GPU VRAM.",
        "Look-ahead speed trick (MTP)", source="env", engine_mapping="$env:FREETOKEN_MTP_RESIDENT",
        plain="Keep the guessing head on the card", advanced=True, effects=("speed:up", "vram:up"),
        info=(
            "Keeps the guess-ahead head on the card all the time (2.25 GiB) instead of loading it when "
            "needed. Faster guesses, less room for expert slots. Only matters when Guess ahead is on."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_SHADOW", "toggle", "0", "boolean", "Run MTP in passive shadow verification mode without returning draft tokens.",
        "Look-ahead speed trick (MTP)", source="env", engine_mapping="$env:FREETOKEN_MTP_SHADOW",
        plain="Measure guesses only", advanced=True, effects=("speed:down",),
        info=(
            "Runs the guessing head alongside normal answering and only records how often its guesses "
            "would have been right. Answers are unchanged and slightly slower. A testing aid."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_SPEC_DEPTH", "number", 5, "tokens", "Maximum draft chain length per speculation step (1-5).",
        "Look-ahead speed trick (MTP)", minimum=1, maximum=5, source="env", engine_mapping="$env:FREETOKEN_MTP_SPEC_DEPTH",
        plain="How many words to guess ahead", slider=(1, 5, 1), effects=("speed:mixed",),
        info=(
            "How many words the guessing head drafts before the main model checks them. Checking "
            "costs about the same whether it checks one guess or five, so fewer guesses rarely helps: "
            "measured 2026-09-01 on this PC, 1 and 2 lost to guessing off everywhere and 5 was the "
            "best. The engine caps this at 5. Only matters when Guess ahead is on."
        ),
    ),
    Dial(
        "FREETOKEN_MTP_SPEC_GRAPH", "toggle", "1", "boolean", "Capture speculation verification cycles inside CUDA graphs for lower latency.",
        "Look-ahead speed trick (MTP)", source="env", engine_mapping="$env:FREETOKEN_MTP_SPEC_GRAPH",
        plain="Fast path for guess checking", effects=("speed:up", "boot:up", "vram:up"),
        info=(
            "Records the guess-checking step once at start-up so it replays with less overhead each "
            "time. Start-up takes about a second longer and the recordings use a little card memory. "
            "Off is never the faster choice: measured 2026-09-05 on this PC, a 9,000-token chat "
            "answered at about 23 words per second with it off. Only matters when Guess ahead is on."
        ),
    ),
    Dial(
        "ExpertLoad", "choice", "parallel", "mode", "Strategy for loading expert banks into RAM (parallel uses unbuffered I/O on Windows).",
        "Loading and diagnostics", options=("auto", "serial", "parallel"), engine_mapping="--expert-load <mode>",
        plain="How the model is read at start-up", option_labels=("Automatic", "One file at a time", "Several files at once"),
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
        "Loading and diagnostics", engine_mapping="--enable-cache-report",
        plain="Write memory reports to the log", advanced=True,
        info=(
            "Adds a regular line to the boot log saying how full the chat memory and expert slots are. "
            "Handy when tuning; no measurable speed cost. Off keeps the log quieter."
        ),
    ),
    Dial(
        "CollectRoutingStats", "toggle", True, "boolean", "Accumulate decode routing frequency histograms accessible via GET /v1/cache/routing.",
        "Loading and diagnostics", engine_mapping="--moe-collect-decode-freq",
        plain="Count which experts get used", advanced=True, effects=("speed:down",),
        info=(
            "Keeps a tally of how often each expert piece is chosen, for research. Adds one small extra "
            "step per layer per word. The tally is what picked the six busiest layers for the card."
        ),
    ),
    Dial(
        "Port", "number", 2020, "port", "TCP port the main FreeToken OpenAI-compatible HTTP server listens on.",
        "Advanced", minimum=1, maximum=65529, engine_mapping="--port <port>",
        plain="Door number apps connect to", advanced=True,
        info=(
            "The number apps use to reach the server on this PC, like a door number. The server also "
            "uses the nine numbers after it for its own helpers, and this settings page lives on 2031. "
            "Change it only if another program already uses 2020."
        ),
    ),
    Dial(
        "DesktopPython", "path", "(Join-Path $env:LOCALAPPDATA 'FreeToken\\venv\\Scripts\\python.exe')", "path", "Path to Python interpreter in the FreeToken Desktop virtual environment.",
        "Advanced", engine_mapping="Launcher interpreter",
        plain="Python program to run the server with", browse="file", advanced=True,
        info=(
            "The Python program that runs the server. The default is the one the FreeToken Desktop app "
            "installed, which already has the graphics-card libraries. Leave it unless that install moves."
        ),
    ),
    Dial(
        "CudaGraphMaxBS", "number", 4, "batch size", "Maximum batch size captured into CUDA graphs (-1 disables graph capture).",
        "Advanced", minimum=-1, maximum=1024, engine_mapping="--cuda-graph-max-bs <N>",
        plain="Fast path up to this many chats", slider=(1, 16, 1), auto_value=-1, auto_label="Off",
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
        text += "Automatic lets the engine fit it to whatever card memory is left after the expert slots."
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
        text += " Slots and chat memory share the same card memory. Automatic lets the engine pick the largest count that fits."
        over["info"] = text

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
) -> list[dict[str, str]]:
    """Return the contract's field/message list without importing any engine package.

    ``model`` (a ModelInfo) applies that model's limits: longest chat, expert-layer count.
    Without one, or with a folder that could not be read, the limits the catalogue was
    measured with apply (REFERENCE_LIMITS). ``ceilings_only`` checks just the technical
    ceilings: the boot-file writer and the profile store use it, since the app has already
    applied the model limits and a stored profile may belong to a different model."""
    errors: list[dict[str, str]] = []
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
    return errors


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
    "Dial",
    "adapt_dial",
    "canonical_value",
    "stored_count",
    "stored_text",
    "dial_value_for_display",
    "normalise_settings",
    "normalize_settings",
    "validate_settings",
]

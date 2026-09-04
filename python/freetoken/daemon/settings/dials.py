"""The torch-free settings catalogue and input checks for the Windows helper."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


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

    def as_dict(self, value: Any) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "control": self.control,
            "value": value,
            "unit": self.unit,
            "help": self.help,
            "options": list(self.options) if self.options is not None else None,
            "min": self.minimum,
            "max": self.maximum,
        }


# Keep this catalogue independent of the model and CUDA packages. The current model path is read
# from the local boot file at runtime rather than copied into tracked source.
DIALS: tuple[Dial, ...] = (
    Dial(
        "ModelPath", "path", "", "", "Filesystem directory containing the model weights and tokenizer config.",
        "Model and context", engine_mapping="--model <path>",
    ),
    Dial(
        "ContextTokens", "number", 262144, "tokens", "Maximum sequence length per request reserved in the KV cache.",
        "Model and context", minimum=64, maximum=262144, engine_mapping="--kv-reserve-tokens <N>",
    ),
    Dial(
        "KVCacheTokens", "number", 262144, "tokens", "Total capacity of the KV token cache pool (0 selects automatic sizing).",
        "Model and context", minimum=0, maximum=4194304, engine_mapping="--num-tokens <N>",
    ),
    Dial(
        "KVDtype", "choice", "bf16", "dtype", "Storage precision for QSA KV cache (bf16 is safe default; fp8 saves ~48% KV VRAM).",
        "Model and context", options=("bf16", "fp8"), engine_mapping="--kv-dtype fp8 (if fp8)",
    ),
    Dial(
        "MaxRunningRequests", "number", 4, "requests", "Maximum number of concurrent inference requests processed in parallel.",
        "Chats", minimum=1, maximum=16, engine_mapping="--max-running-requests <N>",
    ),
    Dial(
        "KVPark", "choice", "off", "backend", "Offload inactive KV cache prefixes to RAM or SSD between multi-turn chat interactions.",
        "KV notes and parking", options=("off", "ram", "ssd"), engine_mapping="--kv-park <mode>",
    ),
    Dial(
        "KVParkIdleMs", "number", 0, "ms", "Idle milliseconds before offloading inactive KV cache prefix to parking store.",
        "KV notes and parking", minimum=0, maximum=86400000, engine_mapping="--kv-park-idle-ms <N>",
    ),
    Dial(
        "KVParkMinTokens", "number", 8192, "tokens", "Minimum token prefix length required before eligible for KV cache parking.",
        "KV notes and parking", minimum=64, maximum=4194304, engine_mapping="--kv-park-min-tokens <N>",
    ),
    Dial(
        "KVParkRAMGiB", "number", 2.0, "GiB", "Maximum pinned host RAM budget allocated for parked KV cache prefixes.",
        "KV notes and parking", minimum=0.125, maximum=128.0, numeric_kind="float", engine_mapping="--kv-park-ram-gib <N>",
    ),
    Dial(
        "KVParkSSDDir", "path", "~/.cache/freetoken/kv-park", "path", "Filesystem directory on high-speed SSD used for parked KV cache files.",
        "KV notes and parking", engine_mapping="--kv-park-ssd-dir <dir>",
    ),
    Dial(
        "KVParkSSDGiB", "number", 32.0, "GiB", "Maximum disk storage budget on SSD allocated for parked KV cache files.",
        "KV notes and parking", minimum=0.125, maximum=8192.0, numeric_kind="float", engine_mapping="--kv-park-ssd-gib <N>",
    ),
    Dial(
        "KVParkWindowMiB", "number", 256, "MiB", "Staging window size in host RAM for overlapping SSD disk reads with GPU copies.",
        "KV notes and parking", minimum=1, maximum=4096, engine_mapping="--kv-park-window-mib <N>",
    ),
    Dial(
        "MoECacheSize", "number", 4188, "slots", "Total number of MoE expert slots allocated in GPU VRAM (0 selects automatic sizing).",
        "Expert slots and card memory", minimum=0, maximum=1048576, engine_mapping="--moe-cache-size <N>",
    ),
    Dial(
        "GpuOwnedLayers", "choice", "auto", "layers", "MoE layers that remain permanently resident in GPU VRAM instead of streaming from host RAM.",
        "Expert slots and card memory", options=("", "auto", "auto:1", "auto:2", "auto:3", "auto:4", "auto:5", "auto:6"), engine_mapping="--moe-gpu-owned-layers <val>",
    ),
    Dial(
        "DenseQuant", "choice", "int8", "format", "Weight-only int8 quantization for dense non-MoE layers, saving ~3.9 GiB VRAM.",
        "Expert slots and card memory", options=("", "int8"), engine_mapping="$env:FREETOKEN_DENSE_QUANT",
    ),
    Dial(
        "MoEVramReserveBytes", "number", -1, "bytes", "VRAM bytes reserved after expert cache sizing for CUDA graphs and draft heads (-1 = auto).",
        "Expert slots and card memory", minimum=-1, maximum=34359738368, engine_mapping="--moe-vram-reserve-bytes <N>",
    ),
    Dial(
        "MoECacheHeadroomBytes", "number", -1, "bytes", "Free VRAM cushion that expert slot cache must leave untouched (-1 = default 1.5 GiB).",
        "Expert slots and card memory", minimum=-1, maximum=34359738368, engine_mapping="--moe-cache-headroom-bytes <N>",
    ),
    Dial(
        "EmbedHost", "toggle", True, "boolean", "Pin the 1.27 GB token embedding table in host RAM to free GPU VRAM for expert slots.",
        "Expert slots and card memory", engine_mapping="$env:FREETOKEN_EMBED_HOST='1'",
    ),
    Dial(
        "EnableVision", "toggle", True, "boolean", "Enable still-picture vision model weights and multimodal image input endpoints.",
        "Picture input", engine_mapping="$env:FREETOKEN_LOAD_VISION='1'",
    ),
    Dial(
        "VisionPackagesPath", "path", "$visionPackages", "path", "Directory containing local Pillow and TorchVision dependencies for image processing.",
        "Picture input", engine_mapping="Added to $env:PYTHONPATH",
    ),
    Dial(
        "VisionExecution", "choice", "layer-stream", "mode", "Vision execution strategy: layer-stream stages bounded layers to GPU; gpu keeps all on GPU.",
        "Picture input", options=("layer-stream", "gpu"), engine_mapping="$env:FREETOKEN_VISION_EXECUTION",
    ),
    Dial(
        "VisionWeights", "choice", "mmap", "mode", "Vision weight placement: mmap demand-pages weights only when an image arrives; ram holds them resident.",
        "Picture input", options=("ram", "mmap"), engine_mapping="$env:FREETOKEN_VISION_WEIGHTS",
    ),
    Dial(
        "FREETOKEN_MTP_SPECULATE", "toggle", "0", "boolean", "Enable Multi-Token Prediction speculative decoding draft engine.",
        "Look-ahead speed trick (MTP)", source="env", engine_mapping="$env:FREETOKEN_MTP_SPECULATE",
    ),
    Dial(
        "FREETOKEN_MTP_RESIDENT", "toggle", "0", "boolean", "Keep MTP draft head model weights permanently resident in GPU VRAM.",
        "Look-ahead speed trick (MTP)", source="env", engine_mapping="$env:FREETOKEN_MTP_RESIDENT",
    ),
    Dial(
        "FREETOKEN_MTP_SHADOW", "toggle", "0", "boolean", "Run MTP in passive shadow verification mode without returning draft tokens.",
        "Look-ahead speed trick (MTP)", source="env", engine_mapping="$env:FREETOKEN_MTP_SHADOW",
    ),
    Dial(
        "FREETOKEN_MTP_SPEC_GRAPH", "toggle", "0", "boolean", "Capture speculation verification cycles inside CUDA graphs for lower latency.",
        "Look-ahead speed trick (MTP)", source="env", engine_mapping="$env:FREETOKEN_MTP_SPEC_GRAPH",
    ),
    Dial(
        "ExpertLoad", "choice", "parallel", "mode", "Strategy for loading expert banks into RAM (parallel uses unbuffered I/O on Windows).",
        "Loading and diagnostics", options=("auto", "serial", "parallel"), engine_mapping="--expert-load <mode>",
    ),
    Dial(
        "EnableCacheReport", "toggle", True, "boolean", "Enable periodic logging and telemetry reports for KV cache and expert slot utilization.",
        "Loading and diagnostics", engine_mapping="--enable-cache-report",
    ),
    Dial(
        "CollectRoutingStats", "toggle", True, "boolean", "Accumulate decode routing frequency histograms accessible via GET /v1/cache/routing.",
        "Loading and diagnostics", engine_mapping="--moe-collect-decode-freq",
    ),
    Dial(
        "Port", "number", 2020, "port", "TCP port the main FreeToken OpenAI-compatible HTTP server listens on.",
        "Advanced", minimum=1, maximum=65529, engine_mapping="--port <port>",
    ),
    Dial(
        "DesktopPython", "path", "(Join-Path $env:LOCALAPPDATA 'FreeToken\\venv\\Scripts\\python.exe')", "path", "Path to Python interpreter in the FreeToken Desktop virtual environment.",
        "Advanced", engine_mapping="Launcher interpreter",
    ),
    Dial(
        "CudaGraphMaxBS", "number", 4, "batch size", "Maximum batch size captured into CUDA graphs (-1 disables graph capture).",
        "Advanced", minimum=-1, maximum=1024, engine_mapping="--cuda-graph-max-bs <N>",
    ),
)

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


def canonical_value(dial: Dial, value: Any) -> Any:
    if dial.control == "toggle":
        enabled = _toggle_value(value, allow_text=dial.source == "env")
        return ("1" if enabled else "0") if dial.source == "env" else enabled
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


def validate_settings(settings: dict[str, Any]) -> list[dict[str, str]]:
    """Return the contract's field/message list without importing any engine package."""
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
        if dial.options is not None and parsed not in dial.options:
            choices = ", ".join(repr(item) for item in dial.options)
            errors.append({"field": name, "message": f"Value {parsed!r} for {name} must be one of {choices}"})
        if dial.minimum is not None and parsed < dial.minimum:
            errors.append({"field": name, "message": f"Value {parsed} below minimum {dial.minimum}"})
        if dial.maximum is not None and parsed > dial.maximum:
            errors.append({"field": name, "message": f"Value {parsed} exceeds maximum {dial.maximum}"})
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
    "PRIMARY_DIALS",
    "ENV_DIALS",
    "EXTENSION_DIALS",
    "Dial",
    "canonical_value",
    "dial_value_for_display",
    "normalise_settings",
    "normalize_settings",
    "validate_settings",
]

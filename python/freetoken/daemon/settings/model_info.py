"""Torch-free reader of a model folder's config.json for the settings helper.

The page sizes its sliders, limits and explanations from what the chosen model actually is:
longest chat (``max_position_embeddings``), how many layers hold routed experts, how many
experts each layer has, and how many bytes one expert occupies in the offload banks. It reads
``config.json`` (and, when present, FTW metadata) plus the small safetensors headers needed to
identify demand-paged PLE/n-gram tensors; weight payloads are never opened.

The per-expert byte formulas are copied from ``freetoken.moe.offload_cache._BANK_BYTES_PER_EXPERT``
(which cannot be imported here: it pulls in torch). Keep the two in step. Checked against the
shipping Qwen3.8-Flash-Next-NVFP4 build: nvfp4(2560, 640) = 2,772,480 bytes per expert, i.e. the
2.77 MB per slot, 2.58 GiB per 1,000 slots and 1.32 GiB per 512-expert layer quoted in
``docs/research/memory-audit-qwen38-rtx5090.md``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Mirrors the keys of freetoken.models.register.MODEL_REGISTRY; tests/settings/test_model_info.py
# checks the two lists agree by reading register.py as text (that module imports torch).
SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
    "DeepseekV4ForCausalLM",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4GGUFForCausalLM",
    "Gemma4UnifiedForCausalLM",
    "Gemma4UnifiedForConditionalGeneration",
    "Glm4MoeForCausalLM",
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
    "GlmMoeDsaForCausalLM",
    "GptOssForCausalLM",
    "LlamaForCausalLM",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
    "Mistral3ForConditionalGeneration",
    "MistralForCausalLM",
    "MuseGlimmerForConditionalGeneration",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen4ExpForConditionalGeneration",
)

# The model every measured number in dials.py was taken on. Other models get computed sizes
# and no speed claims.
REFERENCE_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
REFERENCE_MOE_LAYERS = 48
REFERENCE_EXPERTS = 512

GIB = 1024 ** 3

# Safetensors stores an 8-byte little-endian JSON-header length before the tensor
# metadata. Only that prefix and header are read; the weight payload is never touched.
_MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024
_PLE_TENSOR_MARKERS = ("ple", "ngram", "embed_ngram")


def ple_bytes_from_header(header: bytes | bytearray | memoryview) -> int:
    """Sum data ranges for PLE/n-gram tensors in a safetensors JSON header."""
    try:
        document = json.loads(bytes(header).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError):
        return 0
    if not isinstance(document, dict):
        return 0

    total = 0
    for name, tensor in document.items():
        if not isinstance(name, str) or not any(marker in name.lower() for marker in _PLE_TENSOR_MARKERS):
            continue
        if not isinstance(tensor, dict):
            continue
        offsets = tensor.get("data_offsets")
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
            continue
        try:
            start, end = int(offsets[0]), int(offsets[1])
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= start <= end:
            total += end - start
    return total


def safetensors_ple_bytes(path: str | os.PathLike[str]) -> int:
    """Read one safetensors header and return the PLE/n-gram payload bytes."""
    try:
        with Path(path).open("rb") as fh:
            raw_length = fh.read(8)
            if len(raw_length) != 8:
                return 0
            header_length = int.from_bytes(raw_length, "little", signed=False)
            if not 0 < header_length <= _MAX_SAFETENSORS_HEADER_BYTES:
                return 0
            header = fh.read(header_length)
            if len(header) != header_length:
                return 0
    except (OSError, ValueError, OverflowError):
        return 0
    return ple_bytes_from_header(header)


def _fp8_block_scale_pad(rows: int, cols: int) -> int:
    while (rows * cols * 2) % 16:
        cols += 1
    return cols


BYTES_PER_EXPERT = {
    "bf16": lambda H, I: 3 * I * H * 2,
    "fp8_block": lambda H, I: 3 * I * H + (
        (2 * I // 128) * _fp8_block_scale_pad(2 * I // 128, H // 128)
        + (H // 128) * _fp8_block_scale_pad(H // 128, I // 128)
    ) * 2,
    "q4_0": lambda H, I: 2 * I * (H // 32) * 18 + H * (I // 32) * 18,
    "nvfp4": lambda H, I: 2 * I * (H // 2 + H // 16 + 2) + H * (I // 2 + I // 16 + 2),
    "mxfp4": lambda H, I: 2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2),
    "ds_fp4": lambda H, I: 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32),
}

FORMAT_LABELS = {
    "bf16": "full size (16-bit)",
    "fp8_block": "8-bit",
    "q4_0": "4-bit (GGUF)",
    "nvfp4": "4-bit (NVFP4)",
    "mxfp4": "4-bit (MXFP4)",
    "ds_fp4": "4-bit (DeepSeek FP4)",
}


@dataclass
class ModelInfo:
    path: str
    name: str
    found: bool = False
    error: str = ""
    architecture: str = ""
    model_type: str = ""
    supported: bool = False
    is_reference: bool = False
    max_context_tokens: int | None = None
    num_layers: int | None = None
    num_moe_layers: int | None = None
    num_experts: int | None = None
    experts_per_token: int | None = None
    hidden_size: int | None = None
    moe_intermediate_size: int | None = None
    expert_format: str = ""
    expert_format_label: str = ""
    bytes_per_expert: int | None = None
    bytes_per_layer: int | None = None
    total_expert_bytes: int | None = None
    has_vision: bool = False
    has_ple: bool = False
    has_mtp: bool = False
    ple_bytes: int = 0
    weight_files: int = 0
    weight_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_moe(self) -> bool:
        return bool(self.num_moe_layers and self.num_experts)

    def as_dict(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["isMoe"] = self.is_moe
        # camelCase for the page, matching the rest of the settings payload
        return {_camel(key): value for key, value in doc.items()}


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part.capitalize() for part in rest)


def _int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def expert_format(config: dict[str, Any]) -> str:
    """The offload-bank format tag the engine would pick for the routed experts.

    Follows ``freetoken.models.config``: a ModelOpt/compressed-tensors NVFP4 export is
    ``nvfp4``; ``quant_method: fp8`` with a ``weight_block_size`` is ``fp8_block``;
    ``mxfp4`` is gpt-oss; anything else is stored bf16. Unknown schemes return "" so the
    caller skips the size estimate instead of guessing.
    """
    quant = config.get("quantization_config") or {}
    if not isinstance(quant, dict) or not quant:
        return "bf16"
    algo = str(quant.get("quant_algo") or "").upper()
    method = str(quant.get("quant_method") or "").lower()
    if algo == "NVFP4" or method == "nvfp4":
        return "nvfp4"
    if algo == "MXFP4" or method == "mxfp4":
        return "mxfp4"
    if method == "compressed-tensors":
        groups = quant.get("config_groups") or {}
        text = json.dumps(groups).lower()
        if "nvfp4" in text or ("float" in text and '"num_bits": 4' in text):
            return "nvfp4"
        if '"num_bits": 8' in text and "float" in text:
            return "fp8_block"
        return ""
    if algo == "FP8" or method == "fp8":
        return "fp8_block"
    if method in {"gguf", "q4_0"}:
        return "q4_0"
    return ""


def describe_config(
    config: dict[str, Any],
    name: str,
    path: str | os.PathLike[str] | None = None,
) -> ModelInfo:
    """Describe an already-read ``config.json`` without opening model weights.

    Previewing a Hub model has the same model-shaping rules as reading a local folder, but it
    does not have a folder for ``read_model`` to open.  Keeping the config-only part here makes
    those two paths use one set of formulas.
    """
    info = ModelInfo(path=str(path or ""), name=str(name or ""))
    if not isinstance(config, dict):
        info.error = "config.json is not a settings object."
        return info

    info.found = True
    architectures = config.get("architectures") or []
    info.architecture = str(architectures[0]) if architectures else ""
    info.supported = info.architecture in SUPPORTED_ARCHITECTURES
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    merged: dict[str, Any] = {**config, **text_config}
    info.model_type = str(merged.get("model_type") or "")
    info.max_context_tokens = _int(_first(merged, "max_position_embeddings", "max_sequence_length", "n_positions"))
    info.num_layers = _int(_first(merged, "num_hidden_layers", "n_layer"))
    experts = _int(_first(merged, "num_experts", "n_routed_experts", "num_local_experts"))
    first_dense = int(merged.get("first_k_dense_replace") or 0)
    if experts and info.num_layers:
        info.num_experts = experts
        info.num_moe_layers = max(0, info.num_layers - first_dense)
        info.experts_per_token = _int(_first(merged, "num_experts_per_tok", "num_experts_per_token", "moe_top_k"))
    info.hidden_size = _int(_first(merged, "hidden_size", "n_embd"))
    info.moe_intermediate_size = _int(_first(merged, "moe_intermediate_size", "intermediate_size"))
    info.has_vision = isinstance(config.get("vision_config"), dict)
    info.has_ple = bool(merged.get("ple_layer_ids"))
    info.has_mtp = bool(_int(merged.get("mtp_num_hidden_layers")) or _int(merged.get("num_nextn_predict_layers")))
    info.is_reference = (
        info.architecture == REFERENCE_ARCHITECTURE
        and info.num_moe_layers == REFERENCE_MOE_LAYERS
        and info.num_experts == REFERENCE_EXPERTS
    )

    if info.is_moe:
        fmt = expert_format(config)
        info.expert_format = fmt
        info.expert_format_label = FORMAT_LABELS.get(fmt, fmt or "unknown")
        formula = BYTES_PER_EXPERT.get(fmt)
        if formula and info.hidden_size and info.moe_intermediate_size:
            info.bytes_per_expert = int(formula(info.hidden_size, info.moe_intermediate_size))
            info.bytes_per_layer = info.bytes_per_expert * info.num_experts
            info.total_expert_bytes = info.bytes_per_layer * info.num_moe_layers
    return info


def read_model(path: str | os.PathLike[str] | None) -> ModelInfo:
    """Describe the model folder at ``path``. Never raises: an unreadable folder comes back
    with ``found=False`` and a plain ``error`` so the page can say what is wrong."""
    text = str(path or "").strip()
    folder = Path(os.path.expandvars(os.path.expanduser(text))) if text else None
    info = ModelInfo(path=text, name=folder.name if folder is not None else "")
    if folder is None:
        info.error = "No model folder is set."
        return info
    if text.startswith("$") or text.startswith("("):
        info.error = "The model folder is a script expression the page cannot open."
        return info
    if not folder.is_dir():
        info.error = "That folder does not exist."
        return info
    config_path = folder / "config.json"
    if not config_path.is_file():
        info.error = "No config.json in that folder, so it is not a model folder."
        return info
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError) as exc:
        info.error = f"config.json could not be read: {exc.__class__.__name__}."
        return info
    info = describe_config(config, folder.name, path=str(folder))
    if not info.found:
        return info

    try:
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.endswith(".safetensors"):
                info.weight_files += 1
                try:
                    info.weight_bytes += entry.stat().st_size
                except OSError:
                    pass
                info.ple_bytes += safetensors_ple_bytes(entry.path)
    except OSError:
        pass

    ftw = _ftw_bank_bytes(folder)
    if ftw:
        # exact bank bytes beat the estimate when the checkpoint is pre-packed (FTW)
        info.total_expert_bytes = ftw
        if info.num_moe_layers and info.num_experts:
            info.bytes_per_layer = ftw // info.num_moe_layers
            info.bytes_per_expert = info.bytes_per_layer // info.num_experts
    return info


def _ftw_bank_bytes(folder: Path) -> int | None:
    meta = folder / "freetoken_weight.json"
    if not meta.is_file():
        return None
    try:
        with meta.open("r", encoding="utf-8") as fh:
            tensors = json.load(fh).get("tensors", [])
    except (OSError, ValueError, AttributeError):
        return None
    total = sum(int(t.get("nbytes", 0)) for t in tensors if isinstance(t, dict) and t.get("kind") == "experts_bank")
    return total or None


def gib(value: int | float | None, digits: int = 2) -> str:
    """Plain GiB text for info strings ("1.32 GiB")."""
    if not value:
        return "0 GiB"
    return f"{value / GIB:.{digits}f} GiB"


__all__ = [
    "BYTES_PER_EXPERT",
    "FORMAT_LABELS",
    "GIB",
    "ModelInfo",
    "REFERENCE_ARCHITECTURE",
    "SUPPORTED_ARCHITECTURES",
    "describe_config",
    "expert_format",
    "gib",
    "ple_bytes_from_header",
    "read_model",
    "safetensors_ple_bytes",
]

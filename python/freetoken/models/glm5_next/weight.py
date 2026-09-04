"""Weight loading for GLM-5.3-Flash (``glm5_next``).

Supported checkpoints: NVFP4 exports of GLM-5.3-Flash in the multimodal-wrapper
layout (``model.language_model.*``) -- ModelOpt tensor kinds (LibertAIDAI) or
compressed-tensors kinds (RedHatAI) -- and the EXL3 export in the same layout,
selected by ``quantization_config``. Not supported: bf16-expert originals
(zai-org), text-only key layouts, TP > 1.

Routed experts go to the offload cache via the format-selected provider;
everything else loads bf16 with keys renamed ``model.language_model.X`` ->
``model.X``. ``model.visual.*`` and the trailing MTP layer are never read.

Load-time fusions (must mirror the module split orders):

* KDA ``in_proj``  = q|k|v|b|f_a|g_a projections concatenated on the output axis
* KDA ``conv1d``   = q|k|v depthwise conv weights concatenated on the channel axis

fp32-kept tensors: ``A_log`` / ``dt_bias``, the mHC ``hc_*`` tensors, the indexer
APE, and the router ``e_score_correction_bias``. Optional W8A16 fp8-at-load
follows ``ModelConfig.attn_quant`` / ``dense_quant`` / ``lm_head_quant``
(defaults and env opt-ins: see config.py).
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.glm_moe_dsa.weight import _ShardReader, _quant_fp8_per_row
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .args import Glm5NextArgs
from .config import parse_config

# Checkpoint prefix (multimodal wrapper) -> model prefix.
_CKPT = "model.language_model"
_MODEL = "model"


class _Exl3DirectShardReader:
    """Read EXL3 resident tensors by range from strict Windows unbuffered shards.

    The generic GLM reader keeps ``safe_open`` mappings for every shard until the final
    yielded tensor.  That is fine for ordinary checkpoints, but on Windows its touched
    pages join the file cache beside the 71.29 GiB of EXL3 banks.  The proof's resident
    matrices therefore use the same per-tensor ``DirectShard`` path as the expert loader;
    each small anonymous range buffer is released immediately after its CUDA copy.
    """

    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = device
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        from freetoken.models.weight import DirectShard

        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = DirectShard(
                os.path.join(self._folder, shard), whole=False, unbuffered_only=True
            )
            self._handles[shard] = handle
        tensor = handle.get_tensor(name)
        try:
            return tensor.to(device=self._device)
        finally:
            # DirectShard owns one anonymous range mmap per requested tensor.  The CUDA
            # copy above is final; do not let those ranges accumulate across the layer loop.
            del tensor
            alive = getattr(handle, "_alive", None)
            if alive is not None:
                alive.clear()

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._handles.clear()


# MTP-layer experts (layer == num_layers under the full checkpoint) map to None
# alongside the dense prefix; the bank loader skips them.
def _layer_to_bank(layer, config):
    return (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    )


# ModelOpt export (LibertAIDAI/GLM-5.3-Flash-NVFP4): weight | weight_scale |
# weight_scale_2 (dequant-side global).
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts",
)

# llm-compressor export (RedHatAI/GLM-5.3-Flash-NVFP4): weight_packed |
# weight_scale | weight_global_scale (quant-side global -> reciprocal at ingest).
# ``input_global_scale`` (the calibrated W4A4 activation scale) deliberately does
# not match: our routed-expert paths are W4A16 and never quantize activations.
_NVFP4_CT_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight_packed|weight_global_scale|weight_scale)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts (compressed-tensors)",
    kind_map={"weight_packed": "weight", "weight_global_scale": "weight_scale_2"},
    global_reciprocal=True,
)


def _select_expert_source_spec(model_path: str) -> Nvfp4ExpertSourceSpec:
    quant = getattr(cached_load_hf_config(model_path), "quantization_config", None) or {}
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    method = str(get("quant_method") or "").lower()
    return _NVFP4_CT_SOURCE_SPEC if method == "compressed-tensors" else _NVFP4_SOURCE_SPEC


def _read_linear(reader: _ShardReader, base: str) -> torch.Tensor:
    """Read one resident GLM linear in its checkpoint form.

    EXL3 stores a linear as three tensors plus a scalar codebook marker instead of
    ``<base>.weight``. Reconstructing here keeps the rest of the model loader and its
    optional W8A16 conversion unchanged; only one non-routed matrix is live at a time.
    The plain path remains for the older GLM-5.3 NVFP4 export and for non-quantized state.
    """
    weight_key = f"{base}.weight"
    if reader.has(weight_key):
        return reader.get(weight_key)

    marker_key = f"{base}.mul1"
    mcg_key = f"{base}.mcg"
    if reader.has(mcg_key):
        raise ValueError(f"{base}: EXL3 mcg codebooks are unsupported; expected mul1")
    required = tuple(f"{base}.{part}" for part in ("trellis", "suh", "svh", "mul1"))
    missing = [key.rsplit(".", 1)[1] for key in required if not reader.has(key)]
    if missing:
        raise KeyError(
            f"{base}: missing EXL3 tensor(s) {', '.join(missing)}; expected "
            ".trellis/.suh/.svh/.mul1 or .weight"
        )

    trellis = reader.get(f"{base}.trellis")
    suh = reader.get(f"{base}.suh")
    svh = reader.get(f"{base}.svh")
    marker = reader.get(marker_key)
    if trellis.dtype != torch.int16:
        raise ValueError(f"{base}.trellis must be int16, got {trellis.dtype}")
    if trellis.ndim != 3 or trellis.shape[-1] % 16:
        raise ValueError(
            f"{base}.trellis must have shape [in/16, out/16, 16*K], got "
            f"{tuple(trellis.shape)}"
        )
    if suh.dtype != torch.float16 or svh.dtype != torch.float16:
        raise ValueError(
            f"{base}: EXL3 suh/svh must be float16, got {suh.dtype}/{svh.dtype}"
        )
    if marker.dtype != torch.int32 or marker.ndim != 0:
        raise ValueError(
            f"{marker_key} must be a scalar int32 marker, got {marker.dtype} "
            f"{tuple(marker.shape)}"
        )
    k = trellis.shape[-1] // 16
    # B2 owns the EXL3 arithmetic and the card-only implementation. Keeping this
    # import local lets config/weight tests use a tiny seam stub without loading the
    # optional extension, while production always calls the shared implementation.
    from freetoken.kernel.exl3 import reconstruct

    return reconstruct(trellis, suh, svh, k=k, codebook="mul1")


# KDA in_proj fusion order; MUST match Glm5NextKDA._in_proj_split.
_KDA_IN_PROJ = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")


def load_nvfp4_expert_sources(model_path: str, config, layer_sink=None):
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _select_expert_source_spec(model_path),
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def _maybe_fp8(key: str, w: torch.Tensor, fp8: bool):
    if fp8:
        q, scale = _quant_fp8_per_row(w)
        yield f"{key}.weight", q
        yield f"{key}.weight_scale", scale
    else:
        yield f"{key}.weight", w.to(torch.bfloat16)


def _iter_kda_layer(reader, layer: int, attn_fp8: bool) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"

    # EXL3 already stores q|k|v as one qkv_proj matrix. The KDA module still
    # needs the six-way in_proj layout, so append the plain b|f_a|g_a slices.
    # Older GLM-5.3 exports carry separate q/k/v weights; retain that path.
    if reader.has(f"{src}.qkv_proj.weight") or reader.has(f"{src}.qkv_proj.trellis"):
        qkv = _read_linear(reader, f"{src}.qkv_proj").to(torch.bfloat16)
        bfg = torch.cat(
            [reader.get(f"{src}.{p}.weight").to(torch.bfloat16) for p in ("b_proj", "f_a_proj", "g_a_proj")],
            dim=0,
        )
        if attn_fp8:
            q, scale = _quant_fp8_per_row(qkv)
            yield f"{dst}.in_proj_qkv.weight", q
            yield f"{dst}.in_proj_qkv.weight_scale", scale
            yield f"{dst}.in_proj_bfg.weight", bfg
        else:
            yield f"{dst}.in_proj.weight", torch.cat((qkv, bfg), dim=0)
    elif attn_fp8:
        # fp8 resident: q|k|v (the 201 MB/layer read) as one W8A16 GEMM with
        # per-row scales; the small gate projections b|f_a|g_a stay bf16.
        qkv = torch.cat(
            [reader.get(f"{src}.{p}.weight").to(torch.bfloat16) for p in ("q_proj", "k_proj", "v_proj")],
            dim=0,
        )
        q, scale = _quant_fp8_per_row(qkv)
        yield f"{dst}.in_proj_qkv.weight", q
        yield f"{dst}.in_proj_qkv.weight_scale", scale
        bfg = torch.cat(
            [reader.get(f"{src}.{p}.weight").to(torch.bfloat16) for p in ("b_proj", "f_a_proj", "g_a_proj")],
            dim=0,
        )
        yield f"{dst}.in_proj_bfg.weight", bfg
    else:
        # One fused input GEMM: q|k|v|b|f_a|g_a (output-axis concat).
        fused = torch.cat(
            [reader.get(f"{src}.{p}.weight").to(torch.bfloat16) for p in _KDA_IN_PROJ], dim=0
        )
        yield f"{dst}.in_proj.weight", fused

    # The EXL3 checkpoint carries the already merged q|k|v depthwise convolution;
    # the older layout has three per-stream tensors and is concatenated here.
    if reader.has(f"{src}.conv1d.weight"):
        conv = reader.get(f"{src}.conv1d.weight").to(torch.bfloat16)
    else:
        conv = torch.cat(
            [reader.get(f"{src}.{p}_conv1d.weight").to(torch.bfloat16) for p in ("q", "k", "v")],
            dim=0,
        )
    yield f"{dst}.conv1d.weight", conv
    for p in ("f_b_proj", "g_b_proj"):
        yield f"{dst}.{p}.weight", reader.get(f"{src}.{p}.weight").to(torch.bfloat16)
    yield from _maybe_fp8(f"{dst}.o_proj", _read_linear(reader, f"{src}.o_proj"), attn_fp8)
    # Gate params stay fp32 (the recurrent kernels read them as fp32).
    yield f"{dst}.A_log", reader.get(f"{src}.A_log").to(torch.float32)
    yield f"{dst}.dt_bias", reader.get(f"{src}.dt_bias").to(torch.float32)
    yield f"{dst}.o_norm.weight", reader.get(f"{src}.o_norm.weight").to(torch.bfloat16)


def _iter_dsa_layer(reader, layer: int, attn_fp8: bool) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    fp8_projs = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj") if attn_fp8 else ()
    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        w = (
            reader.get(f"{src}.{proj}.weight")
            if proj == "kv_b_proj"
            else _read_linear(reader, f"{src}.{proj}")
        )
        yield from _maybe_fp8(f"{dst}.{proj}", w, proj in fp8_projs)
    for norm in ("q_a_layernorm", "kv_a_layernorm"):
        yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(torch.bfloat16)
    # kpool indexer (every DSA layer owns one). wq_b may be EXL3; wk and
    # weights_proj remain plain tensors. Kept bf16; the APE is fp32.
    yield f"{dst}.indexer.wq_b.weight", _read_linear(reader, f"{src}.indexer.wq_b").to(
        torch.bfloat16
    )
    for proj in ("wk", "weights_proj"):
        yield f"{dst}.indexer.{proj}.weight", reader.get(
            f"{src}.indexer.{proj}.weight"
        ).to(torch.bfloat16)
    for part, dtype in (
        ("k_norm.weight", torch.bfloat16),
        ("k_norm.bias", torch.bfloat16),
        ("index_kpool_compress_gate", torch.bfloat16),
        ("index_kpool_compress_ape", torch.float32),
    ):
        yield f"{dst}.indexer.{part}", reader.get(f"{src}.indexer.{part}").to(dtype)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.3 stores routed experts as NVFP4 and only supports the offload backend; "
        "experts are loaded into the offload cache via load_nvfp4_expert_sources()."
    )
    assert include_non_moe
    if get_tp_info().size > 1:
        # The loader emits full fused KDA/DSA tensors; TP sharding (per-head q|k|v|b
        # splits, replicated f_a|g_a, row-parallel o_proj) is not implemented yet --
        # same status as every other linear-hybrid / offload-family model in tree.
        raise NotImplementedError("glm5_next weight loading currently supports TP=1 only")
    config = parse_config(cached_load_hf_config(model_path))
    args: Glm5NextArgs = config.glm5_args
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    # The EXL3 proof cannot keep resident-weight safe_open mappings in Windows' file
    # cache while the pinned expert banks are being filled.  Range-read those resident
    # tensors too; Linux and older GLM exports retain the established reader.
    reader_type = (
        _Exl3DirectShardReader
        if os.name == "nt" and getattr(config, "expert_quant", None) == "exl3"
        else _ShardReader
    )
    reader = reader_type(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    attn_fp8 = config.attn_quant == "fp8_pertensor"
    mlp_fp8 = config.dense_quant == "fp8_pertensor"
    head_fp8 = config.lm_head_quant == "fp8_pertensor"
    if primary:
        from freetoken.utils import init_logger

        if reader_type is _Exl3DirectShardReader:
            init_logger(__name__).info(
                "GLM-5.3 EXL3 resident weights: Windows unbuffered range reads"
            )
        init_logger(__name__).info(
            f"GLM-5.3 resident quant: attn={config.attn_quant} dense={config.dense_quant} "
            f"lm_head={config.lm_head_quant} (FREETOKEN_GLM5_ATTN_FP8/FREETOKEN_GLM5_MLP_FP8; "
            "an FTW conversion records these choices implicitly -- serve with the same flags)"
        )
    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading GLM-5.3 dense weights",
            disable=not primary,
        ):
            src = f"{_CKPT}.layers.{layer}"
            dst = f"{_MODEL}.layers.{layer}"
            if args.is_kda_layer(layer):
                yield from _iter_kda_layer(reader, layer, attn_fp8)
            else:
                yield from _iter_dsa_layer(reader, layer, attn_fp8)

            # mHC mixing tensors, fp32 on every layer.
            for hc in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                       "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
                yield f"{dst}.{hc}", reader.get(f"{src}.{hc}").to(torch.float32)

            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(
                    torch.bfloat16
                )

            if layer < config.first_k_dense_replace:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _maybe_fp8(
                        f"{dst}.mlp.{proj}", _read_linear(reader, f"{src}.mlp.{proj}"), mlp_fp8
                    )
            else:
                yield f"{dst}.mlp.gate.weight", reader.get(f"{src}.mlp.gate.weight").to(
                    torch.bfloat16
                )
                yield (
                    f"{dst}.mlp.e_score_correction_bias",
                    # fp32 like HF's router math (the module declares fp32; a bf16
                    # cast would perturb top-8 selection on fp32-bias checkpoints).
                    reader.get(f"{src}.mlp.gate.e_score_correction_bias").to(torch.float32),
                )
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield from _maybe_fp8(
                        f"{dst}.mlp.shared_experts.{proj}",
                        _read_linear(reader, f"{src}.mlp.shared_experts.{proj}"),
                        mlp_fp8,
                    )

        yield f"{_MODEL}.embed_tokens.weight", reader.get(
            f"{_CKPT}.embed_tokens.weight"
        ).to(torch.bfloat16)
        yield f"{_MODEL}.norm.weight", reader.get(f"{_CKPT}.norm.weight").to(torch.bfloat16)
        head = _read_linear(reader, "lm_head")
        if head_fp8 and not config.tie_word_embeddings:
            q, scale = _quant_fp8_per_row(head)
            yield "lm_head.weight", q
            yield "lm_head.weight_scale", scale
        else:
            yield "lm_head.weight", head.to(torch.bfloat16)
    finally:
        reader.close()


__all__ = ["iter_weights", "load_nvfp4_expert_sources"]

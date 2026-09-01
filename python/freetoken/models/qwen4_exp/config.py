from __future__ import annotations

import os
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Mapping, Tuple

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SlotStateSpec,
    vision_load_enabled,
)


@dataclass(frozen=True)
class Qwen4VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_position_embeddings: int
    out_hidden_size: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    in_channels: int
    hidden_act: str
    deepstack_visual_indexes: Tuple[int, ...]


@dataclass(frozen=True)
class Qwen4ExpArgs:
    """Qwen3.8-Flash-Next geometry beyond the generic ModelConfig fields (ModelConfig.qwen4_args)."""

    hidden_size: int
    # Hyper-connections: every layer reads/writes hc_count residual streams [T, hc_count*hidden].
    hc_count: int
    hc_lowrank: int
    # PLE n-gram embedding; layer ids are zero-based decoder layers.
    ple_layer_ids: Tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    # n-gram hash windows never cross this token (the eos id); they restart after it.
    ngram_boundary_token_id: int
    # Qwen multimodal rotary axis widths and layout.
    mrope_section: Tuple[int, int, int]
    mrope_interleaved: bool
    # QSA indexer scoring geometry (the slab/ratio geometry lives on the attention group).
    index_n_heads: int
    index_kv_heads: int
    index_head_dim: int
    index_budget: int
    index_ratio: int

    @property
    def index_topk_blocks(self) -> int:
        return self.index_budget // self.index_ratio

    @property
    def num_ngram_heads(self) -> int:
        # one head group per n-gram order 2..ngram_size (Qwen3.8: 8 x 2-gram + 8 x 3-gram)
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def ngram_head_dim(self) -> int:
        return self.ple_embed_dim // self.num_ngram_heads

    @property
    def ple_conv_dilation(self) -> int:
        # HF Qwen4ExpTextPLELayer sets the depthwise conv dilation to ngram_size
        return self.ngram_size

    @property
    def ple_conv_state_len(self) -> int:
        return (self.ple_conv_kernel_size - 1) * self.ple_conv_dilation

    @property
    def ple_state_width(self) -> int:
        return self.hc_count * self.hidden_size


PLE_CONV_STATE = "ple_conv"
PLE_NGRAM_STATE = "ple_ngram_ctx"


def ple_slot_states(args: Qwen4ExpArgs) -> Tuple[SlotStateSpec, ...]:
    """Per-request PLE state riding the linear-state slots (see LinearStatePool.slot_states)."""
    if not args.ple_layer_ids:
        return ()
    return (
        # dilated-conv left context; replicated (not TP-sharded), model dtype
        SlotStateSpec(
            name=PLE_CONV_STATE,
            shape=(args.ple_state_width, args.ple_conv_state_len),
            layer_ids=args.ple_layer_ids,
        ),
        # last ngram_size-1 token ids, shared by every PLE layer; eos = hash boundary
        SlotStateSpec(
            name=PLE_NGRAM_STATE,
            shape=(args.ngram_size - 1,),
            dtype=torch.int32,
            fill_value=float(args.ngram_boundary_token_id),
        ),
    )


_DENSE_QUANT_ENV = "FREETOKEN_DENSE_QUANT"
_DENSE_QUANT_MODES = ("int8",)


def resolve_dense_quant(environ: Mapping[str, str] | None = None) -> str:
    """Load-time quantization for the dense projections (``FREETOKEN_DENSE_QUANT``).

    The shipping NVFP4 build quantizes only the routed experts: attention, GDN, the
    hyper-connections, the shared expert and ``lm_head`` are all in the modelopt ignore list
    and arrive bf16, which is ~8 GB of weight read per decode token and 6.2 GiB resident.
    ``int8`` converts them at load to weight-only int8 with one scale per output channel --
    half the bytes, half the decode time on those GEMMs, and near-lossless (see
    ``kernel/triton/int8_linear``). Unset (the default) keeps every one of them bf16, byte
    for byte: this flag never changes what a weight the checkpoint ALREADY quantized does.

    An env flag rather than a serving option because it is a private weight layout, exactly
    like the draft head's ``FREETOKEN_MTP_SPEC_DRAFT_LMHEAD``.
    """
    env = os.environ if environ is None else environ
    mode = (env.get(_DENSE_QUANT_ENV, "") or "none").strip().lower() or "none"
    if mode == "none":
        return "none"
    if mode not in _DENSE_QUANT_MODES:
        raise ValueError(
            f"{_DENSE_QUANT_ENV} must be none|{'|'.join(_DENSE_QUANT_MODES)}, got {mode!r}"
        )
    return mode


def _quant_get(hf_config: Any):
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return None
    return quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))


def _ignored(patterns, module_name: str) -> bool:
    return any(fnmatch(module_name, pat) for pat in patterns)


def _parse_vision_config(hf_config: Any) -> Qwen4VisionConfig | None:
    vision = getattr(hf_config, "vision_config", None)
    if vision is None or not vision_load_enabled():
        return None
    return Qwen4VisionConfig(
        depth=int(vision.depth),
        hidden_size=int(vision.hidden_size),
        intermediate_size=int(vision.intermediate_size),
        num_heads=int(vision.num_heads),
        num_position_embeddings=int(vision.num_position_embeddings),
        out_hidden_size=int(vision.out_hidden_size),
        patch_size=int(vision.patch_size),
        spatial_merge_size=int(vision.spatial_merge_size),
        temporal_patch_size=int(vision.temporal_patch_size),
        in_channels=int(vision.in_channels),
        hidden_act=str(vision.hidden_act),
        deepstack_visual_indexes=tuple(
            int(i) for i in (getattr(vision, "deepstack_visual_indexes", None) or ())
        ),
    )


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        # HF Qwen4ExpTextConfig rewrites full_attention to qwen_sparse_attention in __post_init__.
        return [
            "full_attention" if t == "qwen_sparse_attention" else t for t in layer_types
        ]
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    # int(), not round(): HF configuration_qwen4_exp truncates head_dim * partial.
    rotary_dim = int(head_dim * partial)

    # The three-axis MRoPE wrapper owns mrope_section/interleaving. The scalar rotary
    # cache still receives only hashable scalar scaling parameters.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    get = _quant_get(hf_config)
    if get is None:
        expert_quant = attn_quant = dense_quant = lm_head_quant = "none"
    else:
        algo = str(get("quant_algo") or get("quant_method") or "").lower()
        block = get("weight_block_size")
        if algo == "fp8" and block:
            # Official FP8 build (DeepSeek-V3-style block-fp8): only the routed experts
            # are quantized (fp8-e4m3 weights + per-block weight_scale_inv); attention,
            # GDN, the shared expert, HC, PLE and lm_head stay bf16.
            bs = tuple(int(x) for x in block)
            assert bs == (128, 128), f"only 128x128 block-fp8 is supported, got {bs}"
            expert_quant = "fp8_block"
            attn_quant = dense_quant = lm_head_quant = "none"
        else:
            is_fp4 = "fp4" in algo
            ignore = list(get("ignore") or [])

            # The RadixArk NVFP4 build quantizes only the routed experts; attention/GDN,
            # the shared expert, HC, PLE and lm_head all sit in the modelopt ignore list
            # and stay bf16. Derive every flag from that list instead of assuming the split.
            def _quant(probe: str) -> str:
                return "nvfp4" if is_fp4 and not _ignored(ignore, probe) else "none"

            prefix = "model.language_model.layers.0"
            expert_quant = _quant(f"{prefix}.mlp.experts.0.gate_proj")
            dense_quant = _quant(f"{prefix}.mlp.shared_expert.gate_proj")
            attn_quant = _quant(f"{prefix}.self_attn.q_proj")
            lm_head_quant = _quant("lm_head")

    # FREETOKEN_DENSE_QUANT=int8 converts the projections this checkpoint left bf16. It only
    # ever upgrades a "none": a component the checkpoint already ships packed keeps its own
    # (cheaper, exact) format. A tied lm_head is the token embedding under another name --
    # quantizing it would either corrupt the embedding or duplicate 1.27 GB -- so it is left
    # alone; on this checkpoint the head is untied.
    tie_word_embeddings = bool(getattr(text, "tie_word_embeddings", False))
    dense_override = resolve_dense_quant()
    if dense_override != "none":
        attn_quant = dense_override if attn_quant == "none" else attn_quant
        dense_quant = dense_override if dense_quant == "none" else dense_quant
        if lm_head_quant == "none" and not tie_word_embeddings:
            lm_head_quant = dense_override

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    # HF stores ple_layer_ids one-indexed (validated upstream as [1, num_layers]).
    ple_layer_ids = tuple(int(i) - 1 for i in (getattr(text, "ple_layer_ids", None) or ()))
    for lid in ple_layer_ids:
        if layer_types[lid] != "linear_attention":
            raise ValueError(f"PLE must sit on a linear_attention layer, got layer {lid}")

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
        index_head_dim=int(text.indexer_head_dim),
        num_index_layers=len(full_ids),
        index_ratio=int(text.indexer_compress_ratio),
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        # HF resolves a null output_gate_type to hidden_act; mirror that instead of
        # stringifying None.
        output_gate=str(getattr(text, "output_gate_type", None) or text.hidden_act),
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    num_experts = int(getattr(text, "num_experts", 0) or 0)

    # HF accepts int | list here and uses the first entry (modeling_qwen4_exp Qwen4ExpTextNGramEmbedding)
    eos_token_id = text.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]

    mrope_half = rotary_dim // 2
    mrope_section = rope_params.get("mrope_section") or (
        (mrope_half + 2) // 3,
        (mrope_half + 1) // 3,
        mrope_half // 3,
    )
    qwen4_args = Qwen4ExpArgs(
        hidden_size=text.hidden_size,
        hc_count=int(text.hc_count),
        hc_lowrank=int(text.hc_lowrank),
        ple_layer_ids=ple_layer_ids,
        ple_embed_dim=int(text.ple_embed_dim),
        ple_conv_kernel_size=int(text.ple_conv_kernel_size),
        ngram_size=int(text.ngram_size),
        heads_per_ngram=int(text.heads_per_ngram),
        ngram_vocab_size_base=int(text.ngram_vocab_size_base),
        make_ngram_vocab_size_divisible_by=int(text.make_ngram_vocab_size_divisible_by),
        split_ngram_parts=int(text.split_ngram_parts),
        ngram_boundary_token_id=int(eos_token_id),
        mrope_section=tuple(int(value) for value in mrope_section),
        mrope_interleaved=bool(rope_params.get("mrope_interleaved", True)),
        index_n_heads=int(text.indexer_n_heads),
        index_kv_heads=int(text.indexer_kv_heads),
        index_head_dim=int(text.indexer_head_dim),
        index_budget=int(text.indexer_budget),
        index_ratio=int(text.indexer_compress_ratio),
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0) or 0,
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=tie_word_embeddings,
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=int(getattr(text, "num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(getattr(text, "moe_intermediate_size", 0) or 0),
        shared_expert_intermediate_size=int(
            getattr(text, "shared_expert_intermediate_size", 0) or 0
        ),
        # Absent from the shipped config; HF Qwen4ExpTextConfig defaults it True and the
        # Qwen3_5MoE block renormalizes unconditionally -- keep the two in agreement.
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", True)),
        moe_enabled=num_experts > 0,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen4_exp"),
        architectures=getattr(hf_config, "architectures", ["Qwen4ExpForConditionalGeneration"]),
        vision_config=_parse_vision_config(hf_config),
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        attn_quant=attn_quant,
        dense_quant=dense_quant,
        lm_head_quant=lm_head_quant,
        qwen4_args=qwen4_args,
        slot_states=ple_slot_states(qwen4_args),
    )


__all__ = [
    "PLE_CONV_STATE",
    "PLE_NGRAM_STATE",
    "Qwen4ExpArgs",
    "Qwen4VisionConfig",
    "parse_config",
    "ple_slot_states",
    "resolve_dense_quant",
]

"""Meta-only memory planner used by the settings helper subprocess.

This module is intentionally a child boundary.  It may import torch and the engine, but it must
never construct :class:`Engine`, read weight payloads, allocate CUDA tensors, or write anything
other than one bounded JSON result on stdout.  Loggers and diagnostics belong on stderr.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


PROTOCOL_VERSION = 1
_HOST_RUNTIME_ALLOWANCE = 1 << 30


def _error(code: str, message: str) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "status": "unavailable",
        "fits": None,
        "fits_now": None,
        "fits_empty": None,
        "suggestion": None,
        "error": {"code": code, "message": str(message)[:512]},
    }


def _safe_error_message(message: Any, *, settings: Mapping[str, Any] | None = None) -> str:
    text = str(message or "")
    secrets = [os.environ.get("FREETOKEN_MTP_PRIVATE_ROOT", "")]
    if settings is not None:
        secrets.append(str(settings.get("ModelPath", "")))
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    if "FREETOKEN_MTP_PRIVATE_ROOT" in text:
        text = text.split("FREETOKEN_MTP_PRIVATE_ROOT", 1)[0].rstrip(" ,;:")
    return text[:512] or "planner child failed without a diagnostic"


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _ceil_div(value: int, divisor: int) -> int:
    from freetoken.utils import div_ceil

    return div_ceil(value, divisor)


def walk_tensor_storage(
    root: Any,
    *,
    device_for_key: Callable[[str], Any] | None = None,
) -> dict[str, int]:
    """Walk all tensor attributes, including private buffers, with identity de-duplication.

    ``BaseOP.state_dict`` intentionally omits private tensors, and quantized linear layers keep
    their private scales there.  This walker is therefore over the object graph rather than only
    the state dict.  A callback receives the dotted object key and returns the intended resident
    device; tests can use a simple string callback without importing any model.
    """
    callback = device_for_key or (lambda _key: "cuda")
    result: dict[str, int] = {}
    seen_objects: set[int] = set()
    seen_tensors: set[int] = set()

    def visit(value: Any, key: str) -> None:
        if isinstance(value, __import__("torch").Tensor):
            identity = id(value)
            if identity in seen_tensors:
                return
            seen_tensors.add(identity)
            target = callback(key)
            if hasattr(target, "type"):
                target = target.type
            target = str(target)
            result[target] = result.get(target, 0) + int(value.numel()) * int(value.element_size())
            return
        if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool)):
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            identity = id(value)
            if identity in seen_objects:
                return
            seen_objects.add(identity)
            for index, item in enumerate(value):
                visit(item, f"{key}.{index}" if key else str(index))
            return
        if isinstance(value, dict):
            identity = id(value)
            if identity in seen_objects:
                return
            seen_objects.add(identity)
            for name, item in value.items():
                child = f"{key}.{name}" if key else str(name)
                visit(item, child)
            return
        if hasattr(value, "__dict__"):
            identity = id(value)
            if identity in seen_objects:
                return
            seen_objects.add(identity)
            for name, item in vars(value).items():
                child = f"{key}.{name}" if key else name
                visit(item, child)

    visit(root, "")
    return result


def solve_cache_geometry(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    quant_format: str,
) -> tuple[int, int, bool]:
    """Call the engine's exact automatic slot/page policy."""
    from freetoken.engine.cache_budget import resolve_moe_cache_auto

    return resolve_moe_cache_auto(
        baseline_free=baseline_free,
        weights_bytes=weights_bytes,
        memory_ratio=memory_ratio,
        cache_per_page=cache_per_page,
        fixed_cache_size=fixed_cache_size,
        per_expert_bytes=per_expert_bytes,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_tokens=kv_reserve_tokens,
        page_size=page_size,
        quant_format=quant_format,
    )


@contextlib.contextmanager
def _temporary_environment(environment: Mapping[str, str]):
    old = os.environ.copy()
    os.environ.clear()
    os.environ.update({str(k): str(v) for k, v in environment.items()})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def _parse_config(argv: list[str], environment: Mapping[str, str]):
    """Parse and normalize one serving launch without entering server/launch.py."""
    import torch
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.engine.engine import _adjust_config
    from freetoken.layers import set_rope_device
    from freetoken.server.args import parse_args

    with _temporary_environment(environment):
        parsed, _ = parse_args(list(argv), prog="freetoken-memory-plan")
        requested_tp = (0, int(parsed.tp_info.size))
        current_tp = try_get_tp_info()
        if current_tp is None:
            set_tp_info(rank=requested_tp[0], size=requested_tp[1])
        elif (int(current_tp.rank), int(current_tp.size)) != requested_tp:
            raise RuntimeError(
                "planner TP configuration changed within one child request"
            )
        # Rotary tables are shape-only in this process.  CPU is essential: using the default
        # device would initialize a CUDA context during an otherwise meta-only estimate.
        set_rope_device(torch.device("cpu"))
        _adjust_config(parsed)
    return parsed


def _meta_model(config):
    import torch
    from freetoken.models import create_model
    from freetoken.utils import torch_dtype

    with torch.device("meta"), torch_dtype(config.dtype):
        model = create_model(config.model_config)
    # Quantized dense operators convert their visible bf16 placeholders to their final packed
    # dtype and private scale tensors during this load.  The input is still meta-only.
    state = model.state_dict()
    model.load_state_dict(dict(state))
    return model


def _model_weight_bytes(model) -> dict[str, int]:
    import torch

    placement = getattr(model, "weight_device_for_key", None)

    def target(key: str):
        if callable(placement):
            try:
                value = placement(key, torch.device("cuda"))
                return getattr(value, "type", value)
            except Exception:
                # An unknown provider is handled by the caller as an unavailable plan rather than
                # silently treating an unplaceable tensor as zero bytes.
                raise
        return "cuda"

    result = walk_tensor_storage(model, device_for_key=target)
    if getattr(model, "_embed_host", False) and callable(getattr(model, "_load_host_embedding", None)):
        # Qwen's host-table hook rehomes this placed CPU tensor into exact-size mapped pinning.
        # It is already in the CPU total: retain its subset separately for quota/phase accounting.
        if target("model.embed_tokens.weight") != "cpu":
            raise ValueError("host embedding placement is not CPU")
        embedding = model.model.embed_tokens.weight
        result["pinned_cpu"] = int(embedding.numel()) * int(embedding.element_size())
    return result


def _has_routed_moe(config_or_model: Any) -> bool:
    """Whether this normalized configuration owns a routed expert cache.

    A few dense checkpoints share a ``*_moe`` model package while reporting zero experts.  The
    engine's ``is_moe`` marker alone is therefore not sufficient for the cache path; requiring
    both dimensions keeps dense estimates on the ordinary KV/pool path.
    """
    model_config = getattr(config_or_model, "model_config", config_or_model)
    num_layers = _int(getattr(model_config, "num_moe_layers", 0) or 0)
    num_experts = _int(getattr(model_config, "num_experts", 0) or 0)
    marker = getattr(model_config, "is_moe", None)
    if marker is None:
        marker = num_layers > 0 and num_experts > 0
    return bool(marker) and num_layers > 0 and num_experts > 0


def _expert_format(config) -> tuple[str, str]:
    model_config = config.model_config
    source = str(getattr(model_config, "expert_quant", "none") or "none")
    if source == "none":
        source = str(getattr(model_config, "moe_weight_format", None) or "bf16")
    if source == "mxfp4":
        return "mxfp4", "mxfp4_triton"
    if source == "nvfp4":
        backend = str(getattr(config, "nvfp4_backend", "triton") or "triton")
        return source, {
            "marlin": "nvfp4_marlin",
            "flashinfer": "nvfp4_b12x",
        }.get(backend, "nvfp4")
    return source, source


def _expert_slot_bytes(config) -> tuple[int, str, str, int]:
    model_config = config.model_config
    if not _has_routed_moe(config):
        # Dense models still need the KV/pool path, but have no expert rows or slot-cache schema.
        return 0, "none", "none", 0

    import torch
    from freetoken.engine.cache_budget import expert_bytes_per_slot
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT, _BANK_SCHEMAS

    source_format, runtime_format = _expert_format(config)
    formula = _BANK_BYTES_PER_EXPERT.get(source_format)
    if formula is None:
        raise ValueError(f"unsupported expert-bank geometry {source_format!r}")
    hidden = int(getattr(config.model_config, "hidden_size", 0) or 0)
    intermediate = int(getattr(config.model_config, "moe_intermediate_size", 0) or 0)
    tp = max(1, int(config.tp_info.size))
    if hidden <= 0 or intermediate <= 0:
        raise ValueError("expert hidden/intermediate dimensions are unavailable")
    if source_format == "mxfp4":
        from freetoken.models.gpt_oss.weight import local_mxfp4_intermediate_range

        # GPT-OSS shards whole 32-element blocks, including padding on every TP rank.
        # H=I=2880, TP=4 needs I_local=736, not 720 (74,944 extra bytes per slot).
        _, _, local_intermediate = local_mxfp4_intermediate_range(
            intermediate, rank=int(getattr(config.tp_info, "rank", 0)), world_size=tp
        )
    else:
        local_intermediate = _ceil_div(intermediate, tp)
    total = int(formula(hidden, local_intermediate))
    fixed_alpha = 0
    if source_format == "nvfp4" and runtime_format in {"nvfp4_marlin", "nvfp4_b12x"}:
        # The native per-row global banks are folded into one alpha per expert during repack;
        # they are not slot-cache rows. The resulting alpha vectors are fixed resident bytes.
        native_global = (2 * local_intermediate * 2 + hidden * 2)
        total -= native_global
        alpha_itemsize = 2 if runtime_format == "nvfp4_marlin" else 4
        fixed_alpha = (
            2
            * int(getattr(config.model_config, "num_moe_layers", 0) or 0)
            * int(getattr(config.model_config, "num_experts", 0) or 0)
            * alpha_itemsize
        )
    schema = _BANK_SCHEMAS.get(runtime_format) or _BANK_SCHEMAS.get(source_format)
    if not schema or total <= 0:
        raise ValueError(f"no bank schema for expert format {runtime_format!r}")
    # Feed the existing helper metadata-shaped rows. This deliberately exercises the same
    # per-slot summation as OffloadMoeCache; the rows contain no payload and live on meta.
    base, remainder = divmod(total, len(schema))
    sources = {}
    for index, name in enumerate(schema):
        row_bytes = base + (1 if index < remainder else 0)
        sources[name] = [torch.empty((1, row_bytes), dtype=torch.uint8, device="meta")]
    per_slot = int(expert_bytes_per_slot(sources))
    return per_slot, source_format, runtime_format, fixed_alpha


def _owned_layers(config) -> frozenset[int]:
    from freetoken.engine.engine import _validate_gpu_owned_layers

    count = int(getattr(config.model_config, "num_moe_layers", 0) or 0)
    if count <= 0 or not _has_routed_moe(config):
        return frozenset()
    # _adjust_config already validates a non-empty spec.  The second call makes this planner's
    # resolved IDs explicit and keeps learned ranking/auto semantics in engine.py.
    return _validate_gpu_owned_layers(config, count)


def _cpu_layers(config) -> frozenset[int]:
    from freetoken.engine.engine import _resolve_cpu_layers

    if not _has_routed_moe(config):
        return frozenset()
    return _resolve_cpu_layers(config, int(getattr(config.model_config, "num_moe_layers", 0) or 0))


def _pin_cap(environment: Mapping[str, str], reserved: int) -> int | None:
    from freetoken.engine.engine import _pin_budget_bytes

    # ``os.environ`` is a live mapping; copy it before the context manager clears the process
    # environment, otherwise the update would restore an empty mapping and hide the serving
    # launcher's pin-budget override.
    with _temporary_environment(dict(environment)):
        return _pin_budget_bytes(reserved)


def _linear_ladder_bytes(config) -> int:
    """Shape the MTP state-replay arenas without instantiating a real state pool."""
    if not getattr(config.spec_decode, "enabled", False):
        return 0
    group = config.model_config.linear_attention_group()
    if group is None:
        return 0
    import torch
    from freetoken.kvcache.linear_state_pool import (
        _linear_local_dims,
        _linear_pool_num_slots,
        ssm_state_dtype,
    )
    from freetoken.models.qwen4_exp.config import PLE_CONV_STATE, PLE_NGRAM_STATE

    if ssm_state_dtype() != torch.float32:
        raise ValueError(
            "SpecStateLadder needs an fp32 recurrent state; set FREETOKEN_MAMBA_SSM_DTYPE=float32"
        )

    n_layers, conv_dim, v_heads = _linear_local_dims(group, config.tp_info.size)
    width = int(config.spec_decode.batch_width)
    item = int(config.dtype.itemsize)
    total = n_layers * conv_dim * (int(group.conv_kernel_dim) - 1 + width) * item
    total += n_layers * width * conv_dim * item
    total += 2 * n_layers * width * v_heads * item
    # The ladder's snapshot is a borrowed LinearStatePool slot already included in state_pool_bytes.
    # These are its fixed metadata tensors; graph-capture output arenas remain covered by the
    # engine's post-cache graph reserve rather than guessed here.
    total += (width + 1) * 2 * 8  # _cu: (max_width + 1, 2) int64
    total += (_linear_pool_num_slots(config) + 1) * 4  # _slot_ids plus _graph_slot
    for spec in getattr(config.model_config, "slot_states", ()):
        element = int((spec.dtype or config.dtype).itemsize)
        if spec.name == PLE_CONV_STATE:
            # SpecStateLadder._ple_hist: one history per declared PLE layer.
            total += len(spec.layer_ids) * int(spec.shape[0]) * (int(spec.shape[1]) + width) * element
        elif spec.name == PLE_NGRAM_STATE:
            # SpecStateLadder._ngram_hist: one shared token-id history.
            total += (math.prod(spec.shape) + width) * element
        else:
            raise ValueError(f"SpecStateLadder cannot size slot state {spec.name!r}")
    return int(total)


def _module_state_bytes(module: Any) -> int:
    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        raise ValueError("vision component has no state_dict")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise ValueError("vision component state_dict is not a mapping")
    total = 0
    seen: set[int] = set()
    for tensor in state.values():
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if not callable(numel) or not callable(element_size):
            raise ValueError("vision component state_dict contains a non-tensor")
        total += int(numel()) * int(element_size())
    return int(total)


def _vision_stream_workspace(model: Any) -> tuple[int, list[dict[str, Any]]]:
    """Size the patch/block/merger workspaces that layer-stream actually materializes."""
    visual = getattr(model, "visual", None)
    blocks = getattr(visual, "blocks", None)
    op_list = getattr(blocks, "op_list", None)
    try:
        block = op_list[0] if op_list else None
    except (IndexError, TypeError):
        block = None
    modules = (
        ("vision patch workspace", getattr(visual, "patch_embed", None), "vision.patch_embed.state_dict"),
        ("vision block workspace", block, "vision.blocks.op_list[0].state_dict"),
        ("vision merger workspace", getattr(visual, "merger", None), "vision.merger.state_dict"),
    )
    components: list[dict[str, Any]] = []
    sizes: list[int] = []
    for name, component, source in modules:
        if component is None:
            raise ValueError(f"{name} is unavailable")
        size = _module_state_bytes(component)
        if size <= 0:
            raise ValueError(f"{name} has no shape metadata")
        sizes.append(size)
        components.append(
            {
                "name": name,
                "resource": "vram",
                "phase": "graphs",
                "bytes": size,
                "kind": "allowance",
                "source": f"{source}; runtime-only, not idle resident",
                "scenario": "both",
            }
        )
    return max(sizes), components


def _ple_and_vision_bytes(
    config,
    model_weight: Mapping[str, int],
    *,
    model: Any | None = None,
    metadata: _CheckpointMetadata | None = None,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Return pinned/reclaimable host table bytes, runtime vision workspace and components."""
    from freetoken.daemon.settings.model_info import read_model

    info = metadata.model_info if metadata is not None else read_model(config.model_path)
    components: list[dict[str, Any]] = []
    ple_bytes = int(info.ple_bytes or 0)
    backend = str(getattr(config, "ple_backend", "disk"))
    if ple_bytes and backend == "pinned":
        components.append({"name": "PLE table", "resource": "ram", "phase": "host_tables", "bytes": ple_bytes, "kind": "allocation", "source": "safetensors headers", "scenario": "both"})
        pinned = ple_bytes
    elif ple_bytes and backend == "mmap":
        args = getattr(config.model_config, "qwen4_args", None)
        head_dim = int(getattr(args, "ngram_head_dim", 0) or 0)
        rows = int(os.environ.get("FREETOKEN_PLE_ROW_CACHE", "1048576") or 0)
        pinned = max(0, rows) * max(0, head_dim)
        if pinned:
            components.append({"name": "PLE row cache", "resource": "ram", "phase": "host_tables", "bytes": pinned, "kind": "reclaimable", "source": "FREETOKEN_PLE_ROW_CACHE", "scenario": "both"})
    else:
        pinned = 0
    vision_workspace = 0
    if getattr(config.model_config, "is_multimodal", False) and os.environ.get("FREETOKEN_LOAD_VISION", "0") == "1":
        if os.environ.get("FREETOKEN_VISION_EXECUTION", "gpu") == "layer-stream":
            if model is None:
                raise ValueError("vision layer-stream shape metadata is unavailable")
            vision_workspace, vision_components = _vision_stream_workspace(model)
            components.extend(vision_components)
    embedding = int(model_weight.get("pinned_cpu", 0))
    if embedding:
        components.append(_component("pinned token embedding", "ram", "host_tables", embedding, "allocation", "model._load_host_embedding exact-size storage"))
    return pinned + embedding, vision_workspace, components


def _component(name: str, resource: str, phase: str, amount: int, kind: str, source: str, scenario: str = "both") -> dict[str, Any]:
    return {"name": name, "resource": resource, "phase": phase, "bytes": max(0, int(amount)), "kind": kind, "source": source, "scenario": scenario}


def _resource_row(capacity: int, need: int, resident: int, peak: int) -> dict[str, int]:
    return {
        "need_bytes": max(0, int(need)),
        "resident_bytes": max(0, int(resident)),
        "boot_peak_bytes": max(0, int(peak)),
        "shortfall_bytes": max(0, int(need) - int(capacity)),
    }


def _pinned_bank_need(
    bank_total: int,
    *,
    num_moe_layers: int,
    owned_layers: frozenset[int],
    cpu_layers: frozenset[int],
    split_residency: bool,
) -> int:
    """Return the expert-bank bytes that the loader must register with CUDA.

    The engine's split-residency path locks CPU-decoded layers with the OS and pins only the
    remaining banks.  ``bank_total`` already excludes GPU-owned layers; use a conservative
    proportional layer split because FTW metadata exposes the aggregate bank size rather than
    a second allocation for every layer.
    """
    bank_total = max(0, int(bank_total))
    if not split_residency or bank_total == 0:
        return bank_total
    streaming_layers = max(0, int(num_moe_layers) - len(owned_layers))
    locked_layers = len(set(cpu_layers) - set(owned_layers))
    if streaming_layers <= 0 or locked_layers <= 0:
        return bank_total
    locked_layers = min(locked_layers, streaming_layers)
    pinned_layers = streaming_layers - locked_layers
    return (bank_total * pinned_layers + streaming_layers - 1) // streaming_layers


def _placement_plan(
    config,
    *,
    bank_total: int,
    host_tables: int,
    owned: frozenset[int],
) -> tuple[frozenset[int], int | None, int, bool, bool]:
    """Mirror engine CPU-layer auto splitting and pin-quota accounting."""
    from freetoken.engine.engine import _auto_cpu_layers

    num_moe_layers = int(getattr(config.model_config, "num_moe_layers", 0) or 0)
    if not _has_routed_moe(config):
        return frozenset(), _pin_cap(os.environ, 0), host_tables, False, False
    cpu_layers = _cpu_layers(config)
    pin_cap = _pin_cap(os.environ, host_tables)
    backend = str(getattr(config, "moe_backend", "offload"))
    if (
        not cpu_layers
        and getattr(config, "moe_cpu_layers", None) is None
        and backend in ("offload", "hybrid")
        and pin_cap is not None
    ):
        cpu_layers = _auto_cpu_layers(
            config,
            num_moe_layers,
            reserved=host_tables,
            gpu_owned=len(owned),
            bank_bytes=bank_total,
        ) - owned

    split_residency = bool(cpu_layers) and backend in ("offload", "hybrid") and pin_cap is not None
    if backend == "cpu" and not split_residency and pin_cap is not None and bank_total > pin_cap:
        # The engine changes an over-quota all-CPU boot to locked residency for every layer.
        split_residency = True
    pin_need = _pinned_bank_need(
        bank_total,
        num_moe_layers=num_moe_layers,
        owned_layers=owned,
        cpu_layers=cpu_layers,
        split_residency=split_residency,
    )
    overlap = bool(getattr(config, "moe_prefill_overlap", True)) and not split_residency
    # Report whole-process pin need/cap, not only the residual quota offered to expert banks.
    return cpu_layers, _pin_cap(os.environ, 0), pin_need + host_tables, split_residency, overlap


def _pool_bytes_for_pages(pool_cls: type, config: Any, pages: int, cache_per_page: int, pool_fixed: int) -> int:
    """Price the exact pool allocation for the page count returned by its classmethod.

    Uniform pools use the affine ``page * per_page + fixed`` model. DSV4 has several coupled
    window/index/state tiers, so its reviewed cost model is consulted directly instead of applying
    the affine approximation to a dense-model estimate.
    """
    if pool_cls.__name__ == "DSV4PagedKVCache":
        from freetoken.kvcache.dsv4_cost_model import _dsv4_pool_sizes, dsv4_pool_bytes

        sizes = _dsv4_pool_sizes(config, int(pages) + 1)
        return int(dsv4_pool_bytes(sizes, config.model_config.dsv4_args, config.max_running_req + 1))
    return int(pages) * int(cache_per_page) + int(pool_fixed)


def _startup_pool_pages(
    config: Any,
    *,
    pool_cls: type,
    capacity: int,
    weights_gpu: int,
    fixed_pool: int,
    owned_bytes: int,
    lru_bytes: int,
    state_bytes: int,
) -> int:
    """Run the same startup budget and pool solver used immediately before pool creation."""
    from freetoken.engine.engine import _startup_kv_budget

    init_free = int(capacity)
    # At this point in Engine.__init__, weights and the MoE cache are resident, while the KV
    # pool's own fixed bytes are allocated by its constructor after this solve. Keep the ordering
    # identical: _startup_kv_budget sees the post-MoE free value, then state_pool_bytes is removed,
    # then the resolved pool class prices/solves its pages.
    new_free = init_free - int(weights_gpu) - int(fixed_pool) - int(owned_bytes) - int(lru_bytes)
    available = _startup_kv_budget(float(config.memory_ratio), init_free, new_free)
    available -= int(state_bytes)
    return int(pool_cls.solve_num_pages(config, available))


def _dense_scenario_geometry(
    config,
    *,
    capacity: int,
    weights_gpu: int,
    fixed_pool: int,
    state_bytes: int,
    post_reserve: int,
    scenario_name: str,
) -> dict[str, Any]:
    """Resolve the ordinary dense model path without invoking MoE cache policy helpers."""
    from freetoken.kvcache import resolve_pool_class

    model_config = config.model_config
    pool_cls = resolve_pool_class(model_config)
    cache_per_page, pool_fixed, page_tokens, min_reserve = pool_cls.kv_cost(config)
    geometry: dict[str, Any] | None = None
    pages = 0
    issues: list[dict[str, Any]] = []
    try:
        pages = _startup_pool_pages(
            config,
            pool_cls=pool_cls,
            capacity=capacity,
            weights_gpu=weights_gpu,
            fixed_pool=fixed_pool,
            owned_bytes=0,
            lru_bytes=0,
            state_bytes=state_bytes,
        )
        if pages <= 1:
            raise ValueError("not enough KV pages after dense model weights")
        geometry = {
            "total_slots": 0,
            "lru_slots": 0,
            "owned_layers": [],
            "num_pages": int(pages),
            "page_size": int(page_tokens),
            "usable_kv_tokens": int(max(0, pages - 1) * page_tokens),
            "prefill_overlap": False,
        }
    except (AssertionError, ValueError, RuntimeError) as exc:
        issues.append({"code": "geometry_failed", "scope": scenario_name, "message": str(exc)[:512]})

    kv_bytes = _pool_bytes_for_pages(pool_cls, config, pages, cache_per_page, pool_fixed) if geometry else 0
    resident = int(weights_gpu) + int(fixed_pool) + int(state_bytes) + int(kv_bytes) + int(post_reserve)
    boot_peak = resident + _HOST_RUNTIME_ALLOWANCE // 4
    need = max(resident, boot_peak)
    if geometry is None:
        minimum_pages = max(
            2,
            _ceil_div(
                max(_int(getattr(config, "kv_reserve_tokens", 0)), int(min_reserve)),
                int(page_tokens),
            )
            + 1,
        )
        minimum_kv = _pool_bytes_for_pages(pool_cls, config, minimum_pages, cache_per_page, pool_fixed)
        need = max(need, int(weights_gpu) + int(fixed_pool) + int(state_bytes) + minimum_kv + int(post_reserve))

    return {
        "geometry": geometry,
        "resources": {
            "vram": {
                "need": int(need),
                "resident": int(resident),
                "peak": int(boot_peak),
                "policy_need": 0,
            }
        },
        "components": [
            _component("model weights", "vram", "dense", weights_gpu, "allocation", "meta model after conversion"),
            *([_component("dense fixed pool", "vram", "slots", fixed_pool, "allocation", "model pool fixed bytes")] if fixed_pool else []),
            _component("KV cache", "vram", "kv", kv_bytes, "allocation", f"{pool_cls.__name__}.kv_cost"),
            _component("post-cache feature reserve", "vram", "graphs", post_reserve, "allowance", "cache_budget.auto_vram_reserve_bytes"),
        ],
        "issues": issues,
        "effective": {
            "page_size": int(page_tokens),
            "attention_backend": str(config.attention_backend),
            "cache_type": str(getattr(config, "cache_type", "radix")),
            "moe_backend": str(config.moe_backend),
            "ple_backend": str(getattr(config, "ple_backend", "disk")),
            "prefill_overlap": False,
            "memory_ratio": float(config.memory_ratio),
        },
    }


def _scenario_geometry(
    config,
    *,
    capacity: int,
    weights_gpu: int,
    fixed_pool: int,
    state_bytes: int,
    post_reserve: int,
    headroom: int,
    per_expert: int,
    runtime_format: str,
    owned: frozenset[int],
    prefill_overlap: bool,
    scenario_name: str,
) -> dict[str, Any]:
    """Resolve one current/empty geometry with the engine budget helpers."""
    from freetoken.engine.cache_budget import (
        check_explicit_moe_cache_fits,
        gpu_owned_reservation_bytes,
        lru_slots_after_owned_charge,
        net_cache_budget_bytes,
    )
    from freetoken.kvcache import resolve_pool_class

    model_config = config.model_config
    if not _has_routed_moe(config):
        return _dense_scenario_geometry(
            config,
            capacity=capacity,
            weights_gpu=weights_gpu,
            fixed_pool=fixed_pool,
            state_bytes=state_bytes,
            post_reserve=post_reserve,
            scenario_name=scenario_name,
        )
    pool_cls = resolve_pool_class(model_config)
    cache_per_page, pool_fixed, page_tokens, min_reserve = pool_cls.kv_cost(config)
    fixed = int(pool_fixed) + int(state_bytes) + int(fixed_pool)
    num_experts = int(getattr(model_config, "num_experts", 0) or 0)
    total_experts = max(0, int(getattr(model_config, "num_moe_layers", 0) or 0) - len(owned)) * num_experts
    owned_bytes = gpu_owned_reservation_bytes(len(owned), num_experts, per_expert)
    declared_reserve = int(getattr(config, "moe_vram_reserve_bytes", -1))
    from freetoken.engine.cache_budget import resolve_vram_reserve_bytes

    mtp_resident = bool(getattr(config.spec_decode, "enabled", False))
    if os.environ.get("FREETOKEN_MTP_SHADOW", "0") == "1" and os.environ.get("FREETOKEN_MTP_RESIDENT", "0") == "1":
        mtp_resident = True
    policy_reserve = resolve_vram_reserve_bytes(declared_reserve, mtp_resident=mtp_resident)
    fixed_with_post = fixed + owned_bytes + policy_reserve + int(headroom)
    requested_total = int(getattr(config, "moe_cache_size", 0) or 0)
    automatic = bool(getattr(config, "moe_cache_auto", False)) or requested_total <= 0
    issues: list[dict[str, Any]] = []
    geometry: dict[str, Any] | None = None
    lru_slots = 0
    pages = 0
    overlap = bool(prefill_overlap)
    policy_need = 0
    try:
        if automatic:
            slots, solved_pages, overlap = solve_cache_geometry(
                baseline_free=capacity,
                weights_bytes=weights_gpu,
                memory_ratio=float(config.memory_ratio),
                cache_per_page=cache_per_page,
                fixed_cache_size=fixed_with_post,
                per_expert_bytes=per_expert,
                num_experts=num_experts,
                total_experts=total_experts,
                prefill_overlap=overlap,
                kv_reserve_tokens=max(int(config.kv_reserve_tokens), int(min_reserve)),
                page_size=page_tokens,
                quant_format=runtime_format,
            )
            lru_slots = int(slots)
            # Engine commits the auto solver's page result as an override before constructing the
            # pool. Re-run the resolved pool class with that committed value so pool-specific
            # geometry remains the source of truth without spending a second residual budget.
            committed_pages = int(solved_pages)
            max_seq_len = _int(getattr(config, "max_seq_len", 0))
            if max_seq_len > 0:
                committed_pages = min(committed_pages, _ceil_div(max_seq_len, int(page_tokens)) + 1)
            runtime_config = copy.copy(config)
            object.__setattr__(
                runtime_config,
                "num_page_override",
                int(config.num_page_override)
                if config.num_page_override is not None
                else committed_pages,
            )
            pages = _startup_pool_pages(
                runtime_config,
                pool_cls=pool_cls,
                capacity=capacity,
                weights_gpu=weights_gpu,
                fixed_pool=fixed_pool,
                owned_bytes=owned_bytes,
                lru_bytes=lru_slots * int(per_expert),
                state_bytes=state_bytes,
            )
        else:
            floor = 2 * num_experts if overlap else num_experts
            lru_slots = lru_slots_after_owned_charge(
                moe_cache_size=requested_total,
                owned_layers=len(owned),
                num_experts=num_experts,
                floor=floor,
            )
            # The explicit pre-check is the same refusal used by Engine. Its reserve-token floor
            # remains a policy check; it must not replace the actual KV page allocation below.
            reserve_pages = _ceil_div(
                max(int(config.kv_reserve_tokens), int(min_reserve)), page_tokens
            ) + 1
            net = net_cache_budget_bytes(float(config.memory_ratio), capacity, weights_gpu, fixed)
            check_explicit_moe_cache_fits(
                moe_cache_size=lru_slots,
                per_expert_bytes=per_expert,
                budget_bytes=net - reserve_pages * cache_per_page,
                owned_layers=len(owned),
                num_experts=num_experts,
                reserved_bytes=policy_reserve + int(headroom),
                requested_total=requested_total,
            )
            pages = _startup_pool_pages(
                config,
                pool_cls=pool_cls,
                capacity=capacity,
                weights_gpu=weights_gpu,
                fixed_pool=fixed_pool,
                owned_bytes=owned_bytes,
                lru_bytes=lru_slots * int(per_expert),
                state_bytes=state_bytes,
            )
        if pages <= 1:
            raise ValueError("not enough KV pages after the expert cache")
        usable_tokens = max(0, (pages - 1) * page_tokens)
        geometry = {
            "total_slots": int(lru_slots + len(owned) * num_experts),
            "lru_slots": int(lru_slots),
            "owned_layers": sorted(int(i) for i in owned),
            "num_pages": int(pages),
            "page_size": int(page_tokens),
            "usable_kv_tokens": int(usable_tokens),
            "prefill_overlap": bool(overlap),
        }
        # Auto slots reserve their solved pages inside the cache policy budget. Explicit slots
        # check only the requested reserve floor; their later residual KV solve does not set
        # aside post-cache reserve/headroom again. Price that actual pool only in physical need.
        policy_pages = committed_pages if automatic else reserve_pages
        named = weights_gpu + fixed + owned_bytes + lru_slots * per_expert + policy_pages * cache_per_page + policy_reserve
        ratio = float(config.memory_ratio)
        policy_need = math.ceil((named + int(headroom)) / ratio) if ratio > 0 else named + int(headroom)
    except (AssertionError, ValueError, RuntimeError) as exc:
        issues.append({"code": "geometry_failed", "scope": scenario_name, "message": str(exc)[:512]})

    kv_bytes = _pool_bytes_for_pages(pool_cls, config, pages, cache_per_page, pool_fixed) if geometry else 0
    lru_bytes = lru_slots * int(per_expert)
    resident = weights_gpu + int(fixed_pool) + owned_bytes + lru_bytes + kv_bytes + int(state_bytes) + int(post_reserve)
    boot_peak = resident + _HOST_RUNTIME_ALLOWANCE // 4
    need = max(resident, boot_peak, policy_need)
    if geometry is None:
        # Keep a useful non-fit explanation even when the engine's auto solver rejected the
        # minimum geometry; never turn an arithmetic failure into a positive verdict.
        minimum_slots = max(num_experts, 2 * num_experts if overlap else num_experts)
        minimum_pages = _ceil_div(max(int(config.kv_reserve_tokens), int(min_reserve)), page_tokens) + 1
        minimum = weights_gpu + fixed_with_post + minimum_slots * per_expert + minimum_pages * cache_per_page
        need = max(need, minimum)
    return {
        "geometry": geometry,
        "resources": {
            "vram": {
                "need": int(need),
                "resident": int(resident),
                "peak": int(boot_peak),
                "policy_need": int(policy_need),
            }
        },
        "components": [
            _component("model weights", "vram", "dense", weights_gpu, "allocation", "meta model after conversion"),
            *([_component("NVFP4 repack alpha vectors", "vram", "slots", fixed_pool, "allocation", "nvfp4_marlin/b12x set_alphas")] if fixed_pool else []),
            _component("GPU-owned MoE layers", "vram", "slots", owned_bytes, "allocation", "cache_budget.gpu_owned_reservation_bytes"),
            _component("MoE LRU cache", "vram", "slots", lru_bytes, "allocation", "cache_budget.expert_bytes_per_slot"),
            _component("KV cache", "vram", "kv", kv_bytes, "allocation", f"{pool_cls.__name__}.kv_cost"),
            _component("post-cache feature reserve", "vram", "graphs", post_reserve, "allowance", "cache_budget.auto_vram_reserve_bytes"),
            _component("post-cache reserve policy", "vram", "graphs", policy_reserve, "policy", "cache_budget.resolve_vram_reserve_bytes"),
            _component("cache headroom", "vram", "graphs", int(headroom), "policy", "cache_budget.DEFAULT_MOE_CACHE_HEADROOM_BYTES"),
        ],
        "issues": issues,
        "effective": {
            "page_size": int(page_tokens),
            "attention_backend": str(config.attention_backend),
            "cache_type": str(getattr(config, "cache_type", "radix")),
            "moe_backend": str(config.moe_backend),
            "ple_backend": str(getattr(config, "ple_backend", "disk")),
            "prefill_overlap": bool(overlap),
            "memory_ratio": float(config.memory_ratio),
        },
    }


@dataclass(frozen=True)
class _CheckpointMetadata:
    model_info: Any
    ftw_total: int | None
    raw_bank_totals: tuple[int | None, ...]
    shard_sizes: tuple[int, ...]
    scattered: bool
    parallel_shard_sizes: tuple[int, ...]


def _checkpoint_metadata(config) -> _CheckpointMetadata:
    """Read invariant headers once; candidate ownership only selects a pre-sized bank total."""
    from freetoken.daemon.settings.model_info import read_model
    from freetoken.models.weight import experts_scattered
    from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

    model_path = str(getattr(config, "model_path", ""))
    info = read_model(model_path)
    ftw_total = ftw_bank_bytes(model_path)
    layers = int(getattr(config.model_config, "num_moe_layers", 0) or 0)
    raw_totals = tuple(bank_bytes_estimate(config.model_config, gpu_owned=count) for count in range(layers + 1)) if ftw_total is None else ()
    try:
        sizes = tuple(sorted((p.stat().st_size for p in Path(model_path).glob("*.safetensors")), reverse=True)) if ftw_total is None else ()
    except OSError:
        sizes = ()
    parallel_sizes = sizes
    index = Path(model_path) / "model.safetensors.index.json"
    # Root files price _host_ram_fits_parallel's policy, not the indexed reader's buffers.
    if ftw_total is None and (sizes or index.is_file()) and getattr(config.model_config, "expert_quant", None) == "nvfp4":
        from importlib import import_module
        from freetoken.models.nvfp4_banks import _bank_layer, _canon_kind
        from freetoken.models.register import get_model_spec
        from freetoken.models.weight import _ODIRECT_BLK

        module = import_module(get_model_spec(info.architecture).module + ".weight")
        source_spec = module._NVFP4_SOURCE_SPEC
        # Match nvfp4_banks' weight_info predicate: global scales are read serially,
        # while the bulk reader selects indexed shards in lexical filename order.
        with open(index, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        shards = set()
        for name, shard in weight_map.items():
            match = source_spec.key_pattern.match(name)
            if match is None or _bank_layer(source_spec, int(match.group("layer")), config.model_config) is None:
                continue
            if _canon_kind(source_spec, match.group("kind")) in {"weight", "weight_scale"}:
                shards.add(shard)
        # read_shard_direct allocates the entire file rounded to its direct-I/O block,
        # including any nonexpert tensors in a mixed shard (models/weight.py:45-52).
        parallel_sizes = tuple(
            ((Path(model_path, shard).stat().st_size + _ODIRECT_BLK - 1) // _ODIRECT_BLK) * _ODIRECT_BLK
            for shard in sorted(shards)
        )
    scattered = bool(experts_scattered(model_path)) if (sizes or parallel_sizes) and str(getattr(config, "expert_load", "auto")) == "auto" else False
    return _CheckpointMetadata(info, ftw_total, raw_totals, sizes, scattered, parallel_sizes)


def _loader_buffer_bytes(
    config,
    *,
    machine: Mapping[str, Any],
    metadata: _CheckpointMetadata | None = None,
) -> tuple[int, str]:
    """Price scenario-dependent loader policy against request-local immutable header metadata."""
    metadata = metadata if metadata is not None else _checkpoint_metadata(config)
    if metadata.ftw_total is not None:
        return 0, "FTW packed banks use their chunked reader"
    shard_sizes = metadata.shard_sizes
    if not shard_sizes and not metadata.parallel_shard_sizes:
        return 0, "no local safetensors shard metadata"

    requested = str(getattr(config, "expert_load", "auto") or "auto")
    parallel = requested == "parallel"
    if requested == "auto":
        parallel = metadata.scattered and (not shard_sizes or _int(machine.get("ram_free_bytes")) > sum(shard_sizes) + max(shard_sizes))
    if not parallel:
        return 0, "serial/reclaimable expert reader"
    from inspect import signature
    from freetoken.models.weight import iter_expert_tensors_parallel

    queued = max(1, signature(iter_expert_tensors_parallel).parameters["prefetch"].default)
    # At weight.py's q.get -> frombuffer transition, nvfp4_banks.py's caller still owns
    # the previous tensor: previous + active + queued + producer (five at default depth).
    return sum(sorted(metadata.parallel_shard_sizes, reverse=True)[:queued + 3]), f"parallel reader: previous + active + {queued} queued + producer shards"


def _physical_post_cache_reserve(config: Any) -> int:
    """Estimate feature allocations made after the cache, independent of policy headroom.

    ``moe_vram_reserve_bytes`` and ``moe_cache_headroom_bytes`` are budgeting controls, not
    allocations. Even a literal zero must not erase the graph/draft feature bytes from the physical
    estimate; use the existing auto composition as the reviewed initial allowance.
    """
    from freetoken.engine.cache_budget import auto_vram_reserve_bytes

    mtp_resident = bool(getattr(getattr(config, "spec_decode", None), "enabled", False))
    if os.environ.get("FREETOKEN_MTP_SHADOW", "0") == "1" and os.environ.get("FREETOKEN_MTP_RESIDENT", "0") == "1":
        mtp_resident = True
    return int(auto_vram_reserve_bytes(mtp_resident=mtp_resident))


def _evaluate_scenarios(
    configs: tuple[Any, Any],
    *,
    machine: Mapping[str, Any],
    model_bytes: Mapping[str, int],
    host_tables: int,
    vision_workspace: int,
    table_components: list[dict[str, Any]],
    per_expert: int,
    source_format: str,
    runtime_format: str,
    fixed_expert_bytes: int = 0,
    metadata: _CheckpointMetadata | None = None,
) -> dict[str, Any]:
    from freetoken.engine.cache_budget import DEFAULT_MOE_CACHE_HEADROOM_BYTES

    metadata = metadata if metadata is not None else _checkpoint_metadata(configs[0])

    gpu_weights = int(model_bytes.get("cuda", 0))
    cpu_weights = int(model_bytes.get("cpu", 0)) - int(model_bytes.get("pinned_cpu", 0))
    results: dict[str, Any] = {}
    all_components: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []
    for name, config, capacity_key in (
        ("now", configs[0], "vram_free_bytes"),
        ("empty", configs[1], "vram_total_bytes"),
    ):
        try:
            owned = _owned_layers(config)
            ftw_total = metadata.ftw_total
            if ftw_total is not None and owned:
                # The FTW reader cannot fill GPU-owned layers in place; the real engine rejects
                # this combination before allocating banks, so an estimate must not claim a fit.
                raise ValueError("GPU-owned layers are unsupported for FTW expert banks")
            bank_total = ftw_total if ftw_total is not None else metadata.raw_bank_totals[len(owned)]
            if bank_total is None and _has_routed_moe(config):
                raise ValueError("expert-bank provider did not expose metadata-only byte sizing")
            bank_total = int(bank_total or 0)
            cpu_layers, pin_cap, pin_need, _split_residency, overlap = _placement_plan(
                config,
                bank_total=bank_total,
                host_tables=host_tables,
                owned=owned,
            )
            loader_buffers, loader_source = _loader_buffer_bytes(config, machine=machine, metadata=metadata)
            # The config's bank estimate is the host residency after owned layers. Add the dense
            # CPU placements and tables; the explicit 1 GiB allowance is for conversion/runtime.
            host_resident = cpu_weights + bank_total + host_tables
            parking_windows = 2 * int(getattr(config, "kv_park_window_mib", 0) or 0) * (1 << 20) if getattr(config, "kv_park", "off") == "ssd" else 0
            host_resident += parking_windows
            host_peak = host_resident + loader_buffers + _HOST_RUNTIME_ALLOWANCE
            pin_shortfall = max(0, pin_need - pin_cap) if pin_cap is not None else 0
            headroom = int(getattr(config, "moe_cache_headroom_bytes", DEFAULT_MOE_CACHE_HEADROOM_BYTES) or 0)
            from freetoken.kvcache.linear_state_pool import state_pool_bytes

            state_bytes = int(state_pool_bytes(config))
            ladder_bytes = int(_linear_ladder_bytes(config))
            post_reserve = _physical_post_cache_reserve(config)
            scenario = _scenario_geometry(
                config,
                capacity=int(machine.get(capacity_key, 0)),
                weights_gpu=gpu_weights,
                fixed_pool=fixed_expert_bytes,
                state_bytes=state_bytes,
                # Layer-stream vision rows are runtime-only allowances; the shared graph reserve
                # already covers post-cache boot allocations and must not be charged with the
                # lazy picture workspace again.
                post_reserve=post_reserve,
                headroom=headroom,
                per_expert=per_expert,
                runtime_format=runtime_format,
                owned=owned,
                prefill_overlap=overlap,
                scenario_name=name,
            )
            vram = scenario["resources"]["vram"]
            # Engine allocates the ladder after KV/state pools. It occupies ready/peak memory,
            # but is not a sibling pool subtracted by _startup_kv_budget or the cache precheck.
            vram["resident"] += ladder_bytes
            vram["peak"] += ladder_bytes
            vram["need"] = max(vram["need"], vram["resident"], vram["peak"])
            ram_need = max(host_peak, host_resident, pin_need)
            ram_peak = host_peak
            ram_row = {"need": ram_need, "resident": host_resident, "peak": ram_peak}
            results[name] = {
                "geometry": scenario["geometry"],
                "vram": vram,
                "ram": ram_row,
                "pinning": {
                    "need_bytes": pin_need,
                    "cap_bytes": pin_cap,
                    "shortfall_bytes": pin_shortfall,
                    "cpu_layers": sorted(int(i) for i in cpu_layers),
                },
                "components": scenario["components"]
                + [
                    _component("GDN state pool", "vram", "kv", state_bytes, "allocation", "linear_state_pool"),
                    _component("MTP ladder arenas", "vram", "graphs", ladder_bytes, "allocation", "spec_state_ladder"),
                    _component("dense/vision host weights", "ram", "dense", cpu_weights, "allocation", "meta placement"),
                    _component("expert host banks", "ram", "host_banks", bank_total, "allocation", "expert_banks metadata"),
                    _component("expert loader buffers", "ram", "host_banks", loader_buffers, "allowance", loader_source),
                    _component("host runtime allowance", "ram", "host_banks", _HOST_RUNTIME_ALLOWANCE, "allowance", "uncalibrated J2 allowance"),
                ]
                + table_components,
                "issues": scenario["issues"]
                + ([{"code": "pin_quota", "scope": "both", "message": "Pinned host need exceeds the effective pin budget"}] if pin_shortfall else []),
                "effective": scenario["effective"] | {"pin_budget_bytes": pin_cap, "bank_cuda_alloc": os.environ.get("FREETOKEN_BANK_CUDA_ALLOC", "0") in {"1", "true"}},
            }
            all_components.extend(results[name]["components"])
            all_issues.extend(results[name]["issues"])
        except Exception as exc:  # one scenario's provider failure invalidates the whole plan
            raise ValueError(f"{name} scenario: {type(exc).__name__}: {exc}") from exc

    return {
        "results": results,
        "components": all_components,
        "issues": all_issues,
        "formats": {"expert_source": source_format, "expert_runtime": runtime_format},
    }


def _fit(result: Mapping[str, Any], scenario: str, capacity: Mapping[str, Any]) -> bool:
    row = result[scenario]
    if row["vram"]["need"] > _int(capacity.get("vram")):
        return False
    if row["ram"]["need"] > _int(capacity.get("ram")):
        return False
    if row["pinning"]["shortfall_bytes"] > 0:
        return False
    if row["geometry"] is None:
        return False
    return not row["issues"]


def _render_plan(
    request: Mapping[str, Any],
    machine: Mapping[str, Any],
    result: Mapping[str, Any],
    configs: tuple[Any, Any],
) -> dict[str, Any]:
    now_capacity = {"vram": machine.get("vram_free_bytes", 0), "ram": machine.get("ram_free_bytes", 0)}
    empty_capacity = {"vram": machine.get("vram_total_bytes", 0), "ram": machine.get("ram_total_bytes", 0)}
    rows: dict[str, Any] = {}
    geometry: dict[str, Any] = {}
    pinning = result["results"]["now"]["pinning"]
    for name, capacity in (("now", now_capacity), ("empty", empty_capacity)):
        raw = result["results"][name]
        rows[name] = {
            "ram": {"free_bytes": int(machine.get("ram_free_bytes", 0)), "total_bytes": int(machine.get("ram_total_bytes", 0)), **_resource_row(capacity["ram"], raw["ram"]["need"], raw["ram"]["resident"], raw["ram"]["peak"])},
            "vram": {"free_bytes": int(machine.get("vram_free_bytes", 0)), "total_bytes": int(machine.get("vram_total_bytes", 0)), **_resource_row(capacity["vram"], raw["vram"]["need"], raw["vram"]["resident"], raw["vram"]["peak"])},
        }
        geometry[name] = raw["geometry"]
    fits_now = _fit(result["results"], "now", now_capacity)
    fits_empty = _fit(result["results"], "empty", empty_capacity)
    effective = result["results"]["now"]["effective"]
    return {
        "version": PROTOCOL_VERSION,
        "status": "ok",
        "fits": fits_now,
        "fits_now": fits_now,
        "fits_empty": fits_empty,
        "sampled_at": None,
        "effective_settings": dict(request.get("settings") or {}),
        "effective": effective,
        "machine": {
            "ram_source": machine.get("ram_source", "/proc/meminfo:MemAvailable"),
            "gpu_uuid": machine.get("gpu_uuid"),
            "gpu_name": machine.get("gpu_name"),
            "vram_source": machine.get("vram_source", "nvidia-smi"),
            "cgroup_limited": bool(machine.get("cgroup_limited", False)),
        },
        "resources": {
            "ram": {
                "free_bytes": rows["now"]["ram"]["free_bytes"],
                "total_bytes": rows["now"]["ram"]["total_bytes"],
                "now": {key: rows["now"]["ram"][key] for key in ("need_bytes", "resident_bytes", "boot_peak_bytes", "shortfall_bytes")},
                "empty": {key: rows["empty"]["ram"][key] for key in ("need_bytes", "resident_bytes", "boot_peak_bytes", "shortfall_bytes")},
            },
            "vram": {
                "free_bytes": rows["now"]["vram"]["free_bytes"],
                "total_bytes": rows["now"]["vram"]["total_bytes"],
                "now": {key: rows["now"]["vram"][key] for key in ("need_bytes", "resident_bytes", "boot_peak_bytes", "shortfall_bytes")},
                "empty": {key: rows["empty"]["vram"][key] for key in ("need_bytes", "resident_bytes", "boot_peak_bytes", "shortfall_bytes")},
            },
        },
        "geometry": geometry,
        "pinning": pinning,
        "components": result["results"]["now"]["components"],
        "issues": result["issues"],
        "assumptions": [
            "Meta-converted tensor shapes are used; no checkpoint payloads are loaded.",
            "Host loading includes an uncalibrated 1 GiB runtime/conversion allowance.",
            "Post-cache graph/MTP reservation follows the engine policy constants.",
        ],
        "suggestion": None,
    }


def _candidate_config(
    base_config: Any,
    settings: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    launch_builder: Callable[..., Any] | None = None,
    config_parser: Callable[[list[str], Mapping[str, str]], Any] | None = None,
) -> Any:
    """Normalize a suggestion through the same launch and config path as a real boot."""
    if launch_builder is None:
        from freetoken.daemon.settings.linux_launch import build_launch

        launch_builder = build_launch
    if config_parser is None:
        config_parser = _parse_config
    env = dict(os.environ if environment is None else environment)
    try:
        plan = launch_builder(dict(settings), base_env=env)
        argv = list(getattr(plan, "argv", ()) or ())
        if len(argv) < 5:
            raise ValueError("launch normalization returned an incomplete command")
        plan_env = dict(getattr(plan, "env", None) or env)
        return config_parser([str(item) for item in argv[4:]], plan_env)
    except Exception:
        raise


def _suggestion(
    *,
    base_settings: dict[str, Any],
    base_result: Mapping[str, Any],
    base_config: Any,
    machine: Mapping[str, Any],
    environment: Mapping[str, str],
    model_bytes: Mapping[str, int],
    host_tables: int,
    vision_workspace: int,
    table_components: list[dict[str, Any]],
    per_expert: int,
    source_format: str,
    runtime_format: str,
    fixed_expert_bytes: int = 0,
    metadata: _CheckpointMetadata | None = None,
) -> dict[str, Any] | None:
    """Search the permitted memory controls using fresh lightweight configs.

    The meta model and shape totals are immutable for this request. Candidates change only the
    four memory controls and pass through catalogue and launch normalization before sizing.
    """
    from freetoken.daemon.settings.dials import DIAL_BY_NAME, adapt_dial, canonical_value, validate_settings
    from freetoken.daemon.settings.model_info import read_model

    model = metadata.model_info if metadata is not None else read_model(str(base_settings.get("ModelPath", "")))
    control_names = ("MoECacheSize", "GpuOwnedLayers", "ContextTokens", "KVCacheTokens")
    cache: dict[str, tuple[dict[str, Any], dict[str, Any]] | None] = {}

    def key_for(settings: Mapping[str, Any]) -> str:
        return json.dumps(
            {name: settings.get(name) for name in control_names},
            sort_keys=True,
            separators=(",", ":"),
        )

    def evaluate(
        settings: dict[str, Any], *, fresh: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        # Apply, estimate and Save must agree on the full snapshot: on the reference model,
        # the catalogue stores an owned count as auto:N, which can select different IDs than N.
        if validate_settings(settings, model):
            return None
        settings = {
            name: canonical_value(DIAL_BY_NAME[name], value, adapt_dial(DIAL_BY_NAME[name], model).get("storedAs"))
            for name, value in settings.items()
        }
        key = key_for(settings)
        if not fresh and key in cache:
            return cache[key]
        try:
            # Candidate settings are full canonical snapshots. Re-run the real launch builder so
            # inherited environment values (notably GPU-owned layers) cannot disappear, then parse
            # and adjust fresh configs for both scenarios.
            config = _candidate_config(base_config, settings, environment=environment)
            empty_config = _candidate_config(base_config, settings, environment=environment)
            evaluated = _evaluate_scenarios(
                (config, empty_config),
                machine=machine,
                model_bytes=model_bytes,
                host_tables=host_tables,
                vision_workspace=vision_workspace,
                table_components=table_components,
                per_expert=per_expert,
                source_format=source_format,
                runtime_format=runtime_format,
                fixed_expert_bytes=fixed_expert_bytes,
                metadata=metadata,
            )
            document = _render_plan(
                {"settings": settings}, machine, evaluated, (config, empty_config)
            )
            value = (dict(settings), document)
        except Exception:
            # A candidate that cannot pass the normal engine normalization is simply not a
            # suggestion; the original estimate remains a valid response.
            value = None
        cache[key] = value
        return value

    def fits(document: Mapping[str, Any], target: str) -> bool:
        return bool(document.get("fits_now" if target == "now" else "fits_empty"))

    # Use the exact resolved set from the engine rather than reparsing the spelling here.  In
    # particular, fractional specs (for example ``0.5``) and learned ``auto`` rankings must
    # charge suggestions for the same layers as the candidate config.
    original_owned = len(_owned_layers(base_config))
    model_config = base_config.model_config
    num_layers = int(getattr(model_config, "num_moe_layers", 0) or 0)
    num_experts = int(getattr(model_config, "num_experts", 0) or 0)
    total_experts = max(0, num_layers * num_experts)
    geometry = base_result.get("geometry") if isinstance(base_result.get("geometry"), Mapping) else {}
    now_geometry = geometry.get("now") if isinstance(geometry.get("now"), Mapping) else {}
    effective = base_result.get("effective") if isinstance(base_result.get("effective"), Mapping) else {}
    page_size = max(
        1,
        _int(
            now_geometry.get("page_size"),
            _int(effective.get("page_size"), _int(getattr(base_config, "page_size", 1), 1)),
        ),
    )
    requested_slots = _int(base_settings.get("MoECacheSize"))
    upper_slots = requested_slots if requested_slots > 0 else _int(now_geometry.get("total_slots"))
    upper_slots = min(total_experts, upper_slots or total_experts)
    base_overlap = bool(now_geometry.get("prefill_overlap", True))
    original_owned_spec = str(base_settings.get("GpuOwnedLayers") or "")

    def owned_spec(count: int) -> str:
        if count == original_owned:
            return original_owned_spec
        if count <= 0:
            return ""
        if original_owned_spec.strip().lower().startswith("auto"):
            return f"auto:{count}"
        return str(count)

    def slot_search(target: str, variant: dict[str, Any], count: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if num_experts <= 0:
            return evaluate(variant)
        charge = count * num_experts
        floors = [num_experts + charge]
        if base_overlap:
            floors.append(2 * num_experts + charge)
        floors = sorted({floor for floor in floors if floor <= upper_slots})
        if not floors:
            return None
        lower: int | None = None
        best: tuple[dict[str, Any], dict[str, Any]] | None = None
        for floor in floors:
            candidate = dict(variant)
            candidate["MoECacheSize"] = floor
            result = evaluate(candidate)
            if result is not None and fits(result[1], target):
                lower, best = floor, result
                break
        if lower is None or best is None:
            return None
        high = upper_slots + 1
        while lower + 1 < high:
            middle = (lower + high) // 2
            candidate = dict(variant)
            candidate["MoECacheSize"] = middle
            result = evaluate(candidate)
            if result is not None and fits(result[1], target):
                lower, best = middle, result
            else:
                high = middle
        # Recheck the returned boundary with a fresh candidate object.  Policy floors and the
        # Marlin cap can create a discontinuity at the edge of a valid interval.
        boundary = dict(variant)
        boundary["MoECacheSize"] = lower
        result = evaluate(boundary)
        return result if result is not None and fits(result[1], target) else best

    def owned_order() -> list[int]:
        maximum = max(0, num_layers - 1)
        return [
            *range(original_owned, -1, -1),
            *range(original_owned + 1, maximum + 1),
        ]

    def tier_search(target: str, *, context: int | None = None) -> tuple[dict[str, Any], dict[str, Any]] | None:
        variant = dict(base_settings)
        if context is not None:
            variant["ContextTokens"] = context
            page = page_size
            # --num-tokens is normalized after page-size resolution and must be page-aligned.
            variant["KVCacheTokens"] = _ceil_div(max(context + page, 2 * page), page) * page
        if num_experts <= 0:
            result = evaluate(variant)
            return result if result is not None and fits(result[1], target) else None
        for count in owned_order():
            owned_variant = dict(variant)
            owned_variant["GpuOwnedLayers"] = owned_spec(count)
            result = slot_search(target, owned_variant, count)
            if result is not None:
                return result
        return None

    for target in ("now", "empty"):
        if target == "empty" and fits(base_result, "empty"):
            return None
        result = tier_search(target)
        if result is None:
            current_context = _int(base_settings.get("ContextTokens"))
            page = page_size
            if current_context > 64:
                context = max(64, ((current_context // 2) // page) * page)
                while context < current_context:
                    result = tier_search(target, context=context)
                    if result is not None:
                        break
                    if context == 64:
                        break
                    context = max(64, ((context // 2) // page) * page)
        if result is not None:
            candidate_settings, _document = result
            # The search cache is only a speed aid. Rebuild and re-evaluate the winning full
            # snapshot once more so the returned patch is backed by the exact launch-normalized
            # candidate that Apply will submit, not a stale boundary document.
            final_result = evaluate(dict(candidate_settings), fresh=True)
            if final_result is None or not fits(final_result[1], target):
                continue
            candidate_settings, document = final_result
            patch = {
                name: candidate_settings[name]
                for name in control_names
                if candidate_settings.get(name) != base_settings.get(name)
            }
            if not patch:
                continue
            changes = []
            reasons = {
                "MoECacheSize": "Lower the GPU expert-slot total to leave room for the rest of the boot.",
                "GpuOwnedLayers": "Move fewer expert layers onto the card to reduce permanent VRAM use.",
                "ContextTokens": "Reserve less single-chat context memory.",
                "KVCacheTokens": "Use an explicit KV pool sized for the smaller context.",
            }
            for name in control_names:
                if name in patch:
                    changes.append(
                        {
                            "name": name,
                            "from": base_settings.get(name),
                            "to": patch[name],
                            "reason": reasons[name],
                        }
                    )
            return {
                "target": target,
                "settings": patch,
                "fits": bool(document.get("fits_now")),
                "fits_now": bool(document.get("fits_now")),
                "fits_empty": bool(document.get("fits_empty")),
                "changes": changes,
            }
    return None


def estimate_request(request: dict) -> dict:
    """Estimate one request object; all errors are returned in protocol form."""
    if not isinstance(request, dict) or request.get("version") != PROTOCOL_VERSION:
        return _error("planner_failed", "unsupported planner request")
    settings = request.get("settings")
    argv = request.get("argv")
    machine = request.get("machine")
    if not isinstance(settings, dict) or not isinstance(argv, list) or not isinstance(machine, Mapping):
        return _error("planner_failed", "planner request fields have the wrong shape")
    if len(json.dumps(request, separators=(",", ":"))) > 4 * 1024 * 1024:
        return _error("planner_failed", "planner request is too large")
    try:
        environment = dict(os.environ)
        config = _parse_config([str(item) for item in argv], environment)
        # A second fresh config is used for the empty-machine solve: _adjust_config and the auto
        # budget path mutate cache/page fields, so reusing one object would charge twice.
        empty_config = _parse_config([str(item) for item in argv], environment)
        metadata = _checkpoint_metadata(config)
        model = _meta_model(config)
        model_bytes = _model_weight_bytes(model)
        host_tables, vision_workspace, table_components = _ple_and_vision_bytes(
            config,
            model_bytes,
            model=model,
            metadata=metadata,
        )
        per_expert, source_format, runtime_format, fixed_expert_bytes = _expert_slot_bytes(config)
        evaluated = _evaluate_scenarios(
            (config, empty_config),
            machine=machine,
            model_bytes=model_bytes,
            host_tables=host_tables,
            vision_workspace=vision_workspace,
            table_components=table_components,
            per_expert=per_expert,
            source_format=source_format,
            runtime_format=runtime_format,
            fixed_expert_bytes=fixed_expert_bytes,
            metadata=metadata,
        )
        output = _render_plan(request, machine, evaluated, (config, empty_config))
        if not output["fits_now"]:
            output["suggestion"] = _suggestion(
                base_settings=settings,
                base_result=output,
                base_config=config,
                machine=machine,
                environment=environment,
                model_bytes=model_bytes,
                host_tables=host_tables,
                vision_workspace=vision_workspace,
                table_components=table_components,
                per_expert=per_expert,
                source_format=source_format,
                runtime_format=runtime_format,
                fixed_expert_bytes=fixed_expert_bytes,
                metadata=metadata,
            )
            if output["suggestion"] is None and output["fits_empty"]:
                output["issues"].append({
                    "code": "unchanged_empty_fit", "scope": "now",
                    "message": "The current settings already fit an empty machine; no setting change was found that fits right now.",
                })
        return output
    except FileNotFoundError as exc:
        return _error("unsupported_geometry", _safe_error_message(exc, settings=settings))
    except (ImportError, ModuleNotFoundError) as exc:
        return _error("planner_failed", f"planner provider import failed: {type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001 - child must always emit one protocol object
        message = _safe_error_message(f"{type(exc).__name__}: {exc}", settings=settings)
        return _error("planner_failed", message)


def main() -> int:
    """Read one bounded request and write exactly one JSON object to stdout."""
    raw = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        output = _error("planner_failed", "planner request is too large")
    else:
        try:
            request = json.loads(raw.decode("utf-8"))
            with contextlib.redirect_stdout(sys.stderr):
                output = estimate_request(request)
        except Exception as exc:  # malformed input is still a one-object response
            output = _error("planner_failed", f"malformed planner request: {type(exc).__name__}")
    sys.stdout.write(json.dumps(output, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["estimate_request", "main", "solve_cache_geometry", "walk_tensor_storage"]

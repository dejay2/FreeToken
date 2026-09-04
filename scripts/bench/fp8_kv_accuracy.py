from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch


_FP8_MAX = 448.0
_SCALE_FLOOR = 1.0e-12
_BLOCK_SCALE_SIZE = 64
_SCALE_DTYPE = torch.float32
_SCALE_DTYPE_NAME = "fp32"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _load_snapshot(path: Path) -> dict[str, Any]:
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(snapshot, dict):
        raise TypeError(f"snapshot must contain a dictionary, got {type(snapshot).__name__}")
    required = {
        "q",
        "k_cache",
        "v_cache",
        "indices",
        "block_table",
        "token_to_req",
        "seq_lens",
    }
    missing = sorted(required.difference(snapshot))
    if missing:
        raise ValueError(f"snapshot is missing required fields: {', '.join(missing)}")
    return snapshot


def _quantize_fp8(
    tensor: torch.Tensor,
    dtype: torch.dtype,
    scaling: str,
    scale_dtype: torch.dtype,
    active_token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if tensor.ndim != 4:
        raise ValueError(f"paged cache must be [pages, tokens, heads, dim], got {tuple(tensor.shape)}")
    values = tensor.to(torch.float32)
    if active_token_mask is None:
        active_tokens = torch.ones(values.shape[:2], dtype=torch.bool)
    else:
        active_tokens = active_token_mask.to(torch.bool)
        if active_tokens.ndim == 1 and active_tokens.numel() == values.shape[0]:
            # Backward-compatible page-only callers treat every token in an active page as
            # written; the checker itself always supplies the stricter token-level mask.
            active_tokens = active_tokens[:, None].expand(-1, values.shape[1])
        if tuple(active_tokens.shape) != tuple(values.shape[:2]):
            raise ValueError(
                "active token mask must match the cache's [pages, tokens] shape: "
                f"got {tuple(active_tokens.shape)} for {tuple(values.shape[:2])}"
            )
    active_view = active_tokens.unsqueeze(-1).unsqueeze(-1)
    active_abs = values.abs().masked_fill(~active_view, 0.0)

    if scaling == "per-layer":
        group_max = active_abs.amax().reshape(1, 1, 1, 1)
    elif scaling == "per-head":
        group_max = active_abs.amax(dim=(0, 1, 3), keepdim=True)
    elif scaling == "per-head-per-block-64":
        if values.shape[1] != _BLOCK_SCALE_SIZE:
            raise ValueError(
                "per-head-per-block-64 requires a 64-token cache page, "
                f"got {values.shape[1]}"
            )
        group_max = active_abs.amax(dim=(1, 3), keepdim=True)
    else:
        raise ValueError(f"unsupported scaling mode: {scaling}")

    # Keep the scale metadata at FP32, the one scale representation requested by P0.
    scales = (group_max / _FP8_MAX).clamp_min(_SCALE_FLOOR).to(scale_dtype).to(torch.float32)
    normalized = values / scales
    active_normalized = normalized.masked_select(active_view.expand_as(normalized))
    saturated = (active_normalized.abs() >= _FP8_MAX).sum().item() / max(
        active_normalized.numel(), 1
    )
    quantized = normalized.clamp(-_FP8_MAX, _FP8_MAX).to(dtype)
    dequantized = quantized.to(torch.float32) * scales
    return dequantized, scales, group_max, saturated


def _gather_paged(
    cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache.ndim != 4:
        raise ValueError(f"paged cache must be [pages, tokens, heads, dim], got {tuple(cache.shape)}")
    rows, width = logical_indices.shape
    if token_to_req.shape != (rows,):
        raise ValueError(
            f"token_to_req must have shape {(rows,)}, got {tuple(token_to_req.shape)}"
        )
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be [requests, pages], got {tuple(block_table.shape)}")
    if cache.shape[1] != page_size:
        raise ValueError(
            f"cache page size {cache.shape[1]} does not match snapshot page size {page_size}"
        )

    logical = logical_indices.to(torch.int64)
    request = token_to_req.to(torch.int64)
    table = block_table.to(torch.int64)
    page = torch.div(logical, page_size, rounding_mode="floor")
    offset = logical.remainder(page_size)
    valid = (
        (logical >= 0)
        & (request >= 0).unsqueeze(1)
        & (request < table.shape[0]).unsqueeze(1)
        & (page >= 0)
        & (page < table.shape[1])
    )
    safe_request = request.clamp(0, max(table.shape[0] - 1, 0))
    safe_page = page.clamp(0, max(table.shape[1] - 1, 0))
    physical = table[safe_request.unsqueeze(1), safe_page]
    valid &= (physical >= 0) & (physical < cache.shape[0])
    safe_physical = physical.clamp(0, max(cache.shape[0] - 1, 0))
    gathered = cache.to(torch.float32)[safe_physical, offset]
    return gathered, valid


def _replay_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 3:
        raise ValueError(f"q must be [rows, query_heads, head_dim], got {tuple(q.shape)}")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V cache shapes differ")
    keys, valid = _gather_paged(k_cache, indices, block_table, token_to_req, page_size)
    values, values_valid = _gather_paged(v_cache, indices, block_table, token_to_req, page_size)
    if not torch.equal(valid, values_valid):
        raise ValueError("K and V page mappings disagree")
    if q.shape[2] != keys.shape[-1]:
        raise ValueError(f"query head width {q.shape[2]} differs from cache width {keys.shape[-1]}")
    if q.shape[1] % keys.shape[2]:
        raise ValueError("query heads are not evenly grouped over KV heads")

    query = q.to(torch.float32)
    rows, query_heads, head_dim = query.shape
    kv_heads = keys.shape[2]
    group_size = query_heads // kv_heads
    result = torch.zeros_like(query)
    scale = 1.0 / math.sqrt(head_dim)
    for row in range(rows):
        row_valid = valid[row]
        for head in range(query_heads):
            kv_head = head // group_size
            scores = torch.matmul(keys[row, :, kv_head], query[row, head]) * scale
            scores = scores.masked_fill(~row_valid, float("-inf"))
            if bool(row_valid.any()):
                weights = torch.softmax(scores, dim=0)
                result[row, head] = torch.sum(
                    weights.unsqueeze(1) * values[row, :, kv_head], dim=0
                )
    return result, valid


def _selected_agreement(
    bf16_indices: torch.Tensor, fp8_indices: torch.Tensor
) -> tuple[int, int, float]:
    if bf16_indices.shape != fp8_indices.shape:
        raise ValueError("BF16 and FP8 index shapes differ")
    selected = 0
    intersection = 0
    for bf16_row, fp8_row in zip(bf16_indices.tolist(), fp8_indices.tolist()):
        bf16_tokens = {int(token) for token in bf16_row if int(token) >= 0}
        fp8_tokens = {int(token) for token in fp8_row if int(token) >= 0}
        selected += len(bf16_tokens)
        intersection += len(bf16_tokens.intersection(fp8_tokens))
    agreement = intersection / max(selected, 1)
    return selected, intersection, agreement


def _agreement_by_query_block(
    bf16_indices: torch.Tensor, fp8_indices: torch.Tensor
) -> list[dict[str, Any]]:
    """Compare each query row's selected-token set without reusing either K tensor."""
    if bf16_indices.shape != fp8_indices.shape:
        raise ValueError("BF16 and FP8 index shapes differ")
    rows: list[dict[str, Any]] = []
    for query_block, (bf16_row, fp8_row) in enumerate(
        zip(bf16_indices.tolist(), fp8_indices.tolist())
    ):
        bf16_tokens = {int(token) for token in bf16_row if int(token) >= 0}
        fp8_tokens = {int(token) for token in fp8_row if int(token) >= 0}
        selected = len(bf16_tokens)
        intersection = len(bf16_tokens.intersection(fp8_tokens))
        rows.append(
            {
                "query_block": query_block,
                "selected_tokens": selected,
                "selected_intersection": intersection,
                "selected_agreement_pct": f"{100.0 * intersection / max(selected, 1):.9g}",
            }
        )
    return rows


def _query_positions(snapshot: dict[str, Any], rows: int) -> torch.Tensor:
    positions = snapshot.get("query_positions", snapshot.get("positions"))
    if positions is None:
        final_position = snapshot.get("final_position")
        if final_position is None:
            raise ValueError(
                "snapshot must contain query_positions/positions or final_position for selection"
            )
        positions = torch.full((rows,), int(final_position), dtype=torch.int64)
    elif not isinstance(positions, torch.Tensor):
        positions = torch.as_tensor(positions, dtype=torch.int64)
    else:
        positions = positions.to(torch.int64).reshape(-1)
    positions = positions.reshape(-1)
    if positions.numel() == 1 and rows != 1:
        positions = positions.expand(rows)
    if positions.numel() != rows:
        raise ValueError(f"query positions have {positions.numel()} entries for {rows} rows")
    return positions


def _select_main_k_indices(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_positions: torch.Tensor,
    page_size: int,
    index_ratio: int,
    block_topk: int,
    token_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the P0 score/top-k/expand path using the captured main-K slabs.

    The original capture predates the optional compressed-index field, so it contains the
    production main K but not the separate index slab or indexer query.  In that case this is a
    truthful independent fallback: mean each complete ``index_ratio``-token K block, apply the
    same ReLU dot-product block score, top-k, and causal-tail expansion used by QSA.  When a
    future capture includes the compressed index, a dedicated capture can replace this fallback
    without changing the agreement contract.
    """
    if q.ndim != 3 or k_cache.ndim != 4:
        raise ValueError("selection replay expects q=[rows, heads, dim] and paged K")
    if k_cache.shape[1] != page_size:
        raise ValueError("selection page size does not match K cache")
    if page_size % index_ratio:
        raise ValueError("page size must be divisible by index ratio")
    if block_topk <= 0 or token_topk <= 0 or token_topk % index_ratio:
        raise ValueError("selection top-k values must be positive and block-aligned")
    if q.shape[2] != k_cache.shape[3]:
        raise ValueError("selection query and K head widths differ")
    if q.shape[1] % k_cache.shape[2]:
        raise ValueError("selection query heads are not evenly grouped over KV heads")
    if block_table.ndim != 2:
        raise ValueError("selection block table must be [requests, pages]")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("selection token-to-request mapping does not match q rows")

    query = q.to(torch.float32)
    keys = k_cache.to(torch.float32)
    table = block_table.to(torch.int64)
    requests = token_to_req.to(torch.int64)
    lengths = sequence_lengths.to(torch.int64).reshape(-1)
    positions = query_positions.to(torch.int64).reshape(-1)
    output_width = token_topk + index_ratio - 1
    selected_indices = torch.full(
        (q.shape[0], output_width), -1, dtype=torch.int32
    )
    selected_blocks = torch.full(
        (q.shape[0], block_topk), -1, dtype=torch.int32
    )
    group_size = q.shape[1] // keys.shape[2]
    offsets = torch.arange(index_ratio, dtype=torch.int64)
    for row in range(q.shape[0]):
        request = int(requests[row].item())
        if not 0 <= request < lengths.numel():
            continue
        sequence_length = max(int(lengths[request].item()), 0)
        query_position = max(int(positions[row].item()), -1)
        visible_blocks = min(
            max((query_position + 1) // index_ratio, 0),
            sequence_length // index_ratio,
        )
        visible_blocks = min(visible_blocks, table.shape[1] * page_size // index_ratio)
        if visible_blocks <= 0:
            continue
        block_ids = torch.arange(visible_blocks, dtype=torch.int64)
        logical_token = block_ids * index_ratio
        logical_page = torch.div(logical_token, page_size, rounding_mode="floor")
        page_offset = logical_token.remainder(page_size)
        physical_page = table[request, logical_page]
        page_valid = (physical_page >= 0) & (physical_page < keys.shape[0])
        safe_page = physical_page.clamp(0, max(keys.shape[0] - 1, 0))
        token_offsets = page_offset.unsqueeze(1) + offsets.unsqueeze(0)
        block_keys = keys[safe_page.unsqueeze(1), token_offsets]
        block_keys = block_keys.mean(dim=1)
        block_keys = block_keys.masked_fill(~page_valid[:, None, None], 0.0)

        scores = []
        for head in range(q.shape[1]):
            kv_head = head // group_size
            scores.append(
                torch.relu(torch.einsum("bd,d->b", block_keys[:, kv_head], query[row, head]))
            )
        block_scores = torch.stack(scores, dim=0).sum(dim=0) / math.sqrt(q.shape[2])
        block_scores = block_scores.masked_fill(~page_valid, float("-inf"))
        take = min(block_topk, visible_blocks)
        chosen = torch.topk(block_scores, take, largest=True, sorted=False).indices
        selected_blocks[row, :take] = chosen.to(torch.int32)

        expanded_count = take * index_ratio
        selected_indices[row, :expanded_count] = (
            chosen.unsqueeze(1) * index_ratio + offsets.unsqueeze(0)
        ).reshape(-1).to(torch.int32)
        tail_start = ((query_position + 1) // index_ratio) * index_ratio
        tail_count = min(
            max(query_position + 1 - tail_start, 0), index_ratio - 1
        )
        if tail_count:
            tail_begin = expanded_count
            selected_indices[row, tail_begin : tail_begin + tail_count] = torch.arange(
                tail_start, tail_start + tail_count, dtype=torch.int32
            )
    return selected_indices, selected_blocks


def _finite_count(*tensors: torch.Tensor) -> int:
    return sum(int((~torch.isfinite(tensor.to(torch.float32))).sum().item()) for tensor in tensors)


def _active_token_mask(
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    page_size: int,
    page_count: int,
    valid_page_lengths: torch.Tensor | None = None,
    valid_sequence_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the [physical page, token] cells written by this snapshot.

    A page-table row can cover the whole preallocated pool even when only a prefix was written.
    Keep both the page and token mask so scale calibration and saturation never inspect that
    uninitialized tail.  The valid_* fields are capture-time truth when present; the live lengths
    are the fallback for older snapshots.
    """
    active = torch.zeros((page_count, page_size), dtype=torch.bool)
    table = block_table.to(torch.int64)
    lengths = sequence_lengths.to(torch.int64).reshape(-1)
    page_lengths = (
        None
        if valid_page_lengths is None
        else valid_page_lengths.to(torch.int64).reshape(-1)
    )
    token_lengths = (
        lengths
        if valid_sequence_lengths is None
        else valid_sequence_lengths.to(torch.int64).reshape(-1)
    )
    for request_id, length in enumerate(token_lengths.tolist()):
        if request_id >= table.shape[0]:
            continue
        length = max(int(length), 0)
        pages = (length + page_size - 1) // page_size
        if page_lengths is not None and request_id < page_lengths.numel():
            pages = min(pages, max(int(page_lengths[request_id].item()), 0))
        pages = min(pages, table.shape[1])
        for logical_page in range(pages):
            physical = int(table[request_id, logical_page].item())
            if not 0 <= physical < page_count:
                continue
            start = logical_page * page_size
            count = min(page_size, max(length - start, 0))
            if count:
                active[physical, :count] = True
    return active


def _active_page_mask(
    active_token_mask: torch.Tensor,
    sequence_lengths: torch.Tensor | None = None,
    page_size: int | None = None,
    page_count: int | None = None,
    valid_page_lengths: torch.Tensor | None = None,
    valid_sequence_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the physical pages containing at least one written token.

    The optional legacy arguments preserve the old helper's call shape for offline callers; the
    main checker passes a token mask directly so partial final pages remain excluded.
    """
    if sequence_lengths is not None:
        if page_size is None or page_count is None:
            raise ValueError("page_size and page_count are required with sequence_lengths")
        active_token_mask = _active_token_mask(
            active_token_mask,
            sequence_lengths,
            page_size,
            page_count,
            valid_page_lengths,
            valid_sequence_lengths,
        )
    if active_token_mask.ndim != 2:
        raise ValueError(
            f"active token mask must be [pages, tokens], got {tuple(active_token_mask.shape)}"
        )
    return active_token_mask.any(dim=1)


def _format_scale(
    scales: torch.Tensor, active_page_mask: torch.Tensor | None = None
) -> str:
    if (
        active_page_mask is not None
        and scales.ndim == 4
        and scales.shape[0] == active_page_mask.numel()
    ):
        scales = scales[active_page_mask.to(torch.bool)]
    flat = scales.reshape(-1)
    if flat.numel() == 1:
        return f"{float(flat.item()):.9g}"
    return (
        f"min={float(flat.min().item()):.9g};"
        f"max={float(flat.max().item()):.9g};"
        f"groups={flat.numel()}"
    )


def _scale_storage(
    scaling: str,
    scale_dtype: torch.dtype,
    kv_heads: int,
    valid_tokens: int,
    page_size: int,
    active_page_mask: torch.Tensor,
    active_token_count: int | None = None,
) -> tuple[int, int, int, float]:
    if scaling == "per-layer":
        groups = 1
        scale_values = 2
    elif scaling == "per-head":
        groups = 1
        scale_values = 2 * kv_heads
    elif scaling == "per-head-per-block-64":
        if page_size != _BLOCK_SCALE_SIZE:
            raise ValueError(
                "per-head-per-block-64 requires a 64-token cache page, "
                f"got {page_size}"
            )
        groups = max(int(active_page_mask.sum().item()), 1)
        scale_values = 2 * kv_heads * groups
    else:
        raise ValueError(f"unsupported scaling mode: {scaling}")
    scale_bytes = scale_values * torch.empty((), dtype=scale_dtype).element_size()
    denominator = valid_tokens if active_token_count is None else active_token_count
    bytes_per_token = scale_bytes / max(denominator, 1)
    return groups, scale_values, scale_bytes, bytes_per_token


def _scaling_description(scaling: str) -> str:
    return {
        "per-layer": "one K scalar and one V scalar for the layer",
        "per-head": "one K/V scale pair per KV head",
        "per-head-per-block-64": "one K/V scale pair per KV head in each 64-token page",
    }[scaling]


def _percentile(values: torch.Tensor, quantile: float) -> float:
    if not values.numel():
        return 0.0
    return float(torch.quantile(values.to(torch.float64), quantile).item())


def _result_markdown(result: dict[str, Any], args: argparse.Namespace) -> str:
    fields = [
        "scaling",
        "scale_dtype",
        "scale_values",
        "scale_bytes_total",
        "scale_bytes_per_token",
        "commit",
        "layer",
        "prompt_tokens",
        "valid_kv_tokens",
        "k_scale",
        "v_scale",
        "k_max_abs",
        "v_max_abs",
        "k_saturation_pct",
        "v_saturation_pct",
        "selected_tokens",
        "selected_agreement_pct",
        "output_relative_l2_pct",
        "head_relative_l2_mean_pct",
        "head_relative_l2_p99_pct",
        "head_relative_l2_max_pct",
        "output_max_abs",
        "nan_inf_count",
        "bf16_48_hash",
        "fp8_48_hash",
        "first_48_identical",
        "decision",
    ]
    header = "| " + " | ".join(fields) + " |"
    divider = "|" + "|".join("---" for _ in fields) + "|"
    row = "| " + " | ".join(str(result[field]) for field in fields) + " |"
    gate = result["decision"] == "PASS_OFFLINE_ONLY"
    recommended_gate = result.get("recommended_selection_gate", "no")
    per_query_rows = result.get("selected_agreement_by_query_block", [])
    per_query_table = [
        "| query block | selected tokens (BF16) | intersection | agreement |",
        "|---:|---:|---:|---:|",
    ]
    per_query_table.extend(
        "| {query_block} | {selected_tokens} | {selected_intersection} | {selected_agreement_pct}% |".format(
            **item
        )
        for item in per_query_rows
    )
    section = "\n".join(
        [
            f"## Scaling: `{args.scaling}`",
            "",
            f"This run uses {_scaling_description(args.scaling)}. The two selection replays use separate BF16 and FP8-dequantized K tensors; the attention output replay uses the same selected-token input for the quantization error measurement.",
            "",
            f"- Snapshot: `{args.snapshot}`",
            f"- FP8 dtype: `{args.fp8_dtype}`",
            f"- Scale dtype: `{_SCALE_DTYPE_NAME}`",
            f"- Scale metadata: `{result['scale_bytes_total']}` bytes for the captured valid pages, or `{result['scale_bytes_per_token']}` bytes per active KV token.",
            f"- Selection path: `{result.get('selection_source', 'unknown')}`.",
            f"- BF16 replay versus captured production output: `{result.get('reference_output_relative_l2_pct', 'unknown')}%` relative L2, `{result.get('reference_output_max_abs', 'unknown')}` maximum absolute drift.",
            "- Temperature-0 48-token server comparison: **DEFERRED** until FP8 storage exists in B1.",
            "",
            "| Result fields | Value |",
            "|---|---|",
            f"| K/V scale layout | {_scaling_description(args.scaling)} |",
            f"| K scale summary | {result['k_scale']} |",
            f"| V scale summary | {result['v_scale']} |",
            f"| P0 decision | **{result['decision']}** |",
            f"| meets recommended gate (selection agreement 100%) | **{recommended_gate}** |",
            "",
            "### Selection agreement by query block",
            "",
            *per_query_table,
            "",
            header,
            divider,
            row,
            "",
            "- Selected-token agreement must be at least 97.0% under P0.",
            "- Aggregate relative L2 output error must be at most 1.0% under P0.",
            "- Maximum per-head relative L2 error must be at most 2.0% under P0.",
            "- K and V saturation must each be at most 0.01% of active written values.",
            "- NaN/Inf count must be zero.",
            "",
            f"**{'PASS_OFFLINE_ONLY' if gate else 'RP'}** — {'all offline P0 thresholds passed; the server identity row remains DEFERRED.' if gate else 'one or more offline P0 thresholds failed; do not start B1.'}",
            "",
        ]
    )
    superseded = "\n".join(
        [
            "## Superseded (buggy M2/M2b measurements)",
            "",
            "These rows are retained for audit only. The old checker compared `indices` with itself and divided saturation by the entire allocated pool, so they are not evidence for the repaired gate.",
            "",
            "| scaling | scale values | scale bytes | scale bytes/token | agreement | aggregate L2 | worst per-head L2 | K saturation | V saturation | decision |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            "| per-layer | 2 | 8 | 0.0009765625 | 100% (self-comparison) | 2.24723297% | 5.06217033% | 0.00000595464939% (wrong denominator) | 0% | RP |",
            "| per-head | 4 | 16 | 0.001953125 | 100% (self-comparison) | 2.20654501% | 3.78707871% | 0.00000893197409% (wrong denominator) | 0.0000029773247% (wrong denominator) | RP |",
            "| per-head-per-block-64 | 512 | 2048 | 0.25 | 100% (self-comparison) | 2.45133569% | 3.51648256% | 0.00126834032% (wrong denominator) | 0.00133086414% (wrong denominator) | RP |",
            "",
        ]
    )
    return "\n".join(
        [
            "# FP8 KV accuracy — QSA layer 3",
            "",
            "This is the repaired offline FP8 E4M3 replay. The snapshot has one captured final query row; agreement is therefore reported for query block 0 and as an overall value.",
            "",
            superseded,
            section,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a captured QSA row through FP8 E4M3 K/V")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--fp8-dtype", choices=("e4m3fn",), required=True)
    parser.add_argument(
        "--scaling",
        choices=("per-layer", "per-head", "per-head-per-block-64"),
        required=True,
    )
    parser.add_argument("--min-selected-agreement", type=float, required=True)
    parser.add_argument("--max-relative-l2", type=float, required=True)
    parser.add_argument("--max-head-relative-l2", type=float, required=True)
    parser.add_argument("--max-saturation", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    snapshot = _load_snapshot(Path(args.snapshot))
    fp8_dtype = torch.float8_e4m3fn
    page_size = int(snapshot["page_size"])
    index_ratio = int(snapshot.get("index_ratio", 4))
    q = snapshot["q"]
    k_cache = snapshot["k_cache"]
    v_cache = snapshot["v_cache"]
    indices = snapshot["indices"].to(torch.int32)
    block_table = snapshot["block_table"].to(torch.int32)
    token_to_req = snapshot["token_to_req"].to(torch.int32).reshape(-1)
    sequence_lengths = snapshot["seq_lens"].to(torch.int64).reshape(-1)
    if token_to_req.numel() != q.shape[0]:
        raise ValueError("captured token_to_req does not match captured query rows")
    request_id = int(token_to_req[-1].item())
    if not 0 <= request_id < sequence_lengths.numel():
        raise ValueError(f"captured request id {request_id} is outside seq_lens")

    bf16_out, valid = _replay_attention(
        q, k_cache, v_cache, indices, block_table, token_to_req, page_size
    )
    captured_out = snapshot.get("out")
    if not isinstance(captured_out, torch.Tensor):
        raise ValueError("snapshot must contain captured production output in `out`")
    if captured_out.shape != bf16_out.shape:
        raise ValueError(
            f"captured output shape {tuple(captured_out.shape)} differs from replay "
            f"shape {tuple(bf16_out.shape)}"
        )
    reference_delta = bf16_out - captured_out.to(torch.float32)
    reference_denominator = max(
        float(torch.linalg.vector_norm(captured_out.to(torch.float32)).item()), 1.0e-12
    )
    reference_relative_l2 = (
        float(torch.linalg.vector_norm(reference_delta).item()) / reference_denominator
    )
    reference_max_abs = float(reference_delta.abs().max().item())

    valid_page_lengths = snapshot.get("valid_page_lengths")
    valid_sequence_lengths = snapshot.get("valid_sequence_lengths")
    active_token_mask = _active_token_mask(
        block_table,
        sequence_lengths,
        page_size,
        int(k_cache.shape[0]),
        valid_page_lengths,
        valid_sequence_lengths,
    )
    active_page_mask = _active_page_mask(active_token_mask)
    active_token_count = int(active_token_mask.sum().item())
    if active_token_count <= 0:
        raise ValueError("snapshot contains no active written KV tokens")
    valid_kv_tokens = int(sequence_lengths[request_id].item())
    scale_dtype = _SCALE_DTYPE
    scale_groups, scale_values, scale_bytes_total, scale_bytes_per_token = _scale_storage(
        args.scaling,
        scale_dtype,
        int(k_cache.shape[2]),
        valid_kv_tokens,
        page_size,
        active_page_mask,
        active_token_count,
    )
    fp8_k_cache, k_scales, k_max_abs, k_saturation = _quantize_fp8(
        k_cache, fp8_dtype, args.scaling, scale_dtype, active_token_mask
    )
    fp8_v_cache, v_scales, v_max_abs, v_saturation = _quantize_fp8(
        v_cache, fp8_dtype, args.scaling, scale_dtype, active_token_mask
    )
    fp8_out, fp8_valid = _replay_attention(
        q, fp8_k_cache, fp8_v_cache, indices, block_table, token_to_req, page_size
    )
    if not torch.equal(valid, fp8_valid):
        raise AssertionError("FP8 replay changed the valid selected-token mask")

    # Do not use the captured `indices` for this guard: that would only prove that the old
    # selection was copied.  Clone each K/query input so the BF16 and FP8 score calls cannot
    # accidentally share a quantized or unquantized tensor.
    query_positions = _query_positions(snapshot, q.shape[0])
    captured_blocks = snapshot.get("selected_blocks")
    if isinstance(captured_blocks, torch.Tensor) and captured_blocks.ndim == 2:
        block_topk = max(int((captured_blocks >= 0).sum(dim=1).max().item()), 1)
    else:
        block_topk = max((indices.shape[1] - index_ratio + 1) // index_ratio, 1)
    token_topk = block_topk * index_ratio
    if indices.shape[1] < token_topk:
        raise ValueError("captured token selection is narrower than its block selection")
    bf16_selected_indices, bf16_selected_blocks = _select_main_k_indices(
        q.detach().clone(),
        k_cache.detach().clone(),
        block_table,
        token_to_req,
        sequence_lengths,
        query_positions,
        page_size,
        index_ratio,
        block_topk,
        token_topk,
    )
    fp8_selected_indices, fp8_selected_blocks = _select_main_k_indices(
        q.detach().clone(),
        fp8_k_cache.detach().clone(),
        block_table,
        token_to_req,
        sequence_lengths,
        query_positions,
        page_size,
        index_ratio,
        block_topk,
        token_topk,
    )
    selected_tokens, intersection, selected_agreement = _selected_agreement(
        bf16_selected_indices, fp8_selected_indices
    )
    selection_by_query_block = _agreement_by_query_block(
        bf16_selected_indices, fp8_selected_indices
    )
    captured_selected_tokens, captured_intersection, captured_agreement = _selected_agreement(
        indices, bf16_selected_indices
    )

    delta = fp8_out - bf16_out
    aggregate_denominator = max(float(torch.linalg.vector_norm(bf16_out).item()), 1.0e-12)
    aggregate_relative_l2 = float(torch.linalg.vector_norm(delta).item()) / aggregate_denominator
    head_denominator = torch.linalg.vector_norm(bf16_out, dim=2).clamp_min(1.0e-12)
    head_relative_l2 = torch.linalg.vector_norm(delta, dim=2) / head_denominator
    finite_tensors = [
        q,
        k_cache[active_token_mask],
        v_cache[active_token_mask],
        fp8_k_cache[active_token_mask],
        fp8_v_cache[active_token_mask],
        captured_out,
        bf16_out,
        fp8_out,
    ]
    nan_inf_count = _finite_count(*finite_tensors)

    prompt_tokens = int(snapshot.get("prompt_tokens", sequence_lengths.max().item()))
    result: dict[str, Any] = {
        "scaling": args.scaling,
        "scale_dtype": _SCALE_DTYPE_NAME,
        "scale_groups": scale_groups,
        "scale_values": scale_values,
        "scale_bytes_total": scale_bytes_total,
        "scale_bytes_per_token": f"{scale_bytes_per_token:.9g}",
        "commit": _git_commit(),
        "layer": int(snapshot["layer_id"]),
        "prompt_tokens": prompt_tokens,
        "valid_kv_tokens": valid_kv_tokens,
        "active_written_pages": int(active_page_mask.sum().item()),
        "active_written_tokens": active_token_count,
        "k_scale": _format_scale(k_scales, active_page_mask),
        "v_scale": _format_scale(v_scales, active_page_mask),
        "k_max_abs": f"{float(k_max_abs.max().item()):.9g}",
        "v_max_abs": f"{float(v_max_abs.max().item()):.9g}",
        "k_saturation_pct": f"{100.0 * k_saturation:.9g}",
        "v_saturation_pct": f"{100.0 * v_saturation:.9g}",
        "selection_source": "main-K score replay (compressed index absent from snapshot)",
        "selected_tokens": selected_tokens,
        "selected_intersection": intersection,
        "selected_agreement_pct": f"{100.0 * selected_agreement:.9g}",
        "selected_agreement_by_query_block": selection_by_query_block,
        "recommended_selection_gate": "yes" if selected_agreement == 1.0 else "no",
        "captured_selection_tokens": captured_selected_tokens,
        "captured_vs_replayed_bf16_intersection": captured_intersection,
        "captured_vs_replayed_bf16_agreement_pct": f"{100.0 * captured_agreement:.9g}",
        "reference_output_relative_l2_pct": f"{100.0 * reference_relative_l2:.9g}",
        "reference_output_max_abs": f"{reference_max_abs:.9g}",
        "output_relative_l2_pct": f"{100.0 * aggregate_relative_l2:.9g}",
        "head_relative_l2_mean_pct": f"{100.0 * float(head_relative_l2.mean().item()):.9g}",
        "head_relative_l2_p99_pct": f"{100.0 * _percentile(head_relative_l2, 0.99):.9g}",
        "head_relative_l2_max_pct": f"{100.0 * float(head_relative_l2.max().item()):.9g}",
        "output_max_abs": f"{float(delta.abs().max().item()):.9g}",
        "nan_inf_count": nan_inf_count,
        "bf16_48_hash": "DEFERRED",
        "fp8_48_hash": "DEFERRED",
        "first_48_identical": "DEFERRED",
    }
    pass_gate = (
        selected_agreement >= args.min_selected_agreement
        and aggregate_relative_l2 <= args.max_relative_l2
        and float(head_relative_l2.max().item()) <= args.max_head_relative_l2
        and k_saturation <= args.max_saturation
        and v_saturation <= args.max_saturation
        and nan_inf_count == 0
    )
    result["decision"] = "PASS_OFFLINE_ONLY" if pass_gate else "RP"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown = _result_markdown(result, args)
    output.write_text(markdown, encoding="utf-8")
    print(f"SNAPSHOT {args.snapshot}")
    print(
        f"CONFIG fp8_dtype={args.fp8_dtype} scaling={args.scaling} "
        f"scale_dtype={_SCALE_DTYPE_NAME}"
    )
    print("RESULT " + json.dumps(result, sort_keys=True))
    print(f"DECISION {result['decision']}")
    print(f"OUTPUT {output}")


if __name__ == "__main__":
    main()

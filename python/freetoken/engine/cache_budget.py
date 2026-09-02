"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def expert_bytes_per_slot(
    sources: dict[str, "list[torch.Tensor]"],
    gpu_owned_layers: "frozenset[int]" = frozenset(),
) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size. ``gpu_owned_layers`` are permanently resident and never enter the slot
    cache, so the geometry is read from the first STREAMING layer (layer 0 is in
    the default owned set, and an owned layer's tensor lives on the device).
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[layer][0].numel() is the per-row element count (one expert slot); see the
    # matching slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4_paged_pool.py.
    first = next(i for i in range(len(next(iter(sources.values())))) if i not in gpu_owned_layers)
    return sum(t[first][0].numel() * t[first].element_size() for t in sources.values())


def gpu_owned_reservation_bytes(
    owned_layers: int, num_experts: int, per_expert_bytes: int
) -> int:
    """VRAM the GPU-owned MoE layers hold permanently: one full expert layer each.

    Accounted exactly like ``state_pool_bytes`` -- it joins ``fixed_cache_size`` before the
    MoE-vs-KV split, so the greedy slot fill never spends bytes the owned layers already own.
    """
    return owned_layers * num_experts * per_expert_bytes


def lru_slots_after_owned_charge(
    *, moe_cache_size: int, owned_layers: int, num_experts: int, floor: int
) -> int:
    """Split an explicit ``--moe-cache-size`` into the owned reservation and the LRU.

    ``--moe-cache-size`` is the TOTAL expert-slot budget on the card. The GPU-owned layers
    hold one full expert layer each and are CHARGED to that budget; the LRU keeps the rest.
    Charging rather than adding is what makes ``--moe-gpu-owned-layers`` VRAM-neutral:
    turning it on can never raise total MoE residency above the size the operator asked for.

    Measured 2026-09-02 on an RTX 5090 32 GB: 6 owned layers ADDED to 4400 slots put the card
    at 569 MiB free at decode peak and halved throughput (70.4 -> 34.6 tok/s) even though it
    moved 11 % FEWER expert rows per step; the same owned set CHARGED to a 6750-slot budget
    (3678 LRU slots, byte-identical residency to the no-owned baseline) runs at 63 tok/s. See
    ``docs/research/gpu-owned-layers-speed-diagnosis-2026-09-02.md``.

    ``--moe-cache-auto`` needs none of this: there the reservation already joins
    ``fixed_cache_size`` before :func:`plan_cache_budget` splits what is left.
    """
    charge = gpu_owned_reservation_slots(owned_layers, num_experts)
    lru = moe_cache_size - charge
    if lru < floor:
        raise ValueError(
            f"--moe-cache-size {moe_cache_size} is the TOTAL expert-slot budget, and "
            f"{owned_layers} GPU-owned MoE layer(s) charge {charge} slots of it "
            f"({owned_layers} x {num_experts} experts), leaving {lru} for the LRU -- but the "
            f"streaming layers need at least {floor}. Raise --moe-cache-size to "
            f"{floor + charge} or own fewer layers."
        )
    return lru


def gpu_owned_reservation_slots(owned_layers: int, num_experts: int) -> int:
    """Slot-equivalents the GPU-owned layers hold permanently: one full expert layer each.

    The slot twin of :func:`gpu_owned_reservation_bytes` -- same memory, counted in the unit
    ``--moe-cache-size`` is expressed in.
    """
    return owned_layers * num_experts


#: Default post-cache VRAM reservation, in bytes: everything the engine allocates on the card
#: AFTER the MoE slot cache has been sized, and which therefore no budget term used to know
#: about. Measured on an RTX 5090 32 GB with Qwen3.8-Flash-Next-NVFP4 (2026-09-02):
#:   * the integrated MTP resident draft head -- 2.17 GiB (``FREETOKEN_MTP_RESIDENT=1``; boot
#:     logs it as ``draft head 2.17 GiB resident``), built after ``_init_offload_moe_cache``
#:     on purpose, so its bytes cannot be measured before the cache exists;
#:   * the MTP spec/draft CUDA-graph pools and the decode graph pool;
#:   * the vision layer-stream encoder's transient workspace.
#: 3 GiB covers the measured draft head with room for the graph pools. It is the reserve of
#: a boot that runs speculation; :func:`auto_vram_reserve_bytes` composes the reserve a given
#: boot actually needs out of the two components below. Override with
#: ``--moe-vram-reserve-bytes``.
#:
#: The draft-head half: 2.25 GiB, the 2.17 GiB measured resident head plus its slack. Only
#: allocated when integrated speculation (or the MTP shadow observer) is on, so a boot
#: without either must not be charged for it -- 2.25 GiB is ~850 expert slots on this
#: geometry, and reserving bytes nothing will allocate is the same sizing error as spending
#: bytes something will.
MTP_DRAFT_HEAD_RESERVE_BYTES = 9 << 28
#: The always-on half: 0.75 GiB for the decode/spec/draft CUDA-graph pools and the vision
#: layer-stream encoder's transient workspace. Charged on every boot -- the decode graph pool
#: exists whatever else is switched off.
GRAPH_POOL_RESERVE_BYTES = 3 << 28
DEFAULT_MOE_VRAM_RESERVE_BYTES = MTP_DRAFT_HEAD_RESERVE_BYTES + GRAPH_POOL_RESERVE_BYTES


def auto_vram_reserve_bytes(*, mtp_resident: bool) -> int:
    """The post-cache reserve for a boot with these features on (``--moe-vram-reserve-bytes``
    left at its ``-1`` auto default).

    ``mtp_resident`` is whether a resident MTP draft head will be built after the cache:
    integrated speculation (``config.spec_decode.enabled``) or the MTP shadow observer.
    """
    return GRAPH_POOL_RESERVE_BYTES + (MTP_DRAFT_HEAD_RESERVE_BYTES if mtp_resident else 0)


def resolve_vram_reserve_bytes(declared: int, *, mtp_resident: bool) -> int:
    """``--moe-vram-reserve-bytes`` as the budget sees it: ``-1`` = auto, else taken as typed
    (``0`` restores the pre-2026-09 behaviour of reserving nothing)."""
    if declared == -1:
        return auto_vram_reserve_bytes(mtp_resident=mtp_resident)
    if declared < 0:
        raise ValueError(
            f"--moe-vram-reserve-bytes must be >= 0, or -1 for auto, got {declared}"
        )
    return declared

#: Default free-VRAM headroom the MoE cache must leave after every known reservation, in
#: bytes. 1.5 GiB: every healthy configuration measured on this box peaked at 1.3-1.4 GiB
#: free during decode, while the 569 MiB-free run halved decode throughput because the
#: Windows video-memory manager began demoting live allocations to system memory. See
#: ``docs/research/gpu-owned-layers-speed-diagnosis-2026-09-02.md``. Override with
#: ``--moe-cache-headroom-bytes``.
DEFAULT_MOE_CACHE_HEADROOM_BYTES = 3 << 29


def post_cache_reserved_bytes(*, vram_reserve_bytes: int, headroom_bytes: int) -> int:
    """The single place post-cache reservations are declared before the cache is sized.

    Joins ``fixed_cache_size`` exactly like ``state_pool_bytes`` and the GPU-owned
    reservation, so neither ``--moe-cache-auto``'s greedy slot fill nor an explicit
    ``--moe-cache-size`` can spend bytes a later allocation already owns.
    """
    if vram_reserve_bytes < 0:
        raise ValueError(f"--moe-vram-reserve-bytes must be >= 0, got {vram_reserve_bytes}")
    if headroom_bytes < 0:
        raise ValueError(f"--moe-cache-headroom-bytes must be >= 0, got {headroom_bytes}")
    return vram_reserve_bytes + headroom_bytes


def format_vram_ledger(
    *,
    total_bytes: int,
    weights_bytes: int,
    kv_bytes: int,
    gdn_state_bytes: int,
    gpu_owned_bytes: int,
    gpu_owned_layers: "tuple[int, ...]",
    lru_slots: int,
    lru_bytes: int,
    vram_reserve_bytes: int,
    headroom_bytes: int,
) -> str:
    """One block naming every VRAM term the boot budget knows about, largest first.

    ``unaccounted`` is what the ledger cannot name: allocator slack, activations, and
    anything a model allocates outside the budgeted pools. A negative value means the plan
    is over the card -- which is the number to look at when a boot starts paging.
    """
    named = (
        weights_bytes
        + kv_bytes
        + gdn_state_bytes
        + gpu_owned_bytes
        + lru_bytes
        + vram_reserve_bytes
        + headroom_bytes
    )
    owned_note = f" {list(gpu_owned_layers)}" if gpu_owned_layers else ""
    rows = (
        ("weights", f"{_gib(weights_bytes)}"),
        ("KV cache", f"{_gib(kv_bytes)}"),
        ("GDN state pool", f"{_gib(gdn_state_bytes)}"),
        (
            "GPU-owned MoE layers",
            f"{_gib(gpu_owned_bytes)} ({len(gpu_owned_layers)} layers{owned_note})",
        ),
        ("MoE LRU cache", f"{_gib(lru_bytes)} ({lru_slots} slots)"),
        ("post-cache reserve", f"{_gib(vram_reserve_bytes)} (MTP draft head, graphs, vision)"),
        ("headroom", f"{_gib(headroom_bytes)}"),
        ("unaccounted", f"{_gib(total_bytes - named)}"),
    )
    width = max(len(label) for label, _ in rows)
    lines = [f"VRAM ledger ({_gib(total_bytes)} on the card):"]
    lines += [f"  {label:<{width}}  {value}" for label, value in rows]
    lines.append(f"  {'named total':<{width}}  {_gib(named)}")
    return "\n".join(lines)


def _gib(nbytes: int) -> str:
    return f"{nbytes / 2**30:.2f} GiB"


def check_explicit_moe_cache_fits(
    *,
    moe_cache_size: int,
    per_expert_bytes: int,
    budget_bytes: int,
    owned_layers: int,
    num_experts: int,
    reserved_bytes: int = 0,
) -> None:
    """Raise when an explicit ``--moe-cache-size`` plus the GPU-owned reservation plus the
    post-cache reservations exceed the net MoE budget. Operator decision: fail loudly naming
    the largest slot count that fits, never silently shrink the cache the operator asked for.
    """
    owned_bytes = gpu_owned_reservation_bytes(owned_layers, num_experts, per_expert_bytes)
    need = moe_cache_size * per_expert_bytes + owned_bytes + reserved_bytes
    if need <= budget_bytes:
        return
    fits_slots = max(0, (budget_bytes - owned_bytes - reserved_bytes) // per_expert_bytes)
    fits_owned = max(
        0,
        (budget_bytes - moe_cache_size * per_expert_bytes - reserved_bytes)
        // (num_experts * per_expert_bytes),
    )
    owned_clause = (
        f" plus {owned_layers} GPU-owned MoE layers ({owned_bytes} B resident)"
        if owned_layers
        else ""
    )
    owned_advice = (
        f", or own at most {fits_owned} layer(s) at this cache size" if owned_layers else ""
    )
    raise ValueError(
        f"--moe-cache-size {moe_cache_size}{owned_clause} plus {reserved_bytes} B of "
        f"post-cache reservations (--moe-vram-reserve-bytes + --moe-cache-headroom-bytes) "
        f"needs {need} B of the {budget_bytes} B MoE budget. "
        f"Either lower --moe-cache-size to {fits_slots} slots{owned_advice}."
    )


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = max(remaining // cache_per_page, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = moe_cache_size * per_expert_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
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
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = 992 if quant_format == "nvfp4_marlin" else total_experts
    # Every pool keeps page 0 as an unreachable dummy/sentinel. The CLI floor is expressed in
    # usable tokens, so reserve that internal page in addition to the user-visible capacity.
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size) + 1
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )

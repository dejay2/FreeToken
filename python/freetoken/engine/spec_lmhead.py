"""The DRAFT chain's own LM head.

WHY THE DRAFT NEEDS ONE
-----------------------
``SpecDraftHead.propose`` projects one row through the TARGET's ``lm_head`` per drafted token.
On the shipping checkpoint that head is bf16 and untied -- 248,320 x 2,560 -- so every draft
step streams 1.27 GB out of HBM for a single [1, vocab] row. At the RTX 5090's ~1.5 TB/s that
is ~0.8 ms of pure bandwidth, and a depth-5 chain pays it five times: roughly half the whole
chain, spent on a projection whose only consumer is a proposal the target re-checks anyway.

Nothing about the TARGET's logits changes. This is a separate copy of the weight behind a
separate object; ``Qwen4ExpForCausalLM.lm_head`` is neither read nor rewritten, so the served
distribution stays bit-for-bit what it was.

WHAT THE COPY IS
----------------
Weight-only int8 (A16W8), one symmetric scale per output row, run through
``torch._weight_int8pack_mm`` -- ATen's gpt-fast decode GEMV, which takes a bf16 activation
and an int8 [N, K] weight and needs no activation quantization, no M >= 17 padding (the
``torch._int_mm`` IMMA path does), no rescale epilogue and no new kernel. Halves the traffic
(1.27 GB -> 0.64 GB, ~0.4 ms/step, ~2 ms off a depth-5 chain), fixed shapes and no host sync
so it stays graph-legal.

WHY NOT NVFP4
-------------
An NVFP4 copy would be cheaper still (~0.35 GB, and the repo already owns a tuned W4A16 GEMV
in ``kernel/triton/nvfp4_linear.py``), but 4 bits with a per-16 block scale is a ~9.5% RMS
relative weight error against int8's ~0.84%, and on the very test this module is held to --
random Gaussian weights and hidden states, where the top-2 logit gap is small -- that comes
out at 84% top-1 agreement with the bf16 head against int8's 98%+. A draft head that disagrees
with the target's own head one time in six proposes tokens the target will reject, which is
the one thing the chain must not spend its rows on. ``nvfp4`` is therefore accepted by the
flag only when the CHECKPOINT's lm_head is already NVFP4 (then it is the target's own head,
exact and free); asking for a converted copy fails loudly rather than quietly drafting worse.

THE VRAM TRADE
--------------
The int8 copy is 0.64 GB resident (plus 0.5 MB of scales). This box boots at 31.9/32.6 GB with
a 5,000-slot target expert cache, so it is NOT free: at 2.766 MB per NVFP4 expert slot it is
~230 slots, ~4.6% of the cache, and the operator has to give them back
(``-MoECacheSize 4770``) or shrink the KV pool. That is the trade the flag exists to make
reversible: ``FREETOKEN_MTP_SPEC_DRAFT_LMHEAD=bf16`` restores the shared head exactly, cache
and all.
"""

from __future__ import annotations

import os
from typing import Mapping

import torch

_DRAFT_LMHEAD_ENV = "FREETOKEN_MTP_SPEC_DRAFT_LMHEAD"
_PLACEMENTS = ("bf16", "int8", "nvfp4")
# Output rows quantized per pass. The fp32 working copy of the shipping head would be 2.5 GB
# in one go; at this chunk it is 84 MB, which fits beside a card that is already nearly full.
_QUANT_CHUNK_ROWS = 8192


def resolve_draft_lmhead_placement(environ: Mapping[str, str] | None = None) -> str:
    """Which LM head the DRAFT projects through (``FREETOKEN_MTP_SPEC_DRAFT_LMHEAD``).

    Parsed here rather than on ``EngineConfig`` for the same reason the expert placement is:
    it is a private weight layout of this head, not a serving flag.

    The default is ``bf16`` (the shared target head, byte for byte). ``int8`` halves the bytes
    but LOSES time on this box: ``torch._weight_int8pack_mm``'s CUDA path is a naive kernel,
    measured 2026-09-02 on the RTX 5090 at 1.56 ms per [1, 2560] x [248320, 2560] call against
    0.81 ms for the bf16 cuBLAS GEMV, and it scales linearly with rows (4.6 ms at M=6). It
    stays selectable for a box with a real W8A16 GEMV (or once one is written); fp8 row-wise
    ``_scaled_mm`` is not supported on this device by torch 2.11, so it is not an alternative.
    """
    env = os.environ if environ is None else environ
    placement = (env.get(_DRAFT_LMHEAD_ENV, "") or "bf16").strip().lower() or "bf16"
    if placement not in _PLACEMENTS:
        raise ValueError(
            f"{_DRAFT_LMHEAD_ENV} must be one of {'|'.join(_PLACEMENTS)}, got {placement!r}"
        )
    return placement


def quantize_int8_rows(
    weight: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    chunk_rows: int = _QUANT_CHUNK_ROWS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row symmetric int8 for a ``[N, K]`` weight: ``(codes int8, scales)``.

    The scale is stored in the COMPUTE dtype and the codes are then rounded against that
    stored value, not against the fp32 scale it came from. ``_weight_int8pack_mm`` multiplies
    by what is stored, so folding the scale's own rounding in here leaves quantization as the
    only error; leaving it out would add a per-row gain error of up to one bf16 ulp (~0.4%)
    that moves an entire row's logits together -- which is exactly the error argmax notices.

    Chunked over output rows so the fp32 intermediate stays bounded regardless of vocabulary.
    """
    if weight.ndim != 2:
        raise ValueError(f"an int8 LM head weight must be [N, K], got {tuple(weight.shape)}")
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    dtype = weight.dtype if dtype is None else dtype
    rows, width = weight.shape
    codes = torch.empty((rows, width), dtype=torch.int8, device=weight.device)
    scales = torch.empty(rows, dtype=dtype, device=weight.device)
    # An all-zero output row has no scale of its own; any positive one reproduces it exactly
    # (0 / s = 0), and the smallest normal keeps the stored value representable in bf16.
    floor = torch.finfo(torch.float32).tiny
    for lo in range(0, rows, chunk_rows):
        hi = min(lo + chunk_rows, rows)
        block = weight[lo:hi].float()
        scale = (block.abs().amax(dim=1) / 127.0).clamp_min(floor).to(dtype)
        block /= scale.float().unsqueeze(1)
        codes[lo:hi] = block.round_().clamp_(-127.0, 127.0).to(torch.int8)
        scales[lo:hi] = scale
    return codes, scales


def _same_device(wanted: torch.device, actual: torch.device) -> bool:
    """``cuda`` and ``cuda:0`` name the same card; ``torch.device`` equality does not agree."""
    if wanted.type != actual.type:
        return False
    return wanted.index is None or wanted.index == actual.index


class Int8DraftLMHead:
    """A weight-only int8 copy of the target LM head, projecting every row it is given.

    Only ``forward_all`` exists: the draft never takes ``ParallelLMHead.forward``'s
    prefill last-row slice (it projects exactly the one row a draft step produced), and the
    absent method is what keeps this object from being mistaken for a servable head.
    """

    def __init__(self, codes: torch.Tensor, scales: torch.Tensor) -> None:
        if codes.ndim != 2 or codes.dtype is not torch.int8:
            raise ValueError("an int8 draft LM head needs a two-dimensional int8 weight")
        if scales.ndim != 1 or scales.shape[0] != codes.shape[0]:
            raise ValueError("an int8 draft LM head needs one scale per output row")
        if scales.device != codes.device:
            raise ValueError("an int8 draft LM head's weight and scales must share a device")
        self.weight = codes
        self.scales = scales
        self.num_embeddings = int(codes.shape[0])
        self.embedding_dim = int(codes.shape[1])

    @classmethod
    def from_weight(
        cls, weight: torch.Tensor, *, device: torch.device | None = None
    ) -> "Int8DraftLMHead":
        # Quantized where the weight already lives, and moved only if that is genuinely a
        # different device: a spurious ``.to()`` here would briefly hold TWO copies of a
        # 0.64 GB head on a card the operator has deliberately filled.
        codes, scales = quantize_int8_rows(weight)
        if device is not None and not _same_device(torch.device(device), codes.device):
            codes = codes.to(device)
            scales = scales.to(device)
        return cls(codes, scales)

    @property
    def resident_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in (self.weight, self.scales)
        )

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.embedding_dim)
        if rows.dtype is not self.scales.dtype:
            rows = rows.to(self.scales.dtype)
        return torch._weight_int8pack_mm(rows.contiguous(), self.weight, self.scales)


def target_lm_head_weight(lm_head) -> torch.Tensor:
    """The ``[vocab, hidden]`` projection matrix behind a target LM head.

    Follows the tied-embedding indirection ``ParallelLMHead._project`` follows, and refuses a
    sharded head: every row of the vocabulary has to be in this copy or the draft would sample
    from a truncated distribution.
    """
    if int(getattr(lm_head, "tp_size", 1)) != 1:
        raise ValueError("a private draft LM head copy needs an unsharded target head")
    module = getattr(lm_head, "tied_embedding", None) or lm_head
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(f"{type(lm_head).__name__} exposes no LM head weight to copy")
    return weight


def build_draft_lm_head(lm_head, *, placement: str, device: torch.device | None = None):
    """The object ``SpecDraftHead.propose`` projects through, for one resolved placement."""
    if placement == "bf16":
        return lm_head
    already_quantized = type(lm_head).__name__ == "Nvfp4LMHead"
    if placement == "nvfp4":
        if already_quantized:
            # The checkpoint's own 4-bit head: no copy, no extra VRAM, and exactly the logits
            # the target computes -- there is nothing to trade.
            return lm_head
        raise ValueError(
            f"{_DRAFT_LMHEAD_ENV}=nvfp4 is only available when the CHECKPOINT's lm_head is "
            "NVFP4. Converting a bf16 head to NVFP4 measures 84% top-1 agreement against 98% "
            f"for int8 (see this module's docstring); use {_DRAFT_LMHEAD_ENV}=int8"
        )
    if already_quantized:
        # A 4-bit head is already cheaper than the int8 copy would be; quantizing it again
        # would cost VRAM to lose accuracy.
        return lm_head
    return Int8DraftLMHead.from_weight(target_lm_head_weight(lm_head), device=device)


__all__ = [
    "Int8DraftLMHead",
    "build_draft_lm_head",
    "quantize_int8_rows",
    "resolve_draft_lmhead_placement",
    "target_lm_head_weight",
]

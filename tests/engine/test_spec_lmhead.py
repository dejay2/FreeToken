"""The draft chain's private LM head.

The chain projects one row through the LM head per drafted token. On the shipping checkpoint
that head is bf16 and 1.27 GB, so a depth-5 chain streams 6.4 GB of weight out of HBM to
produce five [1, vocab] rows. This module's job is to make that copy cheaper without making
the DRAFT disagree with the target's own head -- a draft the target rejects costs a verify row
and buys nothing.

What is pinned here: the placement flag, that the quantization is exact where it claims to be,
and the bar -- top-1 agreement with the bf16 head on random Gaussian hidden states, which is
the least forgiving distribution there is (the top-2 logit gap is smallest when nothing about
the weights or the inputs is structured).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.spec_lmhead import (
    Int8DraftLMHead,
    build_draft_lm_head,
    quantize_int8_rows,
    resolve_draft_lmhead_placement,
    target_lm_head_weight,
)

CPU = torch.device("cpu")
# The bar. 0.97 is the contract; the measured value on this fixture is reported by
# ``test_the_int8_draft_head_agrees_with_the_bf16_head_on_its_top_1``.
_AGREEMENT_BAR = 0.97


def _weight(vocab: int, hidden: int, *, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(vocab, hidden, generator=generator) * 0.02).to(torch.bfloat16)


def _bf16_head(vocab: int = 512, hidden: int = 256, *, seed: int = 3):
    """A stand-in for ``ParallelLMHead``: a weight, a TP size and ``forward_all``."""
    weight = _weight(vocab, hidden, seed=seed)
    head = SimpleNamespace(weight=weight, tp_size=1, tied_embedding=None)
    head.forward_all = lambda x: torch.nn.functional.linear(x, weight)
    return head


# --------------------------------------------------------------------------- the placement


def test_the_placement_defaults_to_bf16_because_int8pack_mm_is_slower_on_this_box():
    # measured 2026-09-02: _weight_int8pack_mm 1.56 ms vs bf16 GEMV 0.81 ms at M=1 (see module)
    assert resolve_draft_lmhead_placement({}) == "bf16"
    assert resolve_draft_lmhead_placement({"FREETOKEN_MTP_SPEC_DRAFT_LMHEAD": "bf16"}) == "bf16"
    assert resolve_draft_lmhead_placement({"FREETOKEN_MTP_SPEC_DRAFT_LMHEAD": " INT8 "}) == "int8"
    assert resolve_draft_lmhead_placement({"FREETOKEN_MTP_SPEC_DRAFT_LMHEAD": ""}) == "bf16"


def test_an_unknown_placement_is_refused_by_name():
    with pytest.raises(ValueError, match="bf16|int8|nvfp4"):
        resolve_draft_lmhead_placement({"FREETOKEN_MTP_SPEC_DRAFT_LMHEAD": "fp8"})


def test_bf16_hands_back_the_targets_own_head_object():
    """Not a copy of it: ``bf16`` has to be the pre-existing behaviour byte for byte, and a
    copy would also silently double the 1.27 GB."""
    head = _bf16_head()
    assert build_draft_lm_head(head, placement="bf16", device=CPU) is head


def test_an_nvfp4_copy_of_a_bf16_head_is_refused_with_its_reason():
    """4 bits measures 84% top-1 agreement against int8's 98% on this fixture, so the flag
    value exists for a checkpoint whose head is ALREADY NVFP4 and nothing else."""
    with pytest.raises(ValueError, match="int8"):
        build_draft_lm_head(_bf16_head(), placement="nvfp4", device=CPU)


def test_a_checkpoint_native_nvfp4_head_is_reused_rather_than_re_quantized():
    """There is nothing to trade there: it is already 4-bit, and it is the very head the
    target projects through, so the draft's logits are the target's."""
    native = type("Nvfp4LMHead", (), {"forward_all": lambda self, x: x})()
    assert build_draft_lm_head(native, placement="nvfp4", device=CPU) is native
    assert build_draft_lm_head(native, placement="int8", device=CPU) is native


def test_a_sharded_target_head_cannot_be_copied():
    """A per-rank slice of the vocabulary would make the draft sample from a truncated
    distribution -- silently, and only under TP > 1."""
    head = _bf16_head()
    head.tp_size = 2
    with pytest.raises(ValueError, match="unsharded"):
        target_lm_head_weight(head)


def test_the_weight_is_followed_through_a_tied_embedding():
    head = _bf16_head()
    tied = SimpleNamespace(weight=_weight(512, 256, seed=9))
    head.tied_embedding = tied
    assert target_lm_head_weight(head) is tied.weight


# ------------------------------------------------------------------------- the quantization


def test_the_codes_are_rounded_against_the_scale_that_is_actually_stored():
    """``_weight_int8pack_mm`` multiplies by the STORED scale. Rounding the codes against the
    fp32 scale they came from instead would leave a per-row gain error of up to one bf16 ulp,
    which moves a whole row's logits together -- exactly the error argmax notices."""
    weight = _weight(64, 128)
    codes, scales = quantize_int8_rows(weight, dtype=torch.bfloat16)

    assert codes.dtype is torch.int8 and scales.dtype is torch.bfloat16
    expected = (
        (weight.float() / scales.float().unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    )
    assert torch.equal(codes, expected)
    assert int(codes.abs().max()) <= 127


def test_a_zero_output_row_survives_quantization_exactly():
    weight = _weight(8, 32)
    weight[3].zero_()
    codes, scales = quantize_int8_rows(weight)

    assert bool((codes[3] == 0).all())
    assert float(scales[3]) > 0.0  # a zero scale would make the reconstruction NaN


def test_chunking_the_quantization_changes_nothing():
    """The fp32 intermediate is bounded by chunking the output rows; the result must not
    depend on where the chunk boundaries fall."""
    weight = _weight(300, 64)
    whole = quantize_int8_rows(weight, chunk_rows=4096)
    split = quantize_int8_rows(weight, chunk_rows=7)

    assert torch.equal(whole[0], split[0])
    assert torch.equal(whole[1], split[1])


# ------------------------------------------------------------------------------ the bar


@pytest.mark.parametrize("vocab,hidden", [(512, 256), (512, 2560), (1024, 2560)])
def test_the_int8_draft_head_agrees_with_the_bf16_head_on_its_top_1(vocab, hidden, capsys):
    """The contract: >= 97% argmax agreement on random Gaussian hidden states.

    Random weights AND random inputs is the worst case for this measurement -- the logits are
    near-Gaussian, so the top-two gap is as small as it ever gets and any weight perturbation
    has its best chance of flipping the winner. A real head's own structure only helps.
    """
    head = _bf16_head(vocab, hidden)
    draft = build_draft_lm_head(head, placement="int8", device=CPU)
    assert isinstance(draft, Int8DraftLMHead)

    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2000, hidden, generator=generator).to(torch.bfloat16)
    reference = head.forward_all(x)
    got = draft.forward_all(x)

    agreement = float((reference.argmax(-1) == got.argmax(-1)).float().mean())
    with capsys.disabled():
        print(f"\nint8 draft lm_head top-1 agreement (V={vocab}, H={hidden}): {agreement:.4f}")
    assert agreement >= _AGREEMENT_BAR


def test_an_nvfp4_copy_would_miss_the_bar_which_is_why_int8_is_the_default(capsys):
    """The measurement behind ``build_draft_lm_head``'s refusal, kept executable so the
    comparison is a fact rather than a comment. Uses the repo's own NVFP4 reference pair, so
    it is the same arithmetic ``nvfp4_dense_linear`` implements.
    """
    from freetoken.models.qwen4_exp.mtp_spike import (
        dequantize_nvfp4_rows,
        quantize_nvfp4_rows,
    )

    vocab, hidden = 512, 256
    head = _bf16_head(vocab, hidden)
    packed, scale, row_global = quantize_nvfp4_rows(head.weight.float())
    fp4_weight = dequantize_nvfp4_rows(packed, scale, row_global).to(torch.bfloat16)

    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2000, hidden, generator=generator).to(torch.bfloat16)
    reference = head.forward_all(x).argmax(-1)
    fp4 = torch.nn.functional.linear(x, fp4_weight).argmax(-1)
    int8 = build_draft_lm_head(head, placement="int8", device=CPU).forward_all(x).argmax(-1)

    fp4_agreement = float((reference == fp4).float().mean())
    int8_agreement = float((reference == int8).float().mean())
    with capsys.disabled():
        print(f"\nnvfp4 {fp4_agreement:.4f} vs int8 {int8_agreement:.4f} top-1 agreement")
    assert fp4_agreement < _AGREEMENT_BAR <= int8_agreement


# ---------------------------------------------------------------------------- the copy itself


def test_the_int8_head_halves_the_bytes_a_draft_step_reads():
    head = _bf16_head(1024, 256)
    draft = build_draft_lm_head(head, placement="int8", device=CPU)

    bf16_bytes = head.weight.numel() * head.weight.element_size()
    # the scales are one per output row -- a rounding error against the weight itself
    assert draft.resident_bytes < 0.55 * bf16_bytes


def test_the_int8_head_projects_every_row_it_is_given():
    """``forward_all`` is the whole seam: the draft never wants ``ParallelLMHead.forward``'s
    prefill last-row slice, and the absent method is what keeps this from being served."""
    head = _bf16_head(128, 64)
    draft = build_draft_lm_head(head, placement="int8", device=CPU)

    assert draft.forward_all(torch.zeros(3, 64, dtype=torch.bfloat16)).shape == (3, 128)
    assert draft.forward_all(torch.zeros(1, 64, dtype=torch.bfloat16)).shape == (1, 128)
    assert not hasattr(draft, "forward")


def test_a_malformed_int8_head_is_refused():
    with pytest.raises(ValueError, match="int8"):
        Int8DraftLMHead(torch.zeros(4, 8), torch.zeros(4))
    with pytest.raises(ValueError, match="one scale per output row"):
        Int8DraftLMHead(torch.zeros(4, 8, dtype=torch.int8), torch.zeros(3))

"""The speculative sampler and the server must define the SAME distribution.

``SpecSampler`` accepts a draft with probability ``min(1, p/q)`` and corrects from the residual
``(p - q)+``, with both ``p`` and ``q`` coming from ``filtered_probs``. The token the server
would have emitted instead comes from ``Sampler.sample`` -> ``sample_impl``
(``engine/sample.py:24-50``), which dispatches to ``flashinfer.sampling`` when that package is
installed and to ``freetoken.kernel.triton.sampling`` otherwise. If the two disagree,
speculation silently changes client-visible output -- no crash, no divergence report.

``sample_impl`` composes the backend's own ops, so the distribution it draws from is
``top_p_renorm(top_k_renorm(softmax(logits / T), k), p)`` -- top-p on the top-k-RENORMALIZED
probs (``kernel/triton/sampling.py:598-612``; flashinfer's ``filter_apply_order`` default is
the same ``top_k_first``). Both renorm ops are public on both backends, so the comparison below
is exact rather than statistical, and runs on whichever backend is installed. The design
flagged the flashinfer path UNVERIFIED; it is exercised automatically the day the package
appears in the venv and reported as skipped until then, so it cannot stay unverified by
silence. One empirical test then confirms the composed distribution really is the one drawn
from, so the composition is not a fiction.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.engine.sample import BatchSamplingArgs, Sampler
from freetoken.engine.spec_sample import filtered_probs
from freetoken.utils import is_sm90_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the server sampling backends are GPU kernels"
)

VOCAB = 4096
# The threshold search brackets top-p to a bin centre rather than to an exact token boundary,
# so the two filters can disagree by a token or two of tail mass. That is bounded and
# distribution-shaped; a filter-ORDER disagreement is not, and lands past 0.1.
MAX_TOTAL_VARIATION = 0.02


def _backend_params():
    from freetoken.kernel.backend import is_flashinfer_installed

    return [
        pytest.param("triton", id="triton"),
        pytest.param(
            "flashinfer",
            id="flashinfer",
            marks=pytest.mark.skipif(
                not is_flashinfer_installed(),
                reason="flashinfer is not installed in this venv",
            ),
        ),
    ]


@pytest.fixture
def backend(request, monkeypatch):
    """Force ``sample_impl``'s dispatch, and hand back the module it will pick."""
    import freetoken.kernel.backend as probe

    name = request.param
    monkeypatch.setattr(probe, "is_flashinfer_installed", lambda: name == "flashinfer")
    if name == "flashinfer":
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling
    return sampling


def _logit_rows(rows=2):
    """LM-head-shaped logits: a Gaussian body with a dragged-down tail, which is what makes a
    top-k cut land among tokens that still carry real mass."""
    generator = torch.Generator().manual_seed(20260901)
    row = (
        torch.randn(VOCAB, generator=generator) * 2.0
        - 4.0 * torch.rand(VOCAB, generator=generator)
    )
    return row.cuda().expand(rows, VOCAB).contiguous()


def _args(temperature, top_k, top_p, rows):
    full = lambda value, dtype: torch.full((rows,), value, dtype=dtype, device="cuda")
    return BatchSamplingArgs(
        temperatures=full(temperature, torch.float32),
        top_k=None if top_k is None else full(top_k, torch.int32),
        top_p=None if top_p is None else full(top_p, torch.float32),
    )


def _server_probs(sampling, logits, args):
    """What ``sample_impl`` leaves for its inverse-CDF draw, built from the backend's own ops."""
    probs = sampling.softmax(logits.float(), args.temperatures, enable_pdl=is_sm90_supported())
    if args.top_k is not None:
        probs = sampling.top_k_renorm_probs(probs, args.top_k)
    if args.top_p is not None:
        probs = sampling.top_p_renorm_probs(probs, args.top_p)
    return probs / probs.sum(-1, keepdim=True)


CASES = [
    (1.0, None, None),
    (0.7, 8, None),
    (1.0, 40, None),
    (1.0, None, 0.9),
    (0.8, 16, 0.9),
    (1.3, 32, 0.95),
]
CASE_IDS = ["plain", "top-k-8", "top-k-40", "top-p", "top-k-top-p", "hot-top-k-top-p"]


@pytest.mark.parametrize("backend", _backend_params(), indirect=True)
@pytest.mark.parametrize("temperature,top_k,top_p", CASES, ids=CASE_IDS)
def test_the_server_and_the_acceptance_filter_define_the_same_distribution(
    backend, temperature, top_k, top_p
):
    logits = _logit_rows()
    args = _args(temperature, top_k, top_p, logits.shape[0])

    server = _server_probs(backend, logits, args)
    acceptance = filtered_probs(logits, args)

    assert float(0.5 * (server - acceptance).abs().sum(-1).max()) < MAX_TOTAL_VARIATION


@pytest.mark.parametrize("backend", _backend_params(), indirect=True)
def test_the_greedy_fast_path_is_the_filters_one_hot_row(backend):
    logits = _logit_rows()
    args = BatchSamplingArgs(temperatures=None)  # Sampler.sample short-circuits to argmax

    drawn = Sampler(device=torch.device("cuda"), vocab_size=VOCAB).sample(logits, args)
    acceptance = filtered_probs(logits, args)

    assert drawn.tolist() == acceptance.argmax(dim=-1).tolist()
    assert acceptance.sum(-1).tolist() == [1.0] * logits.shape[0]


@pytest.mark.parametrize("backend", _backend_params(), indirect=True)
def test_the_composed_filter_is_the_one_the_server_actually_draws_from(backend):
    """The composition above is only a claim about ``sample_impl``'s internals until a real
    draw agrees with it. top-k 8 keeps the histogram tight enough for 16k draws to settle."""
    rows, calls = 8192, 2
    logits = _logit_rows(rows)
    args = _args(1.0, 8, None, rows)
    sampler = Sampler(device=torch.device("cuda"), vocab_size=VOCAB)

    counts = torch.zeros(VOCAB, device="cuda")
    for _ in range(calls):
        drawn = sampler.sample(logits, args)
        counts.index_add_(0, drawn.long(), torch.ones(rows, device="cuda"))

    expected = filtered_probs(logits[:1], args)[0]
    empirical = counts / counts.sum()
    # a token the filter zeroed must never come back; the converse is only statistical
    assert not bool(((expected == 0) & (empirical > 0)).any())
    assert float(0.5 * (empirical - expected).abs().sum()) < 0.03


# ------------------------------------------------------ where the equality stops holding


def test_the_triton_top_k_pivot_bounds_where_the_two_filters_agree():
    """The triton top-k kernel is an approximation, and its precondition is worth pinning.

    It gathers candidates at ``probs >= max_prob * _FRAC`` in one full-vocab pass and searches
    the k-th value inside that buffer; when FEWER THAN k tokens clear the pivot it cannot find
    a k-th value and deliberately keeps the whole row. So on a distribution peaked enough that
    the k-th token carries under 5 % of the mode's mass, the server applies NO top-k while
    ``filtered_probs`` applies one -- the acceptance filter would be strictly narrower than the
    distribution the server samples from, and speculation would shift client-visible output.

    Not reachable by default (``SamplingParams.top_k`` defaults to -1, which ``Sampler.prepare``
    drops), and flashinfer, if ever installed, has no such pivot. Phase 6's live gate runs
    greedy, where neither filter is consulted at all.
    """
    import freetoken.kernel.triton.sampling as sampling

    top_k = 8
    device = torch.device("cuda")
    # geometric probabilities: token i carries ratio**i of the mode, so exactly
    # floor(log(_FRAC)/log(ratio)) + 1 tokens clear the pivot
    ratio = 0.65
    row = (torch.arange(256, dtype=torch.float32, device=device) * torch.log(
        torch.tensor(ratio, device=device)
    ))
    args = _args(1.0, top_k, None, 1)
    clearing = int((torch.softmax(row, -1) >= torch.softmax(row, -1).max() * sampling._FRAC).sum())
    assert clearing < top_k  # the regime this test is about

    probs = sampling.softmax(row.unsqueeze(0), args.temperatures)
    server = sampling.top_k_renorm_probs(probs, args.top_k)
    acceptance = filtered_probs(row.unsqueeze(0), args)

    assert int((acceptance[0] > 0).sum()) == top_k
    assert int((server[0] > 0).sum()) > top_k          # the cut simply did not happen
    assert float(0.5 * (server - acceptance).abs().sum()) > MAX_TOTAL_VARIATION

    # ... and it does happen as soon as k tokens clear the pivot
    shallow = torch.zeros(1, 256, device=device)
    shallow[0, :40] = torch.linspace(4.0, 3.0, 40, device=device)
    shallow_probs = sampling.softmax(shallow, args.temperatures)
    assert int((shallow_probs[0] >= shallow_probs.max() * sampling._FRAC).sum()) >= top_k
    shallow_server = sampling.top_k_renorm_probs(shallow_probs, args.top_k)

    assert int((shallow_server[0] > 0).sum()) == top_k
    assert torch.allclose(shallow_server, filtered_probs(shallow, args), atol=1e-6)

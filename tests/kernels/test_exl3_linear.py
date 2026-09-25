"""Dense EXL3 linear (spec 2026-09-25 section 2). CPU tests stub the wheel with the
pure-torch reconstruction oracle; the GPU tests at the bottom run on the 5090."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel import exl3 as exl3_kernel
from freetoken.kernel import exl3_linear as el


def _parts(fin, fout, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    trellis = torch.randint(-32768, 32767, (fin // 16, fout // 16, 16 * k), generator=g,
                            dtype=torch.int32).to(torch.int16)
    suh = (torch.randint(0, 2, (fin,), generator=g) * 2 - 1).half()
    svh = (torch.rand(fout, generator=g) * 0.02 + 0.01).half()
    return {"trellis": trellis, "suh": suh, "svh": svh, "mul1": torch.tensor(0, dtype=torch.int32)}


def _ref(x, p, k):
    w = exl3_kernel.reconstruct_reference(p["trellis"], p["suh"], p["svh"], k=k, codebook="mul1")
    return F.linear(x.float(), w.float())


@pytest.fixture
def cpu_wheel(monkeypatch):
    """Route the op's two wheel seams to the CPU oracle."""
    def fake_gemm(x16, trellis, y16, suh, xh, svh):
        k = trellis.shape[-1] // 16
        w = exl3_kernel.reconstruct_reference(trellis, suh, svh, k=k, codebook="mul1")
        y16.copy_(F.linear(x16.float(), w.float()).half())

    def fake_reconstruct(op, out, work):
        out.copy_(exl3_kernel.reconstruct_reference(op.trellis, op.suh, op.svh, k=op.k, codebook="mul1"))
        return out

    monkeypatch.setattr(el, "_exl3_gemm", fake_gemm)
    monkeypatch.setattr(el, "_reconstruct_into", fake_reconstruct)


def _loaded(fin, fout, k, *, bias=False, k_hint=5, **kw):
    op = el.Exl3Linear(fin, fout, has_bias=bias, k_hint=k_hint, **kw)
    state = {f"p.{n}": t for n, t in _parts(fin, fout, k).items()}
    if bias:
        state["p.bias"] = torch.randn(fout).bfloat16()
    # load_state_dict pops consumed keys from its argument by design (top-level calls must
    # end with an empty dict to catch unexpected keys) - pass a copy so the caller's `state`
    # survives for post-hoc comparison, matching the dict(state) copy already used below in
    # test_col_merged_concatenates_in_part_order.
    op.load_state_dict(dict(state), prefix="p")
    return op, state


def test_load_adopts_checkpoint_k():
    op, _ = _loaded(128, 256, 3, k_hint=5)
    assert op.k == 3 and op.trellis.shape == (8, 16, 48)


@pytest.mark.parametrize("bad,match", [
    (lambda s: s.update({"p.trellis": s["p.trellis"][:, :, :40]}), "K"),
    (lambda s: s.update({"p.suh": s["p.suh"].float()}), "suh"),
    (lambda s: s.update({"p.svh": s["p.svh"][:-16]}), "svh"),
    (lambda s: s.pop("p.mul1"), "mul1"),
])
def test_load_rejects_malformed_parts(bad, match):
    op = el.Exl3Linear(128, 256)
    state = {f"p.{n}": t for n, t in _parts(128, 256, 3).items()}
    bad(state)
    with pytest.raises((ValueError, KeyError), match=match):
        op.load_state_dict(state, prefix="p")


def test_load_requires_bias_key_when_op_has_one():
    op = el.Exl3Linear(128, 256, has_bias=True)
    state = {f"p.{n}": t for n, t in _parts(128, 256, 3).items()}  # no "p.bias"
    with pytest.raises(KeyError, match="bias"):
        op.load_state_dict(dict(state), prefix="p")


@pytest.mark.parametrize("bad_bias", [
    torch.randn(256, 1).half(),  # not 1-D
    torch.randn(255).half(),     # wrong length
])
def test_load_rejects_malformed_bias(bad_bias):
    op = el.Exl3Linear(128, 256, has_bias=True)
    state = {f"p.{n}": t for n, t in _parts(128, 256, 3).items()}
    state["p.bias"] = bad_bias
    with pytest.raises(ValueError, match="bias"):
        op.load_state_dict(dict(state), prefix="p")


def test_load_casts_bias_to_bf16_working_dtype():
    # The vision tower's packed linears ship fp16 bias (3.05bpw_h5_ng5 headers); the module's
    # compute dtype is bf16 throughout forward(), so load_state_dict must cast on load.
    op = el.Exl3Linear(128, 256, has_bias=True)
    state = {f"p.{n}": t for n, t in _parts(128, 256, 3).items()}
    state["p.bias"] = torch.randn(256).half()
    op.load_state_dict(dict(state), prefix="p")
    assert op.bias.dtype == torch.bfloat16


@pytest.mark.parametrize("rows", [1, 2, 144, 145, 300])
def test_forward_matches_reference(cpu_wheel, rows):
    op, state = _loaded(128, 256, 3, bias=True)
    el.prepare_exl3_dense_workspace(op, torch.device("cpu"), _reset=True)
    x = torch.randn(rows, 128).bfloat16()
    p = {n: state[f"p.{n}"] for n in ("trellis", "suh", "svh")}
    want = _ref(x, p, 3) + state["p.bias"].float()
    got = op.forward(x).float()
    assert got.shape == (rows, 256)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


def test_forward_keeps_leading_dims(cpu_wheel):
    op, _ = _loaded(128, 256, 3)
    el.prepare_exl3_dense_workspace(op, torch.device("cpu"), _reset=True)
    assert op.forward(torch.randn(2, 4, 128).bfloat16()).shape == (2, 4, 256)


def test_col_merged_concatenates_in_part_order(cpu_wheel):
    op = el.Exl3ColMerged(128, [("q_proj", 256), ("k_proj", 128)])
    state = {}
    for name, fout in (("q_proj", 256), ("k_proj", 128)):
        state.update({f"m.{name}.{n}": t for n, t in _parts(128, fout, 3, seed=fout).items()})
    op.load_state_dict(dict(state), prefix="m")
    el.prepare_exl3_dense_workspace(op, torch.device("cpu"), _reset=True)
    x = torch.randn(3, 128).bfloat16()
    q = _ref(x, {n: state[f"m.q_proj.{n}"] for n in ("trellis", "suh", "svh")}, 3)
    k = _ref(x, {n: state[f"m.k_proj.{n}"] for n in ("trellis", "suh", "svh")}, 3)
    torch.testing.assert_close(op.forward(x).float(), torch.cat([q, k], -1), rtol=2e-2, atol=2e-2)


def test_lm_head_never_reconstructs(cpu_wheel, monkeypatch):
    head = el.Exl3LMHead(num_embeddings=256, embedding_dim=128)
    head.load_state_dict({f"lm_head.{n}": t for n, t in _parts(128, 256, 5).items()}, prefix="lm_head")
    el.prepare_exl3_dense_workspace(head, torch.device("cpu"), _reset=True)
    monkeypatch.setattr(el, "_reconstruct_into", lambda *a, **k: pytest.fail("lm_head reconstructed"))
    assert head.forward_all(torch.randn(300, 128).bfloat16()).shape == (300, 256)


def test_forward_without_workspace_is_a_clear_error(cpu_wheel):
    op, _ = _loaded(128, 256, 3)
    el._WORKSPACES.clear()
    with pytest.raises(RuntimeError, match="prepare_exl3_dense_workspace"):
        op.forward(torch.randn(1, 128).bfloat16())


def test_workspace_never_grows_after_prepare(cpu_wheel):
    small, _ = _loaded(128, 256, 3)
    el.prepare_exl3_dense_workspace(small, torch.device("cpu"), _reset=True)
    big, _ = _loaded(256, 512, 3)
    with pytest.raises(RuntimeError, match="workspace"):
        el.require_exl3_dense_workspace_fits(big, torch.device("cpu"))


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the 5090")


@cuda
@pytest.mark.parametrize("fin,fout,k", [(2560, 12288, 5), (6144, 2560, 5), (2560, 640, 3),
                                         (2560, 2560, 4), (1152, 4352, 5)])
@pytest.mark.parametrize("rows", [1, 2, 8, 144, 145, 4096])
def test_gpu_matches_card_reconstruction(fin, fout, k, rows):
    dev = torch.device("cuda")
    op = el.Exl3Linear(fin, fout, k_hint=k)
    op.load_state_dict({f"p.{n}": t.to(dev) for n, t in _parts(fin, fout, k).items()}, prefix="p")
    el.prepare_exl3_dense_workspace(op, dev, _reset=True)
    x = torch.randn(rows, fin, device=dev).bfloat16()
    w = exl3_kernel.reconstruct(op.trellis, op.suh, op.svh, k=k, codebook="mul1")
    want = F.linear(x.float(), w.float())
    got = op.forward(x).float()
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-2, rel


@cuda
def test_graph_replay_two_linears():
    dev = torch.device("cuda")
    a = el.Exl3Linear(2560, 640, k_hint=5)
    b = el.Exl3Linear(640, 2560, k_hint=5)
    a.load_state_dict({f"a.{n}": t.to(dev) for n, t in _parts(2560, 640, 5, 1).items()}, prefix="a")
    b.load_state_dict({f"b.{n}": t.to(dev) for n, t in _parts(640, 2560, 5, 2).items()}, prefix="b")

    class Both(el.BaseOP):
        def __init__(self):
            self.a, self.b = a, b

    el.prepare_exl3_dense_workspace(Both(), dev, _reset=True)
    x = torch.randn(1, 2560, device=dev).bfloat16()
    eager_a = a.forward(x); eager = b.forward(eager_a) + eager_a.sum()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_a = a.forward(x)
        out = b.forward(out_a) + out_a.sum()
    g.replay(); torch.cuda.synchronize()
    torch.testing.assert_close(out, eager)

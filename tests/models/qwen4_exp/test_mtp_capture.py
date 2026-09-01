from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead
from freetoken.layers.embedding import ParallelLMHead
from freetoken.models.qwen4_exp.model import Qwen4ExpModel


class _Embedding:
    def forward(self, ids):
        return torch.stack((ids.float(), ids.float() + 1, ids.float() + 2), dim=-1)


class _Layer:
    ple = None

    def forward(self, hidden, batch):
        del batch
        return hidden + 0.5


class _Mixer:
    def mix(self, hidden):
        streams = hidden.view(hidden.shape[0], 2, 3)
        return streams.mean(1), None


def _model():
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    model.hc_count = 2
    model.embed_tokens = _Embedding()
    model.layers = SimpleNamespace(op_list=[_Layer()])
    model.hyper_connection_mixer = _Mixer()
    model._ple = ()
    model._image_token_id = 99
    return model


def test_default_path_is_unchanged_and_capture_returns_real_internal_rows():
    ids = torch.tensor([4, 5, 6])
    batch = SimpleNamespace(mm_embeds=None)
    model = _model()
    ordinary = model.forward(ids, batch)
    captured_final, multi, merged = model.forward_mtp_capture(ids, batch)

    assert isinstance(ordinary, torch.Tensor)
    torch.testing.assert_close(captured_final, ordinary, rtol=0, atol=0)
    assert multi.shape == (3, 6)
    assert merged.shape == (3, 3)
    torch.testing.assert_close(merged, _Embedding().forward(ids), rtol=0, atol=0)
    torch.testing.assert_close(multi, merged.repeat(1, 2) + 0.5, rtol=0, atol=0)


def test_capture_contains_actual_picture_soft_embedding():
    ids = torch.tensor([4, 99, 6])
    picture = torch.tensor([[10.0, 11.0, 12.0]])
    batch = SimpleNamespace(mm_embeds=picture)
    _, multi, merged = _model().forward_mtp_capture(ids, batch)
    assert torch.equal(merged[1], picture[0])
    assert torch.equal(multi[1], picture[0].repeat(2) + 0.5)


def test_parallel_lm_head_forward_all_projects_every_row():
    head = ParallelLMHead.__new__(ParallelLMHead)
    head.weight = torch.arange(20, dtype=torch.float32).view(5, 4)
    head.bias = None
    head.tied_embedding = None
    head.tp_size = 1
    head.num_embeddings = 5
    rows = torch.arange(12, dtype=torch.float32).view(3, 4)
    got = head.forward_all(rows)
    assert got.shape == (3, 5)
    torch.testing.assert_close(got, rows @ head.weight.T)


def test_nvfp4_lm_head_forward_all_does_not_slice_rows(monkeypatch):
    head = Nvfp4LMHead.__new__(Nvfp4LMHead)
    head._transposed = False
    head.weight = torch.empty(5, 2, dtype=torch.uint8)
    head.weight_scale = torch.empty(5, 1, dtype=torch.float8_e4m3fn)
    head.weight_global = torch.ones(5, dtype=torch.float16)

    def fake_linear(x, weight, scale, global_scale):
        del weight, scale, global_scale
        return x.sum(-1, keepdim=True).expand(-1, 5)

    monkeypatch.setattr(
        "freetoken.kernel.triton.nvfp4_linear.nvfp4_dense_linear", fake_linear
    )
    rows = torch.arange(12, dtype=torch.bfloat16).view(3, 4)
    got = head.forward_all(rows)
    assert got.shape == (3, 5)
    assert torch.equal(got[:, 0], rows.sum(-1))

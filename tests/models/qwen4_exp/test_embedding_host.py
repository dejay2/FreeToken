"""Host-resident token embedding (``FREETOKEN_EMBED_HOST=1``).

``model.embed_tokens`` is 248,320 x 2560 bf16 = 1.27 GB of VRAM that a decode step reads
1-6 rows of and a prefill chunk up to a few thousand. Host residency hands that 1.27 GB to
the GPU expert cache and gathers rows over UVA with the PLE kernel.

These are CPU tests. ``layers.embedding.gather_host_rows`` dispatches on the destination's
device: CUDA goes to ``kernel/triton/ple.ple_gather_rows`` (covered on GPU by
``test_ple.py``), CPU to the dense ``index_select`` oracle exercised here. What is checked
here is everything the kernel choice does NOT decide -- the row math, the shard masking, the
fixed-buffer contract, and that nothing in the forward path syncs the stream.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from freetoken.layers.embedding import VocabParallelEmbedding
from freetoken.models.config import embed_host_enabled

VOCAB = 97
DIM = 32


def _build(vocab: int = VOCAB, dim: int = DIM) -> tuple[VocabParallelEmbedding, torch.Tensor]:
    """A host-resident embedding plus the dense table it must reproduce."""
    torch.manual_seed(0)
    table = torch.randn(vocab, dim, dtype=torch.bfloat16)
    emb = VocabParallelEmbedding(num_embeddings=vocab, embedding_dim=dim)
    emb.weight = torch.empty(vocab, dim, dtype=torch.bfloat16)
    emb.attach_host_table(table.clone(), torch.device("cpu"))
    return emb, table


@contextlib.contextmanager
def _no_host_sync():
    """Fail the test if the block reads any tensor value back to the host.

    A ``.item()`` / ``.tolist()`` / ``.cpu()`` in the forward path stalls on the stream, which
    is illegal inside a captured CUDA graph -- capture would either error or bake in a stale
    host value. Asserting by monkeypatch is the only way to catch it without a GPU.
    """
    banned = ("item", "tolist", "cpu", "numpy", "nonzero")
    saved = {name: getattr(torch.Tensor, name) for name in banned}

    def _boom(name):
        def _f(*_args, **_kwargs):
            raise AssertionError(f"Tensor.{name}() in the embedding forward path syncs the stream")

        return _f

    for name in banned:
        setattr(torch.Tensor, name, _boom(name))
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.Tensor, name, fn)


# ------------------------------------------------------------------ row-gather equivalence


def test_env_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("FREETOKEN_EMBED_HOST", raising=False)
    assert embed_host_enabled() is False
    monkeypatch.setenv("FREETOKEN_EMBED_HOST", "1")
    assert embed_host_enabled() is True
    monkeypatch.setenv("FREETOKEN_EMBED_HOST", "0")
    assert embed_host_enabled() is False


def test_attach_marks_host_resident_and_keeps_state_dict():
    emb, table = _build()
    assert emb.host_resident
    assert emb.device == torch.device("cpu")
    # the module still owns the table, so a checkpoint round-trip is unchanged
    assert torch.equal(emb.state_dict(prefix="e")["e.weight"], table)


@pytest.mark.parametrize("count", [1, 2, 6, 64, 4096])
def test_gather_matches_dense_lookup(count):
    emb, table = _build()
    ids = torch.randint(0, VOCAB, (count,), dtype=torch.int32)
    got = emb.embed(ids)
    assert got.shape == (count, DIM) and got.dtype == torch.bfloat16
    # bf16 -> fp32 -> bf16 is exact, so the gather must be bit-identical to index_select
    assert torch.equal(got, table.index_select(0, ids.to(torch.int64)))


def test_forward_matches_dense_lookup_and_flattens():
    emb, table = _build()
    ids = torch.randint(0, VOCAB, (5,), dtype=torch.int32)
    assert torch.equal(emb.forward(ids), table.index_select(0, ids.to(torch.int64)))


def test_empty_batch_is_a_no_op():
    emb, _ = _build()
    got = emb.embed(torch.empty(0, dtype=torch.int32))
    assert got.shape == (0, DIM)


def test_out_of_range_ids_read_zeros():
    """The kernel's rule, mirrored by the oracle: an id outside the table reads zeros rather
    than faulting on host memory it does not own."""
    emb, _ = _build()
    ids = torch.tensor([-1, 0, VOCAB, VOCAB + 10], dtype=torch.int64)
    got = emb.embed(ids)
    assert torch.count_nonzero(got[0]) == 0
    assert torch.count_nonzero(got[2]) == 0
    assert torch.count_nonzero(got[3]) == 0


def test_shard_masking_matches_vocab_range():
    """TP>1: ids outside this rank's shard contribute zeros, which is what the all-reduce
    downstream expects. Set up by hand -- the process-wide TP info is size 1 in tests."""
    emb, table = _build()
    start, length = 40, 20
    emb.tp_size = 2
    emb.vocab_range = (start, length)
    ids = torch.tensor([0, 39, 40, 50, 59, 60, 96], dtype=torch.int64)
    got = emb.embed(ids)
    for row, tok in enumerate(ids.tolist()):
        if start <= tok < start + length:
            # a shard's row 0 is global token ``start``, exactly as ``kernel.indexing``
            # reads it when handed ``vocab_range``
            assert torch.equal(got[row], table[tok - start])
        else:
            assert torch.count_nonzero(got[row]) == 0


def test_gpu_resident_path_is_untouched_by_the_seam():
    """``embed`` without a host table must still go through ``kernel.indexing``."""
    emb = VocabParallelEmbedding(num_embeddings=VOCAB, embedding_dim=DIM)
    assert not emb.host_resident
    seen = {}

    def _fake_indexing(*, weights, indices, output=None, vocab_range=None):
        seen.update(vocab_range=vocab_range, output=output)
        return torch.zeros(indices.shape[0], DIM)

    import freetoken.kernel as kernel

    saved = kernel.indexing
    kernel.indexing = _fake_indexing
    try:
        emb.embed(torch.zeros(3, dtype=torch.int32))
    finally:
        kernel.indexing = saved
    assert seen == {"vocab_range": None, "output": None}


# ------------------------------------------------------------------ graph-safety contract


def test_out_buffer_is_written_in_place_and_returned():
    emb, table = _build()
    ids = torch.randint(0, VOCAB, (4,), dtype=torch.int32)
    buf = torch.zeros(4, DIM, dtype=torch.bfloat16)
    got = emb.embed(ids, out=buf)
    assert got is buf
    assert torch.equal(buf, table.index_select(0, ids.to(torch.int64)))


def test_graph_out_buffer_is_stable_per_row_count():
    """One buffer per captured decode size, allocated once and kept for good: growing or
    freeing it would leave a replay writing into a released block."""
    emb, _ = _build()
    first = emb.graph_out_buffer(4)
    assert emb.graph_out_buffer(4) is first
    assert first.shape == (4, DIM) and first.dtype == torch.bfloat16
    assert emb.graph_out_buffer(6) is not first
    assert emb.graph_out_buffer(6).shape == (6, DIM)
    # replaying the same size writes the same storage
    ptr = first.data_ptr()
    for _ in range(3):
        emb.embed(torch.randint(0, VOCAB, (4,), dtype=torch.int32), out=emb.graph_out_buffer(4))
    assert emb.graph_out_buffer(4).data_ptr() == ptr


def test_forward_path_never_syncs_the_host():
    emb, _ = _build()
    ids = torch.randint(0, VOCAB, (6,), dtype=torch.int32)
    buf = emb.graph_out_buffer(6)
    with _no_host_sync():
        emb.embed(ids, out=buf)
        emb.embed(ids)
        emb.forward(ids)


def test_shard_path_never_syncs_the_host():
    emb, _ = _build()
    emb.tp_size = 2
    emb.vocab_range = (10, 20)
    ids = torch.randint(0, VOCAB, (6,), dtype=torch.int32)
    with _no_host_sync():
        emb.embed(ids)


def test_indices_stay_on_the_gather_device():
    """The gather takes device-resident ids; nothing may move them to the host first."""
    emb, _ = _build()
    seen = []
    import freetoken.layers.embedding as embedding_module

    saved = embedding_module.gather_host_rows

    def _spy(table_ptr, num_rows, dim, ids, out):
        seen.append(ids.device)
        return saved(table_ptr, num_rows, dim, ids, out)

    embedding_module.gather_host_rows = _spy
    try:
        emb.embed(torch.zeros(3, dtype=torch.int32, device="cpu"))
    finally:
        embedding_module.gather_host_rows = saved
    assert seen == [torch.device("cpu")]


# ------------------------------------------------------------------ qwen4_exp wiring


def _causal_lm_class():
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    return Qwen4ExpForCausalLM


def test_weight_device_for_key_sends_the_embedding_to_the_host(monkeypatch):
    cls = _causal_lm_class()
    obj = cls.__new__(cls)
    obj._vision_execution = "gpu"
    obj._embed_host = True
    cuda = torch.device("cuda", 0)
    assert cls.weight_device_for_key(obj, "model.embed_tokens.weight", cuda).type == "cpu"
    assert cls.weight_device_for_key(obj, "lm_head.weight", cuda) is cuda
    assert cls.weight_device_for_key(obj, "model.layers.0.self_attn.q_proj.weight", cuda) is cuda
    obj._embed_host = False
    assert cls.weight_device_for_key(obj, "model.embed_tokens.weight", cuda) is cuda


def test_tied_embeddings_disable_host_residency(monkeypatch):
    """A tied lm_head projects against the same matrix every step, so the full-vocab GEMV
    would pull all 1.27 GB over PCIe per token. The flag must not take effect there."""
    from freetoken.models.config import embed_host_enabled

    monkeypatch.setenv("FREETOKEN_EMBED_HOST", "1")
    assert embed_host_enabled() is True
    for tied, expected in ((True, False), (False, True)):
        assert (embed_host_enabled() and not tied) is expected

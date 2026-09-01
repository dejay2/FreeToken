from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_ceil, nvtx_annotate

from .base import BaseOP

# Rows are 2560 bf16 = 5 KB; 8 warps keeps a row ~16 registers deep per lane instead of the
# 128 a single warp would carry. Latency over PCIe still dominates, so this is not tuned hard.
_HOST_GATHER_WARPS = 8


def gather_host_rows(
    table_ptr: int,
    num_rows: int,
    embed_dim: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Gather ``row_ids`` from a pinned host bf16 table into the device buffer ``out``.

    On CUDA this is the PLE UVA gather (``kernel/triton/ple.ple_gather_rows``) reused
    verbatim with ``is_fp8=False`` and ``scale=1.0``. On CPU it is the dense ``index_select``
    oracle the unit tests check the kernel's contract against -- same out-of-range rule (any
    id outside ``[0, num_rows)`` reads zeros), same destination-is-the-return-value rule.
    """
    n = row_ids.numel()
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n == 0:
        return out
    if out.is_cuda:
        from freetoken.kernel.triton.ple import ple_gather_rows

        return ple_gather_rows(
            table_ptr,
            num_rows,
            embed_dim,
            row_ids,
            out,
            1.0,
            False,
            num_warps=_HOST_GATHER_WARPS,
        )
    # CPU oracle: ctypes-free view of the same host storage the device would dereference.
    table = _cpu_table_for_ptr(table_ptr)
    ids = row_ids.reshape(-1).to(torch.int64)
    in_range = (ids >= 0) & (ids < num_rows)
    safe = torch.where(in_range, ids, torch.zeros_like(ids))
    rows = table.index_select(0, safe).to(out.dtype)
    out.copy_(torch.where(in_range.unsqueeze(-1), rows, torch.zeros_like(rows)))
    return out


# data_ptr -> host tensor, so the CPU oracle can resolve the same "address" the GPU takes.
_CPU_TABLES: Dict[int, torch.Tensor] = {}


def _cpu_table_for_ptr(table_ptr: int) -> torch.Tensor:
    table = _CPU_TABLES.get(table_ptr)
    if table is None:
        raise RuntimeError("no host embedding table registered for this address")
    return table


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float | None = None,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        # Gemma scales embeddings by sqrt(hidden_size). The scale is materialized in
        # the weight dtype (bf16) to match HF, which downcasts the scalar. The GPU
        # scalar is built lazily (model __init__ runs on the meta device) and cached
        # so it is not reallocated inside a captured CUDA graph.
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None
        self._comm = DistributedCommunicator()
        # Host-resident mode (FREETOKEN_EMBED_HOST=1): the table stays pinned on the host and
        # rows are gathered over UVA. None keeps the historical GPU-resident behavior.
        self._host_ptr: int | None = None
        self._host_device: torch.device | None = None
        self._graph_out: Dict[int, torch.Tensor] = {}

    # ---------------------------------------------------------------- host residency

    @property
    def host_resident(self) -> bool:
        return self._host_ptr is not None

    @property
    def device(self) -> torch.device:
        """The device rows are produced on -- NOT ``self.weight.device`` once the table is
        host-resident. Callers that used the weight's device to find "where the language
        model runs" must read this instead."""
        if self._host_device is not None:
            return self._host_device
        return self.weight.device

    def attach_host_table(self, weight: torch.Tensor, device: torch.device) -> int:
        """Adopt ``weight`` (pinned, device-mapped, CPU, this module's exact shape/dtype) as
        the row store and return its byte size.

        The caller owns pinning: the gather dereferences host memory from the GPU, so an
        unregistered buffer faults. ``self.weight`` keeps pointing at the same storage so
        ``state_dict`` round-trips unchanged.
        """
        assert weight.device.type == "cpu" and weight.is_contiguous(), weight.device
        assert weight.shape == (self.num_embeddings_tp, self.weight.shape[1]), weight.shape
        assert weight.dtype == torch.bfloat16, weight.dtype
        if device.type == "cuda":
            from freetoken.kernel.pinned import device_ptr

            ptr = device_ptr(weight)
        else:  # CPU oracle path (unit tests): the host address is the address.
            ptr = weight.data_ptr()
            _CPU_TABLES[ptr] = weight
        self.weight = weight
        self._host_ptr = ptr
        self._host_device = device
        self._graph_out.clear()
        return weight.numel() * weight.element_size()

    def graph_out_buffer(self, rows: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        """A fixed ``[rows, embedding_dim]`` destination, one per row count, allocated once.

        Captured decode replays write to a stable address this way. Kept for the life of the
        module: growing or freeing a buffer a replay still writes to is a use-after-free.
        """
        buf = self._graph_out.get(rows)
        if buf is None:
            buf = torch.empty(
                (rows, self.weight.shape[1]),
                dtype=dtype or self.weight.dtype,
                device=self.device,
            )
            self._graph_out[rows] = buf
        return buf

    def embed(self, x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        """Row gather seam shared by the target forward and the MTP draft head.

        ``x`` is a device tensor of token ids and stays on device: nothing here reads a value
        back to the host, so the call is safe inside a captured CUDA graph. ``out``, when
        given, is the destination and is returned as-is (fixed-buffer decode); otherwise a
        fresh buffer is allocated -- under capture that allocation comes from the graph's
        private pool, so its address is baked into the replay exactly as before.
        """
        if self._host_ptr is None:
            from freetoken.kernel import indexing

            return indexing(
                weights=self.weight,
                indices=x,
                output=out,
                vocab_range=self.vocab_range if self.tp_size > 1 else None,
            )
        dim = self.weight.shape[1]
        ids = x.reshape(-1)
        if self.tp_size > 1:
            # Shift into shard-local rows; the gather zeroes anything outside [0, len),
            # which is exactly what ``vocab_range`` masking does before the all-reduce.
            start, length = self.vocab_range
            ids = ids.to(torch.int64) - start
        else:
            length = self.num_embeddings_tp
        if out is None:
            out = torch.empty((ids.numel(), dim), dtype=self.weight.dtype, device=self.device)
        return gather_host_rows(self._host_ptr, length, dim, ids, out)

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embed(x)

        if self.tp_size > 1:
            y = self._comm.all_reduce(y)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(
                    self._embed_scale, dtype=y.dtype, device=y.device
                )
            y = y * self._embed_scale_t
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)
        if input_shape[0] == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Private teacher seam: project every input row without prefill last-row slicing."""
        return self._project(x)

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
            del indices
        return self._project(x)
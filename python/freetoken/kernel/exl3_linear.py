"""Dense EXL3 linears kept packed at runtime (spec 2026-09-25 section 2).

Small row counts run ExLlamaV3's ``exl3_gemm`` straight off the trellis; prompts above
GEMM_MAX_ROWS reconstruct the weight into a shared bf16 scratch and run a normal GEMM, the
same split ExLlamaV3 makes (modules/quant/exl3.py:10, AUTO_RECONSTRUCT_THRESHOLD = 144).
The wheel's own C++ wrapper allocates its Hadamard buffer per call when rows > 1
(exllamav3_ext/libtorch/linear.cpp:41-43), which a CUDA graph cannot hold, so this module
owns fixed fp16 input / Hadamard / output buffers, allocated once after weight load.
Known limit: the wheel's one-token GEMV only covers K 2-4 (exl3_gemv.cu:115-122); the K=5
dense layers of the 3.05bpw build take the general GEMM kernel.

The wheel routines are reached through two seams (_exl3_gemm, _reconstruct_into) so CPU tests
can substitute the pure-torch oracle in kernel/exl3.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F

from freetoken.kernel import exl3 as _exl3
from freetoken.layers.base import BaseOP, OPList, _concat_prefix

GEMM_MAX_ROWS = 144
_COMPONENTS = ("trellis", "suh", "svh", "mul1")


def _load_ext():
    try:
        import exllamav3_ext
    except ImportError as exc:  # pragma: no cover - depends on the optional wheel
        raise RuntimeError("dense EXL3 linears need the ExLlamaV3 exllamav3_ext wheel") from exc
    return exllamav3_ext


def _exl3_gemm(x16, trellis, y16, suh, xh, svh) -> None:
    # Signature verified on the 5090's 1.4.6 wheel in Task 0: exl3_gemm(A, B, C, suh, A_had,
    # svh, force_shape_idx, mcg, mul1, force_num_sms). Adapt here only if Task 0 differs.
    _load_ext().exl3_gemm(x16, trellis, y16, suh, xh, svh, -1, False, True, 0)


def _reconstruct_into(op: "Exl3Linear", out: torch.Tensor, work: torch.Tensor) -> torch.Tensor:
    return _exl3.reconstruct(op.trellis, op.suh, op.svh, k=op.k, codebook="mul1", out=out, work=work)


@dataclass
class Exl3DenseWorkspace:
    device: torch.device
    max_rows: int
    max_in: int
    max_out: int
    recon_elems: int
    x16: torch.Tensor
    xh: torch.Tensor
    y16: torch.Tensor
    recon_work: torch.Tensor
    recon_out: torch.Tensor


_WORKSPACES: dict[torch.device, Exl3DenseWorkspace] = {}


def _dev_key(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def validate_exl3_parts(name, trellis, suh, svh, mul1, in_features, out_features) -> int:
    if trellis.dtype != torch.int16 or trellis.dim() != 3:
        raise ValueError(f"{name}.trellis must be rank-3 int16, got {trellis.dtype} {tuple(trellis.shape)}")
    if trellis.shape[0] * 16 != in_features or trellis.shape[1] * 16 != out_features:
        raise ValueError(f"{name}.trellis {tuple(trellis.shape)} does not match [{in_features}, {out_features}]")
    last = int(trellis.shape[2])
    if last % 16 or not 1 <= last // 16 <= 8:
        raise ValueError(f"{name}.trellis last dim {last} is not 16*K for an integer K in 1..8")
    if suh.dtype != torch.float16 or tuple(suh.shape) != (in_features,):
        raise ValueError(f"{name}.suh must be fp16 [{in_features}], got {suh.dtype} {tuple(suh.shape)}")
    if svh.dtype != torch.float16 or tuple(svh.shape) != (out_features,):
        raise ValueError(f"{name}.svh must be fp16 [{out_features}], got {svh.dtype} {tuple(svh.shape)}")
    if mul1.dtype != torch.int32 or mul1.dim() != 0:
        raise ValueError(f"{name}.mul1 must be a scalar int32 marker, got {mul1.dtype} {tuple(mul1.shape)}")
    return last // 16


class Exl3Linear(BaseOP):
    def __init__(self, in_features: int, out_features: int, has_bias: bool = False, *,
                 k_hint: int = 5, allow_reconstruct: bool = True):
        if in_features % 16 or out_features % 128:
            raise ValueError(f"EXL3 linear needs in%16 == 0 and out%128 == 0, got [{in_features}, {out_features}]")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.k = int(k_hint)
        self.allow_reconstruct = bool(allow_reconstruct)
        self.trellis = torch.empty(in_features // 16, out_features // 16, 16 * self.k, dtype=torch.int16)
        self.suh = torch.empty(in_features, dtype=torch.float16)
        self.svh = torch.empty(out_features, dtype=torch.float16)
        self.mul1 = torch.empty((), dtype=torch.int32)
        # Explicit bf16 -- the module's working dtype (GEMM output, reconstructed weight) --
        # independent of the ambient default dtype and of whatever precision the checkpoint
        # ships the bias in (the vision tower's packed linears ship fp16, 3.05bpw_h5_ng5
        # headers; load_state_dict casts on load, see there). Keeping both the fresh and the
        # loaded tensor in the same dtype also keeps the layer-stream workspace copy
        # (_copy_component_state_'s dtype check) happy regardless of checkpoint precision.
        self.bias = torch.empty(out_features, dtype=torch.bfloat16) if has_bias else None

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        parts = {}
        for comp in _COMPONENTS:
            key = _concat_prefix(prefix, comp)
            if key not in state_dict:
                raise KeyError(f"EXL3 linear {prefix!r} is missing {comp} ({key})")
            parts[comp] = state_dict.pop(key)
        self.k = validate_exl3_parts(prefix, parts["trellis"], parts["suh"], parts["svh"],
                                     parts["mul1"], self.in_features, self.out_features)
        self.trellis = parts["trellis"].contiguous()
        self.suh = parts["suh"].contiguous()
        self.svh = parts["svh"].contiguous()
        self.mul1 = parts["mul1"]
        if self.bias is not None:
            key = _concat_prefix(prefix, "bias")
            if key not in state_dict:
                raise KeyError(f"EXL3 linear {prefix!r} is missing bias ({key})")
            bias = state_dict.pop(key)
            if bias.dim() != 1 or bias.shape[0] != self.out_features:
                raise ValueError(
                    f"EXL3 linear {prefix!r} bias {key!r} must be 1-D [{self.out_features}], "
                    f"got {tuple(bias.shape)}"
                )
            # Cast to the module's bf16 working dtype (see __init__): a checkpoint may ship
            # bias in another precision (the vision tower's packed linears ship fp16).
            self.bias = bias.to(torch.bfloat16)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    @property
    def resident_bytes(self) -> int:
        tensors = [self.trellis, self.suh, self.svh] + ([self.bias] if self.bias is not None else [])
        return sum(t.numel() * t.element_size() for t in tensors)

    def _workspace(self, device) -> Exl3DenseWorkspace:
        ws = _WORKSPACES.get(_dev_key(device))
        if ws is None:
            raise RuntimeError("dense EXL3 workspace missing: call prepare_exl3_dense_workspace after weight load")
        return ws

    def _gemm(self, x2: torch.Tensor) -> torch.Tensor:
        ws = self._workspace(x2.device)
        rows = x2.shape[0]
        x16 = ws.x16[: rows * self.in_features].view(rows, self.in_features)
        xh = ws.xh[: rows * self.in_features].view(rows, self.in_features)
        y16 = ws.y16[: rows * self.out_features].view(rows, self.out_features)
        x16.copy_(x2)
        _exl3_gemm(x16, self.trellis, y16, self.suh, xh, self.svh)
        # A fresh bf16 tensor: y16 is shared by every EXL3 linear on this card.
        return y16.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        rows = x2.shape[0]
        if rows > GEMM_MAX_ROWS and self.allow_reconstruct:
            ws = self._workspace(x2.device)
            n = self.in_features * self.out_features
            w = _reconstruct_into(
                self,
                ws.recon_out[:n].view(self.out_features, self.in_features),
                ws.recon_work[:n].view(self.in_features, self.out_features),
            )
            y = F.linear(x2.to(torch.bfloat16), w, self.bias)
        else:
            if rows <= GEMM_MAX_ROWS:
                y = self._gemm(x2)
            else:
                y = torch.cat([self._gemm(x2[i : i + GEMM_MAX_ROWS])
                               for i in range(0, rows, GEMM_MAX_ROWS)])
            if self.bias is not None:
                y = y + self.bias
        return y.view(*lead, self.out_features)


class Exl3ColMerged(BaseOP):
    """Several EXL3 linears on one input, outputs concatenated in ``parts`` order. Trellis
    tensors cannot be concatenated like bf16 weights, so a fused bf16 projection (q|k|v,
    GDN qkv|z, shared gate|up) becomes one GEMM per part."""

    def __init__(self, in_features: int, parts: list[tuple[str, int]], *, k_hint: int = 5):
        self._names = [name for name, _ in parts]
        self.out_features = sum(size for _, size in parts)
        for name, size in parts:
            setattr(self, name, Exl3Linear(in_features, size, k_hint=k_hint))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([getattr(self, name).forward(x) for name in self._names], dim=-1)


class Exl3LMHead(Exl3Linear):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(embedding_dim, num_embeddings, has_bias=False, k_hint=5,
                         allow_reconstruct=False)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_tp = num_embeddings
        self.vocab_range = (0, num_embeddings)
        self.tp_size = 1

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return super().forward(x)


def iter_exl3_linears(root) -> Iterator[Exl3Linear]:
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Exl3Linear):
            yield node
        if isinstance(node, OPList):
            stack.extend(node.op_list)
        if isinstance(node, BaseOP):
            for value in vars(node).values():
                if isinstance(value, BaseOP):
                    stack.append(value)
                elif isinstance(value, (list, tuple)):
                    stack.extend(v for v in value if isinstance(v, BaseOP))


def _need(root) -> tuple[int, int, int] | None:
    ops = list(iter_exl3_linears(root))
    if not ops:
        return None
    max_in = max(op.in_features for op in ops)
    max_out = max(op.out_features for op in ops)
    recon = max((op.in_features * op.out_features for op in ops if op.allow_reconstruct), default=0)
    return max_in, max_out, recon


def prepare_exl3_dense_workspace(root, device, *, _reset: bool = False):
    need = _need(root)
    if need is None:
        return None
    key = _dev_key(device)
    if _reset:
        _WORKSPACES.pop(key, None)
    if key in _WORKSPACES:
        require_exl3_dense_workspace_fits(root, device)
        return _WORKSPACES[key]
    max_in, max_out, recon = need
    rows = GEMM_MAX_ROWS
    ws = Exl3DenseWorkspace(
        device=key, max_rows=rows, max_in=max_in, max_out=max_out, recon_elems=recon,
        x16=torch.empty(rows * max_in, dtype=torch.float16, device=key),
        xh=torch.empty(rows * max_in, dtype=torch.float16, device=key),
        y16=torch.empty(rows * max_out, dtype=torch.float16, device=key),
        recon_work=torch.empty(recon, dtype=torch.float16, device=key),
        recon_out=torch.empty(recon, dtype=torch.bfloat16, device=key),
    )
    _WORKSPACES[key] = ws
    return ws


def require_exl3_dense_workspace_fits(root, device) -> None:
    need = _need(root)
    if need is None:
        return
    ws = _WORKSPACES.get(_dev_key(device))
    if ws is None:
        raise RuntimeError("dense EXL3 workspace missing: call prepare_exl3_dense_workspace after weight load")
    max_in, max_out, recon = need
    if max_in > ws.max_in or max_out > ws.max_out or recon > ws.recon_elems:
        raise RuntimeError(
            f"dense EXL3 workspace is too small (in {ws.max_in}/{max_in}, out {ws.max_out}/{max_out}, "
            f"reconstruct {ws.recon_elems}/{recon}); it cannot grow after CUDA graphs are captured"
        )


__all__ = ["Exl3ColMerged", "Exl3DenseWorkspace", "Exl3LMHead", "Exl3Linear", "GEMM_MAX_ROWS",
           "iter_exl3_linears", "prepare_exl3_dense_workspace", "require_exl3_dense_workspace_fits",
           "validate_exl3_parts"]

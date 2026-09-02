from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from freetoken.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    # Optional precomputed multimodal soft-token embeddings (GPU tensor). The offline
    # path can provide these directly; online requests carry the processor outputs below.
    mm_embeds: torch.Tensor | None = None
    # CPU picture tensors prepared by the tokenizer worker. Pixel rows travel as BF16
    # because Qwen's first picture projection immediately casts to its BF16 weights.
    mm_pixel_values: torch.Tensor | None = None
    mm_image_grid_thw: torch.Tensor | None = None
    mm_token_type_ids: torch.Tensor | None = None
    # Scheduler-derived Qwen three-axis positions; normally absent on the tokenizer wire.
    mrope_position_ids: torch.Tensor | None = None
    mrope_position_delta: int = 0


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass
class CacheRebuildBackendMsg(BaseBackendMsg):
    # tokenizer worker -> scheduler: request a runtime KV/MoE/GDN cache resize.
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"  # only "if_idle" is supported; "drain" is deferred (rejected)


@dataclass
class RoutingStatsBackendMsg(BaseBackendMsg):
    """tokenizer worker -> scheduler: read the MoE decode routing histogram.

    A pure read of device counters the decode path already accumulates, so unlike a cache
    rebuild it needs no idle scheduler and never touches the maintenance gate."""

    request_id: str
    reset: bool = False  # zero the histogram after reading, to window the next workload


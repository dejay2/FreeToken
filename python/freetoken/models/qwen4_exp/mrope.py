"""Qwen VL multimodal rotary-position helpers."""

from __future__ import annotations

import itertools

import torch

from freetoken.layers import StateLessOP
from freetoken.layers.rotary import get_rope
from freetoken.models.config import ModelConfig


class Qwen4MRoPE(StateLessOP):
    """Partial, interleaved temporal/height/width RoPE for Qwen4-Exp."""

    def __init__(self, config: ModelConfig) -> None:
        rotary = config.rotary_config
        self._base = get_rope(
            head_dim=rotary.head_dim,
            rotary_dim=rotary.rotary_dim,
            max_position=rotary.max_position,
            base=rotary.base,
            rope_scaling=tuple(rotary.scaling.items()) if rotary.scaling else None,
        )
        self.mrope_section = tuple(config.qwen4_args.mrope_section)
        if not config.qwen4_args.mrope_interleaved:
            raise ValueError("Qwen4-Exp picture MRoPE requires interleaved axis sections")
        if sum(self.mrope_section) != self.rotary_dim // 2:
            raise ValueError(
                f"MRoPE sections {self.mrope_section} must sum to rotary_dim / 2 "
                f"({self.rotary_dim // 2})"
            )

    @property
    def scalar(self):
        return self._base

    @property
    def head_size(self) -> int:
        return self._base.head_size

    @property
    def rotary_dim(self) -> int:
        return self._base.rotary_dim

    @property
    def is_neox(self) -> bool:
        return self._base.is_neox

    @property
    def _cos_sin_cache(self) -> torch.Tensor:
        return self._base._cos_sin_cache

    @_cos_sin_cache.setter
    def _cos_sin_cache(self, value: torch.Tensor) -> None:
        self._base._cos_sin_cache = value

    def apply_inplace(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int | None = None,
    ) -> None:
        head_size = self.head_size if head_size is None else int(head_size)
        if positions.ndim == 1 and query.is_cuda:
            self._base.apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=head_size,
                cos_sin_cache=self._cos_sin_cache,
                is_neox=self.is_neox,
            )
            return
        if positions.ndim == 1:
            positions = positions.expand(3, -1)
        if query.is_cuda:
            from freetoken.kernel.triton.rope import (
                apply_mrope_with_cos_sin_cache_inplace,
            )

            apply_mrope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=head_size,
                cos_sin_cache=self._cos_sin_cache,
                mrope_section=self.mrope_section,
                is_neox=self.is_neox,
            )
            return

        if positions.ndim != 2 or positions.shape != (3, query.shape[0]):
            raise ValueError(
                f"MRoPE positions must have shape (3, {query.shape[0]}), got "
                f"{tuple(positions.shape)}"
            )
        half = self.rotary_dim // 2
        pair = torch.arange(half, device=positions.device)
        axis = torch.zeros(half, dtype=torch.long, device=positions.device)
        axis[(pair % 3 == 1) & (pair < self.mrope_section[1] * 3)] = 1
        axis[(pair % 3 == 2) & (pair < self.mrope_section[2] * 3)] = 2
        selected = positions.long().transpose(0, 1)[:, axis]
        dim = pair.view(1, -1).expand_as(selected)
        cos = self._cos_sin_cache[:, :half][selected, dim]
        sin = self._cos_sin_cache[:, half:][selected, dim]
        for tensor in (query, key):
            heads = tensor.shape[1] // head_size
            view = tensor.view(tensor.shape[0], heads, head_size)
            first = view[..., :half].float().clone()
            second = view[..., half : self.rotary_dim].float().clone()
            view[..., :half].copy_(
                (first * cos[:, None] - second * sin[:, None]).to(view.dtype)
            )
            view[..., half : self.rotary_dim].copy_(
                (second * cos[:, None] + first * sin[:, None]).to(view.dtype)
            )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.apply_inplace(positions, query, key)
        return query, key


def build_mrope_positions(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, int]:
    """Build exact Qwen3-VL image/text MRoPE positions for one request.

    ``mm_token_type_ids`` uses 0 for text and 1 for image tokens. Video is
    rejected until the server carries its timestamps and grid metadata.
    """
    tokens = input_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    types = mm_token_type_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    grids = image_grid_thw.detach().to(device="cpu", dtype=torch.int64).reshape(-1, 3)
    if types.numel() != tokens.numel():
        raise ValueError(
            "mm_token_type_ids length must match input_ids: "
            f"{types.numel()} != {tokens.numel()}"
        )
    if spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be positive")

    grid_index = 0
    current_position = 0
    pieces: list[torch.Tensor] = []
    for modality, group in itertools.groupby(enumerate(types.tolist()), lambda item: item[1]):
        members = list(group)
        group_len = len(members)
        if modality == 0:
            positions = torch.arange(current_position, current_position + group_len)
            pieces.append(positions.view(1, -1).expand(3, -1))
            current_position += group_len
            continue
        if modality != 1:
            raise NotImplementedError(
                "Qwen4-Exp video MRoPE is not enabled; image input is supported"
            )
        if grid_index >= grids.shape[0]:
            raise ValueError("mm_token_type_ids contains more image groups than image_grid_thw")
        grid_t, grid_h, grid_w = (int(value) for value in grids[grid_index].tolist())
        grid_index += 1
        if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
            raise ValueError("image grid height and width must divide by spatial_merge_size")
        llm_t = grid_t
        llm_h = grid_h // spatial_merge_size
        llm_w = grid_w // spatial_merge_size
        expected = llm_t * llm_h * llm_w
        if group_len != expected:
            raise ValueError(
                "image-token group length does not match image_grid_thw: "
                f"{group_len} != {expected}"
            )
        temporal = torch.arange(llm_t)
        height = torch.arange(llm_h) + current_position
        width = torch.arange(llm_w) + current_position
        t_grid, h_grid, w_grid = torch.meshgrid(
            temporal, height, width, indexing="ij"
        )
        vision = torch.stack((t_grid, h_grid, w_grid), dim=0).reshape(3, -1)
        vision[0].add_(current_position)
        pieces.append(vision)
        current_position += max(llm_h, llm_w)

    if grid_index != grids.shape[0]:
        raise ValueError("image_grid_thw contains more images than mm_token_type_ids")
    positions = torch.cat(pieces, dim=1) if pieces else torch.empty((3, 0), dtype=torch.int64)
    delta = int(positions.max().item() + 1 - tokens.numel()) if tokens.numel() else 0
    return positions.contiguous(), delta


__all__ = ["Qwen4MRoPE", "build_mrope_positions"]

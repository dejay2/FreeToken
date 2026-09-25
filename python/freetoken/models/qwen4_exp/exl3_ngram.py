"""Decoder for ExLlamaV3's ``exl3_ngram_trellis`` n-gram tables (format version 1).

Port of exllamav3 ``modules/quant/exl3_lib/ngram_codec.py`` at 6b84a21 (MIT License,
Copyright (c) 2025 Turboderp; full notice in ``freetoken/kernel/exl3.py``). A packed row is
``1 + 10*K`` int16 words: word 0 is the fp16 row scale's bit pattern, words 1.. hold a
160*K-bit tail-biting ring where stream bits [i*K, (i+1)*K) are the low K bits of state i.

    row[i] = mul1(state_i) * scale + head_bias[head(row)]

FreeToken serves a converted FP8 table (spec 2026-09-25 section 4); this module is the
converter's decoder and its test oracle, never a serve-time path.
"""

from __future__ import annotations

import torch

ROW_DIM = 160
_MUL1 = 0x83DCD12D


def words_per_row(k: int) -> int:
    return 1 + ROW_DIM * k // 16


def mul1_codebook(device) -> torch.Tensor:
    s = torch.arange(65536, dtype=torch.int64, device=device)
    prod = (s * _MUL1) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    h = (1024 + bsum).float()
    k_inv = torch.tensor([0x1EEE], dtype=torch.uint16).view(torch.float16).float().item()
    k_bias = torch.tensor([0xC931], dtype=torch.uint16).view(torch.float16).float().item()
    return (h * k_inv + k_bias).to(torch.float16)


def pack_rows(states: torch.Tensor, scales_f16: torch.Tensor, k: int) -> torch.Tensor:
    n = states.shape[0]
    dev = states.device
    new_bits = states.to(torch.int64) & ((1 << k) - 1)
    bits = (new_bits.unsqueeze(-1) >> torch.arange(k, device=dev)) & 1
    bits = bits.reshape(n, ROW_DIM * k // 16, 16)
    words = (bits << torch.arange(16, device=dev)).sum(dim=-1)
    words = (words & 0xFFFF).to(torch.uint16).view(torch.int16)
    scale_words = scales_f16.to(torch.float16).view(torch.int16).unsqueeze(1)
    return torch.cat((scale_words, words), dim=1).contiguous()


def unpack_rows(packed: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    dev = packed.device
    scales = packed[:, 0].contiguous().view(torch.float16)
    words = packed[:, 1:].contiguous().view(torch.uint16).to(torch.int64)
    stream = ((words.unsqueeze(-1) >> torch.arange(16, device=dev)) & 1).reshape(
        packed.shape[0], ROW_DIM * k
    )
    i = torch.arange(ROW_DIM, device=dev).unsqueeze(1)
    m = torch.arange(16, device=dev).unsqueeze(0)
    src = ((i - m // k) % ROW_DIM) * k + m % k
    states = (stream[:, src] << m).sum(dim=-1)
    return states, scales


def dequant_rows(packed, k: int, codebook: torch.Tensor, bias=None) -> torch.Tensor:
    if packed.shape[1] != words_per_row(k):
        raise ValueError(f"packed rows have {packed.shape[1]} words, K={k} needs {words_per_row(k)}")
    states, scales = unpack_rows(packed, k)
    out = codebook[states].float() * scales.float().unsqueeze(1)
    if bias is not None:
        out = out + bias.float()
    return out


def head_of_rows(rows: torch.Tensor, head_offsets: torch.Tensor) -> torch.Tensor:
    heads = torch.searchsorted(head_offsets.to(rows.device), rows, right=True) - 1
    return heads.clamp(0, head_offsets.shape[0] - 1)


__all__ = ["ROW_DIM", "dequant_rows", "head_of_rows", "mul1_codebook", "pack_rows",
           "unpack_rows", "words_per_row"]

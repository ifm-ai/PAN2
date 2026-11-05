# Copyright 2026 Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
# SPDX-License-Identifier: Apache-2.0
"""Dense (unpacked) attention used at inference time.

Packed training does not come through here: it calls ``flash_attn_varlen``
directly. Only FlashAttention-4 is supported. Tensors are ``[B, S, H, D]`` and
masks are ``[B, S]`` with True marking real tokens.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from pan.utils.flash_attention import padded_flash_attention

_SUPPORTED_MODES = ("flash", "flash4")


def _check_mode(backend: str) -> None:
    if backend not in _SUPPORTED_MODES:
        raise NotImplementedError(
            f"Unsupported attention mode {backend!r}; only FlashAttention-4 ('flash' / 'flash4') is supported."
        )


def _masked_self_attention(q, k, v, mask):
    """Self-attention where ``mask`` gates both queries and keys; masked query rows come back as zeros."""
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.bool()
    out = padded_flash_attention(q, k, v, mask, mask)
    return out.flatten(2)


@torch.compiler.disable
def masked_attention(query, key, value, attn_mask: Optional[torch.Tensor] = None, backend: str = "flash") -> torch.Tensor:
    """Masked self-attention over ``[B, S, H, D]`` inputs; returns ``[B, S, H*D]``."""
    _check_mode(backend)
    return _masked_self_attention(query, key, value, attn_mask)


@torch.compiler.disable
def joint_attention(video_qkv, text_qkv, text_mask: Optional[torch.Tensor] = None, backend: str = "flash") -> torch.Tensor:
    """Joint attention over video tokens followed by text tokens.

    ``video_qkv`` and ``text_qkv`` are ``(q, k, v)`` triples of ``[B, S, H, D]``
    tensors. ``text_mask`` (``[B, S_text]``) marks valid text tokens; video
    tokens are always valid. Returns ``[B, S_video + S_text, H*D]``.
    """
    _check_mode(backend)
    n_video = video_qkv[0].shape[1]
    query, key, value = (torch.cat(pair, dim=1) for pair in zip(video_qkv, text_qkv))
    mask = None if text_mask is None else F.pad(text_mask.bool(), (n_video, 0), value=True)
    return _masked_self_attention(query, key, value, mask)

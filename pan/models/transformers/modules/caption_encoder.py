# Copyright 2025 The HuggingFace Team. All rights reserved.
# Copyright 2025 HuggingFace Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from Diffusers (https://github.com/huggingface/diffusers): `Attention` with q/k
# RMSNorm (src/diffusers/models/attention_processor.py), the SwiGLU `FeedForward`
# (src/diffusers/models/attention.py) and `SwiGLU` (src/diffusers/models/activations.py).
# Modified by MBZUAI, 2026: fused q/k/v projection (`qkv`) and separate SwiGLU gate / value
# projections (`w1` / `w3`), as stored in PAN 2 checkpoints; padded batches take a key padding
# mask and packed variable-length captions run through varlen flash attention.
"""Caption encoder ("token refiner"): a few pre-norm self-attention blocks that refine the
projected text-encoder states before they join the video tokens.

Each block is ``hidden_states + attn(norm1(hidden_states))`` followed by
``hidden_states + ff(norm2(hidden_states))`` with a SwiGLU feed-forward. Blocks run on
padded batches (``attention_mask`` ``[B, L]``) or on packed variable-length captions
(``cu_seqlens`` / ``max_seqlen``).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm
from .attention import masked_attention
from pan.utils.flash_attention import flash_attn_varlen


class SwiGLUFeedForward(nn.Module):
    r"""
    Bias-free SwiGLU feed-forward: `w2(w3(x) * silu(w1(x)))`. diffusers' `SwiGLU` holds `w3` and `w1` as the two
    halves of one `proj`; `w2` is the output projection.
    """

    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        # Creation order (w1, w3, w2) fixes the seeded initialization.
        self.w1 = nn.Linear(dim, inner_dim, bias=False)
        self.w3 = nn.Linear(dim, inner_dim, bias=False)
        self.w2 = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(hidden_states))
        return self.w2(self.w3(hidden_states) * gate)


def _init_residual_output(linear: nn.Linear, zero_init: bool, init_std: Optional[float]) -> None:
    """Optionally re-initialize the output projection of a residual branch: N(0, init_std^2) when
    ``init_std`` is given, otherwise zeros when ``zero_init`` is set."""
    if init_std is not None and init_std < 0:
        raise ValueError(f"residual init std must be non-negative, got {init_std}")
    if init_std is None and not zero_init:
        return
    if init_std is None:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.normal_(linear.weight, mean=0.0, std=init_std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class CaptionEncoderBlock(nn.Module):
    r"""
    One token-refiner block.

    Args:
        dim (`int`): The token width.
        num_attention_heads (`int`): The number of attention heads.
        mlp_ratio (`float`, defaults to 4.0): Ratio of the feed-forward inner dimension to `dim`.
        qk_norm (`bool`, defaults to `True`): RMS-normalize queries and keys per head.
        eps (`float`, defaults to 1e-5): Epsilon of the RMS norms.
        zero_residual_init / residual_init_std: Optional re-initialization of `attn_out` and `ffn.w2`.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        eps: float = 1e-5,
        zero_residual_init: bool = False,
        residual_init_std: Optional[float] = None,
    ):
        super().__init__()
        self.heads = num_attention_heads
        attention_head_dim = dim // num_attention_heads

        # Attribute order below is the parameter (and RNG) order; keep it.
        self.attn_norm = RMSNorm(dim, eps=eps)  # diffusers: norm1
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)  # diffusers: attn.to_q / to_k / to_v
        self.qkv.weight._muon_chunks_dim_0 = 3 * num_attention_heads  # Muon: one row block per (q|k|v, head)
        self.q_norm = RMSNorm(attention_head_dim, eps=eps) if qk_norm else nn.Identity()  # attn.norm_q
        self.k_norm = RMSNorm(attention_head_dim, eps=eps) if qk_norm else nn.Identity()  # attn.norm_k
        self.attn_out = nn.Linear(dim, dim, bias=False)  # attn.to_out[0]
        self.ffn_norm = RMSNorm(dim, eps=eps)  # norm2
        self.ffn = SwiGLUFeedForward(dim, int(dim * mlp_ratio))  # ff
        _init_residual_output(self.attn_out, zero_residual_init, residual_init_std)
        _init_residual_output(self.ffn.w2, zero_residual_init, residual_init_std)

    def attention(self, hidden_states, attention_mask=None, cu_seqlens=None, max_seqlen=None):
        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)
        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        query = self.q_norm(query).to(value.dtype)
        key = self.k_norm(key).to(value.dtype)

        if cu_seqlens is None:
            hidden_states = masked_attention(query, key, value, attn_mask=attention_mask)
        else:
            hidden_states = flash_attn_varlen(
                query, key, value,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            )
        return self.attn_out(hidden_states)

    def forward(self, hidden_states, attention_mask=None, cu_seqlens=None, max_seqlen=None):
        if cu_seqlens is not None and attention_mask is not None:
            raise ValueError("Pass either attention_mask (padded) or cu_seqlens (packed), not both")
        hidden_states = hidden_states + self.attention(self.attn_norm(hidden_states), attention_mask, cu_seqlens, max_seqlen)
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class CaptionEncoder(nn.Module):
    r"""
    `num_layers` token-refiner blocks applied in sequence (diffusers `context_refiner`). Inputs are already projected
    to `dim` (`cap_embedder` in the transformer).

    With a padding mask, the first caption position is always kept visible so a caption that is entirely padding still
    has a key to attend to (no NaNs).
    """

    def __init__(self, dim: int, num_attention_heads: int, num_layers: int = 4, **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CaptionEncoderBlock(dim=dim, num_attention_heads=num_attention_heads, **block_kwargs)
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states, mask=None, cu_seqlens=None, max_seqlen=None):
        if mask is not None:
            mask = mask.bool() | (torch.arange(mask.shape[1], device=mask.device) == 0)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return hidden_states

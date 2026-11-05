# Copyright 2025 HuggingFace Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from Diffusers (https://github.com/huggingface/diffusers): `AdaLayerNormContinuous`
# (src/diffusers/models/normalization.py) followed by the output projection (stored as
# `norm_out` / `proj_out` in PAN 2's Diffusers-format release).
# Modified by MBZUAI, 2026: the modulation is chunked as (shift, scale), the order stored in
# PAN 2 checkpoints; added the packed-sequence path with the fused LayerNorm + modulation
# kernel and the row-range projection.
"""Output head of the transformer: adaLN-modulated LayerNorm and the projection back to latent patches."""
import math

import torch
import torch.nn as nn

from .layers import AdaLNProjection
from .triton_kernels import packed_modulate
from .mlp_ops import range_linear


class OutputHead(nn.Module):
    r"""
    Args:
        inner_dim (`int`): The token width.
        patch_size (`int` or `tuple[int, int, int]`): The patch size; an `int` means a square 2-D patch.
        out_channels (`int`): The number of latent channels per patch element.
        make_act (`Callable`): Constructor of the modulation activation (SiLU in PAN 2).
        zero_init_modulation (`bool`, defaults to `True`): Start the modulation at zero.
    """

    def __init__(self, inner_dim, patch_size, out_channels, make_act, zero_init_modulation: bool = True):
        super().__init__()
        patch_numel = patch_size * patch_size if isinstance(patch_size, int) else math.prod(patch_size)

        # Creation order (norm, proj, modulation) fixes the seeded initialization.
        self.norm = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(inner_dim, patch_numel * out_channels, bias=True)  # diffusers: proj_out
        nn.init.kaiming_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.modulation = AdaLNProjection(inner_dim, 2, make_act, zero_init=zero_init_modulation)  # norm_out.linear

    def forward(self, hidden_states, temb, cu_seqlens=None, ln_stats=None, cu_seqlens_matmul=None):
        shift, scale = self.modulation(temb).chunk(2, dim=1)
        if cu_seqlens is None:
            hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
            return self.proj(hidden_states)

        # Packed path. hidden_states may contain rows outside cu_seqlens (text / padding in the
        # concat layout); their gradient is always forced to zero. With ln_stats the LayerNorm is
        # fused into packed_modulate (same affine-free statistics over the video tokens), so the
        # raw hidden_states go in.
        if ln_stats is not None:
            fusable = isinstance(self.norm, nn.LayerNorm) and not self.norm.elementwise_affine and self.norm.eps == 1e-6
            assert fusable, "OutputHead fuses norm as affine-free LayerNorm(eps=1e-6)."
            normed = hidden_states
        else:
            normed = self.norm(hidden_states)
        hidden_states = packed_modulate(
            normed, shift=shift, scale=scale, cu_seqlens=cu_seqlens,
            force_grad_zero_init=True, ln_stats=ln_stats,
        )
        return range_linear(hidden_states, self.proj.weight, self.proj.bias, cu_seqlens_matmul[0:2])

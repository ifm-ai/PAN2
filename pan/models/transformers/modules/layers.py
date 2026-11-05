# Copyright 2025 HuggingFace Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from Diffusers (https://github.com/huggingface/diffusers): `RMSNorm` and the
# SiLU -> Linear modulation of `AdaLayerNormZero` / `AdaLayerNormContinuous`
# (src/diffusers/models/normalization.py).
# Modified by MBZUAI, 2026: RMSNorm rounds the normalized value back to the input dtype
# before the learned scale (the order PAN 2 was trained with) and has no bias or NPU path;
# the modulation is a standalone projection whose chunks are applied by `modulate` / `gate`;
# added the activation table.
"""Building blocks shared by the transformer: activations, RMS normalization and
adaptive-norm (adaLN) conditioning.

adaLN: a conditioning vector ``c`` of shape ``[B, C]`` is projected to ``chunks``
pieces (shift / scale / gate); they are then applied per token to a ``[B, L, C]``
activation. ``modulate`` and ``gate`` are plain tensor math and work for any
dtype; the fused packed-sequence equivalents live in ``triton_kernels``.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "gelu_tanh": lambda: nn.GELU(approximate="tanh"),
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


def activation_factory(name: str) -> Callable[[], nn.Module]:
    """Zero-argument constructor for the named activation, so each use site
    builds its own module: ``make_act = activation_factory("silu"); make_act()``."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


class RMSNorm(nn.Module):
    r"""
    RMS Norm as introduced in https://huggingface.co/papers/1910.07467 by Zhang et al.

    The variance and the normalization run in float32 and the result is cast back to
    the input dtype *before* the learned scale is applied. Keep that order: the bf16
    training and inference paths depend on it bit for bit.

    Args:
        dim (`int`): Number of dimensions to use for `weight`.
        eps (`float`): Small value to use when calculating the reciprocal of the square-root.
        elementwise_affine (`bool`, defaults to `True`): Whether to learn a per-channel scale.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states * torch.rsqrt(variance + self.eps)).to(input_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class AdaLNProjection(nn.Module):
    r"""
    The modulation of an adaLN layer: ``linear(act(emb))`` -> ``chunks * width`` values (shift / scale / gate pieces).

    Args:
        width (`int`): Width of the conditioning embedding and of each produced piece.
        chunks (`int`): Number of pieces, e.g. 6 for a block (shift / scale / gate, twice) or 2 for the output norm.
        make_act (`Callable`): Constructor of the activation applied to the embedding (SiLU in PAN 2).
        zero_init (`bool`, defaults to `True`): Start the projection at exactly zero, so a fresh block is the identity.
    """

    def __init__(self, width: int, chunks: int, make_act: Callable[[], nn.Module], zero_init: bool = True):
        super().__init__()
        self.act = make_act()
        self.linear = nn.Linear(width, chunks * width, bias=True)
        if zero_init:
            for param in self.linear.parameters():
                nn.init.zeros_(param)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(emb))


def modulate(
    x: torch.Tensor,
    shift: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``x * (1 + scale) + shift`` with per-sample ``shift``/``scale`` of shape ``[B, C]``.

    Either term may be omitted; with both omitted ``x`` is returned unchanged.
    """
    if scale is not None:
        x = x * (1 + scale.unsqueeze(1))
    if shift is not None:
        x = x + shift.unsqueeze(1)
    return x


def gate(x: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scale ``x`` ([B, L, C]) by a per-sample gate ``g`` ([B, C])."""
    return x if g is None else x * g.unsqueeze(1)

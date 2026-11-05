# Copyright 2026 Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
# SPDX-License-Identifier: Apache-2.0
"""Rectified-flow training math: noising, training-time draws and the time shift.

Time runs on ``[0, T]`` with ``T = num_train_timesteps``: ``t = 0`` is the clean
latent and ``t = T`` pure noise, ``x_t = (1 - t/T) * x_0 + (t/T) * noise``, and the
model regresses the velocity ``noise - x_0``. The functions here are handed to the
packed dataloader, which noises every sample on its worker threads.
"""
from enum import Enum
from typing import Optional

import torch


class TimestepDistribution(str, Enum):
    """How training noise levels are drawn (YAML / CLI key ``flow_snr_type``)."""

    UNIFORM = "uniform"
    LOGNORM = "lognorm"  # logistic of a Gaussian, i.e. logit-normal


class TimestepShiftMode(str, Enum):
    """``fixed``: one shift for every sample; ``dynamic``: a shift from each sample's resolution and length."""

    FIXED = "fixed"
    DYNAMIC = "dynamic"


# Draws stay this far inside (0, 1), so neither end of the path is hit exactly.
_EDGE = 1e-5


def interpolate(clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, dim: int = 0, *, T: int = 1000) -> torch.Tensor:
    """The point at time ``t`` on the straight path from ``clean`` to ``noise``.

    ``t`` holds one time per index of axis ``dim`` of ``clean`` and is broadcast along it.
    """
    shape = [1] * clean.ndim
    shape[dim] = -1
    w = (t / T).view(shape)
    return (1 - w) * clean + w * noise


def draw_timesteps(
    count: int,
    device=None,
    generator: Optional[torch.Generator] = None,
    *,
    distribution: TimestepDistribution,
    T: int = 1000,
    loc: float = 0.0,
    scale: float = 1.0,
    default_device=None,
) -> torch.Tensor:
    """``count`` training times on ``[0, T]``, drawn from ``distribution``."""
    device = device or default_device or torch.device("cuda")
    distribution = TimestepDistribution(distribution)
    if distribution is TimestepDistribution.UNIFORM:
        u = torch.rand((count,), device=device, generator=generator)
    elif distribution is TimestepDistribution.LOGNORM:
        z = torch.normal(mean=loc, std=scale, size=(count,), device=device, generator=generator)
        u = 1.0 / (1.0 + torch.exp(-z))
    else:
        raise ValueError(f"unsupported timestep distribution {distribution}")
    lo, hi = _EDGE, 1.0 - _EDGE
    return (u * (hi - lo) + lo) * T


def _line_through(x: torch.Tensor, p1: tuple, p2: tuple) -> torch.Tensor:
    """Evaluate at ``x`` the straight line through the points ``p1`` and ``p2``."""
    (x1, y1), (x2, y2) = p1, p2
    slope = (y2 - y1) / (x2 - x1)
    return slope * x + (y1 - slope * x1)


def shift_timesteps(
    timesteps: torch.Tensor,
    T: int,
    shift: float = 1.0,
    mode: TimestepShiftMode = TimestepShiftMode.FIXED,
    latent_shapes: Optional[torch.Tensor] = None,
    rescale_factor: float = 0.5,
    max_shift: float = 1000.0,
    frame_stride: int = 4,
    pixel_stride_h: int = 16,
    pixel_stride_w: Optional[int] = None,
) -> torch.Tensor:
    """Warp training times towards the noisy end: ``s * f / (1 + (s - 1) * f)`` with ``f = t / T``.

    In ``fixed`` mode ``s`` is ``shift``. In ``dynamic`` mode ``s`` grows with each sample's pixel
    area and frame count, both read from ``latent_shapes`` (``[N, 3]`` latent token grid sizes
    ``t, h, w``) through the token strides ``frame_stride`` / ``pixel_stride_h`` / ``pixel_stride_w``.
    """
    mode = TimestepShiftMode(mode)
    if mode is TimestepShiftMode.FIXED:
        s = shift
    elif mode is TimestepShiftMode.DYNAMIC:
        if latent_shapes is None:
            raise ValueError("latent_shapes is required when train_timestep_shift_mode=dynamic")
        grid = latent_shapes.to(device=timesteps.device, dtype=torch.float32)
        frames = (grid[:, 0] - 1) * frame_stride + 1
        area = (grid[:, 1] * pixel_stride_h) * (grid[:, 2] * (pixel_stride_w or pixel_stride_h))
        log_s = _line_through(area, (256 * 256, 0.5), (1024 * 1024, 1.15)) + _line_through(frames, (1, 0.0), (121, 1.64))
        s = torch.exp(rescale_factor * log_s).clamp(max=max_shift)
    else:
        raise ValueError(f"Unknown timestep shift mode: {mode}")

    if isinstance(s, float) and s == 1.0:
        return timesteps
    f = timesteps / T
    return s * f / (1 + (s - 1) * f) * T

# Copyright 2026 Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
# SPDX-License-Identifier: Apache-2.0
"""Multi-axis rotary position embeddings (RoPE).

The head dimension is split into one slice per spatial/temporal axis
(``axes_dim``); each slice is rotated by that axis' coordinate. Tables are
returned in the "real" layout, ``cos`` and ``sin`` of shape ``[N, sum(axes_dim)]``
with every frequency repeated twice so that it lines up with the
``(x[2i], x[2i+1])`` pairs that :func:`rotate_half` swaps.

Numerics (float32, ``pos * position_scale``, ``repeat_interleave``) are
shared by the model, the training dataloader and the Triton RoPE kernel, so do
not reorder them.
"""
from functools import cache
from typing import Sequence, Tuple

import torch


def _per_axis(value, ndim: int, name: str) -> list:
    """Broadcast a scalar / 1-tuple to ``ndim`` axes; pass an ``ndim``-tuple through."""
    if isinstance(value, (int, float)):
        return [value] * ndim
    value = list(value)
    if len(value) == 1:
        return value * ndim
    if len(value) != ndim:
        raise ValueError(f"{name} must be a scalar or have one entry per axis ({ndim}), got {value}")
    return value


def _cos_sin(dim: int, positions: torch.Tensor, theta: float,
             position_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device)[: (dim // 2)].float() / dim))
    angles = torch.outer(positions * position_scale, inv_freq)
    return angles.cos().repeat_interleave(2, dim=1), angles.sin().repeat_interleave(2, dim=1)


def _multi_axis(axes_dim, coords, theta, position_scale):
    scales = _per_axis(position_scale, len(axes_dim), "position_scale")
    tables = [_cos_sin(dim, coords[axis], theta, scales[axis]) for axis, dim in enumerate(axes_dim)]
    cos = torch.cat([c for c, _ in tables], dim=1)
    sin = torch.cat([s for _, s in tables], dim=1)
    return cos, sin


@cache
def rope_tables_for_grid(
    axes_dim: Sequence[int],
    grid_sizes,
    theta: float = 10000.0,
    position_scale=1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE tables for a dense grid of ``grid_sizes`` (e.g. ``(T, H, W)``), tokens in row-major order.

    Cached on its arguments: pass tuples, not lists. ``position_scale``
    scales the coordinates of each axis (used to keep the temporal metric fixed
    across data frame rates).
    """
    axes_dim = tuple(axes_dim)
    sizes = (grid_sizes,) * len(axes_dim) if isinstance(grid_sizes, int) else tuple(grid_sizes)
    if len(sizes) != len(axes_dim):
        raise ValueError(f"grid_sizes {sizes} must have one entry per rope axis ({len(axes_dim)})")
    axes = torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in sizes], indexing="ij")
    coords = [axis.reshape(-1) for axis in axes]
    return _multi_axis(axes_dim, coords, theta, position_scale)


def rope_tables_for_positions(
    axes_dim: Sequence[int],
    positions: torch.Tensor,
    theta: float = 10000.0,
    position_scale=1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE tables from explicit per-token coordinates ``positions`` of shape ``[N, len(axes_dim)]``."""
    axes_dim = tuple(axes_dim)
    if positions.ndim != 2 or positions.size(1) != len(axes_dim):
        raise ValueError(
            f"Expected positions with shape [N, {len(axes_dim)}], got {tuple(positions.shape)}"
        )
    positions = positions.float()
    coords = [positions[:, axis] for axis in range(len(axes_dim))]
    return _multi_axis(axes_dim, coords, theta, position_scale)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map every adjacent pair ``(a, b)`` of the last dim to ``(-b, a)``."""
    pairs = x.unflatten(-1, (-1, 2))
    return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)


def _broadcast_table(table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """View a ``[S, D]`` table so it broadcasts over ``x`` laid out ``[..., S, H, D]``.

    A table that already carries its own head axis (``[S, 1, D]``, as the packed
    dataloader provides) is used as is.
    """
    table = table.to(x.device)
    if table.ndim == 2:
        if table.shape != (x.shape[-3], x.shape[-1]):
            raise ValueError(f"RoPE table {tuple(table.shape)} does not match x {tuple(x.shape)}")
        return table.unsqueeze(1)
    return table


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate query/key tensors laid out ``[B, S, H, D]`` (or packed ``[S, H, D]``)
    by the ``(cos, sin)`` tables of :func:`rope_tables_for_grid`.

    The rotation runs in float32 and is cast back to the input dtype.
    """
    cos, sin = (_broadcast_table(table, query) for table in cos_sin)

    def rotate(x):
        x32 = x.float()
        return (x32 * cos + rotate_half(x32) * sin).type_as(x)

    return rotate(query), rotate(key)

# Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from Diffusers (https://github.com/huggingface/diffusers): `get_timestep_embedding`,
# `Timesteps` and `TimestepEmbedding` (src/diffusers/models/embeddings.py).
# Modified by MBZUAI, 2026: the sinusoidal projection and the MLP are one module; the
# frequency table is computed once on the CPU in float32 and kept as a buffer that is rebuilt
# after every cast and load.
"""Input embeddings: video patch embedding and the timestep encoder."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _patch_triple(patch_size) -> tuple:
    if isinstance(patch_size, int):
        return (patch_size, patch_size, patch_size)
    patch_size = tuple(int(p) for p in patch_size)
    if len(patch_size) != 3:
        raise ValueError(f"patch_size must be an int or a (t, h, w) triple, got {patch_size}")
    return patch_size


class VideoPatchEmbed(nn.Module):
    r"""
    Video latents -> tokens with a stride == kernel `Conv3d`.

    Two input layouts are supported:

    * dense `[B, C, T, H, W]` -> `[B, T/pt * H/ph * W/pw, inner_dim]`;
    * packed (`packed=True`): the dataloader has already cut each sample into flat patches
      `[C*pt*ph*pw, N, 1, 1]`, so the conv weights are applied as the equivalent linear map. This is numerically
      identical to the conv and lets samples of different resolution share one sequence.

    Args:
        patch_size (`int` or `tuple[int, int, int]`): The `(t, h, w)` patch size.
        in_channels (`int`): The number of latent channels.
        inner_dim (`int`): The token width.
    """

    def __init__(self, patch_size, in_channels: int, inner_dim: int):
        super().__init__()
        self.patch_size = _patch_triple(patch_size)
        self.proj = nn.Conv3d(in_channels, inner_dim, kernel_size=self.patch_size, stride=self.patch_size)
        # Xavier-uniform over the flattened [inner_dim, C*pt*ph*pw] weight, i.e. the conv initialized
        # as the equivalent linear map; zero bias.
        nn.init.xavier_uniform_(self.proj.weight.view(inner_dim, -1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor, packed: bool = False) -> torch.Tensor:
        if packed:
            patches = hidden_states.squeeze(-1).squeeze(-1).transpose(0, 1)  # [N, C*pt*ph*pw]
            return F.linear(patches, self.proj.weight.flatten(1), self.proj.bias)
        hidden_states = self.proj(hidden_states)  # [B, inner_dim, t, h, w]
        return hidden_states.flatten(2).transpose(1, 2)


def get_timestep_embedding(timesteps: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    r"""
    Sinusoidal timestep embedding in `cos, sin` order (diffusers `flip_sin_to_cos=True`, `downscale_freq_shift=0`),
    for a precomputed frequency table `freqs` of length `embedding_dim // 2`.
    """
    emb = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TimestepEncoder(nn.Module):
    r"""
    Timesteps -> sinusoidal embedding -> `in_proj -> act -> out_proj` (diffusers `Timesteps` + `TimestepEmbedding`,
    whose `linear_1` / `linear_2` are `in_proj` / `out_proj` here).

    The frequency table is a persistent buffer (`freqs`) so checkpoints keep the key, but it is a deterministic
    constant: it is rebuilt in fp32 after every dtype/device cast and checkpoint load, never trusted from a
    low-precision copy.

    Args:
        time_embed_dim (`int`): Width of the produced embedding.
        make_act (`Callable`): Constructor of the activation between the two linear layers.
        num_channels (`int`, defaults to 256): Width of the sinusoidal embedding.
        max_period (`int`, defaults to 10000): Controls the lowest frequency.
    """

    def __init__(self, time_embed_dim: int, make_act, num_channels: int = 256, max_period: int = 10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_period = max_period
        self.register_buffer("freqs", self._make_freqs(None))
        self.in_proj = nn.Linear(num_channels, time_embed_dim)
        self.act = make_act()
        self.out_proj = nn.Linear(time_embed_dim, time_embed_dim)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _make_freqs(self, device):
        # Always computed on the CPU in fp32, as at initialization: a GPU exp, or a bf16 copy cast
        # back to fp32, would put a phase error on the high-frequency channels.
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device="cpu")
        exponent = exponent / half_dim
        return torch.exp(exponent).to(device=device)

    def _apply(self, fn, recurse=True):
        # Module.to()/half()/bfloat16()/to_empty() also transform buffers.
        super()._apply(fn, recurse=recurse)
        self.freqs = self._make_freqs(self.freqs.device)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "freqs"
        saved = state_dict.get(key)
        if isinstance(saved, torch.Tensor) and saved.shape == self.freqs.shape:
            # Substitute before loading (copy; never mutate the caller's dict), which
            # also covers assign=True and meta-device initialization.
            state_dict = state_dict.copy()
            state_dict[key] = self._make_freqs(saved.device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self.freqs = self._make_freqs(self.freqs.device)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        sample = get_timestep_embedding(timestep, self.freqs).to(self.in_proj.weight.dtype)
        sample = self.in_proj(sample)
        sample = self.act(sample)
        return self.out_proj(sample)

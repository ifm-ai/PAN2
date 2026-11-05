# Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from Diffusers (https://github.com/huggingface/diffusers).
# Modified by MBZUAI, 2026: reduced FlowMatchEulerDiscreteScheduler to a deterministic Euler
# sampler on a linspace(1, 0) sigma grid with a static time shift (no dynamic shifting,
# terminal stretching, Karras/exponential/beta sigmas, inverted sigmas or stochastic
# sampling); sigmas stay on the CPU, the step is computed in float32, and a timestep that
# appears twice resolves to its later occurrence.
"""Deterministic Euler sampler for velocity-prediction (rectified-flow) models.

Noise levels run from 1 to 0 on an evenly spaced grid of ``steps + 1`` points,
optionally warped by ``sigma -> shift * sigma / (1 + (shift - 1) * sigma)``.
The model is queried at the first ``steps`` levels (scaled to
``num_train_timesteps``), and each update adds ``velocity * (next - current)``
to the sample in float32.

``sigmas`` stays on the CPU (``pan.schedulers.timestep_schedule`` may replace it
with a refined grid); ``timesteps`` lives on the requested device.
"""
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def shifted_sigma_grid(steps: int, shift: float) -> torch.Tensor:
    """``steps + 1`` noise levels from 1 down to 0, warped by the static time shift."""
    grid = torch.linspace(1, 0, steps + 1)
    if shift == 1.0:
        return grid
    return (shift * grid) / (1 + (shift - 1) * grid)


class FlowEulerScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_inference_steps = None
        self._step_index = None
        # Until set_timesteps() is called the grid is the unshifted training one.
        self._install(torch.linspace(1, 0, num_train_timesteps + 1), num_train_timesteps, device=None)

    def _install(self, sigmas: torch.Tensor, scale: int, device) -> None:
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * scale).to(device=device, dtype=torch.float32)

    @property
    def step_index(self):
        """Position in ``sigmas`` of the interval the next ``step`` call integrates over."""
        return self._step_index

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        self.num_inference_steps = num_inference_steps
        self._step_index = None
        grid = shifted_sigma_grid(num_inference_steps, self.config.shift)
        self._install(grid, self.config.num_train_timesteps, device)

    def index_of(self, timestep) -> int:
        """Where ``timestep`` sits in ``timesteps``.

        A value listed twice means sampling resumed part-way through the grid, so
        its second position is used and no interval is skipped.
        """
        if torch.is_tensor(timestep):
            timestep = timestep.to(self.timesteps.device)
        hits = torch.nonzero(self.timesteps == timestep).view(-1).tolist()
        if not hits:
            raise ValueError(f"timestep {timestep} is not in scheduler.timesteps; call set_timesteps() first")
        return hits[min(1, len(hits) - 1)]

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        """Advance ``sample`` one interval along ``sigmas``; the result is float32."""
        integer_index = isinstance(timestep, int) or (torch.is_tensor(timestep) and not timestep.is_floating_point())
        if integer_index:
            raise ValueError(
                "step() expects a value taken from scheduler.timesteps, not a loop index "
                "such as the counter from enumerate(timesteps)."
            )
        if self._step_index is None:
            self._step_index = self.index_of(timestep)
        current, following = self.sigmas[self._step_index], self.sigmas[self._step_index + 1]
        self._step_index += 1
        return sample.float() + model_output.float() * (following - current)

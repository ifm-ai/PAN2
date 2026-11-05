"""T2V/I2V classifier-free guidance controls and T2V-only STG."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Optional, Tuple


def validate_sampling_extensions(num_steps, timestep_schedule="standard"):
    if timestep_schedule not in ("standard", "early_refine_2x"):
        raise ValueError("Unknown timestep_schedule")
    if timestep_schedule == "early_refine_2x" and num_steps != 50:
        raise ValueError("early_refine_2x requires exactly 50 inference steps")


@dataclass(frozen=True)
class GuidancePlan:
    num_steps: int
    start_step: int
    end_step: int
    stg_scale: float
    skip_interval: Optional[Tuple[int, int]] = None
    warmup_scale: float = 1.0

    def active(self, step_index):
        if isinstance(step_index, bool) or not isinstance(step_index, Integral):
            raise ValueError("step_index must be an integer")
        if not 0 <= step_index < self.num_steps:
            raise ValueError("step_index is outside the scheduler's timestep sequence")
        return (step_index < self.end_step
                and (step_index >= self.start_step or self.warmup_scale > 1)
                and not (self.skip_interval is not None
                         and self.skip_interval[0] <= step_index < self.skip_interval[1]))


def resolve_guidance_plan(
    num_steps, *, start_step=0, end_step=None, stg_scale=0.0,
    skip_interval=None, warmup_scale=1.0,
):
    """Resolve zero-based, end-exclusive bounds against the actual schedule."""
    end_step = num_steps if end_step is None else end_step
    for name, value in (("num_steps", num_steps), ("start_step", start_step), ("end_step", end_step)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if num_steps < 1:
        raise ValueError("Require num_steps > 0")
    if not 0 <= start_step < end_step <= num_steps:
        raise ValueError("Require 0 <= start_step < end_step <= actual scheduler steps")
    if skip_interval is not None:
        if (not isinstance(skip_interval, (list, tuple)) or len(skip_interval) != 2
                or any(isinstance(value, bool) or not isinstance(value, Integral)
                       for value in skip_interval)):
            raise ValueError("skip_interval must be a pair of integer step indices")
        if not 0 <= skip_interval[0] < skip_interval[1] <= num_steps:
            raise ValueError("Require 0 <= skip_interval start < end <= actual scheduler steps")
        skip_interval = tuple(int(value) for value in skip_interval)
    if (isinstance(stg_scale, bool) or not isinstance(stg_scale, Real)
            or not isfinite(stg_scale) or stg_scale < 0):
        raise ValueError("stg_scale must be finite and nonnegative")
    if (isinstance(warmup_scale, bool) or not isinstance(warmup_scale, Real)
            or not isfinite(warmup_scale) or warmup_scale < 1):
        raise ValueError("warmup_scale must be finite and at least 1")
    return GuidancePlan(num_steps, start_step, end_step, float(stg_scale),
                        skip_interval, float(warmup_scale))


def apply_guidance(negative, positive, guidance_scale, *, step_index, plan,
                   weak_predict_fn=None):
    """Skips/tail are positive-only; pre-window CFG uses warmup_scale.

    CFG keeps the prediction dtype. STG mixes in float32 and casts back. Fusing/reordering these operations changes BF16 inference results.
    The weak model is called only when both CFG and STG are active.
    """
    active = plan.active(step_index)
    if negative.shape != positive.shape or negative.dtype != positive.dtype:
        raise ValueError("CFG predictions must have matching shape and dtype")
    if negative.device != positive.device or not positive.is_floating_point():
        raise ValueError("CFG predictions must be floating tensors on the same device")
    if (isinstance(guidance_scale, bool) or not isinstance(guidance_scale, Real)
            or not isfinite(guidance_scale) or guidance_scale <= 1):
        raise ValueError("This guidance path requires a finite guidance_scale > 1")
    if not active:
        return positive
    scale = plan.warmup_scale if step_index < plan.start_step else guidance_scale
    guided = negative + scale * (positive - negative)
    if plan.stg_scale:
        if weak_predict_fn is None:
            raise ValueError("STG requires a weak positive prediction")
        weak = weak_predict_fn()
        if (weak.shape != positive.shape or weak.dtype != positive.dtype or weak.device != positive.device
                or not weak.is_floating_point()):
            raise ValueError("STG prediction must match the positive shape, dtype and device")
        guided = (guided.float() + plan.stg_scale * (positive.float() - weak.float())).to(guided.dtype)
    return guided


def predict_stg_positive(transformer, transformer_input, timesteps, prompt_embeds,
                         prompt_mask, *, batch_size, skip_layers):
    """Slice the full [negative, positive] input, including cond/mask channels."""
    if batch_size < 1:
        raise ValueError("STG requires a nonempty positive batch")
    for name, value in (("input", transformer_input), ("timesteps", timesteps),
                        ("prompt_embeds", prompt_embeds), ("prompt_mask", prompt_mask)):
        if value is not None and value.shape[0] != 2 * batch_size:
            raise ValueError(f"STG {name} must have [negative, positive] batch order")
    return transformer(
        transformer_input[batch_size:], timesteps[batch_size:],
        prompt_embeds[batch_size:],
        None if prompt_mask is None else prompt_mask[batch_size:],
        stg_skip_layers=skip_layers, return_dict=False,
    )[0]

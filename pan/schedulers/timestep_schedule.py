"""Local refinement of an already-shifted 50-step flow-matching sigma grid."""
import torch


def apply_timestep_schedule(scheduler, name):
    if name == "standard":
        return scheduler.timesteps
    if name != "early_refine_2x":
        raise ValueError("Unknown timestep schedule")
    sigmas = scheduler.sigmas
    if (sigmas.ndim != 1 or len(sigmas) != 51 or len(scheduler.timesteps) != 50
            or not torch.isfinite(sigmas).all()
            or not torch.all(sigmas[:-1] > sigmas[1:])
            or not torch.isclose(sigmas[0], sigmas.new_tensor(1.))
            or not torch.isclose(sigmas[-1], sigmas.new_tensor(0.))):
        raise ValueError("early_refine_2x requires a descending 50-step sigma grid from 1 to 0")
    # Original boundary indices 15 and 25 are omitted, saving two evaluations.
    # Insert midpoints AFTER shift, so the first two actual sigma intervals halve.
    keep = torch.ones(51, device=sigmas.device, dtype=torch.bool)
    keep[15] = keep[25] = False
    midpoints = (sigmas[:2] + sigmas[1:3]) / 2
    refined = torch.sort(torch.cat((sigmas[keep], midpoints)), descending=True).values
    if not torch.all(refined[:-1] > refined[1:]):
        raise ValueError("Refined sigma grid must be strictly decreasing")
    scheduler.sigmas = refined
    scheduler.timesteps = (refined[:-1] * scheduler.config.num_train_timesteps).to(scheduler.timesteps)
    scheduler._step_index = None
    return scheduler.timesteps

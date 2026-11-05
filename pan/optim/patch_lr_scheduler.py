"""Keep ``LambdaLR.get_last_lr()`` free of device syncs when the learning rate is a tensor.

With a tensor learning rate (the graph-capturable optimizer setup) PyTorch's
``LambdaLR`` holds ``base_lrs`` as device tensors and reports the last learning
rate as clones of them, so every log line that reads it stalls on a device
sync. The schedule is simply ``base_lr * lr_lambda(last_epoch)``, which is
cheap to recompute on the host once the base values have been read back a
single time. :func:`patch_lr_scheduler` makes ``get_last_lr()`` return those
host floats instead; the learning rates written into the optimizer are
untouched.
"""
from weakref import WeakKeyDictionary

from torch.optim.lr_scheduler import LambdaLR

# scheduler -> (the base_lrs list it was read from, the same values as floats).
# Kept outside the scheduler's __dict__ so it never leaks into state_dict().
_host_base_lrs: "WeakKeyDictionary[LambdaLR, tuple[list, list[float]]]" = WeakKeyDictionary()


def _base_lrs_on_host(scheduler: LambdaLR) -> list[float]:
    cached = _host_base_lrs.get(scheduler)
    if cached is None or cached[0] is not scheduler.base_lrs:
        # One device read per base lr, repeated only if load_state_dict swapped the list.
        cached = (scheduler.base_lrs, [float(lr) for lr in scheduler.base_lrs])
        _host_base_lrs[scheduler] = cached
    return cached[1]


def _last_lr_on_host(self: LambdaLR) -> list[float]:
    """Learning rate of every param group at ``last_epoch``, evaluated on the host."""
    return [
        base * lr_lambda(self.last_epoch)
        for base, lr_lambda in zip(_base_lrs_on_host(self), self.lr_lambdas, strict=True)
    ]


def patch_lr_scheduler() -> None:
    LambdaLR.get_last_lr = _last_lr_on_host

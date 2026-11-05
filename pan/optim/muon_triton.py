import torch
from torch.distributed.tensor import DTensor
from torch.optim._muon import _adjust_lr
import triton
import triton.language as tl


def _autotune_configs():
    configs = []
    # Memory-bound elementwise kernel: large blocks keep the B300's HBM3e
    # bandwidth saturated; sweep warps/stages to find the best launch shape.
    for block_size in (2048, 4096, 8192, 16384, 32768):
        for num_warps in (4, 8, 16):
            for num_stages in (2, 3, 4):
                configs.append(
                    triton.Config(
                        {"BLOCK_SIZE": block_size},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=_autotune_configs(), key=["n_elements"], restore_value=["buf_ptr"], cache_results=True
)
@triton.jit
def _muon_momentum_nesterov_kernel(
    buf_ptr,      # in/out: momentum buffer, updated in place
    grad_ptr,     # in: gradient
    update_ptr,   # out: nesterov update
    momentum,     # scalar
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    buf = tl.load(buf_ptr + offsets, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)

    one_minus = 1.0 - momentum
    # buf.lerp_(grad, 1 - momentum) == momentum*buf + (1-momentum)*grad
    new_buf = momentum * buf + one_minus * grad
    # update = grad.lerp(new_buf, momentum) == (1-momentum)*grad + momentum*new_buf
    update = one_minus * grad + momentum * new_buf

    tl.store(buf_ptr + offsets, new_buf, mask=mask)
    tl.store(update_ptr + offsets, update, mask=mask)


def muon_momentum_nesterov(
    buf: torch.Tensor | DTensor,
    grad: torch.Tensor | DTensor,
    momentum: float,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | DTensor:
    """Elementwise fused replacement for, with nesterov=True:

        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum)

    `buf` is updated in place; `update` is returned. `buf` and `grad` must have
    the same shape and both be contiguous. `update` is materialized in
    `output_dtype` (default: `grad.dtype`) — pass `torch.bfloat16` when the
    caller immediately feeds `update` into Newton-Schulz orthogonalization,
    which casts to bf16 as its first step anyway.

    `out` and `output_dtype` cannot both be passed: if `out` is given (a plain,
    local, contiguous tensor shaped like `grad`'s local shard), the update is
    written directly into it in place -- e.g. a slice of a larger pre-allocated
    batch buffer -- instead of allocating a fresh tensor, letting the caller
    avoid a later stack/concat.
    """
    assert out is None or output_dtype is None, "`out` and `output_dtype` cannot both be passed"
    assert buf.shape == grad.shape, "buf and grad must have the same shape"
    assert buf.is_contiguous() and grad.is_contiguous(), "buf and grad must be contiguous"

    if isinstance(buf, DTensor):
        assert isinstance(grad, DTensor), "If buf is a DTensor, grad must also be a DTensor"
        assert buf.placements == grad.placements, "buf and grad must have the same placements"
        device_mesh = buf.device_mesh
        placements = buf.placements
        buf = buf.to_local()
        grad = grad.to_local()
    else:
        placements = None

    n_elements = buf.numel()
    if out is not None:
        assert out.shape == grad.shape, "out must have the same (local) shape as grad"
        assert out.is_contiguous(), "out must be contiguous"
        update = out
    else:
        update = torch.empty(grad.shape, dtype=output_dtype or grad.dtype, device=grad.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _muon_momentum_nesterov_kernel[grid](
        buf, grad, update, momentum, n_elements
    )

    if placements is not None:
        update = DTensor.from_local(update, device_mesh, placements)

    return update


@triton.autotune(
    configs=_autotune_configs(), key=["n_elements"], restore_value=["param_ptr"], cache_results=True
)
@triton.jit
def _muon_step_param_scalar_kernel(
    param_ptr,        # in/out: parameter, updated in place
    update_ptr,       # in: muon update
    wd_factor,        # scalar (1 - lr * weight_decay)
    neg_lr,           # scalar (-adjusted_lr)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    param = tl.load(param_ptr + offsets, mask=mask).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask).to(tl.float32)

    # param.mul_(1 - lr*wd); param.add_(update, alpha=-adjusted_lr)
    new_param = param * wd_factor + update * neg_lr

    tl.store(param_ptr + offsets, new_param, mask=mask)


@triton.autotune(
    configs=_autotune_configs(), key=["n_elements"], restore_value=["param_ptr"], cache_results=True
)
@triton.jit
def _muon_step_param_tensor_kernel(
    param_ptr,        # in/out: parameter, updated in place
    update_ptr,       # in: muon update
    wd_factor_ptr,    # in: 0-dim scalar tensor (1 - lr * weight_decay)
    neg_lr_ptr,       # in: 0-dim scalar tensor (-adjusted_lr)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    wd_factor = tl.load(wd_factor_ptr).to(tl.float32)
    neg_lr = tl.load(neg_lr_ptr).to(tl.float32)

    param = tl.load(param_ptr + offsets, mask=mask).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask).to(tl.float32)

    # param.mul_(1 - lr*wd); param.add_(update, alpha=-adjusted_lr)
    new_param = param * wd_factor + update * neg_lr

    tl.store(param_ptr + offsets, new_param, mask=mask)


def step_param(param, update, lr, weight_decay, base_adjust_lr_fn, shape) -> None:
    """Fused elementwise replacement for:

        adjusted_lr = _adjust_lr(lr, base_adjust_lr_fn, shape)
        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)

    `param` is updated in place. `param` and `update` must have the same shape
    and both be contiguous, but may differ in dtype (e.g. bf16 `update` from
    Newton-Schulz applied to an fp32 `param`) -- each is loaded/stored through
    its own pointer's dtype, with the elementwise math done in fp32. `lr` may
    be a python float or a 0-dim scalar tensor; the latter path keeps the
    scalar factors as (graph-capturable) device tensors and reads them inside
    the kernel, so nothing is materialized on the host.
    """
    assert param.shape == update.shape, "param and update must have the same shape"
    assert param.is_contiguous() and update.is_contiguous(), "param and update must be contiguous"

    if isinstance(lr, torch.Tensor) and lr.device.type == "cpu":
        # torch's DCP `_init_optim_state` dummy-steps the optimizer with a CPU
        # `torch.tensor(0.0)` lr; Triton can't dereference CPU tensors, so take
        # the scalar-kernel path (.item() on a CPU tensor costs no device sync).
        lr = lr.item()

    adjusted_lr = _adjust_lr(lr, base_adjust_lr_fn, shape)
    wd_factor = 1 - lr * weight_decay  # float, or capturable scalar-tensor op
    neg_lr = -adjusted_lr

    if isinstance(param, DTensor):
        assert isinstance(update, DTensor), "If param is a DTensor, update must also be a DTensor"
        assert param.placements == update.placements, "param and update must have the same placements"
        # In-place on the local shard propagates back to the DTensor (shared storage).
        param = param.to_local()
        update = update.to_local()

    n_elements = param.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if isinstance(lr, torch.Tensor):
        # Scalar factors are device tensors: pass by pointer, no host sync.
        _muon_step_param_tensor_kernel[grid](
            param, update, wd_factor, neg_lr, n_elements
        )
    else:
        _muon_step_param_scalar_kernel[grid](
            param, update, wd_factor, neg_lr, n_elements
        )


# =============================================================================
# Fused batched gram-based Newton-Schulz (SM100/B300 fast path for Muon).
#
# For batched updates whose per-matrix gram is 128x128 (row dim == 128), the
# reference quack-symmetric-gemm path spends most of its time on (a) a wasteful
# full-tensor Frobenius normalization and (b) ~16 tiny [128,128] GEMM launches
# per call.  This fuses the whole [M,M] Newton-Schulz segment into one
# persistent kernel per matrix:
#
#   * gram  R = X @ X^T is streamed and accumulated in fp32,
#   * normalization is folded via  norm^2 = trace(R)  (no full-tensor div),
#   * the composite-Q NS iteration runs entirely in bf16 [128,128] registers
#     (matching the reference's bf16 GEMM precision),
#   * the two big  Q @ X  applies stay as cuBLAS batched GEMMs (already at the
#     memory-bandwidth floor).
#
# Same mathematics as patch_muon._gram_newtonschulz_local; ~3x faster on B300.
# Input orientation: X = [B, M=128, N]  (N = model dim, streamed/reduced).
# =============================================================================


@triton.jit
def _gram_ns_fused_kernel(
    X_ptr,        # [B, M, N] bf16, row-major
    Q_ptr,        # [B, M, M] fp32 out
    N,
    a, b, c, eps,
    NSTEPS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    SCALE_BY_INVNORM: tl.constexpr,
    M: tl.constexpr,      # = 128
    BK: tl.constexpr,     # streaming tile over N
):
    pid = tl.program_id(0)
    offs = tl.arange(0, M)
    diag = offs[:, None] == offs[None, :]

    # ---- gram  R = X @ X^T  (stream over N, fp32 accumulate) ----
    x_base = X_ptr + pid * M * N
    Rf = tl.zeros((M, M), dtype=tl.float32)
    for n0 in tl.range(0, N, BK):
        offs_n = n0 + tl.arange(0, BK)
        mask_n = offs_n < N
        xt = tl.load(
            x_base + offs[:, None] * N + offs_n[None, :],
            mask=mask_n[None, :], other=0.0,
        )                                                 # [M, BK] bf16
        Rf += tl.dot(xt, tl.trans(xt), input_precision="tf32")

    # normalization folded through the trace: norm^2 = trace(X X^T) = sum|X|^2
    norm2 = tl.maximum(tl.sum(tl.where(diag, Rf, 0.0)), eps * eps)
    inv_norm = tl.rsqrt(norm2)
    if NORMALIZE:
        Rf = Rf / norm2
    R = Rf.to(tl.bfloat16)                                # persistent bf16

    # ---- Newton-Schulz core segment, all in [M,M] ----
    Q = tl.zeros((M, M), dtype=tl.bfloat16)
    have_Q = False
    for step in tl.static_range(NSTEPS):
        Z = (b * R.to(tl.float32) + c * tl.dot(R, R, input_precision="tf32")).to(tl.bfloat16)
        if have_Q:
            Q = (a * Q.to(tl.float32) + tl.dot(Z, Q, input_precision="tf32")).to(tl.bfloat16)
        else:
            Q = (tl.where(diag, a, 0.0) + Z.to(tl.float32)).to(tl.bfloat16)
            have_Q = True
        if step < NSTEPS - 1:                             # propagate gram R <- Q R Q
            RZ = (a * R.to(tl.float32) + tl.dot(R, Z, input_precision="tf32")).to(tl.bfloat16)
            R = (a * RZ.to(tl.float32) + tl.dot(Z, RZ, input_precision="tf32")).to(tl.bfloat16)

    # Q is consumed by a bf16 bmm apply, so store bf16 directly (the 1/norm
    # multiply promotes to fp32 internally, then rounds once on store).
    if SCALE_BY_INVNORM:
        Q = (Q.to(tl.float32) * inv_norm).to(tl.bfloat16)
    tl.store(Q_ptr + pid * M * M + offs[:, None] * M + offs[None, :], Q)


def _gram_ns_segment(X, coeffs, nsteps, eps, normalize, scale, out=None, BK=256, num_warps=8):
    """X: [B, M=128, N] bf16 -> composite Q: [B, M, M] bf16 (written into `out` if given)."""
    B, M, N = X.shape
    a, b, c = coeffs
    Q = out if out is not None else torch.empty((B, M, M), device=X.device, dtype=torch.bfloat16)
    _gram_ns_fused_kernel[(B,)](
        X, Q, N, a, b, c, eps,
        NSTEPS=nsteps, NORMALIZE=normalize, SCALE_BY_INVNORM=scale,
        M=M, BK=BK, num_warps=num_warps,
    )
    return Q


def can_use_fused_gram_ns(x: torch.Tensor) -> bool:
    """Fast path applies only to batched CUDA updates with a 128x128 gram."""
    return (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.ndim == 3
        and x.size(0) >= 96          # enough matrices to fill the GPU
        and x.size(-2) == 128        # row dim -> gram is 128x128
        and x.size(-1) >= 128        # col (model) dim is the large streamed axis
    )


def batched_gram_newtonschulz(X, ns_coefficients, ns_steps, eps, reset_iterations=frozenset([2])):
    """Fused Triton equivalent of _gram_newtonschulz_local for X = [B, 128, N] bf16.

    Returns the orthogonalized X (same shape/dtype). Assumes the caller already
    checked can_use_fused_gram_ns(X)."""
    X = X.to(torch.bfloat16) if X.dtype != torch.bfloat16 else X
    # kernel indexes X as row-major [B, M, N]; caller must pass that layout.
    assert X.is_contiguous(), "batched_gram_newtonschulz requires a contiguous [B, 128, N] input"
    # This fused path only implements a single reset boundary; more than one
    # reset would be silently mis-computed (only min() is honored).
    assert len(reset_iterations) <= 1, "fused gram-NS supports at most one reset iteration"
    B, M, N = X.shape
    seg1 = min(reset_iterations) if reset_iterations else ns_steps
    seg2 = ns_steps - seg1

    # One [B,M,M] scratch reused across both segments: Q1 is fully consumed by the
    # apply below before Q2 overwrites it.
    Q = torch.empty((B, M, M), device=X.device, dtype=torch.bfloat16)

    # K1: gram + core steps {0..seg1-1}; Q1 folds in 1/norm so it applies to raw X
    Q1 = _gram_ns_segment(X, ns_coefficients, seg1, eps, normalize=True, scale=True, out=Q)
    Xp = torch.bmm(Q1, X)                                    # X' = Q1 @ X  [B,128,N]
    if seg2 == 0:
        return Xp

    # K2: regram (reference does NOT renormalize at reset) + core steps {seg1..}
    Q2 = _gram_ns_segment(Xp, ns_coefficients, seg2, eps, normalize=False, scale=False, out=Q)
    # Result is [B,128,N] == X.shape, so write it back into X's storage (X is no
    # longer read after Xp is formed). Like _gram_newtonschulz_local's X2, this lets
    # the caller detect a same-storage output and skip the copy-back.
    return torch.bmm(Q2, Xp, out=X)                         # X2 = Q2 @ X'

"""Triton kernels for packed variable-length sequences with backward support."""

import torch
import triton
import triton.language as tl


def autocast_cast(x: torch.Tensor) -> torch.Tensor:
    """Cast a floating-point activation to the active autocast dtype.

    These fused Triton/custom-op kernels aren't ATen ops, so autocast's
    op-level dispatch never sees them (unlike nn.Linear/F.linear) -- without
    this they'd silently compute in whatever dtype the caller happens to hand
    them. Call this on each op's main activation input to mirror autocast's
    lower-precision policy explicitly at the op boundary.
    """
    if x.is_floating_point() and torch.is_autocast_enabled():
        return x.to(torch.get_autocast_dtype("cuda"))
    return x


# ─────────────────── autotune configs ───────────────────────

def _autotune_configs():
    """Configs for packed_mean kernels (1D accumulation over sequence)."""
    configs = []
    for BLOCK_C in [64, 128]:
        for BLOCK_L in [64, 128, 256]:
            for num_warps in [4, 8]:
                configs.append(
                    triton.Config(
                        {"BLOCK_C": BLOCK_C, "BLOCK_L": BLOCK_L},
                        num_warps=num_warps,
                    )
                )
    return configs


# ─────────────────────────── segment_sum ───────────────────────────
# Triton replacement for torch.segment_reduce(x, "sum", offsets=..., axis=0),
# used for grad_bias reductions in mlp_ops.py. torch.segment_reduce accumulates
# in x's own dtype, which loses most of its precision when summing bf16 over
# thousands of rows -- this always accumulates in fp32 instead (an atomic-add
# buffer), regardless of x's dtype or the requested output_dtype, then casts
# the fp32 accumulator to output_dtype with a plain .to() (a standard, already
# elementwise-fused op -- no need for a hand-rolled kernel). That final cast
# can't be fused into the reduction itself: the atomic-add buffer is shared
# across many programs (one per row-chunk), so it has to stay in one
# fixed-precision dtype for the whole accumulation, not output_dtype.
#
# Segments can be very unequal in length (e.g. one segment with tens of
# thousands of rows next to one with a few hundred) with very few segments
# (often just 2: video/text) -- a single-program-per-segment reduction would
# under-utilize the GPU badly. So each segment's row range is additionally
# split into BLOCK_M chunks (grid axis 2), with partial sums combined via
# tl.atomic_add, giving good occupancy regardless of how lopsided segments are.

def _segment_reduce_autotune_configs():
    configs = []
    for BLOCK_N in [64, 128, 256, 1024]:
        for BLOCK_M in [64, 128, 256]:
            for num_warps in [4, 8]:
                configs.append(
                    triton.Config({"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}, num_warps=num_warps)
                )
    return configs


@triton.autotune(configs=_segment_reduce_autotune_configs(), key=["N"], restore_value=["out_ptr"], cache_results=True)
@triton.jit
def _segment_reduce_sum_kernel(
    x_ptr, offsets_ptr, out_ptr,
    N,
    stride_xm, stride_xn,
    stride_os, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    seg = tl.program_id(0)
    n_block = tl.program_id(1)
    m_chunk = tl.program_id(2)

    seg_start = tl.load(offsets_ptr + seg)
    seg_end = tl.load(offsets_ptr + seg + 1)

    row_start = seg_start + m_chunk * BLOCK_M
    if row_start >= seg_end:
        return

    col_start = n_block * BLOCK_N
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)
    row_mask = rows < seg_end
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    x_ptrs = x_ptr + rows[:, None].to(tl.int64) * stride_xm + cols[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(x, axis=0)  # (BLOCK_N,)

    out_ptrs = out_ptr + seg * stride_os + cols * stride_on
    tl.atomic_add(out_ptrs, partial, mask=col_mask)


@torch.compiler.disable
def segment_sum(
    x: torch.Tensor, offsets: torch.Tensor, output_dtype: torch.dtype,
) -> torch.Tensor:
    """Drop-in replacement for torch.segment_reduce(x, "sum", offsets=offsets,
    axis=0): sums rows of `x` [M, N] into per-segment sums [S, N] where S =
    offsets.numel() - 1. Always accumulates in fp32 regardless of x.dtype or
    output_dtype -- `x` is used as-is (no dtype conversion before the kernel).
    Forward-only; this is only ever used to compute a gradient directly; not
    itself part of an autograd graph.
    """
    assert x.ndim == 2 and x.is_cuda
    assert offsets.dtype in (torch.int32, torch.int64)
    M, N = x.shape
    num_segments = offsets.numel() - 1

    offsets_i32 = offsets.to(torch.int32)

    out_fp32 = torch.zeros((num_segments, N), dtype=torch.float32, device=x.device)

    # No segment can have more than M rows -- use that as a static upper bound
    # on the row-chunk grid dimension instead of syncing on the true max
    # segment length (which would require a .item() device sync).
    grid = lambda META: (
        num_segments,
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(M, META["BLOCK_M"]),
    )
    _segment_reduce_sum_kernel[grid](
        x, offsets_i32, out_fp32,
        N,
        x.stride(0), x.stride(1),
        out_fp32.stride(0), out_fp32.stride(1),
    )

    return out_fp32.to(output_dtype)


# ─────────────────────────── packed_modulate ───────────────────────────

@triton.autotune(configs=_autotune_configs(), key=["C"], cache_results=True)
@triton.jit
def _packed_modulate_fwd_kernel(
    X_ptr,          # [total_L, C]
    Out_ptr,        # [total_L, C]
    Shift_ptr,      # [B, C]
    Scale_ptr,      # [B, C]
    CuSeqlens_ptr,  # [B+1]
    C: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    seq_start = tl.load(CuSeqlens_ptr + pid_b)
    seq_end = tl.load(CuSeqlens_ptr + pid_b + 1)
    seq_len = seq_end - seq_start

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    if HAS_SCALE:
        scale_val = tl.load(Scale_ptr + pid_b * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    if HAS_SHIFT:
        shift_val = tl.load(Shift_ptr + pid_b * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    for l_start in range(0, seq_len, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < seq_len
        mask = l_mask[:, None] & c_mask[None, :]

        x_ptrs = X_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
        x_val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        if HAS_SCALE:
            x_val = x_val * (1.0 + scale_val[None, :])
        if HAS_SHIFT:
            x_val = x_val + shift_val[None, :]

        out_ptrs = Out_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
        tl.store(out_ptrs, x_val, mask=mask)


@triton.autotune(configs=_autotune_configs(), key=["C"], cache_results=True)
@triton.jit
def _packed_modulate_bwd_kernel(
    GradOut_ptr,    # [total_L, C]
    X_ptr,          # [total_L, C]
    GradX_ptr,      # [total_L, C]
    GradShift_ptr,  # [B, C]
    GradScale_ptr,  # [B, C]
    Scale_ptr,      # [B, C]
    CuSeqlens_ptr,  # [B+1]
    C: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    seq_start = tl.load(CuSeqlens_ptr + pid_b)
    seq_end = tl.load(CuSeqlens_ptr + pid_b + 1)
    seq_len = seq_end - seq_start

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    if HAS_SCALE:
        scale_val = tl.load(Scale_ptr + pid_b * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    acc_shift = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_scale = tl.zeros([BLOCK_C], dtype=tl.float32)

    for l_start in range(0, seq_len, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < seq_len
        mask = l_mask[:, None] & c_mask[None, :]

        go_ptrs = GradOut_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
        go_block = tl.load(go_ptrs, mask=mask, other=0.0).to(tl.float32)

        # grad_x = grad_out * (1 + scale) if HAS_SCALE else grad_out
        if HAS_SCALE:
            grad_x = go_block * (1.0 + scale_val[None, :])
        else:
            grad_x = go_block

        gx_ptrs = GradX_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
        tl.store(gx_ptrs, grad_x, mask=mask)

        # accumulate grad_shift = sum(grad_out)
        if HAS_SHIFT:
            acc_shift += tl.sum(go_block, axis=0)

        # accumulate grad_scale = sum(grad_out * x)
        if HAS_SCALE:
            x_ptrs = X_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
            x_block = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
            acc_scale += tl.sum(go_block * x_block, axis=0)

    if HAS_SHIFT:
        tl.store(GradShift_ptr + pid_b * C + c_offs, acc_shift, mask=c_mask)
    if HAS_SCALE:
        tl.store(GradScale_ptr + pid_b * C + c_offs, acc_scale, mask=c_mask)


# ─────────────────────────────── ln_stats ────────────────────────────────
# ``ln_stats`` is the fused-LayerNorm statistics cache shared by these kernels so
# an affine-free LayerNorm can be fused onto the front of packed_modulate without
# a separate normalization pass. It is NOT (mean, variance):
#
#   * Shape/content: [total_L, 2] fp32, per token, = (sum_c x, sum_c x^2) — the
#     raw 1st/2nd moment SUMS over channels of the tensor whose LayerNorm we want.
#     mean/rstd are finalized on read (see _ln_mean_rstd) with a fixed eps=1e-6, so
#     the producer never needs to know C or eps and the form stays atomic-friendly.
#   * Producers: packed_gate(output_ln_stats=True) and
#     packed_simple_concat(output_ln_stats=True) emit it for their output tensor,
#     atomic-accumulating the two channel-partial sums across channel blocks (the
#     grid parallelizes C, so no single program sees a token's full width).
#   * Consumer: packed_modulate(ln_stats=...) fuses LayerNorm(affine=False,
#     eps=1e-6) in front of the modulate: out = ((x-mean)*rstd)*(1+scale)+shift.
#   * Gradient: ln_stats is a cached forward statistic (mark_non_differentiable).
#     It carries no gradient; the exact LayerNorm backward w.r.t. x is recomputed
#     inside packed_modulate's backward from x + ln_stats, so producing it as a
#     detached side output does not drop or double-count any gradient.
#
# Pattern (see JointAttentionBlock.forward_packed_concat): gate emits ln_stats
# for the tensor it just produced (fused residual add) -> the next packed_modulate
# consumes it as that block/section's pre-norm, chaining across blocks and into
# the output head's norm.

@triton.jit
def _ln_mean_rstd(s0, s1, C: tl.constexpr, eps):
    mean = s0 / C
    var = s1 / C - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    return mean, rstd


@triton.autotune(configs=_autotune_configs(), key=["C"], cache_results=True)
@triton.jit
def _packed_modulate_ln_fwd_kernel(
    X_ptr,          # [total_L, C]
    Out_ptr,        # [total_L, C]
    Shift_ptr,      # [B, C]
    Scale_ptr,      # [B, C]
    LnStats_ptr,    # [total_L, 2] = (sum_c x, sum_c x^2)
    CuSeqlens_ptr,  # [B+1]
    eps,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    seq_start = tl.load(CuSeqlens_ptr + pid_b)
    seq_end = tl.load(CuSeqlens_ptr + pid_b + 1)
    seq_len = seq_end - seq_start

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    scale_val = tl.load(Scale_ptr + pid_b * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    shift_val = tl.load(Shift_ptr + pid_b * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    for l_start in range(0, seq_len, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < seq_len
        tok = seq_start + l_offs
        mask = l_mask[:, None] & c_mask[None, :]

        x_ptrs = X_ptr + tok[:, None] * C + c_offs[None, :]
        x_val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        s0 = tl.load(LnStats_ptr + tok * 2 + 0, mask=l_mask, other=0.0)
        s1 = tl.load(LnStats_ptr + tok * 2 + 1, mask=l_mask, other=0.0)
        mean, rstd = _ln_mean_rstd(s0, s1, C, eps)

        n = (x_val - mean[:, None]) * rstd[:, None]
        out_val = n * (1.0 + scale_val[None, :]) + shift_val[None, :]

        out_ptrs = Out_ptr + tok[:, None] * C + c_offs[None, :]
        tl.store(out_ptrs, out_val, mask=mask)


@triton.autotune(configs=_autotune_configs(), key=["C"], restore_value=["AB_ptr"], cache_results=True)
@triton.jit
def _packed_modulate_ln_bwd_reduce_kernel(
    GradOut_ptr,    # [total_L, C]
    X_ptr,          # [total_L, C]
    Scale_ptr,      # [B, C]
    LnStats_ptr,    # [total_L, 2]
    GradShift_ptr,  # [B, C]
    GradScale_ptr,  # [B, C]
    AB_ptr,         # [total_L, 2] = (sum_c g, sum_c g*n) — atomic accumulated
    CuSeqlens_ptr,  # [B+1]
    eps,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Pass 1: grad_shift = sum_tok go, grad_scale = sum_tok go*n (per-segment,
    # clean per-program writes), and accumulate a=sum_c g, b=sum_c g*n per token
    # (atomics across channel blocks) for the grad_x pass.
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    seq_start = tl.load(CuSeqlens_ptr + pid_b)
    seq_end = tl.load(CuSeqlens_ptr + pid_b + 1)
    seq_len = seq_end - seq_start

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    scale_val = tl.load(Scale_ptr + pid_b * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    acc_shift = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_scale = tl.zeros([BLOCK_C], dtype=tl.float32)

    for l_start in range(0, seq_len, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < seq_len
        tok = seq_start + l_offs
        mask = l_mask[:, None] & c_mask[None, :]

        go = tl.load(GradOut_ptr + tok[:, None] * C + c_offs[None, :], mask=mask, other=0.0).to(tl.float32)
        x_val = tl.load(X_ptr + tok[:, None] * C + c_offs[None, :], mask=mask, other=0.0).to(tl.float32)
        s0 = tl.load(LnStats_ptr + tok * 2 + 0, mask=l_mask, other=0.0)
        s1 = tl.load(LnStats_ptr + tok * 2 + 1, mask=l_mask, other=0.0)
        mean, rstd = _ln_mean_rstd(s0, s1, C, eps)
        n = (x_val - mean[:, None]) * rstd[:, None]
        g = go * (1.0 + scale_val[None, :])

        acc_shift += tl.sum(go, axis=0)
        acc_scale += tl.sum(go * n, axis=0)

        # Masked channels carry go == 0 -> g == 0, so the channel-partial sums are exact.
        tl.atomic_add(AB_ptr + tok * 2 + 0, tl.sum(g, axis=1), mask=l_mask)
        tl.atomic_add(AB_ptr + tok * 2 + 1, tl.sum(g * n, axis=1), mask=l_mask)

    tl.store(GradShift_ptr + pid_b * C + c_offs, acc_shift, mask=c_mask)
    tl.store(GradScale_ptr + pid_b * C + c_offs, acc_scale, mask=c_mask)


@triton.autotune(configs=_autotune_configs(), key=["C"], cache_results=True)
@triton.jit
def _packed_modulate_ln_bwd_gradx_kernel(
    GradOut_ptr,    # [total_L, C]
    X_ptr,          # [total_L, C]
    Scale_ptr,      # [B, C]
    LnStats_ptr,    # [total_L, 2]
    AB_ptr,         # [total_L, 2] = (sum_c g, sum_c g*n)
    GradX_ptr,      # [total_L, C]
    CuSeqlens_ptr,  # [B+1]
    eps,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Pass 2: grad_x = rstd * (g - a - n*b), a = mean_c(g), b = mean_c(g*n).
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    seq_start = tl.load(CuSeqlens_ptr + pid_b)
    seq_end = tl.load(CuSeqlens_ptr + pid_b + 1)
    seq_len = seq_end - seq_start

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    scale_val = tl.load(Scale_ptr + pid_b * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    for l_start in range(0, seq_len, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < seq_len
        tok = seq_start + l_offs
        mask = l_mask[:, None] & c_mask[None, :]

        go = tl.load(GradOut_ptr + tok[:, None] * C + c_offs[None, :], mask=mask, other=0.0).to(tl.float32)
        x_val = tl.load(X_ptr + tok[:, None] * C + c_offs[None, :], mask=mask, other=0.0).to(tl.float32)
        s0 = tl.load(LnStats_ptr + tok * 2 + 0, mask=l_mask, other=0.0)
        s1 = tl.load(LnStats_ptr + tok * 2 + 1, mask=l_mask, other=0.0)
        mean, rstd = _ln_mean_rstd(s0, s1, C, eps)
        n = (x_val - mean[:, None]) * rstd[:, None]
        g = go * (1.0 + scale_val[None, :])

        a = tl.load(AB_ptr + tok * 2 + 0, mask=l_mask, other=0.0) / C
        b = tl.load(AB_ptr + tok * 2 + 1, mask=l_mask, other=0.0) / C
        grad_x = rstd[:, None] * (g - a[:, None] - n * b[:, None])

        tl.store(GradX_ptr + tok[:, None] * C + c_offs[None, :], grad_x, mask=mask)


_MODULATE_LN_EPS = 1e-6


class PackedModulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale, cu_seqlens, force_grad_zero_init, ln_stats):
        total_L, C = x.shape
        out = torch.empty_like(x)
        B = cu_seqlens.shape[0] - 1

        shift = shift.contiguous() if shift is not None else None
        scale = scale.contiguous() if scale is not None else None

        grid = lambda META: (B, triton.cdiv(C, META['BLOCK_C']))

        HAS_LN_STATS = ln_stats is not None
        if HAS_LN_STATS:
            # Fused LayerNorm(affine=False, eps=1e-6) + modulate; ln_stats carries the
            # (sum_c x, sum_c x^2) statistics produced upstream.
            assert shift is not None and scale is not None, "fused ln_stats modulate needs shift and scale"
            _packed_modulate_ln_fwd_kernel[grid](
                x, out, shift, scale, ln_stats, cu_seqlens, _MODULATE_LN_EPS, C=C,
            )
            ctx.save_for_backward(x, scale, cu_seqlens, ln_stats)
        else:
            HAS_SHIFT = shift is not None
            HAS_SCALE = scale is not None
            _packed_modulate_fwd_kernel[grid](
                x, out,
                shift if HAS_SHIFT else x,
                scale if HAS_SCALE else x,
                cu_seqlens,
                C=C, HAS_SHIFT=HAS_SHIFT, HAS_SCALE=HAS_SCALE,
            )
            ctx.save_for_backward(x, scale, cu_seqlens)
            ctx.HAS_SHIFT = HAS_SHIFT
            ctx.HAS_SCALE = HAS_SCALE

        ctx.HAS_LN_STATS = HAS_LN_STATS
        ctx.force_grad_zero_init = force_grad_zero_init

        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Unpack saved tensors exactly once (accessing ctx.saved_tensors more than
        # once re-triggers activation-checkpoint unpacking and raises).
        saved = ctx.saved_tensors
        x, scale, cu_seqlens = saved[0], saved[1], saved[2]
        total_L, C = x.shape
        B = cu_seqlens.shape[0] - 1
        grad_out = grad_out.contiguous()
        grid = lambda META: (B, triton.cdiv(C, META['BLOCK_C']))

        # The output head needs zero gradients outside the active video segments.
        # Other callers write every gradient row they consume and may use an
        # uninitialized allocation. Keep this per-call correctness requirement.
        grad_x = torch.zeros_like(x) if ctx.force_grad_zero_init else torch.empty_like(x)

        if ctx.HAS_LN_STATS:
            ln_stats = saved[3]
            grad_shift = torch.empty(B, C, dtype=grad_out.dtype, device=grad_out.device)
            grad_scale = torch.empty(B, C, dtype=grad_out.dtype, device=grad_out.device)
            # a = sum_c g, b = sum_c g*n per token; zeroed for atomic accumulation.
            ab = torch.zeros((total_L, 2), dtype=torch.float32, device=x.device)
            _packed_modulate_ln_bwd_reduce_kernel[grid](
                grad_out, x, scale, ln_stats, grad_shift, grad_scale, ab, cu_seqlens, _MODULATE_LN_EPS, C=C,
            )
            _packed_modulate_ln_bwd_gradx_kernel[grid](
                grad_out, x, scale, ln_stats, ab, grad_x, cu_seqlens, _MODULATE_LN_EPS, C=C,
            )
            return grad_x, grad_shift, grad_scale, None, None, None

        HAS_SHIFT = ctx.HAS_SHIFT
        HAS_SCALE = ctx.HAS_SCALE
        grad_shift = torch.empty(B, C, dtype=grad_out.dtype, device=grad_out.device) if HAS_SHIFT else None
        grad_scale = torch.empty(B, C, dtype=grad_out.dtype, device=grad_out.device) if HAS_SCALE else None
        _packed_modulate_bwd_kernel[grid](
            grad_out, x,
            grad_x,
            grad_shift if HAS_SHIFT else grad_out,
            grad_scale if HAS_SCALE else grad_out,
            scale if HAS_SCALE else x,
            cu_seqlens,
            C=C, HAS_SHIFT=HAS_SHIFT, HAS_SCALE=HAS_SCALE,
        )
        return grad_x, grad_shift, grad_scale, None, None, None


@torch.compiler.disable
def packed_modulate(x, cu_seqlens, shift=None, scale=None, force_grad_zero_init=False, ln_stats=None):
    """Fused packed modulate: x * (1 + scale[seq]) + shift[seq].

    Args:
        x: [total_L, C]
        cu_seqlens: [B+1] int32
        shift: [B, C] or None
        scale: [B, C] or None
        force_grad_zero_init: zero-init grad_x in backward so rows of x outside cu_seqlens (which
            the kernel never writes) get a zero gradient. Needed when x has rows
            outside the modulated segments, e.g. the text/padding tail of the
            concat layout in the final layer.
        ln_stats: [total_L, 2] = (sum_c x, sum_c x^2) per token, from
            packed_gate/packed_simple_concat(output_ln_stats=True). When given,
            an affine-free LayerNorm(eps=1e-6) is fused in front of the modulate:
            out = ((x - mean) * rstd) * (1 + scale) + shift.

    Returns:
        [total_L, C]
    """
    if scale is None and shift is None and ln_stats is None:
        return x
    assert x.is_contiguous(), "packed_modulate expects contiguous [L, C] input"
    x = autocast_cast(x)
    return PackedModulate.apply(x, shift, scale, cu_seqlens, force_grad_zero_init, ln_stats)


# ─────────────────────────── packed_gate ───────────────────────────

@triton.autotune(configs=_autotune_configs(), key=["C"], restore_value=["LnStats_ptr"], cache_results=True)
@triton.jit
def _packed_gate_fwd_kernel(
    X_ptr,          # [total_L, C]
    Gate_ptr,       # [B, C]
    Out_ptr,        # [total_L, C]
    CuSeqlens_ptr,  # [B+1]
    Bias_ptr,       # [total_L, C] (fused residual add) — dummy when not HAS_BIAS
    LnStats_ptr,    # [total_L, 2] = (sum_c out, sum_c out^2) — atomic accumulated
    C: tl.constexpr,
    USE_TANH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_LN_STATS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    seq_start = tl.load(CuSeqlens_ptr + pid_b)
    seq_end = tl.load(CuSeqlens_ptr + pid_b + 1)
    seq_len = seq_end - seq_start

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    g_val = tl.load(Gate_ptr + pid_b * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    if USE_TANH:
        g_val = tl.extra.cuda.libdevice.tanh(g_val)

    for l_start in range(0, seq_len, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < seq_len
        mask = l_mask[:, None] & c_mask[None, :]

        x_ptrs = X_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
        x_val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        out_val = x_val * g_val[None, :]
        if HAS_BIAS:
            b_ptrs = Bias_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
            out_val = out_val + tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)

        out_ptrs = Out_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
        tl.store(out_ptrs, out_val, mask=mask)

        if OUTPUT_LN_STATS:
            # Masked channels/rows carry out_val == 0, so partial sums over this
            # channel block are exact; atomics accumulate across channel blocks.
            tok = seq_start + l_offs
            tl.atomic_add(LnStats_ptr + tok * 2 + 0, tl.sum(out_val, axis=1), mask=l_mask)
            tl.atomic_add(LnStats_ptr + tok * 2 + 1, tl.sum(out_val * out_val, axis=1), mask=l_mask)


@triton.autotune(configs=_autotune_configs(), key=["C"], cache_results=True)
@triton.jit
def _packed_gate_bwd_kernel(
    GradOut_ptr,    # [total_L, C]
    X_ptr,          # [total_L, C]
    Gate_ptr,       # [B, C]
    GradX_ptr,      # [total_L, C]
    GradGate_ptr,   # [B, C]
    CuSeqlens_ptr,  # [B+1]
    C: tl.constexpr,
    USE_TANH: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    seq_start = tl.load(CuSeqlens_ptr + pid_b)
    seq_end = tl.load(CuSeqlens_ptr + pid_b + 1)
    seq_len = seq_end - seq_start

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    g_val = tl.load(Gate_ptr + pid_b * C + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    if USE_TANH:
        tanh_g = tl.extra.cuda.libdevice.tanh(g_val)
        dtanh = 1.0 - tanh_g * tanh_g
        g_for_x = tanh_g
    else:
        g_for_x = g_val

    acc_gate = tl.zeros([BLOCK_C], dtype=tl.float32)

    for l_start in range(0, seq_len, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < seq_len
        mask = l_mask[:, None] & c_mask[None, :]

        go_ptrs = GradOut_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
        go_block = tl.load(go_ptrs, mask=mask, other=0.0).to(tl.float32)

        # grad_x = grad_out * gate (or tanh(gate))
        grad_x = go_block * g_for_x[None, :]
        gx_ptrs = GradX_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
        tl.store(gx_ptrs, grad_x, mask=mask)

        # accumulate grad_gate = sum(grad_out * x)
        x_ptrs = X_ptr + (seq_start + l_offs[:, None]) * C + c_offs[None, :]
        x_block = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        acc_gate += tl.sum(go_block * x_block, axis=0)

    if USE_TANH:
        acc_gate = acc_gate * dtanh

    tl.store(GradGate_ptr + pid_b * C + c_offs, acc_gate, mask=c_mask)


class PackedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, cu_seqlens, tanh, bias, output_ln_stats):
        total_L, C = x.shape
        B = cu_seqlens.shape[0] - 1
        out = torch.empty_like(x)

        gate = gate.contiguous()
        HAS_BIAS = bias is not None
        if HAS_BIAS:
            bias = bias.contiguous()
        # (sum_c out, sum_c out^2) per token; zeroed for atomic accumulation.
        ln_stats = torch.zeros((total_L, 2), dtype=torch.float32, device=x.device) if output_ln_stats else None

        grid = lambda META: (B, triton.cdiv(C, META['BLOCK_C']))

        _packed_gate_fwd_kernel[grid](
            x, gate, out, cu_seqlens,
            bias if HAS_BIAS else x,
            ln_stats if output_ln_stats else x,
            C=C, USE_TANH=tanh, HAS_BIAS=HAS_BIAS, OUTPUT_LN_STATS=output_ln_stats,
        )

        ctx.save_for_backward(x, gate, cu_seqlens)
        ctx.tanh = tanh
        ctx.has_bias = HAS_BIAS

        if output_ln_stats:
            # ln_stats is a cached forward statistic (of out); its LayerNorm gradient
            # is handled by the consuming packed_modulate, so it carries no grad here.
            ctx.mark_non_differentiable(ln_stats)
            return out, ln_stats
        return out

    @staticmethod
    def backward(ctx, grad_out, grad_ln_stats=None):
        x, gate, cu_seqlens = ctx.saved_tensors
        tanh = ctx.tanh
        total_L, C = x.shape
        B = cu_seqlens.shape[0] - 1

        grad_x = torch.empty_like(x)
        grad_gate = torch.empty_like(gate)

        grid = lambda META: (B, triton.cdiv(C, META['BLOCK_C']))

        grad_out = grad_out.contiguous()
        _packed_gate_bwd_kernel[grid](
            grad_out, x, gate,
            grad_x, grad_gate, cu_seqlens,
            C=C, USE_TANH=tanh,
        )

        # out = x * gate + bias, so d out / d bias = 1 -> grad_bias = grad_out.
        grad_bias = grad_out if ctx.has_bias else None
        return grad_x, grad_gate, None, None, grad_bias, None


@torch.compiler.disable
def packed_gate(x, gate, cu_seqlens, tanh=False, bias=None, output_ln_stats=False):
    """Fused packed gate: x * gate[seq] (+ bias), optionally with tanh.

    Args:
        x: [total_L, C]
        gate: [B, C]
        cu_seqlens: [B+1] int32
        tanh: whether to apply tanh to gate
        bias: [total_L, C] fused residual add (out = x * gate + bias), or None
        output_ln_stats: also return [total_L, 2] = (sum_c out, sum_c out^2) per
            token — the LayerNorm statistics consumed by a downstream
            packed_modulate(ln_stats=...). Rows outside cu_seqlens stay zero.

    Returns:
        out [total_L, C], or (out, ln_stats) when output_ln_stats.
    """
    if gate is None and bias is None and not output_ln_stats:
        return x
    assert x.is_contiguous(), "packed_gate expects contiguous [L, C] input"
    x = autocast_cast(x)
    return PackedGate.apply(x, gate, cu_seqlens, tanh, bias, output_ln_stats)

# ─────────────────── packed_simple_concat ───────────────────────

@triton.autotune(configs=_autotune_configs(), key=["C"], restore_value=["LnStats_ptr"], cache_results=True)
@triton.jit
def _packed_simple_concat_fwd_kernel(
    Img_ptr,            # [total_video, C]
    Txt_ptr,            # [total_text, C]
    CondEmb_ptr,        # [C] (single vector, broadcast to all text tokens)
    Out_ptr,            # [total_seqlen, C]
    LnStats_ptr,        # [total_seqlen, 2] = (sum_c out, sum_c out^2) — atomic accumulated
    Img_stride_0,
    Img_stride_1,
    Txt_stride_0,
    Txt_stride_1,
    CuMatmul_ptr,       # [3] = [0, total_video, total_video + total_text]
    C: tl.constexpr,
    HAS_TEXT_TYPE_VEC: tl.constexpr,
    OUTPUT_LN_STATS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Concat layout: [v0, v1, ..., vn, t0, t1, ..., tn, padding].

    Group 0 is the whole video block, group 1 is the whole text block; their
    output offsets are exactly cu_seqlens_matmul, so the source row within a
    group equals the within-group offset (img[l] / txt[l])."""
    pid_g = tl.program_id(0)   # 0 = video block, 1 = text block
    pid_l = tl.program_id(1)
    pid_c = tl.program_id(2)

    g_start = tl.load(CuMatmul_ptr + pid_g)
    g_end = tl.load(CuMatmul_ptr + pid_g + 1)
    g_len = g_end - g_start

    is_text = pid_g == 1

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l_offs < g_len
    mask = l_mask[:, None] & c_mask[None, :]

    # Load in fp32 in both branches (the store casts back to the output dtype) so
    # the text_type_vec add runs in fp32 and the two branches agree on val's type.
    if is_text:
        src_ptrs = (
            Txt_ptr
            + l_offs[:, None] * Txt_stride_0
            + c_offs[None, :] * Txt_stride_1
        )
        val = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)
        if HAS_TEXT_TYPE_VEC:
            emb_val = tl.load(CondEmb_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
            val = val + emb_val[None, :]
    else:
        src_ptrs = (
            Img_ptr
            + l_offs[:, None] * Img_stride_0
            + c_offs[None, :] * Img_stride_1
        )
        val = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

    tok = g_start + l_offs
    dst_ptrs = Out_ptr + tok[:, None] * C + c_offs[None, :]
    tl.store(dst_ptrs, val, mask=mask)

    if OUTPUT_LN_STATS:
        # Masked channels/rows carry val == 0 (other=0.0), so channel-partial sums
        # are exact; atomics accumulate across channel blocks.
        tl.atomic_add(LnStats_ptr + tok * 2 + 0, tl.sum(val, axis=1), mask=l_mask)
        tl.atomic_add(LnStats_ptr + tok * 2 + 1, tl.sum(val * val, axis=1), mask=l_mask)


@triton.autotune(configs=_autotune_configs(), key=["C"], restore_value=["GradCondEmb_ptr"], cache_results=True)
@triton.jit
def _packed_simple_concat_bwd_kernel(
    GradOut_ptr,        # [total_seqlen, C]
    GradImg_ptr,        # [total_video, C]
    GradTxt_ptr,        # [total_text, C]
    GradCondEmb_ptr,    # [C] — atomically accumulated across L-blocks
    CuMatmul_ptr,       # [3]
    C: tl.constexpr,
    HAS_TEXT_TYPE_VEC: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Backward of the concat: scatter grad_out's video/text blocks back to
    grad_img/grad_txt, and reduce the text block into grad_text_type_vec[C]."""
    pid_g = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_c = tl.program_id(2)

    g_start = tl.load(CuMatmul_ptr + pid_g)
    g_end = tl.load(CuMatmul_ptr + pid_g + 1)
    g_len = g_end - g_start

    is_text = pid_g == 1

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l_offs < g_len
    mask = l_mask[:, None] & c_mask[None, :]

    go_ptrs = GradOut_ptr + (g_start + l_offs[:, None]) * C + c_offs[None, :]
    go_val = tl.load(go_ptrs, mask=mask, other=0.0)

    if is_text:
        gt_ptrs = GradTxt_ptr + l_offs[:, None] * C + c_offs[None, :]
        tl.store(gt_ptrs, go_val, mask=mask)
        # grad_text_type_vec = sum of grad_out over all text tokens
        if HAS_TEXT_TYPE_VEC:
            acc_emb = tl.sum(go_val.to(tl.float32), axis=0)
            tl.atomic_add(GradCondEmb_ptr + c_offs, acc_emb, mask=c_mask)
    else:
        gi_ptrs = GradImg_ptr + l_offs[:, None] * C + c_offs[None, :]
        tl.store(gi_ptrs, go_val, mask=mask)


class PackedSimpleConcat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, txt, text_type_vec, cu_seqlens_matmul, total_seqlen, output_ln_stats):
        """
        Args:
            img: [total_video, C]
            txt: [total_text, C]
            text_type_vec: [C] (single embedding vector) or None
            cu_seqlens_matmul: [3] int32 = [0, total_video, total_video + total_text]
            total_seqlen: int — total output sequence length (>= total_video + total_text)
            output_ln_stats: also produce [total_seqlen, 2] = (sum_c out, sum_c out^2)
        Returns:
            out: [total_seqlen, C] with layout [v0, v1, ..., t0+emb, t1+emb, ..., padding]
            (and ln_stats [total_seqlen, 2] when output_ln_stats)
        """
        total_video, C = img.shape
        total_text = txt.shape[0]
        G = cu_seqlens_matmul.shape[0] - 1  # 2: video block, text block
        max_L = max(total_video, total_text)

        HAS_TEXT_TYPE_VEC = text_type_vec is not None
        # The kernel writes active video/text rows. Padding rows are not consumed.
        out = torch.empty(total_seqlen, C, dtype=img.dtype, device=img.device)
        # (sum_c out, sum_c out^2) per token; zeroed for atomic accumulation.
        ln_stats = torch.zeros((total_seqlen, 2), dtype=torch.float32, device=img.device) if output_ln_stats else None

        grid = lambda META: (G, triton.cdiv(max_L, META['BLOCK_L']), triton.cdiv(C, META['BLOCK_C']))

        _packed_simple_concat_fwd_kernel[grid](
            img, txt,
            text_type_vec if HAS_TEXT_TYPE_VEC else img,  # dummy
            out,
            ln_stats if output_ln_stats else out,  # dummy
            img.stride(0), img.stride(1),
            txt.stride(0), txt.stride(1),
            cu_seqlens_matmul,
            C=C, HAS_TEXT_TYPE_VEC=HAS_TEXT_TYPE_VEC, OUTPUT_LN_STATS=output_ln_stats,
        )

        ctx.save_for_backward(cu_seqlens_matmul)
        ctx.total_video = total_video
        ctx.total_text = total_text
        ctx.C = C
        ctx.HAS_TEXT_TYPE_VEC = HAS_TEXT_TYPE_VEC
        ctx.G = G
        ctx.max_L = max_L

        if output_ln_stats:
            ctx.mark_non_differentiable(ln_stats)
            return out, ln_stats
        return out

    @staticmethod
    def backward(ctx, grad_out, grad_ln_stats=None):
        (cu_seqlens_matmul,) = ctx.saved_tensors
        total_video = ctx.total_video
        total_text = ctx.total_text
        C = ctx.C
        HAS_TEXT_TYPE_VEC = ctx.HAS_TEXT_TYPE_VEC
        G = ctx.G
        max_L = ctx.max_L

        grad_img = torch.zeros(total_video, C, dtype=grad_out.dtype, device=grad_out.device)
        grad_txt = torch.zeros(total_text, C, dtype=grad_out.dtype, device=grad_out.device)
        # Zero-init for atomic accumulation
        grad_text_type_vec = torch.zeros(C, dtype=torch.float32, device=grad_out.device) if HAS_TEXT_TYPE_VEC else None

        grid = lambda META: (G, triton.cdiv(max_L, META['BLOCK_L']), triton.cdiv(C, META['BLOCK_C']))

        _packed_simple_concat_bwd_kernel[grid](
            grad_out.contiguous(),
            grad_img, grad_txt,
            grad_text_type_vec if HAS_TEXT_TYPE_VEC else grad_out,  # dummy
            cu_seqlens_matmul,
            C=C, HAS_TEXT_TYPE_VEC=HAS_TEXT_TYPE_VEC,
        )

        if HAS_TEXT_TYPE_VEC:
            grad_text_type_vec = grad_text_type_vec.to(grad_out.dtype)

        return grad_img, grad_txt, grad_text_type_vec, None, None, None


@torch.compiler.disable
def packed_simple_concat(img, txt, cu_seqlens_matmul, total_seqlen=None, text_type_vec=None, output_ln_stats=False):
    """Simple-concat packed img and txt into [v0, v1, ..., t0, t1, ..., padding].

    All video tokens are placed first, then all text tokens, matching the
    concat layout used for grouped matmuls.

    Args:
        img: [total_video, C]
        txt: [total_text, C]
        cu_seqlens_matmul: [3] int32 = [0, total_video, total_video + total_text]
        total_seqlen: int — total output sequence length. If None, uses
                      img.shape[0] + txt.shape[0]. Extra rows are uninitialized
                      and must not be consumed by downstream operations.
        text_type_vec: [C] single vector broadcast-added to all text tokens, or None
        output_ln_stats: also return [total_seqlen, 2] = (sum_c out, sum_c out^2)
            per token — the LayerNorm stats for a downstream packed_modulate.
            Padding statistics stay zero; padding activations are uninitialized.

    Returns:
        out [total_seqlen, C], or (out, ln_stats) when output_ln_stats.
    """
    if total_seqlen is None:
        total_seqlen = img.shape[0] + txt.shape[0]
    img = autocast_cast(img)
    txt = autocast_cast(txt)
    return PackedSimpleConcat.apply(img, txt, text_type_vec, cu_seqlens_matmul, total_seqlen, output_ln_stats)


# ─────────────────── packed_concat_mse_loss ───────────────────────

@triton.autotune(configs=_autotune_configs(), key=["C"], restore_value=["LossAccum_ptr", "ActiveCount_ptr"], cache_results=True)
@triton.jit
def _packed_concat_mse_fwd_kernel(
    Pred_ptr,           # [total_seqlen, C] — concat [v0, v1, ..., vn, t0, ..., tn, padding]
    Target_ptr,         # [total_video, C] — contiguous packed video
    LossAccum_ptr,      # [1] float32 — atomically accumulated loss
    ActiveCount_ptr,    # [1] float32 — atomically accumulated active element count
    CuMatmul_ptr,       # [3] = [0, total_video, total_video + total_text]
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Sum of squared differences over the video block of a concat-format pred.

    The video block is the leading ``cu_seqlens_matmul[1]`` rows of pred, aligned
    1:1 with the contiguous packed video target; text/padding rows are ignored.
    Atomically accumulates loss and active element count."""
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)

    # cu_seqlens_matmul[0] == 0, so the video length is just cu_seqlens_matmul[1].
    loss_len = tl.load(CuMatmul_ptr + 1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l_offs < loss_len
    mask = l_mask[:, None] & c_mask[None, :]

    pred_ptrs = Pred_ptr + l_offs[:, None] * C + c_offs[None, :]
    pred_val = tl.load(pred_ptrs, mask=mask, other=0.0).to(tl.float32)

    tgt_ptrs = Target_ptr + l_offs[:, None] * C + c_offs[None, :]
    tgt_val = tl.load(tgt_ptrs, mask=mask, other=0.0).to(tl.float32)

    diff = pred_val - tgt_val
    tl.atomic_add(LossAccum_ptr, tl.sum(diff * diff))

    # Only pid_c == 0 accumulates element count to avoid double-counting across C-blocks.
    if pid_c == 0:
        active_rows = tl.sum(l_mask.to(tl.int32))
        active_elements = (active_rows * C).to(tl.float32)
        tl.atomic_add(ActiveCount_ptr, active_elements)


@triton.autotune(configs=_autotune_configs(), key=["C"], cache_results=True)
@triton.jit
def _packed_concat_mse_bwd_kernel(
    Pred_ptr,           # [total_seqlen, C]
    Target_ptr,         # [total_video, C]
    GradPred_ptr,       # [total_seqlen, C] — zero-init'd; only the video block is written
    GradOutput_ptr,     # [1] float32 — scalar grad_output
    ActiveCount_ptr,    # [1] float32 — total active element count from forward
    CuMatmul_ptr,       # [3]
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Backward: write grad into the video block of GradPred; text/padding rows
    are left as whatever GradPred was allocated with (empty -- see caller).
    scale = 2.0 * grad_output / active_count."""
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)

    loss_len = tl.load(CuMatmul_ptr + 1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l_offs < loss_len
    mask = l_mask[:, None] & c_mask[None, :]

    grad_output = tl.load(GradOutput_ptr)
    active_count = tl.load(ActiveCount_ptr)
    scale = 2.0 * grad_output / active_count

    pred_ptrs = Pred_ptr + l_offs[:, None] * C + c_offs[None, :]
    pred_val = tl.load(pred_ptrs, mask=mask, other=0.0).to(tl.float32)

    tgt_ptrs = Target_ptr + l_offs[:, None] * C + c_offs[None, :]
    tgt_val = tl.load(tgt_ptrs, mask=mask, other=0.0).to(tl.float32)

    grad = (pred_val - tgt_val) * scale

    gp_ptrs = GradPred_ptr + l_offs[:, None] * C + c_offs[None, :]
    tl.store(gp_ptrs, grad, mask=mask)


class PackedConcatMSELoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target, cu_seqlens_matmul):
        """
        MSE loss between the video block of a concat-format pred and contiguous target.

        Args:
            pred: [total_seqlen, C] — concat [v0, v1, ..., vn, t0, ..., tn, padding]
            target: [total_video, C] — contiguous packed video tokens
            cu_seqlens_matmul: [3] int32 = [0, total_video, total_video + total_text];
                cu_seqlens_matmul[1] is the number of video tokens the loss covers.

        Returns:
            scalar loss
        """
        total_seqlen, C = pred.shape
        # Host-side upper bound on the video length for the grid; the exact (active)
        # length is read from cu_seqlens_matmul[1] inside the kernel.
        max_video = target.shape[0]

        # Scalar accumulators for atomic add in kernel
        loss_and_count_accum = torch.zeros(2, dtype=torch.float32, device=pred.device)

        grid = lambda META: (triton.cdiv(max_video, META['BLOCK_L']), triton.cdiv(C, META['BLOCK_C']))

        _packed_concat_mse_fwd_kernel[grid](
            pred.contiguous(), target.contiguous(),
            loss_and_count_accum[0], loss_and_count_accum[1],
            cu_seqlens_matmul,
            C=C,
        )

        torch.distributed.all_reduce(loss_and_count_accum, op=torch.distributed.ReduceOp.AVG)

        loss = loss_and_count_accum[0] / loss_and_count_accum[1]

        ctx.save_for_backward(
            pred, target,
            cu_seqlens_matmul,
            loss_and_count_accum[1],
        )
        ctx.max_video = max_video

        return loss.squeeze(0)

    @staticmethod
    def backward(ctx, grad_output):
        (
            pred, target,
            cu_seqlens_matmul,
            active_count,
        ) = ctx.saved_tensors
        max_video = ctx.max_video
        total_seqlen, C = pred.shape

        # Experiment: always empty, ignoring the previous zero-init default (the
        # kernel only writes the video block; text/padding rows used to rely on
        # zero-init here).
        grad_pred = torch.empty(total_seqlen, C, dtype=pred.dtype, device=pred.device)

        grid = lambda META: (triton.cdiv(max_video, META['BLOCK_L']), triton.cdiv(C, META['BLOCK_C']))

        _packed_concat_mse_bwd_kernel[grid](
            pred.contiguous(), target.contiguous(),
            grad_pred,
            grad_output,
            active_count,
            cu_seqlens_matmul,
            C=C,
        )

        return grad_pred, None, None


@torch.compiler.disable
def packed_concat_mse_loss(pred, target, cu_seqlens_matmul):
    """MSE loss over the video tokens of a concat-format pred and contiguous packed target.

    Concat pred layout: [v0, v1, ..., vn, t0, ..., tn, padding]. The video block is the
    leading cu_seqlens_matmul[1] rows, aligned 1:1 with the packed video target. Loss is
    averaged over the video elements (text/padding excluded) and AVG all-reduced across
    ranks.

    Args:
        pred: [total_seqlen, C] concat format
        target: [total_video, C] contiguous packed video
        cu_seqlens_matmul: [3] int32 = [0, total_video, total_video + total_text]

    Returns:
        scalar loss
    """
    return PackedConcatMSELoss.apply(pred, target, cu_seqlens_matmul)


# ─────────────── qkv permute + per-head rms (forward-only kernels) ───────────
#
# Part of the qkv_proj fusion: linear -> (permute + weight + rstd-calc) -> (rstd
# + rope). These are raw kernels + plain (non-autograd) wrapper functions; the
# combined forward/backward autograd op lives in mlp_ops.py (a
# torch.library.custom_op, for SAC compatibility), which calls these
# directly rather than going through their own separate autograd registration.
#
# This step reads qkv_raw (output of the "linear" gemm, concat layout: rows are
# [video tokens | text tokens | padding], columns are [Q | K | V] each of width
# heads*head_dim) and, for Q and K only:
#   - scatters concat-order row `r` to interleave-order row `scatter_idx[r]`
#     (scatter_idx = indices_interleave_to_concat: same identity used by
#     grouped_out_proj's forward, "concat[p] = interleave[scatter_idx[p]]" means
#     token p's value belongs at interleave position scatter_idx[p])
#   - multiplies by the per-head_dim affine (q_weight/k_weight), selected by
#     modality (video vs text) via the grid's segment index
#   - computes rstd = rsqrt(mean(raw^2, dim=head_dim) + eps), the per-token-
#     per-head RMS statistic, cached (not yet applied) for the next kernel
# V is not touched here -- it has no norm, so its permute stays a plain
# index_select at the call site (nothing to fuse it with).


def _qkv_rms_autotune_configs():
    configs = []
    for BLOCK_L in [32, 64, 128]:
        for num_warps in [4, 8]:
            configs.append(triton.Config({"BLOCK_L": BLOCK_L}, num_warps=num_warps))
    return configs


@triton.autotune(configs=_qkv_rms_autotune_configs(), key=["heads_num", "head_dim"], cache_results=True)
@triton.jit
def _qkv_permute_rms_fwd_kernel(
    QKVRaw_ptr,      # [total_seqlen, NUM_PROJ * heads_num * head_dim], concat layout
    QWeight_ptr,     # [2 * head_dim] (video || text)
    KWeight_ptr,     # [2 * head_dim]
    ScatterIdx_ptr,  # [total_seqlen] int32/int64, dest row (interleave) per concat row
    CuSeqlens_ptr,   # [3] int32: [0, total_video, total_active]
    QOut_ptr,        # [total_seqlen, heads_num, head_dim] interleave layout (scattered)
    KOut_ptr,        # [total_seqlen, heads_num, head_dim] interleave layout (scattered)
    QRstd_ptr,       # [total_seqlen, heads_num] fp32, interleave layout (scattered)
    KRstd_ptr,       # [total_seqlen, heads_num] fp32, interleave layout (scattered)
    total_seqlen,
    heads_num: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    NUM_PROJ: tl.constexpr,     # projections per modality in qkv_raw's columns: 3 (q,k,v)
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Flat grid over the *whole* concat-order range (not just cu_seqlens_m's
    # real [0, total_active) segments): the gather/rstd have no cross-token
    # dependency, so nothing stops us from computing every row, including the
    # padding tail. Modality (for q/k weight selection) is the only thing
    # cu_seqlens_m is used for here, so it's resolved per-row instead of via a
    # segment-indexed program: video rows (< cu_seqlens_m[1]) get the video
    # affine, everything else (real text rows *and* padding) gets the text
    # affine -- an arbitrary but harmless choice for padding, since qkv_raw is
    # zero-inited there (raw=0 -> rstd=rsqrt(eps), weighted=0, both finite) and
    # nothing downstream reads those rows anyway. This also means grid sizing
    # only depends on total_seqlen (a shape), never a cu_seqlens value.
    pid_l = tl.program_id(0)
    pid_h = tl.program_id(1)

    text_start = tl.load(CuSeqlens_ptr + 1)

    hidden = heads_num * head_dim
    row_stride = NUM_PROJ * hidden
    q_col_base = 0 * hidden + pid_h * head_dim
    k_col_base = 1 * hidden + pid_h * head_dim

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim

    qw_video = tl.load(QWeight_ptr + 0 * head_dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    qw_text = tl.load(QWeight_ptr + 1 * head_dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    kw_video = tl.load(KWeight_ptr + 0 * head_dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    kw_text = tl.load(KWeight_ptr + 1 * head_dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l_offs < total_seqlen
    rows = l_offs.to(tl.int64)
    mask2d = l_mask[:, None] & d_mask[None, :]

    is_text = rows >= text_start
    qw = tl.where(is_text[:, None], qw_text[None, :], qw_video[None, :])
    kw = tl.where(is_text[:, None], kw_text[None, :], kw_video[None, :])

    scatter_rows = tl.load(ScatterIdx_ptr + rows, mask=l_mask, other=0).to(tl.int64)

    q_ptrs = QKVRaw_ptr + rows[:, None] * row_stride + (q_col_base + d_offs)[None, :]
    q_val = tl.load(q_ptrs, mask=mask2d, other=0.0).to(tl.float32)
    q_rstd = tl.rsqrt(tl.sum(q_val * q_val, axis=1) / head_dim + eps)
    q_weighted = q_val * qw

    k_ptrs = QKVRaw_ptr + rows[:, None] * row_stride + (k_col_base + d_offs)[None, :]
    k_val = tl.load(k_ptrs, mask=mask2d, other=0.0).to(tl.float32)
    k_rstd = tl.rsqrt(tl.sum(k_val * k_val, axis=1) / head_dim + eps)
    k_weighted = k_val * kw

    out_row_stride = heads_num * head_dim
    qout_ptrs = QOut_ptr + scatter_rows[:, None] * out_row_stride + pid_h * head_dim + d_offs[None, :]
    tl.store(qout_ptrs, q_weighted.to(QOut_ptr.dtype.element_ty), mask=mask2d)
    kout_ptrs = KOut_ptr + scatter_rows[:, None] * out_row_stride + pid_h * head_dim + d_offs[None, :]
    tl.store(kout_ptrs, k_weighted.to(KOut_ptr.dtype.element_ty), mask=mask2d)

    tl.store(QRstd_ptr + scatter_rows * heads_num + pid_h, q_rstd, mask=l_mask)
    tl.store(KRstd_ptr + scatter_rows * heads_num + pid_h, k_rstd, mask=l_mask)


def qkv_permute_rms_forward(
    qkv_raw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    scatter_idx: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    heads_num: int,
    head_dim: int,
    eps: float = 1e-6,
):
    """Plain (non-autograd) forward for the permute + weight + rstd-calc step.

    Returns (q_weighted, k_weighted, q_rstd, k_rstd), all in interleave layout
    (length == qkv_raw.shape[0]), fully written for every row -- including the
    padding tail (given a zero-inited qkv_raw, those rows come out finite: raw=0
    -> rstd=rsqrt(eps), weighted=0). cu_seqlens_m is only used to pick the
    video/text affine per row, not to bound which rows get processed.
    """
    total_seqlen, raw_width = qkv_raw.shape
    num_proj = 3
    assert raw_width == num_proj * heads_num * head_dim
    q_weighted = torch.empty(total_seqlen, heads_num, head_dim, dtype=qkv_raw.dtype, device=qkv_raw.device)
    k_weighted = torch.empty(total_seqlen, heads_num, head_dim, dtype=qkv_raw.dtype, device=qkv_raw.device)
    q_rstd = torch.empty(total_seqlen, heads_num, dtype=torch.float32, device=qkv_raw.device)
    k_rstd = torch.empty(total_seqlen, heads_num, dtype=torch.float32, device=qkv_raw.device)

    block_d = triton.next_power_of_2(head_dim)
    grid = lambda META: (triton.cdiv(total_seqlen, META["BLOCK_L"]), heads_num)
    _qkv_permute_rms_fwd_kernel[grid](
        qkv_raw, q_weight, k_weight, scatter_idx, cu_seqlens_m,
        q_weighted, k_weighted, q_rstd, k_rstd, total_seqlen,
        heads_num=heads_num, head_dim=head_dim, eps=eps,
        NUM_PROJ=num_proj, BLOCK_D=block_d,
    )
    return q_weighted, k_weighted, q_rstd, k_rstd


# ─────────────── rstd-multiply + rope (forward-only kernel) ──────────────────
#
# Second half of the qkv norm: y = x_weighted * rstd, then RoPE(y). Operates
# purely in interleave layout (no cu_seqlens/modality dependency at all -- the
# permute already happened in qkv_permute_rms_forward), so the grid just covers
# every row unconditionally; padding rows read/write uninitialized garbage from
# the previous step, which is harmless (never read downstream).
#
# RoPE here matches rope.apply_rope exactly:
#   out = y * cos + rotate_half(y) * sin
#   rotate_half(y)[2i] = -y[2i+1], rotate_half(y)[2i+1] = y[2i]
# i.e. out[d] = y[d] * cos[d] + sign(d) * y[d^1] * sin[d], sign(d) = -1 if d
# even else +1 (d^1 flips the last bit, giving each element's pair partner).


@triton.autotune(configs=_qkv_rms_autotune_configs(), key=["heads_num", "head_dim"], cache_results=True)
@triton.jit
def _qkv_rstd_rope_fwd_kernel(
    XWeighted_ptr,  # [total_seqlen, heads_num, head_dim] interleave layout
    Rstd_ptr,       # [total_seqlen, heads_num] fp32
    Cos_ptr,        # [total_seqlen, head_dim] fp32 (middle broadcast dim squeezed)
    Sin_ptr,        # [total_seqlen, head_dim] fp32
    Out_ptr,        # [total_seqlen, heads_num, head_dim] interleave layout
    total_seqlen,
    heads_num: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_h = tl.program_id(1)

    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l_offs < total_seqlen
    rows = l_offs.to(tl.int64)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim
    partner = d_offs ^ 1
    sign = tl.where(d_offs % 2 == 0, -1.0, 1.0)

    mask2d = l_mask[:, None] & d_mask[None, :]

    row_stride = heads_num * head_dim
    base = rows[:, None] * row_stride + pid_h * head_dim

    x_val = tl.load(XWeighted_ptr + base + d_offs[None, :], mask=mask2d, other=0.0).to(tl.float32)
    x_partner = tl.load(XWeighted_ptr + base + partner[None, :], mask=mask2d, other=0.0).to(tl.float32)

    rstd_val = tl.load(Rstd_ptr + rows * heads_num + pid_h, mask=l_mask, other=0.0)

    y = x_val * rstd_val[:, None]
    y_partner = x_partner * rstd_val[:, None]
    rotated = y_partner * sign[None, :]

    cos_val = tl.load(Cos_ptr + rows[:, None] * head_dim + d_offs[None, :], mask=mask2d, other=0.0)
    sin_val = tl.load(Sin_ptr + rows[:, None] * head_dim + d_offs[None, :], mask=mask2d, other=0.0)

    out_val = y * cos_val + rotated * sin_val

    tl.store(Out_ptr + base + d_offs[None, :], out_val.to(Out_ptr.dtype.element_ty), mask=mask2d)


def qkv_rstd_rope_forward(
    x_weighted: torch.Tensor,
    rstd: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads_num: int,
    head_dim: int,
):
    """Plain (non-autograd) forward for the rstd-multiply + RoPE step.

    x_weighted: [total_seqlen, heads_num, head_dim] interleave layout (from
    qkv_permute_rms_forward). rstd: [total_seqlen, heads_num] fp32. cos/sin:
    [total_seqlen, 1, head_dim] or [total_seqlen, head_dim]. Returns roped
    output, same shape/layout as x_weighted.
    """
    total_seqlen = x_weighted.shape[0]
    cos = cos.reshape(total_seqlen, head_dim)
    sin = sin.reshape(total_seqlen, head_dim)
    out = torch.empty_like(x_weighted)

    block_d = triton.next_power_of_2(head_dim)
    grid = lambda META: (triton.cdiv(total_seqlen, META["BLOCK_L"]), heads_num)
    _qkv_rstd_rope_fwd_kernel[grid](
        x_weighted, rstd, cos, sin, out,
        total_seqlen, heads_num=heads_num, head_dim=head_dim, BLOCK_D=block_d,
    )
    return out


# ─────────────── rstd-multiply + rope backward ────────────────────────────
#
# Forward (per element d, partner = d^1, sign(d) = -1 if d even else +1):
#   y = x_weighted * rstd
#   out[d] = y[d] * cos[d] + sign(d) * y[partner(d)] * sin[d]
#
# Backward (out = J @ y with J the per-pair rotation matrix; grad_y = J^T @ grad_out):
#   grad_y[d]           = cos[d] * grad_out[d] - sign(d) * sin[partner(d)] * grad_out[partner(d)]
#   grad_x_weighted[d]  = grad_y[d] * rstd
#   grad_rstd           = sum_d( grad_y[d] * x_weighted[d] )     (per token, per head)
#
# Note sin is read at `partner(d)`, not `d` -- this is NOT the same index pattern
# as the forward kernel (which reads sin at `d`), so it isn't literally "run the
# forward kernel backward with the sign flipped"; derived directly from the
# per-pair Jacobian transpose instead.


@triton.autotune(configs=_qkv_rms_autotune_configs(), key=["heads_num", "head_dim"], cache_results=True)
@triton.jit
def _qkv_rstd_rope_bwd_kernel(
    GradOut_ptr,     # [total_seqlen, heads_num, head_dim] interleave layout
    XWeighted_ptr,   # [total_seqlen, heads_num, head_dim] interleave layout (saved)
    Rstd_ptr,        # [total_seqlen, heads_num] fp32 (saved)
    Cos_ptr,         # [total_seqlen, head_dim] fp32
    Sin_ptr,         # [total_seqlen, head_dim] fp32
    GradXWeighted_ptr,  # [total_seqlen, heads_num, head_dim] out
    GradRstd_ptr,       # [total_seqlen, heads_num] fp32 out
    total_seqlen,
    heads_num: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_h = tl.program_id(1)

    l_offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l_offs < total_seqlen
    rows = l_offs.to(tl.int64)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim
    partner = d_offs ^ 1
    sign = tl.where(d_offs % 2 == 0, -1.0, 1.0)

    mask2d = l_mask[:, None] & d_mask[None, :]

    row_stride = heads_num * head_dim
    base = rows[:, None] * row_stride + pid_h * head_dim

    go = tl.load(GradOut_ptr + base + d_offs[None, :], mask=mask2d, other=0.0).to(tl.float32)
    go_partner = tl.load(GradOut_ptr + base + partner[None, :], mask=mask2d, other=0.0).to(tl.float32)

    cos_val = tl.load(Cos_ptr + rows[:, None] * head_dim + d_offs[None, :], mask=mask2d, other=0.0)
    sin_partner = tl.load(Sin_ptr + rows[:, None] * head_dim + partner[None, :], mask=mask2d, other=0.0)

    grad_y = cos_val * go - sign[None, :] * sin_partner * go_partner

    rstd_val = tl.load(Rstd_ptr + rows * heads_num + pid_h, mask=l_mask, other=0.0)
    grad_x_weighted = grad_y * rstd_val[:, None]

    x_weighted_val = tl.load(XWeighted_ptr + base + d_offs[None, :], mask=mask2d, other=0.0).to(tl.float32)
    grad_rstd = tl.sum(grad_y * x_weighted_val, axis=1)

    tl.store(
        GradXWeighted_ptr + base + d_offs[None, :],
        grad_x_weighted.to(GradXWeighted_ptr.dtype.element_ty), mask=mask2d,
    )
    tl.store(GradRstd_ptr + rows * heads_num + pid_h, grad_rstd, mask=l_mask)


def qkv_rstd_rope_backward(
    grad_out: torch.Tensor,
    x_weighted: torch.Tensor,
    rstd: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads_num: int,
    head_dim: int,
):
    """Backward for qkv_rstd_rope_forward. Returns (grad_x_weighted, grad_rstd)."""
    total_seqlen = grad_out.shape[0]
    cos = cos.reshape(total_seqlen, head_dim)
    sin = sin.reshape(total_seqlen, head_dim)
    grad_x_weighted = torch.empty_like(x_weighted)
    grad_rstd = torch.empty(total_seqlen, heads_num, dtype=torch.float32, device=grad_out.device)

    block_d = triton.next_power_of_2(head_dim)
    grid = lambda META: (triton.cdiv(total_seqlen, META["BLOCK_L"]), heads_num)
    _qkv_rstd_rope_bwd_kernel[grid](
        grad_out, x_weighted, rstd, cos, sin, grad_x_weighted, grad_rstd,
        total_seqlen, heads_num=heads_num, head_dim=head_dim, BLOCK_D=block_d,
    )
    return grad_x_weighted, grad_rstd


# ─────────────── permute + weight + rstd-calc backward ────────────────────
#
# Backward for qkv_permute_rms_forward. Undoes the weight-multiply and the
# RMSNorm self-referential term, and -- for free, since this kernel already
# iterates concat-order segments (for modality/weight selection) -- gathers
# grad_x_weighted/grad_rstd/rstd from their interleave-order storage back into
# concat order via the same scatter_idx used by the forward pass (forward's
# "write to interleave[scatter_idx[p]]" becomes backward's "read from
# interleave[scatter_idx[p]]"), so grad_raw comes out already in concat order --
# no separate scatter/gather step, and no gather_A needed in the linear
# backward that consumes it.
#
# raw = qkv_raw[p, col]; rstd = rsqrt(mean_d(raw^2) + eps); x_weighted = raw*weight
#   grad_raw[d]    = grad_x_weighted[d] * weight[d]
#                    - (rstd^3 / head_dim) * grad_rstd * raw[d]      (self-referential term)
#   grad_weight[d] = sum_over_tokens_in_segment( grad_x_weighted[d] * raw[d] )
#
# grad_weight is shared across all heads_num programs of a segment, so it's
# atomically accumulated (zero-inited by the caller) -- restore_value is needed
# so autotune's benchmark re-runs don't re-accumulate into it.


def _qkv_rms_bwd_autotune_configs():
    configs = []
    for BLOCK_L in [32, 64, 128]:
        for num_warps in [4, 8]:
            configs.append(triton.Config({"BLOCK_L": BLOCK_L}, num_warps=num_warps))
    return configs


@triton.autotune(
    configs=_qkv_rms_bwd_autotune_configs(),
    key=["heads_num", "head_dim"],
    restore_value=["GradWeightQ_ptr", "GradWeightK_ptr"],
    cache_results=True,
)
@triton.jit
def _qkv_permute_rms_bwd_kernel(
    QKVRaw_ptr,          # [total_seqlen, NUM_PROJ * heads_num * head_dim], concat layout
    QWeight_ptr,         # [2 * head_dim]
    KWeight_ptr,         # [2 * head_dim]
    ScatterIdx_ptr,      # [total_seqlen], gather source (interleave row) per concat row
    CuSeqlens_ptr,       # [3] int32
    GradXWeightedQ_ptr,  # [total_seqlen, heads_num, head_dim] interleave layout (saved grad, gathered)
    GradXWeightedK_ptr,
    GradRstdQ_ptr,       # [total_seqlen, heads_num] fp32 interleave layout (saved grad, gathered)
    GradRstdK_ptr,
    RstdQ_ptr,           # [total_seqlen, heads_num] fp32 interleave layout (saved from forward, gathered)
    RstdK_ptr,
    GradQKVRaw_ptr,      # [total_seqlen, NUM_PROJ * heads_num * head_dim] concat layout, out (Q/K columns only)
    GradWeightQ_ptr,     # [2 * head_dim] fp32, atomic accumulate, zero-inited by caller
    GradWeightK_ptr,
    heads_num: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    NUM_PROJ: tl.constexpr,     # projections per modality in qkv_raw's columns: 3 (q,k,v)
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    seq_start = tl.load(CuSeqlens_ptr + pid_b)
    seq_end = tl.load(CuSeqlens_ptr + pid_b + 1)
    seq_len = seq_end - seq_start

    hidden = heads_num * head_dim
    row_stride = NUM_PROJ * hidden
    q_col_base = 0 * hidden + pid_h * head_dim
    k_col_base = 1 * hidden + pid_h * head_dim

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim

    qw = tl.load(QWeight_ptr + pid_b * head_dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    kw = tl.load(KWeight_ptr + pid_b * head_dim + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    acc_gw_q = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_gw_k = tl.zeros([BLOCK_D], dtype=tl.float32)

    for l_start in range(0, seq_len, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < seq_len
        rows = (seq_start + l_offs).to(tl.int64)
        mask2d = l_mask[:, None] & d_mask[None, :]

        gather_rows = tl.load(ScatterIdx_ptr + rows, mask=l_mask, other=0).to(tl.int64)

        # ---- Q ----
        q_ptrs = QKVRaw_ptr + rows[:, None] * row_stride + (q_col_base + d_offs)[None, :]
        raw_q = tl.load(q_ptrs, mask=mask2d, other=0.0).to(tl.float32)

        out_row_stride = heads_num * head_dim
        gxw_q_ptrs = GradXWeightedQ_ptr + gather_rows[:, None] * out_row_stride + pid_h * head_dim + d_offs[None, :]
        grad_xw_q = tl.load(gxw_q_ptrs, mask=mask2d, other=0.0).to(tl.float32)
        grstd_q = tl.load(GradRstdQ_ptr + gather_rows * heads_num + pid_h, mask=l_mask, other=0.0)
        rstd_q = tl.load(RstdQ_ptr + gather_rows * heads_num + pid_h, mask=l_mask, other=0.0)

        grad_raw_q = grad_xw_q * qw[None, :] - (rstd_q * rstd_q * rstd_q / head_dim)[:, None] * grstd_q[:, None] * raw_q
        tl.store(GradQKVRaw_ptr + rows[:, None] * row_stride + (q_col_base + d_offs)[None, :],
                 grad_raw_q.to(GradQKVRaw_ptr.dtype.element_ty), mask=mask2d)
        acc_gw_q += tl.sum(tl.where(mask2d, grad_xw_q * raw_q, 0.0), axis=0)

        # ---- K ----
        k_ptrs = QKVRaw_ptr + rows[:, None] * row_stride + (k_col_base + d_offs)[None, :]
        raw_k = tl.load(k_ptrs, mask=mask2d, other=0.0).to(tl.float32)

        gxw_k_ptrs = GradXWeightedK_ptr + gather_rows[:, None] * out_row_stride + pid_h * head_dim + d_offs[None, :]
        grad_xw_k = tl.load(gxw_k_ptrs, mask=mask2d, other=0.0).to(tl.float32)
        grstd_k = tl.load(GradRstdK_ptr + gather_rows * heads_num + pid_h, mask=l_mask, other=0.0)
        rstd_k = tl.load(RstdK_ptr + gather_rows * heads_num + pid_h, mask=l_mask, other=0.0)

        grad_raw_k = grad_xw_k * kw[None, :] - (rstd_k * rstd_k * rstd_k / head_dim)[:, None] * grstd_k[:, None] * raw_k
        tl.store(GradQKVRaw_ptr + rows[:, None] * row_stride + (k_col_base + d_offs)[None, :],
                 grad_raw_k.to(GradQKVRaw_ptr.dtype.element_ty), mask=mask2d)
        acc_gw_k += tl.sum(tl.where(mask2d, grad_xw_k * raw_k, 0.0), axis=0)

    tl.atomic_add(GradWeightQ_ptr + pid_b * head_dim + d_offs, acc_gw_q, mask=d_mask)
    tl.atomic_add(GradWeightK_ptr + pid_b * head_dim + d_offs, acc_gw_k, mask=d_mask)


def qkv_permute_rms_backward(
    qkv_raw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    scatter_idx: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    grad_x_weighted_q: torch.Tensor,
    grad_x_weighted_k: torch.Tensor,
    grad_rstd_q: torch.Tensor,
    grad_rstd_k: torch.Tensor,
    rstd_q: torch.Tensor,
    rstd_k: torch.Tensor,
    heads_num: int,
    head_dim: int,
    eps: float = 1e-6,
):
    """Backward for qkv_permute_rms_forward.

    Writes grad_raw for Q and K directly into a buffer shaped like qkv_raw
    (concat layout, [total_seqlen, 3*hidden]) -- the V columns are left
    uninitialized, the caller fills them separately. Returns (grad_qkv_raw,
    grad_q_weight, grad_k_weight); grad_q_weight/grad_k_weight are [2*head_dim]
    fp32.
    """
    total_seqlen, raw_width = qkv_raw.shape
    hidden = heads_num * head_dim
    num_proj = 3
    assert raw_width == num_proj * hidden
    grad_qkv_raw = torch.empty(total_seqlen, raw_width, dtype=qkv_raw.dtype, device=qkv_raw.device)
    grad_q_weight = torch.zeros(2 * head_dim, dtype=torch.float32, device=qkv_raw.device)
    grad_k_weight = torch.zeros(2 * head_dim, dtype=torch.float32, device=qkv_raw.device)

    block_d = triton.next_power_of_2(head_dim)
    grid = (cu_seqlens_m.shape[0] - 1, heads_num)
    _qkv_permute_rms_bwd_kernel[grid](
        qkv_raw, q_weight, k_weight, scatter_idx, cu_seqlens_m,
        grad_x_weighted_q, grad_x_weighted_k, grad_rstd_q, grad_rstd_k, rstd_q, rstd_k,
        grad_qkv_raw, grad_q_weight, grad_k_weight,
        heads_num=heads_num, head_dim=head_dim, eps=eps,
        NUM_PROJ=num_proj, BLOCK_D=block_d,
    )
    return grad_qkv_raw, grad_q_weight, grad_k_weight

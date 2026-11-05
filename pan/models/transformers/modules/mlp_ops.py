import re
from typing import Optional, Tuple

import torch

from quack.gemm_interface import gemm as quack_gemm, gemm_act, gemm_dact
from .quack_autotune_patch import patch_quack_autotune_shape_buckets
from .triton_kernels import (
    autocast_cast,
    qkv_permute_rms_forward,
    qkv_permute_rms_backward,
    qkv_rstd_rope_forward,
    qkv_rstd_rope_backward,
    segment_sum,
)

# Unpadded packed batches give every step a fresh total token count; without
# this, quack's exact-shape autotune key re-autotunes each gemm (minutes) on
# every new shape. See quack_autotune_patch for details.
patch_quack_autotune_shape_buckets()


# Two things here assume a quack >= 0.6.3 contract, and both fail quietly or
# confusingly on older builds, so check once at import:
#   * _varlen_b below hands varlen_m gemms a per-sequence (l, k, n) B. 0.5.x/0.6.0
#     unsqueeze a 2D B themselves and pass a 3D one through, so our explicit
#     broadcast is compatible there -- but 0.6.3+ REJECTS the 2D form outright,
#     which is what forced the change.
#   * quack_autotune_patch mirrors quack's autotune cache-key format, which
#     changed in 0.6.x (raw values, not stringified). Against 0.5.x it mismatches
#     and the patch disables itself, silently restoring a per-sequence-length
#     autotune that costs minutes per gemm.
_MIN_QUACK_VERSION = (0, 6, 3)


def _require_quack_version():
    import quack

    raw = getattr(quack, "__version__", "")
    parts = tuple(int(n) for n in re.findall(r"\d+", raw)[:3])
    if len(parts) == 3 and parts >= _MIN_QUACK_VERSION:
        return
    want = ".".join(str(n) for n in _MIN_QUACK_VERSION)
    raise RuntimeError(
        f"quack-kernels {raw or '<unknown>'} is too old: this fork needs >= {want}. "
        "This is a defensive upgrade past an upstream quack bug (varlen_m read out of "
        "bounds for a shared 2D B); it never affected the correctness of our code, but "
        "newer quack rejects the old call form, and our autotune-cache patch mirrors the "
        "newer cache-key format. Upgrade FA4 first, then quack, as two pip runs:\n"
        '    pip install --pre "flash-attn-4[cu13]" --no-build-isolation -U\n'
        '    pip install "quack-kernels[cu13]" -U\n'
        "(use [cu12] instead on driver < 575; see the README's Compatibility section)."
    )


_require_quack_version()


def _varlen_b(b: torch.Tensor, cu_seqlens_m: torch.Tensor) -> torch.Tensor:
    """Hand a shared 2-D B to a SINGLE-SEGMENT varlen_m gemm as (1, k, n).

    quack >= 0.6.3 requires B to be per-sequence indexed, (l, k, n) with l =
    number of segments, whenever cu_seqlens_m is given: it used to unsqueeze a
    2-D B itself, which read OOB garbage past the first sequence (their fix,
    2026-07-14), and now rejects 2-D outright.

    Every call in this file that shares one weight across its rows is a
    single-segment varlen_m gemm -- that is the point of it: the segment
    boundary (e.g. total_video) is a CUDA scalar, and a plain gemm on x[:bound]
    would need .item(), a device sync per call. With exactly one segment the
    per-sequence B is just b[None], a free view. Multi-segment shared weights
    would need a real broadcast and a decision about what that means, so they
    are refused here rather than guessed at; the per-modality gemms are already
    stacked [2, k, n] against a two-segment cu_seqlens_m and never come through.
    """
    assert cu_seqlens_m.shape[0] == 2, (
        f"_varlen_b is for single-segment varlen_m gemms; got cu_seqlens_m of shape "
        f"{tuple(cu_seqlens_m.shape)} ({cu_seqlens_m.shape[0] - 1} segments)")
    return b.unsqueeze(0)


# ---- grouped_out_proj: fuses the interleave->concat gather with the per-modality
# linear projection (video/text weight blocks selected via cu_seqlens_m) into a
# single quack varlen_m gemm. Registered as a torch.library custom op (rather than
# a bare torch.autograd.Function) so activation checkpointing / SAC can see it --
# see pan/utils/flash_attention.py for the same pattern applied to flash attention.
#
# gather_idx is the *whole* concat<->interleave permutation (length == attn.shape[0],
# padding included as identity), not just its first total_active entries -- so every
# output row (including the padding tail) has a well-defined, distinct source row.
#
# cu_seqlens_m is passed as-is by the caller and is NOT required to extend to
# attn.shape[0] -- e.g. [0, total_video, total_active], stopping short of the
# padding tail. Rows beyond cu_seqlens_m's last boundary are simply never written
# by the kernel (left uninitialized, not zeroed) in both the forward output and
# grad_attn_gathered below; index_copy_ still scatters those uninitialized rows
# into grad_attn's padding positions via the identity part of gather_idx. This is
# fine because nothing downstream ever reads attn's (or its output's) padding
# rows, and grad_out for them is always exactly zero (packed_gate zero-inits
# grad_x outside cu_seqlens_concat), so they contribute nothing to
# grad_weight/grad_bias regardless of what garbage sits in the unwritten rows.
# Default to NOT extending cu_seqlens_m to attn.shape[0] -- it's unnecessary work.
# This also sidesteps ever reading a cu_seqlens boundary with .item() (a device sync).
@torch.library.custom_op("gmlp::grouped_out_proj_fwd", mutates_args=())
def _grouped_out_proj_fwd(
    attn: torch.Tensor,          # [total_seqlen, K], attention output, interleave layout
    weight: torch.Tensor,        # [2, N, K], per-modality weight (nn.Linear convention: out, in)
    bias: torch.Tensor,          # [2, N]
    gather_idx: torch.Tensor,    # [total_seqlen] int32 (interleave source row per concat row)
    cu_seqlens_m: torch.Tensor,  # [3] int32: [0, total_video, total_seqlen]
) -> torch.Tensor:
    # weight/bias are fp32 master params; quack has no autocast dispatch of its
    # own, so cast to the activation dtype ourselves (standard mixed-precision
    # practice: compute in attn's dtype, accumulate grads in the param's dtype).
    B = weight.transpose(-1, -2).to(attn.dtype)  # [2, K, N] view -- already K-contiguous, no copy needed
    bias = bias.to(attn.dtype)
    return quack_gemm(attn, B, bias=bias, cu_seqlens_m=cu_seqlens_m, A_idx=gather_idx)


@_grouped_out_proj_fwd.register_fake
def _(attn, weight, bias, gather_idx, cu_seqlens_m):
    N = weight.shape[-2]
    return attn.new_empty(attn.shape[0], N)


def _grouped_out_proj_setup_context(ctx, inputs, output):
    attn, weight, bias, gather_idx, cu_seqlens_m = inputs
    ctx.save_for_backward(attn, weight, gather_idx, cu_seqlens_m)


def _grouped_out_proj_backward(ctx, grad_out):
    attn, weight, gather_idx, cu_seqlens_m = ctx.saved_tensors
    weight_compute = weight.to(grad_out.dtype)  # cast fp32 master weight to compute dtype

    # grad_attn: gather_idx is a genuine permutation over the full length, so
    # every row of grad_attn is written exactly once by the scatter -- but
    # grad_attn_gathered itself is zero-inited (not empty) because cu_seqlens_m
    # doesn't cover its padding-tail rows, and this becomes flash attention's
    # dout: an uninitialized (garbage/NaN) padding tail there gets incorrectly
    # folded into the *real* segments' dk/dv by flash-attn's varlen backward
    # (verified: a cu_seqlens with repeated trailing boundaries merges the last
    # real segment's dk/dv reduction with the padding tail). Zero is harmless
    # and correct there (padding queries contribute nothing), unlike NaN.
    grad_attn_gathered = torch.zeros(grad_out.shape[0], weight_compute.shape[-1], dtype=grad_out.dtype, device=grad_out.device)
    quack_gemm(grad_out, weight_compute, out=grad_attn_gathered, cu_seqlens_m=cu_seqlens_m)
    grad_attn = torch.empty_like(attn)
    grad_attn.index_copy_(0, gather_idx.long(), grad_attn_gathered)

    # grad_weight: varlen_k contraction over the (gathered) token dimension. A is
    # passed as attn.transpose(0, 1) (a view, not .contiguous()) so it satisfies
    # quack's "m-major" stride requirement for varlen_k without a copy. out_dtype
    # writes the result directly in weight's own (fp32) dtype -- the reduction
    # itself is fp32 either way (tensor-core accumulation), so this just skips a
    # separate elementwise cast, it doesn't change precision.
    grad_weight_T = quack_gemm(
        attn.transpose(0, 1), grad_out, cu_seqlens_k=cu_seqlens_m, A_idx=gather_idx,
        out_dtype=weight.dtype,
    )  # [2, K, N]
    grad_weight = grad_weight_T.transpose(-1, -2)

    # grad_bias: M=1 crashes quack's varlen_k gemm kernel, so this segment-sum
    # (L=2 total) uses segment_sum directly against cu_seqlens_m as
    # `offsets` -- no .item() read needed, cu_seqlens_m is consumed as a tensor.
    # Always accumulated in fp32 (unlike torch.segment_reduce, which accumulates
    # in grad_out's own dtype), then cast to the bias parameter's dtype.
    grad_bias = segment_sum(grad_out, cu_seqlens_m, weight.dtype)

    return grad_attn, grad_weight, grad_bias, None, None


_grouped_out_proj_fwd.register_autograd(
    _grouped_out_proj_backward, setup_context=_grouped_out_proj_setup_context
)


def grouped_out_proj(
    attn: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gather_idx: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
) -> torch.Tensor:
    """Fused out_proj for the packed/concat path.

    Gathers rows of `attn` (interleave layout) via `gather_idx` (the full
    concat<->interleave permutation) directly into the concat-layout output
    (video rows first, then text), applying the per-modality (weight, bias)
    selected by `cu_seqlens_m` = [0, total_video, total_active]. cu_seqlens_m
    does not need to extend to attn.shape[0]: rows past its last boundary (the
    padding tail) are left uninitialized rather than computed, which is fine
    since nothing downstream reads them. Replaces
    attn.index_select(0, indices_interleave_to_concat) + per-modality F.linear +
    _concat_pad with a single quack varlen_m gemm (gather_A + cu_seqlens_m).
    """
    attn = autocast_cast(attn)
    return _grouped_out_proj_fwd(attn, weight, bias, gather_idx, cu_seqlens_m)


# ---- range_linear: a plain (single-weight, not per-modality) nn.Linear
# replacement that only computes over [cu_seqlens_m[0], cu_seqlens_m[-1]) -- the
# rest of x (e.g. the output head's padding tail) is never read by the gemm at all.
#
# This matters beyond just skipping wasted work: nn.Linear's backward reduces
# grad_weight = grad_output^T @ x over *every* row. Even though grad_output is
# exactly zero for padding rows (force_grad_zero_init upstream), IEEE754 gives
# 0 * NaN = NaN, not 0 -- so if x's padding rows ever carry NaN (leftover
# residual-stream garbage, uninitialized memory, ...), a plain nn.Linear would
# let that NaN poison grad_weight/grad_bias for the *whole* layer. Restricting
# the gemm to cu_seqlens_m's single segment means the padding rows are never
# read by either the forward or backward gemm, so they can't contaminate
# anything regardless of what garbage they contain.
#
# Callers must pass an already 2-element, contiguous, offset-0 cu_seqlens_m
# (e.g. cu_seqlens_matmul[0:2]) -- quack's varlen_m gemm requires cu_seqlens_m
# to be contiguous (stride 1); a strided slice (e.g. cu_seqlens[::L-1]) is
# rejected outright, and a generic first/last extraction from a longer tensor
# needs a copy (torch.stack/.clone()) to become contiguous.
@torch.library.custom_op("gmlp::range_linear_fwd", mutates_args=())
def _range_linear_fwd(
    x: torch.Tensor,           # [total_seqlen, K]
    weight: torch.Tensor,      # [N, K], nn.Linear convention (out, in)
    bias: torch.Tensor,        # [N]
    cu_seqlens_m: torch.Tensor,  # [2] int32: [0, active_len]
) -> torch.Tensor:
    B = weight.transpose(-1, -2).to(x.dtype)  # [K, N] view, no copy
    bias_c = bias.to(x.dtype)
    return quack_gemm(x, _varlen_b(B, cu_seqlens_m), bias=bias_c, cu_seqlens_m=cu_seqlens_m)


@_range_linear_fwd.register_fake
def _(x, weight, bias, cu_seqlens_m):
    return x.new_empty(x.shape[0], weight.shape[0])


def _range_linear_setup_context(ctx, inputs, output):
    x, weight, bias, cu_seqlens_m = inputs
    ctx.save_for_backward(x, weight, cu_seqlens_m)


def _range_linear_backward(ctx, grad_out):
    x, weight, cu_seqlens_m = ctx.saved_tensors
    weight_compute = weight.to(grad_out.dtype)

    grad_x = quack_gemm(
        grad_out, _varlen_b(weight_compute, cu_seqlens_m), cu_seqlens_m=cu_seqlens_m)

    grad_weight_T = quack_gemm(
        x.transpose(0, 1), grad_out, cu_seqlens_k=cu_seqlens_m, out_dtype=weight.dtype,
    )  # [K, N]
    grad_weight = grad_weight_T.transpose(-1, -2)

    grad_bias = segment_sum(grad_out, cu_seqlens_m, weight.dtype).squeeze(0)

    return grad_x, grad_weight, grad_bias, None


_range_linear_fwd.register_autograd(
    _range_linear_backward, setup_context=_range_linear_setup_context
)


def range_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
) -> torch.Tensor:
    """nn.Linear-equivalent that only touches rows [cu_seqlens_m[0], cu_seqlens_m[-1])
    of `x` -- forward and backward both skip the rest entirely (see module
    comment above for why this matters beyond just saved compute).

    Pass an already 2-element, contiguous, offset-0 slice here (e.g.
    cu_seqlens_matmul[0:2], not video_cu_seqlens[0]/[-1] stacked) -- quack's
    varlen_m gemm requires cu_seqlens_m to be contiguous (stride 1), and a
    generic first/last extraction from a longer tensor needs a copy
    (torch.stack/.clone()) to satisfy that, whereas cu_seqlens_matmul[0:2] is
    already a valid zero-copy view when the range you want is its first
    segment (as it is for anything restricting to "video tokens only").
    """
    x = autocast_cast(x)
    return _range_linear_fwd(x, weight, bias, cu_seqlens_m)


# ---- grouped_qkv_proj: the full qkv_proj pipeline (linear -> permute+weight+
# rstd-calc -> rstd+rope) as ONE torch.library.custom_op / one autograd node,
# per the same pattern as grouped_out_proj above (SAC needs one opaque
# forward/backward pair here, not three separately-checkpointed ops).
#
# Q/K go through the full per-head RMSNorm + RoPE; V has neither (never did --
# only q/k get _rms in the original project()), so V is just gathered from
# qkv_raw's V columns via a plain index_select (nothing to fuse it with).
#
# The permute (concat<->interleave) is fused into qkv_permute_rms_forward /
# _backward on the Q/K path: forward scatters concat-order results into
# interleave storage via `interleave_to_concat`; backward, needing to go the
# other way, reads (gathers) the interleave-order saved/upstream tensors using
# that *same* array (forward's scatter destination is backward's gather
# source), so grad_qkv_raw for Q/K comes out already in concat order with no
# extra scatter step -- the linear-step backward that consumes it needs no gather_A of its
# own.
#
# The custom_op's own "public" output is (q_out, k_out, v_out); qkv_raw and the
# per-head (weighted, rstd) intermediates are returned too (register_autograd's
# setup_context can only see `inputs` and `output`, not arbitrary locals), and
# the public `grouped_qkv_proj` wrapper below discards them.
@torch.library.custom_op("gmlp::grouped_qkv_proj_fwd", mutates_args=())
def _grouped_qkv_proj_fwd(
    x_mod: torch.Tensor,                 # [total_seqlen, hidden] concat layout
    weight: torch.Tensor,                # [2, 3*hidden, hidden]
    bias: torch.Tensor,                  # [2, 3*hidden]
    q_weight: torch.Tensor,              # [2*head_dim]
    k_weight: torch.Tensor,              # [2*head_dim]
    interleave_to_concat: torch.Tensor,  # [total_seqlen] int32: concat[p] = interleave[itc[p]]
    concat_to_interleave: torch.Tensor,  # [total_seqlen] int64/int32: interleave[q] = concat[cti[q]]
    cu_seqlens_m: torch.Tensor,          # [3] int32: [0, total_video, total_active]
    cos: torch.Tensor,                   # [total_seqlen, 1, head_dim] or [total_seqlen, head_dim]
    sin: torch.Tensor,
    heads_num: int,
    head_dim: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = heads_num * head_dim
    B = weight.transpose(-1, -2).to(x_mod.dtype)
    bias_c = bias.to(x_mod.dtype)
    N = weight.shape[-2]
    qkv_raw = torch.zeros(x_mod.shape[0], N, dtype=x_mod.dtype, device=x_mod.device)
    quack_gemm(x_mod, B, out=qkv_raw, bias=bias_c, cu_seqlens_m=cu_seqlens_m)

    q_weighted, k_weighted, q_rstd, k_rstd = qkv_permute_rms_forward(
        qkv_raw, q_weight, k_weight, interleave_to_concat, cu_seqlens_m, heads_num, head_dim, eps,
    )
    q_out = qkv_rstd_rope_forward(q_weighted, q_rstd, cos, sin, heads_num, head_dim)
    k_out = qkv_rstd_rope_forward(k_weighted, k_rstd, cos, sin, heads_num, head_dim)

    v_raw = qkv_raw[:, 2 * hidden:3 * hidden].reshape(-1, heads_num, head_dim)
    v_out = v_raw.index_select(0, concat_to_interleave.long())

    return q_out, k_out, v_out, qkv_raw, q_weighted, k_weighted, q_rstd, k_rstd


@_grouped_qkv_proj_fwd.register_fake
def _(x_mod, weight, bias, q_weight, k_weight, interleave_to_concat, concat_to_interleave,
      cu_seqlens_m, cos, sin, heads_num, head_dim, eps):
    total_seqlen = x_mod.shape[0]
    three_hidden = weight.shape[-2]
    q_out = x_mod.new_empty(total_seqlen, heads_num, head_dim)
    k_out = x_mod.new_empty(total_seqlen, heads_num, head_dim)
    v_out = x_mod.new_empty(total_seqlen, heads_num, head_dim)
    qkv_raw = x_mod.new_empty(total_seqlen, three_hidden)
    q_weighted = x_mod.new_empty(total_seqlen, heads_num, head_dim)
    k_weighted = x_mod.new_empty(total_seqlen, heads_num, head_dim)
    q_rstd = x_mod.new_empty(total_seqlen, heads_num, dtype=torch.float32)
    k_rstd = x_mod.new_empty(total_seqlen, heads_num, dtype=torch.float32)
    return q_out, k_out, v_out, qkv_raw, q_weighted, k_weighted, q_rstd, k_rstd


def _grouped_qkv_proj_setup_context(ctx, inputs, output):
    (x_mod, weight, bias, q_weight, k_weight, interleave_to_concat, concat_to_interleave,
     cu_seqlens_m, cos, sin, heads_num, head_dim, eps) = inputs
    q_out, k_out, v_out, qkv_raw, q_weighted, k_weighted, q_rstd, k_rstd = output
    ctx.save_for_backward(
        x_mod, weight, q_weight, k_weight, interleave_to_concat, concat_to_interleave,
        cu_seqlens_m, cos, sin, qkv_raw, q_weighted, k_weighted, q_rstd, k_rstd,
    )
    ctx.heads_num = heads_num
    ctx.head_dim = head_dim
    ctx.eps = eps
    # qkv_raw/q_weighted/k_weighted/q_rstd/k_rstd are internal saved state, not
    # part of the public API (grouped_qkv_proj discards them) -- no gradient
    # flows into them from outside this op.
    ctx.mark_non_differentiable(qkv_raw, q_weighted, k_weighted, q_rstd, k_rstd)


def _grouped_qkv_proj_backward(ctx, grad_q_out, grad_k_out, grad_v_out, *_grad_extra):
    (x_mod, weight, q_weight, k_weight, interleave_to_concat, concat_to_interleave,
     cu_seqlens_m, cos, sin, qkv_raw, q_weighted, k_weighted, q_rstd, k_rstd) = ctx.saved_tensors
    heads_num, head_dim, eps = ctx.heads_num, ctx.head_dim, ctx.eps
    hidden = heads_num * head_dim

    # ---- step 3 backward (rope + rstd-multiply), Q and K independently ----
    grad_xw_q, grad_rstd_q = qkv_rstd_rope_backward(grad_q_out, q_weighted, q_rstd, cos, sin, heads_num, head_dim)
    grad_xw_k, grad_rstd_k = qkv_rstd_rope_backward(grad_k_out, k_weighted, k_rstd, cos, sin, heads_num, head_dim)

    # ---- step 2 backward (weight-undo + RMSNorm self-referential term),
    # fused with the permute-undo gather: grad_qkv_raw comes out in concat
    # order already, Q/K columns filled ----
    grad_qkv_raw, grad_q_weight, grad_k_weight = qkv_permute_rms_backward(
        qkv_raw, q_weight, k_weight, interleave_to_concat, cu_seqlens_m,
        grad_xw_q, grad_xw_k, grad_rstd_q, grad_rstd_k, q_rstd, k_rstd,
        heads_num, head_dim, eps,
    )
    grad_q_weight = grad_q_weight.to(q_weight.dtype)
    grad_k_weight = grad_k_weight.to(k_weight.dtype)

    # ---- V: no norm, so its permute-undo is a plain gather (forward's
    # index_select(0, concat_to_interleave) inverted the same way as above:
    # forward's gather source becomes backward's... here V's forward gathered
    # FROM concat INTO interleave via concat_to_interleave, so backward gathers
    # FROM interleave INTO concat via interleave_to_concat, its inverse) ----
    grad_v_raw = grad_v_out.index_select(0, interleave_to_concat.long()).reshape(-1, hidden)
    grad_qkv_raw[:, 2 * hidden:3 * hidden] = grad_v_raw.to(grad_qkv_raw.dtype)

    # ---- step 1 backward (linear) -- grad_qkv_raw is already in concat order,
    # no gather_A needed ----
    weight_compute = weight.to(grad_qkv_raw.dtype)
    grad_x_mod = quack_gemm(grad_qkv_raw, weight_compute, cu_seqlens_m=cu_seqlens_m)
    grad_weight_T = quack_gemm(
        x_mod.transpose(0, 1), grad_qkv_raw, cu_seqlens_k=cu_seqlens_m, out_dtype=weight.dtype,
    )
    grad_weight = grad_weight_T.transpose(-1, -2)
    grad_bias = segment_sum(grad_qkv_raw, cu_seqlens_m, weight.dtype)

    return (
        grad_x_mod, grad_weight, grad_bias, grad_q_weight, grad_k_weight,
        None, None, None, None, None, None, None, None,
    )


_grouped_qkv_proj_fwd.register_autograd(
    _grouped_qkv_proj_backward, setup_context=_grouped_qkv_proj_setup_context
)


def grouped_qkv_proj(
    x_mod: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    interleave_to_concat: torch.Tensor,
    concat_to_interleave: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads_num: int,
    head_dim: int,
    eps: float = 1e-6,
):
    """Fused qkv_proj for the packed/concat path: linear -> per-head RMSNorm
    (Q/K only) -> permute (concat<->interleave) -> RoPE (Q/K only), as one
    autograd op. Returns (q, k, v), each [total_seqlen, heads_num, head_dim] in
    interleave layout, ready for flash attention.
    """
    x_mod = autocast_cast(x_mod)
    q_out, k_out, v_out, *_ = _grouped_qkv_proj_fwd(
        x_mod, weight, bias, q_weight, k_weight, interleave_to_concat, concat_to_interleave,
        cu_seqlens_m, cos, sin, heads_num, head_dim, eps,
    )
    return q_out, k_out, v_out


# ---- grouped_ffn: the per-modality (video/text) GELU(tanh) MLP (linear -> act
# -> linear) fused into one autograd op, avoiding the host-side total_video/
# total_active .item() split that the naive x[:total_video]/x[total_video:...]
# slicing needs. video and text have their own weights (video_hidden can be
# wider than text_hidden, e.g. 8x vs 4x), so unlike grouped_out_proj
# this is NOT a single batched (L=2) gemm -- it's two gemm_act
# / gemm_dact calls (one per modality), each restricted to its row range via a
# 2-element slice of cu_seqlens_m (cu_seqlens_m[0:2] = [0, total_video] for
# video, cu_seqlens_m[1:3] = [total_video, total_active] for text), exactly the
# range_linear pattern generalized to a non-zero-start window. This sidesteps
# ever reading cu_seqlens_m with .item().
#
# The up-projection's post-activation z lands in a single [total_seqlen,
# video_hidden] buffer (video writes all columns, text writes only its first
# text_hidden columns -- requires video_hidden >= text_hidden, asserted by the
# caller); the down-projection then reads back out of that same buffer (full
# width for video, [:, :text_hidden] for text) into a shared [total_seqlen,
# hidden] output, again one gemm call per modality. Padding rows (beyond
# cu_seqlens_m's last boundary) are left uninitialized throughout -- nothing
# downstream reads them (same reasoning as grouped_out_proj/range_linear).
#
# Backward only saves `preact` (the up-projection's pre-activation, video-width)
# from forward, NOT the post-activation z -- gemm_dact recomputes z as a byproduct
# of fusing the down-projection's dz = grad_y @ w_down with the activation
# derivative (dact(preact) * dz -> grad_u), so the down-projection's own weight
# gradient (z^T @ grad_y) can reuse that recomputed z instead of an extra saved
# tensor. This is the whole point of gemm_dact over a plain gemm + elementwise
# activation-derivative multiply: one kernel launch instead of two, and one
# fewer saved activation.
_FFN_ACTIVATION = "gelu_tanh_approx"


@torch.library.custom_op("gmlp::grouped_ffn_fwd", mutates_args=())
def _grouped_ffn_fwd(
    x: torch.Tensor,             # [total_seqlen, hidden]
    w_up_video: torch.Tensor,    # [video_hidden, hidden]
    b_up_video: torch.Tensor,    # [video_hidden]
    w_down_video: torch.Tensor,  # [hidden, video_hidden]
    b_down_video: torch.Tensor,  # [hidden]
    w_up_text: torch.Tensor,     # [text_hidden, hidden]
    b_up_text: torch.Tensor,     # [text_hidden]
    w_down_text: torch.Tensor,   # [hidden, text_hidden]
    b_down_text: torch.Tensor,   # [hidden]
    cu_seqlens_m: torch.Tensor,  # [3] int32: [0, total_video, total_active]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_seqlen, hidden = x.shape
    video_hidden = w_up_video.shape[0]
    text_hidden = w_up_text.shape[0]
    compute_dtype = x.dtype

    video_seg = cu_seqlens_m[0:2]
    # A non-zero-start slice (cu_seqlens_m[1:3]) is contiguous but shares
    # cu_seqlens_m's storage at a +4-byte offset, which fails the varlen gemm
    # kernel's 16-byte alignment requirement on the cu_seqlens pointer -- stack
    # into a fresh (aligned) tensor instead. No .item()/host sync involved.
    # Returned below (not just used locally) so backward can reuse it instead
    # of launching a second stack kernel for the same two scalars.
    text_seg = torch.stack([cu_seqlens_m[1], cu_seqlens_m[2]])

    # preact/z are allocated video_hidden-wide; text only ever touches the
    # first text_hidden columns (video_hidden >= text_hidden, asserted by caller).
    preact = torch.empty(total_seqlen, video_hidden, dtype=compute_dtype, device=x.device)
    z = torch.empty(total_seqlen, video_hidden, dtype=compute_dtype, device=x.device)

    Bu_video = w_up_video.transpose(-1, -2).to(compute_dtype)
    Bu_text = w_up_text.transpose(-1, -2).to(compute_dtype)

    gemm_act(
        x, _varlen_b(Bu_video, video_seg), bias=b_up_video.to(compute_dtype), activation=_FFN_ACTIVATION,
        preact_out=preact, postact_out=z, cu_seqlens_m=video_seg,
    )
    gemm_act(
        x, _varlen_b(Bu_text, text_seg), bias=b_up_text.to(compute_dtype), activation=_FFN_ACTIVATION,
        preact_out=preact[:, :text_hidden], postact_out=z[:, :text_hidden], cu_seqlens_m=text_seg,
    )

    y = torch.empty(total_seqlen, hidden, dtype=compute_dtype, device=x.device)
    Bd_video = w_down_video.transpose(-1, -2).to(compute_dtype)
    Bd_text = w_down_text.transpose(-1, -2).to(compute_dtype)

    quack_gemm(z, _varlen_b(Bd_video, video_seg), out=y,
               bias=b_down_video.to(compute_dtype), cu_seqlens_m=video_seg)
    quack_gemm(z[:, :text_hidden], _varlen_b(Bd_text, text_seg), out=y,
               bias=b_down_text.to(compute_dtype), cu_seqlens_m=text_seg)

    return y, preact, text_seg


@_grouped_ffn_fwd.register_fake
def _(x, w_up_video, b_up_video, w_down_video, b_down_video,
      w_up_text, b_up_text, w_down_text, b_down_text, cu_seqlens_m):
    total_seqlen, hidden = x.shape
    video_hidden = w_up_video.shape[0]
    y = x.new_empty(total_seqlen, hidden)
    preact = x.new_empty(total_seqlen, video_hidden)
    text_seg = cu_seqlens_m.new_empty(2)
    return y, preact, text_seg


def _grouped_ffn_setup_context(ctx, inputs, output):
    (x, w_up_video, b_up_video, w_down_video, b_down_video,
     w_up_text, b_up_text, w_down_text, b_down_text, cu_seqlens_m) = inputs
    y, preact, text_seg = output
    ctx.save_for_backward(x, w_up_video, w_down_video, w_up_text, w_down_text, cu_seqlens_m, preact, text_seg)
    # preact/text_seg are internal saved state (grouped_ffn discards them) --
    # no external gradient flows into either.
    ctx.mark_non_differentiable(preact, text_seg)


def _grouped_ffn_backward(ctx, grad_y, _grad_preact, _grad_text_seg):
    x, w_up_video, w_down_video, w_up_text, w_down_text, cu_seqlens_m, preact, text_seg = ctx.saved_tensors
    total_seqlen, hidden = x.shape
    video_hidden = w_up_video.shape[0]
    text_hidden = w_up_text.shape[0]

    video_seg = cu_seqlens_m[0:2]
    # text_seg is reused from forward (saved via ctx) instead of stacking again.

    grad_y = grad_y.contiguous()
    compute_dtype = grad_y.dtype

    # ---- down-proj backward (dz = grad_y @ w_down) fused with the activation
    # derivative (grad_u = dact(preact) * dz) via gemm_dact; postact_out
    # recomputes z (not saved from forward) for the weight-gradient gemm below.
    # w_down_* is used AS-IS (not transposed): nn.Linear convention [out, in] is
    # already the [K=out, N=in] layout gemm's B expects for a dy@W backward.
    w_down_video_c = w_down_video.to(compute_dtype)
    w_down_text_c = w_down_text.to(compute_dtype)

    grad_u = torch.empty(total_seqlen, video_hidden, dtype=compute_dtype, device=grad_y.device)
    z = torch.empty(total_seqlen, video_hidden, dtype=compute_dtype, device=grad_y.device)

    gemm_dact(
        grad_y, _varlen_b(w_down_video_c, video_seg), preact, activation=_FFN_ACTIVATION,
        dx_out=grad_u, postact_out=z, cu_seqlens_m=video_seg,
    )
    gemm_dact(
        grad_y, _varlen_b(w_down_text_c, text_seg), preact[:, :text_hidden], activation=_FFN_ACTIVATION,
        dx_out=grad_u[:, :text_hidden], postact_out=z[:, :text_hidden], cu_seqlens_m=text_seg,
    )

    # grad_w_down: contraction over each modality's own row range via cu_seqlens_k
    # (mirrors grouped_out_proj's grad_weight, but L=1 per call since video/text
    # weights are separate parameters here rather than stacked along dim 0).
    grad_w_down_video = quack_gemm(
        z.transpose(0, 1), grad_y, cu_seqlens_k=video_seg, out_dtype=w_down_video.dtype,
    ).squeeze(0).transpose(-1, -2)
    grad_w_down_text = quack_gemm(
        z[:, :text_hidden].transpose(0, 1), grad_y, cu_seqlens_k=text_seg, out_dtype=w_down_text.dtype,
    ).squeeze(0).transpose(-1, -2)

    grad_b_down_video = segment_sum(grad_y, video_seg, w_down_video.dtype).squeeze(0)
    grad_b_down_text = segment_sum(grad_y, text_seg, w_down_text.dtype).squeeze(0)

    # ---- up-proj backward (plain linear, grad_u already carries the activation
    # derivative from gemm_dact above) ----
    grad_x = torch.empty(total_seqlen, hidden, dtype=compute_dtype, device=grad_y.device)
    quack_gemm(grad_u, _varlen_b(w_up_video.to(compute_dtype), video_seg),
               out=grad_x, cu_seqlens_m=video_seg)
    quack_gemm(grad_u[:, :text_hidden], _varlen_b(w_up_text.to(compute_dtype), text_seg),
               out=grad_x, cu_seqlens_m=text_seg)

    grad_w_up_video = quack_gemm(
        x.transpose(0, 1), grad_u, cu_seqlens_k=video_seg, out_dtype=w_up_video.dtype,
    ).squeeze(0).transpose(-1, -2)
    grad_w_up_text = quack_gemm(
        x.transpose(0, 1), grad_u[:, :text_hidden], cu_seqlens_k=text_seg, out_dtype=w_up_text.dtype,
    ).squeeze(0).transpose(-1, -2)

    grad_b_up_video = segment_sum(grad_u, video_seg, w_up_video.dtype).squeeze(0)
    grad_b_up_text = segment_sum(grad_u[:, :text_hidden], text_seg, w_up_text.dtype).squeeze(0)

    return (
        grad_x, grad_w_up_video, grad_b_up_video, grad_w_down_video, grad_b_down_video,
        grad_w_up_text, grad_b_up_text, grad_w_down_text, grad_b_down_text, None,
    )


_grouped_ffn_fwd.register_autograd(
    _grouped_ffn_backward, setup_context=_grouped_ffn_setup_context
)


def grouped_ffn(
    x: torch.Tensor,
    w_up_video: torch.Tensor,
    b_up_video: torch.Tensor,
    w_down_video: torch.Tensor,
    b_down_video: torch.Tensor,
    w_up_text: torch.Tensor,
    b_up_text: torch.Tensor,
    w_down_text: torch.Tensor,
    b_down_text: torch.Tensor,
    cu_seqlens_m: torch.Tensor,
) -> torch.Tensor:
    """Fused per-modality GELU(tanh) MLP (linear -> act -> linear) for the
    packed/concat path: one quack gemm_act + one quack gemm per modality in
    forward, replacing GroupedFFN.project's F.linear/F.gelu with the host-side
    total_video/total_active slice. cu_seqlens_m = [0, total_video,
    total_active] drives both modalities' row ranges without ever calling
    .item() on it. Requires w_up_video.shape[0] >= w_up_text.shape[0] (video's
    hidden width must be at least text's).
    """
    x = autocast_cast(x)
    y, _, _ = _grouped_ffn_fwd(
        x, w_up_video, b_up_video, w_down_video, b_down_video,
        w_up_text, b_up_text, w_down_text, b_down_text, cu_seqlens_m,
    )
    return y

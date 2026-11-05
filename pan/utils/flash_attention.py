# Copyright 2026 Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
# SPDX-License-Identifier: Apache-2.0
"""FlashAttention-4 entry points: a torch.library varlen op for packed training (visible to
selective activation checkpointing) and padded / dense attention for inference."""
from einops import rearrange
import torch

from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd


# ---- Register the varlen flash-attn call used by the packed training path as a
# torch.library custom op (rather than relying on flash-attn's own bare
# torch.autograd.Function) so activation checkpointing / SAC can see and target
# it by name -- mlp_ops.py applies the same pattern to the grouped projections.
# flash-attn-4's cute-dsl backend has no torch.library registration of its own,
# so without this it's opaque to SAC's torch_dispatch-based policy_fn.

@torch.library.custom_op("fav::flash_varlen_fwd", mutates_args=())
def _flash_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,  # unused here; threaded through so backward can read it back from ctx
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, _, _ = _flash_attn_fwd(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=True,
    )
    return out, lse


@_flash_varlen_fwd.register_fake
def _(q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
      max_seqlen_q, max_seqlen_k, softmax_scale, causal, deterministic):
    total_q, num_head = q.shape[0], q.shape[-2]
    out = q.new_empty(total_q, num_head, v.shape[-1])
    lse = q.new_empty(num_head, total_q, dtype=torch.float32)
    return out, lse


@torch.library.custom_op("fav::flash_varlen_bwd", mutates_args=())
def _flash_varlen_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None,
    causal: bool,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _flash_attn_bwd(
        q, k, v, out, dout, lse,
        softmax_scale, causal, 0.0,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        deterministic=deterministic,
    )
    return dq, dk, dv


@_flash_varlen_bwd.register_fake
def _(dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
      max_seqlen_q, max_seqlen_k, softmax_scale, causal, deterministic):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _flash_varlen_setup_context(ctx, inputs, output):
    (q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
     max_seqlen_q, max_seqlen_k, softmax_scale, causal, deterministic) = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.deterministic = deterministic


def _flash_varlen_backward(ctx, dout, dlse):
    q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors
    dq, dk, dv = _flash_varlen_bwd(
        dout, q, k, v, out, lse,
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
        ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.softmax_scale, ctx.causal, ctx.deterministic,
    )
    return dq, dk, dv, None, None, None, None, None, None, None, None, None


_flash_varlen_fwd.register_autograd(_flash_varlen_backward, setup_context=_flash_varlen_setup_context)


def _gather_valid_rows(tensor, padding_mask):
    """Unpad without depending on the FA2-only ``flash_attn.bert_padding``."""
    lengths = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    return (
        tensor.flatten(0, 1).index_select(0, indices),
        indices,
        cu_seqlens,
        int(lengths.max().item()),
    )


def padded_flash_attention(
    query,
    key,
    value,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    """FA4 inference for padded or dense Q/KV, including rectangular attention.

    Inputs and outputs use [B, S, H, D]; True/nonzero mask entries are valid.
    Padded query positions are restored as zeros.
    """
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

    if dropout_p != 0.0:
        raise ValueError("FlashAttention-4 inference path does not support dropout")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("FlashAttention-4 Q/KV expects [B, S, H, D] tensors.")
    if key.shape != value.shape:
        raise ValueError("FlashAttention-4 requires matching key/value shapes.")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("FlashAttention-4 query and key/value batch/head shapes must match.")

    batch_size, query_len, _, _ = query.shape
    key_len = key.shape[1]
    masks = []
    for name, mask, tensor in (
        ("query_padding_mask", query_padding_mask, query),
        ("key_padding_mask", key_padding_mask, key),
    ):
        if mask is not None:
            if tuple(mask.shape) != tuple(tensor.shape[:2]):
                raise ValueError(f"{name} must have shape {tuple(tensor.shape[:2])}.")
            mask = mask.to(device=tensor.device, dtype=torch.bool)
        masks.append(mask)
    query_padding_mask, key_padding_mask = masks

    if query_len == 0:
        return torch.empty_like(query)
    if all(mask is None or bool(mask.all()) for mask in masks):
        output = flash_attn_func(
            query, key, value,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
        return output[0] if isinstance(output, tuple) else output

    if query_padding_mask is None:
        query_padding_mask = torch.ones(
            (batch_size, query_len), device=query.device, dtype=torch.bool
        )
    if key_padding_mask is None:
        key_padding_mask = torch.ones(
            (batch_size, key_len), device=key.device, dtype=torch.bool
        )
    query_unpad, query_indices, cu_seqlens_q, max_seqlen_q = _gather_valid_rows(
        query, query_padding_mask
    )
    key_unpad, key_indices, cu_seqlens_k, max_seqlen_k = _gather_valid_rows(
        key, key_padding_mask
    )
    value_unpad = value.flatten(0, 1).index_select(0, key_indices)
    output_unpad = flash_attn_varlen_func(
        query_unpad, key_unpad, value_unpad,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    if isinstance(output_unpad, tuple):
        output_unpad = output_unpad[0]
    output = output_unpad.new_zeros((batch_size * query_len, *output_unpad.shape[1:]))
    output.index_copy_(0, query_indices, output_unpad)
    return output.reshape(batch_size, query_len, *output_unpad.shape[1:])


def flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    """
    Run FlashAttention varlen directly on packed tokens using explicit segment boundaries.
    """
    output, _ = _flash_varlen_fwd(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        seqused_q, seqused_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale, causal, deterministic,
    )
    output = rearrange(output, "nnz h d -> nnz (h d)")
    return output

"""Ulysses context parallelism for the packed (THD) dense-attention path.

Layout contract
---------------
Within a cp group of size ``cp``, joint token i lives on cp_rank ``i % cp``
(STRIDED: a sequence of 15 at cp=4 shards as 012301230123012), so shard sizes
differ by at most one and no divisibility constraint exists on the sequence
length. Outside attention every rank holds its shard with all ``H`` heads; the
attention all-to-alls trade tokens for heads, so after the gather each rank
holds the FULL sequence for its own ``H / cp`` head block, runs the ordinary
varlen flash attention against the FULL-sequence ``cu_seqlens``, and the reverse
all-to-all restores token shards with all heads.

Grouped vs full order
---------------------
An all-to-all concatenates the received chunks in SOURCE-RANK order, so the
gathered sequence arrives "grouped" -- all of rank 0's tokens ``[0::cp]``, then
rank 1's ``[1::cp]``, and so on -- which is a permutation of the real sequence.
Attention needs the real order (its varlen ``cu_seqlens`` delimit samples there),
so the gather is followed by one ``index_select`` per tensor and the scatter is
preceded by its inverse. Both maps are cached per (length, cp, device).

The strided layout is what keeps each rank's shard representative: every rank
gets ~1/cp of EVERY sample's video and caption tokens, so no rank ends up with,
say, an empty text group in the per-modality grouped GEMMs. A contiguous split
would make the permutations unnecessary but does not have that property.

``dataset.cp_batch`` splits batches with the same :func:`shard_sizes`, which is
the single source of truth for both sides.
"""

import functools
from typing import Sequence

import torch
import torch.distributed as dist

from pan.utils.flash_attention import flash_attn_varlen


@functools.lru_cache(maxsize=64)
def shard_sizes(total: int, cp: int) -> tuple[int, ...]:
    """Rows of shard r = tokens ``[r::cp]``; sizes differ by at most one."""
    return tuple((total - r + cp - 1) // cp for r in range(cp))


@functools.lru_cache(maxsize=64)
def _perms(total: int, cp: int, device_str: str):
    """``(to_full, to_grouped)``: index_select maps between the two orders.

    grouped = cat over ranks r of full[r::cp]   (the all-to-all arrival order)
    full    = grouped.index_select(0, to_full)
    grouped = full.index_select(0, to_grouped)
    """
    device = torch.device(device_str)
    sizes = shard_sizes(total, cp)
    to_grouped = torch.cat([
        torch.arange(sizes[r], device=device) * cp + r for r in range(cp)])
    return torch.argsort(to_grouped), to_grouped


def _a2a(recv, send, out_splits, in_splits, group):
    return dist.all_to_all_single(
        recv, send, output_split_sizes=list(out_splits),
        input_split_sizes=list(in_splits), group=group, async_op=True,
    )


def _tokens_to_heads(x: torch.Tensor, sizes: Sequence[int], group):
    """[s_local, H, D] (my tokens, all heads) -> [S, H/cp, D] (all tokens, my heads).

    Returns ``(work, send, finish)``; the caller keeps ``send`` alive until
    ``work.wait()`` and then calls ``finish()``.
    """
    cp = len(sizes)
    s_local, H, D = x.shape
    hl = H // cp
    S = sum(sizes)
    # Destination-major, so each destination's payload is one contiguous chunk.
    send = x.reshape(s_local, cp, hl, D).transpose(0, 1).contiguous().view(-1)
    recv = send.new_empty(S * hl * D)
    work = _a2a(recv, send, [s * hl * D for s in sizes], [s_local * hl * D] * cp, group)
    to_full, _ = _perms(S, cp, str(x.device))
    # Chunks arrive in source-rank order, i.e. grouped; permute to the real order.
    return work, send, lambda: recv.view(S, hl, D).index_select(0, to_full)


def _heads_to_tokens(y: torch.Tensor, sizes: Sequence[int], group):
    """[S, H/cp, D] (all tokens, my heads) -> [s_local, H, D] (my tokens, all heads)."""
    cp = len(sizes)
    S, hl, D = y.shape
    s_local = sizes[dist.get_group_rank(group, dist.get_rank())]
    # Back to grouped order, which puts each destination's tokens together.
    _, to_grouped = _perms(S, cp, str(y.device))
    send = y.index_select(0, to_grouped).view(-1)
    recv = send.new_empty(cp * s_local * hl * D)
    work = _a2a(recv, send, [s_local * hl * D] * cp, [s * hl * D for s in sizes], group)
    # Source r contributed my tokens for head block r -> heads land at r * hl + j.
    # .contiguous() is load-bearing: at hl >= 2 the reshape must copy (it merges
    # non-adjacent dims), but at hl == 1 the size-1 dim drops for free and
    # reshape returns a strided VIEW -- which the triton RoPE/RMSNorm backward
    # kernels then read with contiguous [total, H, D] indexing (they never look
    # at strides). That was the one-head-per-rank gradient corruption.
    return work, send, lambda: (
        recv.view(cp, s_local, hl, D).transpose(0, 1).reshape(s_local, cp * hl, D).contiguous()
    )


class _SeqAllToAll(torch.autograd.Function):
    """Both Ulysses directions; each one's backward is the other one.

    Takes/returns a tuple of tensors so q, k and v ride on three collectives
    issued back to back and waited on once.
    """

    @staticmethod
    def forward(ctx, sizes, group, to_heads, *tensors):
        ctx.sizes, ctx.group, ctx.to_heads = sizes, group, to_heads
        fn = _tokens_to_heads if to_heads else _heads_to_tokens
        pending = [fn(t, sizes, group) for t in tensors]
        outs = []
        for work, _send, finish in pending:
            work.wait()
            outs.append(finish())
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grads):
        grads = _SeqAllToAll.apply(
            ctx.sizes, ctx.group, not ctx.to_heads, *[g.contiguous() for g in grads]
        )
        return (None, None, None) + tuple(grads)


@torch.compiler.disable
def ulysses_flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    joint_total: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Varlen flash attention over a cp-sharded packed sequence.

    Args:
        q, k, v: ``[s_local, H, D]`` -- this rank's strided token shard, all heads.
        cu_seqlens: FULL-sequence segment boundaries, the same tensor on every cp
            rank; attention runs on the whole sequence for a slice of the heads.
        max_seqlen: host upper bound on one segment's length (full sequence).
        joint_total: ``S``, the full joint packed length.
        group: the cp process group.

    Returns:
        ``[s_local, H * D]`` -- this rank's token shard again with all heads,
        exactly the shape ``flash_attn_varlen`` returns without cp.
    """
    cp = dist.get_world_size(group)
    H = q.shape[1]
    if H % cp != 0:
        raise ValueError(f"heads_num={H} must be divisible by cp={cp} for Ulysses.")
    sizes = shard_sizes(joint_total, cp)
    rank = dist.get_group_rank(group, dist.get_rank())
    if q.shape[0] != sizes[rank]:
        raise ValueError(
            f"local token count {q.shape[0]} != cp shard size {sizes[rank]} "
            f"(joint_total={joint_total}, cp={cp}); the batch was not split by "
            "dataset.cp_batch."
        )

    q_f, k_f, v_f = _SeqAllToAll.apply(sizes, group, True, q, k, v)
    attn = flash_attn_varlen(
        q_f, k_f, v_f,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )  # [S, (H/cp) * D]
    attn = attn.view(joint_total, H // cp, -1)
    (out,) = _SeqAllToAll.apply(sizes, group, False, attn)
    return out.reshape(q.shape[0], -1)

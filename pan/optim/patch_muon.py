"""PAN 2's Muon step, installed over ``torch.optim.Muon``'s inner loop by :func:`patch_muon`.

``torch.optim.Muon`` orthogonalizes one parameter at a time with plain matmuls.
Here the momentum update and the parameter step are fused Triton kernels
(``muon_triton``), Newton-Schulz runs on the Gram matrix with symmetric GEMMs
(batched across same-shape chunks), and FSDP-sharded parameters are gathered
with an ``all_to_all`` pipeline that overlaps communication with the
orthogonalization. The replaced function keeps PyTorch's keyword interface so
``torch.optim.Muon.step`` can call it unchanged; a param group's
``adjust_lr_fn`` may additionally be a ``(base_fn, chunk_dim, n_chunks)`` tuple
to orthogonalize a weight as ``n_chunks`` independent row blocks.
"""
from functools import lru_cache
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard, Placement
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from quack.gemm_interface import (
    gemm_symmetric,
    gemm_symmetric_out,
    gemm_out,
)
from .muon_triton import (
    muon_momentum_nesterov,
    step_param,
    batched_gram_newtonschulz,
    can_use_fused_gram_ns,
)


def _scalar_lr(lr: float | Tensor) -> float | Tensor:
    """A 1-element tensor lr is used as a 0-dim tensor (what the step kernels expect); floats pass through."""
    return lr.reshape(()) if isinstance(lr, Tensor) and lr.ndim > 0 else lr


def _require_matrix(grad: Tensor) -> None:
    if grad.ndim != 2:
        raise ValueError(f"Muon updates 2-D weights only, got a gradient of shape {tuple(grad.shape)}")


@torch.compiler.disable
def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float | Tensor,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float] | tuple[tuple[float, float, float], ...],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | tuple | None,
    has_complex: bool,
) -> None:
    lr = _scalar_lr(lr)
    if has_complex:
        raise ValueError("PAN 2's Muon step has no complex-parameter path")

    if isinstance(adjust_lr_fn, tuple):
        base_adjust_lr_fn, chunk_dim, total_chunks = adjust_lr_fn
    else:
        base_adjust_lr_fn = adjust_lr_fn
        chunk_dim = None
        total_chunks = None

    chunk_dim0_mesh = grads[0].device_mesh if grads and isinstance(grads[0], DTensor) else None
    chunk_dim0_evenly_shardable = chunk_dim == 0 and (
        chunk_dim0_mesh is None or total_chunks % chunk_dim0_mesh.get_group(-1).size() == 0
    )

    if chunk_dim0_evenly_shardable:
        # Newton-Schulz batches poorly at just num_local_chunks-per-param (kernel
        # util is low). Instead: precompute each param's local per-chunk shape
        # (R // num_local_chunks, N) up front with no GPU work and group params
        # that share that shape. Each group is then run to completion (fill via
        # muon_momentum_nesterov's `out` -> batched NS -> step_param) before
        # moving to the next group, so only one group's buffers are ever live at
        # once instead of all groups' simultaneously.
        assert nesterov
        coeffs = ns_coefficients

        keys: list[tuple[int, int]] = []
        num_local_chunks_list: list[int] = []
        offsets: list[int] = []
        device_meshes: list[DeviceMesh | None] = []
        placements_list: list[tuple[Placement, ...] | None] = []
        key_device: dict[tuple[int, int], torch.device] = {}
        groups: dict[tuple[int, int], list[int]] = {}
        group_cursor: dict[tuple[int, int], int] = {}

        for idx, grad in enumerate(grads):
            _require_matrix(grad)
            if isinstance(grad, DTensor):
                local_grad = grad.to_local()
                device_mesh = grad.device_mesh
                placements = grad.placements
                num_local_chunks = total_chunks // device_mesh.get_group(-1).size()
            else:
                local_grad = grad
                device_mesh = None
                placements = None
                num_local_chunks = total_chunks
            R, N = local_grad.shape
            assert R % num_local_chunks == 0, "# rows must be divisible by # local chunks (chunks must be same-shape)"

            key = (R // num_local_chunks, N)
            keys.append(key)
            num_local_chunks_list.append(num_local_chunks)
            device_meshes.append(device_mesh)
            placements_list.append(placements)
            key_device.setdefault(key, local_grad.device)

            offsets.append(group_cursor.get(key, 0))
            group_cursor[key] = group_cursor.get(key, 0) + num_local_chunks
            groups.setdefault(key, []).append(idx)

        for key, member_idxs in groups.items():
            rows_per_chunk, N = key
            gbuf = torch.empty((group_cursor[key], rows_per_chunk, N), dtype=torch.bfloat16, device=key_device[key])

            for idx in member_idxs:
                num_local_chunks = num_local_chunks_list[idx]
                offset = offsets[idx]
                out_slice = gbuf[offset:offset + num_local_chunks].reshape(num_local_chunks * rows_per_chunk, N)
                muon_momentum_nesterov(muon_momentum_bufs[idx], grads[idx], momentum, out=out_slice)

            ortho = _gram_newtonschulz_local(gbuf, coeffs, ns_steps, eps)

            for idx in member_idxs:
                num_local_chunks = num_local_chunks_list[idx]
                offset = offsets[idx]
                local_ortho = ortho[offset:offset + num_local_chunks].reshape(num_local_chunks * rows_per_chunk, N)
                device_mesh = device_meshes[idx]
                if device_mesh is not None:
                    update = DTensor.from_local(local_ortho, device_mesh=device_mesh, placements=placements_list[idx])
                else:
                    update = local_ortho
                step_param(params[idx], update, lr, weight_decay, base_adjust_lr_fn, key)
    elif chunk_dim == 1:
        assert False, "Chunking along dim 1 is not currently supported (previously supported without comm overlap)"
    else:
        dtensors: list[tuple[DTensor, DTensor]] = []
        device_mesh: DeviceMesh = None

        for param, grad, buf in zip(params, grads, muon_momentum_bufs):
            _require_matrix(grad)
            assert nesterov
            update = muon_momentum_nesterov(buf, grad, momentum, output_dtype=torch.bfloat16)

            coeffs = ns_coefficients

            if isinstance(update, DTensor):
                assert isinstance(update.placements[-1], Shard) and update.placements[-1].dim == 0, "Current Muon impl only supports DTensors sharded on dim 0"
                if device_mesh is None:
                    device_mesh = update.device_mesh
                else:
                    assert device_mesh == update.device_mesh, "Current Muon impl only supports all DTensors on the same device mesh"
                dtensors.append((param, update))
            else:
                if chunk_dim == 0:
                    batched_update = update.reshape(total_chunks, -1, update.size(-1))
                    update = _gram_newtonschulz_local(batched_update, coeffs, ns_steps, eps).reshape(update.shape)
                    shape = batched_update.shape[1:]
                else:
                    update = _gram_newtonschulz_local(update, coeffs, ns_steps, eps)
                    shape = update.shape
                step_param(param, update, lr, weight_decay, base_adjust_lr_fn, shape)

        dtensors = sorted(dtensors, key=lambda x: (x[0].numel(), x[0].shape[0]))

        if dtensors:
            world_size = device_mesh.size(-1)
            group = device_mesh.get_group(-1)
            rank = dist.get_rank(group)

            # ping/pong compute streams + a dedicated (normal) comm stream.
            streams = get_muon_streams()
            compute_streams = streams[:2]
            comm_stream = streams[2]

            # The send shards were produced on the main stream (buf.lerp_ / lerp above).
            for stream in streams:
                stream.wait_stream(torch.cuda.current_stream())

            chunks = [dtensors[s:s + world_size] for s in range(0, len(dtensors), world_size)]
            n = len(chunks)

            def prep_and_forward(chunk):
                """Allocate send/recv/full buffers for one chunk and issue its forward
                all_to_all on the comm stream.

                Mirrors the synchronous path: rank r gathers the full r-th tensor.
                `recv` views into `full`, so the inverse all_to_all can reuse it.
                Returns ((send, recv, full), forward-done event).
                """
                k = len(chunk)  # may be < world_size for the trailing chunk
                ref = chunk[0][1].to_local()  # device/dtype for newly-allocated buffers
                send = [chunk[j][1].to_local() if j < k else ref.new_empty(0) for j in range(world_size)]
                if rank < k:
                    full = ref.new_empty(chunk[rank][1].shape)
                    recv = list(full.chunk(world_size, dim=0))
                else:
                    full = None
                    recv = [ref.new_empty(0) for _ in range(world_size)]
                with torch.cuda.stream(comm_stream):
                    dist.all_to_all(recv, send, group=group)
                    return (send, recv, full), comm_stream.record_event()

            # depth-2 rolling pipeline: only the current and next chunk's `full` buffers
            # are live at once, so ping-pong two vars instead of length-n lists.
            cur_state, cur_fwd = prep_and_forward(chunks[0])  # prologue forward

            for i in range(n):
                compute_stream = compute_streams[i % 2]
                send, recv, full = cur_state

                # Prefetch the next forward so it overlaps NS(i) on the comm stream.
                if i + 1 < n:
                    next_state, next_fwd = prep_and_forward(chunks[i + 1])

                # NS(i): waits only on its own forward. NS on ping/pong may run
                # concurrently (NS is not serialized across the two compute streams).
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(cur_fwd)
                    if full is not None:
                        if chunk_dim == 0:
                            batched_full = full.reshape(total_chunks, -1, full.size(-1))
                            ortho = _gram_newtonschulz_local(batched_full, ns_coefficients, ns_steps, eps).reshape(full.shape)
                        else:
                            ortho = _gram_newtonschulz_local(full, ns_coefficients, ns_steps, eps)
                        same_tensor = (
                            full.data_ptr() == ortho.data_ptr()
                            and full.shape == ortho.shape
                            and full.stride() == ortho.stride()
                            and full.dtype == ortho.dtype
                        )
                        if not same_tensor:
                            full.copy_(ortho)  # cast back to input dtype, reusing full's storage
                            print(f"full and ortho are different tensors, copy back to full; this is numerically correct and the slowdown is marginal, but not expected")
                            print(f"full:  data_ptr={full.data_ptr()}, shape={full.shape}, stride={full.stride()}, dtype={full.dtype}")
                            print(f"ortho: data_ptr={ortho.data_ptr()}, shape={ortho.shape}, stride={ortho.stride()}, dtype={ortho.dtype}")
                            print(f"chunk_dim={chunk_dim}, total_chunks={total_chunks}")

                # Inverse all_to_all back to sharding; `recv` still views `full`.
                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_stream(compute_stream)
                    dist.all_to_all(send, recv, group=group)

                # upd's local data now holds the orthogonalized update.
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_stream(comm_stream)
                    for param, upd in chunks[i]:
                        if chunk_dim == 0:
                            shape = (param.shape[0] // total_chunks, *param.shape[1:])
                        else:
                            shape = param.shape
                        step_param(param, upd, lr, weight_decay, base_adjust_lr_fn, shape)

                # Advance the pipeline; old cur_state is dropped here (buffers freed).
                if i + 1 < n:
                    cur_state, cur_fwd = next_state, next_fwd

            for stream in streams:
                torch.cuda.current_stream().wait_stream(stream)


@lru_cache(maxsize=1)
def get_muon_streams():
    return [torch.cuda.Stream() for _ in range(3)] + [torch.cuda.Stream(priority=12)]

def _gram_newtonschulz_local(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float,
):
    """Orthogonalize ``grad`` (``[..., M, N]``, one matrix per batch entry) in bf16.

    Iterates on the Gram matrix ``R = X X^T`` and accumulates the polynomial
    ``Q`` so that only two GEMMs ever touch the full ``[M, N]`` matrix; after
    the second iteration ``Q`` is applied and the Gram matrix recomputed to keep
    bf16 round-off in check. A bf16 ``grad`` is normalized in place.
    """
    a, b, c = ns_coefficients
    reset_iterations = frozenset([2]) if ns_steps > 2 else frozenset()

    # Fused Triton fast path (SM100/B300): batched updates with a 128x128 gram.
    # Same mathematics; ~3x faster by folding the normalization and fusing the
    # per-matrix Newton-Schulz segment into one persistent kernel.
    if can_use_fused_gram_ns(grad):
        return batched_gram_newtonschulz(
            grad, ns_coefficients, ns_steps, eps, reset_iterations=reset_iterations
        )

    # Work on the bf16 matrices scaled to unit Frobenius norm, oriented so that
    # the Gram matrix is the small side (one orientation for the whole batch).
    X = grad if grad.dtype == torch.bfloat16 else grad.to(torch.bfloat16)
    X.div_(X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps))
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    *batch, M, N = X.shape

    R = torch.empty((*batch, M, M), dtype=X.dtype, device=X.device)
    Z = torch.empty((*batch, M, M), dtype=X.dtype, device=X.device)
    RZ = torch.empty((*batch, M, M), dtype=X.dtype, device=X.device)
    X2 = torch.empty_like(X)

    gemm_symmetric_out(X, X.mT, out=R)  # initial gram

    Q = None
    for i in range(ns_steps):
        if i in reset_iterations:
            # Apply Q to X, then recompute R.
            gemm_out(Q, X, out=X2)
            X, X2 = X2, X  # swap to avoid copy
            gemm_symmetric_out(X, X.mT, out=R)  # recompute gram
            Q = None

        gemm_symmetric_out(R, R, out=Z, C=R, alpha=c, beta=b)
        if Q is None:
            Q = Z.clone()
            # trick for per-matrix diagonal add of Q=a*I+Z (flatten each (M, M) slice)
            Q.flatten(-2)[..., ::M + 1].add_(a)
        else:
            Q = gemm_symmetric(Q, Z, C=Q, beta=a)

        if i < ns_steps - 1 and (i + 1) not in reset_iterations:
            gemm_symmetric_out(R, Z,  out=RZ, C=R,  alpha=1.0, beta=a)
            gemm_symmetric_out(Z, RZ, out=R,  C=RZ, alpha=1.0, beta=a)

    gemm_out(Q, X, out=X2)
    return X2.mT if transpose else X2


def patch_muon():
    torch.optim._muon._single_tensor_muon = _single_tensor_muon
    torch.optim._muon._zeropower_via_newtonschulz = _gram_newtonschulz_local

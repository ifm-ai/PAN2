# Copyright 2026 Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
# SPDX-License-Identifier: Apache-2.0
"""Process group and device meshes for one PAN 2 process.

The world is factored as ``dp x cp``:

* ``cp`` is Ulysses context parallelism. Within a cp group every rank holds a
  CONTIGUOUS slice of the packed token sequence (rank r owns joint tokens
  [start_r, start_r + size_r); sizes differ by at most one) and the attention
  all-to-alls trade tokens for heads, so the gathered sequence is the original
  one in order.
* ``dp`` is the data-parallel axis.

FSDP gets a mesh of its own, ``(dp_replicate, fsdp_shard)``: with
``hybrid_shard`` parameters are sharded over the GPUs of one node and replicated
across nodes, otherwise they are sharded over the whole world.

``init_parallel_layout`` builds all of this once per process;
``current_parallel_layout`` hands it to the rest of the code.
"""
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

_BACKEND = "cpu:gloo,cuda:nccl"


def _start_process_group() -> None:
    if "RANK" in os.environ:
        # Started by a launcher (torchrun, srun + torchrun): rendezvous through its env:// variables.
        dist.init_process_group(backend=_BACKEND)
    else:
        # A plain `python generate.py`: a group of one, so no address or port is needed.
        dist.init_process_group(backend=_BACKEND, store=dist.HashStore(), rank=0, world_size=1)


class ParallelLayout:
    def __init__(self, cp: int = 1, hybrid_shard: bool = True):
        _start_process_group()
        world = dist.get_world_size()
        if world % cp != 0:
            raise ValueError(f"world size {world} is not divisible by cp={cp}")
        self.cp = cp
        self.world_size = world
        self.hybrid_shard = hybrid_shard

        # Every rank must create the meshes (and so their process groups) in this same order.
        self._dp_cp = init_device_mesh("cuda", (world // cp, cp), mesh_dim_names=("dp", "cp"))
        per_shard = min(world, torch.cuda.device_count()) if hybrid_shard else world
        self.fsdp_mesh = init_device_mesh(
            "cuda", (world // per_shard, per_shard), mesh_dim_names=("dp_replicate", "fsdp_shard")
        )

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def dp_mesh(self):
        return self._dp_cp["dp"]

    @property
    def cp_group(self):
        """Process group of this rank's cp slice, or None without context parallelism."""
        return self._dp_cp["cp"].get_group() if self.cp_enabled else None

    @property
    def cp_rank(self) -> int:
        """Position within the cp group; the global rank without context parallelism."""
        return self._dp_cp["cp"].get_local_rank() if self.cp_enabled else dist.get_rank()


_layout = None


def init_parallel_layout(cp: int = 1, hybrid_shard: bool = True) -> ParallelLayout:
    """Initialize the process group and meshes for this process and remember them."""
    global _layout
    _layout = ParallelLayout(cp=cp, hybrid_shard=hybrid_shard)
    return _layout


def current_parallel_layout() -> ParallelLayout:
    """The layout from ``init_parallel_layout``; on first use without one, a plain layout with cp = 1."""
    if _layout is None:
        init_parallel_layout()
    return _layout

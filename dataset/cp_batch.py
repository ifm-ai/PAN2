"""Context-parallel batch distribution: dataloader on cp_rank 0 only.

Within a cp group only cp_rank 0 runs the packed dataloader (CSV streaming,
latent download, text encoder). Each step the assembled PackedBatch is broadcast
to the group and every rank takes its STRIDED shard of the joint token sequence
-- joint token j lives on cp_rank ``j % cp`` -- matching the layout contract in
:mod:`pan.utils.ulysses`.

Strided, not contiguous: it gives every rank ~1/cp of EVERY sample's video and
caption tokens. A contiguous split would let a shard land inside one sample's
video run, leaving that rank with an empty text group in the per-modality
grouped GEMMs, and it makes what a shard holds depend on how the packer happened
to bin the batch. The cost is the gather/scatter permutation inside the attention
(see pan.utils.ulysses).

Because the joint layout is per-sample ``v_i`` then ``t_i``, a strided shard is
still, within each sample, an in-order subset of that sample's video tokens
followed by an in-order subset of its caption tokens. So the local batch has the
SAME shape as any packed batch -- same segment count, same two layouts -- just
with shorter segments, and all the packed metadata is rebuilt in local
coordinates.

What is sharded vs replicated
-----------------------------
* sharded -- hidden_states / target (video tokens), freqs_cos / freqs_sin (joint
  tokens, already in interleave order).
* rebuilt in LOCAL coordinates -- video_cu_seqlens, cu_seqlens_concat,
  cu_seqlens_matmul and the interleave<->concat permutations. A sample can
  contribute zero rows to a rank (its block being shorter than cp), and then it
  keeps a zero-length local segment so the per-segment modulation rows still line
  up 1:1 with ``timestep``.
* replicated -- text_states and caption_cu_seqlens: the caption encoder
  runs varlen ATTENTION over each caption, so it must see whole captions; every
  cp rank runs it redundantly (captions are tiny) and the model gathers its
  output down to this rank's rows with ``cp_text_index``. Also replicated:
  timestep, and joint_cu_seqlens -- the ONLY consumer of the latter is flash
  attention, which under cp runs on the gathered full sequence.
* the loss needs no cp handling: PackedConcatMSELoss all-reduces (sum, count)
  with AVG over the world and divides, which is the exact global element mean
  for any dp x cp composition, and its backward's averaged-count divisor cancels
  FSDP's 1/world gradient averaging. Every rank contributes the partial
  derivative of the one global loss through the rows it owns, so those partials
  sum to the true gradient -- including for the redundantly-run caption encoder.
"""

import dataclasses

import torch
import torch.distributed as dist

from dataset.packed_dataloader import PackedBatch
from pan.utils.ulysses import shard_sizes


@dataclasses.dataclass
class _TSpec:
    shape: tuple
    dtype: torch.dtype


def _strip(obj):
    """Structure with every tensor replaced by its (shape, dtype) spec."""
    if torch.is_tensor(obj):
        return _TSpec(tuple(obj.shape), obj.dtype)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return type(obj)(**{f.name: _strip(getattr(obj, f.name))
                            for f in dataclasses.fields(obj)})
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items()}
    return obj


def _collect(obj, out):
    """Tensors (or specs) of a structure, in one deterministic order."""
    if torch.is_tensor(obj) or isinstance(obj, _TSpec):
        out.append(obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _collect(getattr(obj, f.name), out)
    elif isinstance(obj, dict):
        for k in sorted(obj):
            _collect(obj[k], out)
    return out


def _fill(obj, tensors_iter):
    """The stripped structure with specs replaced by tensors, same order."""
    if isinstance(obj, _TSpec):
        return next(tensors_iter)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return type(obj)(**{f.name: _fill(getattr(obj, f.name), tensors_iter)
                            for f in dataclasses.fields(obj)})
    if isinstance(obj, dict):
        return {k: _fill(obj[k], tensors_iter) for k in obj}
    return obj


def distribute_packed_batch(batch, meta, group, device=None):
    """Broadcast (PackedBatch, PackedDataloaderMeta) from cp_rank 0 of ``group``.

    cp_rank 0 passes its (batch, meta); the others pass (None, None) and get
    reconstructed copies back. ``meta`` rides in the object broadcast whole (it
    is small and partly host-side), minus ``dataloader_state``, which stays
    rank-0-only -- it is what the checkpoint records, and only cp_rank 0 ever
    resumes a dataloader from it.
    """
    src = dist.get_global_rank(group, 0)
    is_src = dist.get_rank() == src
    if is_src:
        # cp_meta is host-side and tiny: it rides in the object broadcast, not
        # the tensor broadcast (which would land it on the GPU).
        obj = [_strip(dataclasses.replace(batch, cp_meta=None)), batch.cp_meta,
               dataclasses.replace(meta, dataloader_state=None)]
    else:
        obj = [None, None, None]
    dist.broadcast_object_list(obj, src=src, group=group)
    spec, cp_meta, meta_spec = obj

    if is_src:
        tensors = [t.contiguous() for t in _collect(dataclasses.replace(batch, cp_meta=None), [])]
    else:
        device = device or torch.device("cuda", torch.cuda.current_device())
        tensors = [torch.empty(s.shape, dtype=s.dtype, device=device)
                   for s in _collect(spec, [])]
    works = [dist.broadcast(t, src=src, group=group, async_op=True) for t in tensors]
    for w in works:
        w.wait()
    if is_src:
        return batch, meta
    return dataclasses.replace(_fill(spec, iter(tensors)), cp_meta=cp_meta), meta_spec


def _strided_count(length, offset, cp):
    """Rows a `length`-long block keeps when sampled at stride cp from `offset`."""
    return max(0, -(-(length - offset) // cp)) if offset < length else 0


def strided_split_batch(batch: PackedBatch, cp: int, rank: int) -> PackedBatch:
    """This cp_rank's strided shard of a full PackedBatch.

    Rank r keeps joint tokens ``[r::cp]``. Inside sample i that means its video
    block ``[joint_cu[i], joint_cu[i] + vl)`` contributes rows at stride cp
    starting ``(r - joint_cu[i]) % cp`` into the block, and its caption block
    likewise -- so the shard stays an ordinary packed batch, one segment per
    sample, with shorter segments.
    """
    if batch.cp_meta is None:
        raise ValueError("strided_split_batch needs a FULL batch with cp_meta "
                         "(PackedIterator(cp_enabled=True)); got None -- already sharded?")
    p = batch.packed_seq_params
    dev = batch.hidden_states.device
    # Host-side segment lengths, so none of this arithmetic syncs on the GPU.
    video_cu = batch.cp_meta.video_cu_seqlens_cpu.tolist()
    caption_cu = batch.cp_meta.caption_cu_seqlens_cpu.tolist()
    n = len(video_cu) - 1
    # joint_cu[i] = video_cu[i] + caption_cu[i]: the joint layout is exactly the
    # two token streams interleaved per sample.
    joint_cu = [v + c for v, c in zip(video_cu, caption_cu)]
    V, L, S = video_cu[n], caption_cu[n], joint_cu[n]
    if (S, V, L) != (p.joint_packed_length, batch.hidden_states.shape[1],
                     batch.text_states.shape[0]):
        raise ValueError(
            f"batch shapes (joint={p.joint_packed_length}, video="
            f"{batch.hidden_states.shape[1]}, text={batch.text_states.shape[0]}) "
            f"disagree with the meta cu_seqlens ({S}, {V}, {L}); cp splitting "
            "assumes an unpadded packed batch.")

    # Per-sample local lengths, and the global rows they select.
    video_rows, text_rows, local_video, local_caption = [], [], [], []
    for i in range(n):
        vl = video_cu[i + 1] - video_cu[i]
        cl = caption_cu[i + 1] - caption_cu[i]
        v_off = (rank - joint_cu[i]) % cp                 # into the video block
        t_off = (rank - joint_cu[i] - vl) % cp            # into the caption block
        video_rows.append(range(video_cu[i] + v_off, video_cu[i + 1], cp))
        text_rows.append(range(caption_cu[i] + t_off, caption_cu[i + 1], cp))
        local_video.append(_strided_count(vl, v_off, cp))
        local_caption.append(_strided_count(cl, t_off, cp))

    n_v, n_t = sum(local_video), sum(local_caption)
    assert n_v + n_t == shard_sizes(S, cp)[rank], (n_v, n_t, S, cp, rank)
    if n_v == 0:
        # Only reachable when every sample's video block is shorter than cp.
        raise ValueError(
            f"cp shard {rank} of a {S}-token batch holds no video tokens "
            f"(cp={cp}); every sample's video block is shorter than cp. Use a "
            "smaller cp_size.")

    idx_video = torch.tensor([j for rows in video_rows for j in rows], dtype=torch.long)
    idx_text = torch.tensor([j for rows in text_rows for j in rows], dtype=torch.long)

    # interleave (local joint order) <-> concat ([video | text]) permutations,
    # mirroring PackedIterator._build_concat_layout_cpu in local coordinates.
    itc = torch.empty(n_v + n_t, dtype=torch.int32)
    cti = torch.empty(n_v + n_t, dtype=torch.int32)
    v0 = t0 = j0 = 0
    for vl, cl in zip(local_video, local_caption):
        for lo, hi, c0 in ((j0, j0 + vl, v0), (j0 + vl, j0 + vl + cl, n_v + t0)):
            itc[c0:c0 + hi - lo] = torch.arange(lo, hi, dtype=torch.int32)
            cti[lo:hi] = torch.arange(c0, c0 + hi - lo, dtype=torch.int32)
        v0, t0, j0 = v0 + vl, t0 + cl, j0 + vl + cl

    video_bounds = torch.tensor([0] + torch.tensor(local_video).cumsum(0).tolist(),
                                dtype=torch.int32, device=dev)   # [n + 1]
    idx_video, idx_text = idx_video.to(dev), idx_text.to(dev)
    new_p = dataclasses.replace(
        p,
        # LOCAL: one modulation row per sample, zero-length where this rank holds
        # none of that sample's video tokens.
        video_cu_seqlens=video_bounds,
        cu_seqlens_concat=torch.cat(
            [video_bounds, torch.tensor([n_v + n_t], dtype=torch.int32, device=dev)]),
        cu_seqlens_matmul=torch.tensor([0, n_v, n_v + n_t], dtype=torch.int32, device=dev),
        indices_interleave_to_concat=itc.pin_memory().to(device=dev, non_blocking=True),
        indices_concat_to_interleave=cti.pin_memory().to(device=dev, non_blocking=True),
        # freqs are already in joint interleave order, so the shard is the plain
        # stride (unlike the video/text rows, whose stride offset differs per
        # sample); .contiguous() because the RoPE kernels want packed rows.
        freqs_cos=p.freqs_cos[rank::cp].contiguous(),
        freqs_sin=p.freqs_sin[rank::cp].contiguous(),
        joint_packed_length=n_v + n_t,
        # GLOBAL, and deliberately so: joint_cu_seqlens / caption_cu_seqlens feed
        # the two ops that must see whole sequences (flash attention after the
        # Ulysses gather, and the caption encoder).
        cp_text_index=idx_text,
        cp_joint_total=S,
    )
    return PackedBatch(
        hidden_states=batch.hidden_states.index_select(1, idx_video),
        timestep=batch.timestep,
        text_states=batch.text_states,
        target=batch.target.index_select(0, idx_video),
        packed_seq_params=new_p,
        cp_meta=None,          # consumed: the shard is an ordinary packed batch
    )


def split_packed_batch_for_cp(batch, meta, parallel_state):
    """Broadcast then shard, the two steps the train loop needs each iteration."""
    batch, meta = distribute_packed_batch(batch, meta, parallel_state.cp_group)
    return strided_split_batch(batch, parallel_state.cp, parallel_state.cp_rank), meta

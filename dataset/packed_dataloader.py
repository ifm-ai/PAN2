"""
Part 3: GPU-side assembly (PackedIterator).

Consumes :class:`PackedPlan` objects produced by the packing worker,
copies the precomputed VAE latents to the GPU, assembles the final packed
batch, and yields ``(PackedBatch, PackedDataloaderMeta)`` pairs.  The VAE
forward is done offline; no encoding happens here.
"""

from copy import deepcopy
from dataclasses import dataclass
import operator
from typing import Callable, Iterator, List, Optional, Tuple

from einops import rearrange
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedModel

from pan.models.text_encoders import build_cfg_empty_video_token_ids
from pan.models.transformers.pan_transformer import PackedSequenceParams

from .packing_dataset import (
    DataloaderState,
    PackedPlan,
    PackingDataset,
)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class CpMeta:
    """What context-parallel sharding needs that the batch only holds on GPU.

    Host int32 copies of the per-sample boundaries, [num_sequences + 1] each.
    dataset.cp_batch.strided_split_batch derives every rank's local packed
    metadata (segment counts, index tensors -- i.e. tensor SHAPES) from these,
    which needs Python ints; reading them back from the batch's CUDA tensors
    would be a device sync in the train loop every step. Present only on a
    FULL batch under cp: None when cp is off, and None again on the per-rank
    shard the splitter returns.
    """
    video_cu_seqlens_cpu: torch.Tensor
    caption_cu_seqlens_cpu: torch.Tensor


@dataclass
class PackedBatch:
    """Output of the GPU decode + assembly step."""
    hidden_states: torch.Tensor
    timestep: torch.Tensor
    text_states: torch.Tensor
    target: torch.Tensor
    packed_seq_params: PackedSequenceParams
    cp_meta: CpMeta | None = None


@dataclass
class PackedDataloaderMeta:
    """Sidecar metadata yielded alongside each PackedBatch."""
    active_sequence_count: int
    active_joint_length: int
    active_video_length: int
    active_caption_length: int
    active_image_count: int = 0  # image-modality sequences (videos = seq_count - this)
    dataloader_state: DataloaderState | None = None


# ---------------------------------------------------------------------------
# PackedIterator
# ---------------------------------------------------------------------------


class PackedIterator:
    """
    End-to-end packed latent loader.

    A single packing worker (DataLoader ``num_workers=1``) downloads the
    precomputed VAE latents via a thread pool, bin-packs them, and builds the
    CPU RoPE tensors.  The main thread copies the latents to GPU and assembles
    the final packed batch — no VAE forward is run.

    Args:
        num_workers: Thread count passed to the packing worker's internal
            latent-download thread pool (Part 1).  The packing DataLoader
            itself always uses exactly one worker.
    """

    def __init__(
        self,
        csv_paths: List[str],
        text_encoder: PreTrainedModel,
        text_tokenizer_path: str,
        text_template_video_cfg: dict,
        timestep_sample_fn: Callable[[int, torch.device, torch.Generator], torch.Tensor],
        timestep_transform_fn: Callable[..., torch.Tensor],
        add_noise_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
        rope_embed_fn: Callable[[tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]],
        random_seed: int = 42,
        cfg_drop_prob: float = 0.0,
        image_cond_prob: float = 0.0,
        video_cond_prob: float = 0.0,
        conditioning_schema: str = "token_concat",
        num_buffers: int = 4,
        packed_video_token_budget: int = 8192,
        packed_joint_token_budget: int = 8192,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        resize: tuple[int, int] = (256, 256),
        num_workers: int = 4,
        prefetch_factor: int = 2,
        dtype: torch.dtype = torch.float16,
        max_num_seq: int = 16,
        packed_caption_token_budget: int = 2048,
        packed_caption_segment_limit: int = 256,
        packed_joint_segment_limit: int = 2048,
        image_tar_shards: List[str] | None = None,
        image_sample_prob: float = 0.0,
        resume_state: DataloaderState | None = None,
        cp_enabled: bool = False,
    ):
        self.caption_segment_limit = packed_caption_segment_limit
        self.joint_segment_limit = packed_joint_segment_limit
        self.patch_size = patch_size
        self.resize = resize
        self.timestep_sample_fn = timestep_sample_fn
        self.timestep_transform_fn = timestep_transform_fn
        if conditioning_schema not in {"token_concat", "channel_concat"}:
            raise ValueError(
                f"Unsupported conditioning_schema {conditioning_schema!r}; "
                "expected 'token_concat' or 'channel_concat'."
            )
        self.conditioning_schema = conditioning_schema
        self.add_noise_fn = add_noise_fn
        self.dtype = dtype
        self.rng = torch.Generator(device="cpu")
        self.cuda_rng = torch.Generator(device="cuda")
        self.rng.manual_seed(random_seed)
        self.cuda_rng.manual_seed(random_seed)
        self.video_copy_stream = torch.cuda.current_stream()
        self.video_stream = torch.cuda.current_stream()
        self.meta_stream = torch.cuda.current_stream()
        self.text_stream = torch.cuda.current_stream()

        self.pending_cuda_frames: Optional[List[torch.Tensor]] = None
        self.text_template_video_cfg = deepcopy(text_template_video_cfg)
        text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, padding_side="right")
        build_cfg_empty_video_token_ids(
            tokenizer=text_tokenizer,
            template_cfg=self.text_template_video_cfg,
            max_caption_tokens=packed_caption_segment_limit,
        )
        self.text_template_prefix_len = self.text_template_video_cfg.get("prefix_len", 0)
        self.qwen_token_budget = packed_caption_token_budget + max_num_seq * self.text_template_prefix_len
        self.qwen_segment_limit = packed_caption_segment_limit + self.text_template_prefix_len

        self.max_num_seq = max_num_seq
        # Under context parallelism the batch also carries CpMeta (host copies of
        # the boundaries) for dataset.cp_batch; off otherwise, so a non-cp batch
        # is exactly what it always was.
        self.cp_enabled = cp_enabled

        self._dataset = PackingDataset(
            csv_paths=csv_paths,
            text_tokenizer_path=text_tokenizer_path,
            text_template_video_cfg=deepcopy(self.text_template_video_cfg),
            rope_embed_fn=rope_embed_fn,
            patch_size=patch_size,
            resize=resize,
            dtype=dtype,
            num_buffers=num_buffers,
            max_tokens=packed_joint_token_budget,
            qwen_token_budget=self.qwen_token_budget,
            packed_video_token_budget=packed_video_token_budget,
            packed_caption_token_budget=packed_caption_token_budget,
            random_seed=random_seed,
            cfg_drop_prob=cfg_drop_prob,
            image_cond_prob=image_cond_prob,
            video_cond_prob=video_cond_prob,
            caption_segment_limit=packed_caption_segment_limit,
            qwen_segment_limit=self.qwen_segment_limit,
            joint_segment_limit=self.joint_segment_limit,
            conditioning_schema=conditioning_schema,
            max_num_seq=max_num_seq,
            num_decode_workers=num_workers,
            image_tar_shards=image_tar_shards,
            image_sample_prob=image_sample_prob,
            resume_state=resume_state,
        )

        self._loader = DataLoader(
            self._dataset,
            batch_size=1,
            num_workers=1,
            collate_fn=operator.itemgetter(0),
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )

        self._text_encoder = text_encoder

    def _prepare_next_packed_latents(self, latents: list[torch.Tensor]):
        """Kick off async H2D copy for the next batch's precomputed latents."""
        self.video_copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.video_copy_stream), torch.cuda.nvtx.range("copy latents"):
            self.pending_cuda_frames = [
                t.to(device="cuda", non_blocking=True) for t in latents
            ]

    # -- public iterator ---------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[PackedBatch, PackedDataloaderMeta]]:
        ret = None

        for plan in self._loader:
            self._prepare_next_packed_latents(plan.decoded_frames)
            if ret is not None:
                yield ret
            ret = self._assemble(plan)

    def _encode_text(
        self,
        qwen_input_ids: torch.Tensor,
        qwen_position_ids: torch.Tensor,
        qwen_cu_seqlens: torch.Tensor,
        qwen_seq_idx: torch.Tensor,
        caption_row_indices: torch.Tensor,
    ):
        qwen_cu_seqlens = qwen_cu_seqlens.to(device="cuda", non_blocking=True)
        full_output = self._text_encoder(
            qwen_input_ids.to(device="cuda", non_blocking=True).unsqueeze(0),
            position_ids=qwen_position_ids.to(device="cuda", non_blocking=True).unsqueeze(0),
            seq_idx=qwen_seq_idx.to(device="cuda", non_blocking=True).unsqueeze(0),
            cu_seq_lens_q=qwen_cu_seqlens,
            cu_seq_lens_k=qwen_cu_seqlens,
            max_length_q=self.qwen_segment_limit,
            max_length_k=self.qwen_segment_limit,
            use_cache=False,
        ).last_hidden_state[0].to(self.dtype)
        return full_output.index_select(0, caption_row_indices.to(device="cuda", non_blocking=True))

    def _build_concat_layout_cpu(
        self,
        plan: PackedPlan,
        num_sequences: int,
        joint_packed_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build interleave<->concat layout metadata on CPU (no CUDA sync).

        Two packed token layouts share the same set of tokens:
            interleave: v1, t1, v2, t2, ..., vn, tn, padding
            concat:     v1, v2, ..., vn, t1, t2, ..., tn, padding

        Everything is derived from the CPU plan metadata and pinned so the H2D
        copies in ``_assemble`` stay async (non_blocking) without forcing a sync.

        Returns (interleave_to_concat, concat_to_interleave, cu_seqlens_concat,
        cu_seqlens_matmul) where the index tensors satisfy
        ``concat = interleave[interleave_to_concat]`` and
        ``interleave = concat[concat_to_interleave]``.
        """
        n = num_sequences
        video_cu = plan.video_cu_seqlens
        caption_cu = plan.caption_cu_seqlens
        joint_cu = plan.joint_cu_seqlens
        total_video = int(video_cu[n].item())
        total_caption = int(caption_cu[n].item())

        # arange init makes the trailing padding region an identity (no-op) part of
        # the permutation, so the gather is bijective and backward stays correct.
        # int32 throughout (not torch.long): these only ever feed index_select /
        # quack's A_idx (both accept int32), and .index_copy_ callers cast to long
        # locally where that specific op still requires it.
        interleave_to_concat = torch.arange(joint_packed_length, dtype=torch.int32)
        concat_to_interleave = torch.arange(joint_packed_length, dtype=torch.int32)
        for i in range(n):
            v0 = int(video_cu[i].item())
            v1 = int(video_cu[i + 1].item())
            c0 = int(caption_cu[i].item())
            c1 = int(caption_cu[i + 1].item())
            j0 = int(joint_cu[i].item())
            vl = v1 - v0
            cl = c1 - c0
            # interleave: [j0, j0+vl) video then [j0+vl, j0+vl+cl) caption
            vid_interleave = torch.arange(j0, j0 + vl, dtype=torch.int32)
            cap_interleave = torch.arange(j0 + vl, j0 + vl + cl, dtype=torch.int32)
            # concat: [v0, v1) video then [total_video+c0, total_video+c1) caption
            vid_concat = torch.arange(v0, v1, dtype=torch.int32)
            cap_concat = torch.arange(total_video + c0, total_video + c1, dtype=torch.int32)
            # concat[p] = interleave[interleave_to_concat[p]]
            interleave_to_concat[v0:v1] = vid_interleave
            interleave_to_concat[total_video + c0:total_video + c1] = cap_interleave
            # interleave[q] = concat[concat_to_interleave[q]]
            concat_to_interleave[j0:j0 + vl] = vid_concat
            concat_to_interleave[j0 + vl:j0 + vl + cl] = cap_concat

        # Per-video segment boundaries plus ONE merged text segment, in the concat
        # layout: [n + 2] (n video segments + 1 text segment). Text modulation is
        # timestep-independent (a single shared value across all captions), so all
        # caption tokens can share one segment in packed_modulate / packed_gate
        # -- both are per-token elementwise with a per-segment broadcast param, with
        # no cross-token reduction in the forward, so merging is exact. The n video
        # segments align 1:1 with cat([video_mod, single_text_mod])'s n video rows.
        video_bounds = video_cu[:n + 1].to(torch.int32)                       # [n + 1]: 0 .. total_video
        text_bound = torch.tensor([total_video + total_caption], dtype=torch.int32)  # [1]: end of merged text segment
        cu_seqlens_concat = torch.cat([video_bounds, text_bound])
        # Grouped-matmul boundaries over the concat layout: [0, total_video, total_video + total_caption].
        cu_seqlens_matmul = torch.tensor(
            [0, total_video, total_video + total_caption], dtype=torch.int32
        )

        return (
            interleave_to_concat.pin_memory(),
            concat_to_interleave.pin_memory(),
            cu_seqlens_concat.pin_memory(),
            cu_seqlens_matmul.pin_memory(),
        )

    @torch.no_grad()
    def _assemble(self, plan: PackedPlan) -> Tuple[PackedBatch, PackedDataloaderMeta]:
        """Copy latents to GPU, run text encoder, assemble packed batch."""
        num_sequences = len(plan.decoded_frames)
        joint_packed_length = int(plan.joint_cu_seqlens[-1].item())
        (
            interleave_to_concat_cpu,
            concat_to_interleave_cpu,
            cu_seqlens_concat_cpu,
            cu_seqlens_matmul_cpu,
        ) = self._build_concat_layout_cpu(plan, num_sequences, joint_packed_length)

        with torch.cuda.stream(self.text_stream), torch.cuda.nvtx.range("assemble text"):
            caption_states = self._encode_text(
                plan.qwen_input_ids,
                plan.qwen_position_ids,
                plan.qwen_cu_seqlens,
                plan.qwen_seq_idx,
                plan.caption_row_indices,
            )

        with torch.cuda.stream(self.meta_stream), torch.cuda.nvtx.range("assemble cu_seqlens"):
            video_cu_seqlens = plan.video_cu_seqlens.to(device="cuda", non_blocking=True)
            caption_cu_seqlens = plan.caption_cu_seqlens.to(device="cuda", non_blocking=True)
            joint_cu_seqlens = plan.joint_cu_seqlens.to(device="cuda", non_blocking=True)
            indices_interleave_to_concat = interleave_to_concat_cpu.to(device="cuda", non_blocking=True)
            indices_concat_to_interleave = concat_to_interleave_cpu.to(device="cuda", non_blocking=True)
            cu_seqlens_concat = cu_seqlens_concat_cpu.to(device="cuda", non_blocking=True)
            cu_seqlens_matmul = cu_seqlens_matmul_cpu.to(device="cuda", non_blocking=True)
            cos = plan.freqs_cos.to(device="cuda", non_blocking=True)
            sin = plan.freqs_sin.to(device="cuda", non_blocking=True)

        with torch.cuda.stream(self.video_stream), torch.cuda.nvtx.range("assemble video"):
            active_sequence_count = torch.empty((), device="cuda", dtype=torch.int32).fill_(num_sequences)

            # Latents are precomputed offline; just consume the GPU copies.
            torch.cuda.current_stream().wait_stream(self.video_copy_stream)
            latents = self.pending_cuda_frames  # one [C, T, H, W] latent per sample, on GPU
            pt, ph, pw = self.patch_size

            # Per-sample timestep transform uses each latent's own grid, so
            # variable resolutions are supported.
            timestep_latent_shapes = torch.tensor(
                [[l.shape[1] // pt, l.shape[2] // ph, l.shape[3] // pw] for l in latents],
                dtype=torch.float32,
                device="cpu",
            )
            timesteps = self.timestep_sample_fn(num_sequences, torch.device("cpu"), self.rng)
            timesteps = self.timestep_transform_fn(
                timesteps,
                latent_shapes=timestep_latent_shapes,
            ).to(self.dtype)

            timesteps_cuda = timesteps.to(device="cuda", non_blocking=True)

            # One noisy timestep per sample drives the video modulation (vec).
            # Channel-concat supplies clean context via condition channels, so every
            # frame is fully noised and there is no clean/noisy interleave anymore.
            model_timesteps = timesteps.to(device="cuda", non_blocking=True)

            # Patchify each latent individually so samples of different H,W pack
            # into one flat token sequence (pt == 1, enforced in the plan). Each
            # token's features are the C*pt*ph*pw values inside its patch, in the
            # same (C pt ph pw) order as the Conv3d patch_embed weight; tokens are in
            # (T H W) order, matching the per-sample RoPE in _build_packed_plan.
            channel_concat = self.conditioning_schema == "channel_concat"
            noised_chunks: List[torch.Tensor] = []   # each [C*pt*ph*pw, n_i]
            target_chunks: List[torch.Tensor] = []   # each [n_i, C*pt*ph*pw]
            for i, latent in enumerate(latents):
                _, T, H, W = latent.shape
                tpf = (H // ph) * (W // pw)
                # Channel-concat fully noises every frame (clean context is supplied
                # via the condition channels below), so all frames share the sample's
                # single noisy timestep.
                per_frame_t = timesteps_cuda[i].repeat(T)
                noise = torch.empty_like(latent).normal_(generator=self.cuda_rng)
                noised = self.add_noise_fn(latent, noise, per_frame_t, 1)
                target_lat = noise - latent

                model_input = noised
                if channel_concat:
                    # Append [clean-condition (C ch) | binary mask (1 ch)]; non-prefix
                    # frames stay zero so the model only sees the conditioned frames.
                    cond_steps = min(int(plan.condition_prefix_tokens[i]) // tpf, T) if tpf > 0 else 0
                    cond = noised.new_zeros((latent.shape[0] + 1, T, H, W))
                    if cond_steps > 0:
                        cond[:-1, :cond_steps] = latent[:, :cond_steps]
                        cond[-1:, :cond_steps] = 1.0
                    model_input = torch.cat([noised, cond], dim=0)  # [2C+1, T, H, W]

                noised_chunks.append(
                    rearrange(model_input, "C (T pt) (H ph) (W pw) -> (C pt ph pw) (T H W)", pt=pt, ph=ph, pw=pw)
                )
                target_chunks.append(
                    rearrange(target_lat, "C (T pt) (H ph) (W pw) -> (T H W) (C pt ph pw)", pt=pt, ph=ph, pw=pw)
                )

            all_noised = torch.cat(noised_chunks, dim=1)  # [C*pt*ph*pw, sum_n]
            target = torch.cat(target_chunks, dim=0)      # [sum_n, C*pt*ph*pw]

            # patch_embed (packed) expects [C*pt*ph*pw, total_tokens, 1, 1].
            model_input = all_noised.unsqueeze(-1).unsqueeze(-1)
            target = target.contiguous()

        torch.cuda.current_stream().wait_stream(self.video_stream)
        torch.cuda.current_stream().wait_stream(self.text_stream)
        torch.cuda.current_stream().wait_stream(self.meta_stream)

        active_video_length = int(plan.video_cu_seqlens[num_sequences].item())
        active_caption_length = int(plan.caption_cu_seqlens[num_sequences].item())

        batch = PackedBatch(
            hidden_states=model_input,
            timestep=model_timesteps,
            text_states=caption_states,
            target=target,
            packed_seq_params=PackedSequenceParams(
                video_cu_seqlens=video_cu_seqlens,
                caption_cu_seqlens=caption_cu_seqlens,
                caption_segment_limit=self.caption_segment_limit,
                joint_cu_seqlens=joint_cu_seqlens,
                indices_interleave_to_concat=indices_interleave_to_concat,
                indices_concat_to_interleave=indices_concat_to_interleave,
                cu_seqlens_concat=cu_seqlens_concat,
                cu_seqlens_matmul=cu_seqlens_matmul,
                joint_segment_limit=self.joint_segment_limit,
                freqs_cos=cos,
                freqs_sin=sin,
                active_sequence_count=active_sequence_count,
                joint_packed_length=joint_packed_length,
            ),
            cp_meta=CpMeta(
                video_cu_seqlens_cpu=plan.video_cu_seqlens[:num_sequences + 1].clone(),
                caption_cu_seqlens_cpu=plan.caption_cu_seqlens[:num_sequences + 1].clone(),
            ) if self.cp_enabled else None,
        )
        meta = PackedDataloaderMeta(
            active_sequence_count=num_sequences,
            active_joint_length=active_video_length + active_caption_length,
            active_video_length=active_video_length,
            active_caption_length=active_caption_length,
            active_image_count=plan.num_image_sequences,
            dataloader_state=plan.dataloader_state,
        )

        return batch, meta

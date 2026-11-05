# Copyright 2026 Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
# SPDX-License-Identifier: Apache-2.0
"""Inference pipeline: Qwen caption encoding, Euler sampling of the DiT velocity, Wan 2.2 decoding.

The transformer is built from the training YAML's architecture (``build_transformer``);
the VAE and the text encoder come from the Hugging Face cache. ``create_pipeline``
assembles the three, ``__call__`` generates one prompt batch.
"""
import glob
import inspect
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import loguru
import safetensors.torch
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from pan.commons import parse_json_object
from pan.models.text_encoders import QwenTextEncoder
from pan.models.transformers.pan_transformer import PanVideoDiffusionTransformer, translate_architecture_config
from pan.schedulers.scheduling_flow_match_discrete import FlowEulerScheduler
from pan.schedulers.timestep_schedule import apply_timestep_schedule
from pan.utils.inference_guidance import (
    apply_guidance, predict_stg_positive, resolve_guidance_plan, validate_sampling_extensions,
)

# Wan 2.2 VAE latent space: 48 channels, 16x spatial and 4x temporal compression
# (the first frame gets its own latent frame).
WAN22_LATENT_CHANNELS = 48
WAN22_SPATIAL_STRIDE = 16
WAN22_TEMPORAL_STRIDE = 4
WAN22_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

OFFLOAD_MODES = ("none", "models", "blocks")


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def latent_grid(num_frames: int, height: int, width: int, patch_size) -> Tuple[int, int, int]:
    """``(frames, rows, cols)`` of the Wan latent of a ``num_frames x height x width`` video.

    The pixel size must be a multiple of the VAE stride, and the latent size a multiple
    of the transformer's spatial patch.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height=} {width=}")
    if height % WAN22_SPATIAL_STRIDE or width % WAN22_SPATIAL_STRIDE:
        raise ValueError(f"{height}x{width} is not a multiple of the VAE stride {WAN22_SPATIAL_STRIDE}")
    rows, cols = height // WAN22_SPATIAL_STRIDE, width // WAN22_SPATIAL_STRIDE
    patch = (patch_size,) * 2 if isinstance(patch_size, int) else tuple(int(p) for p in patch_size)[-2:]
    if rows % patch[0] or cols % patch[1]:
        raise ValueError(f"latent {rows}x{cols} of a {height}x{width} video is not a multiple of the patch {patch}")
    grid = ((num_frames - 1) // WAN22_TEMPORAL_STRIDE + 1, rows, cols)
    if min(grid) <= 0:
        raise ValueError(f"Empty latent grid {grid} for {num_frames} frames at {height}x{width}")
    return grid


@dataclass
class PanVideoPipelineOutput(BaseOutput):
    videos: torch.Tensor


class PanVideoPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: QwenTextEncoder,
        transformer: PanVideoDiffusionTransformer,
        scheduler: FlowEulerScheduler,
        flow_shift: float = 5.0,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
        progress_bar_config: Optional[Dict[str, Any]] = None,
        execution_device=None,
    ):
        super().__init__()
        # Per-call defaults for the sampling knobs.
        self.register_to_config(flow_shift=flow_shift, guidance_scale=guidance_scale,
                                num_inference_steps=num_inference_steps)
        self._progress_bar_config = {**getattr(self, "_progress_bar_config", {}), **(progress_bar_config or {})}
        self.register_modules(vae=vae, text_encoder=text_encoder, transformer=transformer, scheduler=scheduler)

        self.vae_scale_factor = WAN22_SPATIAL_STRIDE
        self.vae_dtype = self.vae.dtype
        # The DiT runs under bf16 autocast. Noise is drawn on the CPU so a seed gives the
        # same video on any GPU.
        self.dit_dtype = torch.bfloat16
        self.noise_device = torch.device("cpu")
        self.compute_device = torch.device(execution_device)
        self.offload_models = False

    # ---- runtime ---------------------------------------------------------------------

    def configure_runtime(self, compile_blocks: bool = False, offload: str = "none", prefetch: bool = True):
        """Runtime knobs set once after construction.

        ``compile_blocks`` wraps every transformer block in ``torch.compile``. ``offload``:
        ``none`` keeps everything on the GPU; ``models`` parks the text encoder, DiT and VAE
        on the CPU between uses; ``blocks`` additionally streams the DiT's blocks to the GPU
        while it runs, one block at a time with a prefetch stream when ``prefetch`` is set
        (fast, much more host memory), otherwise four at a time.
        """
        if offload not in OFFLOAD_MODES:
            raise ValueError(f"offload must be one of {OFFLOAD_MODES}, got {offload!r}")
        if compile_blocks:
            # All blocks share one compiled code object; leave room for every instance.
            torch._dynamo.config.cache_size_limit = torch._dynamo.config.accumulated_cache_size_limit = 192
            for block in self.transformer.blocks:
                block.forward = torch.compile(block.forward)
        self.offload_models = offload != "none"
        if offload == "blocks":
            stream = {"use_stream": True} if prefetch else {}
            self.transformer.enable_group_offload(
                onload_device=torch.device("cuda"), num_blocks_per_group=1 if prefetch else 4, **stream,
            )

    @staticmethod
    def dit_fits_on_gpu(dit_bytes: int, device=None) -> bool:
        """Whether the DiT's weights alone fit in the free memory of ``device`` (the current GPU)."""
        if not torch.cuda.is_available():
            return False
        free, _ = torch.cuda.mem_get_info(device)
        return dit_bytes < free

    @contextmanager
    def _resident(self, *modules):
        """With model offloading on, keep ``modules`` on the GPU inside the block and back on the CPU after."""
        modules = [m for m in modules if m is not None]
        if not self.offload_models:
            yield
            return
        for module in modules:
            module.to(self.compute_device)
        try:
            yield
        finally:
            for module in modules:
                module.to(torch.device("cpu"))

    # ---- VAE -------------------------------------------------------------------------

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE latents -> the normalized space the DiT was trained in."""
        mean, std = (stat.to(latents.device, latents.dtype) for stat in (self.vae._pan_latents_mean, self.vae._pan_latents_std))
        return (latents - mean) / std

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = (stat.to(latents.device, latents.dtype) for stat in (self.vae._pan_latents_mean, self.vae._pan_latents_std))
        return latents * std + mean

    def _encode_reference_image(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        """Normalized VAE latents ``[1, C, 1, h, w]`` of ``image`` fitted to ``height x width``.

        The image is scaled (Lanczos) just enough to cover the target, center-cropped,
        mapped to [-1, 1] and encoded with the VAE posterior mode.
        """
        src_w, src_h = image.size
        zoom = max(width / src_w, height / src_h)
        image = TF.resize(image, [int(round(src_h * zoom)), int(round(src_w * zoom))],
                          interpolation=InterpolationMode.LANCZOS)
        pixels = TF.normalize(TF.to_tensor(TF.center_crop(image, [height, width])), [0.5], [0.5])
        pixels = pixels[None, :, None].to(self.compute_device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self._normalize_latents(self.vae.encode(pixels).latent_dist.mode())

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized latents -> float ``[B, C, T, H, W]`` CPU video in [0, 1]."""
        latents = self._denormalize_latents(latents)
        tiling = self.vae.memory_efficient_context() if hasattr(self.vae, "memory_efficient_context") else nullcontext()
        with torch.autocast(device_type="cuda", dtype=self.vae_dtype), self._resident(self.vae), tiling:
            pixels = self.vae.decode(latents, return_dict=False)[0]
        # VAE output is in [-1, 1]; x * 0.5 is exact, so this equals (x / 2 + 0.5).
        return pixels.mul(0.5).add(0.5).clamp(0, 1).cpu().float()

    # ---- text ------------------------------------------------------------------------

    def encode_prompt(self, prompt, device, with_negative, negative_prompt=None):
        """Encode the prompt(s) and, under CFG, the negative prompt(s) with the Qwen text encoder.

        Encoding mirrors the packed training loader (video chat template, full
        prompt through Qwen, template prefix cropped). Positive and negative
        prompts go through Qwen as one batch so both come back padded to the same
        length, which lets ``__call__`` stack them for CFG.

        Returns ``(prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask)``;
        the negative pair is ``None`` without CFG.
        """
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        n = len(prompts)
        texts = list(prompts)
        if with_negative:
            negatives = (list(negative_prompt) if isinstance(negative_prompt, (list, tuple))
                         else [negative_prompt or ""] * n)
            if len(negatives) != n:
                raise ValueError(f"Got {len(negatives)} negative prompts for {n} prompts.")
            texts += negatives

        embeds, mask = self.text_encoder.encode(texts, device=device)
        embeds, mask = embeds.to(device=device, dtype=self.text_encoder.dtype), mask.to(device)
        if not with_negative:
            return embeds, None, mask, None
        return embeds[:n], embeds[n:], mask[:n], mask[n:]

    # ---- sampling --------------------------------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def _use_cfg(self) -> bool:
        """CFG runs a negative branch only for a weight above 1 (1 would equal the positive prediction)."""
        return (self._guidance_scale or 0) > 1

    def _condition_channels(self, batch: int, grid, image_latents: Optional[torch.Tensor], like: torch.Tensor):
        """``(cond, mask)`` stacked onto the noisy latent: zeros for t2v; for i2v the reference
        latent frame(s) and a mask of ones over them, as in packed training."""
        frames, rows, cols = grid
        cond = like.new_zeros((batch, WAN22_LATENT_CHANNELS, frames, rows, cols))
        mask = like.new_zeros((batch, 1, frames, rows, cols))
        if image_latents is not None:
            if image_latents.shape[0] not in (1, batch):
                raise ValueError(f"reference latents have batch {image_latents.shape[0]}, expected 1 or {batch}")
            source = image_latents.repeat(batch, 1, 1, 1, 1) if image_latents.shape[0] != batch else image_latents
            known = min(int(source.shape[2]), frames)
            cond[:, :, :known].copy_(source[:, :, :known])
            mask[:, :, :known].fill_(1.0)
        return cond, mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: "str | list[str]",
        video_length: int,
        num_inference_steps: "int | None" = None,
        guidance_scale: "float | None" = None,
        negative_prompt: "str | list[str] | None" = None,
        seed: Optional[int] = None,
        flow_shift: "float | None" = None,
        reference_image=None,
        video_height: int = None,
        video_width: int = None,
        cfg_start_step: int = 0,
        cfg_end_step: Optional[int] = None,
        stg_scale: float = 0.0,
        stg_skip_layers: Optional[List[int]] = None,
        i2v_skip_first_latent_cfg: bool = True,
        cfg_skip_interval: Optional[Tuple[int, int]] = None,
        cfg_warmup_scale: float = 1.0,
        timestep_schedule: str = "standard",
        first_step_velocity_scale: float = 1.0,
    ):
        """Generate a video for ``prompt`` (text-to-video), or animate ``reference_image``
        (a PIL image or a path) when one is given (image-to-video).

        ``guidance_scale``, ``flow_shift`` and ``num_inference_steps`` default to
        the pipeline config. ``seed`` (random when ``None`` or -1) seeds the CPU
        noise generator. The ``cfg_*``, ``stg_*``, ``timestep_schedule`` and
        ``first_step_velocity_scale`` options are the guidance controls handled by
        ``pan.utils.inference_guidance`` and ``pan.schedulers.timestep_schedule``.

        Returns a ``PanVideoPipelineOutput`` whose ``videos`` is a float ``[B, C, T, H, W]``
        CPU tensor in ``[0, 1]``.
        """
        cfg = self.config
        guidance_scale = cfg.guidance_scale if guidance_scale is None else guidance_scale
        flow_shift = cfg.flow_shift if flow_shift is None else flow_shift
        num_inference_steps = cfg.num_inference_steps if num_inference_steps is None else num_inference_steps

        reference_label = reference_image  # what the caller passed, for the banner
        task_type = "t2v" if reference_image is None else "i2v"
        if isinstance(reference_image, str):
            reference_image = Image.open(reference_image).convert("RGB")
        elif reference_image is not None and not isinstance(reference_image, Image.Image):
            raise ValueError("reference_image must be a file path or a PIL image")
        if not isinstance(i2v_skip_first_latent_cfg, bool):
            raise ValueError("i2v_skip_first_latent_cfg must be a boolean")
        if type(first_step_velocity_scale) not in (int, float) or not 0 <= first_step_velocity_scale <= 1:
            raise ValueError("first_step_velocity_scale must be finite and in [0,1]")
        validate_sampling_extensions(num_inference_steps, timestep_schedule)
        guidance_requested = (
            timestep_schedule != "standard" or cfg_start_step != 0
            or cfg_end_step is not None or stg_scale != 0
            or stg_skip_layers is not None or cfg_skip_interval is not None
            or cfg_warmup_scale != 1.0 or first_step_velocity_scale != 1.0
        )
        if guidance_requested:
            if not guidance_scale > 1:
                raise ValueError("Guidance controls require guidance_scale > 1")
            if stg_scale != 0 and task_type != "t2v":
                raise ValueError("STG supports T2V only")
        if stg_scale == 0 and stg_skip_layers is not None:
            raise ValueError("stg_skip_layers requires stg_scale > 0")
        if video_height is None or video_width is None:
            raise ValueError("video_height and video_width are required.")
        height, width = int(video_height), int(video_width)

        self.scheduler = FlowEulerScheduler(shift=flow_shift)  # fresh: set_timesteps mutates it
        if seed in (None, -1):
            seed = random.randint(100000, 999999)  # random six-digit seed, printed below for reproduction
        generator = torch.Generator(device=self.noise_device).manual_seed(seed)
        grid = latent_grid(video_length, height, width, self.transformer.config.patch_size)
        self._guidance_scale = guidance_scale
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self.compute_device

        if _is_main_process():
            ref = f"{reference_label} {reference_image.size}" if reference_image is not None else "none"
            rows = [
                ("mode", task_type), ("prompt", prompt), ("frames", video_length),
                ("size (w x h)", f"{width} x {height}"), ("reference", ref), ("cfg weight", guidance_scale),
                ("i2v first-latent cfg skip", task_type == "i2v" and i2v_skip_first_latent_cfg),
                ("flow shift", flow_shift), ("steps", num_inference_steps), ("seed", seed),
                ("attention", f"FlashAttention-4 (attention_backend={self.transformer.attention_backend})"),
                ("dit dtype", self.transformer.dtype),
            ]
            rule = "-" * 64
            print("\n".join(["", rule, "PAN 2 generation"] + [f"  {k:<26}{v}" for k, v in rows] + [rule, ""]))

        with self._resident(self.text_encoder):
            text, neg_text, text_mask, neg_text_mask = self.encode_prompt(prompt, device, self._use_cfg, negative_prompt)
        if self._use_cfg:
            # Row order [negative; positive] matches the chunk(2) split after the forward.
            text, text_mask = torch.cat([neg_text, text]), torch.cat([neg_text_mask, text_mask])

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        if timestep_schedule != "standard":
            timesteps = apply_timestep_schedule(self.scheduler, timestep_schedule)
        guidance_plan = resolve_guidance_plan(
            len(timesteps), start_step=cfg_start_step, end_step=cfg_end_step, stg_scale=stg_scale,
            skip_interval=cfg_skip_interval, warmup_scale=cfg_warmup_scale,
        )
        if guidance_plan.stg_scale:
            stg_skip_layers = self.transformer.validate_stg_layers(stg_skip_layers)
            if not stg_skip_layers:
                raise ValueError("Active STG requires nonempty stg_skip_layers")

        # The DiT input is [noisy latent (48), condition (48), condition mask (1)] = 97 channels.
        if int(self.transformer.in_channels) != 2 * WAN22_LATENT_CHANNELS + 1:
            raise ValueError(f"expected a {2 * WAN22_LATENT_CHANNELS + 1}-channel (channel-concat) transformer, "
                             f"got in_channels={self.transformer.in_channels}")
        # Unit Gaussian [B, C, T, H, W] noise from the CPU generator, then moved to the GPU.
        latents = torch.randn((batch_size, WAN22_LATENT_CHANNELS, *grid), generator=generator,
                              device=self.noise_device, dtype=self.dit_dtype).to(device)
        with self._resident(self.vae):
            image_latents = self._encode_reference_image(reference_image, height, width) if task_type == "i2v" else None
        cond, mask = self._condition_channels(batch_size, grid, image_latents, like=latents)
        if self._use_cfg:
            cond, mask = torch.cat([cond] * 2), torch.cat([mask] * 2)

        with self.progress_bar(total=len(timesteps)) as progress_bar, self._resident(self.transformer):
            for i, t in enumerate(timesteps):
                x_in = torch.cat([latents] * 2) if self._use_cfg else latents
                t_batch = t.expand(x_in.shape[0])  # one timestep per batch row
                # The condition follows the latent's dtype, which is float32 after the first Euler step.
                dit_in = torch.cat([x_in, cond.to(device=x_in.device, dtype=x_in.dtype),
                                    mask.to(device=x_in.device, dtype=x_in.dtype)], dim=1)
                with torch.autocast(device_type="cuda", dtype=self.dit_dtype):
                    velocity = self.transformer(dit_in, t_batch, text, text_mask, return_dict=False)[0]

                if self._use_cfg:
                    v_neg, v_pos = velocity.chunk(2)
                    if guidance_requested:
                        def weak_predict_fn():
                            with torch.autocast(device_type="cuda", dtype=self.dit_dtype):
                                return predict_stg_positive(
                                    self.transformer, dit_in, t_batch, text, text_mask,
                                    batch_size=v_pos.shape[0], skip_layers=stg_skip_layers,
                                )

                        velocity = apply_guidance(
                            v_neg, v_pos, self.guidance_scale,
                            step_index=i, plan=guidance_plan, weak_predict_fn=weak_predict_fn,
                        )
                    else:
                        # Plain CFG in bf16, the exact arithmetic of the default sampler.
                        velocity = v_neg + self.guidance_scale * (v_pos - v_neg)

                if self._use_cfg and task_type == "i2v" and i2v_skip_first_latent_cfg and guidance_plan.active(i):
                    # The image-aligned latent frame takes the positive prediction only;
                    # outside the CFG window the whole prediction is positive-only already.
                    velocity[:, :, :1] = v_pos[:, :, :1]
                # Scale the complete first update, after the i2v handling; 1 skips the multiply.
                if i == 0 and first_step_velocity_scale != 1.0:
                    velocity = velocity * first_step_velocity_scale

                latents = self.scheduler.step(velocity, t, latents)  # Euler step to the next noise level
                progress_bar.update()

        videos = self._decode(latents)
        self.maybe_free_model_hooks()
        return PanVideoPipelineOutput(videos=videos)

    # ---- construction ----------------------------------------------------------------

    @staticmethod
    def _read_state_dict(path, transformer_dtype):
        """Transformer state dict from ``path``: a .pt / .pth / .safetensors file, or a directory
        holding model_state.pt, *.safetensors, or a diffusers-style checkpoint (config.json + weights)."""
        if os.path.isfile(path):
            if path.endswith(".safetensors"):
                return safetensors.torch.load_file(path, device="cpu")
            state = torch.load(path, map_location="cpu", weights_only=True)
            for wrapper in ("model", "state_dict", "module"):
                if (isinstance(state, dict) and isinstance(state.get(wrapper), dict)
                        and not any(hasattr(v, "shape") for v in state.values())):
                    return state[wrapper]
            return state
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Pretrained transformer weights not found: {path}")
        if os.path.isfile(os.path.join(path, "model_state.pt")):
            return PanVideoPipeline._read_state_dict(os.path.join(path, "model_state.pt"), transformer_dtype)
        shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if shards and not os.path.isfile(os.path.join(path, "config.json")):
            state = {}
            for shard in shards:
                state.update(safetensors.torch.load_file(shard, device="cpu"))
            return state
        reference = PanVideoDiffusionTransformer.from_pretrained(path, torch_dtype=transformer_dtype, low_cpu_mem_usage=True)
        return reference.state_dict()

    @staticmethod
    def _load_matching_weights(transformer, pretrained_path, transformer_dtype):
        """Copy every tensor of ``pretrained_path`` whose name and shape match into ``transformer``."""
        source = PanVideoPipeline._read_state_dict(pretrained_path, transformer_dtype)
        target = transformer.state_dict()
        matching = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
        mismatched = [(k, tuple(v.shape), tuple(target[k].shape)) for k, v in source.items()
                      if k in target and k not in matching]
        if not any(key.startswith("blocks.") for key in matching):
            raise ValueError(
                f"{pretrained_path} has no tensor that matches this transformer's joint blocks "
                f"({len(matching)}/{len(target)} keys matched by name and shape); "
                "refusing to warm start from what would be a mostly random initialization."
            )
        result = transformer.load_state_dict(matching, strict=False)
        loguru.logger.info(
            f"Loaded compatible pretrained weights: {len(matching)}/{len(target)} keys, "
            f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}, "
            f"shape_mismatched={len(mismatched)}."
        )
        if mismatched:
            preview = ", ".join(f"{k}: {src}->{dst}" for k, src, dst in mismatched[:8])
            loguru.logger.warning(f"Sample shape mismatches (first 8): {preview}")
        not_loaded = [k for k in target if k not in matching]
        if not_loaded:
            # A partial match is legal (e.g. a changed input layer), but every tensor
            # listed here keeps its fresh random init.
            blocks = sorted({k.split(".")[1] for k in not_loaded if k.startswith("blocks.")}, key=int)
            loguru.logger.warning(
                f"PARTIAL WARM START: {len(not_loaded)}/{len(target)} transformer tensors were not in "
                f"{pretrained_path} (or had another shape) and keep their random init"
                + (f", including tensors of blocks {', '.join(blocks)}" if blocks else "")
                + f". First 8: {', '.join(not_loaded[:8])}")
        unused = [k for k in source if k not in target]
        if unused:
            loguru.logger.warning(
                f"{len(unused)} tensors of {pretrained_path} have no counterpart in this transformer "
                f"and were skipped, e.g. {', '.join(unused[:8])}")

    @staticmethod
    def _load_wan_vae(device, wan_vae_checkpoint=None):
        """The Wan 2.2 VAE (diffusers ``AutoencoderKLWan``) whose normalized latents the DiT was
        trained on. ``wan_vae_checkpoint`` may point at a local diffusers root."""
        vae = AutoencoderKLWan.from_pretrained(wan_vae_checkpoint or WAN22_REPO, subfolder="vae",
                                               torch_dtype=torch.bfloat16).to(device)
        # Latent normalization statistics, cached as broadcastable buffers.
        for name, values in (("_pan_latents_mean", vae.config.latents_mean), ("_pan_latents_std", vae.config.latents_std)):
            stat = torch.tensor(values, device=device, dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
            vae.register_buffer(name, stat, persistent=False)
        return vae

    @staticmethod
    def architecture(transformer_config_overrides=None) -> dict:
        """The transformer config: its constructor defaults updated by the (possibly legacy-named) overrides."""
        overrides = translate_architecture_config(parse_json_object(transformer_config_overrides, "transformer_config_overrides") or {})
        known = set(inspect.signature(PanVideoDiffusionTransformer.__init__).parameters) - {"self"}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise KeyError(f"Unknown transformer config keys: {unknown}. Known keys: {sorted(known)}")
        return overrides

    @classmethod
    def build_transformer(
        cls,
        transformer_config_overrides=None,
        pretrained_transformer_path=None,
        load_transformer_weights=False,
        transformer_dtype=torch.bfloat16,
        device=None,
    ):
        """Build the transformer from its constructor defaults plus ``transformer_config_overrides``.

        With ``load_transformer_weights`` every name- and shape-matching tensor of
        ``pretrained_transformer_path`` (a .pt / .safetensors state dict, or a folder with one) is loaded.
        """
        device = torch.device("cuda") if device is None else device
        config = cls.architecture(transformer_config_overrides)
        loguru.logger.info(f"Building transformer with config overrides {config}")
        transformer = PanVideoDiffusionTransformer.from_config(config)
        if load_transformer_weights:
            if not pretrained_transformer_path:
                raise ValueError("load_transformer_weights=True requires pretrained_transformer_path "
                                 "(a .pt / .safetensors state dict, or a directory holding one).")
            loguru.logger.info(f"Loading pretrained transformer weights from {pretrained_transformer_path}")
            cls._load_matching_weights(transformer, pretrained_transformer_path, transformer_dtype)
        elif pretrained_transformer_path:
            loguru.logger.info(f"load_transformer_weights=False: ignoring pretrained_transformer_path={pretrained_transformer_path}")
        return transformer.to(device, dtype=transformer_dtype)

    @staticmethod
    def transformer_nbytes(transformer_config_overrides=None, dtype=torch.bfloat16) -> int:
        """Size of the transformer's weights in ``dtype``, measured on the meta device (no allocation)."""
        with torch.device("meta"):
            model = PanVideoDiffusionTransformer.from_config(PanVideoPipeline.architecture(transformer_config_overrides))
        return sum(p.numel() for p in model.parameters()) * torch.empty((), dtype=dtype).element_size()

    @classmethod
    def create_pipeline(
        cls,
        transformer_config_overrides=None,
        transformer_dtype=torch.bfloat16,
        device=None,
        transformer_init_device=None,
        wan_vae_checkpoint=None,
        text_encoder_hidden_state_skip_layer=2,
        text_encoder_max_length=None,
        flow_shift=5.0,
        guidance_scale=4.0,
        num_inference_steps=50,
    ):
        """Assemble the inference pipeline.

        The transformer comes from ``transformer_config_overrides`` (the training YAML's
        model section) with random weights; load the checkpoint afterwards (see
        generate.py). The Wan 2.2 VAE and the Qwen3.5 text encoder come from the Hugging
        Face cache. ``flow_shift`` / ``guidance_scale`` / ``num_inference_steps`` are the
        defaults a call uses when it passes None.
        """
        device = torch.device("cuda") if device is None else device
        transformer = cls.build_transformer(
            transformer_config_overrides=transformer_config_overrides,
            transformer_dtype=transformer_dtype,
            device=device if transformer_init_device is None else transformer_init_device,
        )
        if transformer.out_channels != WAN22_LATENT_CHANNELS:
            raise ValueError(f"out_channels must be the Wan 2.2 latent width {WAN22_LATENT_CHANNELS}, got {transformer.out_channels}")
        vae = cls._load_wan_vae(device=device, wan_vae_checkpoint=wan_vae_checkpoint)
        # Same Qwen3.5 loader as training, so the inference embeddings match what the DiT saw.
        text_encoder = QwenTextEncoder(
            hidden_state_skip_layer=text_encoder_hidden_state_skip_layer,
            max_caption_tokens=1000 if text_encoder_max_length is None else int(text_encoder_max_length),
            device=device,
        )
        return cls(vae=vae, text_encoder=text_encoder, transformer=transformer,
                   scheduler=FlowEulerScheduler(shift=flow_shift), execution_device="cuda",
                   flow_shift=flow_shift, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)

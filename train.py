"""
Packed-only PAN training entrypoint.

This fork keeps the packed CSV metadata path and removes the alternative
streaming/WebDataset training inputs. The trainer expects packed metadata CSVs
and builds packed batches with `dataset.packed_dataloader.PackedIterator`.
"""

from functools import partial
from copy import deepcopy
from collections.abc import Mapping
import os
import glob
import re
import random
import shutil
import math
import time
import argparse
from pathlib import Path
import sys
import threading
import faulthandler
from dataclasses import MISSING, dataclass, field, fields
from typing import Dict, Any, Optional
from enum import Enum
import json
import loguru
import safetensors
import safetensors.torch
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import create_selective_checkpoint_contexts
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from dataset.cp_batch import split_packed_batch_for_cp
from dataset.packed_dataloader import PackedBatch, PackedDataloaderMeta, PackedIterator

# Wan 2.2 latent geometry. The VAE is not loaded during training — latents are
# precomputed offline and streamed by the packed loader — so these constants
# live here instead of on a VAE object.
WAN22_LATENT_CHANNELS = 48
WAN22_SPATIAL_RATIO = 16
WAN22_TEMPORAL_RATIO = 4
# Fixed seed for the deterministic shuffle of the image-tar shard list. Shuffling
# the full sorted list with this constant (before per-rank sharding) mixes shard
# order while staying reproducible across runs and on resume.
_IMAGE_TAR_SHUFFLE_SEED = 42
from dataset.packing_dataset import DataloaderState
from pan.models.text_encoders import PROMPT_TEMPLATE, QWEN35_TEXT_ENCODER_ID, load_qwen_text_encoder
from pan.models.transformers.pan_transformer import PanVideoDiffusionTransformer
from pan.models.transformers.modules.caption_encoder import CaptionEncoderBlock
from pan.models.transformers.modules.rope import rope_tables_for_grid
from loguru import logger
from pan.pipelines.pan_video_pipeline import PanVideoPipeline
from pan.commons import parse_json_object, str_to_bool
from pan.commons.parallel_layout import current_parallel_layout, init_parallel_layout
from pan.optim.muon import build_native_muon_optimizers
from pan.flow_matching import TimestepDistribution, TimestepShiftMode, draw_timesteps, interpolate, shift_timesteps
from pan.models.transformers.modules.quack_autotune_patch import register_quack_dynamic_sizes

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

import flash_attn.cute

flash_attn.cute.flash_attn_varlen_func = torch.compiler.disable(flash_attn.cute.flash_attn_varlen_func)
flash_attn.cute.flash_attn_func = torch.compiler.disable(flash_attn.cute.flash_attn_func)

def logging_format(record):
    msg = record["message"]
    if all(i in msg for i in ("Step", "Loss", "LR")) or msg.startswith("(avg) sequence count"):
        return "{time:YYYY-MM-DD HH:mm:ss} - {message}\n{exception}"  # shorter format for training step logs
    return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:4.4} | {name}:{function}:{line} - {message}\n{exception}"


def configure_logging():
    rank = int(os.environ.get("RANK", "0"))
    logger.remove()
    logger.add(
        sys.stderr,
        format=logging_format,
        filter=lambda record: rank == 0 or record["level"].no >= logger.level("WARNING").no,
    )


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ConditioningSchema(str, Enum):
    TOKEN_CONCAT = "token_concat"
    CHANNEL_CONCAT = "channel_concat"


def load_training_config_defaults(config_path: str, parser: argparse.ArgumentParser) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML training configs require omegaconf or PyYAML. Install project requirements first."
            ) from exc

        with open(config_path, "r", encoding="utf-8") as f:
            config_obj = yaml.safe_load(f)
    else:
        config_obj = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if config_obj is None:
        return {}
    if not isinstance(config_obj, Mapping):
        raise ValueError(f"Training config {config_path} must be a YAML mapping")

    action_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.dest not in ("help", argparse.SUPPRESS)
    }
    valid_keys = set(action_by_dest)
    defaults: Dict[str, Any] = {}
    # Deprecated YAML keys kept working so older configs still load.
    legacy_keys = {"sp_size": "cp_size", "i2v_prob": "image_cond_prob", "v2v_prob": "video_cond_prob"}
    # Removed keys that are still accepted when they carry their only meaningful value.
    ignored_keys = {"single_stream_only": (True, False), "single_stream_block_type": ("mm_joint_attn",),
                    "single_stream_depth": (None,)}

    def visit(node: Mapping[str, Any], path: tuple[str, ...] = ()) -> None:
        for raw_key, value in node.items():
            key = str(raw_key)
            current_path = (*path, key)
            key = legacy_keys.get(key, key)
            if key in ignored_keys:
                if value not in ignored_keys[key]:
                    raise ValueError(
                        f"Training config key {'.'.join(current_path)!r} is no longer supported; "
                        "set the architecture in model.transformer_config_overrides (e.g. num_layers)."
                    )
                continue
            if key in valid_keys:
                if key in defaults:
                    raise ValueError(
                        f"Duplicate training config key {key!r}; "
                        f"second occurrence at {'.'.join(current_path)}"
                    )
                defaults[key] = value
            elif isinstance(value, Mapping):
                visit(value, current_path)
            else:
                raise ValueError(
                    f"Unknown training config key {'.'.join(current_path)!r}. "
                    "Use argparse destination names, grouped however you like."
                )

    visit(config_obj)

    for key, value in list(defaults.items()):
        action = action_by_dest[key]
        if action.type is str_to_bool and isinstance(value, str):
            value = str_to_bool(value)
        if action.nargs in ("+", "*") and value is not None and not isinstance(value, list):
            value = [value]
        defaults[key] = value

    return defaults



def _resolve_dcp_transformer_dir(path: Optional[str]) -> Optional[str]:
    """If `path` is a checkpoint-N/ dir or its transformer/ DCP dir, return the
    DCP dir; otherwise (plain state-dict file / dir, or None) return None."""
    if not path or not os.path.isdir(path):
        return None
    for candidate in (os.path.join(path, "transformer"), path):
        if os.path.isfile(os.path.join(candidate, ".metadata")):
            return candidate
    return None


LR_SCHEDULER_TYPES = ("constant_with_warmup", "linear_decay", "cosine_decay")


def _opt(default, kind, help, factory=None, **cli):
    """Declare one training option.

    ``kind`` is the argument type (``int`` / ``float`` / ``str``) or one of ``"switch"``
    (``--x``, ``--x true`` or ``--x false``), ``"set"`` / ``"unset"`` (a bare flag that
    stores True / False) or ``"list"`` (one or more strings). ``cli`` may hold ``flags``,
    ``dest``, ``choices``, ``convert`` (applied to the parsed value) or ``cli=False``.
    """
    metadata = {"kind": kind, "help": help, **cli}
    if factory is not None:
        return field(default_factory=factory, metadata=metadata)
    return field(default=default, metadata=metadata)


@dataclass
class TrainingConfig:
    """Every training option, declared once.

    Each field is also a command-line flag (``--<name>`` unless noted) and a key of
    the YAML file given with ``--config``; YAML keys may be grouped under any
    sections. Command-line flags override the YAML file, which overrides the
    defaults below.
    """

    # -- model ------------------------------------------------------------------------
    transformer_config_overrides: Optional[Dict[str, Any]] = _opt(
        None, str, "Architecture on top of PanVideoDiffusionTransformer's defaults: a JSON object, a JSON "
        "file path, or (in YAML) a mapping, e.g. '{\"mm_single_blocks_depth\": 64}'.",
        convert=lambda raw: parse_json_object(raw, "transformer_config_overrides"))
    zero_init_modulation: bool = _opt(False, "switch", "Start every adaLN modulation projection at zero.")
    text_encoder_hidden_state_skip_layer: int = _opt(2, int, "Drop this many final Qwen layers; the caption features "
                                                     "are the output of the layer before them.")
    load_transformer_weights: bool = _opt(False, "switch", "Warm-start from pretrained_transformer_path (every "
                                          "shape-compatible tensor) instead of training from the fresh init.")
    # A plain state dict (.pt / .safetensors file, or a directory holding model_state.pt)
    # is loaded before FSDP; a checkpoint-N/ directory written by this trainer (or its
    # transformer/ DCP folder) is read after FSDP, weights only: optimizer, step and
    # dataloader position start fresh. Ignored once output_dir holds a checkpoint.
    pretrained_transformer_path: Optional[str] = _opt(None, str, "Weights for load_transformer_weights: a state-dict "
                                                      "file or folder, or a checkpoint-N/ directory of this trainer.")

    # -- optimization -----------------------------------------------------------------
    use_muon: bool = _opt(True, "switch", "Muon for the 2-D weight matrices and AdamW for everything else; "
                          "off trains everything with AdamW.")
    learning_rate: float = _opt(1e-5, float, "Peak learning rate (Muon and AdamW).")
    weight_decay: float = _opt(0.05, float, "Decoupled weight decay.")
    max_steps: int = _opt(10000, int, "Number of optimizer steps to train for.")
    warmup_steps: int = _opt(500, int, "Steps of linear learning-rate warmup.")
    lr_warmup_start_ratio: float = _opt(0.0, float, "Warmup starts at this fraction of the peak learning rate.")
    lr_scheduler_type: str = _opt("constant_with_warmup", str, "Learning-rate shape after warmup.", choices=LR_SCHEDULER_TYPES)
    lr_decay_steps: Optional[int] = _opt(None, int, "Length of the decay after warmup; max_steps - warmup_steps if unset.")
    lr_decay_floor_ratio: float = _opt(0.30, float, "Decay schedules end at this fraction of the peak learning rate.")
    gradient_accumulation_steps: int = _opt(1, int, "Micro-batches accumulated per optimizer step.")
    max_grad_norm: float = _opt(1.0, float, "Clip the global gradient norm to this value; 0 turns clipping off.")

    # -- flow matching ----------------------------------------------------------------
    num_train_timesteps: int = _opt(1000, int, "Length T of the training time axis.", cli=False)
    snr_type: TimestepDistribution = _opt(
        TimestepDistribution.LOGNORM, str, "Distribution of the training noise levels: uniform or logit-normal.",
        flags=["--flow_snr_type"], dest="flow_snr_type", choices=[d.value for d in TimestepDistribution],
        convert=TimestepDistribution)
    flow_lognorm_loc: float = _opt(0.0, float, "Mean of the Gaussian behind the logit-normal draws.")
    flow_lognorm_scale: float = _opt(1.0, float, "Standard deviation of the Gaussian behind the logit-normal draws.")
    train_timestep_shift_mode: TimestepShiftMode = _opt(
        TimestepShiftMode.FIXED, str, "fixed: shift every draw by train_timestep_shift; dynamic: by a shift that grows "
        "with each sample's resolution and length.", choices=[m.value for m in TimestepShiftMode], convert=TimestepShiftMode)
    train_timestep_shift: float = _opt(5.0, float, "Time shift s applied to the draws in fixed mode "
                                       "(t/T -> s*t/T / (1 + (s-1)*t/T)).")
    train_timestep_rescale_factor: float = _opt(0.5, float, "Dynamic mode: exponent scale of the resolution/length shift.")
    train_timestep_max_shift: float = _opt(1000.0, float, "Dynamic mode: upper bound of the shift.")

    # -- conditioning -----------------------------------------------------------------
    image_cond_prob: float = _opt(0.3, float, "Fraction of video samples trained as image-to-video (first frame given).",
                                  flags=["--image_cond_prob", "--i2v_prob"])
    video_cond_prob: float = _opt(0.0, float, "Fraction of video samples trained as video continuation (a prefix given).",
                                  flags=["--video_cond_prob", "--v2v_prob"])
    cfg_drop_prob: float = _opt(0.1, float, "Fraction of samples with every condition dropped, for classifier-free guidance.")
    conditioning_schema: ConditioningSchema = _opt(
        ConditioningSchema.TOKEN_CONCAT, str, "How i2v / v2v context frames enter the model.",
        choices=[c.value for c in ConditioningSchema], convert=ConditioningSchema)

    # -- data -------------------------------------------------------------------------
    packed_dataloader_meta_csv_path: list = _opt(None, "list", "Video metadata: one or more CSV files or directories of them.",
                                                 factory=lambda: ["dummy_meta.csv"])
    # Image-latent tar shards (WebDataset style) interleaved with the video stream.
    image_tar_path: list = _opt(None, "list", "Image-latent shards: directories of .tar files, .tar files, or text files "
                                "listing either. Empty: videos only.", factory=list)
    image_sample_prob: float = _opt(0.0, float, "Chance that an item is drawn from image_tar_path instead of the videos.")
    video_height: int = _opt(480, int, "Frame height the packed loader resizes videos to.")
    video_width: int = _opt(832, int, "Frame width the packed loader resizes videos to.")
    batch_size: int = _opt(1, int, "Per-rank batch size; packed batches are sized by their token budgets.")
    packed_joint_token_budget: int = _opt(7312, int, "Video + caption tokens per packed batch.")
    packed_video_token_budget: int = _opt(7056, int, "Video tokens per packed batch.")
    packed_caption_token_budget: int = _opt(1024, int, "Caption tokens per packed batch, counted after cropping the Qwen output.")
    packed_joint_segment_limit: int = _opt(7312, int, "Video + caption tokens of a single packed sequence.")
    packed_caption_segment_limit: int = _opt(256, int, "Caption tokens of a single packed sequence, after cropping.")
    packed_dataloader_max_num_seq: int = _opt(8, int, "Sequences per packed batch at most.")
    packed_dataloader_num_buffers: int = _opt(4, int, "Packing buffers kept by the loader.")
    packed_dataloader_prefetch_factor: int = _opt(2, int, "Batches each packing worker prepares ahead.")
    packed_dataloader_num_workers: int = _opt(32, int, "Threads decoding videos and tokenizing captions.")
    # On resume: also restore the video-CSV position. False keeps model, optimizer and
    # step but streams the (possibly new) CSVs from the start; the image cursor is kept.
    resume_video_csv: bool = _opt(True, "switch", "On resume, continue the video CSV stream where the checkpoint left it.")

    # -- parallelism and memory -------------------------------------------------------
    cp_size: int = _opt(1, int, "Ulysses context-parallel degree; must divide the number of ranks.",
                        flags=["--cp_size", "--sp_size"])
    hybrid_shard: bool = _opt(False, "set", "FSDP: shard within a node and replicate across nodes.")
    enable_gradient_checkpointing: bool = _opt(True, "switch", "Recompute block activations during the backward pass.")
    use_dedicated_gradient_checkpointing: bool = _opt(False, "set", "Selective recomputation that keeps the flash-attention "
                                                      "and grouped-QKV outputs (tuned for the 29B model on B300).")
    compile_text_encoder: bool = _opt(False, "set", "torch.compile the caption text encoder.")
    compile_transformer: bool = _opt(False, "set", "torch.compile the transformer.")
    compile_optimizer: bool = _opt(False, "set", "torch.compile the optimizer step.")
    dtype: str = _opt("bf16", str, "Autocast precision of the forward pass.", choices=("bf16", "fp32"))
    # Must stay True: the only validated execution mode (side-stream execution, tensor
    # learning rates and capturable optimizers, no autocast cache, FSDP resharding; no
    # CUDA graph is captured).
    legacy_eager_mode: bool = _opt(True, "switch", "Validated eager execution mode; false is not supported.")

    # -- checkpoints, logging and safety ----------------------------------------------
    output_dir: str = _opt("./outputs", str, "Folder for checkpoint-N/ directories and logs; a run resumes the latest one.")
    save_interval: int = _opt(1000, int, "Save a checkpoint every this many steps.")
    save_total_limit: int = _opt(0, int, "Keep only this many most recent checkpoints; 0 keeps them all.")
    log_interval: int = _opt(10, int, "Log a step summary every this many steps.")
    report_to: str = _opt("none", str, "Metrics sink.", choices=("none", "wandb"))
    seed: int = _opt(42, int, "Seed: the model init uses it unchanged on every rank; data sampling adds the data rank.")
    anomaly_detection: bool = _opt(True, "unset", "Turn off the loss / gradient anomaly checks.",
                                   flags=["--no_anomaly"], dest="anomaly_detection")
    anomaly_detection_warmup_steps: int = _opt(100, int, "No spike-based shutdown during these first steps "
                                               "(non-finite values and hard overflows still stop the run).")
    max_consecutive_skipped_steps: int = _opt(5, int, "Non-finite updates skipped in a row before recovering from the last checkpoint.")

    def check(self) -> None:
        """Reject inconsistent settings with an error that names the option."""
        if not self.legacy_eager_mode:
            raise ValueError("legacy_eager_mode must remain true; disabling it selects an unvalidated training execution path.")
        if self.warmup_steps < 0:
            raise ValueError("--warmup_steps must be >= 0")
        if self.lr_decay_steps is not None and self.lr_decay_steps < 0:
            raise ValueError("--lr_decay_steps must be >= 0 when set")
        for name in ("lr_warmup_start_ratio", "lr_decay_floor_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"--{name} must be finite and in [0, 1]")
        if self.anomaly_detection_warmup_steps < 0:
            raise ValueError("--anomaly_detection_warmup_steps must be >= 0")
        if not (math.isfinite(self.image_cond_prob) and math.isfinite(self.video_cond_prob)):
            raise ValueError(f"Require finite image_cond_prob and video_cond_prob, got image_cond_prob={self.image_cond_prob}, video_cond_prob={self.video_cond_prob}")
        if self.image_cond_prob < 0 or self.video_cond_prob < 0 or self.image_cond_prob + self.video_cond_prob > 1:
            raise ValueError("Require 0 <= image_cond_prob, video_cond_prob and image_cond_prob + video_cond_prob <= 1")
        if not (math.isfinite(self.flow_lognorm_loc) and math.isfinite(self.flow_lognorm_scale)):
            raise ValueError(f"Require finite flow_lognorm_loc and flow_lognorm_scale, "
                             f"got loc={self.flow_lognorm_loc}, scale={self.flow_lognorm_scale}")
        if self.flow_lognorm_scale <= 0:
            raise ValueError(f"Require flow_lognorm_scale > 0, got {self.flow_lognorm_scale}")
        if self.train_timestep_shift <= 0:
            raise ValueError(f"Require train_timestep_shift > 0, got {self.train_timestep_shift}")
        if self.train_timestep_rescale_factor < 0:
            raise ValueError(f"Require train_timestep_rescale_factor >= 0, got {self.train_timestep_rescale_factor}")
        if self.train_timestep_max_shift <= 0:
            raise ValueError(f"Require train_timestep_max_shift > 0, got {self.train_timestep_max_shift}")
        if self.pretrained_transformer_path and not self.load_transformer_weights:
            raise ValueError(
                f"pretrained_transformer_path is set ({self.pretrained_transformer_path}) but "
                "load_transformer_weights is false, so the run would train from scratch. Set "
                "load_transformer_weights: true to warm start, or clear pretrained_transformer_path.")
        if self.load_transformer_weights and not self.pretrained_transformer_path:
            raise ValueError("--load_transformer_weights true requires --pretrained_transformer_path.")
        if self.load_transformer_weights and not os.path.exists(self.pretrained_transformer_path):
            raise FileNotFoundError(f"--pretrained_transformer_path not found: {self.pretrained_transformer_path}")


def _cli_options():
    """``(field, metadata)`` of every TrainingConfig field that is a command-line flag."""
    return [(f, f.metadata) for f in fields(TrainingConfig) if f.metadata.get("cli", True)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PAN on video data")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML file with training options; command-line flags take precedence over it.")
    for f, meta in _cli_options():
        default = f.default_factory() if f.default is MISSING else f.default
        if isinstance(default, Enum):
            default = default.value
        spec = {"dest": meta.get("dest", f.name), "help": meta["help"]}
        kind = meta["kind"]
        if kind == "switch":
            spec.update(type=str_to_bool, nargs="?", const=True, default=default)
        elif kind == "set":
            spec.update(action="store_true")
        elif kind == "unset":
            spec.update(action="store_false")
        elif kind == "list":
            spec.update(type=str, nargs="+", default=default)
        else:
            spec.update(type=kind, default=default)
        if "choices" in meta:
            spec["choices"] = list(meta["choices"])
        parser.add_argument(*meta.get("flags", [f"--{f.name}"]), **spec)
    return parser


def parse_training_config(argv=None) -> TrainingConfig:
    """Defaults <- YAML (``--config``) <- command line, converted and checked."""
    parser = build_arg_parser()
    probe, _ = parser.parse_known_args(argv)
    if probe.config:
        parser.set_defaults(**load_training_config_defaults(probe.config, parser))
    args = parser.parse_args(argv)
    values = {}
    for f, meta in _cli_options():
        raw = getattr(args, meta.get("dest", f.name))
        values[f.name] = meta["convert"](raw) if "convert" in meta else raw
    config = TrainingConfig(**values)
    config.check()
    return config


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    warmup_steps: int,
    decay_steps: Optional[int],
    max_steps: int,
    warmup_start_ratio: float,
    decay_floor_ratio: float,
) -> LambdaLR:
    """Build a warmup scheduler with optional post-warmup decay."""
    if scheduler_type not in LR_SCHEDULER_TYPES:
        raise ValueError(f"Unknown lr_scheduler_type={scheduler_type!r}; expected one of {LR_SCHEDULER_TYPES}")
    effective_decay_steps = decay_steps if decay_steps is not None else max(max_steps - warmup_steps, 0)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            if warmup_steps == 1:
                return 1.0
            progress = current_step / float(warmup_steps - 1)
            return warmup_start_ratio * (1.0 - progress) + progress

        if scheduler_type == "constant_with_warmup":
            return 1.0

        if effective_decay_steps <= 1:
            return decay_floor_ratio

        decay_step = min(max(current_step - warmup_steps, 0), effective_decay_steps - 1)
        progress = decay_step / float(effective_decay_steps - 1)
        if scheduler_type == "linear_decay":
            return 1.0 * (1.0 - progress) + decay_floor_ratio * progress
        if scheduler_type == "cosine_decay":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return decay_floor_ratio + (1.0 - decay_floor_ratio) * cosine
        raise AssertionError(f"Unhandled lr_scheduler_type={scheduler_type!r}")

    return LambdaLR(optimizer, lr_lambda)


def _checkpoint_layout(ckpt_root: str):
    """``(transformer dir, optimizer dir, training-state file)`` inside a ``checkpoint-N`` directory."""
    return (os.path.join(ckpt_root, "transformer"), os.path.join(ckpt_root, "optimizer"),
            os.path.join(ckpt_root, "training_state.pt"))


class PanVideoTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

        # torchrun exports these; a plain `python train.py` is a single process.
        self.rank, self.world_size, self.local_rank = (
            int(os.environ.get(name, fallback)) for name, fallback in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
        )
        self.device = torch.device("cuda", self.local_rank)
        self.is_main_process = self.rank == 0
        # Also rejects cp_size > world_size, since then world_size % cp_size == world_size.
        if self.world_size % config.cp_size:
            raise ValueError(f"cp_size={config.cp_size} must evenly divide the number of ranks ({self.world_size})")

        init_parallel_layout(cp=config.cp_size, hybrid_shard=config.hybrid_shard)
        torch.cuda.set_device(self.device)
        self.side_stream = torch.cuda.Stream(device=self.device, priority=4)  # lower priority
        self.metrics_ready_event = torch.cuda.Event()
        self.parallel_state = current_parallel_layout()
        self.dp_rank = self.parallel_state.dp_mesh.get_local_rank()

        self.distributed_mode = "fsdp" if self.world_size > 1 else "single"

        # FSDP2 does not broadcast initial parameters between replica groups.
        # All ranks must construct the same model before sharding; the packed
        # dataloader seeds its generators separately for each data rank below.
        self._seed_everything(config.seed)
        self._build_models()
        self._build_optimizer()
        
        self.add_noise = partial(interpolate, T=config.num_train_timesteps)
        self.draw_timesteps = partial(
            draw_timesteps, distribution=config.snr_type, T=config.num_train_timesteps,
            loc=config.flow_lognorm_loc, scale=config.flow_lognorm_scale, default_device=self.device,
        )
        
        self.global_step = 0
        self.running_loss = None
        self.running_grad_norm = None
        self.consecutive_skipped_steps = 0
        self.total_skipped_steps = 0
        self.resume_dataloader_state: Optional[DataloaderState] = None
        self.last_dataloader_state: Optional[DataloaderState] = None
        
        if self.is_main_process:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        if config.report_to == "wandb" and self.is_main_process:
            import wandb
            wandb.init(
                settings=wandb.Settings(init_timeout=7200),
                project=os.environ.get("WANDB_PROJECT", "pan-video-training"),
                name=os.environ.get("WANDB_RUN_NAME", os.path.basename(os.path.normpath(config.output_dir))),
                config={
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "batch_size": config.batch_size,
                    "max_steps": config.max_steps,
                    "warmup_steps": config.warmup_steps,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "max_grad_norm": config.max_grad_norm,  
                    "train_timestep_shift": config.train_timestep_shift,
                    "train_timestep_shift_mode": config.train_timestep_shift_mode.value,
                    "train_timestep_rescale_factor": config.train_timestep_rescale_factor,
                    "train_timestep_max_shift": config.train_timestep_max_shift,
                    "flow_lognorm_loc": config.flow_lognorm_loc,
                    "flow_lognorm_scale": config.flow_lognorm_scale,
                    "snr_type": config.snr_type.value,
                    "cfg_drop_prob": config.cfg_drop_prob,
                    "cp_size": config.cp_size,
                    "world_size": self.world_size,
                    "use_muon": config.use_muon,
                    "zero_init_modulation": config.zero_init_modulation,
                },
            )
            logger.info("Wandb initialized")
        self.loss_cpu = torch.tensor(0.0, dtype=torch.float32, device="cpu", pin_memory=True)
        self.grad_norm_cpu = torch.tensor(0.0, dtype=torch.float32, device="cpu", pin_memory=True)
        self.running_seq_count = None
        self.running_image_count = None
        self.running_joint_length = None
        self.running_video_length = None
        self.running_caption_length = None

    @staticmethod
    def _seed_everything(seed: int):
        # Python's generator (data helpers), then torch's CPU and per-device CUDA generators.
        for seed_fn in (random.seed, torch.manual_seed):
            seed_fn(seed)

    @staticmethod
    def _make_channels_last(model):
        """Convert Conv2d and Conv3d weights to channels-last memory format."""
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data = module.weight.data.to(memory_format=torch.channels_last)
            elif isinstance(module, nn.Conv3d):
                module.weight.data = module.weight.data.to(memory_format=torch.channels_last_3d)
    
    def _build_text_encoder(self):
        """The frozen Qwen encoder the packed dataloader runs captions through
        (same loader as inference: pan.models.text_encoders)."""
        text_encoder = load_qwen_text_encoder(
            hidden_state_skip_layer=self.config.text_encoder_hidden_state_skip_layer,
            device="cuda",
        )
        if self.config.compile_text_encoder:
            text_encoder.compile(mode="max-autotune")
        return text_encoder

    def _build_models(self):
        # Parameters live in fp32 (the optimizer's master copy); FSDP's mixed-precision
        # policy casts them to bf16 for compute.
        master_dtype = torch.float32

        transformer_overrides = dict(self.config.transformer_config_overrides or {})
        if "zero_init_modulation" in transformer_overrides:
            if transformer_overrides["zero_init_modulation"] != self.config.zero_init_modulation:
                if self.is_main_process:
                    logger.warning(
                        "Overriding transformer_config_overrides['zero_init_modulation'] "
                        f"({transformer_overrides['zero_init_modulation']}) with CLI/config value "
                        f"({self.config.zero_init_modulation})."
                    )
        transformer_overrides["zero_init_modulation"] = self.config.zero_init_modulation

        # The VAE is not loaded: the packed loader streams precomputed, normalized
        # Wan 2.2 latents.
        latent_channels = WAN22_LATENT_CHANNELS
        expected_in_channels = (2 * latent_channels + 1
                                if self.config.conditioning_schema == ConditioningSchema.CHANNEL_CONCAT
                                else latent_channels)
        expected_out_channels = latent_channels
        for key, expected in (("in_channels", expected_in_channels), ("out_channels", expected_out_channels)):
            if key in transformer_overrides and int(transformer_overrides[key]) != expected:
                raise ValueError(
                    f"{self.config.conditioning_schema.value} conditioning with Wan 2.2 latents expects "
                    f"transformer_config_overrides.{key}={expected}, got {transformer_overrides[key]}."
                )
            transformer_overrides.setdefault(key, expected)

        # A DCP training checkpoint can only be loaded into the FSDP-wrapped model
        # (see _warm_start_from_dcp); plain state-dict files load here, pre-wrap.
        self._warm_start_dcp_dir = (
            _resolve_dcp_transformer_dir(self.config.pretrained_transformer_path)
            if self.config.load_transformer_weights else None
        )
        self.transformer: PanVideoDiffusionTransformer = PanVideoPipeline.build_transformer(
            transformer_config_overrides=transformer_overrides,
            pretrained_transformer_path=self.config.pretrained_transformer_path,
            load_transformer_weights=self.config.load_transformer_weights and self._warm_start_dcp_dir is None,
            transformer_dtype=master_dtype,
            device=self.device,
        )

        self.text_tokenizer_path = QWEN35_TEXT_ENCODER_ID
        self.text_template_video_cfg = deepcopy(PROMPT_TEMPLATE["video"])
        # Only the cp rank that runs the dataloader ever encodes text, so the rest
        # of the cp group skips loading the (multi-GB, frozen) Qwen encoder
        # altogether -- HBM is the reason cp is on in the first place.
        if self.parallel_state.cp_enabled and self.parallel_state.cp_rank != 0:
            loguru.logger.info(
                f"cp_rank={self.parallel_state.cp_rank}: skipping the text encoder load; "
                "cp_rank 0 of this group runs the dataloader."
            )
            self.text_encoder = None
        else:
            self.text_encoder = self._build_text_encoder()

        self.transformer.train()

        if self.distributed_mode != "fsdp":
            self._make_channels_last(self.transformer)

        if self.config.enable_gradient_checkpointing:
            self._wrap_activation_checkpointing()

        if self.distributed_mode == "fsdp":
            self._apply_fsdp()

        if self.config.compile_transformer:
            # Packed batches have variable token counts, so always compile
            # with dynamic shapes and without CUDA graph capture.
            self.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=True)

        if self.is_main_process:
            base_transformer = self._unwrap_transformer()
            arch = base_transformer.config
            logger.info(f"Models loaded. Transformer dtype: {master_dtype}")
            self._log_model_parameter_summary()
            logger.info(
                f"Transformer: hidden_size={arch.hidden_size}, num_attention_heads={arch.num_attention_heads}, "
                f"num_layers={arch.num_layers}, mlp_ratio={arch.mlp_ratio}, context_mlp_ratio={arch.context_mlp_ratio}, "
                f"attention_backend={arch.attention_backend}"
            )
            logger.info(f"Load transformer weights: {self.config.load_transformer_weights}")
            if self.config.load_transformer_weights:
                logger.info(f"Pretrained transformer path: {self.config.pretrained_transformer_path}")
            if self.config.transformer_config_overrides:
                logger.info(f"Transformer config overrides: {self.config.transformer_config_overrides}")
            logger.info(f"Zero-init modulation: {self.config.zero_init_modulation}")
            logger.info(f"Distributed mode: {self.distributed_mode}")
            logger.info(f"Hybrid shard enabled: {self.config.hybrid_shard}")
            logger.info(f"Activation checkpointing: {'on' if self.config.enable_gradient_checkpointing else 'off'}")
            logger.info(f"Timestep distribution: {self.config.snr_type.value}")
            logger.info(
                "Timestep lognorm params: "
                f"loc={self.config.flow_lognorm_loc}, scale={self.config.flow_lognorm_scale}"
            )
            logger.info(
                "Timestep shift: "
                f"mode={self.config.train_timestep_shift_mode.value}, "
                f"fixed_shift={self.config.train_timestep_shift}, "
                f"rescale_factor={self.config.train_timestep_rescale_factor}, "
                f"max_shift={self.config.train_timestep_max_shift}"
            )

    def _unwrap_transformer(self):
        model = self.transformer
        model = getattr(model, "_orig_mod", model)  # Unwrap torch.compile
        model = getattr(model, "module", model)  # Unwrap DDP or FSDP
        return model

    @staticmethod
    def _resolve_module(obj):
        if isinstance(obj, nn.Module):
            return obj
        maybe_model = getattr(obj, "model", None)
        if isinstance(maybe_model, nn.Module):
            return maybe_model
        return None

    @staticmethod
    def _new_parameter_stats():
        return {
            "numel": 0,
            "trainable_numel": 0,
            "tensors": 0,
            "trainable_tensors": 0,
        }

    @classmethod
    def _add_parameter_to_stats(cls, stats, param):
        stats["tensors"] += 1
        stats["numel"] += param.numel()
        if param.requires_grad:
            stats["trainable_tensors"] += 1
            stats["trainable_numel"] += param.numel()

    @classmethod
    def _module_parameter_stats(cls, module: Optional[nn.Module], seen_param_ids=None):
        stats = cls._new_parameter_stats()
        if module is None:
            return stats
        for param in module.parameters():
            if seen_param_ids is not None:
                param_id = id(param)
                if param_id in seen_param_ids:
                    continue
                seen_param_ids.add(param_id)
            cls._add_parameter_to_stats(stats, param)
        return stats

    @staticmethod
    def _format_parameter_stats(stats):
        frozen_numel = stats["numel"] - stats["trainable_numel"]
        frozen_tensors = stats["tensors"] - stats["trainable_tensors"]
        return (
            f"total={stats['numel']:,}, trainable={stats['trainable_numel']:,}, "
            f"frozen={frozen_numel:,}, tensors={stats['tensors']:,} "
            f"(trainable={stats['trainable_tensors']:,}, frozen={frozen_tensors:,})"
        )

    def _log_model_parameter_summary(self):
        if not self.is_main_process:
            return

        transformer = self._unwrap_transformer()
        text_encoder = self._resolve_module(self.text_encoder)
        components = {
            "transformer": transformer,
            "text_encoder": text_encoder,
        }

        logger.info("Model parameter count summary:")
        seen_param_ids = set()
        total_stats = self._new_parameter_stats()
        for component_name, module in components.items():
            if module is None:
                logger.warning(
                    f"Could not resolve {component_name} nn.Module when counting parameters; skipping."
                )
                continue
            stats = self._module_parameter_stats(module, seen_param_ids=seen_param_ids)
            for key, value in stats.items():
                total_stats[key] += value
            logger.info(f"  {component_name}: {self._format_parameter_stats(stats)}")
        logger.info(f"  total_unique: {self._format_parameter_stats(total_stats)}")

        if transformer is None:
            return

        top_level_stats = {}
        for param_name, param in transformer.named_parameters():
            top_name = param_name.split(".", 1)[0]
            stats = top_level_stats.setdefault(top_name, self._new_parameter_stats())
            self._add_parameter_to_stats(stats, param)

        if top_level_stats:
            logger.info("Transformer parameter count by top-level module:")
            for module_name in sorted(top_level_stats):
                logger.info(f"  transformer.{module_name}: {self._format_parameter_stats(top_level_stats[module_name])}")

        blocks = getattr(transformer, "blocks", None)
        if blocks is not None:
            block_type_counts = {}
            for block in blocks:
                block_type = type(block).__name__
                block_type_counts[block_type] = block_type_counts.get(block_type, 0) + 1
            if block_type_counts:
                block_types = ", ".join(
                    f"{block_type} x{count}" for block_type, count in sorted(block_type_counts.items())
                )
                logger.info(f"Transformer block types: {block_types}")

    
    def _apply_fsdp(self):
        """Shard each joint block, then the remaining top-level parameters, with FSDP2.

        Compute runs in bf16 with fp32 gradient reduction. Muon chunk hints are read
        off the parameters first, because sharding replaces them with DTensors.
        """
        if self.is_main_process:
            logger.info("Sharding the transformer with FSDP2")
        shard_kwargs = dict(
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            mesh=current_parallel_layout().fsdp_mesh,
        )
        if self.config.legacy_eager_mode:
            shard_kwargs["reshard_after_forward"] = True

        self._muon_chunks_config = {}
        for name, param in self.transformer.named_parameters():
            for dim in (0, 1):
                n_chunks = getattr(param, f"_muon_chunks_dim_{dim}", None)
                if n_chunks is not None:
                    self._muon_chunks_config[name] = (dim, n_chunks)
                    break

        for block in self.transformer.blocks:
            fully_shard(block, **shard_kwargs)
        fully_shard(self.transformer, **shard_kwargs)
        if self.is_main_process:
            logger.info("FSDP2 sharding done")

    def _wrap_activation_checkpointing(self):
        """Recompute activations of the caption-encoder blocks, joint blocks, patch embed and output head.

        With ``use_dedicated_gradient_checkpointing`` the flash-attention forward and the
        grouped QKV projection outputs are kept instead of recomputed (selective AC).
        """
        if self.is_main_process:
            logger.info("Wrapping transformer blocks for activation checkpointing")
        keep_ops = [
            torch.ops.fav.flash_varlen_fwd.default,
            torch.ops.gmlp.grouped_qkv_proj_fwd.default,
        ] if self.config.use_dedicated_gradient_checkpointing else []

        def wrap(module):
            return checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                context_fn=lambda: create_selective_checkpoint_contexts(keep_ops),
            )

        wrapped_types = {CaptionEncoderBlock, type(self.transformer.patch_embed), type(self.transformer.output_head)}
        wrapped_types |= {type(block) for block in self.transformer.blocks if block is not None}
        apply_activation_checkpointing(
            self.transformer,
            checkpoint_wrapper_fn=wrap,
            check_fn=lambda m: isinstance(m, tuple(wrapped_types)),
        )
        if self.is_main_process:
            logger.info("Activation checkpointing in place")

    def _build_optimizer(self):
        optimizer_model = self._unwrap_transformer()
        self.optimizer_map = {}
        self.optimizers = []
        self.lr_scheduler_map = {}
        self.lr_schedulers = []
        if self.config.legacy_eager_mode:
            lr = torch.tensor(self.config.learning_rate, dtype=torch.float32, device=self.device)
        else:
            lr = self.config.learning_rate

        if self.config.use_muon:
            native_bundle = build_native_muon_optimizers(
                model=optimizer_model,
                lr=lr,
                weight_decay=self.config.weight_decay,
                capturable=self.config.legacy_eager_mode,
                muon_chunks=getattr(self, '_muon_chunks_config', {}),
            )
            self.optimizer_map = native_bundle["optimizers"]
            if self.is_main_process:
                replace_number = re.compile(r"\.\d+\.")
                param_names = native_bundle.get("param_names", {})
                muon_stats = native_bundle.get("stats")
                if muon_stats is not None:
                    logger.info(
                        "Muon split summary: "
                        f"muon_tensors={muon_stats['muon_param_tensors']} "
                        f"(numel={muon_stats['muon_param_numel']:,}), "
                        f"adamw_tensors={muon_stats['adamw_param_tensors']} "
                        f"(numel={muon_stats['adamw_param_numel']:,}), "
                        f"excluded_embedding_or_head_tensors={muon_stats['excluded_embedding_or_head_tensors']}"
                    )
                muon_param_names = param_names.get("muon", [])
                if muon_param_names:
                    logger.info(f"Muon parameter list ({len(muon_param_names)} tensors):")
                    for name in dict.fromkeys(replace_number.sub(".x.", n) for n in muon_param_names):
                        logger.info(f"  [muon] {name}")
                muon_chunked_param_names = param_names.get("muon_chunked", [])
                if muon_chunked_param_names:
                    logger.info(f"Muon chunked parameter list ({len(muon_chunked_param_names)} tensors):")
                    for name, dim, n_chunks in dict.fromkeys(
                        (replace_number.sub('.x.', name), dim, n_chunks)
                        for name, dim, n_chunks in muon_chunked_param_names
                    ):
                        logger.info(f"  [muon_chunked] {name} dim={dim} n_chunks={n_chunks}")
                adamw_param_names = param_names.get("adamw", [])
                if adamw_param_names:
                    logger.info(f"AdamW parameter list ({len(adamw_param_names)} tensors):")
                    for name in dict.fromkeys(replace_number.sub(".x.", n) for n in adamw_param_names):
                        logger.info(f"  [adamw] {name}")
        else:
            # PyTorch's default betas (0.9, 0.999) and eps (1e-8).
            self.optimizer_map["adamw"] = torch.optim.AdamW(
                list(optimizer_model.parameters()), lr=lr, weight_decay=self.config.weight_decay,
                capturable=self.config.legacy_eager_mode,
            )

        for name in ("muon", "adamw"):
            optimizer = self.optimizer_map.get(name)
            if optimizer is None:
                continue
            self.optimizers.append(optimizer)
            self.lr_scheduler_map[name] = build_lr_scheduler(
                optimizer,
                scheduler_type=self.config.lr_scheduler_type,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.lr_decay_steps,
                max_steps=self.config.max_steps,
                warmup_start_ratio=self.config.lr_warmup_start_ratio,
                decay_floor_ratio=self.config.lr_decay_floor_ratio,
            )
            self.lr_schedulers.append(self.lr_scheduler_map[name])

        if not self.optimizers:
            raise RuntimeError("No optimizers were created.")

        if self.config.compile_optimizer:
            for optimizer in self.optimizers:
                optimizer.step = torch.compile(optimizer.step, mode="max-autotune-no-cudagraphs", dynamic=False)

        if self.is_main_process:
            logger.info("Optimizer and scheduler initialized")
            optimizer_names = [type(opt).__name__ for opt in self.optimizers]
            logger.info(f"Optimizers: {' + '.join(optimizer_names)}")
            resolved_decay_steps = self.config.lr_decay_steps
            if resolved_decay_steps is None:
                resolved_decay_steps = max(self.config.max_steps - self.config.warmup_steps, 0)
            if self.config.lr_scheduler_type == "constant_with_warmup":
                logger.info(
                    "LR schedule: constant_with_warmup; linear warmup "
                    f"{self.config.lr_warmup_start_ratio:.4g}->1.0 over {self.config.warmup_steps} steps; "
                    "then hold 1.0"
                )
            else:
                decay_name = "linear" if self.config.lr_scheduler_type == "linear_decay" else "cosine"
                logger.info(
                    f"LR schedule: {self.config.lr_scheduler_type}; linear warmup "
                    f"{self.config.lr_warmup_start_ratio:.4g}->1.0 over {self.config.warmup_steps} steps; "
                    f"{decay_name} decay 1.0->{self.config.lr_decay_floor_ratio:.4g} over {resolved_decay_steps} steps"
                )

    def train_step(self, batch: PackedBatch) -> Dict[str, Any]:
        bf16 = self.config.dtype == "bf16"
        accum = self.config.gradient_accumulation_steps

        # Tell the quack autotune patch this step's packed sequence length so
        # seq-length dims are dropped from the tuning key a priori -- the first
        # autotune's key then already covers every future length (no re-tunes).
        register_quack_dynamic_sizes(batch.packed_seq_params.joint_packed_length)

        def forward_backward():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if bf16 else torch.float32,
                enabled=bf16,
                cache_enabled=not self.config.legacy_eager_mode,
            ), torch.cuda.nvtx.range("forward"):
                if self.config.compile_transformer:
                    torch._dynamo.mark_dynamic(batch.hidden_states, 1)
                    torch._dynamo.mark_dynamic(batch.text_states, 0, min=1, max=self.config.packed_caption_token_budget)
                    torch._dynamo.mark_dynamic(batch.timestep, 0)
                    torch._dynamo.mark_dynamic(batch.target, 0)
                    batch.packed_seq_params.mark_dynamic()
                loss = self.transformer(**batch.__dict__)

            with torch.cuda.nvtx.range("backward"):
                loss = loss / accum  # the logged loss is this per-micro-step value
                loss.backward()

            with torch.cuda.nvtx.range("grad_norm"):
                last_micro_step = (self.global_step + 1) % accum == 0
                max_norm = self.config.max_grad_norm
                if last_micro_step and max_norm > 0:
                    grad_norm = clip_grad_norm_(self.transformer.parameters(), max_norm)
                else:
                    grad_norm = torch.tensor(0.0, device=loss.device)

            with torch.cuda.nvtx.range("metrics_copy"):
                self.loss_cpu.copy_(loss.detach(), non_blocking=True)
                grad_norm_local = grad_norm.to_local() if hasattr(grad_norm, 'to_local') else grad_norm
                self.grad_norm_cpu.copy_(grad_norm_local.detach(), non_blocking=True)

        def optimizer_step(apply_update: bool = True):
            # zero_grad runs even when the update is skipped: set_to_none makes the
            # next backward assign a fresh grad, whereas a non-finite grad left in
            # place would be accumulated into and stay non-finite forever.
            with torch.cuda.nvtx.range("optimizer_step"):
                for optimizer in self.optimizers:
                    if apply_update:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        def update_is_finite() -> bool:
            """Whether this step's update may be applied.

            This has to be answered *before* optimizer.step(): Muon asserts on
            non-finite input from inside a CUDA kernel (patch_muon._assert_no_nan),
            which kills the context on every rank, and it folds the gradient into
            its momentum buffers before that assert runs. _maybe_stop_for_anomaly()
            checks the same values but only sees them after the step, too late.

            The host wait is on the event train_step already blocks on below for
            metrics, so it costs a step's worth of launch lead-in, not a stall.
            """
            with torch.cuda.nvtx.range("finite_check"):
                self.metrics_ready_event.synchronize()
                local_ok = math.isfinite(self.loss_cpu.item()) and math.isfinite(self.grad_norm_cpu.item())
                if self.world_size == 1:
                    return local_ok
                # Every rank must reach the same verdict: Muon and FSDP both
                # collect inside step(), so a split decision deadlocks the job.
                # The clipped grad norm is already all-reduced and agrees
                # everywhere, but each rank has its own loss.
                ok = torch.tensor(int(local_ok), device="cpu", dtype=torch.int)
                dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                return bool(ok.item())

        if self.config.legacy_eager_mode:
            # Keep the validated eager execution stream and synchronization.
            self.side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side_stream):
                forward_backward()
                self.metrics_ready_event.record()
                step_applied = update_is_finite()
                optimizer_step(step_applied)
        else:
            forward_backward()
            self.metrics_ready_event.record()
            step_applied = update_is_finite()
            optimizer_step(step_applied)

        # Scheduler updates on the current stream must follow the optimizer.
        with torch.cuda.nvtx.range("lr_scheduler_step"):
            if self.config.legacy_eager_mode:
                torch.cuda.current_stream().wait_stream(self.side_stream)
            for scheduler in self.lr_schedulers:
                scheduler.step()

        with torch.cuda.nvtx.range("sync_for_metrics"):
            self.metrics_ready_event.synchronize()
            metrics = {
                "loss": self.loss_cpu.item() * self.config.gradient_accumulation_steps,
                "grad_norm": self.grad_norm_cpu.item(),
                "lr": self.lr_schedulers[0].get_last_lr()[0] if hasattr(self.lr_schedulers[0], "get_last_lr") else self.config.learning_rate,
                "step_skipped": not step_applied,
            }

        return metrics
    
    def _checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.config.output_dir, f"checkpoint-{step}")

    def save_checkpoint(self):
        """Write ``checkpoint-<global_step>/``: weights, optimizer state, schedulers + step, and each
        data rank's dataloader cursor. Every rank must call this."""
        step = self.global_step
        ckpt_root = self._checkpoint_dir(step)
        model_dir, optim_dir, state_file = _checkpoint_layout(ckpt_root)
        if self.is_main_process:
            os.makedirs(ckpt_root, exist_ok=True)
        # Metrics may still be in flight on the side stream.
        torch.cuda.current_stream().wait_stream(self.side_stream)

        self._save_model_and_optimizer(model_dir, optim_dir)
        if self.is_main_process:
            self._save_training_state(state_file, step)
        if self.last_dataloader_state is not None:
            cursor_dir = os.path.join(ckpt_root, "dataloader")
            os.makedirs(cursor_dir, exist_ok=True)
            Path(cursor_dir, f"{self.dp_rank}.json").write_text(self.last_dataloader_state.to_json())
        dist.barrier()

        if self.is_main_process:
            logger.info(f"Step {step}: checkpoint written to {ckpt_root}")
            record_dir = os.path.join(self.config.output_dir, "save_logging")
            os.makedirs(record_dir, exist_ok=True)
            with open(os.path.join(record_dir, f"checkpoint_{step}.txt"), "a") as f:
                f.write(f"{step}: {ckpt_root}\n")
            if self.config.save_total_limit > 0:
                self._cleanup_old_checkpoints()

    def _save_model_and_optimizer(self, model_dir: str, optim_dir: str) -> None:
        if self.distributed_mode == "fsdp":
            # Sharded DCP folders; every rank writes its own shards.
            dcp.save(get_model_state_dict(self.transformer), checkpoint_id=model_dir)
            dcp.save(get_optimizer_state_dict(self.transformer, self.optimizers), checkpoint_id=optim_dir)
            return
        if not self.is_main_process:
            return
        for folder in (model_dir, optim_dir):
            os.makedirs(folder, exist_ok=True)
        self._unwrap_transformer().save_pretrained(model_dir, safe_serialization=True)
        bundle = {"format_version": 2, "optimizers": {name: opt.state_dict() for name, opt in self.optimizer_map.items()}}
        torch.save(bundle, os.path.join(optim_dir, "optimizer_state.pt"))

    def _save_training_state(self, state_file: str, step: int) -> None:
        def scheduler_state(scheduler: LRScheduler):
            state = scheduler.state_dict()
            if isinstance(scheduler, LambdaLR):
                # base_lrs would force a device sync; resume with the same base learning rate instead.
                loguru.logger.warning(
                    "We don't save base_lr for LambdaLR schedulers to avoid synchrnozing device. "
                    "Be sure to run with the same base LR when resuming training."
                )
                state.pop("base_lrs", None)
            return state

        torch.save({
            "lr_schedulers": {name: scheduler_state(sched) for name, sched in self.lr_scheduler_map.items()},
            "global_step": step,
        }, state_file)

    def _cleanup_old_checkpoints(self):
        """Remove oldest checkpoints to keep only save_total_limit most recent."""
        checkpoint_dirs = []
        for d in os.listdir(self.config.output_dir):
            match = re.match(r"checkpoint-(\d+)$", d)
            if match:
                checkpoint_dirs.append((int(match.group(1)), os.path.join(self.config.output_dir, d)))
        checkpoint_dirs.sort(key=lambda x: x[0])
        while len(checkpoint_dirs) > self.config.save_total_limit:
            _, dir_to_remove = checkpoint_dirs.pop(0)
            logger.info(f"Removing old checkpoint: {dir_to_remove}")
            shutil.rmtree(dir_to_remove, ignore_errors=True)

    def _find_latest_two_checkpoints(self) -> tuple[str | None, str | None]:
        if not os.path.isdir(self.config.output_dir):
            return None, None
        checkpoint_dirs = []
        for d in os.listdir(self.config.output_dir):
            match = re.match(r"checkpoint-(\d+)$", d)
            if match:
                checkpoint_dirs.append((int(match.group(1)), os.path.join(self.config.output_dir, d)))
        checkpoint_dirs.sort(key=lambda x: x[0])
        if len(checkpoint_dirs) == 0:
            return None, None
        elif len(checkpoint_dirs) == 1:
            return checkpoint_dirs[-1][1], None
        else:
            return checkpoint_dirs[-1][1], checkpoint_dirs[-2][1]


    def _warm_start_from_dcp(self, model_dir: str):
        """Transformer weights only, from a DCP checkpoint written by this trainer.

        Loaded through the hybrid-shard reader into the FSDP-wrapped model, so
        each rank only materializes its own shard. Nothing else is restored.
        """
        self._dcp_load_into(get_model_state_dict(self.transformer), model_dir)
        if self.is_main_process:
            logger.info(
                f"Warm start from {model_dir}: transformer weights only; "
                "optimizer, LR schedulers, global_step and the dataloader cursor stay fresh")

    def _dcp_load_into(self, state_dict, path):
        """In-place DCP load where each rank reads only its own FSDP shard."""
        dcp.load(state_dict=state_dict, checkpoint_id=path)

    def _load_checkpoint(self, checkpoint_path: str, load_dataloader_state: bool = True):
        if not Path(checkpoint_path).exists():
            raise ValueError(f"No checkpoint at {checkpoint_path}")
        given = os.path.normpath(checkpoint_path)
        if os.path.basename(given) == "transformer":  # pointed at checkpoint-N/transformer
            ckpt_root, model_dir = os.path.dirname(given), checkpoint_path
        else:
            ckpt_root, model_dir = checkpoint_path, os.path.join(checkpoint_path, "transformer")
        _, optim_dir, state_file = _checkpoint_layout(ckpt_root)
        optim_file = os.path.join(optim_dir, "optimizer_state.pt")
        shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
        base_transformer = self._unwrap_transformer()
        if self.is_main_process:
            logger.info(f"Resuming from {ckpt_root}")

        if shards:
            weights = {}
            for shard in shards:
                weights |= safetensors.torch.load_file(shard, device="cpu")
            missing = base_transformer.load_state_dict(weights, strict=False).missing_keys
            if missing:
                raise ValueError(f"{model_dir} is missing {len(missing)} transformer tensors, e.g. {missing[:5]}")
            if self.distributed_mode != "fsdp":
                self._make_channels_last(base_transformer)
            if self.is_main_process:
                logger.info("Transformer weights restored from safetensors")
        elif os.path.exists(model_dir):
            self._dcp_load_into(get_model_state_dict(self.transformer), model_dir)
            if self.is_main_process:
                logger.info("Transformer weights restored from DCP")
        else:
            raise ValueError(f"{checkpoint_path} holds no transformer weights")

        if os.path.exists(optim_file):
            saved = torch.load(optim_file, map_location="cpu")["optimizers"]
            for name, optimizer in self.optimizer_map.items():
                optimizer.load_state_dict(saved[name])
                if self.is_main_process:
                    logger.info(f"Optimizer '{name}' restored from {optim_file}")
        elif os.path.exists(optim_dir):
            self._dcp_load_into(get_optimizer_state_dict(self.transformer, self.optimizers), optim_dir)
            if self.is_main_process:
                logger.info("Optimizer state restored from DCP")
        else:
            raise ValueError(f"{checkpoint_path} holds no optimizer state")

        # A failed load attempt can abort torch's DCP `_init_optim_state` between its
        # lr swap (to a CPU torch.tensor(0.0)) and the restore, leaving a wrong lr in
        # the param groups. Fail fast here instead of crashing in the muon Triton
        # kernel (or silently training with lr=0) on the first step after resume.
        for opt_name, optimizer in self.optimizer_map.items():
            for group in optimizer.param_groups:
                group_lr = group.get("lr")
                if isinstance(group_lr, torch.Tensor) and group_lr.device.type == "cpu":
                    raise RuntimeError(
                        f"Optimizer '{opt_name}' param_group lr is a CPU tensor "
                        f"(value={group_lr.item()}) after checkpoint load; a prior "
                        "load attempt likely failed mid-way through DCP's "
                        "_init_optim_state and leaked its temporary lr."
                    )

        self._load_training_state(state_file)

        if load_dataloader_state:
            dl_path = os.path.join(ckpt_root, "dataloader", f"{self.dp_rank}.json")
            if not os.path.exists(dl_path):
                raise ValueError(
                    f"Checkpoint {ckpt_root} has no dataloader state for dp_rank "
                    f"{self.dp_rank} ({dl_path}). The checkpoint was written by a run "
                    f"with a different data-parallel size; re-run without the dataloader "
                    f"resume, or use a checkpoint from a matching DP layout.")
            with open(dl_path) as f:
                self.resume_dataloader_state = DataloaderState.from_json(f.read())
            if not self.config.resume_video_csv:
                # Keep the image-tar cursor; start the video stream from scratch.
                self.resume_dataloader_state = self.resume_dataloader_state.reset_video_stream()
                if self.is_main_process:
                    logger.info("resume_video_csv=false: video CSV stream reset; image-tar resume preserved")

        if self.is_main_process:
            logger.info(f"Resume complete at step {self.global_step}")

    def _load_training_state(self, state_file: str):
        """LR schedulers + global_step."""
        state = torch.load(state_file, map_location=self.device)
        for name, scheduler in self.lr_scheduler_map.items():
            scheduler.load_state_dict(state["lr_schedulers"][name])
        self.global_step = state["global_step"]
        if self.is_main_process:
            logger.info(f"LR schedulers and step counter restored (step {self.global_step})")

    def load_checkpoint(self, load_dataloader_state: bool = True) -> str | None:
        """Resume from the latest checkpoint-N in output_dir (falling back to the
        one before it if the latest is unreadable). Returns the dir used, or None
        when output_dir has no checkpoint yet."""
        last, second_last = self._find_latest_two_checkpoints()
        if last is None:
            return None

        try:
            self._load_checkpoint(last, load_dataloader_state=load_dataloader_state)
            return last
        except (Exception, dcp.api.CheckpointException) as e1:
            if second_last is None:
                raise ValueError(
                    f"Failed to load checkpoint: {e1!s} for the last checkpoint, no second last checkpoint"
                ) from e1
            try:
                self._load_checkpoint(second_last, load_dataloader_state=load_dataloader_state)
                return second_last
            except (Exception, dcp.api.CheckpointException) as e2:
                raise ValueError(
                    f"Failed to load checkpoint: {e1!s} for the last checkpoint, {e2!s} for the second last checkpoint"
                ) from e2

    def build_packed_dataloader(self, resume_state: Optional[DataloaderState] = None):
        csv_paths = []
        for p in self.config.packed_dataloader_meta_csv_path:
            if os.path.isdir(p):
                csv_paths.extend(sorted(glob.glob(os.path.join(p, "*.csv"))))
            else:
                csv_paths.append(p)

        csv_paths = sorted(csv_paths)

        dp_group = current_parallel_layout().dp_mesh.get_group()
        # Each data-parallel rank streams whole CSVs. A rank with none would spin in
        # the loader forever while the others time out in a collective ~30 min later,
        # so fail every rank now, with the reason.
        if len(csv_paths) < dp_group.size():
            message = (
                f"{len(csv_paths)} CSV file(s) in packed_dataloader_meta_csv_path for "
                f"{dp_group.size()} data-parallel ranks: every rank needs at least one CSV, so "
                f"{dp_group.size() - len(csv_paths)} rank(s) would get none and hang. Split the "
                f"metadata into at least {dp_group.size()} CSVs (ideally a multiple of it), or "
                "use fewer nodes / a larger cp_size.")
            loguru.logger.error(message)
            raise ValueError(message)
        csv_paths = csv_paths[dp_group.rank()::dp_group.size()]

        loguru.logger.info(f"Streaming {len(csv_paths)} CSV file(s) lazily")

        image_tar_shards: list[str] = []
        if self.config.image_tar_path and self.config.image_sample_prob > 0.0:
            from dataset.image_item_processor import resolve_image_tar_shards
            image_tar_shards = resolve_image_tar_shards(self.config.image_tar_path)
            # Deterministic global shuffle (fixed seed) BEFORE per-rank sharding:
            # every DP rank applies the same permutation to the same sorted list,
            # so the slice each rank takes is reproducible across runs and the
            # shard indices stored on resume stay valid.
            random.Random(_IMAGE_TAR_SHUFFLE_SEED).shuffle(image_tar_shards)
            image_tar_shards = image_tar_shards[dp_group.rank()::dp_group.size()]
            loguru.logger.info(
                f"Interleaving {len(image_tar_shards)} image tar shard(s) at "
                f"image_sample_prob={self.config.image_sample_prob}"
            )

        base_transformer = self._unwrap_transformer()
        # Latents are precomputed offline; validate that the configured latent
        # geometry (resolution / downsample ratio / patch size) matches the
        # token-per-frame count the loader assumes.
        spatial_ratio = WAN22_SPATIAL_RATIO
        pt, ph, pw = base_transformer.patch_size
        if self.config.video_height % spatial_ratio != 0 or self.config.video_width % spatial_ratio != 0:
            raise ValueError(
                f"Configured resolution {(self.config.video_height, self.config.video_width)} is not divisible by "
                f"latent spatial ratio {spatial_ratio}."
            )
        latent_h = self.config.video_height // spatial_ratio
        latent_w = self.config.video_width // spatial_ratio
        if latent_h % ph != 0 or latent_w % pw != 0:
            raise ValueError(
                f"Latent grid {(latent_h, latent_w)} derived from resolution "
                f"{(self.config.video_height, self.config.video_width)} and latent spatial ratio "
                f"{spatial_ratio} is not divisible by transformer patch size {base_transformer.patch_size}."
            )
        rope_axes, rope_theta, rope_scale = base_transformer.rope_settings()
        return PackedIterator(
            csv_paths=csv_paths,
            text_encoder=self.text_encoder,
            text_tokenizer_path=self.text_tokenizer_path,
            text_template_video_cfg=deepcopy(self.text_template_video_cfg),
            timestep_sample_fn=self.draw_timesteps,
            timestep_transform_fn=partial(
                shift_timesteps,
                T=self.config.num_train_timesteps,
                shift=self.config.train_timestep_shift,
                mode=self.config.train_timestep_shift_mode,
                rescale_factor=self.config.train_timestep_rescale_factor,
                max_shift=self.config.train_timestep_max_shift,
                frame_stride=WAN22_TEMPORAL_RATIO * base_transformer.patch_size[0],
                pixel_stride_h=WAN22_SPATIAL_RATIO * base_transformer.patch_size[1],
                pixel_stride_w=WAN22_SPATIAL_RATIO * base_transformer.patch_size[2],
            ),
            add_noise_fn=self.add_noise,
            # theta and position_scale go in as keywords: the packing dataset rebuilds the call
            # for explicit positions from this partial's args and keywords
            # (dataset/packing_dataset.py `_rope_kwargs_from_callable`).
            rope_embed_fn=partial(rope_tables_for_grid, rope_axes, theta=rope_theta, position_scale=rope_scale),
            random_seed=hash((self.config.seed, self.dp_rank, self.global_step)),
            cfg_drop_prob=self.config.cfg_drop_prob,
            image_cond_prob=self.config.image_cond_prob,
            video_cond_prob=self.config.video_cond_prob,
            conditioning_schema=self.config.conditioning_schema.value,
            num_buffers=self.config.packed_dataloader_num_buffers,
            packed_joint_token_budget=self.config.packed_joint_token_budget,
            packed_video_token_budget=self.config.packed_video_token_budget,
            packed_caption_token_budget=self.config.packed_caption_token_budget,
            max_num_seq=self.config.packed_dataloader_max_num_seq,
            patch_size=base_transformer.patch_size,
            resize=(self.config.video_height, self.config.video_width),
            num_workers=self.config.packed_dataloader_num_workers,
            prefetch_factor=self.config.packed_dataloader_prefetch_factor,
            dtype=torch.bfloat16 if self.config.dtype == "bf16" else torch.float32,
            packed_caption_segment_limit=self.config.packed_caption_segment_limit,
            packed_joint_segment_limit=self.config.packed_joint_segment_limit,
            image_tar_shards=image_tar_shards,
            image_sample_prob=self.config.image_sample_prob,
            resume_state=resume_state,
            cp_enabled=self.parallel_state.cp_enabled,
        )

    def train(self):
        cfg = self.config
        max_steps = cfg.max_steps
        if self.is_main_process:
            logger.info(
                f"Training for {max_steps} steps: batch_size={cfg.batch_size} lr={cfg.learning_rate} "
                f"weight_decay={cfg.weight_decay} zero_init_modulation={cfg.zero_init_modulation}"
            )

        if self.is_main_process:
            logger.info(
                "The first step compiles and autotunes kernels. On a cold QUACK_CACHE_DIR "
                f"({os.environ.get('QUACK_CACHE_DIR', 'unset')}) this takes over 30 minutes at "
                "production sizes; [quack] lines below report each kernel as it is tuned.")
        training_start_time = prev_step_time = time.perf_counter()
        train_step_time = 0.0

        resumed = self.load_checkpoint()
        if self._warm_start_dcp_dir is not None:
            if resumed is None:
                # Warm start: transformer weights only, no dataloader cursor (the
                # source is usually a different run on a different dataset / DP
                # layout). Only while output_dir has no checkpoints of its own:
                # once this run has saved, a restarted job (e.g. the next link of
                # a submit chain) resumes its own progress instead.
                self._warm_start_from_dcp(self._warm_start_dcp_dir)
            elif self.is_main_process:
                logger.info(
                    f"{self.config.output_dir} already has checkpoints; resumed {resumed} "
                    f"instead of warm starting from {self._warm_start_dcp_dir}")
        dist.barrier()
        # Under context parallelism only cp_rank 0 of each cp group runs the
        # dataloader (CSV streaming, latent download, text encoder); the assembled
        # batch is broadcast to the group and sharded per rank in the loop below.
        ps = self.parallel_state
        if ps.cp_enabled and ps.cp_rank != 0:
            dataloader = None
        else:
            packed_dataloader = self.build_packed_dataloader(resume_state=self.resume_dataloader_state)
            dataloader = iter(packed_dataloader)

        self.transformer.train()

        # Heartbeat watchdog: per-rank thread that force-exits the process if the
        # main loop stops making progress, so silent hangs become diagnosable
        # (rank + last stage land in this rank's torchrun stderr).
        self._heartbeat_last_time = time.monotonic()
        self._heartbeat_last_stage = "init"
        self._heartbeat_last_step = self.global_step
        _heartbeat_threshold_s = 50000.0
        _heartbeat_poll_s = 5.0
        _heartbeat_rank = self.rank

        def _heartbeat_loop():
            while True:
                time.sleep(_heartbeat_poll_s)
                stuck = time.monotonic() - self._heartbeat_last_time
                if stuck > _heartbeat_threshold_s:
                    msg = (
                        f"[heartbeat] rank={_heartbeat_rank} stuck for {stuck:.1f}s "
                        f"at stage='{self._heartbeat_last_stage}' "
                        f"step={self._heartbeat_last_step}; dumping all-thread tracebacks then force exiting"
                    )
                    print(msg, file=sys.stderr, flush=True)
                    try:
                        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    except Exception:
                        pass
                    sys.stderr.flush()
                    os._exit(1)

        threading.Thread(target=_heartbeat_loop, daemon=True).start()

        pre_data_event = torch.cuda.Event(enable_timing=True)
        post_data_event = torch.cuda.Event(enable_timing=True)
        post_step_event = torch.cuda.Event(enable_timing=True)
        pre_data2_event = torch.cuda.Event(enable_timing=True)
        post_data2_event = torch.cuda.Event(enable_timing=True)
        post_step2_event = torch.cuda.Event(enable_timing=True)
        pre_data2_event.record()
        post_data2_event.record()
        post_step2_event.record()

        while self.global_step < max_steps:
            self._heartbeat_last_stage = "dataloader_next"
            self._heartbeat_last_time = time.monotonic()
            with torch.cuda.nvtx.range("dataloader_next"):
                pre_data_event.record()
                batch, meta = next(dataloader) if dataloader is not None else (None, None)
                if ps.cp_enabled:
                    batch, meta = split_packed_batch_for_cp(batch, meta, ps)
                # None on the cp ranks that do not run a dataloader; only the one
                # that does has a cursor worth checkpointing.
                if meta.dataloader_state is not None:
                    self.last_dataloader_state = meta.dataloader_state
                post_data_event.record()

            if self.global_step >= max_steps:
                break

            self._heartbeat_last_stage = "train_step"
            self._heartbeat_last_time = time.monotonic()
            stats = self.train_step(batch)
            self._update_utilization_ema(meta)
            self._heartbeat_last_step = self.global_step

            if self.config.legacy_eager_mode:
                post_step_event.record(self.side_stream)
            else:
                post_step_event.record()

            with torch.cuda.nvtx.range("logging_and_checkpointing"):
                elapsed = time.perf_counter() - training_start_time
                step_time = time.perf_counter() - prev_step_time
                prev_step_time = time.perf_counter()
                data_time = pre_data2_event.elapsed_time(post_data2_event) / 1000 if pre_data2_event.query() and post_data2_event.query() else 0.0
                train_step_time = post_data2_event.elapsed_time(post_step2_event) / 1000 if post_data2_event.query() and post_step2_event.query() else 0.0

                if self.is_main_process and self.global_step % cfg.log_interval == 0:
                    # Keep this layout: log parsers match "Step N/M | Loss: ... | Grad Norm: ...".
                    fields = [
                        f"Step {self.global_step}/{max_steps}",
                        f"Loss: {stats['loss']:.6f}",
                        f"Grad Norm: {stats['grad_norm']:.4f}",
                        f"LR: {stats['lr']:.2e}",
                        f"CPU Wall: {step_time:.3f}s",
                        f"(Prev Step CUDA) Data: {data_time:.3f}s",
                        f"Train: {train_step_time:.3f}s",
                        f"Elapsed: {format_duration(elapsed)}",
                    ]
                    logger.info(" | ".join(fields))

                if self.config.report_to == "wandb" and self.is_main_process:
                    wandb_payload = {}
                    if self.global_step % self.config.log_interval == 0:
                        wandb_payload.update(
                            {
                                "train/loss": stats["loss"],
                                "train/grad_norm": stats["grad_norm"],
                                "train/lr": stats["lr"],
                            }
                        )
                    if wandb_payload:
                        import wandb
                        wandb.log(wandb_payload, step=self.global_step)

                step_skipped = bool(stats.get("step_skipped"))
                if step_skipped:
                    self.consecutive_skipped_steps += 1
                    self.total_skipped_steps += 1
                    if self.is_main_process:
                        logger.warning(
                            f"Skipped optimizer step at step {self.global_step}: non-finite update "
                            f"(loss={stats['loss']}, grad_norm={stats['grad_norm']}); "
                            f"{self.consecutive_skipped_steps} consecutive, {self.total_skipped_steps} total"
                        )
                else:
                    self.consecutive_skipped_steps = 0

                self.global_step += 1
                self._heartbeat_last_step = self.global_step
                self._heartbeat_last_time = time.monotonic()

                # Keeps the EMAs fed on every step, so it has to run unconditionally.
                should_recover = self.config.anomaly_detection and self._maybe_stop_for_anomaly(stats)
                if should_recover and step_skipped and self.consecutive_skipped_steps <= self.config.max_consecutive_skipped_steps:
                    # The update was never applied, so model and optimizer state are
                    # still the pre-anomaly state and there is nothing to roll back
                    # to; recovering would only discard the good steps since the last
                    # checkpoint. If the skips keep coming, fall through to recovery.
                    should_recover = False
                    if self.is_main_process:
                        logger.warning(
                            "Not recovering: the anomalous update was skipped, so training state is clean "
                            f"({self.consecutive_skipped_steps}/{self.config.max_consecutive_skipped_steps} consecutive skips allowed)"
                        )

                if should_recover:
                    self._heartbeat_last_stage = "recover_from_checkpoint"
                    self._heartbeat_last_time = time.monotonic()
                    self._recover_from_latest_checkpoint()
                elif self.global_step % self.config.save_interval == 0:  # no need to save if we just recovered
                    self._heartbeat_last_stage = "save_checkpoint"
                    self._heartbeat_last_time = time.monotonic()
                    self._log_packing_utilization()
                    self.save_checkpoint()

                pre_data_event, pre_data2_event = pre_data2_event, pre_data_event
                post_data_event, post_data2_event = post_data2_event, post_data_event
                post_step_event, post_step2_event = post_step2_event, post_step_event

                if self.global_step == 10:
                    self._log_packing_utilization()
                    # let's try to make next(dataloader) in next iteration also profiled
                    # may behave incorrectly if dataloader has pre-fetch
                    dist.barrier()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()

        torch.cuda.cudart().cudaProfilerStop()

        self._log_packing_utilization()
        self.save_checkpoint()
        logger.info(f"Training finished at step {self.global_step}")
        if self.config.report_to == "wandb":
            import wandb
            wandb.finish()

        if self.distributed_mode == "fsdp":
            dist.barrier()
            dist.destroy_process_group()

    def _recover_from_latest_checkpoint(self) -> bool:
        if self.is_main_process:
            logger.warning("Recovering from latest checkpoint")

        path = self.load_checkpoint(load_dataloader_state=False)
        if path is None:
            raise ValueError("No checkpoint found for recovery")

        # The EMAs describe the pre-rollback trajectory, which we just discarded;
        # re-seed them from post-recovery steps so spike detection stays armed.
        self.running_loss = None
        self.running_grad_norm = None
        self.consecutive_skipped_steps = 0

        if self.is_main_process:
            logger.warning(f"Recovery complete from {path}")

    def _update_utilization_ema(self, meta: PackedDataloaderMeta):
        decay = 0.99
        def _ema(cur, val):
            return float(val) if cur is None else decay * cur + (1 - decay) * float(val)
        self.running_seq_count = _ema(self.running_seq_count, meta.active_sequence_count)
        self.running_image_count = _ema(self.running_image_count, meta.active_image_count)
        self.running_joint_length = _ema(self.running_joint_length, meta.active_joint_length)
        self.running_video_length = _ema(self.running_video_length, meta.active_video_length)
        self.running_caption_length = _ema(self.running_caption_length, meta.active_caption_length)

    def _log_packing_utilization(self):
        if self.running_seq_count is None:
            return
        buf = torch.tensor(
            [self.running_seq_count, self.running_image_count, self.running_joint_length,
             self.running_video_length, self.running_caption_length],
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf /= self.world_size
        if self.is_main_process:
            seq, image, joint, video, caption = buf.tolist()
            jb = self.config.packed_joint_token_budget
            vb = self.config.packed_video_token_budget
            cb = self.config.packed_caption_token_budget
            logger.info(
                f"(avg) sequence count: {seq:5.2f} (image: {image:5.2f}, video: {seq - image:5.2f}); "
                f"joint: {joint:5.0f}/{jb:5} = {joint / jb * 100:2.0f}%; "
                f"video/image: {video:5.0f}/{vb:5} = {video / vb * 100:2.0f}%; "
                f"caption: {caption:4.0f}/{cb:4} = {caption / cb * 100:2.0f}%"
            )

    def _maybe_stop_for_anomaly(self, metrics: Dict[str, float]) -> bool:
        decay = 0.99
        loss_spike_factor = 20.0
        grad_norm_spike_factor = 6.0
        grad_norm_spike_min = 1e-3
        grad_norm_hard_limit = 1e6
        anomaly_warmup_steps = max(int(self.config.anomaly_detection_warmup_steps), 0)
        spike_detection_enabled = self.global_step >= anomaly_warmup_steps

        loss_value = float(metrics["loss"])
        grad_norm = float(metrics["grad_norm"])

        # Only finite values may enter the EMAs. A single inf/nan would otherwise
        # poison the running average for the lifetime of the process, silently
        # disabling both spike branches (x > factor * inf is always False).
        if math.isfinite(loss_value):
            if self.running_loss is None:
                self.running_loss = loss_value
            else:
                self.running_loss = decay * self.running_loss + (1 - decay) * loss_value

        if math.isfinite(grad_norm):
            if self.running_grad_norm is None:
                self.running_grad_norm = grad_norm
            else:
                self.running_grad_norm = decay * self.running_grad_norm + (1 - decay) * grad_norm

        ema_loss_usable = self.running_loss is not None and math.isfinite(self.running_loss)
        ema_grad_norm_usable = self.running_grad_norm is not None and math.isfinite(self.running_grad_norm)

        local_should_stop = False
        reasons = []

        if not math.isfinite(loss_value):
            local_should_stop = True
            reasons.append(f"non-finite loss={loss_value}")
        elif (
            spike_detection_enabled
            and ema_loss_usable
            and self.running_loss > 0
            and loss_value > loss_spike_factor * self.running_loss
        ):
            local_should_stop = True
            reasons.append(
                f"loss spike: {loss_value:.6f} > {loss_spike_factor:.1f} * ema_loss({self.running_loss:.6f})"
            )

        if not math.isfinite(grad_norm):
            local_should_stop = True
            reasons.append(f"non-finite grad_norm={grad_norm}")
        elif grad_norm > grad_norm_hard_limit:
            local_should_stop = True
            reasons.append(
                f"grad norm overflow: {grad_norm:.6f} > {grad_norm_hard_limit:.1f}"
            )
        elif (
            spike_detection_enabled
            and grad_norm > grad_norm_spike_min
            and ema_grad_norm_usable
            and self.running_grad_norm > 0
            and grad_norm > grad_norm_spike_factor * self.running_grad_norm
        ):
            local_should_stop = True
            reasons.append(
                f"grad norm spike: {grad_norm:.6f} > {grad_norm_spike_factor:.1f} * ema_grad_norm({self.running_grad_norm:.6f})"
            )

        if local_should_stop and self.is_main_process:
            logger.error(
                f"Stopping training due to anomaly at step {self.global_step} on rank {self.rank}: "
                + "; ".join(reasons)
            )

        if self.world_size > 1:
            stop_tensor = torch.tensor(local_should_stop, device="cpu", dtype=torch.int)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            global_should_stop = stop_tensor.item()
            return global_should_stop

        return local_should_stop
    
def main():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.compiler.config.accumulated_recompile_limit = 2048
    torch._dynamo.config.cache_size_limit = 2048
    configure_logging()
    config = parse_training_config()

    if config.legacy_eager_mode:
        logger.info("legacy_eager_mode=true: using the validated eager training configuration; no CUDA graph capture/replay.")
        # Preserve the NCCL setting used by the validated execution path.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        assert config.gradient_accumulation_steps == 1, "legacy_eager_mode requires gradient_accumulation_steps == 1"
    trainer = PanVideoTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

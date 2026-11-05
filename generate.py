"""Video generation entry point for PAN 2 checkpoints.

Builds the transformer from the training YAML (``--training_config``), loads a
checkpoint (``--checkpoint_path``), and samples T2V / I2V videos for a prompt
JSON (``--prompt_json``) or direct ``--prompt`` strings. One process holds the
full model; run one worker per GPU and shard the prompts with
``--prompt_shard_index / --prompt_shard_count``. Launch with ``torchrun`` or plain ``python``.

The sampler controls (steps, flow shift, CFG window, STG, timestep
schedule, ...) mirror the pipeline's ``__call__`` keyword arguments.
"""
import os

# Set before torch is imported: expandable segments cut fragmentation for the large,
# variable-size video activations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import datetime
import glob
import json
import sys

import imageio
import loguru
import safetensors.torch
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from inference.inputs import GUIDANCE_KEYS, is_complete, load_inference_inputs
from pan.commons import parse_json_object, str_to_bool
from pan.commons.parallel_layout import init_parallel_layout
from pan.pipelines.pan_video_pipeline import OFFLOAD_MODES, PanVideoPipeline
from pan.utils.inference_guidance import validate_sampling_extensions

MODEL_LABEL = "pan2"  # recorded in each <video>_settings next to the output


def _log(message, level="INFO"):
    """Log from the first process only."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        loguru.logger.log(level, message)


def write_video(video, path, fps=24, video_crf=None, video_preset=None, video_pixel_format=None):
    """Write a float video in [0, 1], ``[C, F, H, W]`` or ``[1, C, F, H, W]``; one frame becomes a still image."""
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"write_video takes a single video, got a batch of {video.shape[0]}")
        video = video[0]
    frames = (video * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0)  # f h w c
    if frames.shape[0] == 1:
        imageio.imwrite(path, frames[0].cpu().numpy())
        return
    ffmpeg_params, options = [], {}
    if video_crf is not None:
        options.update(codec="libx264", quality=None)
        ffmpeg_params += ["-crf", str(video_crf)]
    if video_preset is not None:
        ffmpeg_params += ["-preset", video_preset]
    if ffmpeg_params:
        options["ffmpeg_params"] = ffmpeg_params
    if video_pixel_format is not None:
        options["pixelformat"] = video_pixel_format
    imageio.mimwrite(path, frames, fps=fps, **options)


def save_settings(args, video_path, task):
    """Write ``<video stem>_config.json``: the task and every argument the video was made with.

    ``inference.inputs.is_complete`` compares the ``arguments`` field with a rerun's arguments
    to decide whether the video can be skipped.
    """
    def as_json(value):
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    settings = {
        "model": MODEL_LABEL,
        "task": task,
        "video": video_path,
        "written_at": datetime.datetime.now().isoformat(),
        "arguments": {k: as_json(v) for k, v in vars(args).items() if not k.startswith("_") and not callable(v)},
    }
    settings_path = os.path.splitext(video_path)[0] + "_config.json"
    with open(settings_path, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2, ensure_ascii=False)
    print(f"Wrote generation settings to {settings_path}")
    return settings_path


def read_training_config(args):
    """Take the architecture from the training YAML, so inference builds exactly the trained model.

    ``model.transformer_config_overrides`` (with ``model.zero_init_modulation`` folded in) and
    ``model.text_encoder_hidden_state_skip_layer`` replace the corresponding flags;
    ``data.packed_caption_segment_limit`` becomes the text-encoder length.
    """
    if not args.training_config:
        return
    if not os.path.isfile(args.training_config):
        raise ValueError(f"--training_config not found: {args.training_config}")
    with open(args.training_config, "r", encoding="utf-8") as f:
        sections = yaml.safe_load(f) or {}
    model, data = sections.get("model") or {}, sections.get("data") or {}
    if not isinstance(model, dict) or not isinstance(data, dict):
        raise ValueError(f"'model' and 'data' in {args.training_config} must be mappings")
    if model.get("single_stream_block_type", "mm_joint_attn") != "mm_joint_attn":
        raise ValueError("model.single_stream_block_type must be mm_joint_attn (the only PAN 2 block)")

    arch = dict(model.get("transformer_config_overrides") or {})
    zero_init = model.get("zero_init_modulation")
    if zero_init is not None:
        if arch.get("zero_init_modulation", zero_init) != zero_init:
            _log(f"model.zero_init_modulation ({zero_init}) overrides transformer_config_overrides "
                 f"['zero_init_modulation'] ({arch['zero_init_modulation']}).", "WARNING")
        arch["zero_init_modulation"] = zero_init
    if arch:
        args.transformer_config_overrides = json.dumps(arch)
    if model.get("text_encoder_hidden_state_skip_layer") is not None:
        args.text_encoder_hidden_state_skip_layer = model["text_encoder_hidden_state_skip_layer"]
    if data.get("packed_caption_segment_limit") is not None:
        args.text_encoder_max_length = int(data["packed_caption_segment_limit"])
    _log(f"Architecture from {args.training_config}; text encoder: skip_layer="
         f"{args.text_encoder_hidden_state_skip_layer}, max_length={args.text_encoder_max_length}")


def load_transformer_weights(pipe, checkpoint_path):
    """Load DiT weights from a safetensors directory, a plain state-dict file, or a DCP checkpoint.

    Formats are tried in that order; every load is strict on keys (and, for DCP, shapes).
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"No checkpoint at {checkpoint_path}")
    _log(f"Reading transformer weights from {checkpoint_path}", "INFO")

    normalized = os.path.normpath(checkpoint_path)
    base_name = os.path.basename(normalized)

    plain_candidates = []
    dcp_candidates = []

    if os.path.isfile(normalized):
        weights_dir = os.path.dirname(normalized)
        plain_candidates.append(normalized)
    else:
        if base_name == "transformer":
            weights_dir = normalized
            ckpt_root = os.path.dirname(normalized)
        else:
            ckpt_root = normalized
            weights_dir = os.path.join(ckpt_root, "transformer")  # training layout: checkpoint-N/transformer

        plain_candidates.extend([
            os.path.join(weights_dir, "model_state.pt"),
            os.path.join(weights_dir, "pytorch_model.bin"),
            os.path.join(ckpt_root, "model_state.pt"),
        ])
        dcp_candidates.extend([weights_dir, normalized])

    def _has_dcp_metadata(path):
        if not os.path.isdir(path):
            return False
        metadata_names = {
            ".metadata",
            "metadata",
            "_metadata",
            ".metadata.tmp",
        }
        return any(os.path.exists(os.path.join(path, name)) for name in metadata_names)


    def _format_key_sample(keys, limit=20):
        keys = sorted(str(k) for k in keys)
        sample = ", ".join(keys[:limit])
        if len(keys) > limit:
            sample += f", ... (+{len(keys) - limit} more)"
        return sample

    def _raise_on_incompatible_keys(missing, unexpected, source):
        missing = list(missing)
        unexpected = list(unexpected)
        if missing or unexpected:
            raise ValueError(
                f"Strict checkpoint load failed for {source}: "
                f"missing={len(missing)} [{_format_key_sample(missing)}], "
                f"unexpected={len(unexpected)} [{_format_key_sample(unexpected)}]"
            )

    def _validate_dcp_metadata(path, expected_state_dict, key_prefix=""):
        metadata = FileSystemReader(path).read_metadata()
        checkpoint_meta = metadata.state_dict_metadata
        checkpoint_keys = set(checkpoint_meta.keys())
        expected_keys = {f"{key_prefix}{key}" for key in expected_state_dict.keys()}
        missing = expected_keys - checkpoint_keys
        unexpected = checkpoint_keys - expected_keys
        if missing or unexpected:
            raise ValueError(
                f"Strict DCP metadata check failed for {path}: "
                f"missing={len(missing)} [{_format_key_sample(missing)}], "
                f"unexpected={len(unexpected)} [{_format_key_sample(unexpected)}]"
            )

        shape_mismatches = []
        for key, tensor in expected_state_dict.items():
            expected_key = f"{key_prefix}{key}"
            expected_size = getattr(tensor, "size", None)
            metadata_size = getattr(checkpoint_meta[expected_key], "size", None)
            if expected_size is None or metadata_size is None:
                continue
            expected_shape = torch.Size(expected_size())
            checkpoint_shape = torch.Size(metadata_size)
            if checkpoint_shape != expected_shape:
                shape_mismatches.append((expected_key, checkpoint_shape, expected_shape))
        if shape_mismatches:
            sample = "; ".join(
                f"{key}: checkpoint={checkpoint_shape} model={expected_shape}"
                for key, checkpoint_shape, expected_shape in shape_mismatches[:20]
            )
            if len(shape_mismatches) > 20:
                sample += f"; ... (+{len(shape_mismatches) - 20} more)"
            raise ValueError(
                f"Strict DCP shape check failed for {path}: "
                f"mismatches={len(shape_mismatches)} [{sample}]"
            )
        _log(
            f"DCP metadata strict match: {path} | keys={len(expected_keys)} prefix={key_prefix!r}",
            "INFO",
        )

    def _try_load_safetensors(path):
        safetensors_files = glob.glob(f"{path}/*.safetensors")
        if not safetensors_files:
            raise ValueError("no safetensors")
        state_dict = {}
        for stf in safetensors_files:
            state_dict |= safetensors.torch.load_file(stf)
        missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
        _raise_on_incompatible_keys(missing, unexpected, path)
        _log(
            f"Transformer loaded from safe tensors: {path} | strict_keys=ok",
            "INFO",
        )

    def _try_load_plain(path):
        # mmap: the file is paged in on demand and the page cache is shared by
        # every worker on the node, instead of each holding its own 100+ GB copy.
        state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported plain checkpoint format in {path}")

        keys = list(state_dict.keys())
        if keys and all(k.startswith("module.") for k in keys):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
        _raise_on_incompatible_keys(missing, unexpected, path)
        _log(
            f"Transformer loaded from plain checkpoint: {path} | strict_keys=ok",
            "INFO",
        )

    try:
        _try_load_safetensors(weights_dir)
        return
    except Exception as e:
        _log(f"safetensors load skipped/failed: {e}", "WARNING")

    plain_errors = []
    for candidate in plain_candidates:
        if os.path.isfile(candidate):
            try:
                _try_load_plain(candidate)
                return
            except Exception as e:
                plain_errors.append((candidate, e))
                _log(f"State-dict file {candidate} could not be loaded: {e}", "WARNING")

    dcp_errors = []
    for candidate in dcp_candidates:
        if not (os.path.isdir(candidate) and _has_dcp_metadata(candidate)):
            continue
        # Transformer-only DCP dirs hold bare keys; full training checkpoints nest them under "model".
        for prefix in ("", "model."):
            try:
                target = get_model_state_dict(pipe.transformer)
                _validate_dcp_metadata(candidate, target, key_prefix=prefix)
                dcp.load(state_dict={"model": target} if prefix else target, checkpoint_id=candidate)
                _log(f"Transformer weights restored from DCP {candidate} (keys and shapes match)", "INFO")
                return
            except BaseException as e:
                dcp_errors.append((candidate, e))
                _log(f"DCP load from {candidate} (prefix {prefix!r}) failed: {e}", "WARNING")

    details = []
    if plain_errors:
        details.append(
            "Plain checkpoint load attempts failed: "
            + "; ".join(f"{path}: {err}" for path, err in plain_errors)
        )
    if dcp_errors:
        details.append(
            "DCP checkpoint load attempts failed: "
            + "; ".join(f"{path}: {err}" for path, err in dcp_errors)
        )
    if dcp_candidates and not dcp_errors:
        missing_dcp_metadata = [path for path in dcp_candidates if os.path.isdir(path) and not _has_dcp_metadata(path)]
        if missing_dcp_metadata:
            details.append(
                "Skipped DCP load because checkpoint metadata was missing in: "
                + ", ".join(missing_dcp_metadata)
            )

    raise ValueError(
        f"Could not load checkpoint from {checkpoint_path}. "
        "Expected either a plain model file (model_state.pt/pytorch_model.bin), a *.safetensors "
        "directory, or a DCP transformer directory. "
        + " ".join(details)
    )


def prepare_generation_inputs(args, task):
    """Resolve/skip prompt work before model construction, retaining legacy inputs."""
    if args.image_root is not None and not args.prompt_json:
        raise ValueError("--image_root requires --prompt_json; direct inputs should use --image_path")
    if task == "auto" and not args.prompt_json:
        raise ValueError("--task auto requires --prompt_json")
    if args.prompt_json:
        if task not in {"t2v", "i2v", "auto"} or args.output_path or not args.output_dir or args.image_path:
            raise ValueError("--prompt_json requires T2V/I2V and --output_dir; put reference images in JSON, not --image_path")
        if (type(args.prompt_shard_count) is not int or type(args.prompt_shard_index) is not int
                or not 0 <= args.prompt_shard_index < args.prompt_shard_count):
            raise ValueError("Require 0 <= prompt_shard_index < prompt_shard_count")
        records = load_inference_inputs(args.prompt_json, args.output_dir, task=task, fps=args.fps,
                                 fallback=args.video_length,
                                 shard_index=0 if task == "auto" else args.prompt_shard_index,
                                 shard_count=1 if task == "auto" else args.prompt_shard_count,
                                 default_negative=args.negative_prompt, image_root=args.image_root)
        if task == "auto":
            # Validate the entire mixed collection before sharding/resume, so
            # one T2V-only worker cannot start an invalid mixed STG run.
            if args.stg_scale != 0 and any(record["task"] == "i2v" for record in records):
                raise ValueError("STG supports T2V only; auto collection contains I2V entries")
            records = records[args.prompt_shard_index::args.prompt_shard_count]
    else:
        if args.output_dir or (args.prompt_shard_index, args.prompt_shard_count) != (0, 1):
            raise ValueError("--output_dir and prompt sharding require --prompt_json")
        prompts = args.prompt if isinstance(args.prompt, list) else [args.prompt]
        refs = args.image_path if task == "i2v" else None
        if task == "i2v" and not refs:
            raise ValueError("I2V requires --image_path (or per-entry images with --prompt_json)")
        if refs is not None and len(refs) != len(prompts):
            raise ValueError("Reference input count must match --prompt count")
        records = []
        for index, prompt in enumerate(prompts):
            output = args.output_path
            if output and len(prompts) > 1:
                base, ext = os.path.splitext(output)
                output = f"{base}_{index}{ext}"
            record = dict(prompt=prompt, negative_prompt=args.negative_prompt,
                          video_length=args.video_length, output_path=output)
            if refs is not None:
                record["image_path"] = refs[index]
            records.append(record)
    tasks = []
    for record in records:
        effective = copy.copy(args)
        vars(effective).update(record)
        effective.task = record.get("task", task)
        effective.i2v_skip_first_latent_cfg = effective.task == "i2v" and args.i2v_skip_first_latent_cfg
        if args.skip_existing and effective.output_path and is_complete(effective):
            _log(f"Skipping complete output: {effective.output_path}", "INFO")
        else:
            tasks.append(effective)
    return tasks


def _choose_offload(args, overrides):
    """``--offload auto``: stream DiT blocks from the CPU only when its bf16 weights do not fit the GPU."""
    if args.offload != "auto":
        return args.offload
    dit_bytes = PanVideoPipeline.transformer_nbytes(overrides, dtype=torch.bfloat16)
    return "none" if PanVideoPipeline.dit_fits_on_gpu(dit_bytes) else "blocks"


def generate_video(args):
    overrides = parse_json_object(args.transformer_config_overrides, "transformer_config_overrides")
    task = "i2v" if args.image_path else (args.task or "t2v")
    if args.task is not None and task != args.task:
        raise ValueError("--task conflicts with the supplied reference image")

    tasks = prepare_generation_inputs(args, task)
    if not tasks:
        _log("No pending prompts in this shard; checkpoint loading skipped.")
        return

    dit_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    offload = _choose_offload(args, overrides)
    # With offloading the models are built on the CPU and moved to the GPU while they run.
    device = torch.device("cpu" if offload != "none" else "cuda")
    pipe = PanVideoPipeline.create_pipeline(
        transformer_config_overrides=overrides,
        transformer_dtype=dit_dtype,
        device=device,
        wan_vae_checkpoint=args.wan_vae_checkpoint,
        text_encoder_hidden_state_skip_layer=args.text_encoder_hidden_state_skip_layer,
        text_encoder_max_length=args.text_encoder_max_length,
    )
    loguru.logger.info(f"offload={offload} prefetch={args.offload_prefetch}")
    pipe.configure_runtime(compile_blocks=args.compile, offload=offload, prefetch=args.offload_prefetch)
    if args.checkpoint_path:
        load_transformer_weights(pipe, args.checkpoint_path)
    pipe.transformer.eval()  # one load per worker; every prompt of the shard reuses it

    defaults = {name: getattr(pipe.config, name) for name in ("num_inference_steps", "flow_shift", "guidance_scale")}
    _log(f"Inference settings: task={task}, frames={args.video_length}, fps={args.fps}, "
         + ", ".join(f"{name}={getattr(args, name) if getattr(args, name) is not None else value}"
                     for name, value in defaults.items()))

    for index, job in enumerate(tasks, start=1):
        for name, value in defaults.items():
            if getattr(job, name) is None:
                setattr(job, name, value)
        _log(f"Generating {index}/{len(tasks)}, steps={job.num_inference_steps}, shift={job.flow_shift}, "
             f"CFG={job.guidance_scale}, frames={job.video_length}: {job.prompt[:80]}...")
        out = pipe(
            prompt=job.prompt,
            video_height=args.video_height,
            video_width=args.video_width,
            num_inference_steps=job.num_inference_steps,
            video_length=job.video_length,
            negative_prompt=job.negative_prompt,
            guidance_scale=job.guidance_scale,
            i2v_skip_first_latent_cfg=job.i2v_skip_first_latent_cfg,
            cfg_skip_interval=job.cfg_skip_interval,
            cfg_warmup_scale=job.cfg_warmup_scale,
            timestep_schedule=getattr(job, "timestep_schedule", "standard"),
            first_step_velocity_scale=getattr(job, "first_step_velocity_scale", 1.0),
            flow_shift=job.flow_shift,
            **{key: getattr(job, key) for key in GUIDANCE_KEYS},
            seed=args.seed,
            # The reference image belongs to this record only.
            reference_image=job.image_path if job.task == "i2v" else None,
        )
        if dist.is_initialized() and dist.get_rank() != 0:
            continue
        output_path = job.output_path or os.path.join(
            ".", "outputs", f"output_{MODEL_LABEL}_{datetime.datetime.now():%Y-%m-%d %H:%M:%S}.mp4")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        write_video(out.videos, output_path, fps=args.fps, video_crf=args.video_crf,
                    video_preset=args.video_preset, video_pixel_format=args.video_pixel_format)
        print(f"Video written to {output_path}")
        if args.save_settings:
            job.output_path = output_path
            save_settings(job, output_path, job.task)
        _log(f"Completed {index}/{len(tasks)}: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate videos with a PAN 2 checkpoint")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt", type=str, nargs="+", help="Text prompt(s) for video generation")
    inputs.add_argument("--prompt_json", help="T2V/I2V prompt-entry JSON; preserves per-entry text, reference images and durations")
    parser.add_argument('--task', choices=['t2v', 'i2v', 'auto'], default=None,
                        help='JSON task filter (default: t2v); auto runs all T2V/I2V entries with one shared model and task-namespaced outputs; direct inputs infer I2V from --image_path')
    parser.add_argument('--output_dir', help='Output root for --prompt_json relative paths')
    parser.add_argument('--image_root', help='Optional JSON I2V reference directory: resolve each image by its original basename without editing the input JSON')
    parser.add_argument('--prompt_shard_index', type=int, default=0)
    parser.add_argument('--prompt_shard_count', type=int, default=1)
    parser.add_argument('--skip_existing', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--timestep_schedule', choices=['standard','early_refine_2x'], default='standard',
                        help='early_refine_2x splits the first two shifted sigma intervals and coarsens two middle intervals; exactly 50 updates.')
    parser.add_argument('--first_step_velocity_scale', type=float, default=1.0,
                        help='Scale the complete velocity at index 0 only, in [0,1]')
    parser.add_argument('--cfg_start_step', type=int, default=0)
    parser.add_argument('--cfg_end_step', type=int, default=None)
    parser.add_argument('--cfg_warmup_scale', type=float, default=1.0,
                        help='Constant CFG before cfg_start_step (>=1); default 1 preserves positive-only warmup')
    parser.add_argument('--cfg_skip_interval', type=int, nargs=2, metavar=('START', 'END'), default=None,
                        help='Optional zero-based, end-exclusive positive-only interval inside the CFG window; 2 15 skips displayed steps 3-15')
    parser.add_argument('--i2v_skip_first_latent_cfg', type=str_to_bool, nargs='?', const=True, default=True,
                        help='I2V only: use the positive prediction without CFG on temporal latent 0 (default: true)')
    parser.add_argument('--stg_scale', type=float, default=0.0)
    parser.add_argument('--stg_skip_layers', type=int, nargs='+', default=None)
    parser.add_argument('--video_crf', type=int, default=None, help='Optional libx264 CRF; omitted preserves legacy encoding')
    parser.add_argument('--video_preset', choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow', 'placebo'], default=None)
    parser.add_argument('--video_pixel_format', choices=['yuv420p', 'yuv444p'], default=None)
    add = parser.add_argument
    toggle = dict(type=str_to_bool, nargs="?", const=True)  # "--flag", "--flag true", "--flag 0", ...
    add("--negative_prompt", default="", type=str,
        help="What the video should avoid; empty by default. Used for --prompt_json entries "
             "that have no negative_prompt.")
    add("--negative_prompt_file", default=None, type=str,
        help="Text file whose full contents become --negative_prompt.")
    add("--guidance_scale", default=None, type=float, help="CFG weight; the pipeline default when omitted.")
    add("--flow_shift", default=None, type=float, help="Sigma time shift of the sampler; the pipeline default when omitted.")
    add("--video_height", default=704, type=int, help="Frame height in pixels (704).")
    add("--video_width", default=1248, type=int, help="Frame width in pixels (1248).")
    add("--num_inference_steps", default=None, type=int, help="Denoising steps; the pipeline default (50) when omitted.")
    add("--video_length", default=241, type=int, help="Frame count for prompts that do not specify a duration (241).")
    add("--fps", default=24.0, type=float, help="Frame rate of the written video; the training data is 24 fps.")
    add("--offload", default="auto", choices=("auto",) + OFFLOAD_MODES,
        help="none: everything on the GPU; models: text encoder, DiT and VAE wait on the CPU between uses; "
             "blocks: also stream DiT blocks to the GPU while it runs; auto (default): blocks only when the "
             "bf16 DiT does not fit in free GPU memory, else none.")
    add("--offload_prefetch", default=True, **toggle,
        help="With --offload blocks: prefetch the next block on a side stream (faster, much more host memory; on).")
    add("--dtype", default="bf16", choices=["bf16", "fp32"], help="Transformer precision: bf16 (default) or fp32.")
    add("--seed", default=42, type=int, help="Seed of the CPU noise generator (42); -1 draws a random one.")
    add("--image_path", default=None, type=str, nargs="+", help="Reference image(s); giving one switches to image-to-video.")
    add("--output_path", default=None, type=str, help="Where to write the video; a timestamped file under ./outputs/ when omitted.")
    add("--compile", default=False, **toggle, help="torch.compile each transformer block (off).")
    add("--save_settings", default=True, **toggle,
        help="Write <video>_config.json with the arguments used (on); --skip_existing reads it.")
    add("--checkpoint_path", default=None, type=str,
        help="Transformer weights: a training checkpoint-N/transformer DCP dir, a checkpoint-N root, "
             "a state-dict file such as model_state.pt, or a folder of *.safetensors.")
    add("--training_config", default=None, type=str,
        help="Training YAML (e.g. config/sft_29B_720p.yaml); its model section and "
             "data.packed_caption_segment_limit replace the architecture flags below.")
    add("--transformer_config_overrides", default=None, type=str,
        help="Architecture as a JSON object or JSON file, e.g. '{\"num_layers\": 44}' (prefer a file in shell scripts).")
    add("--text_encoder_hidden_state_skip_layer", default=2, type=int,
        help="Final Qwen layers to drop: 0 keeps the last hidden state, 2 matches the shipped training configs.")
    add("--text_encoder_max_length", default=None, type=int,
        help="Caption tokens kept after the template prefix; the training YAML's packed_caption_segment_limit when given.")
    add("--wan_vae_checkpoint", default=None, type=str,
        help="Local diffusers root or Hub repo of the Wan 2.2 VAE (default: Wan-AI/Wan2.2-TI2V-5B-Diffusers).")
    return parser


def _check_args(args, argv):
    validate_sampling_extensions(args.num_inference_steps or 50, args.timestep_schedule)
    if not 0 <= args.first_step_velocity_scale <= 1:
        raise ValueError("--first_step_velocity_scale must be finite and in [0,1]")
    if not 1.0 <= args.cfg_warmup_scale < float("inf"):
        raise ValueError("--cfg_warmup_scale must be finite and at least 1")
    if args.stg_scale == 0 and not any(t == "--stg_skip_layers" or t.startswith("--stg_skip_layers=") for t in argv):
        args.stg_skip_layers = None
    if args.skip_existing and (not args.save_settings or any(
            getattr(args, key) is None for key in ("num_inference_steps", "flow_shift", "guidance_scale"))):
        raise ValueError("--skip_existing requires saved settings and explicit steps/shift/CFG")
    if args.skip_existing and not (args.output_path or args.output_dir):
        raise ValueError("--skip_existing requires a stable --output_path or --output_dir")
    if args.video_crf is not None and not 0 <= args.video_crf <= 51:
        raise ValueError("--video_crf must be within [0, 51]")
    if args.negative_prompt_file:
        if args.negative_prompt:
            raise ValueError("--negative_prompt and --negative_prompt_file are mutually exclusive")
        with open(args.negative_prompt_file, "r", encoding="utf-8") as f:
            args.negative_prompt = f.read().strip()
    read_training_config(args)
    if args.image_root is not None and not args.prompt_json:
        raise ValueError("--image_root requires --prompt_json; direct inputs should use --image_path")
    if args.task == "auto" and not args.prompt_json:
        raise ValueError("--task auto requires --prompt_json")
    # A literal "none" from a templated launcher means "no reference image".
    if isinstance(args.image_path, str) and args.image_path.strip().lower() == "none":
        args.image_path = None
    if args.video_height <= 0 or args.video_width <= 0:
        raise ValueError("--video_height and --video_width must be positive integers")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_arg_parser().parse_args(argv)
    _check_args(args, argv)
    # One process holds the whole model; prompts are parallelized by launching independent
    # single-GPU workers with --prompt_shard_index / --prompt_shard_count.
    init_parallel_layout(cp=1)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    try:
        generate_video(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

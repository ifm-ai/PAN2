# Inference

[← back to README](../README.md) · prev: [Training](training.md) · next: [Compatibility](compatibility.md)

## 1. Run it

```bash
bash inference/inference.sh <CKPT_PATH> <OUTPUT_DIR> [PROMPTS_JSON]    # on a GPU node; sbatch works too
```

`inference/inference.sh` starts one `generate.py` worker per GPU on the node.
Every worker holds the full 29B model and takes every N-th prompt
(`--prompt_shard_index / --prompt_shard_count`), so throughput scales with the
GPU count and no context parallelism is involved. `CKPT_PATH` is a training
`checkpoint-N/transformer` directory (DCP), a `model_state.pt`, or a directory
of `*.safetensors`. `PROMPTS_JSON` defaults to `inference/prompts/examples.json`,
one t2v and one i2v record with structured negatives and a reference image.

The architecture is read from the training YAML (`TRAINING_CONFIG`, default
`config/sft_29B_720p.yaml`), so no model root is needed: the Wan 2.2 VAE and
the Qwen3.5 text encoder come from the Hugging Face cache (`HF_HOME`), the
transformer weights from `CKPT_PATH`. Everything else is an environment
variable with a default in the script:

| variable | default | meaning |
| --- | --- | --- |
| `NUM_STEPS`, `FLOW_SHIFT`, `GUIDANCE_SCALE` | 50, 9, 2.5 | sampler (section 2) |
| `TASK` | `auto` | `t2v`, `i2v`, or both from one JSON |
| `VIDEO_HEIGHT` x `VIDEO_WIDTH`, `FPS` | 704 x 1248, 24 | output size and frame rate |
| `VIDEO_LENGTH` | 241 | frames for entries without a duration |
| `SEED` | 123 | |
| `NUM_GPUS` | all visible GPUs | workers on this node |
| `NEGATIVE_PROMPT_FILE` | `inference/prompts/negative_prompt.json` | fallback negative for entries without one; its time ranges are rendered to each clip's duration |
| `EXTRA_ARGS` | | appended to the `generate.py` command |

Outputs land in `<OUTPUT_DIR>/<t2v|i2v>/<relative_path>.mp4`, each
next to a `*_config.json` with the effective arguments. A rerun skips a video
whose config matches and fails closed when it differs, so use a new output
directory for a new checkpoint, negative prompt or sampler.

## 2. Default sampler

The launcher passes these values to `generate.py`:

| control | value |
| --- | --- |
| steps / flow shift | 50 / 9, standard timestep schedule |
| CFG | 2.5 on every update (indices 0-49) |
| skip interval, STG | off |
| I2V | `--i2v_skip_first_latent_cfg true`: temporal latent 0 uses the positive prediction |
| precision | bf16 transformer; storage precision of the checkpoint is independent |

Indices are zero-based and end-exclusive.

## 3. Guidance controls

`generate.py` exposes the pipeline's sampler controls as flags. Append any of
them through `EXTRA_ARGS` when using the launcher.

| flag | effect |
| --- | --- |
| `--cfg_start_step`, `--cfg_end_step`, `--cfg_warmup_scale` | full-scale CFG window; before it a constant warm-up scale (1 = positive-only), after it positive-only |
| `--cfg_skip_interval a b` | positive-only inside `[a, b)` even within the window |
| `--stg_scale s --stg_skip_layers i ...` | skip-layer guidance: one extra positive pass with whole joint blocks skipped; t2v only, no offloading hooks |
| `--i2v_skip_first_latent_cfg` | default true; `false` restores CFG on the reference latent |
| `--first_step_velocity_scale` | scale the whole first update, in [0, 1] |
| `--timestep_schedule early_refine_2x` | halve the first two sigma intervals of a 50-step grid, drop two middle boundaries |

The math lives in `pan/utils/inference_guidance.py` (plan, CFG, STG) and
`pan/schedulers/timestep_schedule.py`; the pipeline
keeps the exact legacy bf16 arithmetic whenever no control is requested.

## 4. Prompt JSON

`--prompt_json` is a list of records (or `{"entries": [...]}`):

| field | notes |
| --- | --- |
| `refined_prompt` / `refined_prompts` / `prompt` / `caption` | positive text, the first nonempty one in this order, so a rewrite wins over the raw prompt; structured captions in the training style work best |
| `negative_prompt` | optional; falls back to `--negative_prompt` / `--negative_prompt_file`, whose `time_range_sec` fields are rendered to the clip's duration |
| `type` or `task` | `t2v` / `i2v`; inferred from `image_path` when absent |
| `image_path` (`image`, `ref_image`) | i2v reference; `--image_root DIR` remaps by basename |
| `duration_sec`, `frames`, `num_target_frames` | frames = `4 * ceil(duration_sec * fps / 4) + 1`; else `--video_length` |
| `id` or `relative_path` | output name: `<task>_<id>.mp4`, or the given relative path |

Direct use without JSON: `--prompt "..." [--image_path ref.png] --output_path out.mp4`.
The output size is always explicit (`--video_height` / `--video_width`, default 704 x 1248).
`inference/prompts/examples.json` shows one t2v and one i2v record in this format.

## 5. Prompt rewriting

Training captions are structured (`video_style`, `lighting`,
`perspective_view`, `subjects_lifecycle`, timed events). Free-text prompts
work best after being rewritten into that style. `inference/prompts_rewritter.py`
does this offline with a local instruction-tuned model (`--model-path`,
by default `Qwen/Qwen3.5-9B`, the same model PAN 2 uses as its text encoder)
and writes a JSON prompt file the launcher accepts. There is no in-process
rewriting; `generate.py` passes prompts to the text encoder verbatim.

The same rewriting policy is also packaged as an agent skill in
[`inference/pan-prompt-rewriter/`](../inference/pan-prompt-rewriter/SKILL.md)
for Codex, Claude Code or any assistant that can follow a `SKILL.md`. The
skill authors the positive caption from `system_prompt.md`, attaches the fixed
negative prompt from `assets/generic_negative.txt` (only the duration is
substituted), and `scripts/prompt_tools.py` assembles and validates the
records, including the text-encoder token count when pointed at the Qwen3.5
tokenizer with `--tokenizer`. To install it, copy or symlink the directory to
`~/.codex/skills/pan-prompt-rewriter` or `.claude/skills/pan-prompt-rewriter`.

## 6. Attention backend

The pinned environment ships FlashAttention-4 only. The non-packed inference
path runs `attention_backend: flash` on FA4 (the only supported backend), for both
the caption encoder and the joint video-text attention; the startup banner prints
`attention  FlashAttention-4 (attention_backend=flash)`.

## 6b. Memory

`--offload` picks where the models live: `none` (everything on the GPU, what the
launcher uses), `models` (text encoder, DiT and VAE wait on the CPU between uses),
`blocks` (additionally stream the DiT's blocks to the GPU while it runs;
`--offload_prefetch` overlaps the copies at the cost of host memory) or `auto` (the
default: `blocks` when the bf16 DiT does not fit in free GPU memory, else `none`).

## 7. Converting checkpoints

Released weights (`IFM/PAN2` on Hugging Face) use the diffusers layout of
`PAN2Transformer3DModel`; this repository's checkpoints use the native layout of
`PanVideoDiffusionTransformer`. `scripts/convert_checkpoint.py` converts between
the two:

```bash
# diffusers transformer/ folder -> <DIR>/transformer/model.safetensors
python scripts/convert_checkpoint.py to-native \
    --input /path/to/PAN2/transformer --output <DIR>/transformer
bash inference/inference.sh <DIR>/transformer <OUTPUT_DIR>

# checkpoint-N/transformer (DCP), model_state.pt or safetensors -> diffusers transformer/ folder
python scripts/convert_checkpoint.py to-diffusers \
    --input /path/to/checkpoint-N/transformer --output /path/to/release/transformer \
    --training_config config/sft_29B_720p.yaml --dtype bf16
```

`generate.py` reads a safetensors checkpoint from a folder named `transformer`
(or a folder containing one), so keep that name for `--output` of `to-native`.
`to-diffusers` takes the architecture from `--training_config` and writes
5 GB shards, the shard index and `config.json`; `--dtype` casts the weights
(default: keep the stored dtype). The same native output also works as
`model.pretrained_transformer_path` for a training warm start, together with
`model.load_transformer_weights: true`.

The layouts hold the same tensors, so both directions are exact. The diffusers
layout splits the fused QKV, output and q/k-norm projections per modality,
stores the caption-encoder SwiGLU as `[w3; w1]` and the output-head modulation
as `[scale; shift]`, keeps only row 0 of the text type table (the only row the
model reads) and drops the timestep frequency table (a constant rebuilt on
load); `to-native` restores both.

# Training

[← back to README](../README.md) · prev: [Data](data.md) · next: [Inference](inference.md)

- [1. Launch](#1-launch)
- [2. The YAML](#2-the-yaml)
- [3. Pretrain vs SFT](#3-pretrain-vs-sft)
- [4. The launch scripts](#4-the-launch-scripts)
- [5. Checkpoints and resume](#5-checkpoints-and-resume)
- [6. Context parallelism (`cp_size`)](#6-context-parallelism-cp_size)

## 1. Launch

```bash
sbatch scripts/pretrain.sh      # pretraining
sbatch scripts/sft.sh           # SFT, warm-started from a pretrain checkpoint

# Another YAML, plus train.py overrides on top of it:
sbatch scripts/sft.sh config/my_run.yaml --output_dir /path/to/run --learning_rate 5e-5
```

Submit from the repository root, or set `PAN_ROOT` to it.

Each script sets up Slurm, activates the conda env, exports caches and
networking, resolves the master address and runs

```
torchrun ... train.py --config config/<run>.yaml
```

on every node. **Nearly everything about a run is in the YAML**, not the
script. The script only owns the cluster side.

## 2. The YAML

Keys are train.py argument names grouped into sections. The section names are
free-form, and any key can be overridden from the command line
(`--learning_rate 1e-5` beats the file). The groups and what you change in each:

**`model`** — how the transformer is built and initialized.

- `transformer_config_overrides`: the complete architecture, every constructor
  argument of `PanVideoDiffusionTransformer` spelled out (`hidden_size`,
  `num_attention_heads`, `num_layers`, `mlp_ratio`, `rope_axes_dim`, ...). Nothing is
  read from a pretrained-model directory. Configs written with the older key names
  (`heads_num`, `mm_single_blocks_depth`, `rope_dim_list`, ...) still load; they are
  translated with a warning. `zero_init_modulation` is a top-level field and must
  not be repeated inside the overrides.
- `pretrained_transformer_path` + `load_transformer_weights`: the warm start,
  see [§ 3](#3-pretrain-vs-sft).
- `text_encoder_hidden_state_skip_layer`: how many final Qwen layers to drop
  (2). Inference reads the same value; do not change it between the two.

**`training`** — optimizer and schedule: `learning_rate`, `weight_decay`,
`max_steps`, `warmup_steps`, `lr_scheduler_type`, `lr_decay_steps`,
`lr_decay_floor_ratio`, `max_grad_norm`, `gradient_accumulation_steps`,
`use_muon` (Muon by default, AdamW for embeddings, norms and modulation),
`legacy_eager_mode`, `compile_transformer`, `seed`, `dtype`.

Keep `legacy_eager_mode: true` (the default). This preserves the validated
side-stream execution, tensor learning rates, capturable optimizers, disabled
autocast caching, and FSDP resharding. It does not capture or replay CUDA
graphs. Disabling this mode is unsupported and is rejected before training.
The validated mode requires `gradient_accumulation_steps: 1` and retains its
NCCL async-error-handling setting. Configure this execution mode only through
`legacy_eager_mode`; the former CUDA graph option has been removed.

**`diffusion`** — flow matching: `flow_snr_type` (`uniform`, `lognorm`, `mix`,
`mode`), `flow_lognorm_loc` / `flow_lognorm_scale`, `train_timestep_shift`,
`train_timestep_shift_mode` (`fixed` or `dynamic`).

**`data`** — paths and budgets, see [Data](data.md). Also
`packed_dataloader_num_workers` (loader threads) and
`packed_dataloader_prefetch_factor`.

**`tasks`** — `image_cond_prob`, `video_cond_prob`, `cfg_drop_prob`, `conditioning_schema`.

**`distributed`** — `hybrid_shard` (shard within a node,
replicate across), `enable_gradient_checkpointing`,
`use_dedicated_gradient_checkpointing`, `cp_size` ([§ 6](#6-context-parallelism-cp_size)).

**`output`** — `output_dir`, `save_interval`, `save_total_limit`,
`log_interval`, `report_to` (`none` or `wandb`).

## 3. Pretrain vs SFT

The two configs share the architecture, budgets and tasks and differ in three
places:

| | `config/pretrain_29B_720p.yaml` | `config/sft_29B_720p.yaml` |
| --- | --- | --- |
| model init | from scratch, or a plain state-dict warm start | **`pretrained_transformer_path`: the pretrain run's `checkpoint-N/` directory**, `load_transformer_weights: true` |
| data | pretraining CSVs, image mixing on | curated SFT CSVs, `image_sample_prob: 0.0` |
| `output_dir` | pretrain run dir | its own dir |

`pretrained_transformer_path` accepts

- a plain state dict: `.pt`, `.safetensors`, or a directory holding
  `model_state.pt`. Shape-compatible tensors are loaded before FSDP wrapping.
- a training checkpoint written by this trainer: a `checkpoint-N/` directory or
  its `transformer/` DCP directory. Transformer weights only, loaded after FSDP
  through the hybrid-shard reader. This is the SFT case.

Either way optimizer state, LR schedule, step and dataloader cursor start
fresh, so SFT gets its own warmup from step 0. The warm start is used only while
`output_dir` has no checkpoint of its own; a restarted SFT job resumes itself.
For a true from-scratch pretrain set `load_transformer_weights: false` and
`pretrained_transformer_path: null`; a path with the switch off is rejected at
startup rather than silently ignored.

## 4. The launch scripts

`scripts/pretrain.sh` and `scripts/sft.sh` are the same script apart from the
Slurm header, the job name and the default YAML. Both take an optional YAML
path as the first argument and pass everything after it to `train.py`. What
you set there:

- **Slurm header**: `--nodes`, `--exclude`, plus any `--partition` / `--qos` /
  `--account` your cluster requires. `pretrain.sh` is set to 1
  node for debugging, `sft.sh` to 240.
- **Conda env** (`CONDA_ENV`, default `pan`) and `HF_HOME`. Keep
  `HF_HUB_OFFLINE=1`; compute nodes have no network.
- **Kernel caches.** Autotuned kernels are cached on disk and reused across
  runs, so a new job starts warm when these point at persistent storage:
  ```bash
  export TRITON_CACHE_DIR="/tmp/triton_${USER}"
  export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${USER}"
  export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="${PAN2_CACHE_ROOT:-$HOME/.cache/pan2}/fa4_cute_dsl"
  export QUACK_CACHE_DIR="${PAN2_CACHE_ROOT:-$HOME/.cache/pan2}/quack"
  ```
  Triton and Inductor are node-local `/tmp`, so each node compiles once per
  boot. The FA4 and quack caches are the expensive ones (minutes per GEMM
  shape); point `PAN2_CACHE_ROOT` at shared storage you own so every node
  reuses them. A quack version bump
  invalidates the quack cache by design (the version is in the key), so the
  first run after an upgrade is always cold. **A cold first step takes a long
  time** (over 30 minutes at production sizes) while quack autotunes its
  GEMMs; the log says when this starts and prints each kernel as it is tuned,
  so a quiet stretch after "autotuning" is expected, not a hang.
- **Weights & Biases**: `WANDB_PROJECT`, `WANDB_RUN_ID`, and `WANDB_API_KEY`
  exported from your shell, together with `report_to: wandb` in the YAML. Do
  not commit the key into the script.
- `MASTER_PORT`, the EFA variables and the NIC-pinning block normally stay as
  they are.

## 5. Checkpoints and resume

Every `save_interval` steps the trainer writes `output_dir/checkpoint-N/`:
DCP `transformer/` and `optimizer/` shards, `training_state.pt` with step and
LR schedulers, and `dataloader/<dp_rank>.json` cursors. `save_total_limit`
prunes old ones.

A run **always resumes from the latest checkpoint in its own `output_dir`**,
falling back to the previous one if the latest is unreadable. There is no
explicit resume path; to continue, launch again with the same `output_dir`.
Dataloader cursors are per data-parallel rank, so resuming needs the same
`world_size / cp_size`. `resume_video_csv: false` keeps model and optimizer but
restarts the video stream from the top of the CSVs; the image-tar cursor is
kept.

## 6. Context parallelism (`cp_size`)

Ulysses context parallelism for the packed path, to fit a long packed sequence
on GPUs with less HBM. `cp_size N` splits the world as `dp x cp`; within a cp
group joint token `j` lives on cp rank `j % N`, and the attention all-to-alls
trade tokens for heads so each rank runs varlen attention over the full
sequence for `num_attention_heads / N` heads. The shard is strided so every rank holds a
share of every sample's video and caption tokens.

Only cp rank 0 of each group runs the dataloader and loads the text encoder;
the assembled batch is broadcast to the group and sharded per rank. Loss and
gradients are exact for any `dp x cp`. Constraint: `num_attention_heads`
must be divisible by `cp_size`.

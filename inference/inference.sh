#!/bin/bash
#SBATCH --job-name=pan2_inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=192
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#
# Single-node inference. Run on a GPU node, either directly or through Slurm:
#   bash  inference/inference.sh <CKPT_PATH> [OUTPUT_DIR] [PROMPTS_JSON]
#   sbatch inference/inference.sh <CKPT_PATH> [OUTPUT_DIR] [PROMPTS_JSON]
#
# OUTPUT_DIR defaults to outputs/<timestamp> under the repository.
#
# One independent generate.py worker per GPU, each holding the full 29B model;
# worker k of NUM_GPUS takes prompts k, k+NUM_GPUS, k+2*NUM_GPUS, ...
#
# The sampler: 50 steps, flow shift 9, CFG 2.5, 704x1248 at 24 fps, seed 123,
# bf16, I2V first-latent CFG exclusion on. Override any variable below with an
# environment variable, e.g. VIDEO_LENGTH=121 bash inference/inference.sh ...
#
# A finished video is skipped on rerun when its saved *_config.json matches;
# use a new OUTPUT_DIR for a different checkpoint, negative prompt or sampler.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-pan}"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}       # compute nodes usually have no network
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
# Find the repository: under sbatch $0 is Slurm's spooled copy, so also try the
# submitted script's path and the submit directory. PAN_ROOT overrides all three.
repo_root() {
    local cmd dir
    [[ -n ${PAN_ROOT:-} ]] && { echo "$PAN_ROOT"; return; }
    if [[ -n ${SLURM_JOB_ID:-} ]]; then
        cmd=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n1)
        [[ -n $cmd && $cmd != /* ]] && cmd="${SLURM_SUBMIT_DIR:-.}/$cmd"
    fi
    for dir in "$(dirname "$0")/.." "${cmd:+$(dirname "$cmd")/..}" "${SLURM_SUBMIT_DIR:-}"; do
        [[ -n $dir && -f $dir/generate.py ]] && { (cd "$dir" && pwd); return; }
    done
}
ROOT=$(repo_root)
if [[ -z $ROOT || ! -f $ROOT/generate.py ]]; then
    echo "Cannot find the PAN 2 repository (no generate.py); set PAN_ROOT=/path/to/PAN2" >&2
    exit 1
fi
cd "$ROOT"

CHECKPOINT_PATH=${1:?usage: inference.sh <CKPT_PATH> [OUTPUT_DIR] [PROMPTS_JSON]}
OUTPUT_DIR=${2:-${OUTPUT_DIR:-outputs/$(date +%Y%m%d_%H%M%S)}}
PROMPTS_JSON=${3:-${PROMPTS_JSON:-inference/prompts/examples.json}}

TRAINING_CONFIG=${TRAINING_CONFIG:-config/sft_29B_720p.yaml}   # architecture; must match the checkpoint
NUM_STEPS=${NUM_STEPS:-50}                    # default sampler: 50 Euler steps, flow shift 9, CFG 2.5
FLOW_SHIFT=${FLOW_SHIFT:-9}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-2.5}
TASK=${TASK:-auto}                            # auto: t2v and i2v entries from one JSON
VIDEO_HEIGHT=${VIDEO_HEIGHT:-704}
VIDEO_WIDTH=${VIDEO_WIDTH:-1248}
FPS=${FPS:-24}
VIDEO_LENGTH=${VIDEO_LENGTH:-241}             # fallback for entries without duration_sec / frames
SEED=${SEED:-123}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
# Fallback negative prompt for entries without one of their own (rendered to each clip's duration
# by generate.py).
NEGATIVE_PROMPT_FILE=${NEGATIVE_PROMPT_FILE:-inference/prompts/negative_prompt.json}
EXTRA_ARGS=${EXTRA_ARGS:-}

if (( NUM_GPUS < 1 )); then echo "No GPU found (set NUM_GPUS)" >&2; exit 1; fi
mkdir -p "$OUTPUT_DIR" logs
JOB=${SLURM_JOB_ID:-$$}

# Kernel caches (FA4 CuTe, Triton, quack, inductor) are per run so concurrent
# runs never race on one directory; point these at persistent storage to start warm.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/pan2_triton_${USER}_${JOB}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/pan2_inductor_${USER}_${JOB}}
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-/tmp/pan2_fa4_${USER}_${JOB}}
export QUACK_CACHE_DIR=${QUACK_CACHE_DIR:-/tmp/pan2_quack_${USER}_${JOB}}

pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nnodes=1 --nproc_per_node=1 generate.py \
        --task "$TASK" --prompt_json "$PROMPTS_JSON" --output_dir "$OUTPUT_DIR" \
        --prompt_shard_index "$gpu" --prompt_shard_count "$NUM_GPUS" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --training_config "$TRAINING_CONFIG" \
        --num_inference_steps "$NUM_STEPS" --flow_shift "$FLOW_SHIFT" --guidance_scale "$GUIDANCE_SCALE" \
        --negative_prompt_file "$NEGATIVE_PROMPT_FILE" \
        --video_height "$VIDEO_HEIGHT" --video_width "$VIDEO_WIDTH" --fps "$FPS" --video_length "$VIDEO_LENGTH" \
        --seed "$SEED" --dtype bf16 --i2v_skip_first_latent_cfg true \
        --offload none \
        --skip_existing true --save_settings true \
        --video_crf 8 --video_preset slow --video_pixel_format yuv444p \
        $EXTRA_ARGS \
        > "logs/inference_${JOB}_gpu${gpu}.log" 2>&1 &
    pids+=($!)
done
echo "Launched $NUM_GPUS workers (logs/inference_${JOB}_gpu*.log); outputs -> $OUTPUT_DIR"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status

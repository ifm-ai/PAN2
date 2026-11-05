#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --nodes=240
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=180
#SBATCH --gpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#
# Usage: sbatch scripts/sft.sh [CONFIG] [train.py overrides...]
#   CONFIG defaults to config/sft_29B_720p.yaml; anything after it is passed to train.py, e.g.
#   sbatch scripts/sft.sh config/sft_29B_720p.yaml --output_dir /path/to/run --learning_rate 5e-5
# CONDA_ENV picks the conda env (default: pan); PAN_ROOT the repository, when it is not
# the directory you submit from.

set -eo pipefail
CONFIG=config/sft_29B_720p.yaml
if [[ $# -gt 0 && $1 != -* ]]; then CONFIG=$1; shift; fi
TRAIN_ARGS=
if [[ $# -gt 0 ]]; then TRAIN_ARGS=$(printf ' %q' "$@"); fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-pan}"


export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Under sbatch $0 is Slurm's spooled copy, so look for train.py in PAN_ROOT, the submit
# directory and the current directory.
REPO_ROOT=
for dir in "${PAN_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
    [[ -n $dir && -f $dir/train.py ]] && { REPO_ROOT=$(cd "$dir" && pwd); break; }
done
if [[ -z $REPO_ROOT ]]; then
    echo "Cannot find the PAN 2 repository (no train.py); submit from it or set PAN_ROOT" >&2
    exit 1
fi
cd "$REPO_ROOT"
[[ -f $CONFIG ]] || { echo "Config not found: $CONFIG (relative paths are relative to $REPO_ROOT)" >&2; exit 1; }

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_HOST" hostname -I | awk '{print $1}')
if [[ -z "$MASTER_ADDR" ]]; then
    echo "Failed to resolve IPv4 address for MASTER_HOST=$MASTER_HOST" >&2
    exit 1
fi
export MASTER_PORT=29505



export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1





export TRITON_CACHE_DIR="/tmp/triton_${USER}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${USER}"



export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="${PAN2_CACHE_ROOT:-$HOME/.cache/pan2}/fa4_cute_dsl"
export QUACK_CACHE_ENABLED=1
export QUACK_CACHE_DIR="${PAN2_CACHE_ROOT:-$HOME/.cache/pan2}/quack"




# export TRITON_CACHE_DIR="$REPO_ROOT/triton"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="ATEN,CUTEDSL,NVGEMM"
export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS="ATEN,CUTEDSL,NVGEMM"
export TORCHINDUCTOR_NVGEMM_MAX_PROFILING_CONFIGS="10"
export TORCHINDUCTOR_DISTRIBUTED_MAX_AUTOTUNE_GEMM="1"
# export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/inductor"
# mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

srun --kill-on-bad-exit=1 --wait=60 bash -c "
# Pin Gloo/NCCL to the real inter-node NIC so peers do not advertise 127.0.1.1
NODE_IP=\$(hostname -I | awk '{print \$1}')
NODE_IFACE=\$(ip -o -4 addr show | awk -v ip=\"\$NODE_IP\" '\$4 ~ ip\"/\" {print \$2; exit}')
if [[ -z \"\$NODE_IFACE\" ]]; then
    echo \"Failed to detect NIC for IP \$NODE_IP on \$(hostname)\" >&2
    exit 1
fi
export GLOO_SOCKET_IFNAME=\"\$NODE_IFACE\"
export NCCL_SOCKET_IFNAME=\"\$NODE_IFACE\"
echo \"[\$(hostname)] Using NIC \$NODE_IFACE (\$NODE_IP) for Gloo/NCCL\"

ulimit -n 131072
cd $(printf %q "$REPO_ROOT")
\"\${NSYS_PREFIX[@]}\" torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config $(printf %q "$CONFIG")$TRAIN_ARGS
"
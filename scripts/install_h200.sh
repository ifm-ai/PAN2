#!/usr/bin/env bash
# Copyright (c) 2026 MBZUAI. SPDX-License-Identifier: Apache-2.0
# Install the reference CUDA dependencies in a separate Hopper environment.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-pan2-h200}"
PYTHON_VERSION="${PYTHON_VERSION:-3.14}"
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v conda >/dev/null || { echo 'conda must be available on PATH' >&2; exit 1; }
eval "$(conda shell.bash hook)"
if conda env list --json | python -c 'import json,pathlib,sys; sys.exit(not any(pathlib.Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]))' "$CONDA_ENV"; then
    echo "Environment already exists: $CONDA_ENV. Choose another CONDA_ENV to keep it intact." >&2
    exit 1
fi

conda create --name "$CONDA_ENV" "python=$PYTHON_VERSION" pip -y
conda activate "$CONDA_ENV"
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132
python -m pip install --pre 'flash-attn-4[cu13]==4.0.0b29' --no-build-isolation
python -m pip install 'quack-kernels[cu13]==0.6.4'
python -m pip install -r "$REPO_DIR/requirements.txt"
python -m pip check
python - <<'PY'
import importlib.metadata as metadata
import torch
import flash_attn.cute
import quack
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
for name in ('flash-attn-4', 'quack-kernels', 'transformers', 'diffusers'):
    print(name, metadata.version(name))
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
else:
    print('Installation imports passed; CUDA execution requires a GPU node.')
PY
echo "Next: conda activate $CONDA_ENV"
echo 'Training and inference: docs/environment_h200.md'

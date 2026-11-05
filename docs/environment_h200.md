# H200 setup

H200 uses the same model architecture, trainer and inference entry points as
main. The environment below uses the reference CUDA dependencies on Hopper
(compute capability 9.0).

## Installation

Run from the repository root with conda on PATH:

```bash
CONDA_ENV=pan2-h200 bash scripts/install_h200.sh
conda activate pan2-h200
```

The installer creates a separate environment and refuses to overwrite an
existing one. It installs Python 3.14, PyTorch 2.12.1 with CUDA 13.2,
FlashAttention 4.0.0b29 and quack-kernels 0.6.4, then the repository
requirements. A compatible NVIDIA driver is required; see the
[reference environment](environment.md) for prerequisites and model downloads.

## Training

Set dataset and checkpoint paths in `config/pretrain_29B_720p.yaml` or
`config/sft_29B_720p.yaml`. Keep `legacy_eager_mode: true`, as required by
the validated training path, and choose `cp_size` to divide the GPU count.
Adjust the Slurm resource directives for your cluster.

```bash
mkdir -p logs
CONDA_ENV=pan2-h200 sbatch scripts/pretrain.sh
CONDA_ENV=pan2-h200 sbatch scripts/sft.sh
```

See [Training](training.md) for packed data and checkpoint configuration.

## Inference

```bash
CONDA_ENV=pan2-h200 TRAINING_CONFIG=/path/to/training.yaml \
  bash inference/inference.sh /path/to/checkpoint/transformer /path/to/output /path/to/prompts.json
```

Use the checkpoint's training YAML to preserve its architecture and text
encoder settings. See [Inference](inference.md) for sampler and prompt options.

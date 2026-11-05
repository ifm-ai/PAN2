<!-- ============================================================
     HERO
     ============================================================ -->
<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/pan2-logo-dark.png">
  <img alt="PAN 2" src="assets/pan2-logo-light.png" width="420">
</picture>


<a href="https://huggingface.co/IFM/PAN2" target="_blank"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face"></a>
<a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/pan2" target="_blank"><img src="https://img.shields.io/badge/Diffusers-FF9D00?logo=huggingface&logoColor=white" alt="Diffusers"></a>
<a href="https://recipes.vllm.ai/IFM/PAN2" target="_blank"><img src="https://img.shields.io/badge/vLLM-30A2FF?logoColor=white" alt="vLLM"></a>
<a href="MODEL_CARD.md"><img src="https://img.shields.io/badge/Model%20card-ffd21e?logo=huggingface&logoColor=black" alt="Model card"></a>
<a href="#documentation"><img src="https://img.shields.io/badge/Docs-4f8cff?logo=readthedocs&logoColor=white" alt="Docs"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-Apache%202.0-4CAF50?logo=apache&logoColor=white" alt="LICENSE"></a>
<a href="docs/environment.md"><img src="https://img.shields.io/badge/PyTorch-2.12%20%7C%20CUDA%2013.2-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
<a href="docs/compatibility.md"><img src="https://img.shields.io/badge/FlashAttention-4-76b900" alt="FlashAttention 4"></a>

[**Quickstart**](#quickstart) · [**Find your path**](#find-your-path) · [**Examples**](#examples) · [**Model**](#model) · [**Docs**](#documentation)

</div>

# PAN 2

Developed by the Institute of Foundation Models (IFM) at MBZUAI.

---

## Table of contents

- [What is PAN 2?](#what-is-pan-2)
- [Find your path](#find-your-path)
- [Quickstart](#quickstart)
- [Examples](#examples)
- [Model](#model)
- [Repository map](#repository-map)
- [Documentation](#documentation)
- [Results](#results)
- [License and Attribution](#license-and-attribution)
- [Citation](#citation)

## What is PAN 2?

PAN 2 is a general-purpose world model: it simulates how the world will change in response to actions. Given a current observation (an optional image plus text) and actions described in text, it simulates the outcome as a video of up to 20 seconds. This supports *simulative reasoning*: an agent can imagine the outcomes of its candidate actions before choosing among them, by simulating each plan with PAN 2 and acting on the one whose simulated outcome is best.

PAN 2 follows the Generative Latent Prediction (GLP) architecture formalized in [Critique of World Models](https://arxiv.org/abs/2507.05169) and couples **conceptual reasoning** by a large language model with **perceptual reasoning** by a latent diffusion model. The language model, the **conceptual reasoner**, writes the next world state as a structured, time-grounded description of events, timed actions bound to actors, and camera motion. The 29B-parameter MMDiT, the **perceptual reasoning backbone**, predicts the video latents that realize that state, and the Wan 2.2 VAE decoder renders them as the next observation.

This repository holds the **training code** for the perceptual reasoning backbone: a token-budget packed dataloader over
precomputed latents, FSDP2 hybrid sharding, optional context
parallelism, and Muon. No
pixel decoding and no VAE forward happen during training; everything the loader
reads is produced offline and read from local disk. It also holds the **inference code** and the conceptual reasoner's prompt and agent skill. PAN 2 is also supported in [Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pan2) and [vLLM](https://recipes.vllm.ai/IFM/PAN2).

## Find your path

<table>
  <tr><th align="left" width="46%">I want to…</th><th align="left">Go to</th></tr>
  <tr><td><b>Set up the environment</b></td><td><a href="docs/environment.md">Environment setup</a> — conda, FA4, quack, model downloads</td></tr>
  <tr><td><b>Set up H200</b></td><td><a href="docs/environment_h200.md">H200 setup</a> — installation, training and inference</td></tr>
  <tr><td><b>Train on a small debug set</b></td><td><a href="#quickstart">Quickstart ↓</a></td></tr>
  <tr><td><b>Prepare my own data</b> — latents, CSVs, image tars</td><td><a href="docs/data.md">Data</a></td></tr>
  <tr><td><b>Pretrain or SFT at scale</b></td><td><a href="docs/training.md">Training</a> — YAML, launch scripts, checkpoints, resume</td></tr>
  <tr><td><b>Fit long sequences on smaller GPUs</b></td><td><a href="docs/training.md#6-context-parallelism-cp_size">Context parallelism</a></td></tr>
  <tr><td><b>Generate videos from a checkpoint</b></td><td><a href="docs/inference.md">Inference</a> — launcher, sampler controls, prompt JSON, rewriting</td></tr>
  <tr><td><b>Run the conceptual reasoner with a coding agent</b></td><td><a href="inference/pan-prompt-rewriter/SKILL.md">PAN 2 Conceptual Reasoner (Prompt Rewriter) skill</a> — Codex / Claude Code skill: caption policy, fixed negative, token validation</td></tr>
  <tr><td><b>See results, safety guardrails and limitations</b></td><td><a href="MODEL_CARD.md">Model card</a></td></tr>
  <tr><td><b>Upgrade torch / FA4 / quack</b></td><td><a href="docs/compatibility.md">Compatibility</a></td></tr>
</table>

## Quickstart

Train the 29B model on a one-node debug set. Full detail for each step is one
click away.

> **H200 GPUs:** see the [H200 setup guide](docs/environment_h200.md) for installation, training and inference.

**1. Environment** ([docs/environment.md](docs/environment.md))

```bash
conda create -n pan python=3.14 -y && conda activate pan
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132
pip install --pre "flash-attn-4[cu13]==4.0.0b29" --no-build-isolation
pip install "quack-kernels[cu13]==0.6.4"
pip install -r requirements.txt
```

**2. Models** (text encoder for training; the VAE for encoding data and inference)

```bash
export HF_HOME=YOUR_HF_PATH
hf download Qwen/Qwen3.5-9B
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

**3. Data.** Encode video clips with the Wan 2.2 VAE and list them in CSVs; the
formats are shown by `python scripts/make_sample_data.py`, which writes a random-latent
`sample_data/videos.csv` and `sample_data/images.tar`. About
8 CSVs with 100 video latents and 8 image tars serve one node; point the `data`
paths in the YAML at them. See [docs/data.md](docs/data.md).

```yaml
# config/pretrain_29B_720p.yaml
data:
  packed_dataloader_meta_csv_path: [/path/to/csvs/]      # one CSV per data-parallel rank
  image_tar_path: [/path/to/image_tars/]                 # optional
  image_sample_prob: 0.2
```

**4. Launch**

```bash
sbatch scripts/pretrain.sh      # config/pretrain_29B_720p.yaml
sbatch scripts/sft.sh           # warm-started from a pretrain checkpoint
```

Everything about a run lives in the YAML; the script only owns the cluster
side. A run always resumes from the latest checkpoint in its own `output_dir`.
See [docs/training.md](docs/training.md).

**5. Generate** (single node, one worker per GPU)

```bash
bash inference/inference.sh <CKPT_PATH> <OUTPUT_DIR> [PROMPTS_JSON]   # or sbatch, same arguments
```

`CKPT_PATH` is a training `checkpoint-N/transformer` directory, a
`model_state.pt`, or a directory of `*.safetensors`; the architecture comes
from `config/sft_29B_720p.yaml`, the VAE and text encoder from the Hugging
Face cache (step 2). The launcher starts one `generate.py` worker per GPU,
shards the prompts across them, and samples with the default recipe: 50 steps, flow shift 9, CFG 2.5, 704x1248 at 24 fps, bf16. `PROMPTS_JSON` defaults
to `inference/prompts/examples.json` (one t2v and one i2v record). Videos land
in `<OUTPUT_DIR>/{t2v,i2v}/` next to a `*_config.json`; a rerun into the same
directory skips finished clips. Override any default through environment
variables (`NUM_STEPS`, `FLOW_SHIFT`, `GUIDANCE_SCALE`, `VIDEO_LENGTH`,
`SEED`, `NUM_GPUS`, ...) or pass extra `generate.py` flags in `EXTRA_ARGS`.
See [docs/inference.md](docs/inference.md).

**6. Convert checkpoints** (diffusers layout <-> this repository)

The released weights (e.g. `IFM/PAN2`) are in the diffusers layout
(`PAN2Transformer3DModel`). `scripts/convert_checkpoint.py` converts them for
`generate.py` and training warm starts, and converts a training checkpoint back
to the diffusers layout for release:

```bash
# Released diffusers weights -> this repository; then pass <DIR>/transformer as CKPT_PATH.
hf download IFM/PAN2 --include "transformer/*" --local-dir /path/to/PAN2
python scripts/convert_checkpoint.py to-native \
    --input /path/to/PAN2/transformer --output <DIR>/transformer

# Training checkpoint -> diffusers transformer/ folder (architecture from the YAML).
python scripts/convert_checkpoint.py to-diffusers \
    --input /path/to/checkpoint-N/transformer --output /path/to/release/transformer \
    --training_config config/sft_29B_720p.yaml --dtype bf16
```

To warm start training from the converted weights, set both keys in the YAML:

```yaml
model:
  pretrained_transformer_path: <DIR>/transformer
  load_transformer_weights: true    # the pretrain YAML ships false
```

The conversion only renames, splits and concatenates tensors, so it is exact in
both directions. See [docs/inference.md](docs/inference.md#7-converting-checkpoints).

## Examples

### Prompts

Training captions are structured, and prompts work best in the same style.
`inference/prompts/examples.json` ships one t2v and one i2v record in that
format, each with its structured negative prompt, duration and (for i2v) a
reference image. Free-text prompts like these two also work:

> A muscular male swimmer in his early 30s, wearing a sleek professional swim
> cap and mirrored goggles, stands poised on a starting block. With an
> explosive burst of energy, he leans forward and dives headfirst into the
> crystalline blue water of an Olympic-sized swimming pool. The camera captures
> the moment of impact in a dynamic shot, showing a massive plume of white
> spray and chaotic bubbles erupting around him […]

> A wide, cinematic shot opens in a high-end, minimalist showroom characterized
> by polished white marble floors that mirror the overhead soft-box lighting
> and stark, charcoal-grey walls. In the center of the frame, a futuristic
> concept car is unveiled […] Luminous neon-blue accents trace the contours of
> the chassis and the thin, blade-like LED headlights, which pulse slowly with
> a breathing light effect […]

A conceptual reasoner expands them into the training caption style in two ways: offline with
`inference/prompts_rewritter.py` and a local instruction-tuned model, or
interactively through the [PAN 2 Conceptual Reasoner (Prompt Rewriter)](inference/pan-prompt-rewriter/SKILL.md)
agent skill, which carries the same caption policy and fixed negative prompt
and validates the result against the text encoder's token budget. See
[docs/inference.md](docs/inference.md#5-prompt-rewriting).

### Training data row

One CSV row per clip: the latent path and the caption.
`scripts/make_sample_data.py` writes ten of these with random latents.

```csv
"latent_path","caption"
"/data/latents/video_00.safetensors","{""video_style"": ""..."", ""lighting"": ""..."", ""perspective_view"": ""..."", ""events"": [...], ...}"
```

### Image-to-video

Set `image_cond_prob` in the `tasks:` section of the YAML (0.3 in the shipped
configs). Per row the loader draws i2v with that probability and conditions on
the first latent frame through channel concatenation, so one model serves both
tasks. At generation time any prompt-JSON entry with an `image_path` (or
`type: i2v`) is run as image-to-video by the same launcher; the shipped
`examples.json` has one of each.

## Model

| | |
| --- | --- |
| Parameters | 29.2B |
| Backbone | 44 dual-stream (video + state) joint-attention blocks, hidden size 4096, 32 heads, RMS qk-norm, 3-D RoPE |
| Latent space | Wan 2.2 VAE, 48 channels, 16x spatial / 4x temporal downsampling, patch `[1, 2, 2]` |
| Text encoder | Qwen3.5-9B hidden states (last 2 layers dropped), caption-encoder projection, up to 1,536 caption tokens |
| Resolution | 704x1248 at 24 fps: a 44x78 latent grid, 858 tokens per latent frame; up to 481 frames (20 s) |
| Objective | flow matching, fixed timestep shift 9; logit-normal timestep sampling in pretraining, uniform in SFT |
| Tasks | t2v, i2v, channel-concat conditioning: 48 noisy + 48 condition + 1 mask = 97 input channels |
| Training | three stages: 480p pretraining, 720p continued pretraining, 720p SFT |


This repository covers pretraining and SFT. Reward learning and the model soup
that produce the released weights are described in the technical report and are
not part of the training code.


| Config | Purpose |
| --- | --- |
| [`config/pretrain_29B_720p.yaml`](config/pretrain_29B_720p.yaml) | dense 720p pretraining / continued pretraining |
| [`config/sft_29B_720p.yaml`](config/sft_29B_720p.yaml) | SFT, warm-started from a pretrain checkpoint, image mixing off |

Results, safety guardrails and limitations: [MODEL_CARD.md](MODEL_CARD.md).

## Repository map

```text
pan2/
├── train.py            # entry point
├── config/             # training YAMLs
├── scripts/            # Slurm launchers: pretrain.sh, sft.sh; convert_checkpoint.py (diffusers <-> native weights);
│                       #   make_sample_data.py (random-latent data in the training formats)
├── dataset/            # packed dataloader: CSV stream, latent reads, image/video mixing, bin packing
├── pan/                # model, inference pipeline, text encoder, optimizer
│   ├── models/         # PanVideoDiffusionTransformer, Qwen3.5 text encoder
│   ├── optim/          # Muon, patched LR schedulers
│   ├── pipelines/      # inference pipeline
│   └── utils/          # Ulysses CP, guidance controls
├── generate.py         # inference entry point (one process per GPU, prompt sharding)
├── inference/          # single-node launcher, example prompts, prompt rewriter, pan-prompt-rewriter/ agent skill
├── docs/               # the guides linked below
├── assets/             # logo and demo media
└── MODEL_CARD.md       # overview, usage, results, safety guardrails, limitations
```

## Documentation

| Guide | Covers |
| --- | --- |
| [Environment setup](docs/environment.md) | conda env, CUDA stack install order, install check, model downloads |
| [Data](docs/data.md) | Wan 2.2 VAE encoding and normalization, CSV and image-tar formats, image/video mixing, token budgets, debug set |
| [Training](docs/training.md) | launching, the YAML sections, pretrain vs SFT, launch scripts and kernel caches, checkpoints and resume, context parallelism |
| [Inference](docs/inference.md) | launcher, default sampler, guidance controls, prompt JSON format, resume |
| [Compatibility](docs/compatibility.md) | version pins, FA4/quack upgrade contract, cu12 vs cu13 |
| [Model card](MODEL_CARD.md) | overview, usage, results, safety guardrails, limitations |


## Results

PAN 2 with Kimi K3 as the conceptual reasoner, from the technical report; benchmarks, human evaluation and the evaluation protocols are in the [model card](MODEL_CARD.md#headline-results).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/report/results_capabilities_dark.png">
    <img alt="Capability and multi-step plan results, PAN 2 highlighted" src="assets/report/results_capabilities_light.png" width="100%">
  </picture>
</p>

## License and Attribution

PAN 2 is developed by the Institute of Foundation Models (IFM) at MBZUAI.

PAN 2's own code and weights are released by MBZUAI under the [Apache License, Version 2.0](LICENSE).

PAN 2 includes or is used with third-party components, each under its own license. The license files are:

- [LICENSE](LICENSE): the license for PAN 2's own code and weights.
- [NOTICE](NOTICE): MBZUAI's copyright notice and attribution for the third-party components.
- [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES): the full license text of each third-party component, with its original copyright line.

### Modifications

The Wan 2.2 and Qwen (Qwen3.5-9B) files and weights are used as released by their authors and have not been modified or fine-tuned by MBZUAI. Third-party code adapted in this repository keeps its original copyright and license header and, where MBZUAI has changed the code, a notice describing the changes.

## Citation

```bibtex
@misc{pan2_2026,
  title        = {PAN 2},
  author       = {{PAN Team}},
  organization = {Institute of Foundation Models, Mohamed bin Zayed University of Artificial Intelligence},
  year         = {2026},
  month        = oct,
  howpublished = {\url{https://github.com/ifm-ai/PAN2/tree/main}},
  url          = {https://github.com/ifm-ai/PAN2/tree/main}
}
```

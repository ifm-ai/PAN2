# Environment setup

[← back to README](../README.md) · next: [Data](data.md)

The reference environment is the `pan` conda env: Python 3.14.4, PyTorch
2.12.1 built for CUDA 13.2, FlashAttention 4 and quack-kernels for the fused
attention and GEMM paths. `requirements.txt` pins everything to the versions
installed there.

## 1. Create the env

```bash
conda create -n pan python=3.14 -y
conda activate pan
```

## 2. Install the CUDA stack, in this order

The order matters: FA4 pulls in its own quack and cutlass-dsl versions, and
quack has to be installed after FA4 so the pin below wins.

```bash
# 1. PyTorch for CUDA 13.2
pip install torch==2.12.1 torchvision==0.27.1 \
    --index-url https://download.pytorch.org/whl/cu132

# 2. FlashAttention 4 (pre-release wheel), then quack-kernels
pip install --pre "flash-attn-4[cu13]==4.0.0b29" --no-build-isolation
pip install "quack-kernels[cu13]==0.6.4"

# 3. Everything else
pip install -r requirements.txt
```

`cu12` or `cu13`: with NVIDIA driver `>= 575` use the cu13 wheels above;
otherwise use `[cu12]` extras and the `cu129` PyTorch index. Run the three
steps together when upgrading, so a later step does not silently downgrade an
earlier one; add `--force-reinstall` if versions get tangled.

## 3. Check the install

```bash
python -c "import torch, importlib.metadata as m; print(torch.__version__, torch.version.cuda); \
  [print(p, m.version(p)) for p in ('flash-attn-4','quack-kernels','transformers','diffusers')]"
```

Expected: `2.12.1+cu132 13.2`, `flash-attn-4 4.0.0b29`, `quack-kernels 0.6.4`,
`transformers 5.13.0`, `diffusers 0.38.0`. `quack-kernels >= 0.6.3` is
enforced at import time by `pan/models/transformers/modules/mlp_ops.py`; an
older quack fails fast with upgrade instructions.

## 4. Download the models

```bash
export HF_HOME=YOUR_HF_PATH
hf download Qwen/Qwen3.5-9B                    # text encoder, ~19 GB; training and inference
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers    # VAE, ~2.7 GB; latent encoding and inference
```

Training loads only the Qwen checkpoint. The Wan VAE is needed when you encode
data ([Data § 1](data.md#1-encode-with-the-wan-22-vae)) and at inference; it is
never loaded during training.

## Upgrading FA4 and quack

See [Compatibility](compatibility.md) for the upgrade order and the two
contracts the kernels rely on.

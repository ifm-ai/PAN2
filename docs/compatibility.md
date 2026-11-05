# Compatibility

[← back to README](../README.md) · prev: [Inference](inference.md)

- FSDP2 + compile need `torch >= 2.11`. Training uses eager execution, without CUDA graph capture/replay.
- `quack-kernels >= 0.6.3` is checked at import in
  `pan/models/transformers/modules/mlp_ops.py`. To upgrade, FA4 first, then
  quack:
  ```bash
  pip install --pre "flash-attn-4[cu13]" --no-build-isolation -U
  pip install "quack-kernels[cu13]" -U
  ```
  Two contracts to keep in mind. varlen_m GEMMs take a per-sequence
  `(l, k, n)` B, and since 0.6.3 quack rejects a shared 2D B; our shared-weight
  sites are single-segment varlen_m GEMMs and pass `b[None]`.
  `quack_autotune_patch.py` mirrors quack's autotune cache-key format
  (verified on 0.6.3 and 0.6.4); if quack changes it, the patch disables itself
  and every step re-autotunes, costing minutes per GEMM.
- cu12 or cu13: NVIDIA driver `>= 575` takes cu13, otherwise cu12.

## Pinned versions

| package | version |
| --- | --- |
| Python | 3.14.4 |
| torch / torchvision | 2.12.1 / 0.27.1 (cu132) |
| flash-attn-4 | 4.0.0b29 |
| quack-kernels | 0.6.4 |
| transformers | 5.13.0 |
| diffusers | 0.38.0 |

The full list is in [`requirements.txt`](../requirements.txt).

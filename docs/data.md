# Data

[← back to README](../README.md) · prev: [Environment](environment.md) · next: [Training](training.md)

Three inputs, all local files:

| input | format | where it is set |
| --- | --- | --- |
| video latents | one `.safetensors` per clip, Wan 2.2 VAE output, normalized | named per row in the CSV |
| video CSVs | `latent_path`, `caption` | `data.packed_dataloader_meta_csv_path` |
| image tars (optional) | WebDataset tars of single-frame latents + captions | `data.image_tar_path`, `data.image_sample_prob` |

No pixel decoding and no VAE forward happen during training. Everything the
loader reads is produced offline and read from local disk.

- [1. Encode with the Wan 2.2 VAE](#1-encode-with-the-wan-22-vae)
- [2. Video CSV format](#2-video-csv-format)
- [3. Image tar format](#3-image-tar-format)
- [4. Mixing images and video](#4-mixing-images-and-video)
- [5. Where to change the data paths](#5-where-to-change-the-data-paths)
- [6. Token budgets](#6-token-budgets)
- [7. Sample and debug data](#7-sample-and-debug-data)

## 1. Encode with the Wan 2.2 VAE

Use the diffusers `AutoencoderKLWan` from
[`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
(`subfolder="vae"`). It is the same class and checkpoint the inference pipeline
loads, and the only VAE the code supports. Its geometry is hard-coded in the
trainer:

| | |
| --- | --- |
| latent channels | 48 |
| spatial downsample | 16x |
| temporal downsample | 4x, so `T_latent = 1 + (frames - 1) / 4` |

**Normalize before saving.** The loader feeds latents straight into the
noise/target math with no scaling of its own, so the cache must hold

```
z_norm = (z - latents_mean) / latents_std
```

with `latents_mean` and `latents_std` from the VAE's `config.json`, per
channel, broadcast as `[1, C, 1, 1, 1]`. Inference applies the inverse before
decoding, so the two sides agree only if the cache is normalized this way.

**File format.** One `.safetensors` per clip holding a single tensor named
`latent`, shape `[C, T, H, W]`, bf16. A leading batch dim of 1 is tolerated and
dropped. Anything else, including a different tensor name, is rejected at load
time and the row is skipped with a warning.

**Resolution.** Encode at the training resolution. The latent grid must be
divisible by the spatial patch size, 2 with the default `patch_size: [1, 2, 2]`.
The 720p data is 704x1248 pixels at 24 fps: a 44x78 latent grid, 22x39 = 858
tokens per latent frame. `video_height` and `video_width` in the YAML only
validate this geometry; they do not resize anything.

## 2. Video CSV format

`data.packed_dataloader_meta_csv_path` takes one or more CSV files or
directories; a directory expands to every `*.csv` directly under it. Standard
`csv.DictReader` input: a header row, quoted fields, embedded newlines inside
quotes are fine. Two columns are read:

| column | meaning |
| --- | --- |
| `latent_path` | local path of this clip's latent `.safetensors`, opened as-is |
| `caption` | text fed to Qwen **verbatim**. Structured JSON captions are fine, the whole string is the prompt |

```csv
"latent_path","caption"
"/path/to/latents/video_00.safetensors","{""video_style"": ""..."", ...}"
```

`python scripts/make_sample_data.py` writes a complete 10-row example in this format (see section 7).

Every other column is ignored.

**One CSV per rank.** The sorted CSV list is dealt round-robin across
data-parallel ranks, `world_size / cp_size` of them, so provide at least that
many files or some ranks get no data. The stream loops forever over its shards;
one pass is one epoch. Rows with an empty caption or an unreadable latent are
skipped with a warning; check the skip count in the log after preparing a new
set.

**Per-row sampling.** For each row the loader draws, seeded by the row's
position so a resume reproduces it:

- the task: `v2v` with `video_cond_prob` (a random clean prefix of 1 to T/2 latent
  frames), `i2v` with `image_cond_prob` (the first latent frame), else `t2v`
- caption dropout with `cfg_drop_prob` (the caption is replaced by the empty
  prompt template, for classifier-free guidance)

With `conditioning_schema: channel_concat` the condition is concatenated on
channels: 48 noisy + 48 condition + 1 mask = 97 input channels, hence
`in_channels: 97` and `out_channels: 48` in the model config.

## 3. Image tar format

WebDataset-style tars. Each sample is two members with a shared stem,
contiguous in the tar:

```
00000000.safetensors   tensor "latent", shape [48, 1, H, W], normalized exactly like video
00000000.txt           caption, UTF-8
00000001.safetensors
00000001.txt
```

`scripts/make_sample_data.py` also writes a 10-sample example of this layout.

Encode images with the same Wan 2.2 VAE as one-frame videos, same
normalization. Images may be **any resolution** and resolutions may be mixed
within a shard; the only constraint is a latent grid divisible by the patch
size. Samples missing either member are skipped.

`data.image_tar_path` is a list; each entry may be a directory (expanded
recursively to every `.tar`), a `.tar` file, or a text file listing either.
The resolved shard list is shuffled with a fixed seed and dealt round-robin to
data-parallel ranks, then streamed in a background thread with one tar of
prefetch, cycling forever. As with CSVs, provide at least one tar per rank.

## 4. Mixing images and video

```yaml
data:
  image_tar_path:
    - /path/to/image_tars/
  image_sample_prob: 0.2     # 0 disables mixing entirely
```

Per item the loader draws **image with probability `image_sample_prob`,
otherwise video**, from a counter-seeded RNG, so the interleaving is
deterministic and survives a resume. Images and videos go through the same
packing and the same token budgets. Images always train as `t2v` (no prefix
conditioning) and get their own caption-dropout draw at `cfg_drop_prob`. With
`image_sample_prob: 0` the tar list is not even resolved.

## 5. Where to change the data paths

All in the `data:` section of the YAML:

```yaml
data:
  packed_dataloader_meta_csv_path:          # video CSVs: files or directories
    - /path/to/csvs/
  image_tar_path:                           # image tars: dirs, .tar files, or list files
    - /path/to/image_tars/
  image_sample_prob: 0.2
```

Latent files move with the CSV: edit the `latent_path` column, nothing else
refers to them.

## 6. Token budgets

The packer fills `packed_dataloader_num_buffers` bins up to a joint budget and
emits a batch when a bin is full or nothing fits. Per-item limits reject
oversize samples at load time with a warning; per-batch limits bound the packed
sequence:

| key | scope | meaning |
| --- | --- | --- |
| `packed_joint_token_budget` | batch | video + caption tokens in one packed batch |
| `packed_video_token_budget` | batch, item | video tokens |
| `packed_caption_token_budget` | batch, item | caption tokens after template cropping |
| `packed_dataloader_max_num_seq` | batch | max samples packed together |
| `packed_joint_segment_limit` | item | video + caption tokens of one sample |
| `packed_caption_segment_limit` | item | caption tokens of one sample |

Video tokens per sample are `T_latent * (H/16/ph) * (W/16/pw)`. The 720p
budgets, 109,142 video and 110,576 joint, were sized for one 121-frame clip at
up to 880 tokens per frame plus a long caption; at 44x78 a 121-frame clip is
103,818 tokens. Under context parallelism the budgets describe the full
sequence before it is split across the cp group.

## 7. Sample and debug data

`scripts/make_sample_data.py` writes a small synthetic dataset in the exact
input formats. The latents are Gaussian noise and the captions are placeholders
with the same JSON fields as real ones, so it contains no real images or videos:

```bash
python scripts/make_sample_data.py --output sample_data   # about 100 MB
```

| file | contents |
| --- | --- |
| `sample_data/videos.csv` | 10 video rows, `latent_path` (absolute) + `caption` |
| `sample_data/latents/video_XX.safetensors` | one `latent` tensor `[48, 31, 44, 78]` per row (121 frames at 704x1248); change with `--latent_frames`, `--height`, `--width` |
| `sample_data/images.tar` | 10 image samples, `<key>.safetensors` + `<key>.txt` pairs, at mixed resolutions |

It is for format checks and smoke tests of the training loop, not for training.
Both files can be fed to the loader directly, but one CSV and one tar only
serve a single data-parallel rank.

**Sizing a debug set.** A small set of 8 CSVs with about 100 video rows and
8 image tars of 100 samples each is enough to exercise one node, 8 DP ranks
at `cp_size: 1`. Point the `data` paths in the YAML at your own copy.

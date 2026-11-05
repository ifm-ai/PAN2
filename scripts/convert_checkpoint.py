"""Convert PAN 2 transformer weights between this repository's layout and the
diffusers layout (``PAN2Transformer3DModel``, as in the ``IFM/PAN2`` releases).

to-native: a diffusers ``transformer/`` folder -> ``<out>/model.safetensors``, which
``generate.py --checkpoint_path <out>`` and ``model.pretrained_transformer_path`` load.

    python scripts/convert_checkpoint.py to-native \\
        --input /path/to/PAN2/transformer --output /path/to/pan2-native

to-diffusers: a training ``checkpoint-N/transformer`` (DCP), a ``model_state.pt`` or a
safetensors file/folder -> a diffusers ``transformer/`` folder (sharded safetensors,
index and ``config.json``). The architecture is read from ``--training_config``.

    python scripts/convert_checkpoint.py to-diffusers \\
        --input /path/to/checkpoint-N/transformer --output /path/to/PAN2/transformer --dtype bf16

The two layouts hold the same tensors, apart from three differences:
* diffusers splits the fused projections: ``qkv_proj`` into ``to_q/k/v`` and
  ``add_q/k/v_proj``, ``out_proj`` into ``to_out.0`` and ``to_add_out``, the caption
  encoder ``qkv`` into ``to_q/k/v``, and the per-modality q/k norms into
  ``norm_q/k`` and ``norm_added_q/k``;
* diffusers' SwiGLU stores ``[w3; w1]`` and ``AdaLayerNormContinuous`` stores
  ``[scale; shift]`` (the native output head reads ``[shift; scale]``);
* diffusers keeps only row 0 of the 3-row text type table (the only row the model
  reads) and drops the timestep frequency table (a constant both sides rebuild).
"""
import argparse
import glob
import json
import math
import os
import sys

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QKV_NAMES = ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"]
SHARD_BYTES = 5 * 1024**3


def timestep_freqs(num_features=256, max_period=10000):
    half = num_features // 2
    return torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)


# ---------------------------------------------------------------------------
# diffusers -> native
# ---------------------------------------------------------------------------


def diffusers_to_native(get, num_layers, num_refiner_layers):
    sd = {
        "patch_embed.proj.weight": get("x_embedder.weight"),
        "patch_embed.proj.bias": get("x_embedder.bias"),
        "timestep_encoder.freqs": timestep_freqs(),
        "timestep_encoder.in_proj.weight": get("time_embedder.linear_1.weight"),
        "timestep_encoder.in_proj.bias": get("time_embedder.linear_1.bias"),
        "timestep_encoder.out_proj.weight": get("time_embedder.linear_2.weight"),
        "timestep_encoder.out_proj.bias": get("time_embedder.linear_2.bias"),
    }
    for key in ("0.weight", "1.weight", "1.bias"):
        sd[f"cap_embedder.{key}"] = get(f"context_embedder.{key}")

    for i in range(num_refiner_layers):
        src, dst = f"context_refiner.{i}.", f"caption_encoder.blocks.{i}."
        sd[dst + "attn_norm.weight"] = get(src + "norm1.weight")
        sd[dst + "ffn_norm.weight"] = get(src + "norm2.weight")
        sd[dst + "qkv.weight"] = torch.cat([get(src + f"attn.to_{n}.weight") for n in "qkv"])
        sd[dst + "q_norm.weight"] = get(src + "attn.norm_q.weight")
        sd[dst + "k_norm.weight"] = get(src + "attn.norm_k.weight")
        sd[dst + "attn_out.weight"] = get(src + "attn.to_out.0.weight")
        w3, w1 = get(src + "ff.net.0.proj.weight").chunk(2)
        sd[dst + "ffn.w1.weight"], sd[dst + "ffn.w3.weight"] = w1, w3
        sd[dst + "ffn.w2.weight"] = get(src + "ff.net.2.weight")

    row0 = get("context_type_embedding")
    sd["text_type_embedding.weight"] = torch.stack([row0, torch.zeros_like(row0), torch.zeros_like(row0)])

    for i in range(num_layers):
        src, dst = f"transformer_blocks.{i}.", f"blocks.{i}."
        sd[dst + "modulation.linear.weight"] = get(src + "norm1.linear.weight")
        sd[dst + "modulation.linear.bias"] = get(src + "norm1.linear.bias")
        sd[dst + "text_modulation"] = get(src + "scale_shift_table_context")
        sd[dst + "qkv_proj.weight"] = torch.cat([get(src + f"attn.{n}.weight") for n in QKV_NAMES])
        sd[dst + "qkv_proj.bias"] = torch.cat([get(src + f"attn.{n}.bias") for n in QKV_NAMES])
        sd[dst + "qkv_proj.q_weight"] = torch.cat([get(src + "attn.norm_q.weight"), get(src + "attn.norm_added_q.weight")])
        sd[dst + "qkv_proj.k_weight"] = torch.cat([get(src + "attn.norm_k.weight"), get(src + "attn.norm_added_k.weight")])
        sd[dst + "out_proj.weight"] = torch.cat([get(src + "attn.to_out.0.weight"), get(src + "attn.to_add_out.weight")])
        sd[dst + "out_proj.bias"] = torch.cat([get(src + "attn.to_out.0.bias"), get(src + "attn.to_add_out.bias")])
        for modality, ff in (("video", "ff"), ("text", "ff_context")):
            sd[dst + f"ffn.w_up_{modality}"] = get(src + f"{ff}.net.0.proj.weight")
            sd[dst + f"ffn.b_up_{modality}"] = get(src + f"{ff}.net.0.proj.bias")
            sd[dst + f"ffn.w_down_{modality}"] = get(src + f"{ff}.net.2.weight")
            sd[dst + f"ffn.b_down_{modality}"] = get(src + f"{ff}.net.2.bias")

    for suffix in ("weight", "bias"):
        scale, shift = get(f"norm_out.linear.{suffix}").chunk(2)
        sd[f"output_head.modulation.linear.{suffix}"] = torch.cat([shift, scale])
    sd["output_head.proj.weight"] = get("proj_out.weight")
    sd["output_head.proj.bias"] = get("proj_out.bias")
    return sd


def run_to_native(args):
    config = json.load(open(os.path.join(args.input, "config.json")))
    index_path = os.path.join(args.input, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
    else:
        weight_map = {}
        for shard in glob.glob(os.path.join(args.input, "*.safetensors")):
            with safe_open(shard, "pt") as f:
                weight_map.update({key: os.path.basename(shard) for key in f.keys()})
    handles, used = {}, set()

    def get(key):
        shard = weight_map[key]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(args.input, shard), "pt")
        used.add(key)
        return handles[shard].get_tensor(key)

    sd = diffusers_to_native(get, config["num_layers"], config["num_refiner_layers"])
    leftover = sorted(set(weight_map) - used)
    if leftover:
        raise ValueError(f"Unconverted diffusers tensors: {leftover[:10]}")
    os.makedirs(args.output, exist_ok=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(args.output, "model.safetensors"),
              metadata={"format": "pt"})
    print(f"{len(weight_map)} diffusers tensors -> {len(sd)} native tensors in {args.output}/model.safetensors")


# ---------------------------------------------------------------------------
# native -> diffusers
# ---------------------------------------------------------------------------


def native_to_diffusers(sd, num_layers, num_refiner_layers, head_dim):
    sd = dict(sd)
    out = {}

    def move(src, dst):
        out[dst] = sd.pop(src)

    move("patch_embed.proj.weight", "x_embedder.weight")
    move("patch_embed.proj.bias", "x_embedder.bias")
    sd.pop("timestep_encoder.freqs")
    move("timestep_encoder.in_proj.weight", "time_embedder.linear_1.weight")
    move("timestep_encoder.in_proj.bias", "time_embedder.linear_1.bias")
    move("timestep_encoder.out_proj.weight", "time_embedder.linear_2.weight")
    move("timestep_encoder.out_proj.bias", "time_embedder.linear_2.bias")
    for key in ("0.weight", "1.weight", "1.bias"):
        move(f"cap_embedder.{key}", f"context_embedder.{key}")

    for i in range(num_refiner_layers):
        src, dst = f"caption_encoder.blocks.{i}.", f"context_refiner.{i}."
        move(src + "attn_norm.weight", dst + "norm1.weight")
        move(src + "ffn_norm.weight", dst + "norm2.weight")
        for name, w in zip("qkv", sd.pop(src + "qkv.weight").chunk(3)):
            out[dst + f"attn.to_{name}.weight"] = w
        move(src + "q_norm.weight", dst + "attn.norm_q.weight")
        move(src + "k_norm.weight", dst + "attn.norm_k.weight")
        move(src + "attn_out.weight", dst + "attn.to_out.0.weight")
        out[dst + "ff.net.0.proj.weight"] = torch.cat([sd.pop(src + "ffn.w3.weight"), sd.pop(src + "ffn.w1.weight")])
        move(src + "ffn.w2.weight", dst + "ff.net.2.weight")

    out["context_type_embedding"] = sd.pop("text_type_embedding.weight")[0]

    for i in range(num_layers):
        src, dst = f"blocks.{i}.", f"transformer_blocks.{i}."
        move(src + "modulation.linear.weight", dst + "norm1.linear.weight")
        move(src + "modulation.linear.bias", dst + "norm1.linear.bias")
        move(src + "text_modulation", dst + "scale_shift_table_context")
        weights = sd.pop(src + "qkv_proj.weight").chunk(6)
        biases = sd.pop(src + "qkv_proj.bias").chunk(6)
        for name, w, b in zip(QKV_NAMES, weights, biases):
            out[dst + f"attn.{name}.weight"], out[dst + f"attn.{name}.bias"] = w, b
        q_norm, k_norm = sd.pop(src + "qkv_proj.q_weight"), sd.pop(src + "qkv_proj.k_weight")
        out[dst + "attn.norm_q.weight"], out[dst + "attn.norm_added_q.weight"] = q_norm[:head_dim], q_norm[head_dim:]
        out[dst + "attn.norm_k.weight"], out[dst + "attn.norm_added_k.weight"] = k_norm[:head_dim], k_norm[head_dim:]
        out[dst + "attn.to_out.0.weight"], out[dst + "attn.to_add_out.weight"] = sd.pop(src + "out_proj.weight").chunk(2)
        out[dst + "attn.to_out.0.bias"], out[dst + "attn.to_add_out.bias"] = sd.pop(src + "out_proj.bias").chunk(2)
        for modality, ff in (("video", "ff"), ("text", "ff_context")):
            move(src + f"ffn.w_up_{modality}", dst + f"{ff}.net.0.proj.weight")
            move(src + f"ffn.b_up_{modality}", dst + f"{ff}.net.0.proj.bias")
            move(src + f"ffn.w_down_{modality}", dst + f"{ff}.net.2.weight")
            move(src + f"ffn.b_down_{modality}", dst + f"{ff}.net.2.bias")

    for suffix in ("weight", "bias"):
        shift, scale = sd.pop(f"output_head.modulation.linear.{suffix}").chunk(2)
        out[f"norm_out.linear.{suffix}"] = torch.cat([scale, shift])
    move("output_head.proj.weight", "proj_out.weight")
    move("output_head.proj.bias", "proj_out.bias")
    if sd:
        raise ValueError(f"Unconverted native tensors: {sorted(sd)[:10]}")
    return out


def load_native(path):
    if os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata")):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata

        reader = dcp.FileSystemReader(path)
        metadata = reader.read_metadata()
        sd = {key: torch.empty(meta.size, dtype=meta.properties.dtype)
              for key, meta in metadata.state_dict_metadata.items() if isinstance(meta, TensorStorageMetadata)}
        dcp.load(sd, storage_reader=reader, no_dist=True)
    elif os.path.isdir(path):
        sd = {}
        for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            sd |= load_file(shard)
        if not sd:
            raise ValueError(f"No DCP .metadata or *.safetensors in {path}")
    elif path.endswith(".safetensors"):
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        for key in ("state_dict", "model"):
            if isinstance(sd.get(key), dict):
                sd = sd[key]
    return {k.removeprefix("module.").removeprefix("model."): v for k, v in sd.items()}


def diffusers_config(training_config):
    import inspect
    from pan.models.transformers.pan_transformer import PanVideoDiffusionTransformer
    from pan.pipelines.pan_video_pipeline import PanVideoPipeline

    overrides = yaml.safe_load(open(training_config))["model"]["transformer_config_overrides"]
    defaults = {name: p.default for name, p in inspect.signature(PanVideoDiffusionTransformer.__init__).parameters.items()
                if p.default is not inspect.Parameter.empty}
    cfg = {**defaults, **PanVideoPipeline.architecture(overrides)}  # legacy key names are translated
    return {
        "_class_name": "PAN2Transformer3DModel",
        "attention_head_dim": cfg["hidden_size"] // cfg["num_attention_heads"],
        "context_mlp_ratio": float(cfg["context_mlp_ratio"]),
        "in_channels": cfg["in_channels"],
        "mlp_ratio": float(cfg["mlp_ratio"]),
        "num_attention_heads": cfg["num_attention_heads"],
        "num_layers": cfg["num_layers"],
        "num_refiner_layers": 4,
        "out_channels": cfg["out_channels"],
        "patch_size": list(cfg["patch_size"]),
        "refiner_mlp_ratio": 4.0,
        "rope_axes_dim": list(cfg["rope_axes_dim"]),
        "rope_temporal_scale": float(cfg["rope_temporal_scale"]),
        "rope_theta": float(cfg["rope_theta"]),
        "text_embed_dim": cfg["text_embed_dim"],
    }


def run_to_diffusers(args):
    config = diffusers_config(args.training_config)
    sd = native_to_diffusers(load_native(args.input), config["num_layers"], config["num_refiner_layers"],
                             config["attention_head_dim"])
    if args.dtype:
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
        sd = {k: v.to(dtype) for k, v in sd.items()}
    os.makedirs(args.output, exist_ok=True)

    shards, current, size = [], {}, 0
    for key in sorted(sd):
        nbytes = sd[key].numel() * sd[key].element_size()
        if current and size + nbytes > SHARD_BYTES:
            shards.append(current)
            current, size = {}, 0
        current[key] = sd[key].contiguous()
        size += nbytes
    shards.append(current)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards, 1):
        name = f"diffusion_pytorch_model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(args.output, name), metadata={"format": "pt"})
        for key, tensor in shard.items():
            weight_map[key] = name
            total += tensor.numel() * tensor.element_size()
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(args.output, "diffusion_pytorch_model.safetensors.index.json"), "w"), indent=2)
    json.dump(config, open(os.path.join(args.output, "config.json"), "w"), indent=2)
    print(f"{len(sd)} diffusers tensors in {len(shards)} shard(s) -> {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="direction", required=True)
    p = sub.add_parser("to-native", help="diffusers transformer folder -> this repository's layout")
    p.add_argument("--input", required=True, help="diffusers transformer/ folder (config.json + safetensors)")
    p.add_argument("--output", required=True, help="output folder; model.safetensors is written there")
    p = sub.add_parser("to-diffusers", help="this repository's checkpoint -> diffusers transformer folder")
    p.add_argument("--input", required=True, help="checkpoint-N/transformer (DCP), model_state.pt or safetensors")
    p.add_argument("--output", required=True, help="output transformer/ folder")
    p.add_argument("--training_config", default="config/sft_29B_720p.yaml", help="YAML defining the architecture")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default=None, help="cast the weights (default: keep)")
    args = parser.parse_args()
    {"to-native": run_to_native, "to-diffusers": run_to_diffusers}[args.direction](args)


if __name__ == "__main__":
    main()

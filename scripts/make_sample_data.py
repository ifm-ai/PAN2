"""Write a small synthetic dataset in the exact formats the training loaders read.

Latents are Gaussian noise and captions are a fixed placeholder, so nothing here
comes from a real image or video. Use it to check the formats and to smoke-test
the training loop, not to train a model.

    python scripts/make_sample_data.py --output sample_data

writes

    sample_data/videos.csv                  latent_path,caption rows (absolute paths)
    sample_data/latents/video_XX.safetensors one "latent" tensor [48, T, H, W] per clip
    sample_data/images.tar                  <key>.safetensors ("latent" [48, 1, H, W]) + <key>.txt pairs

See docs/data.md for the formats.
"""
import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import torch
from safetensors.torch import save, save_file

LATENT_CHANNELS = 48
# Latent (h, w) grids for the image samples; 44x78 is a 704x1248 frame.
IMAGE_GRIDS = [(44, 78), (78, 44), (48, 28), (32, 32), (40, 60)]


def video_caption(index: int, seconds: float) -> str:
    """A placeholder structured caption with the same fields as the real ones."""
    half = round(seconds / 2, 1)
    return json.dumps({
        "video_style": "Placeholder",
        "lighting": "Placeholder",
        "perspective_view": "Placeholder",
        "subjects_lifecycle": [{
            "subject_id": "<|Subject1|>",
            "display_name": f"Placeholder subject {index}",
            "subject_attribute": "Placeholder",
        }],
        "init_frame_visual_description": f"Synthetic sample {index}; the latent is random noise.",
        "events": [
            {
                "event_index": i,
                "timestamp": f"{start} - {end}",
                "event_description": f"Placeholder event {i}.",
                "camera_motion": [{"timestamp": f"{start} - {end}", "camera_motion": ["static"]}],
                "contained_actions": [{"timestamp": f"{start} - {end}", "action": "<|Subject1|> placeholder action"}],
            }
            for i, (start, end) in enumerate([(0.0, half), (half, round(seconds, 1))])
        ],
        "quality": "Placeholder",
    }, indent=2)


def image_caption(index: int) -> str:
    return json.dumps({"events": [{"init_frame_visual_description": f"Synthetic image {index}; the latent is random noise."}]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="sample_data", help="Output directory.")
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--latent_frames", type=int, default=31, help="T of each video latent; 31 is 121 frames.")
    parser.add_argument("--height", type=int, default=44, help="Latent height of the videos (44 = 704 px).")
    parser.add_argument("--width", type=int, default=78, help="Latent width of the videos (78 = 1248 px).")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output).resolve()
    (out / "latents").mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    seconds = ((args.latent_frames - 1) * 4 + 1) / args.fps

    with open(out / "videos.csv", "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["latent_path", "caption"])
        for i in range(args.num_videos):
            shape = (LATENT_CHANNELS, args.latent_frames, args.height, args.width)
            path = out / "latents" / f"video_{i:02d}.safetensors"
            save_file({"latent": torch.randn(shape, generator=generator).to(torch.bfloat16)}, path)
            writer.writerow([str(path), video_caption(i, seconds)])

    with tarfile.open(out / "images.tar", "w") as tar:
        for i in range(args.num_images):
            h, w = IMAGE_GRIDS[i % len(IMAGE_GRIDS)]
            latent = torch.randn((LATENT_CHANNELS, 1, h, w), generator=generator).to(torch.bfloat16)
            for name, data in ((f"{i:08d}.safetensors", save({"latent": latent})),
                               (f"{i:08d}.txt", image_caption(i).encode())):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    print(f"Wrote {args.num_videos} videos and {args.num_images} images to {out}")


if __name__ == "__main__":
    main()

"""CPU-only T2V/I2V input preparation and conservative per-output resume checks."""

from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
from pathlib import Path, PurePosixPath
import re


GUIDANCE_KEYS = ("cfg_start_step", "cfg_end_step", "stg_scale", "stg_skip_layers")




def _positive(value, name, integer=False):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be positive and finite") from None
    if isinstance(value, bool) or not number.is_finite() or number <= 0 or (integer and number != number.to_integral_value()):
        raise ValueError(f"{name} must be positive and finite" + (" integer" if integer else ""))
    return number


def resolve_frames(entry, fps=24, fallback=121):
    """Duration wins; round up to the 4k+1 frame counts the Wan VAE encodes."""
    fps_value = _positive(fps, "fps")
    fallback_value = _positive(fallback, "video_length", integer=True)
    if entry.get("duration_sec") is not None:
        duration = _positive(entry["duration_sec"], "duration_sec")
        return int((duration * fps_value / 4).to_integral_value(rounding=ROUND_CEILING)) * 4 + 1
    for key in ("frames", "num_target_frames"):
        if entry.get(key) is not None:
            return int(_positive(entry[key], key, integer=True))
    return int(fallback_value)


def render_negative_duration(text, video_length, fps=24):
    """Stamp the clip's duration into a structured negative prompt.

    The shipped negative is a template whose ``time_range_sec`` fields span a
    nominal ``0 to 10`` seconds; the per-record negatives of the evaluation sets
    are the same template rendered for each record's duration. Plain-text
    negatives and anything that is not a JSON object are returned unchanged.
    """
    try:
        negative = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    if not isinstance(negative, dict):
        return text
    seconds = (int(video_length) - 1) / float(fps)
    span = f"0 to {seconds:g}"

    def visit(node):
        if isinstance(node, dict):
            return {key: (span if key == "time_range_sec" else visit(value)) for key, value in node.items()}
        if isinstance(node, list):
            return [visit(item) for item in node]
        return node

    return json.dumps(visit(negative), ensure_ascii=False, separators=(",", ":"))




def load_inference_inputs(path, output_dir, *, task="t2v", fps=24, fallback=121,
                    shard_index=0, shard_count=1, default_negative="", validate_images=True, image_root=None):
    if task not in {"t2v", "i2v", "auto"}:
        raise ValueError("JSON input task must be t2v, i2v, or auto")
    if type(validate_images) is not bool:
        raise ValueError("validate_images must be a boolean")
    if image_root is not None:
        if not isinstance(image_root, (str, Path)) or not str(image_root).strip():
            raise ValueError("image_root must be a nonempty local directory path")
        image_root = Path(image_root).resolve()
        if validate_images and not image_root.is_dir():
            raise ValueError(f"image_root must be an existing local directory: {image_root}")
    if type(shard_count) is not int or type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= prompt_shard_index < prompt_shard_count")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        keys = [key for key in ("entries", "prompts", "data") if key in payload]
        if len(keys) != 1:
            raise ValueError("Prompt wrapper needs one entries/prompts/data list")
        payload = payload[keys[0]]
    if not isinstance(payload, list):
        raise ValueError("Prompt JSON must contain an entry list")
    root = Path(output_dir).resolve()
    records, seen = [], set()
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Prompt entries must be objects")
        tasks = [str(entry[k]).strip().lower() for k in ("task", "type") if entry.get(k)]
        if len(set(tasks)) > 1:
            raise ValueError("Conflicting task/type fields")
        image_aliases = [key for key in ("image_path", "image", "ref_image") if entry.get(key)]
        entry_task = tasks[0] if tasks else ("i2v" if image_aliases else "t2v")
        if entry_task not in {"t2v", "i2v"}:
            raise ValueError(f"Unsupported task: {entry_task}")
        if task != "auto" and entry_task != task:
            continue
        image_path = None
        if entry_task == "i2v":
            if not image_aliases:
                raise ValueError("I2V entry requires image_path (or image/ref_image)")
            if any(entry[key] != entry[image_aliases[0]] for key in image_aliases[1:]):
                raise ValueError("Conflicting reference image aliases")
            image_path = entry[image_aliases[0]]
            if not isinstance(image_path, str) or not image_path.strip():
                raise ValueError("I2V reference image must be an existing local file")
            if image_root is not None:
                basename = Path(image_path).name
                if basename in {"", ".", ".."}:
                    raise ValueError("I2V reference image requires a filename for image_root remapping")
                image_path = image_root / basename
            image_path = str(Path(image_path).resolve())
            if validate_images and not Path(image_path).is_file():
                raise ValueError(f"I2V reference image must be an existing local file: {image_path}")
        elif image_aliases:
            raise ValueError("T2V entry cannot contain an I2V reference image")
        # A rewrite (prompts_rewritter.py adds refined_prompts next to the raw prompt)
        # wins over the raw prompt, whether or not the entry has an id.
        fields = ("refined_prompt", "refined_prompts", "prompt", "caption")
        prompt = next((entry[k] for k in fields if entry.get(k)), None)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{entry_task.upper()} entry needs a nonempty positive prompt")
        entry_id = "" if entry.get("id") is None else str(entry["id"]).strip()
        if entry_id:
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", entry_id).strip("._-")
            if not name:
                raise ValueError("Empty output id after sanitization")
            relative = f"{entry_task}_{name}.mp4"
        else:
            relative = str(entry.get("relative_path") or "").strip() or f"{entry_task}_{len(records):04d}.mp4"
        rel = PurePosixPath(relative)
        # Mixed collections may reuse the same relative name across tasks.
        # Explicit task filters retain their historical output layout.
        task_root = root / entry_task if task == "auto" else root
        target = (task_root / relative).resolve()
        if rel.is_absolute() or ".." in rel.parts or "\\" in relative or rel.suffix.lower() != ".mp4" or not target.is_relative_to(root):
            raise ValueError(f"Unsafe output path: {relative}")
        if target in seen:
            raise ValueError(f"Duplicate output path: {relative}")
        seen.add(target)
        video_length = resolve_frames(entry, fps, fallback)
        aliases = [k for k in ("negative_prompt", "negative_prompts", "neg_prompt") if k in entry]
        if any(entry[k] != entry[aliases[0]] for k in aliases[1:]):
            raise ValueError("Conflicting negative prompt aliases")
        if aliases:
            negative = entry[aliases[0]]
        else:
            # The shared fallback is a template; render it for this clip.
            negative = render_negative_duration(default_negative, video_length, fps)
        records.append(dict(prompt=prompt, negative_prompt=negative,
                            video_length=video_length, output_path=str(target)))
        if task == "auto":
            records[-1]["task"] = entry_task
        if entry_task == "i2v":
            records[-1]["image_path"] = image_path
    if not records:
        raise ValueError(f"Prompt JSON contains no {task.upper()} entries")
    return records[shard_index::shard_count]


def is_complete(args):
    """Skip only a nonempty video/config pair with matching effective arguments.

    A partial pair is resumable. A complete but differently configured pair
    fails closed: use another output directory or explicitly disable skipping.
    """
    video = Path(args.output_path)
    config = video.with_name(video.stem + "_config.json")
    if not video.is_file() or not video.stat().st_size or not config.is_file() or not config.stat().st_size:
        return False
    try:
        saved = json.loads(config.read_text(encoding="utf-8"))["arguments"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not isinstance(saved, dict):
        return False
    # Routing and labelling arguments only decide where a video lands; every
    # other argument is part of what was generated and must match exactly.
    controls = {"prompt_json", "output_dir", "prompt_shard_index", "prompt_shard_count", "skip_existing",
                "output_path", "task", "image_root"}
    mismatches = [key for key, value in vars(args).items()
                  if not key.startswith("_") and key not in controls and (key not in saved or saved[key] != value)]
    if mismatches:
        raise ValueError(f"Existing output settings differ for {video}: {', '.join(mismatches)}")
    return True

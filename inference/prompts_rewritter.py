from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-9B"  # the same Apache-2.0 model PAN 2 uses as its text encoder
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_DTYPE = "auto"
DEFAULT_NUM_GPUS = 8
REFINED_PROMPT_FIELD = "refined_prompts"
IMAGE_PATH_KEYS = (
    "image_path",
    "image",
    "ref_image",
    "reference_image",
    "input_frame",
    "input_image",
)


@dataclass(frozen=True)
class RewriteRequest:
    prompt: str
    image_path: str | None = None

REWRITE_SYSTEM_PROMPT = """You are a video-generation caption rewriter. Rewrite user-provided text into the same natural-language structure and style as the `global_caption` captions from the latest video captioning system.

Output only the rewritten caption text. Do not output JSON, markdown, headings, bullets, explanations, or quotes.

Write exactly four prose paragraphs separated by blank lines:

General input handling:
- The input may be a short prompt, a long caption, a partially structured description, or a detailed multi-paragraph caption.
- Preserve user-specified semantics, subject identities, requested action, camera intent, and constraints. Improve clarity and completeness without changing the requested video.
- If the input is already detailed, reorganize and normalize it into the four-paragraph caption style instead of expanding it as if it were only a simple sentence.

Reference-image handling for image-to-video:
- If a reference image is attached, treat it as the exact first visible frame and the visual authority for initial scene content.
- In that case, paragraph 2 must rely on the attached image for setting, subject/object inventory, appearance, position, readable text, lighting, style, and initial camera framing. Do not hallucinate initial-state details that are not visible in the image.
- In that case, paragraph 3 must describe motion as a continuation from the attached image. The moving subjects, their starting positions, directions, contacts, and camera behavior must remain consistent with the reference image and the user text.
- If the user asks for a new subject or object that is not visible in the reference image, introduce it only as an entering or newly appearing element during paragraph 3, not as something already present in paragraph 2.
- If the user text conflicts with the reference image, keep the image-grounded initial state and satisfy requested motion only where it can be made visually consistent.

Paragraph 1, summary:
- Write one or two concise sentences at the global level.
- Mention the camera range or shot type, visual style or medium, major subject/topic, setting category, and what the video is about when those details are implied or can be added without changing the intent.
- Keep this paragraph at the global level. Do not put detailed subject appearance, object inventories, text/logo details, lighting specifics, or background layout here; save those details for paragraph 2.
- Prefer phrasing like "A close-up photorealistic shot of..." or "A wide handheld-style view of..." when appropriate.

Paragraph 2, initial state:
- Describe only the first visible moment as a still image.
- Include the setting, layout, foreground/midground/background placement, subject appearance, object color/shape/size/material, readable text if explicitly present, lighting, color palette, and composition.
- Use frame-relative position words such as `frame-left`, `frame-right`, `upper frame`, `lower frame`, and `center`.
- For major foreground subjects, use specific visible descriptions rather than coarse labels when more detail is available.
- Do not include objects, text, poses, lighting changes, or layout details that appear only later; those belong in paragraph 3.
- End this paragraph with the initial camera description: shot type/range, angle, framing/composition, perspective, focus, and stability.

Paragraph 3, dynamic motion:
- Describe what happens in chronological order using natural temporal phrasing such as "at first", "then", "as", "next", "while", "afterward", or "eventually".
- Preserve every action, interaction, object state change, and physical relationship from the input text.
- Explicitly describe motion for any moving subject or object, including path, direction, rotation, bending, swinging, contact, support, resistance, settling, or lack of significant movement.
- If important elements remain static or persist throughout, say so.
- End this paragraph with the dynamic camera description, including whether the camera is static, panning, tilting, tracking, dollying, zooming, handheld, reframing, changing focus, or only slightly shaky. If the camera is static, fixed, or only slightly shaky, state that explicitly instead of omitting camera behavior.

Paragraph 4, other key attributes:
- Add remaining important physics, material, contact/support, continuity, lighting, visual style, scene-context, text/logo persistence, or image-quality details that help the caption stand alone for text-to-video training.
- Mention visual artifacts, watermarks, borders, compression, blur, overexposure, or disruptive edits only if the input explicitly implies them.

Style examples:

Example 1:
A close-up cinematic shot of a performer in traditional Chinese opera attire performing a stylized sequence of movements in an ornate indoor setting.

At the beginning, a performer is centered in the frame, wearing elaborate white face makeup with pink shading around the eyes and red lips. They are adorned with a massive, intricate headdress featuring blue and red accents, numerous white pearls, and dangling beaded strands. The performer wears a white robe with wide sleeves and a colorful garment underneath featuring blue, red, and gold embroidery. They hold a small, rectangular golden box in front of their face with both hands, partially obscuring their features. In the background, a large, ornate yellow screen with blue dragon motifs is visible, set against a red backdrop with purple cloud patterns. Two other performers in similar opera makeup and headdresses are partially visible on the frame-left and frame-right. The lighting is soft and warm, creating a rich, saturated color palette. The camera is a static close-up shot with a centered composition and stable focus.

As the sequence progresses, the performer slowly lowers the golden box, revealing their face. They then lift the box higher, momentarily resting it atop their headdress while glancing toward the frame-left. Afterward, the performer brings the box back down and moves it to the frame-left, while their right hand moves toward their face in a graceful, stylized gesture. Throughout the motion, the beaded strands of the headdress sway gently. The other performers in the background remain mostly static, maintaining their poised expressions. The camera remains static throughout the entire sequence, with no movement, zoom, or change in framing.

The materials of the costumes are highly detailed, showing the shimmer of the gold embroidery and the luster of the pearls. The makeup is applied with a smooth, matte finish, contrasting with the glossy surface of the golden box. The scene maintains consistent continuity in lighting and composition, with no visible visual artifacts or disruptive edits.

Example 2:
A wide handheld-style shot captures a variety of potted citrus and flowering plants for sale at an outdoor market, set against a backdrop of a parking lot and blossoming trees.

At the beginning, the scene is filled with numerous black plastic pots containing small citrus trees laden with orange fruits and some yellow-flowering shrubs. The plants are arranged in rows on a light gray concrete pavement. In the background, a blue canopy tent is visible on the frame-left, with several people walking nearby. Further back, a parking lot contains various white and silver cars, and the horizon is lined with white-blossomed trees under an overcast sky. The lighting is soft and diffused, creating a natural, low-contrast color palette. The camera starts in a wide shot with a stable, eye-level perspective and deep focus.

As the video progresses, the camera begins a complex movement, panning to the frame-right while simultaneously tracking right and moving backward. This motion shifts the focus from the citrus trees on the frame-left toward the yellow-flowering shrubs on the frame-right. The camera also tilts slightly upward and rolls clockwise, creating a dynamic, sweeping view of the plant nursery. The people in the background continue to move slowly, and the cars remain stationary in the parking lot. The camera movement is fluid but exhibits slight handheld jitter as it reframes the composition to emphasize the yellow blossoms near the end.

The plants are supported by black plastic nursery pots, some with small white labels attached. The citrus fruits are bright orange, contrasting with the deep green foliage, while the shrubs feature vibrant yellow flowers. The concrete ground is scattered with a few fallen orange fruits and yellow petals. The overall scene maintains temporal continuity, with the background elements remaining consistent as the perspective shifts.

Constraints:
- Preserve the input text semantics and intent.
- Add plausible visual detail only to make the caption complete; do not introduce a new plot, new subjects, or extra events that extend the implied video before or after the input.
- If the input only describes the start of an action, describe only that start and do not invent the middle or end.
- If a reference image is attached, any added visual detail must be consistent with the visible image.
- Avoid strong speed words such as "fast", "rapid", "quick", "explosive", or "frantic" unless the input explicitly requires them.
- Do not refer to frames by number.
- Do not use JSON reference IDs such as `object_1`, `subject_1`, `actor_id`, or `entity_id`.
- The result should read like a dense video caption, not an instruction to create a video.
"""


def format_elapsed_time(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"

    minutes, remaining_seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"

    hours, remaining_minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.2f}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite one prompt or a prompt file using Hugging Face Transformers."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        help="Prompt text to rewrite.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help=(
            "Prompt file for batch rewriting when used with --output-json. "
            "Plain text files are read as one prompt per non-empty line; JSON/JSONL are also supported."
        ),
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help=(
            "Optional reference image/input frame for I2V rewriting. For JSON/JSONL prompt files, "
            "per-record image_path/image/ref_image/reference_image is used first; this value is a fallback."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help=(
            "Write rewrite results as JSON. For JSON prompt files, preserves the "
            f"input records and adds {REFINED_PROMPT_FIELD!r}."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Hugging Face repo id or local path of the rewriter model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max generated tokens. Default: {DEFAULT_MAX_NEW_TOKENS}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE}",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Sampling top_p. Default: {DEFAULT_TOP_P}",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=DEFAULT_DTYPE,
        help=f"Model dtype. Default: {DEFAULT_DTYPE}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for the single-device model load. Defaults to cuda:0 if CUDA is available, else cpu.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Optional CUDA device id for single-prompt rewriting, for example 0.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help=f"Number of GPUs to use for batch rewriting. Default: {DEFAULT_NUM_GPUS}",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated CUDA device ids for batch rewriting, for example 0,1,2,3,4,5,6,7. "
            "Overrides --num-gpus."
        ),
    )
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None and args.prompt_file is not None:
        raise SystemExit("Use either a positional prompt or --prompt-file, not both.")

    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None:
        if not args.prompt_file.exists():
            raise SystemExit(f"Prompt file does not exist: {args.prompt_file}")
        prompt = args.prompt_file.read_text(encoding="utf-8")
    else:
        raise SystemExit("Provide a prompt or --prompt-file.")

    prompt = prompt.strip()
    if not prompt:
        raise SystemExit("Prompt is blank.")
    return prompt


def resolve_image_path(
    image_value: Any,
    source: str,
    prompt_file: Path | None = None,
) -> str | None:
    if image_value is None:
        return None
    if not isinstance(image_value, str):
        raise ValueError(f"{source} image path must be a string, got {type(image_value)!r}.")

    raw_value = image_value.strip()
    if not raw_value or raw_value.lower() in {"none", "null", "nil"}:
        return None
    if raw_value.startswith(("http://", "https://", "data:image")):
        return raw_value

    image_path = Path(raw_value).expanduser()
    if image_path.is_absolute():
        return str(image_path)

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [Path.cwd() / image_path]
    if prompt_file is not None:
        candidates.append(prompt_file.parent / image_path)
    candidates.append(repo_root / image_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(image_path)


def extract_prompt_text(item: Any, source: str) -> str:
    if isinstance(item, str):
        prompt = item.strip()
    elif isinstance(item, dict):
        prompt = ""
        for key in ("prompts", "prompt", "caption", "input", "text"):
            value = item.get(key)
            if isinstance(value, str):
                prompt = value.strip()
                break
        if not prompt:
            raise ValueError(
                f"{source} must contain a string prompt, caption, input, or text field."
            )
    else:
        raise ValueError(f"{source} must be a string or dict, got {type(item)!r}.")

    if not prompt:
        raise ValueError(f"{source} contains a blank prompt.")
    return prompt


def extract_image_path(
    item: Any,
    source: str,
    prompt_file: Path | None,
    fallback_image_path: str | None,
) -> str | None:
    if isinstance(item, dict):
        for key in IMAGE_PATH_KEYS:
            if key in item:
                image_path = resolve_image_path(item.get(key), f"{source}.{key}", prompt_file)
                if image_path:
                    return image_path
    return fallback_image_path


def extract_rewrite_request(
    item: Any,
    source: str,
    prompt_file: Path | None = None,
    fallback_image_path: str | None = None,
) -> RewriteRequest:
    return RewriteRequest(
        prompt=extract_prompt_text(item, source),
        image_path=extract_image_path(item, source, prompt_file, fallback_image_path),
    )


def extract_rewrite_requests_from_json(
    payload: Any,
    source: str,
    prompt_file: Path | None = None,
    fallback_image_path: str | None = None,
) -> list[RewriteRequest]:
    if isinstance(payload, dict):
        for key in ("entries", "prompts", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break

    if isinstance(payload, list):
        requests = [
            extract_rewrite_request(
                item,
                f"{source}[{idx}]",
                prompt_file=prompt_file,
                fallback_image_path=fallback_image_path,
            )
            for idx, item in enumerate(payload)
        ]
    else:
        requests = [
            extract_rewrite_request(
                payload,
                source,
                prompt_file=prompt_file,
                fallback_image_path=fallback_image_path,
            )
        ]

    if not requests:
        raise ValueError(f"{source} contains no prompts.")
    return requests


def read_batch_prompts(
    prompt_file: Path,
    fallback_image_path: str | None = None,
) -> list[RewriteRequest]:
    if not prompt_file.exists():
        raise SystemExit(f"Prompt file does not exist: {prompt_file}")

    text = prompt_file.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Prompt file is blank.")

    fallback_image_path = resolve_image_path(
        fallback_image_path,
        "--image-path",
        prompt_file,
    )

    suffix = prompt_file.suffix.lower()
    try:
        if suffix == ".json":
            return extract_rewrite_requests_from_json(
                json.loads(text),
                str(prompt_file),
                prompt_file=prompt_file,
                fallback_image_path=fallback_image_path,
            )

        if suffix == ".jsonl":
            requests: list[RewriteRequest] = []
            for line_idx, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    item = stripped
                requests.append(
                    extract_rewrite_request(
                        item,
                        f"{prompt_file}:{line_idx}",
                        prompt_file=prompt_file,
                        fallback_image_path=fallback_image_path,
                    )
                )
            if not requests:
                raise ValueError(f"{prompt_file} contains no prompts.")
            return requests

        if text[0] in "[{":
            try:
                return extract_rewrite_requests_from_json(
                    json.loads(text),
                    str(prompt_file),
                    prompt_file=prompt_file,
                    fallback_image_path=fallback_image_path,
                )
            except json.JSONDecodeError:
                pass

        requests = [
            RewriteRequest(line.strip(), fallback_image_path)
            for line in text.splitlines()
            if line.strip()
        ]
        if not requests:
            raise ValueError(f"{prompt_file} contains no prompts.")
        return requests
    except ValueError as e:
        raise SystemExit(str(e)) from e


def build_messages(prompt: str, image_path: str | None = None) -> list[dict[str, Any]]:
    instruction = REWRITE_SYSTEM_PROMPT.strip()
    if not instruction:
        raise SystemExit("REWRITE_SYSTEM_PROMPT is blank.")

    if image_path is None:
        user_text = f"{instruction}\n\nInput text:\n{prompt}"
        return [{"role": "user", "content": user_text}]

    user_text = (
        f"{instruction}\n\n"
        "Reference image: the attached image is the exact first visible frame "
        "for this image-to-video rewrite.\n\n"
        f"Input text:\n{prompt}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def render_prompt(processor: Any, prompt: str, image_path: str | None = None) -> str:
    messages = build_messages(prompt, image_path)
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def resolve_dtype(dtype_name: str) -> str | Any:
    if dtype_name == "auto":
        return "auto"

    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def disable_broken_torchvision_imports() -> None:
    """Avoid importing torchvision for this text-only rewrite path.

    The tokenizer/config import path of multimodal models touches Transformers'
    image/video utilities. In environments where torchvision's compiled ops do
    not match the installed torch, importing it raises before the tokenizer is
    even loaded, so report it as unavailable here.
    """
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    def unavailable(*_: Any, **__: Any) -> bool:
        return False

    for name in ("is_torchvision_available", "is_torchvision_v2_available"):
        checker = getattr(import_utils, name, None)
        if hasattr(checker, "cache_clear"):
            checker.cache_clear()
        setattr(import_utils, name, unavailable)
        setattr(transformers_utils, name, unavailable)


def load_runtime(
    model_path: str,
    dtype_name: str,
    device: str,
    needs_vision: bool = False,
) -> tuple[Any, Any]:
    if needs_vision:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    else:
        disable_broken_torchvision_imports()

        from transformers import AutoModelForMultimodalLM, AutoTokenizer

        processor = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            extra_special_tokens={"video_token": "<|video|>"},
        )

    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_path,
        "local_files_only": True,
        "trust_remote_code": True,
        "device_map": {"": device},
        "dtype": resolve_dtype(dtype_name),
    }
    try:
        model = AutoModelForMultimodalLM.from_pretrained(**model_kwargs)
    except TypeError as e:
        if "dtype" not in str(e):
            raise
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForMultimodalLM.from_pretrained(**model_kwargs)

    model.eval()
    return processor, model


def build_model_inputs(processor: Any, prompt: str, image_path: str | None) -> Any:
    if image_path is None:
        tokenizer = getattr(processor, "tokenizer", processor)
        rendered_prompt = render_prompt(processor, prompt)
        return tokenizer(rendered_prompt, return_tensors="pt")

    messages = build_messages(prompt, image_path)
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        try:
            return processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError as e:
            raise SystemExit(
                "Image-backed rewriting requires a multimodal processor that supports "
                "apply_chat_template(..., tokenize=True, return_dict=True)."
            ) from e


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def decode_completion(processor: Any, output_ids: Any, input_length: int) -> str:
    completion_ids = output_ids[:, input_length:]
    decoder = getattr(processor, "tokenizer", processor)
    if hasattr(processor, "batch_decode"):
        try:
            return processor.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        except TypeError:
            return processor.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
    return decoder.decode(
        completion_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def rewrite_prompt(
    prompt: str,
    processor: Any,
    model: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    image_path: str | None = None,
) -> str:
    import torch

    inputs = build_model_inputs(processor, prompt, image_path)
    input_length = inputs["input_ids"].shape[-1]

    model_device = next(model.parameters()).device
    inputs = move_inputs_to_device(inputs, model_device)

    do_sample = temperature > 0.0
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    rewritten = decode_completion(processor, output_ids, input_length)
    if not rewritten:
        raise SystemExit("Model returned an empty rewrite.")
    return rewritten


def resolve_batch_gpu_ids(num_gpus: int, gpu_ids: str | None) -> list[int]:
    if gpu_ids is not None:
        try:
            ids = [int(part.strip()) for part in gpu_ids.split(",") if part.strip()]
        except ValueError as e:
            raise SystemExit("--gpu-ids must be a comma-separated list of integers.") from e
        if not ids:
            raise SystemExit("--gpu-ids did not contain any device ids.")
        if any(gpu_id < 0 for gpu_id in ids):
            raise SystemExit("--gpu-ids must be non-negative CUDA device ids.")
        if len(set(ids)) != len(ids):
            raise SystemExit("--gpu-ids must not contain duplicate device ids.")
        return ids

    if num_gpus <= 0:
        raise SystemExit("--num-gpus must be positive.")
    return list(range(int(num_gpus)))


def split_prompt_work(
    requests: list[RewriteRequest],
    gpu_ids: list[int],
) -> list[tuple[int, list[tuple[int, RewriteRequest]]]]:
    indexed_requests = list(enumerate(requests))
    chunk_size = max(1, math.ceil(len(indexed_requests) / len(gpu_ids)))

    work: list[tuple[int, list[tuple[int, RewriteRequest]]]] = []
    for worker_idx, gpu_id in enumerate(gpu_ids):
        start = worker_idx * chunk_size
        chunk = indexed_requests[start : start + chunk_size]
        if chunk:
            work.append((gpu_id, chunk))
    return work


def rewrite_prompt_chunk(
    gpu_id: int,
    indexed_requests: list[tuple[int, RewriteRequest]],
    model_path: str,
    dtype_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[tuple[int, RewriteRequest, str]]:
    import torch

    torch.cuda.set_device(gpu_id)
    needs_vision = any(request.image_path is not None for _, request in indexed_requests)
    processor, model = load_runtime(
        model_path,
        dtype_name,
        f"cuda:{gpu_id}",
        needs_vision=needs_vision,
    )
    results: list[tuple[int, RewriteRequest, str]] = []
    for prompt_idx, request in indexed_requests:
        refined_prompt = rewrite_prompt(
            prompt=request.prompt,
            image_path=request.image_path,
            processor=processor,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        results.append((prompt_idx, request, refined_prompt))
    return results


def rewrite_prompts_batch(
    requests: list[RewriteRequest],
    model_path: str,
    dtype_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_gpus: int,
    gpu_ids_arg: str | None,
) -> list[dict[str, str]]:
    if not requests:
        raise SystemExit("No prompts to rewrite.")

    gpu_ids = resolve_batch_gpu_ids(num_gpus, gpu_ids_arg)
    work = split_prompt_work(requests, gpu_ids)
    if not work:
        raise SystemExit("No batch work was created.")

    image_count = sum(1 for request in requests if request.image_path is not None)
    image_summary = f", {image_count} with reference images" if image_count else ""
    print(
        f"Rewriting {len(requests)} prompts{image_summary} across {len(work)} GPU workers: "
        f"{', '.join(str(gpu_id) for gpu_id, _ in work)}",
        file=sys.stderr,
    )

    batch_start = time.perf_counter()
    completed: list[tuple[int, RewriteRequest, str]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(work), mp_context=context) as executor:
        futures = [
            executor.submit(
                rewrite_prompt_chunk,
                gpu_id,
                chunk,
                model_path,
                dtype_name,
                max_new_tokens,
                temperature,
                top_p,
            )
            for gpu_id, chunk in work
        ]
        for future in as_completed(futures):
            chunk_results = future.result()
            completed.extend(chunk_results)
            elapsed = time.perf_counter() - batch_start
            print(
                f"Finished {len(completed)}/{len(requests)} prompts "
                f"in {format_elapsed_time(elapsed)}.",
                file=sys.stderr,
            )

    completed.sort(key=lambda item: item[0])
    results: list[dict[str, str]] = []
    for _, request, refined_prompt in completed:
        result = {"prompt": request.prompt, "refined_prompt": refined_prompt}
        if request.image_path is not None:
            result["image_path"] = request.image_path
        results.append(result)
    return results


def add_refined_prompt_field(item: Any, refined_prompt: str) -> dict[str, Any]:
    if isinstance(item, dict):
        result = dict(item)
    else:
        result = {"prompts": str(item)}
    result[REFINED_PROMPT_FIELD] = refined_prompt
    return result


def build_flat_json_result(result: dict[str, str]) -> dict[str, str]:
    row = {"prompts": result["prompt"], REFINED_PROMPT_FIELD: result["refined_prompt"]}
    image_path = result.get("image_path")
    if image_path:
        row["image_path"] = image_path
    return row


def build_preserved_json_results(
    prompt_file: Path,
    results: list[dict[str, str]],
) -> Any:
    try:
        payload = json.loads(prompt_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [
            build_flat_json_result(result)
            for result in results
        ]

    refined_prompts = [result["refined_prompt"] for result in results]

    if isinstance(payload, list) and len(payload) == len(refined_prompts):
        return [
            add_refined_prompt_field(item, refined_prompt)
            for item, refined_prompt in zip(payload, refined_prompts)
        ]

    if isinstance(payload, dict):
        for list_key in ("prompts", "entries", "data"):
            payload_items = payload.get(list_key)
            if isinstance(payload_items, list) and len(payload_items) == len(refined_prompts):
                output_payload = dict(payload)
                output_payload[list_key] = [
                    add_refined_prompt_field(item, refined_prompt)
                    for item, refined_prompt in zip(payload_items, refined_prompts)
                ]
                return output_payload

        if len(refined_prompts) == 1:
            output_payload = dict(payload)
            output_payload[REFINED_PROMPT_FIELD] = refined_prompts[0]
            return output_payload

    return [
        build_flat_json_result(result)
        for result in results
    ]


def write_json(output_json: Path, payload: Any) -> None:
    if output_json.parent != Path("."):
        output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_batch_results_json(
    output_json: Path,
    prompt_file: Path,
    results: list[dict[str, str]],
) -> None:
    write_json(output_json, build_preserved_json_results(prompt_file, results))


def main() -> int:
    script_start = time.perf_counter()
    args = parse_args()
    if args.prompt_file is not None and args.output_json is not None:
        if args.prompt is not None:
            raise SystemExit("Use either a positional prompt or --prompt-file, not both.")
        if args.gpu_id is not None:
            raise SystemExit("Use --gpu-ids for batch rewriting; --gpu-id is single-prompt only.")

        requests = read_batch_prompts(
            args.prompt_file,
            fallback_image_path=args.image_path,
        )
        results = rewrite_prompts_batch(
            requests=requests,
            model_path=args.model_path,
            dtype_name=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_gpus=args.num_gpus,
            gpu_ids_arg=args.gpu_ids,
        )
        write_batch_results_json(args.output_json, args.prompt_file, results)
        elapsed = time.perf_counter() - script_start
        print(
            f"Wrote {len(results)} rewrites to {args.output_json} "
            f"in {format_elapsed_time(elapsed)}"
        )
        return 0

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    prompt = read_prompt(args)
    image_path = resolve_image_path(args.image_path, "--image-path")
    device = resolve_device(args.device)
    processor, model = load_runtime(
        args.model_path,
        args.dtype,
        device,
        needs_vision=image_path is not None,
    )
    rewritten = rewrite_prompt(
        prompt=prompt,
        image_path=image_path,
        processor=processor,
        model=model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.output_json is not None:
        output_item = {"prompts": prompt, REFINED_PROMPT_FIELD: rewritten}
        if image_path is not None:
            output_item["image_path"] = image_path
        write_json(args.output_json, [output_item])
        elapsed = time.perf_counter() - script_start
        print(f"Wrote 1 rewrite to {args.output_json} in {format_elapsed_time(elapsed)}")
    else:
        print(rewritten)
        elapsed = time.perf_counter() - script_start
        print(f"Elapsed time: {format_elapsed_time(elapsed)}", file=sys.stderr)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

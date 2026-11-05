"""
Part 1: per-item processing (video).

Loads precomputed Wan 2.2 VAE latents (one local ``.safetensors`` per CSV row,
at the path in the row's ``latent_path`` column) and tokenizes captions.  Called from
ThreadPoolExecutor threads inside PackingDataset.  No pixel decoding or VAE
forward happens here — the latents are cached offline and consumed directly.

The caption-tokenization + token-budget validation is factored into
:func:`finalize_item` so the image processor (``image_item_processor.py``) can
reuse the exact same checks on its own latents.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Any

from loguru import logger
import torch
from safetensors.torch import load as safetensors_load

from pan.models.text_encoders import (
    build_cfg_empty_video_token_ids,
    build_templated_video_token_ids,
)
if TYPE_CHECKING:
    from .packing_dataset import DataloaderItemRef


@dataclass
class VideoItem:
    path: str
    caption: str
    prefix_tokens: int
    n_video_tokens: int
    ref: DataloaderItemRef
    modality: str = "video"
    # State needed to resume the (possibly interleaved video/image) stream right
    # after this item; attached by the packing stream, consumed at flush time.
    resume_cursor: Any = None


@dataclass(frozen=True)
class _TokenizedCaption:
    full_input_ids: torch.Tensor
    prefix_len: int
    full_length: int
    cropped_length: int


def decode_latent_bytes(raw: bytes, dtype: torch.dtype, where: str = "") -> torch.Tensor:
    """Decode a ``.safetensors`` blob holding a single ``latent`` tensor into a
    normalized ``[C, T, H, W]`` CPU tensor.

    Shared by the video loader (one ``.safetensors`` file per clip) and the
    image loader (one ``.safetensors`` member per tar sample) so both honor the
    exact same on-disk format. The latents are already normalized offline — no
    VAE forward and no mean/std here.
    """
    tensors = safetensors_load(raw)
    if "latent" not in tensors:
        raise KeyError(f"Expected a 'latent' tensor{f' in {where}' if where else ''}, got keys {list(tensors)}")
    latent = tensors["latent"]
    if latent.dim() == 5 and latent.shape[0] == 1:
        latent = latent[0]  # drop a leading batch dim if present
    if latent.dim() != 4:
        raise ValueError(f"Expected a [C, T, H, W] latent{f' in {where}' if where else ''}, got shape {tuple(latent.shape)}")
    return latent.to(dtype=dtype, memory_format=torch.contiguous_format)


def load_latent(latent_path: str, dtype: torch.dtype) -> torch.Tensor:
    """Read a precomputed, already-normalized VAE latent from local disk.

    The file is ``.safetensors`` with a single ``latent`` tensor (``[C, T, H, W]``,
    same convention as the image tars). Returns a CPU tensor ready to feed
    straight into the noise/target math.
    """
    with torch.cuda.nvtx.range("load latent"):
        with open(latent_path, "rb") as f:
            raw = f.read()
        return decode_latent_bytes(raw, dtype, where=latent_path)


def tokenize_caption(
    caption: str,
    tokenizer,
    template_cfg: dict,
    caption_segment_limit: int,
) -> _TokenizedCaption:
    tokenized = build_templated_video_token_ids(
        tokenizer=tokenizer,
        text=caption,
        template_cfg=template_cfg,
        max_caption_tokens=caption_segment_limit,
    )
    return tokenized


def build_empty_caption(
    tokenizer,
    template_cfg: dict,
    caption_segment_limit: int,
) -> _TokenizedCaption:
    return build_cfg_empty_video_token_ids(
        tokenizer=tokenizer,
        template_cfg=template_cfg,
        max_caption_tokens=caption_segment_limit,
    )


def finalize_item(
    *,
    latent: torch.Tensor,
    caption: str,
    ref: DataloaderItemRef,
    path: str,
    prefix_latent_steps: int,
    drop_caption: bool,
    modality: str,
    cfg_empty_caption: _TokenizedCaption,
    tokenizer,
    template_cfg: dict,
    caption_segment_limit: int,
    qwen_segment_limit: int,
    max_tokens: int,
    qwen_token_budget: int,
    packed_video_token_budget: int,
    packed_caption_token_budget: int,
    joint_segment_limit: int,
    tokens_per_frame: int,
) -> tuple[_TokenizedCaption, torch.Tensor, VideoItem] | None:
    """Build the :class:`VideoItem`, tokenize the caption, and apply the shared
    token-budget checks. Returns ``None`` if any budget is exceeded.

    Shared by the video and image processors so both enforce identical limits;
    the caller is responsible for choosing ``prefix_latent_steps`` (image
    samples are single-frame and always pass ``0``) and ``drop_caption``.
    """
    latent_steps = int(latent.shape[1])
    item = VideoItem(
        path=path,
        caption=caption,
        prefix_tokens=prefix_latent_steps * tokens_per_frame,
        n_video_tokens=latent_steps * tokens_per_frame,
        ref=ref,
        modality=modality,
    )

    tokenized_caption = cfg_empty_caption if drop_caption else tokenize_caption(
        item.caption, tokenizer, template_cfg, caption_segment_limit,
    )
    caption_length = tokenized_caption.cropped_length
    if item.n_video_tokens + caption_length > max_tokens:
        logger.warning(f"Item needs {item.n_video_tokens + caption_length} tokens > {max_tokens=}, skipping")
        return None
    if tokenized_caption.full_length > qwen_token_budget:
        logger.warning(f"Item needs {tokenized_caption.full_length} full templated text tokens > {qwen_token_budget=}, skipping")
        return None
    if item.n_video_tokens > packed_video_token_budget:
        logger.warning(f"Item needs {item.n_video_tokens} video tokens > {packed_video_token_budget=}, skipping")
        return None
    if caption_length > packed_caption_token_budget:
        logger.warning(f"Item needs {caption_length} caption tokens > {packed_caption_token_budget=}, skipping")
        return None
    if tokenized_caption.full_length > qwen_segment_limit:
        logger.warning(f"Item needs {tokenized_caption.full_length} full templated text tokens > {qwen_segment_limit=}, skipping")
        return None
    if item.n_video_tokens + caption_length > joint_segment_limit:
        logger.warning(f"Item needs {item.n_video_tokens + caption_length} joint tokens > {joint_segment_limit=}, skipping")
        return None
    if caption_length > caption_segment_limit:
        logger.warning(f"Item needs {caption_length} caption tokens > {caption_segment_limit=}, skipping")
        return None

    return tokenized_caption, latent, item


def load_latent_row(
    row_and_ref: tuple[dict[str, Any], DataloaderItemRef],
    *,
    seed: int,
    cfg_empty_caption: _TokenizedCaption,
    tokenizer,
    template_cfg: dict,
    caption_segment_limit: int,
    qwen_segment_limit: int,
    dtype: torch.dtype,
    max_tokens: int,
    qwen_token_budget: int,
    packed_video_token_budget: int,
    packed_caption_token_budget: int,
    joint_segment_limit: int,
    patch_hw: tuple[int, int],
    video_cond_prob: float,
    image_cond_prob: float,
    cfg_drop_prob: float,
) -> tuple[_TokenizedCaption, torch.Tensor, VideoItem] | None:
    row, ref = row_and_ref
    try:
        latent_path = row["latent_path"].strip()
        if not latent_path:
            raise ValueError("empty latent_path")
        # The caption column is used verbatim as the text input (structured JSON
        # captions are consumed as-is).
        caption = row["caption"]
    except Exception as exc:
        logger.warning(f"Skipping row {ref.local_index} in csv[{ref.csv_index}]: {exc}")
        return None

    if not isinstance(caption, str) or not caption.strip():
        logger.warning(f"Skipping row {ref.local_index} in csv[{ref.csv_index}]: empty/invalid caption")
        return None
    # Strip lone UTF-16 surrogates (broken emoji from data prep): valid in a Python
    # str but not encodable UTF-8, and the fast tokenizer's encode_batch rejects them
    # with "TextEncodeInput must be Union[...]", which crashes the whole training job.
    caption = caption.encode("utf-8", "ignore").decode("utf-8")

    try:
        latent = load_latent(latent_path, dtype)  # [C, T, H, W]
    except Exception as exc:
        logger.warning(f"Failed to load latent for row {ref.local_index} in csv[{ref.csv_index}] ({latent_path!r}): {exc}")
        return None

    # Latent temporal length is the ground truth for token counts; the cached
    # latent was produced offline by the VAE, so derive everything from it.
    latent_steps = int(latent.shape[1])

    ph, pw = int(patch_hw[0]), int(patch_hw[1])
    h, w = int(latent.shape[2]), int(latent.shape[3])
    if h % ph != 0 or w % pw != 0:
        logger.warning(
            f"Skipping row {ref.local_index} in csv[{ref.csv_index}]: latent grid {(h, w)} "
            f"not divisible by patch {(ph, pw)}"
        )
        return None
    tokens_per_frame = (h // ph) * (w // pw)

    rng = random.Random(hash((seed, ref.epoch_index, ref.global_index, ref.local_index, ref.local_file_pos)))
    u = rng.random()
    if latent_steps < 2:
        prefix_latent_steps = 0
    elif u < video_cond_prob:
        prefix_latent_steps = rng.randint(1, max(1, latent_steps // 2))
    elif u < video_cond_prob + image_cond_prob:
        prefix_latent_steps = 1
    else:
        prefix_latent_steps = 0

    drop_caption = rng.random() < cfg_drop_prob
    return finalize_item(
        latent=latent,
        caption=caption,
        ref=ref,
        path=latent_path,
        prefix_latent_steps=prefix_latent_steps,
        drop_caption=drop_caption,
        modality="video",
        cfg_empty_caption=cfg_empty_caption,
        tokenizer=tokenizer,
        template_cfg=template_cfg,
        caption_segment_limit=caption_segment_limit,
        qwen_segment_limit=qwen_segment_limit,
        max_tokens=max_tokens,
        qwen_token_budget=qwen_token_budget,
        packed_video_token_budget=packed_video_token_budget,
        packed_caption_token_budget=packed_caption_token_budget,
        joint_segment_limit=joint_segment_limit,
        tokens_per_frame=tokens_per_frame,
    )

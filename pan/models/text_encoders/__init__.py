"""Qwen3.5 text encoder shared by training and inference.

Training (``train.py`` / ``dataset/packed_dataloader.py``) runs the packed
captions through :func:`load_qwen_text_encoder`; inference wraps the very same
module in :class:`QwenTextEncoder`, which also owns the prompt template and the
crop that removes the template prefix. Keeping both sides on one loader is what
guarantees the inference embeddings match what the DiT saw in training.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import os

import loguru
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

__all__ = [
    "QWEN35_TEXT_ENCODER_ID",
    "PROMPT_TEMPLATE",
    "TemplatedVideoTokenIds",
    "build_templated_video_token_ids",
    "build_cfg_empty_video_token_ids",
    "load_qwen_text_encoder",
    "QwenTextEncoder",
]

# Hardcoded to Qwen3.5 (heterogeneous full/linear-attention layers; packing
# requires threading seq_idx to reset the GatedDeltaNet conv per sequence).
QWEN35_TEXT_ENCODER_ID = "Qwen/Qwen3.5-9B"

# The tokenizer is called from dataloader worker threads; its own thread pool only adds fork warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Chat template every caption is wrapped in before Qwen encodes it. The caption
# replaces "{}" in the user turn; only the tokens after the user-turn header are
# kept for the video backbone (see ``prefix_len``).
VIDEO_CAPTION_CHAT = (
    {"role": "system", "content": "You are a helpful assistant. Generate the video with the following instructions."},
    {"role": "user", "content": "{}"},
)

# ``prefix_len`` is the number of template tokens before the caption; -1 means
# "measure it from the tokenizer on first use".
PROMPT_TEMPLATE = {
    "video": {"template": VIDEO_CAPTION_CHAT, "prefix_len": -1},
}

_USER_TURN_HEADER = "<|im_start|>user\n"


# ---------------------------------------------------------------------------
# Prompt templating
# ---------------------------------------------------------------------------


@dataclass
class TemplatedVideoTokenIds:
    # Full templated prompt token ids that Qwen sees during training/inference.
    full_input_ids: torch.LongTensor
    # Caption-only slice after the template prefix is removed.
    cropped_input_ids: torch.LongTensor
    prefix_len: int
    full_length: int
    cropped_length: int


def _chat_with_caption(template, caption):
    """Copy of the chat ``template`` with ``caption`` substituted for ``{}``.

    An empty caption is replaced by a single space so the user turn always
    carries at least one token.
    """
    caption = caption or " "
    return [{**turn, "content": turn["content"].format(caption)} for turn in template]


def _tokenize_chat(tokenizer, chat, max_length, padding):
    """Token ids of ``chat`` rendered by the tokenizer's chat template, with the assistant turn opened."""
    return tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=True, return_dict=True,
        truncation=True, max_length=max_length, padding=padding, return_tensors="pt",
    )


def _caption_offset(tokenizer, input_ids):
    """Index of the first caption token: the position right after the user-turn header."""
    header = tokenizer(_USER_TURN_HEADER, add_special_tokens=False)["input_ids"]
    ids = input_ids.tolist()
    for pos in range(len(ids) - len(header) + 1):
        if ids[pos:pos + len(header)] == header:
            return pos + len(header)
    raise ValueError(f"User-turn header {_USER_TURN_HEADER!r} not found in the templated prompt.")


def _caption_prefix_len(tokenizer, template_cfg, sample_text):
    """Number of template tokens in front of the caption; measured on first use and cached in ``template_cfg``."""
    if template_cfg.get("prefix_len", -1) == -1:
        probe = _tokenize_chat(tokenizer, _chat_with_caption(template_cfg["template"], sample_text),
                               max_length=256, padding="max_length")
        template_cfg["prefix_len"] = _caption_offset(tokenizer, probe["input_ids"][0])
    return template_cfg["prefix_len"]


def build_templated_video_token_ids(tokenizer, text, template_cfg, max_caption_tokens):
    """Tokenize ``text`` inside the chat template. Qwen sees the whole templated prompt; the video
    backbone only gets the caption span behind the ``prefix_len`` template tokens."""
    prefix_len = _caption_prefix_len(tokenizer, template_cfg, sample_text=text)
    budget = max(prefix_len, 0) + max_caption_tokens  # the template prefix does not count against the caption
    chat = _chat_with_caption(template_cfg["template"], text)
    ids = _tokenize_chat(tokenizer, chat, max_length=budget, padding=False)["input_ids"][0]
    if ids.numel() > prefix_len + max_caption_tokens:
        loguru.logger.warning(
            f"Caption is {ids.numel() - prefix_len} tokens, more than {max_caption_tokens=}; it is truncated."
        )
    caption_ids = ids[prefix_len:]
    return TemplatedVideoTokenIds(
        full_input_ids=ids,
        cropped_input_ids=caption_ids,
        prefix_len=prefix_len,
        full_length=ids.numel(),
        cropped_length=caption_ids.numel(),
    )


def build_cfg_empty_video_token_ids(tokenizer, template_cfg, max_caption_tokens):
    return build_templated_video_token_ids(
        tokenizer=tokenizer,
        text="",
        template_cfg=template_cfg,
        max_caption_tokens=max_caption_tokens,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_qwen_text_encoder(
    hidden_state_skip_layer: int = 2,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_4",
) -> nn.Module:
    """The frozen Qwen3.5 language model used as the text encoder.

    The last ``hidden_state_skip_layer`` decoder layers are dropped at load time
    and the final RMSNorm is replaced by identity, so ``last_hidden_state`` is the
    raw output of layer ``num_hidden_layers - hidden_state_skip_layer``. The
    vision tower and LM head are discarded. Used verbatim by training and
    inference, so both see the same features.
    """
    hidden_state_skip_layer = int(hidden_state_skip_layer)
    if hidden_state_skip_layer < 0:
        raise ValueError(f"hidden_state_skip_layer must be >= 0, got {hidden_state_skip_layer}")
    full_config = AutoConfig.from_pretrained(QWEN35_TEXT_ENCODER_ID)
    te_config = full_config.text_config
    if hidden_state_skip_layer >= te_config.num_hidden_layers:
        raise ValueError(
            f"hidden_state_skip_layer={hidden_state_skip_layer} must be smaller than "
            f"Qwen3.5 num_hidden_layers={te_config.num_hidden_layers}."
        )
    te_config.num_hidden_layers -= hidden_state_skip_layer
    loguru.logger.info(
        f"Loading {QWEN35_TEXT_ENCODER_ID} text encoder: hidden_state_skip_layer={hidden_state_skip_layer}, "
        f"apply_final_norm=False, dtype={dtype}, attn={attn_implementation}"
    )
    full_model = AutoModelForImageTextToText.from_pretrained(
        QWEN35_TEXT_ENCODER_ID,
        attn_implementation=attn_implementation,
        dtype=dtype,
        device_map=str(device),
        config=full_config,
    )
    text_encoder = full_model.model.language_model
    del full_model.model.visual, full_model.lm_head
    if (
        not hasattr(te_config, "layer_types")
        or te_config.layer_types is None
        or all(lt == te_config.layer_types[0] for lt in te_config.layer_types)
    ):
        for module in text_encoder.modules():
            if hasattr(module, "layer_idx"):
                del module.layer_idx
    text_encoder.norm = nn.Identity()
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    return text_encoder


class QwenTextEncoder(nn.Module):
    """Inference-side wrapper: template + tokenize + Qwen forward + prefix crop.

    Mirrors the packed training path: every caption is wrapped in the video chat
    template, Qwen runs on the full templated prompt, and only the rows from the
    user-caption start (``prefix_len``) onward are handed to the DiT.
    """

    def __init__(
        self,
        hidden_state_skip_layer: int = 2,
        max_caption_tokens: int = 1000,
        device: Optional[Union[str, torch.device]] = None,
        template_cfg: Optional[dict] = None,
        model: Optional[nn.Module] = None,
        tokenizer=None,
    ):
        super().__init__()
        max_caption_tokens = int(max_caption_tokens)
        if max_caption_tokens <= 0:
            raise ValueError(f"max_caption_tokens must be > 0, got {max_caption_tokens}")
        self.hidden_state_skip_layer = int(hidden_state_skip_layer)
        self.max_caption_tokens = max_caption_tokens
        self.template_cfg = deepcopy(template_cfg or PROMPT_TEMPLATE["video"])
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            QWEN35_TEXT_ENCODER_ID, padding_side="right",
        )
        self.model = model if model is not None else load_qwen_text_encoder(
            hidden_state_skip_layer, device=device or "cuda",
        )

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def tokenize(self, texts: Sequence[str]) -> List[TemplatedVideoTokenIds]:
        return [
            build_templated_video_token_ids(self.tokenizer, text, self.template_cfg, self.max_caption_tokens)
            for text in texts
        ]

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, Sequence[str]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of prompts.

        Returns ``(embeds, mask)``: ``embeds`` is ``[B, L, D]`` holding the
        hidden states of the caption span (template prefix already cropped),
        right-padded to the longest caption in the batch; ``mask`` is ``[B, L]``
        long with 1 on real caption rows.
        """
        if isinstance(texts, str):
            texts = [texts]
        device = torch.device(device) if device is not None else self.device
        tokenized = self.tokenize(texts)
        prefix_len = tokenized[0].prefix_len
        full_len = max(t.full_length for t in tokenized)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = torch.full((len(tokenized), full_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(tokenized), full_len), dtype=torch.long)
        for i, t in enumerate(tokenized):
            input_ids[i, : t.full_length] = t.full_input_ids
            attention_mask[i, : t.full_length] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        # The template prefix is identical for every prompt, so one slice crops
        # all of them; padding beyond each prompt's full_length is already masked.
        return hidden[:, prefix_len:].to(self.dtype), attention_mask[:, prefix_len:]

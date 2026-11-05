"""
Part 2: virtual bin-packing and CPU metadata assembly.

Reads CSV rows, downloads and decodes videos via a thread pool, bin-packs
decoded items by token budget, and assembles CPU-side PackedPlan objects
(including RoPE cos/sin tensors).  No GPU work is done here.

Token counts and RoPE are derived per-sample from each latent's own
``[C, T, H, W]`` shape, so samples of different spatial size (e.g. variable
image resolutions interleaved with fixed-size video) pack correctly.
"""

import csv
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
import itertools
import json
import logging
import random
from typing import Callable, Iterator, NamedTuple, Optional

import torch
from torch.utils.data import IterableDataset

from transformers import AutoTokenizer

from pan.models.transformers.modules.rope import rope_tables_for_positions

from .video_item_processor import (
    _TokenizedCaption,
    VideoItem,
    build_empty_caption,
    load_latent_row,
)
from .image_item_processor import (
    ImageSample,
    ImageStreamCursor,
    ImageTarStreamer,
    load_image_latent_from_ref,
    load_image_sample,
)

logger = logging.getLogger(__name__)

# Integer salt for the video/image mix-draw RNG. Kept an int (not a string) so
# ``hash(...)`` is stable across processes regardless of PYTHONHASHSEED,
# matching the all-int seeds used by the item processors.
_MIX_DRAW_SALT = 0x6D1809A2


# ---------------------------------------------------------------------------
# Shared checkpoint types
# ---------------------------------------------------------------------------


class _LineTracker:
    """Wraps a file, remembering the byte offset of each physical line.

    A logical CSV row may span several physical lines (quoted fields with
    embedded newlines), so the byte offset of a row is the offset of its
    *first* physical line. Before handing each line to the CSV reader we record
    ``f.tell()`` keyed by a 1-based ``line_num``. Callers capture ``line_num``
    just before ``next(reader)`` and recover the row's start offset with
    ``start_pos`` once the (possibly multi-line) row has been read.

    The counter is the tracker's own (not ``csv.reader.line_num``) so it can be
    restored on resume via :meth:`seek`, where the reader's internal counter
    cannot. Offsets are kept in a bounded LRU map; a single row's physical-line
    span stays far below ``_MAX_CACHE``, so the entry is always still present
    when looked up immediately after the row is read.
    """

    _MAX_CACHE = 1024

    def __init__(self, f):
        self._f = f
        self.line_num: int = 1  # next 1-based line number the reader will see
        self.row_start: "OrderedDict[int, int]" = OrderedDict()

    def __iter__(self):
        while True:
            self.row_start[self.line_num] = self._f.tell()
            if len(self.row_start) >= self._MAX_CACHE:
                self.row_start.popitem(last=False)
            self.line_num += 1
            line = self._f.readline()
            if not line:
                return
            yield line

    def seek(self, seek_pos: int, line_num: int) -> None:
        """Reposition the file and restore the logical line counter on resume."""
        self._f.seek(seek_pos)
        self.line_num = line_num

    def start_pos(self, line_num: int) -> int:
        return self.row_start[line_num]


@dataclass
class DataloaderItemRef:
    """JSON-serializable pointer to a single data item (video CSV row or image
    tar member). Video uses the CSV locator fields; image uses the tar locator
    fields. ``modality`` selects which set is meaningful; the other defaults to
    sentinels.
    """
    global_index: int
    modality: str = "video"
    # --- video CSV locator ---
    csv_index: int = -1
    local_index: int = -1
    local_file_pos: int = -1
    # 1-based physical line where the row begins; paired with ``local_file_pos``
    # so a resumed stream can restore the line tracker exactly (see _LineTracker).
    local_line_num: int = -1
    epoch_index: int = 0
    # --- image tar locator (range-read of the .safetensors member) ---
    shard_index: int = -1
    sample_index: int = -1
    cycle: int = 0
    shard_url: str = ""
    tar_offset: int = -1
    tar_size: int = -1
    caption: str = ""


@dataclass
class DataloaderState:
    """JSON-serializable dataloader checkpoint state.

    ``next_stream_ref`` is the video CSV cursor (last consumed video row);
    ``image_cursor`` is the analogous image-stream cursor; ``draw_count`` is the
    number of mix draws made, so a resumed run reproduces the exact
    interleaving. ``buffer_items`` holds the in-flight packed-buffer contents
    (mixed modality), each reloadable from its ref.
    """
    buffer_items: list[list[DataloaderItemRef]]
    next_stream_ref: Optional[DataloaderItemRef]
    image_cursor: Optional[ImageStreamCursor] = None
    draw_count: int = 0

    def to_json(self) -> str:
        return json.dumps({
            "buffer_items": [[asdict(r) for r in buf] for buf in self.buffer_items],
            "next_stream_ref": asdict(self.next_stream_ref) if self.next_stream_ref is not None else None,
            "image_cursor": asdict(self.image_cursor) if self.image_cursor is not None else None,
            "draw_count": self.draw_count,
        })

    def reset_video_stream(self) -> "DataloaderState":
        """Return a copy with the video-CSV resume position cleared while the
        image-tar cursor is left untouched.

        Used to resume a checkpoint (model/optimizer/step) but start the video
        stream fresh from a new/different CSV: the old ``next_stream_ref`` and any
        in-flight video buffer refs point at rows that no longer exist in the new
        CSV, so they are dropped. Image refs (``modality == "image"``) stay in the
        buffers and ``image_cursor``/``draw_count`` are preserved, so image-tar
        resume behaves exactly as before.
        """
        return DataloaderState(
            buffer_items=[[r for r in buf if r.modality == "image"] for buf in self.buffer_items],
            next_stream_ref=None,
            image_cursor=self.image_cursor,
            draw_count=self.draw_count,
        )

    @classmethod
    def from_json(cls, s: str) -> "DataloaderState":
        d = json.loads(s)
        ic = d.get("image_cursor")
        nsr = d.get("next_stream_ref")
        return cls(
            buffer_items=[[DataloaderItemRef(**r) for r in buf] for buf in d["buffer_items"]],
            next_stream_ref=DataloaderItemRef(**nsr) if nsr is not None else None,
            image_cursor=ImageStreamCursor(**ic) if ic is not None else None,
            draw_count=d.get("draw_count", 0),
        )


@dataclass
class _StreamResume:
    """Snapshot of stream cursors *after* producing a given item; attached to
    that item so the flush can persist exact resume state."""
    video_next_ref: Optional[DataloaderItemRef]
    image_cursor: Optional[ImageStreamCursor]
    draw_count: int


# ---------------------------------------------------------------------------
# PackedPlan: bridge from Part 2 (packing) to Part 3 (GPU assembly)
# ---------------------------------------------------------------------------


class PackedPlan(NamedTuple):
    """CPU-only output of the packing step, ready for GPU assembly."""
    # Precomputed, already-normalized VAE latents; one [C, T, H, W] CPU tensor per video.
    decoded_frames: list[torch.Tensor]
    # Per-video: sampled conditioning prefix length, independent of how conditioning is represented.
    condition_prefix_tokens: list[int]
    # Text encoder (Qwen) inputs.
    qwen_input_ids: torch.Tensor            # [concat_qwen_tokens] long
    qwen_position_ids: torch.Tensor         # [concat_qwen_tokens] long
    qwen_cu_seqlens: torch.Tensor           # [num_sequences + 1] int32
    qwen_seq_idx: torch.Tensor              # [concat_qwen_tokens] int32, resets causal_conv1d per sequence (Qwen3.5 GatedDeltaNet)
    caption_row_indices: torch.Tensor       # [concat_caption_tokens] long
    # Attention mask boundaries.
    video_cu_seqlens: torch.Tensor          # [num_sequences + 1] int32
    caption_cu_seqlens: torch.Tensor        # [num_sequences + 1] int32
    joint_cu_seqlens: torch.Tensor          # [num_sequences + 1] int32, one video+caption segment per sample
    # CPU RoPE tensors; GPU copy stays in Part 3.
    freqs_cos: torch.Tensor                 # [joint_L, 1, rope_dim]
    freqs_sin: torch.Tensor                 # [joint_L, 1, rope_dim]
    # Checkpoint sidecar.
    dataloader_state: DataloaderState
    # Number of image-modality sequences in this batch (videos = seq_count - this).
    num_image_sequences: int


# ---------------------------------------------------------------------------
# Packing utilities
# ---------------------------------------------------------------------------


def seq_idx_from_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Per-token sequence index (int32), derived from cu_seqlens."""
    counts = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.int32), counts.to(torch.int64)
    )


def _rope_dim_list_from_callable(rope_embed_fn: Callable) -> tuple[int, ...]:
    args = getattr(rope_embed_fn, "args", ())
    if not args:
        return ()
    return tuple(int(v) for v in args[0])


def _rope_kwargs_from_callable(rope_embed_fn: Callable) -> dict:
    return dict(getattr(rope_embed_fn, "keywords", {}) or {})


def _build_joint_rope_position_ids(
    video_t: int,
    video_h: int,
    video_w: int,
    text_len: int,
) -> torch.Tensor:
    text_pos = torch.zeros((text_len, 4), dtype=torch.float32)
    if text_len > 0:
        text_pos[:, 3] = torch.arange(text_len, dtype=torch.float32)
    video_seq_start = float(text_pos[-1, 3].item() + 1) if text_len > 0 else 0.0

    t = torch.arange(video_t, dtype=torch.float32)
    h = torch.arange(video_h, dtype=torch.float32)
    w = torch.arange(video_w, dtype=torch.float32)
    grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")
    video_pos = torch.stack(
        (grid_t.reshape(-1), grid_h.reshape(-1), grid_w.reshape(-1)), dim=1
    )
    video_seq = torch.full((video_pos.size(0), 1), video_seq_start, dtype=torch.float32)
    video_pos = torch.cat((video_pos, video_seq), dim=1)

    return torch.cat((video_pos, text_pos), dim=0)


# ---------------------------------------------------------------------------
# Virtual packing buffer (CPU only)
# ---------------------------------------------------------------------------


@dataclass
class _BufferedSample:
    item: VideoItem
    tokenized_caption: _TokenizedCaption
    decoded_frames: torch.Tensor


@dataclass
class _VirtualBuffer:
    max_tokens: int
    qwen_token_budget: int
    packed_video_token_budget: int
    packed_caption_token_budget: int
    max_num_seq: int
    used_tokens: int = 0
    used_qwen_tokens: int = 0
    used_video_tokens: int = 0
    used_caption_tokens: int = 0
    samples: list[_BufferedSample] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.max_tokens - self.used_tokens

    @property
    def remaining_qwen_tokens(self) -> int:
        return self.qwen_token_budget - self.used_qwen_tokens

    @property
    def remaining_video_tokens(self) -> int:
        return self.packed_video_token_budget - self.used_video_tokens

    @property
    def remaining_caption_tokens(self) -> int:
        return self.packed_caption_token_budget - self.used_caption_tokens

    @property
    def remaining_seqs(self) -> int:
        return self.max_num_seq - len(self.samples)

    def can_fit(self, sample: _BufferedSample) -> bool:
        caption_tokens = sample.tokenized_caption.cropped_length
        return (
            sample.item.n_video_tokens + caption_tokens <= self.remaining
            and sample.tokenized_caption.full_length <= self.remaining_qwen_tokens
            and sample.item.n_video_tokens <= self.remaining_video_tokens
            and caption_tokens <= self.remaining_caption_tokens
            and self.remaining_seqs > 0
        )

    def add(self, sample: _BufferedSample):
        caption_tokens = sample.tokenized_caption.cropped_length
        self.samples.append(sample)
        self.used_tokens += sample.item.n_video_tokens + caption_tokens
        self.used_qwen_tokens += sample.tokenized_caption.full_length
        self.used_video_tokens += sample.item.n_video_tokens
        self.used_caption_tokens += caption_tokens

    def flush(self) -> list[_BufferedSample]:
        out_samples = list(self.samples)
        self.samples.clear()
        self.used_tokens = 0
        self.used_qwen_tokens = 0
        self.used_video_tokens = 0
        self.used_caption_tokens = 0
        return out_samples


# ---------------------------------------------------------------------------
# Plan assembly (CPU only) — includes RoPE cos/sin construction
# ---------------------------------------------------------------------------


def _build_packed_plan(
    samples: list[_BufferedSample],
    dataloader_state: DataloaderState,
    rope_embed_fn: Callable[[tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]],
    patch_size: tuple[int, int, int],
) -> PackedPlan:
    """
    Assemble all CPU metadata from a flushed packing buffer, including RoPE.
    Runs in the packing worker — no CUDA.

    Token counts and RoPE are derived per-sample from each latent's own
    ``[C, T, H, W]`` shape, so samples of different spatial size (e.g. variable
    image resolutions interleaved with fixed-size video) pack correctly.
    """
    pt, ph, pw = patch_size
    if pt != 1:
        raise ValueError(f"Packed i2v requires patch_size[0] == 1, got {patch_size}.")

    def _video_tokens(s: "_BufferedSample") -> int:
        _, t, h, w = s.decoded_frames.shape
        return (int(t) // pt) * (int(h) // ph) * (int(w) // pw)

    condition_prefix_tokens: list[int] = []
    all_qwen_input_ids: list[torch.Tensor] = []
    all_qwen_position_ids: list[torch.Tensor] = []
    all_caption_row_indices: list[torch.Tensor] = []

    video_lengths: list[int] = []
    caption_token_lengths: list[int] = []
    qwen_token_lengths: list[int] = []
    qwen_token_offset = 0

    for sample in samples:
        condition_prefix_tokens.append(sample.item.prefix_tokens)

        vl = _video_tokens(sample)
        tl = sample.tokenized_caption.cropped_length
        full_tl = sample.tokenized_caption.full_length
        video_lengths.append(vl)
        caption_token_lengths.append(tl)
        qwen_token_lengths.append(full_tl)

        all_qwen_input_ids.append(sample.tokenized_caption.full_input_ids)
        all_qwen_position_ids.append(torch.arange(full_tl, dtype=torch.long))
        if tl > 0:
            all_caption_row_indices.append(
                torch.arange(
                    sample.tokenized_caption.prefix_len,
                    sample.tokenized_caption.full_length,
                    dtype=torch.long,
                ) + qwen_token_offset
            )
        qwen_token_offset += full_tl

    video_cu_seqlens = torch.zeros(len(video_lengths) + 1, dtype=torch.int32)
    caption_cu_seqlens = torch.zeros(len(caption_token_lengths) + 1, dtype=torch.int32)
    joint_cu_seqlens = torch.zeros(len(video_lengths) + 1, dtype=torch.int32)

    for i, (cl, vl) in enumerate(zip(caption_token_lengths, video_lengths)):
        caption_cu_seqlens[i + 1] = caption_cu_seqlens[i] + cl
        video_cu_seqlens[i + 1] = video_cu_seqlens[i] + vl
        joint_cu_seqlens[i + 1] = joint_cu_seqlens[i] + vl + cl

    qwen_cu_seqlens = torch.zeros(len(qwen_token_lengths) + 1, dtype=torch.int32)
    for i, tl in enumerate(qwen_token_lengths):
        qwen_cu_seqlens[i + 1] = qwen_cu_seqlens[i] + tl

    qwen_seq_idx = seq_idx_from_cu_seqlens(qwen_cu_seqlens)

    qwen_input_ids = torch.cat(all_qwen_input_ids) if all_qwen_input_ids else torch.empty(0, dtype=torch.long)
    qwen_position_ids = torch.cat(all_qwen_position_ids) if all_qwen_position_ids else torch.empty(0, dtype=torch.long)
    caption_row_indices = (
        torch.cat(all_caption_row_indices) if all_caption_row_indices else torch.empty(0, dtype=torch.long)
    )

    # --- CPU RoPE construction ------------------------------------------------
    # Latents are precomputed [C, T, H, W]; T/H/W are already in latent space.
    # Each sample uses its own spatial grid so variable resolutions are supported.
    rope_dim_list = _rope_dim_list_from_callable(rope_embed_fn)
    rope_ndim = len(rope_dim_list) if rope_dim_list else 3
    rope_kwargs = _rope_kwargs_from_callable(rope_embed_fn)

    # Per-sample video and text freqs kept separate, then interleaved below.
    video_cos: list[torch.Tensor] = []
    video_sin: list[torch.Tensor] = []
    text_cos: list[torch.Tensor] = []
    text_sin: list[torch.Tensor] = []
    for i, sample in enumerate(samples):
        _, t, lat_h, lat_w = sample.decoded_frames.shape
        input_ids_len = int(caption_cu_seqlens[i + 1].item() - caption_cu_seqlens[i].item())
        rope_sizes = (int(t) // pt, int(lat_h) // ph, int(lat_w) // pw)
        if rope_ndim == 3:
            cos, sin = rope_embed_fn(rope_sizes)
            tcos = torch.ones((input_ids_len, cos.size(1)), dtype=cos.dtype)
            tsin = torch.zeros((input_ids_len, sin.size(1)), dtype=sin.dtype)
        elif rope_ndim == 4:
            position_ids = _build_joint_rope_position_ids(*rope_sizes, input_ids_len)
            cos_all, sin_all = rope_tables_for_positions(
                rope_dim_list,
                position_ids,
                **rope_kwargs,
            )
            vl = video_lengths[i]
            cos, tcos = cos_all[:vl], cos_all[vl:]
            sin, tsin = sin_all[:vl], sin_all[vl:]
        else:
            raise ValueError(f"Packed RoPE supports 3 or 4 axes, got {rope_ndim}")
        video_cos.append(cos)
        video_sin.append(sin)
        text_cos.append(tcos)
        text_sin.append(tsin)

    # Interleave order [v1, t1, v2, t2, ...] -- what flash attention's
    # joint_cu_seqlens layout expects.
    cos_s = [x for pair in zip(video_cos, text_cos) for x in pair]
    sin_s = [x for pair in zip(video_sin, text_sin) for x in pair]

    freqs_cos = torch.cat(cos_s, dim=0).unsqueeze(1)  # [joint_L, 1, rope_dim]
    freqs_sin = torch.cat(sin_s, dim=0).unsqueeze(1)

    return PackedPlan(
        decoded_frames=[s.decoded_frames for s in samples],
        condition_prefix_tokens=condition_prefix_tokens,
        qwen_input_ids=qwen_input_ids,
        qwen_position_ids=qwen_position_ids,
        qwen_cu_seqlens=qwen_cu_seqlens,
        qwen_seq_idx=qwen_seq_idx,
        caption_row_indices=caption_row_indices,
        video_cu_seqlens=video_cu_seqlens,
        caption_cu_seqlens=caption_cu_seqlens,
        joint_cu_seqlens=joint_cu_seqlens,
        freqs_cos=freqs_cos,
        freqs_sin=freqs_sin,
        dataloader_state=dataloader_state,
        num_image_sequences=sum(1 for s in samples if s.item.modality == "image"),
    )


# ---------------------------------------------------------------------------
# IterableDataset: virtual packing + video decoding in worker
# ---------------------------------------------------------------------------


class PackingDataset(IterableDataset):
    """
    Streams CSV rows, decodes videos via a thread pool, and bin-packs
    decoded items into fixed-token-budget :class:`PackedPlan` objects.

    Designed to run inside a DataLoader with ``num_workers=1``.  The internal
    thread pool (``num_decode_workers``) handles concurrent latent reads and
    caption tokenization within that single worker.

    Args:
        num_decode_workers: Threads used concurrently for latent reads / tokenization.
        image_tar_shards: Resolved local ``.tar`` image-latent shards. When set
            (with ``image_sample_prob > 0``) images are interleaved into the
            video stream at that per-item probability.
    """

    def __init__(
        self,
        csv_paths: list[str],
        text_tokenizer_path: str,
        text_template_video_cfg: dict,
        rope_embed_fn: Callable[[tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]],
        patch_size: tuple[int, int, int],
        resize: tuple[int, int],
        dtype: torch.dtype,
        num_buffers: int = 4,
        max_tokens: int = 8192,
        qwen_token_budget: int = 8192,
        packed_video_token_budget: int = 8192,
        packed_caption_token_budget: int = 8192,
        random_seed: int = 42,
        cfg_drop_prob: float = 0.0,
        image_cond_prob: float = 0.0,
        video_cond_prob: float = 0.0,
        caption_segment_limit: int = 256,
        qwen_segment_limit: int = 256,
        max_num_seq: int = 16,
        num_decode_workers: int = 4,
        joint_segment_limit: int = 0,
        conditioning_schema: str = "token_concat",
        image_tar_shards: Optional[list[str]] = None,
        image_sample_prob: float = 0.0,
        resume_state: Optional[DataloaderState] = None,
    ):
        super().__init__()
        if not torch.isfinite(torch.tensor([image_cond_prob, video_cond_prob], dtype=torch.float32)).all().item():
            raise ValueError(
                f"Require finite image_cond_prob and video_cond_prob, got image_cond_prob={image_cond_prob}, video_cond_prob={video_cond_prob}"
            )
        if image_cond_prob < 0 or video_cond_prob < 0 or image_cond_prob + video_cond_prob > 1:
            raise ValueError("Require 0 <= image_cond_prob, video_cond_prob and image_cond_prob + video_cond_prob <= 1")
        self.csv_paths = csv_paths
        self.num_buffers = num_buffers
        self.max_tokens = max_tokens
        self.qwen_token_budget = qwen_token_budget
        self.packed_video_token_budget = packed_video_token_budget
        self.packed_caption_token_budget = packed_caption_token_budget
        self.random_seed = random_seed
        self.cfg_drop_prob = cfg_drop_prob
        self.image_cond_prob = image_cond_prob
        self.video_cond_prob = video_cond_prob
        self.text_template_video_cfg = text_template_video_cfg
        self.caption_segment_limit = caption_segment_limit
        self.qwen_segment_limit = qwen_segment_limit
        self.max_num_seq = max_num_seq
        self.rope_embed_fn = rope_embed_fn
        self.patch_size = patch_size
        self.resize = resize
        self.dtype = dtype
        self.num_decode_workers = max(1, num_decode_workers)
        if conditioning_schema not in {"token_concat", "channel_concat"}:
            raise ValueError(
                f"Unsupported conditioning_schema {conditioning_schema!r}; "
                "expected 'token_concat' or 'channel_concat'."
            )
        self.conditioning_schema = conditioning_schema
        self.joint_segment_limit = joint_segment_limit
        self.image_tar_shards = list(image_tar_shards) if image_tar_shards else []
        self.image_sample_prob = image_sample_prob if self.image_tar_shards else 0.0
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer_path,
            padding_side="right",
        )
        self.cfg_empty_caption = build_empty_caption(
            tokenizer=self.text_tokenizer,
            template_cfg=self.text_template_video_cfg,
            caption_segment_limit=self.caption_segment_limit,
        )
        self.resume_state = resume_state

    def _item_stream(
        self, resume_ref: Optional[DataloaderItemRef] = None
    ) -> Iterator[tuple[dict, DataloaderItemRef]]:
        seek_csv_index: Optional[int] = resume_ref.csv_index if resume_ref is not None else None
        seek_pos: Optional[int] = resume_ref.local_file_pos if resume_ref is not None else None
        global_counter = itertools.count(resume_ref.global_index + 1 if resume_ref is not None else 0)
        epoch_index: int = resume_ref.epoch_index if resume_ref is not None else 0

        if not self.csv_paths:
            raise ValueError("This data-parallel rank has no video CSV to stream; "
                             "packed_dataloader_meta_csv_path needs at least one CSV per rank.")
        while True:
            # A pass that starts at the first row and yields nothing would loop forever.
            full_pass, rows = seek_csv_index is None, 0
            for csv_index, csv_path in enumerate(self.csv_paths):
                if seek_csv_index is not None and csv_index != seek_csv_index:
                    continue
                seek_csv_index = None

                with open(csv_path, newline="") as f:
                    tracker = _LineTracker(f)
                    reader = csv.DictReader(tracker)
                    _ = reader.fieldnames

                    if seek_pos is not None:
                        # Seek to the resumed-from row and realign the tracker's
                        # line counter with the freshly-created reader (the saved
                        # absolute line number can't be restored into
                        # ``csv.reader.line_num``), then skip that row so iteration
                        # continues at the following one.
                        tracker.seek(seek_pos, reader.line_num + 1)
                        next(reader)
                        local_counter = itertools.count(resume_ref.local_index + 1)
                        seek_pos = None
                    else:
                        local_counter = itertools.count(0)

                    # A logical CSV row may span multiple physical lines (quoted
                    # fields with embedded newlines). ``reader.line_num`` after the
                    # previous row points at that row's last physical line, so
                    # ``+ 1`` is the next row's first line; ``start_pos`` maps it
                    # back to that line's byte offset.
                    while True:
                        start_line_num = reader.line_num + 1
                        try:
                            row = next(reader)
                        except StopIteration:
                            break
                        ref = DataloaderItemRef(
                            global_index=next(global_counter),
                            csv_index=csv_index,
                            local_index=next(local_counter),
                            local_file_pos=tracker.start_pos(start_line_num),
                            local_line_num=start_line_num,
                            epoch_index=epoch_index,
                        )
                        rows += 1
                        yield row, ref

            if full_pass and rows == 0:
                message = (f"The video CSVs of this data-parallel rank have no data rows: "
                           f"{', '.join(self.csv_paths)}")
                logger.error(message)
                raise ValueError(message)
            epoch_index += 1
            global_counter = itertools.count(0)

    # -- per-item processors ----------------------------------------------

    def _make_video_process_fn(self):
        return partial(
            load_latent_row,
            seed=self.random_seed,
            cfg_empty_caption=self.cfg_empty_caption,
            tokenizer=self.text_tokenizer,
            template_cfg=self.text_template_video_cfg,
            caption_segment_limit=self.caption_segment_limit,
            qwen_segment_limit=self.qwen_segment_limit,
            dtype=self.dtype,
            max_tokens=self.max_tokens,
            qwen_token_budget=self.qwen_token_budget,
            packed_video_token_budget=self.packed_video_token_budget,
            packed_caption_token_budget=self.packed_caption_token_budget,
            joint_segment_limit=self.joint_segment_limit,
            patch_hw=(self.patch_size[1], self.patch_size[2]),
            video_cond_prob=self.video_cond_prob,
            image_cond_prob=self.image_cond_prob,
            cfg_drop_prob=self.cfg_drop_prob,
        )

    def _make_image_process_fn(self):
        return partial(
            load_image_sample,
            seed=self.random_seed,
            cfg_empty_caption=self.cfg_empty_caption,
            tokenizer=self.text_tokenizer,
            template_cfg=self.text_template_video_cfg,
            caption_segment_limit=self.caption_segment_limit,
            qwen_segment_limit=self.qwen_segment_limit,
            max_tokens=self.max_tokens,
            qwen_token_budget=self.qwen_token_budget,
            packed_video_token_budget=self.packed_video_token_budget,
            packed_caption_token_budget=self.packed_caption_token_budget,
            joint_segment_limit=self.joint_segment_limit,
            patch_hw=(self.patch_size[1], self.patch_size[2]),
            cfg_drop_prob=self.cfg_drop_prob,
        )

    @staticmethod
    def _image_ref(sample: ImageSample) -> DataloaderItemRef:
        return DataloaderItemRef(
            global_index=sample.global_index,
            modality="image",
            shard_index=sample.shard_index,
            sample_index=sample.sample_index,
            cycle=sample.cycle,
            shard_url=sample.shard_url,
            tar_offset=sample.tar_offset,
            tar_size=sample.tar_size,
            caption=sample.caption,
        )

    # -- resume reload -----------------------------------------------------

    def _load_sample_from_ref(self, ref: DataloaderItemRef, video_fn, image_fn) -> Optional["_BufferedSample"]:
        """Reconstruct one in-flight buffered sample from its ref on resume.

        Returns ``None`` if the item no longer passes the budget checks, e.g. a
        caption that now crops to a different length than when it was buffered.
        Losing one in-flight sample is preferable to failing the whole job.
        """
        if ref.modality == "image":
            latent = load_image_latent_from_ref(ref.shard_url, ref.tar_offset, ref.tar_size, self.dtype)
            sample = ImageSample(
                latent=latent,
                caption=ref.caption,
                shard_index=ref.shard_index,
                sample_index=ref.sample_index,
                cycle=ref.cycle,
                global_index=ref.global_index,
                shard_url=ref.shard_url,
                tar_offset=ref.tar_offset,
                tar_size=ref.tar_size,
            )
            result = image_fn((sample, ref))
        else:
            with open(self.csv_paths[ref.csv_index], newline="") as f:
                tracker = _LineTracker(f)
                reader = csv.DictReader(tracker)
                _ = reader.fieldnames
                f.seek(ref.local_file_pos)
                row = next(reader)
            result = video_fn((row, ref))
        if result is None:
            logger.warning(
                "Resumed item %s failed validation on reload; dropping it "
                "(check that the config matches the checkpoint if this is frequent)",
                ref,
            )
            return None
        tokenized_caption, decoded_frames, item = result
        return _BufferedSample(
            item=item,
            tokenized_caption=tokenized_caption,
            decoded_frames=decoded_frames,
        )

    # -- interleaved stream ------------------------------------------------

    def _mixed_stream(
        self,
        video_resume_ref: Optional[DataloaderItemRef],
        streamer: Optional[ImageTarStreamer],
        image_cursor: Optional[ImageStreamCursor],
        start_draw: int,
    ) -> Iterator[tuple]:
        """Interleave the video CSV stream and the image tar stream.

        Per item, draw image with probability ``image_sample_prob`` (else
        video) using a counter-seeded RNG, so the sequence is deterministic and
        resumable. Each yielded work item carries a :class:`_StreamResume`
        snapshot of the cursors *after* producing it, so the flush can persist
        exact resume state.
        """
        video_gen = self._item_stream(resume_ref=video_resume_ref)
        mixing = streamer is not None and self.image_sample_prob > 0.0
        draw = start_draw
        video_last = video_resume_ref
        image_last = image_cursor
        while True:
            if mixing:
                use_image = random.Random(
                    hash((self.random_seed, _MIX_DRAW_SALT, draw))
                ).random() < self.image_sample_prob
                draw += 1
            else:
                use_image = False
            if use_image:
                sample = streamer.next_sample()
                image_last = ImageStreamCursor(
                    sample.shard_index, sample.sample_index, sample.cycle, sample.global_index,
                )
                resume = _StreamResume(video_last, image_last, draw)
                yield ("image", sample, self._image_ref(sample), resume)
            else:
                row, ref = next(video_gen)
                video_last = ref
                resume = _StreamResume(video_last, image_last, draw)
                yield ("video", row, ref, resume)

    def __iter__(self) -> Iterator[PackedPlan]:
        """Yield :class:`PackedPlan` with decoded frames and CPU RoPE tensors."""
        buffers = [
            _VirtualBuffer(
                max_tokens=self.max_tokens,
                qwen_token_budget=self.qwen_token_budget,
                packed_video_token_budget=self.packed_video_token_budget,
                packed_caption_token_budget=self.packed_caption_token_budget,
                max_num_seq=self.max_num_seq,
            )
            for _ in range(self.num_buffers)
        ]

        video_fn = self._make_video_process_fn()
        image_fn = self._make_image_process_fn()

        def dispatch(work):
            kind, payload, ref, resume = work
            result = image_fn((payload, ref)) if kind == "image" else video_fn((payload, ref))
            if result is None:
                return None
            _, _, item = result
            item.resume_cursor = resume  # exact stream cursors as of this item
            return result

        streamer: Optional[ImageTarStreamer] = None
        try:
            with ThreadPoolExecutor(max_workers=self.num_decode_workers) as executor:
                video_resume_ref: Optional[DataloaderItemRef] = None
                image_cursor: Optional[ImageStreamCursor] = None
                start_draw = 0
                if self.resume_state is not None:
                    load_fn = partial(self._load_sample_from_ref, video_fn=video_fn, image_fn=image_fn)
                    for buf, refs in zip(buffers, self.resume_state.buffer_items):
                        for sample in executor.map(load_fn, refs):
                            if sample is None:
                                continue
                            # A rebuilt sample is not guaranteed byte-identical to
                            # the one that was buffered pre-checkpoint (caption
                            # variant, tokenizer/config drift), so re-check the
                            # budget instead of trusting it the way the steady
                            # state does. Adding blind here overflows the buffer
                            # and later yields a negative padding tail.
                            if buf.can_fit(sample):
                                buf.add(sample)
                                continue
                            fitting = [b for b in buffers if b.can_fit(sample)]
                            if fitting:
                                min(fitting, key=lambda b: b.remaining).add(sample)
                            else:
                                logger.warning(
                                    "Resumed item %s no longer fits any packing buffer; dropping it",
                                    sample.item.ref,
                                )
                    video_resume_ref = self.resume_state.next_stream_ref
                    image_cursor = self.resume_state.image_cursor
                    start_draw = self.resume_state.draw_count

                if self.image_tar_shards and self.image_sample_prob > 0.0:
                    streamer = ImageTarStreamer(
                        self.image_tar_shards, dtype=self.dtype, resume_after=image_cursor,
                    )

                for result in executor.map(
                    dispatch,
                    self._mixed_stream(video_resume_ref, streamer, image_cursor, start_draw),
                    buffersize=self.num_decode_workers * 8,
                ):
                    if result is None:
                        continue

                    tokenized_caption, decoded_frames, item = result
                    sample = _BufferedSample(
                        item=item,
                        tokenized_caption=tokenized_caption,
                        decoded_frames=decoded_frames,
                    )

                    fitting = list(filter(lambda b: b.can_fit(sample), buffers))
                    if fitting:
                        min(fitting, key=lambda b: b.remaining).add(sample)
                    else:
                        flush_buf = max(buffers, key=lambda b: b.used_tokens)
                        flushed = flush_buf.flush()
                        flush_buf.add(sample)

                        resume: _StreamResume = item.resume_cursor
                        yield _build_packed_plan(
                            samples=flushed,
                            dataloader_state=DataloaderState(
                                buffer_items=[[s.item.ref for s in buf.samples] for buf in buffers],
                                next_stream_ref=resume.video_next_ref,
                                image_cursor=resume.image_cursor,
                                draw_count=resume.draw_count,
                            ),
                            rope_embed_fn=self.rope_embed_fn,
                            patch_size=self.patch_size,
                        )
        finally:
            if streamer is not None:
                streamer.close()

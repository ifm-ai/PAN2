"""
Part 1 (image): streaming image-latent source + per-item processing.

Counterpart to ``video_item_processor.py``. Image VAE embeddings are stored as
local WebDataset-style tar shards: each sample is a pair ``<key>.safetensors``
(tensor key ``latent`` -> normalized ``[C, 1, H, W]`` bf16 latent, the same
convention as the cached video latents) and ``<key>.txt`` (caption).

Unlike videos (random-access ``.safetensors`` per CSV row), image latents live
inside tars that must be streamed sequentially. :class:`ImageTarStreamer` runs a
background thread that reads one tar ahead of consumption and emits individual
:class:`ImageSample` objects onto a bounded queue, so shard reads overlap with
training. The shard list is cycled forever, mirroring how the video CSV stream
loops over epochs.

Resume mirrors the video CSV cursor: each sample carries an
:class:`ImageStreamCursor` (shard/sample/cycle position) plus its
``.safetensors`` byte range, so a resumed run continues the stream from the
right place and can range-read any in-flight buffered sample straight back from
the tar without re-streaming it.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import glob
import logging
import os
import queue
import random
import tarfile
import threading
from typing import TYPE_CHECKING, BinaryIO, Iterator, List, Optional, Tuple

import torch

from .packed_metadata import FileSegment
from .video_item_processor import _TokenizedCaption, VideoItem, decode_latent_bytes, finalize_item

if TYPE_CHECKING:
    from .packing_dataset import DataloaderItemRef

logger = logging.getLogger(__name__)

_CAPTION_EXT = "txt"
_LATENT_EXT = "safetensors"

# Integer salt for the image caption-dropout RNG. Must stay an int (not a
# string) so ``hash(...)`` is stable across processes regardless of
# PYTHONHASHSEED — matching the all-int seeds used by the video processor.
_IMG_CFG_DROP_SALT = 0x1A6E1C0F


# ---------------------------------------------------------------------------
# Shard resolution
# ---------------------------------------------------------------------------


def _open_tar_fileobj(path: str) -> BinaryIO:
    """Open a tar shard for streaming straight off disk (``tarfile`` ``r|`` reads
    sequentially), so the multi-GB tars are never fully loaded into RAM."""
    return open(path, "rb")


def _read_member_bytes(path: str, offset: int, size: int) -> bytes:
    """Range-read a single tar member's bytes from a local shard."""
    seg = FileSegment(path, offset, size)
    try:
        return seg.read()
    finally:
        seg.f.close()


def list_local_tar_shards(root: str) -> List[str]:
    """Expand a directory into one absolute path per ``.tar`` under it (recursively)."""
    return sorted(
        os.path.abspath(p)
        for p in glob.glob(os.path.join(root, "**", "*.tar"), recursive=True)
    )


def resolve_image_tar_shards(specs: List[str]) -> List[str]:
    """Resolve a list of specs into a flat, sorted list of local tar-shard paths.

    Each spec may be:
      * a directory (expanded recursively to all ``.tar`` files under it),
      * a ``*.tar`` file (used as-is),
      * a text file with one spec per line (each resolved recursively; lines may
        themselves be directories or tar paths).
    """
    shards: List[str] = []
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue
        if os.path.isdir(spec):
            shards.extend(list_local_tar_shards(spec))
        elif spec.endswith(".tar"):
            shards.append(os.path.abspath(spec))
        else:
            # Text file of specs, one per line; resolve each recursively so it
            # can list directories or tar paths.
            with open(spec) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            shards.extend(resolve_image_tar_shards(lines))
    return sorted(set(shards))


# ---------------------------------------------------------------------------
# Sample / cursor types
# ---------------------------------------------------------------------------


@dataclass
class ImageSample:
    """One decoded image latent plus everything needed to resume/reload it."""
    latent: torch.Tensor       # [C, T, H, W] normalized latent (T == 1 for images)
    caption: str
    shard_index: int           # index into the resolved shard list
    sample_index: int          # 0-based index of the sample within its shard
    cycle: int                 # how many times the shard list has been wrapped
    global_index: int          # monotonic counter across the whole image stream
    shard_url: str
    tar_offset: int            # byte offset of the .safetensors member data
    tar_size: int              # byte size of the .safetensors member data


@dataclass
class ImageStreamCursor:
    """Position of the last *consumed* image sample, for exact resume."""
    shard_index: int
    sample_index: int
    cycle: int
    global_index: int


# ---------------------------------------------------------------------------
# Per-shard streaming
# ---------------------------------------------------------------------------


def _decode_latent(raw: bytes, dtype: torch.dtype) -> torch.Tensor:
    # Shared with the video loader so both honor the same `.safetensors` format.
    return decode_latent_bytes(raw, dtype)


def _iter_shard_members(fileobj) -> Iterator[Tuple[str, bytes, int, int, str]]:
    """Yield ``(key, latent_bytes, tar_offset, tar_size, caption)`` per sample.

    WebDataset groups members of one sample under a common ``<key>`` stem with
    different extensions; members of a sample are contiguous in the tar. The
    image bytes are read eagerly (we need the latent now), while the member's
    byte range is recorded so a resume can range-read it directly.
    """
    tf = tarfile.open(fileobj=fileobj, mode="r|")
    pending: dict = {}
    last_stem: Optional[str] = None

    def _ready(stem: str):
        d = pending.pop(stem, None)
        if not d or "latent_bytes" not in d or "caption" not in d:
            return None
        return (stem, d["latent_bytes"], d["offset"], d["size"], d["caption"])

    for member in tf:
        if not member.isfile():
            continue
        stem, _, ext = member.name.partition(".")
        ext = ext.lower()
        if last_stem is not None and stem != last_stem:
            out = _ready(last_stem)
            if out is not None:
                yield out
        last_stem = stem
        d = pending.setdefault(stem, {})
        if ext == _LATENT_EXT:
            d["offset"] = member.offset_data
            d["size"] = member.size
            d["latent_bytes"] = tf.extractfile(member).read()
        elif ext == _CAPTION_EXT:
            try:
                d["caption"] = tf.extractfile(member).read().decode("utf-8", "replace").strip()
            except Exception:
                pass
    if last_stem is not None:
        out = _ready(last_stem)
        if out is not None:
            yield out


def load_image_latent_from_ref(shard_url: str, tar_offset: int, tar_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Range-read a single ``.safetensors`` member back from its tar and decode it.

    Used on resume to reconstruct an in-flight buffered image sample without
    re-streaming the whole tar.
    """
    raw = _read_member_bytes(shard_url, tar_offset, tar_size)
    return _decode_latent(raw, dtype)


# ---------------------------------------------------------------------------
# Background-prefetching streamer
# ---------------------------------------------------------------------------


class ImageTarStreamer:
    """Infinite stream of :class:`ImageSample` over a fixed list of tar shards.

    Two levels of prefetch keep the GPU fed without stalls at shard boundaries:

    * **Tar-ahead prefetch** — a small download pool fetches the next
      ``tar_prefetch`` whole shards *concurrently* while the producer decodes
      the current one, so the next tar's download overlaps consumption of the
      current tar (and there is no cold-start stall beyond the very first tar).
    * **Sample-level buffer** — a daemon producer thread decodes members from
      the ready shards and fills a bounded :class:`queue.Queue` of decoded
      samples, smoothing per-sample jitter.

    The shard list is cycled forever. Consume with :meth:`next_sample`.
    """

    _SENTINEL = object()

    def __init__(
        self,
        shards: List[str],
        dtype: torch.dtype,
        resume_after: Optional[ImageStreamCursor] = None,
        queue_size: int = 256,
        tar_prefetch: int = 2,
        download_workers: int = 2,
    ):
        if not shards:
            raise ValueError("ImageTarStreamer requires at least one tar shard")
        self.shards = shards
        self.dtype = dtype
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._resume_after = resume_after
        self._tar_prefetch = max(1, tar_prefetch)
        self._download_pool = ThreadPoolExecutor(
            max_workers=max(1, download_workers), thread_name_prefix="image-tar-dl"
        )
        self._thread = threading.Thread(target=self._produce, name="image-tar-streamer", daemon=True)
        self._thread.start()

    # -- shard schedule ----------------------------------------------------

    def _shard_plan(self) -> Iterator[Tuple[int, int, int]]:
        """Infinite ``(shard_pos, cycle, skip_through)`` schedule.

        Starts at the resume position (cycling forever). ``skip_through`` is the
        last already-consumed ``sample_index`` to skip on the first resumed
        shard, and ``-1`` everywhere else.
        """
        cur = self._resume_after
        shard_pos = cur.shard_index if cur is not None else 0
        cycle = cur.cycle if cur is not None else 0
        first = True
        while True:
            skip_through = cur.sample_index if (first and cur is not None) else -1
            yield shard_pos, cycle, skip_through
            first = False
            shard_pos += 1
            if shard_pos >= len(self.shards):
                shard_pos = 0
                cycle += 1

    # -- producer ----------------------------------------------------------

    def _produce(self) -> None:
        plan = self._shard_plan()
        global_index = (self._resume_after.global_index + 1) if self._resume_after is not None else 0

        # Sliding window of in-flight whole-tar downloads, kept ``tar_prefetch``
        # deep so the next shards download while the current one is decoded.
        pending: "deque[Tuple[Future, int, int, int]]" = deque()

        def submit_next() -> None:
            shard_pos, cycle, skip_through = next(plan)
            fut = self._download_pool.submit(self._download, self.shards[shard_pos])
            pending.append((fut, shard_pos, cycle, skip_through))

        try:
            for _ in range(self._tar_prefetch):
                submit_next()

            while not self._stop.is_set():
                fut, shard_pos, cycle, skip_through = pending.popleft()
                submit_next()  # keep the prefetch window full
                shard_url = self.shards[shard_pos]
                body = None
                try:
                    body = fut.result()
                    if body is None:  # download skipped because the streamer stopped
                        continue
                    for sample_index, (key, latent_bytes, off, size, caption) in enumerate(
                        _iter_shard_members(body)
                    ):
                        if self._stop.is_set():
                            return
                        if sample_index <= skip_through:
                            continue
                        try:
                            latent = _decode_latent(latent_bytes, self.dtype)
                        except Exception as exc:
                            logger.warning("Skipping image %s in %s: %s", key, shard_url, exc)
                            continue
                        self._put(ImageSample(
                            latent=latent,
                            caption=caption,
                            shard_index=shard_pos,
                            sample_index=sample_index,
                            cycle=cycle,
                            global_index=global_index,
                            shard_url=shard_url,
                            tar_offset=off,
                            tar_size=size,
                        ))
                        global_index += 1
                except Exception as exc:
                    logger.warning("Failed to stream image tar %s: %s", shard_url, exc)
                finally:
                    # Close the per-shard fileobj so file descriptors don't leak
                    # across the (infinite) shard cycle.
                    if body is not None:
                        try:
                            body.close()
                        except Exception:
                            pass
        finally:
            # Unblock a consumer waiting on a stopped stream.
            try:
                self._queue.put(self._SENTINEL, timeout=1.0)
            except queue.Full:
                pass

    def _download(self, url: str):
        if self._stop.is_set():
            return None
        return _open_tar_fileobj(url)

    def _put(self, sample: ImageSample) -> None:
        # Block until space is available, waking periodically to honour stop.
        while not self._stop.is_set():
            try:
                self._queue.put(sample, timeout=1.0)
                return
            except queue.Full:
                continue

    # -- consumer ----------------------------------------------------------

    def next_sample(self) -> ImageSample:
        item = self._queue.get()
        if item is self._SENTINEL:
            raise StopIteration("image tar stream ended")
        return item

    def close(self) -> None:
        self._stop.set()
        # Drain one slot so a blocked producer can observe the stop flag.
        try:
            self._queue.get_nowait()
        except Exception:
            pass
        # Guarded: during interpreter shutdown the concurrent.futures/queue
        # internals may already be torn down, so swallow anything here.
        try:
            self._download_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------


def load_image_sample(
    sample_and_ref: tuple[ImageSample, "DataloaderItemRef"],
    *,
    seed: int,
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
    patch_hw: Tuple[int, int],
    cfg_drop_prob: float,
) -> tuple[_TokenizedCaption, torch.Tensor, VideoItem] | None:
    """Process one already-decoded image latent into a packable item.

    The latent is supplied by :class:`ImageTarStreamer` (or range-read from a
    ref on resume), so there is no download here — only caption dropout and the
    shared token-budget checks. Images are single-frame, so ``prefix_tokens`` is
    always 0 (no image/video conditioning prefix).

    Images may be any resolution: the per-image token count is ``(H//ph) *
    (W//pw)`` and the packer patchifies each sample with its own H,W. The only
    requirement is that the latent grid is divisible by the patch size.
    """
    sample, ref = sample_and_ref

    ph, pw = int(patch_hw[0]), int(patch_hw[1])
    h, w = int(sample.latent.shape[2]), int(sample.latent.shape[3])
    if h % ph != 0 or w % pw != 0:
        logger.warning(
            "Skipping image %s: latent grid %s not divisible by patch %s",
            sample.shard_url, (h, w), (ph, pw),
        )
        return None
    tokens_per_frame = (h // ph) * (w // pw)

    # Caption dropout, seeded deterministically by the image's stream position
    # (independent of the video RNG) so resume reproduces the same draw.
    rng_seed = hash((seed, _IMG_CFG_DROP_SALT, sample.cycle, sample.shard_index, sample.sample_index))
    drop_caption = random.Random(rng_seed).random() < cfg_drop_prob

    return finalize_item(
        latent=sample.latent,
        caption=sample.caption,
        ref=ref,
        path=sample.shard_url,
        prefix_latent_steps=0,
        drop_caption=drop_caption,
        modality="image",
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

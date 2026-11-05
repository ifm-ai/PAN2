"""Bucket quack's autotune cache key over tensor shapes.

quack's Autotuner keys its config cache on the EXACT shape of every tensor
argument (see quack.autotuner.Autotuner.__call__). With fixed-size padded
batches every step had identical buffer shapes, so each gemm autotuned
exactly once. Unpadded packed batches have a different total token count every
step, so every step became a fresh cache miss: a full re-autotune -- subprocess
compile pools plus benchmarking dozens of configs -- costing minutes per gemm
at production sizes, dominating (especially) the backward pass.

The tuned config only encodes tile/cluster/scheduler choices; kernels are
compiled shape-dynamically, so a config tuned at one sequence length is valid
(and near optimal) for any other. This patch therefore mirrors the config cache
under a bucketed key and seeds the exact-shape cache from it. The bucketed key
keeps every tensor dim at its EXACT size except sequence-length dims, which are
DROPPED from the key entirely -- one config per gemm's static geometry, not one
per sequence length (or per power of two of it).

Sequence-length dims are identified two ways:
  * a priori: the trainer calls register_quack_dynamic_sizes(T) each step with
    the packed sequence length; any dim FIRST observed at a registered size is
    marked dynamic immediately, so the very first autotune already uses the
    final (dropped-dim) key and later sequence lengths hit it -- zero re-tunes.
    Marking is first-observation-only: a static dim that merely coincides with
    T on a later step keeps its exact key (no flip, no orphaned entry).
  * fallback: a dim observed at two different sizes flips to dynamic (costs the
    one re-tune the a-priori path exists to avoid; also catches anything the
    trainer didn't register).

Bucket entries additionally persist to disk (via quack's own FileCacheManager,
atomic writes, QUACK_CACHE_DIR-controlled), keyed on (quack version, torch
version, GPU, kernel, bucket key, config set): a config tuned once on a machine
is reused by every later process, so restarts skip the cold autotune entirely.
Set QUACK_AUTOTUNE_PATCH_NO_DISK=1 to disable persistence.

The exact-key computation below must mirror quack's own (verified against
quack-kernels 0.6.3 and 0.6.4; see the "cache-hit fast path" in
Autotuner.__call__). 0.6.x keys hold RAW values
-- tuple(shape), tuple(strides), torch.dtype -- where 0.5.x stringified them, and
it reads the `key` fields from kwargs only, walking positional args before
keyword args. If a quack upgrade changes the format again, seeding silently
stops matching; we detect that (a benchmarked call whose key we predicted wrong)
and disable the patch with a warning rather than let every step re-autotune.
tests/test_quack_autotune_key.py asserts the two agree, so an upgrade that moves
the format fails a test instead of quietly costing throughput.
"""

import hashlib
import json
import logging
import os
import sys
import time

import torch

logger = logging.getLogger(__name__)

_DYNAMIC = "*"  # placeholder for a dim dropped from the bucketed key
_registered_dynamic_sizes: set[int] = set()


def register_quack_dynamic_sizes(*sizes: int) -> None:
    """Declare the per-step dynamic size(s) (the packed sequence length).

    Call once per step BEFORE the forward (the trainer does). Tensor dims first
    observed at one of these sizes are dropped from the autotune bucket key
    immediately, so the first tune's key already covers every future sequence
    length. Replaces the previous registration.
    """
    _registered_dynamic_sizes.clear()
    _registered_dynamic_sizes.update(int(s) for s in sizes if s is not None)


def _bucket_shape(self, name: str, shape) -> tuple:
    """Shape key for one tensor arg: exact sizes, with sequence-length dims
    replaced by a placeholder (a str, so a bucketed shape can never collide
    with a real all-int shape tuple). A dim becomes dynamic (permanently, per
    (arg name, dim index), per Autotuner instance) either on FIRST observation
    at a registered dynamic size, or on being observed at two different sizes
    (fallback; orphans at most one cached bucket entry -> one extra autotune)."""
    seen = self._bucket_dim_seen        # {(name, i): first-seen size}
    dynamic = self._bucket_dynamic_dims  # {(name, i)}
    out = []
    for i, d in enumerate(shape):
        key = (name, i)
        d = int(d)
        if key in dynamic:
            out.append(_DYNAMIC)
            continue
        if key not in seen and d in _registered_dynamic_sizes:
            dynamic.add(key)
            out.append(_DYNAMIC)
            continue
        prev = seen.setdefault(key, d)
        if prev != d:
            dynamic.add(key)
            out.append(_DYNAMIC)
        else:
            out.append(d)
    return tuple(out)


def _make_keys(self, args, kwargs):
    """Compute (exact_key, bucketed_key); exact_key must match upstream's.

    Mirrors quack.autotuner.Autotuner.__call__: the `key` fields come from
    kwargs only, then every positional Tensor arg contributes
    (shape, bucketed strides, dtype) in positional order, then every keyword
    Tensor arg whose name is a real parameter does the same. Values stay raw
    (tuples / torch.dtype); upstream stringifies only when persisting to disk.
    """
    arg_name_set = self._arg_name_set
    arg_names = self.arg_names
    exact = [kwargs[k] for k in self.keys if k in kwargs]
    bucket = list(exact)

    def add(name, arg):
        shape = tuple(arg.shape)
        # Strides are never negative, so this matches upstream's `s if s < 2 else 2`.
        stride = tuple(s if s < 2 else 2 for s in arg.stride())
        exact.extend((shape, stride, arg.dtype))
        bucket.extend((_bucket_shape(self, name, shape), stride, arg.dtype))

    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            add(arg_names[i] if i < len(arg_names) else f"__arg{i}", arg)
    for name, arg in kwargs.items():
        if isinstance(arg, torch.Tensor) and name in arg_name_set:
            add(name, arg)
    return tuple(exact), tuple(bucket)


# ---------------------------------------------------------------------------
# Disk persistence of bucket entries (config identified by str(config), quack's
# own serialization convention -- see Autotuner.check_disk_cache).
# ---------------------------------------------------------------------------

def _disk_manager(self, bucket_key):
    from quack.autotuner import FileCacheManager, VERSION, _base32

    fn_name = getattr(self.fn, "__name__", repr(self.fn))
    config_strs = sorted(str(c) for c in self.configs)
    ident = "-".join(
        ["bucket_patch_v2", VERSION, torch.__version__, torch.cuda.get_device_name(),
         fn_name, str(bucket_key)] + config_strs
    )
    key = hashlib.sha256(ident.encode("utf-8")).hexdigest()
    return FileCacheManager(_base32(key)), f"{fn_name[:150]}.bucket_autotune.json"


def _disk_load(self, bucket_key):
    if os.environ.get("QUACK_AUTOTUNE_PATCH_NO_DISK") == "1":
        return None
    try:
        cache, file_name = _disk_manager(self, bucket_key)
        path = cache.get_file(file_name)
        if not path:
            return None
        with open(path, "r") as f:
            config_str = json.load(f)["config"]
        return {str(c): c for c in self.configs}.get(config_str)
    except Exception as e:  # disk cache is best-effort; never break the call
        logger.warning(f"quack_autotune_patch: disk cache read failed ({e!r})")
        return None


def _disk_store(self, bucket_key, config):
    if os.environ.get("QUACK_AUTOTUNE_PATCH_NO_DISK") == "1":
        return
    try:
        cache, file_name = _disk_manager(self, bucket_key)
        cache.put(
            # str(): the key holds torch dtypes and shape tuples, and this field
            # is informational -- only "config" is read back (same convention as
            # quack's own check_disk_cache).
            json.dumps({"bucket_key": str(bucket_key), "config": str(config)}),
            file_name, binary=False,
        )
    except Exception as e:
        logger.warning(f"quack_autotune_patch: disk cache write failed ({e!r})")


def patch_quack_autotune_shape_buckets():
    from quack.autotuner import Autotuner

    if getattr(Autotuner, "_shape_bucket_patched", False):
        return
    Autotuner._shape_bucket_patched = True

    orig_call = Autotuner.__call__

    def bucketed_call(self, *args, **kwargs):
        if len(self.configs) <= 1 or getattr(self, "_bucket_disabled", False):
            return orig_call(self, *args, **kwargs)

        if not hasattr(self, "_bucket_cache"):
            self._bucket_cache = {}
            self._bucket_dim_seen = {}
            self._bucket_dynamic_dims = set()
        bcache = self._bucket_cache

        exact_key, bucket_key = _make_keys(self, args, kwargs)

        if bucket_key not in bcache:
            disk_config = _disk_load(self, bucket_key)
            if disk_config is not None:
                bcache[bucket_key] = disk_config

        seeded = False
        if exact_key not in self.cache and bucket_key in bcache:
            self.cache[exact_key] = bcache[bucket_key]
            seeded = True

        # A cold autotune compiles and benchmarks every config and can take minutes
        # with nothing else in the log, so say so (global rank 0 only).
        cold = exact_key not in self.cache and os.environ.get("RANK", "0") == "0"
        if cold:
            name = getattr(self.fn, "__name__", "gemm")
            print(f"[quack] autotuning {name} over {len(self.configs)} configs (cold cache; "
                  "can take minutes per kernel, cached in QUACK_CACHE_DIR afterwards)",
                  file=sys.stderr, flush=True)
            start = time.perf_counter()
        ret = orig_call(self, *args, **kwargs)
        if cold:
            print(f"[quack] tuned {name} in {time.perf_counter() - start:.0f}s", file=sys.stderr, flush=True)

        config = self.cache.get(exact_key)
        if config is None:
            # Upstream used a different key than we predicted (quack version
            # skew) -- our seeding can never hit. Fall back to stock behavior.
            self._bucket_disabled = True
            logger.warning(
                "quack_autotune_patch: exact-key mismatch against installed quack; "
                "shape-bucketed autotune caching disabled (per-shape autotune cost returns)."
            )
            return ret

        prev = bcache.get(bucket_key)
        bcache[bucket_key] = config
        if prev is None or str(prev) != str(config):
            # Freshly tuned (or changed) entry: persist for future processes.
            _disk_store(self, bucket_key, config)
        if seeded:
            # Keep memory flat: one persistent entry per bucket is enough.
            self.cache.pop(exact_key, None)
        return ret

    Autotuner.__call__ = bucketed_call

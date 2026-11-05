# Copyright 2026 Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
# SPDX-License-Identifier: Apache-2.0
"""Split model parameters between ``torch.optim.Muon`` and AdamW and build both optimizers."""
import torch
from collections import defaultdict
from typing import Dict, List, Tuple

from .patch_lr_scheduler import patch_lr_scheduler
from .patch_muon import patch_muon


def _looks_like_embedding_or_head(name: str) -> bool:
    """
    Model-aware exclusions for params that should stay on AdamW with native Muon.

    In this repo's diffusion transformer, we exclude:
    - Explicit embeddings / token-type embeddings
    - Conditioning/input projection stacks (time/text/vector guidance paths)
    - Output head (`output_head.*`)
    """
    lower = name.lower()

    # Generic patterns (portable across model variants).
    generic_substrings = (
        "embedding",
        "embed_tokens",
        "token_embed",
        "token_emb",
        "pos_embed",
        "position_embed",
        "lm_head",
        "output_head",
        "classifier",
    )
    if any(p in lower for p in generic_substrings):
        return True

    # Modulation/gating layers are zero-initialized in this codebase and are better kept on AdamW.
    if ".modulation." in lower or "text_modulation" in lower:
        return True

    # PAN transformer-specific paths.
    exact_prefixes = (
        "output_head.",           # output projection + its modulation
        "timestep_encoder.",      # timestep MLP
        "text_type_embedding",    # nn.Embedding(3, hidden_size)
        "patch_embed.",           # patch embedding (conv today; keep future-proof)
    )
    if lower.startswith(exact_prefixes):
        return True

    return False


def split_params_for_native_muon(
    model,
) -> Tuple[List[Tuple[str, torch.nn.Parameter]], List[Tuple[str, torch.nn.Parameter]], Dict[str, int]]:
    """
    Split trainable params for native torch.optim.Muon best-practice usage:
    - Muon: hidden-layer 2D params
    - AdamW: everything else (1D, >2D, embeddings/heads, etc.)
    """
    muon_named_params: List[Tuple[str, torch.nn.Parameter]] = []
    adamw_named_params: List[Tuple[str, torch.nn.Parameter]] = []
    stats = {
        "trainable_param_tensors": 0,
        "muon_param_tensors": 0,
        "adamw_param_tensors": 0,
        "muon_param_numel": 0,
        "adamw_param_numel": 0,
        "excluded_embedding_or_head_tensors": 0,
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        stats["trainable_param_tensors"] += 1

        # Native torch Muon is intended for 2D hidden-layer weights only.
        use_muon = p.ndim == 2 and not _looks_like_embedding_or_head(name)
        if use_muon:
            muon_named_params.append((name, p))
            stats["muon_param_tensors"] += 1
            stats["muon_param_numel"] += p.numel()
        else:
            adamw_named_params.append((name, p))
            stats["adamw_param_tensors"] += 1
            stats["adamw_param_numel"] += p.numel()
            if p.ndim == 2 and _looks_like_embedding_or_head(name):
                stats["excluded_embedding_or_head_tensors"] += 1

    return muon_named_params, adamw_named_params, stats


def build_native_muon_optimizers(
    model,
    *,
    lr=1e-3,
    momentum=0.95,
    weight_decay=0.1,
    muon_adjust_lr_fn="match_rms_adamw",
    adamw_betas=(0.95, 0.95),
    adamw_eps=1e-8,
    capturable=False,
    muon_chunks: dict[str, tuple[int, int]] | None = None,
):
    """
    Build native torch Muon + AdamW optimizers following official best practice.

    Returns:
        dict with keys:
        - optimizers: ordered dict-like mapping {"muon": ..., "adamw": ...} (missing keys omitted)
        - stats: split summary
        - param_names: {"muon": [...], "adamw": [...]}
    """
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "torch.optim.Muon is not available in this PyTorch build. "
            f"Installed torch version: {torch.__version__}. Please upgrade to a version that includes torch.optim.Muon."
        )

    muon_named_params, adamw_named_params, stats = split_params_for_native_muon(model)
    optimizers: Dict[str, torch.optim.Optimizer] = {}

    if muon_named_params:
        if isinstance(lr, torch.Tensor):
            patch_muon()
            patch_lr_scheduler()

        # Separate params by chunk config; each unique (dim, n_chunks) gets its own
        # param group with adjust_lr_fn encoded as (base_fn, dim, n_chunks).
        chunks_map = muon_chunks or {}
        chunked_groups: dict[tuple[int, int], list] = defaultdict(list)
        regular_params: list = []
        for name, p in muon_named_params:
            if name in chunks_map:
                chunked_groups[chunks_map[name]].append(p)
            else:
                regular_params.append(p)

        param_groups: list[dict] = [{'params': regular_params}]
        for (dim, n_chunks), group_params in chunked_groups.items():
            param_groups.append({
                'params': group_params,
                'adjust_lr_fn': (muon_adjust_lr_fn, dim, n_chunks),
            })

        optimizers["muon"] = torch.optim.Muon(
            param_groups,
            lr=lr.clone() if isinstance(lr, torch.Tensor) else lr,
            weight_decay=weight_decay,
            momentum=momentum,
            adjust_lr_fn=muon_adjust_lr_fn,
        )
    if adamw_named_params:
        optimizers["adamw"] = torch.optim.AdamW(
            adamw_named_params,
            lr=lr.clone() if isinstance(lr, torch.Tensor) else lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
            capturable=capturable,
            fused=True,
        )

    if not optimizers:
        raise ValueError("No trainable parameters found to build optimizers.")

    chunks_map = muon_chunks or {}
    return {
        "optimizers": optimizers,
        "stats": stats,
        "param_names": {
            "muon": [name for name, _ in muon_named_params if name not in chunks_map],
            "adamw": [name for name, _ in adamw_named_params],
            "muon_chunked": [(name, *chunks_map[name]) for name, _ in muon_named_params if name in chunks_map],
        },
    }

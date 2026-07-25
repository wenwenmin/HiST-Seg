"""Utilities for loading direct 512-D scGPT spot features.

The preprocessing cache stores one feature vector per ST spot together with a
1/4-resolution RGB thumbnail, tissue mask, and spot coordinates in the same
thumbnail coordinate system. Runtime normalization is label-free and performed
per slice before any patch-specific spot selection.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _get(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default)


def resolve_context_cache_dir(cfg: Any, ds_cfg: Any, fold: Optional[str] = None) -> str:
    """Return the configured tagged context-cache directory.

    ``fold`` is accepted for call-site compatibility. The current all-gene
    pipeline stores caches by tag rather than by fold because feature extraction
    itself is label-free.
    """
    del fold
    root = str(ds_cfg.context_cache_root)
    tag = str(_get(getattr(cfg, "data", None), "context_cache_tag", "") or "").strip()
    return os.path.join(root, tag) if tag else root


def resolve_context_cache_path(
    cfg: Any,
    ds_cfg: Any,
    slice_id: str,
    fold: Optional[str] = None,
) -> str:
    return os.path.join(resolve_context_cache_dir(cfg, ds_cfg, fold), f"{slice_id}_ctx.npz")


def normalize_scgpt_spot_features(
    features: np.ndarray,
    cfg: Any,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply optional spot LayerNorm followed by label-free slice centering."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"ST features must be [N,C], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("ST features contain NaN/Inf before normalization")

    norm_cfg = getattr(cfg, "st_feature_normalization", None)
    enabled = bool(_get(norm_cfg, "enabled", False))
    before_mean = float(np.mean(x)) if x.size else 0.0
    before_std = float(np.std(x)) if x.size else 0.0
    if not enabled:
        return np.ascontiguousarray(x), {
            "enabled": 0.0,
            "before_mean": before_mean,
            "before_std": before_std,
            "after_mean": before_mean,
            "after_std": before_std,
        }

    eps = float(_get(norm_cfg, "eps", 1e-6))
    if eps <= 0:
        raise ValueError("st_feature_normalization.eps must be positive")

    x = x.copy()
    if bool(_get(norm_cfg, "spot_layer_norm", True)):
        spot_mean = x.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        spot_var = ((x - spot_mean) ** 2).mean(axis=1, keepdims=True, dtype=np.float64)
        x = (x - spot_mean) / np.sqrt(spot_var + eps).astype(np.float32)

    if bool(_get(norm_cfg, "slice_center", True)):
        mode = str(_get(norm_cfg, "center_mode", "mean")).lower()
        if mode == "mean":
            center = x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        elif mode == "median":
            center = np.median(x, axis=0, keepdims=True).astype(np.float32)
        else:
            raise ValueError("st_feature_normalization.center_mode must be 'mean' or 'median'")
        x -= center

    if not np.isfinite(x).all():
        raise ValueError("ST features contain NaN/Inf after normalization")

    return np.ascontiguousarray(x), {
        "enabled": 1.0,
        "before_mean": before_mean,
        "before_std": before_std,
        "after_mean": float(np.mean(x)) if x.size else 0.0,
        "after_std": float(np.std(x)) if x.size else 0.0,
    }


def load_runtime_context_cache(
    path: str,
    cfg: Any,
    expected_dim: int,
    expected_fold: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and validate one tagged H&E/ST context cache."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        required = {"features", "coords_thumb", "tissue_thumb", "thumb"}
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path}: missing cache keys {missing}")

        features = np.asarray(data["features"], dtype=np.float32)
        coords = np.asarray(data["coords_thumb"], dtype=np.float32)
        tissue = np.asarray(data["tissue_thumb"], dtype=np.uint8)
        thumb = np.asarray(data["thumb"], dtype=np.uint8)
        cache_ds = int(np.asarray(data["ds"]).item()) if "ds" in data.files else int(cfg.geometry.downsample)
        full_height = int(np.asarray(data["full_h"]).item()) if "full_h" in data.files else int(round(thumb.shape[0] * cache_ds))
        full_width = int(np.asarray(data["full_w"]).item()) if "full_w" in data.files else int(round(thumb.shape[1] * cache_ds))

        if features.ndim != 2 or features.shape[1] != int(expected_dim):
            raise ValueError(f"{path}: expected features [N,{int(expected_dim)}], got {features.shape}")
        if coords.shape != (features.shape[0], 2):
            raise ValueError(f"{path}: coords/features mismatch {coords.shape} vs {features.shape}")
        if thumb.ndim != 3 or thumb.shape[2] != 3:
            raise ValueError(f"{path}: thumb must be HxWx3 RGB, got {thumb.shape}")
        if tissue.shape != thumb.shape[:2]:
            raise ValueError(f"{path}: tissue/thumb shape mismatch {tissue.shape} vs {thumb.shape}")

        # Old fold-aware caches are still rejected when they explicitly carry a
        # mismatching held-out identifier. New tagged caches normally omit it.
        if expected_fold and "held_out" in data.files:
            value = np.asarray(data["held_out"])
            cache_fold = str(value.item() if value.ndim == 0 else value.reshape(-1)[0])
            if cache_fold != str(expected_fold):
                raise RuntimeError(
                    f"Fold cache mismatch: expected={expected_fold}, cache={cache_fold}, path={path}"
                )

        metadata: Dict[str, Any] = {}
        for key in ("feature_type", "preprocessing_mode", "cache_tag"):
            if key in data.files:
                value = np.asarray(data[key])
                metadata[key] = str(value.item() if value.ndim == 0 else value.reshape(-1)[0])

    # A ds=4 cache is required by the supplied configs. Rejecting mismatches
    # prevents accidental reuse of the old ds=16 / 4096-FOV caches.
    expected_ds = int(cfg.geometry.downsample)
    if cache_ds != expected_ds:
        raise ValueError(
            f"{path}: cache downsample={cache_ds}, config expects {expected_ds}. "
            "Regenerate context caches with the new preprocessing configuration."
        )

    features, norm_stats = normalize_scgpt_spot_features(features, cfg)
    return {
        "features": features,
        "thumb_native": np.ascontiguousarray(thumb),
        "coords_thumb_native": np.ascontiguousarray(coords),
        "tissue_thumb_native": np.ascontiguousarray(tissue),
        "cache_downsample": cache_ds,
        "full_height": full_height,
        "full_width": full_width,
        "normalization_stats": norm_stats,
        **metadata,
    }

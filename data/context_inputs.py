"""Utilities for spatially aligned H&E/ST context extraction.

The new multimodal context branch operates on one physical field of view:

* detail patch: ``patch_size x patch_size`` pixels;
* context field: ``ctx_scale`` times wider/taller in the original slide;
* H&E context crop: read from the cached thumbnail at ``geometry.downsample``;
* ST spots: selected from exactly the same crop and represented by relative
  coordinates in ``[0, 1] x [0, 1]``.

The dataset returns padded spot sets rather than a dense 512-channel raster.
The supplied configuration uses a 512-pixel detail patch, 4x context scale, and
a 1/4-resolution context cache, so every context sample covers 2048x2048 source
pixels while the model receives a 512x512 H&E context crop.
The learnable 512->32 spot encoder therefore runs before differentiable
rasterization, avoiding the memory cost of a dense high-dimensional map.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _safe_window(
    array: np.ndarray,
    center_x: int,
    center_y: int,
    size: int,
    *,
    pad_mode: str = "constant",
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop a square window around ``(center_x, center_y)`` with padding."""
    height, width = array.shape[:2]
    half = int(size) // 2
    x0 = int(center_x) - half
    y0 = int(center_y) - half
    x1 = x0 + int(size)
    y1 = y0 + int(size)

    xs, xe = max(0, x0), min(width, x1)
    ys, ye = max(0, y0), min(height, y1)
    crop = array[ys:ye, xs:xe]

    pads = [
        (max(0, -y0), max(0, y1 - height)),
        (max(0, -x0), max(0, x1 - width)),
    ]
    if array.ndim == 3:
        pads.append((0, 0))

    if any(left or right for left, right in pads):
        try:
            crop = np.pad(crop, pads, mode=pad_mode)
        except ValueError:
            crop = np.pad(crop, pads, mode="constant")

    return crop, (x0, y0)



def _safe_rect_window(
    array: np.ndarray,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    *,
    pad_mode: str = "constant",
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop a rectangular window with padding and return its source origin."""
    source_h, source_w = array.shape[:2]
    x0 = int(center_x) - int(width) // 2
    y0 = int(center_y) - int(height) // 2
    x1 = x0 + int(width)
    y1 = y0 + int(height)
    xs, xe = max(0, x0), min(source_w, x1)
    ys, ye = max(0, y0), min(source_h, y1)
    crop = array[ys:ye, xs:xe]
    pads = [
        (max(0, -y0), max(0, y1 - source_h)),
        (max(0, -x0), max(0, x1 - source_w)),
    ]
    if array.ndim == 3:
        pads.append((0, 0))
    if any(left or right for left, right in pads):
        try:
            crop = np.pad(crop, pads, mode=pad_mode)
        except ValueError:
            crop = np.pad(crop, pads, mode="constant")
    return crop, (x0, y0)

def transform_relative_positions(
    positions: np.ndarray,
    hflip: bool,
    vflip: bool,
    rotations_ccw: int,
) -> np.ndarray:
    """Apply the same square-image geometry transform to relative XY positions."""
    output = np.asarray(positions, dtype=np.float32).copy()
    if output.size == 0:
        return output

    valid = np.isfinite(output).all(axis=1)
    xy = output[valid]

    if hflip:
        xy[:, 0] = 1.0 - xy[:, 0]
    if vflip:
        xy[:, 1] = 1.0 - xy[:, 1]

    k = int(rotations_ccw) % 4
    x = xy[:, 0].copy()
    y = xy[:, 1].copy()
    if k == 1:
        xy[:, 0] = y
        xy[:, 1] = 1.0 - x
    elif k == 2:
        xy[:, 0] = 1.0 - x
        xy[:, 1] = 1.0 - y
    elif k == 3:
        xy[:, 0] = 1.0 - y
        xy[:, 1] = x

    output[valid] = np.clip(xy, 0.0, 1.0)
    return output


def _spatially_balanced_indices(
    positions: np.ndarray,
    maximum: int,
    *,
    randomize: bool,
    rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Select at most ``maximum`` spots while preserving spatial coverage.

    We lexicographically sort spots by coarse spatial bins and then take evenly
    spaced samples. During training a random phase offset is allowed, but the
    selection remains spatially spread rather than becoming a pure random draw.
    """
    count = int(len(positions))
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    if maximum <= 0:
        raise ValueError("maximum must be positive")

    xy = np.asarray(positions, dtype=np.float32)
    bins = max(4, int(np.ceil(np.sqrt(maximum))))
    bx = np.clip((xy[:, 0] * bins).astype(np.int64), 0, bins - 1)
    by = np.clip((xy[:, 1] * bins).astype(np.int64), 0, bins - 1)
    order = np.lexsort((xy[:, 0], xy[:, 1], bx, by))

    if randomize:
        rng = rng or np.random.default_rng()
        phase = float(rng.random())
    else:
        phase = 0.5

    locations = ((np.arange(maximum, dtype=np.float64) + phase) * count / maximum)
    locations = np.clip(locations.astype(np.int64), 0, count - 1)
    return order[locations]


def extract_aligned_context(
    cache: Dict[str, np.ndarray],
    patch_x: int,
    patch_y: int,
    cfg: Any,
    *,
    is_train: bool,
    keep_probability: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Extract one aligned H&E context crop and padded ST spot set.

    Required cache entries:
    ``thumb_native``, ``tissue_thumb_native``, ``coords_thumb_native``,
    ``features``, and ``cache_downsample``.
    """
    patch_size = int(cfg.geometry.patch_size)
    ctx_scale = float(cfg.geometry.ctx_scale)
    cache_ds = int(cache.get("cache_downsample", cfg.geometry.downsample))
    target_context_size = int(
        getattr(
            cfg.geometry,
            "context_image_size",
            round(patch_size * ctx_scale / cache_ds),
        )
    )

    max_spots = int(getattr(cfg.data, "max_context_spots", 1024))
    if max_spots <= 0:
        raise ValueError("data.max_context_spots must be positive")

    thumb_native = np.asarray(cache["thumb_native"], dtype=np.uint8)
    tissue_native = np.asarray(cache["tissue_thumb_native"], dtype=np.uint8)
    thumb_height, thumb_width = thumb_native.shape[:2]
    full_width = float(cache.get("full_width", round(thumb_width * cache_ds)))
    full_height = float(cache.get("full_height", round(thumb_height * cache_ds)))
    if full_width <= 0 or full_height <= 0:
        raise ValueError("Invalid full slide dimensions in context cache")

    scale_x = float(thumb_width) / full_width
    scale_y = float(thumb_height) / full_height
    physical_context = float(patch_size) * ctx_scale
    native_width = max(1, int(round(physical_context * scale_x)))
    native_height = max(1, int(round(physical_context * scale_y)))
    center_x = int(round((float(patch_x) + patch_size / 2.0) * scale_x))
    center_y = int(round((float(patch_y) + patch_size / 2.0) * scale_y))

    context_rgb_native, origin = _safe_rect_window(
        thumb_native,
        center_x,
        center_y,
        native_width,
        native_height,
        pad_mode="constant",
    )
    context_tissue_native, _ = _safe_rect_window(
        tissue_native,
        center_x,
        center_y,
        native_width,
        native_height,
        pad_mode="constant",
    )
    if context_rgb_native.shape[:2] != (target_context_size, target_context_size):
        context_rgb = cv2.resize(
            context_rgb_native,
            (target_context_size, target_context_size),
            interpolation=cv2.INTER_AREA,
        )
        context_tissue = cv2.resize(
            context_tissue_native,
            (target_context_size, target_context_size),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        context_rgb = context_rgb_native
        context_tissue = context_tissue_native

    coordinates = np.asarray(cache["coords_thumb_native"], dtype=np.float32)
    features = np.asarray(cache["features"], dtype=np.float32)
    if coordinates.shape != (features.shape[0], 2):
        raise ValueError(
            f"Coordinate/feature mismatch: {coordinates.shape} vs {features.shape}"
        )

    x0, y0 = origin
    relative = np.empty_like(coordinates, dtype=np.float32)
    relative[:, 0] = (coordinates[:, 0] - float(x0)) / float(native_width)
    relative[:, 1] = (coordinates[:, 1] - float(y0)) / float(native_height)
    inside = (
        (relative[:, 0] >= 0.0)
        & (relative[:, 0] < 1.0)
        & (relative[:, 1] >= 0.0)
        & (relative[:, 1] < 1.0)
    )

    selected_features = features[inside]
    selected_positions = relative[inside]

    if is_train and keep_probability < 1.0 and len(selected_features):
        rng = rng or np.random.default_rng()
        keep = rng.random(len(selected_features)) < float(np.clip(keep_probability, 0.0, 1.0))
        # Keep at least one spot when a non-empty context was selected.
        if not bool(keep.any()):
            keep[int(rng.integers(0, len(keep)))] = True
        selected_features = selected_features[keep]
        selected_positions = selected_positions[keep]

    if len(selected_features) > max_spots:
        indices = _spatially_balanced_indices(
            selected_positions,
            max_spots,
            randomize=is_train,
            rng=rng,
        )
        selected_features = selected_features[indices]
        selected_positions = selected_positions[indices]

    feature_dim = int(features.shape[1])
    padded_features = np.zeros((max_spots, feature_dim), dtype=np.float32)
    padded_positions = np.zeros((max_spots, 2), dtype=np.float32)
    padded_mask = np.zeros((max_spots,), dtype=np.float32)

    count = int(len(selected_features))
    if count:
        padded_features[:count] = selected_features
        padded_positions[:count] = selected_positions
        padded_mask[:count] = 1.0

    return {
        "he_context": np.ascontiguousarray(context_rgb, dtype=np.uint8),
        "context_tissue": np.ascontiguousarray(context_tissue, dtype=np.uint8),
        "spot_features": padded_features,
        "spot_positions": padded_positions,
        "spot_mask": padded_mask,
        "spot_count": np.asarray(count, dtype=np.int64),
        "context_origin_thumb": np.asarray(origin, dtype=np.int64),
        "context_native_size_wh": np.asarray(
            [native_width, native_height], dtype=np.int64
        ),
    }

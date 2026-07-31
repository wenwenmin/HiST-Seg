from __future__ import annotations

import argparse
import copy
import json
import math
import os

import cv2
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree


DEFAULT_SLICES = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]


class AutoScaler:
    """Scale coordinates temporarily while preserving their original geometry."""

    def __init__(self, target_spacing: float = 2.0):
        if target_spacing <= 0:
            raise ValueError("target_spacing must be positive")
        self.target_spacing = float(target_spacing)
        self.scale_factor = 1.0
        self.x_min = 0.0
        self.y_min = 0.0
        self.original_median_spacing = 1.0

    def fit_transform(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Expected coordinates with shape (N, 2), got {coords.shape}")
        if len(coords) < 2:
            return coords.copy()

        tree = KDTree(coords)
        dists, _ = tree.query(coords, k=2)
        median_spacing = float(np.median(dists[:, 1]))
        if not np.isfinite(median_spacing) or median_spacing <= 0:
            median_spacing = 1.0

        self.original_median_spacing = median_spacing
        self.scale_factor = self.target_spacing / median_spacing
        self.x_min = float(coords[:, 0].min())
        self.y_min = float(coords[:, 1].min())
        return (coords - [self.x_min, self.y_min]) * self.scale_factor

    def inverse_transform(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64)
        return coords / self.scale_factor + [self.x_min, self.y_min]


class GridGeometry:
    """Geometry required for interpolation at one adaptive density."""

    def __init__(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        nearest_indices: np.ndarray,
        background_mask: np.ndarray,
        hr_grid_x: np.ndarray,
        hr_grid_y: np.ndarray,
        valid_mask: np.ndarray,
        new_h: int,
        new_w: int,
        target_spacing: float,
        background_radius: float,
        valid_radius: float,
    ):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.nearest_indices = nearest_indices
        self.background_mask = background_mask
        self.hr_grid_x = hr_grid_x
        self.hr_grid_y = hr_grid_y
        self.valid_mask = valid_mask
        self.new_h = int(new_h)
        self.new_w = int(new_w)
        self.target_spacing = float(target_spacing)
        self.background_radius = float(background_radius)
        self.valid_radius = float(valid_radius)

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def valid_coords(self) -> np.ndarray:
        flat_coords = np.column_stack(
            (self.hr_grid_x.ravel(), self.hr_grid_y.ravel())
        )
        return flat_coords[self.valid_mask.ravel()]


def extract_raw_coordinates(adata: sc.AnnData) -> np.ndarray:
    """Return full-resolution pixel coordinates aligned with the raw H&E image.

    This follows the crop/resize script: ``adata.obsm["spatial"]`` is
    preferred. If it is unavailable, the Visium full-resolution pixel columns
    are used directly. No ``/16`` conversion is applied, because crop and
    resize coordinates must remain in the same pixel coordinate system as
    ``he-raw.jpg``.
    """
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        source = "obsm['spatial']"
    elif {
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    }.issubset(set(adata.obs.columns)):
        coords = np.stack(
            [
                np.asarray(adata.obs["pxl_col_in_fullres"], dtype=np.float64),
                np.asarray(adata.obs["pxl_row_in_fullres"], dtype=np.float64),
            ],
            axis=1,
        )
        source = "obs full-resolution pixel columns"
    else:
        raise KeyError(
            "No image-aligned coordinates found. Expected obsm['spatial'] or "
            "obs['pxl_col_in_fullres']/obs['pxl_row_in_fullres']."
        )

    if coords.shape != (adata.n_obs, 2):
        raise ValueError(
            f"Coordinate shape {coords.shape} does not match AnnData observations "
            f"({adata.n_obs}, 2)"
        )
    if not np.isfinite(coords).all():
        raise ValueError("Coordinates contain NaN or Inf")
    print(f"  coordinate source: {source}")
    return coords


def build_grid_geometry(
    norm_coords: np.ndarray,
    target_spacing: float,
    base_spacing: float = 2.0,
    delta_x: float = 1.0,
    delta_y: float = 1.0,
) -> GridGeometry:
    """Build interpolation geometry without touching the expression matrix.

    The original code used target_spacing=2.0 together with fixed radii 3.0 and
    1.6. When target_spacing changes, all geometric radii and margins are scaled
    by the same ratio. This changes sampling density while preserving the same
    tissue support in the original physical coordinate system.
    """
    if len(norm_coords) == 0:
        raise ValueError("No coordinates available")

    ratio = float(target_spacing) / float(base_spacing)
    margin = 2.0 * ratio
    background_radius = 3.0 * ratio
    valid_radius = 1.6 * ratio

    x_min = float(norm_coords[:, 0].min() - margin)
    y_min = float(norm_coords[:, 1].min() - margin)
    x_max = float(norm_coords[:, 0].max() + margin)
    y_max = float(norm_coords[:, 1].max() + margin)

    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError("Degenerate coordinate extent")

    grid_x, grid_y = np.mgrid[x_min:x_max:delta_x, y_min:y_max:delta_y]
    if grid_x.size == 0:
        raise RuntimeError("Generated an empty coarse grid")

    for row in range(1, grid_y.shape[0], 2):
        grid_y[row] += delta_y / 2.0

    tree = KDTree(norm_coords)
    coarse_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    coarse_dists, nearest_indices = tree.query(coarse_points, k=1)
    background_mask = (coarse_dists > background_radius).reshape(grid_x.shape)

    h, w = grid_x.shape
    new_h, new_w = 2 * h - 1, 2 * w - 1

    y_range = np.arange(y_min, y_max, delta_y)
    if y_range.size == 0:
        raise RuntimeError("Generated an empty high-resolution y range")

    hr_grid_x, hr_grid_y = np.mgrid[
        x_min:x_max:delta_x / 2.0,
        y_min:y_range[-1] + delta_y:delta_y / 2.0,
    ]

    for row in range(1, hr_grid_y.shape[0], 2):
        hr_grid_y[row] -= delta_y / 4.0
    for row in range(2, hr_grid_y.shape[0], 4):
        hr_grid_y[row : row + 2] += delta_y / 2.0

    hr_points = np.column_stack((hr_grid_x.ravel(), hr_grid_y.ravel()))
    hr_dists, _ = tree.query(hr_points, k=1)
    valid_mask = (hr_dists <= valid_radius).reshape(hr_grid_x.shape)

    return GridGeometry(
        grid_x=grid_x,
        grid_y=grid_y,
        nearest_indices=np.asarray(nearest_indices, dtype=np.int64),
        background_mask=background_mask,
        hr_grid_x=hr_grid_x,
        hr_grid_y=hr_grid_y,
        valid_mask=valid_mask,
        new_h=new_h,
        new_w=new_w,
        target_spacing=target_spacing,
        background_radius=background_radius,
        valid_radius=valid_radius,
    )


def evaluate_spacing(raw_coords: np.ndarray, spacing: float) -> Tuple[int, AutoScaler, GridGeometry]:
    scaler = AutoScaler(target_spacing=spacing)
    norm_coords = scaler.fit_transform(raw_coords)
    geometry = build_grid_geometry(norm_coords, target_spacing=spacing)
    return geometry.valid_count, scaler, geometry


def find_adaptive_spacing(
    raw_coords: np.ndarray,
    min_spots: int = 3500,
    max_spots: int = 4500,
    target_spots: int = 4000,
    initial_spacing: float = 2.0,
    min_spacing: float = 0.25,
    max_spacing: float = 12.0,
    max_search_steps: int = 18,
) -> Tuple[AutoScaler, GridGeometry, List[Dict[str, float]]]:
    """Search a slice-specific target_spacing using only coordinate geometry."""
    if not (0 < min_spots <= target_spots <= max_spots):
        raise ValueError("Require 0 < min_spots <= target_spots <= max_spots")

    cache: Dict[float, Tuple[int, AutoScaler, GridGeometry]] = {}
    history: List[Dict[str, float]] = []

    def evaluate(value: float) -> Tuple[int, AutoScaler, GridGeometry]:
        value = float(np.clip(value, min_spacing, max_spacing))
        key = round(value, 8)
        if key not in cache:
            count, scaler, geometry = evaluate_spacing(raw_coords, value)
            cache[key] = (count, scaler, geometry)
            history.append({"target_spacing": value, "spot_count": int(count)})
            print(f"  geometry search: target_spacing={value:.6f}, spots={count}")
        return cache[key]

    initial_count, _, _ = evaluate(initial_spacing)

    if initial_count < target_spots:
        low = float(initial_spacing)
        high = float(initial_spacing)
        high_count = initial_count
        while high_count < target_spots and high < max_spacing:
            high = min(max_spacing, high * 1.4)
            high_count, _, _ = evaluate(high)
            if high >= max_spacing:
                break
        if high_count < min_spots:
            raise RuntimeError(
                f"Even target_spacing={high:.3f} produced only {high_count} spots; "
                f"cannot reach the requested minimum {min_spots}."
            )
    else:
        high = float(initial_spacing)
        low = float(initial_spacing)
        low_count = initial_count
        while low_count > target_spots and low > min_spacing:
            low = max(min_spacing, low / 1.4)
            low_count, _, _ = evaluate(low)
            if low <= min_spacing:
                break
        if low_count > max_spots:
            print(
                f"  [warn] minimum spacing still gives {low_count} spots; "
                "a deterministic spatial reduction will be applied."
            )

    for _ in range(max_search_steps):
        mid = (low + high) / 2.0
        mid_count, _, _ = evaluate(mid)
        if mid_count < target_spots:
            low = mid
        else:
            high = mid
        if abs(high - low) < 1e-4:
            break

    candidates = list(cache.values())
    in_range = [item for item in candidates if min_spots <= item[0] <= max_spots]
    if in_range:
        best = min(in_range, key=lambda item: abs(item[0] - target_spots))
    else:
        above = [item for item in candidates if item[0] >= target_spots]
        if above:
            best = min(above, key=lambda item: item[0] - target_spots)
        else:
            best = min(candidates, key=lambda item: abs(item[0] - target_spots))

    best_count, best_scaler, best_geometry = best
    print(
        f"  selected target_spacing={best_geometry.target_spacing:.6f}, "
        f"geometric spots={best_count}"
    )
    return best_scaler, best_geometry, history


def deterministic_spatial_reduce(
    coords: np.ndarray,
    target_count: int,
    seed: int,
) -> np.ndarray:
    """Select a reproducible, spatially spread subset when geometry overshoots."""
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    if target_count >= n:
        return np.arange(n, dtype=np.int64)
    if target_count <= 0:
        raise ValueError("target_count must be positive")

    x_span = max(float(np.ptp(coords[:, 0])), 1e-8)
    y_span = max(float(np.ptp(coords[:, 1])), 1e-8)
    aspect = x_span / y_span
    nx = max(1, int(round(math.sqrt(target_count * aspect))))
    ny = max(1, int(math.ceil(target_count / nx)))

    x_norm = (coords[:, 0] - coords[:, 0].min()) / x_span
    y_norm = (coords[:, 1] - coords[:, 1].min()) / y_span
    bx = np.minimum((x_norm * nx).astype(int), nx - 1)
    by = np.minimum((y_norm * ny).astype(int), ny - 1)
    cell_id = bx * ny + by

    selected: List[int] = []
    for cid in np.unique(cell_id):
        members = np.flatnonzero(cell_id == cid)
        cx = (bx[members[0]] + 0.5) / nx
        cy = (by[members[0]] + 0.5) / ny
        dist2 = (x_norm[members] - cx) ** 2 + (y_norm[members] - cy) ** 2
        selected.append(int(members[np.argmin(dist2)]))

    selected = list(dict.fromkeys(selected))
    rng = np.random.default_rng(seed)

    if len(selected) > target_count:
        chosen = rng.choice(np.asarray(selected), size=target_count, replace=False)
        return np.sort(chosen.astype(np.int64))

    if len(selected) < target_count:
        selected_set = set(selected)
        remaining = np.asarray(
            [idx for idx in range(n) if idx not in selected_set], dtype=np.int64
        )
        extra = rng.choice(
            remaining, size=target_count - len(selected), replace=False
        )
        selected.extend(extra.tolist())

    return np.sort(np.asarray(selected[:target_count], dtype=np.int64))


def interpolate_expression_in_chunks(
    expression,
    geometry: GridGeometry,
    selected_valid_indices: Optional[np.ndarray],
    chunk_size: int = 256,
) -> sp.csr_matrix:
    """Interpolate expression in gene chunks to avoid a huge dense tensor."""
    n_genes = int(expression.shape[1])
    nearest = geometry.nearest_indices
    background_flat = geometry.background_mask.ravel()
    valid_flat = geometry.valid_mask.ravel()
    hr_h, hr_w = geometry.hr_grid_x.shape

    output_chunks: List[sp.csr_matrix] = []
    print(
        f"Interpolating {n_genes} genes in chunks of {chunk_size}; "
        f"valid geometric spots={geometry.valid_count}"
    )

    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        if sp.issparse(expression):
            source_chunk = expression[:, start:end].toarray()
        else:
            source_chunk = np.asarray(expression[:, start:end])
        source_chunk = np.asarray(source_chunk, dtype=np.float32)

        coarse = source_chunk[nearest, :]
        coarse[background_flat, :] = 0.0
        coarse = coarse.T.reshape(end - start, *geometry.grid_x.shape)

        tensor = torch.from_numpy(coarse).unsqueeze(0)
        hr_tensor = F.interpolate(
            tensor,
            size=(geometry.new_h, geometry.new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pad_h = hr_h - hr_tensor.shape[1]
        pad_w = hr_w - hr_tensor.shape[2]
        if pad_h > 0 or pad_w > 0:
            hr_tensor = F.pad(
                hr_tensor,
                (0, max(0, pad_w), 0, max(0, pad_h)),
                mode="constant",
                value=0.0,
            )
        hr_tensor = hr_tensor[:, :hr_h, :hr_w]

        chunk_values = (
            hr_tensor.numpy().reshape(end - start, -1)[:, valid_flat].T
        )
        if selected_valid_indices is not None:
            chunk_values = chunk_values[selected_valid_indices]

        chunk_values[np.abs(chunk_values) < 1e-8] = 0.0
        output_chunks.append(sp.csr_matrix(chunk_values, dtype=np.float32))
        print(f"  genes {start + 1}-{end}/{n_genes}")

    return sp.hstack(output_chunks, format="csr", dtype=np.float32)



def cv_imread(file_path: str, flags: int = cv2.IMREAD_COLOR):
    """Read images safely from paths that may contain non-ASCII characters."""
    if not os.path.exists(file_path):
        return None
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)


def cv_imwrite(save_path: str, image: np.ndarray) -> None:
    """Write images safely to paths that may contain non-ASCII characters."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    extension = os.path.splitext(save_path)[1]
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode image: {save_path}")
    buffer.tofile(save_path)


def validate_image_alignment(
    he_image: np.ndarray,
    label_image: np.ndarray,
    tissue_image: np.ndarray,
    slice_id: str,
) -> None:
    """Ensure the three source images share one full-resolution coordinate system."""
    he_h, he_w = he_image.shape[:2]
    label_h, label_w = label_image.shape[:2]
    tissue_h, tissue_w = tissue_image.shape[:2]
    if (label_h, label_w) != (he_h, he_w):
        raise ValueError(
            f"[{slice_id}] label image size {(label_w, label_h)} does not match "
            f"H&E size {(he_w, he_h)}"
        )
    if (tissue_h, tissue_w) != (he_h, he_w):
        raise ValueError(
            f"[{slice_id}] tissue-mask size {(tissue_w, tissue_h)} does not match "
            f"H&E size {(he_w, he_h)}"
        )


def compute_crop_and_resize_transform(
    coords_original: np.ndarray,
    image_width: int,
    image_height: int,
    padding: int,
    max_size: int,
) -> Dict[str, object]:
    """Compute the same crop translation and longest-side resize as attachment 2."""
    coords = np.asarray(coords_original, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError(f"Invalid coordinates for crop transform: {coords.shape}")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    min_x = int(np.floor(coords[:, 0].min()))
    max_x = int(np.ceil(coords[:, 0].max()))
    min_y = int(np.floor(coords[:, 1].min()))
    max_y = int(np.ceil(coords[:, 1].max()))

    start_x = max(0, min_x - int(padding))
    end_x = min(int(image_width), max_x + int(padding))
    start_y = max(0, min_y - int(padding))
    end_y = min(int(image_height), max_y + int(padding))

    if end_x <= start_x or end_y <= start_y:
        raise RuntimeError(
            f"Invalid crop box: x=[{start_x}, {end_x}), y=[{start_y}, {end_y})"
        )

    crop_width = end_x - start_x
    crop_height = end_y - start_y
    longest_side = max(crop_width, crop_height)
    scale = min(1.0, float(max_size) / float(longest_side))
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))

    coords_cropped = coords.copy()
    coords_cropped[:, 0] -= float(start_x)
    coords_cropped[:, 1] -= float(start_y)
    coords_resized = coords_cropped * float(scale)

    tolerance = 1.0
    if (
        coords_cropped[:, 0].min() < -tolerance
        or coords_cropped[:, 1].min() < -tolerance
        or coords_cropped[:, 0].max() > crop_width + tolerance
        or coords_cropped[:, 1].max() > crop_height + tolerance
    ):
        raise RuntimeError("Cropped coordinates fall outside the cropped image")
    if (
        coords_resized[:, 0].min() < -tolerance
        or coords_resized[:, 1].min() < -tolerance
        or coords_resized[:, 0].max() > resized_width + tolerance
        or coords_resized[:, 1].max() > resized_height + tolerance
    ):
        raise RuntimeError("Resized coordinates fall outside the resized image")

    return {
        "start_x": int(start_x),
        "end_x": int(end_x),
        "start_y": int(start_y),
        "end_y": int(end_y),
        "crop_width": int(crop_width),
        "crop_height": int(crop_height),
        "scale": float(scale),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "coords_cropped": coords_cropped.astype(np.float32),
        "coords_resized": coords_resized.astype(np.float32),
    }


def _set_primary_coordinates(
    adata: sc.AnnData,
    primary_coords: np.ndarray,
    coordinate_system: str,
) -> None:
    """Set coordinates used by generic readers and synchronized obs columns."""
    coords = np.asarray(primary_coords, dtype=np.float32)
    adata.obsm["spatial"] = coords.copy()
    adata.obs["spatial_x"] = coords[:, 0]
    adata.obs["spatial_y"] = coords[:, 1]
    adata.uns["active_coordinate_system"] = str(coordinate_system)


def create_coordinate_variant(
    hr_adata: sc.AnnData,
    primary_coords: np.ndarray,
    coordinate_system: str,
    transform: Dict[str, object],
    coords_original: np.ndarray,
    coords_cropped: Optional[np.ndarray] = None,
    coords_resized: Optional[np.ndarray] = None,
) -> sc.AnnData:
    """Create one coordinate variant while keeping peak memory bounded."""
    variant = hr_adata.copy()
    variant.obsm["spatial_original"] = np.asarray(coords_original, dtype=np.float32)
    if coords_cropped is not None:
        variant.obsm["spatial_cropped"] = np.asarray(coords_cropped, dtype=np.float32)
    if coords_resized is not None:
        variant.obsm["spatial_resized"] = np.asarray(coords_resized, dtype=np.float32)
    _set_primary_coordinates(variant, primary_coords, coordinate_system)

    transform_metadata = {
        "crop_start_xy": [int(transform["start_x"]), int(transform["start_y"])],
        "crop_end_xy_exclusive": [int(transform["end_x"]), int(transform["end_y"])],
        "cropped_size_wh": [int(transform["crop_width"]), int(transform["crop_height"])],
        "resize_scale": float(transform["scale"]),
        "resized_size_wh": [
            int(transform["resized_width"]),
            int(transform["resized_height"]),
        ],
        "formula_cropped": "[x_crop, y_crop] = [x_original-start_x, y_original-start_y]",
        "formula_resized": "[x_resized, y_resized] = [x_crop, y_crop] * scale",
    }
    variant.uns["crop_resize_transform"] = transform_metadata

    if coordinate_system == "resized":
        if "spatial" in variant.uns and isinstance(variant.uns["spatial"], dict):
            for _, library_data in variant.uns["spatial"].items():
                if not isinstance(library_data, dict):
                    continue
                scalefactors = library_data.get("scalefactors", {})
                if "spot_diameter_fullres" in scalefactors:
                    scalefactors["spot_diameter_resized"] = (
                        float(scalefactors["spot_diameter_fullres"])
                        * float(transform["scale"])
                    )
    return variant

def save_crop_and_resize_images(
    slice_id: str,
    he_image: np.ndarray,
    label_image: np.ndarray,
    tissue_image: np.ndarray,
    transform: Dict[str, object],
    cropped_output_dir: str,
    resized_output_dir: str,
) -> Dict[str, str]:
    """Save images using exactly the directory structure of attachment 2."""
    sx, ex = int(transform["start_x"]), int(transform["end_x"])
    sy, ey = int(transform["start_y"]), int(transform["end_y"])
    new_w = int(transform["resized_width"])
    new_h = int(transform["resized_height"])
    scale = float(transform["scale"])

    he_cropped = he_image[sy:ey, sx:ex]
    label_cropped = label_image[sy:ey, sx:ex]
    tissue_cropped = tissue_image[sy:ey, sx:ex]

    out_he_dir = os.path.join(cropped_output_dir, "he_cropped")
    out_label_dir = os.path.join(cropped_output_dir, "label_cropped")
    out_tissue_dir = os.path.join(cropped_output_dir, "tissue_mask_cropped")
    os.makedirs(out_he_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_tissue_dir, exist_ok=True)

    he_cropped_path = os.path.join(out_he_dir, f"{slice_id}_he_cropped.jpg")
    label_cropped_path = os.path.join(out_label_dir, f"{slice_id}_label_cropped.png")
    tissue_cropped_path = os.path.join(out_tissue_dir, f"{slice_id}.png")
    cv_imwrite(he_cropped_path, he_cropped)
    cv_imwrite(label_cropped_path, label_cropped)
    cv_imwrite(tissue_cropped_path, tissue_cropped)

    if scale < 1.0:
        he_resized = cv2.resize(
            he_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        label_resized = cv2.resize(
            label_cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )
        mask_resized = cv2.resize(
            tissue_cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        he_resized = he_cropped.copy()
        label_resized = label_cropped.copy()
        mask_resized = tissue_cropped.copy()

    slice_resized_dir = os.path.join(resized_output_dir, slice_id)
    os.makedirs(slice_resized_dir, exist_ok=True)
    he_resized_path = os.path.join(slice_resized_dir, "he_resized.png")
    label_resized_path = os.path.join(slice_resized_dir, "label_resized.png")
    mask_resized_path = os.path.join(slice_resized_dir, "mask_resized.png")
    cv_imwrite(he_resized_path, he_resized)
    cv_imwrite(label_resized_path, label_resized)
    cv_imwrite(mask_resized_path, mask_resized)

    return {
        "he_cropped": he_cropped_path,
        "label_cropped": label_cropped_path,
        "tissue_cropped": tissue_cropped_path,
        "he_resized": he_resized_path,
        "label_resized": label_resized_path,
        "mask_resized": mask_resized_path,
    }


def run_interpolation_crop_resize(
    slice_id: str,
    adata_path: str,
    he_path: str,
    label_path: str,
    tissue_path: str,
    original_output_dir: str,
    cropped_output_dir: str,
    resized_output_dir: str,
    min_spots: int,
    max_spots: int,
    target_spots: int,
    padding: int,
    max_size: int,
    seed: int,
    chunk_size: int,
    overwrite: bool,
) -> Dict[str, object]:
    """Interpolate once and save original/cropped/resized coordinate versions."""
    original_slice_dir = os.path.join(original_output_dir, slice_id)
    cropped_slice_dir = os.path.join(cropped_output_dir, slice_id)
    resized_slice_dir = os.path.join(resized_output_dir, slice_id)
    os.makedirs(original_slice_dir, exist_ok=True)
    os.makedirs(cropped_slice_dir, exist_ok=True)
    os.makedirs(resized_slice_dir, exist_ok=True)

    original_h5ad = os.path.join(original_slice_dir, "adata_HR_original.h5ad")
    cropped_h5ad = os.path.join(cropped_slice_dir, "adata_HR_cropped.h5ad")
    resized_h5ad = os.path.join(resized_slice_dir, "adata_HR_resized.h5ad")
    metadata_file = os.path.join(original_slice_dir, "interpolation_metadata.json")
    scalefactors_file = os.path.join(resized_slice_dir, "scalefactors.txt")

    required_outputs = [original_h5ad, cropped_h5ad, resized_h5ad]
    if all(os.path.exists(path) for path in required_outputs) and not overwrite:
        existing = sc.read_h5ad(original_h5ad, backed="r")
        n_obs = int(existing.n_obs)
        existing.file.close()
        print(f"Skip existing three-file output for {slice_id}")
        return {
            "slice": slice_id,
            "original_h5ad": original_h5ad,
            "cropped_h5ad": cropped_h5ad,
            "resized_h5ad": resized_h5ad,
            "output_spots": n_obs,
            "status": "skipped_existing",
        }

    he_image = cv_imread(he_path, cv2.IMREAD_COLOR)
    label_image = cv_imread(label_path, cv2.IMREAD_GRAYSCALE)
    tissue_image = cv_imread(tissue_path, cv2.IMREAD_GRAYSCALE)
    missing_images = [
        name
        for name, image in [
            (he_path, he_image),
            (label_path, label_image),
            (tissue_path, tissue_image),
        ]
        if image is None
    ]
    if missing_images:
        raise FileNotFoundError(f"[{slice_id}] missing image(s): {missing_images}")
    validate_image_alignment(he_image, label_image, tissue_image, slice_id)

    print(f"Loading AnnData from: {adata_path}")
    source_adata = sc.read_h5ad(adata_path)
    raw_coords = extract_raw_coordinates(source_adata)

    scaler, geometry, search_history = find_adaptive_spacing(
        raw_coords=raw_coords,
        min_spots=min_spots,
        max_spots=max_spots,
        target_spots=target_spots,
    )

    valid_norm_coords = geometry.valid_coords
    selected_indices: Optional[np.ndarray] = None
    if geometry.valid_count > max_spots:
        selected_indices = deterministic_spatial_reduce(
            valid_norm_coords, target_count=target_spots, seed=seed
        )
        final_norm_coords = valid_norm_coords[selected_indices]
        reduction_applied = True
        print(f"  spatial reduction: {geometry.valid_count} -> {len(final_norm_coords)}")
    else:
        final_norm_coords = valid_norm_coords
        reduction_applied = False

    final_count = len(final_norm_coords)
    if not min_spots <= final_count <= max_spots:
        raise RuntimeError(
            f"Final spot count {final_count} is outside [{min_spots}, {max_spots}]"
        )

    sparse_data = interpolate_expression_in_chunks(
        expression=source_adata.X,
        geometry=geometry,
        selected_valid_indices=selected_indices,
        chunk_size=chunk_size,
    )
    if sparse_data.shape != (final_count, source_adata.n_vars):
        raise RuntimeError(
            f"Interpolated matrix shape {sparse_data.shape} does not match "
            f"expected {(final_count, source_adata.n_vars)}"
        )

    coords_original = scaler.inverse_transform(final_norm_coords).astype(np.float32)
    image_height, image_width = he_image.shape[:2]
    transform = compute_crop_and_resize_transform(
        coords_original=coords_original,
        image_width=image_width,
        image_height=image_height,
        padding=padding,
        max_size=max_size,
    )
    coords_cropped = transform["coords_cropped"]
    coords_resized = transform["coords_resized"]

    obs = pd.DataFrame(index=[f"interp_{i:05d}" for i in range(final_count)])
    obs["is_interpolated"] = True
    base_hr = sc.AnnData(X=sparse_data, obs=obs, var=source_adata.var.copy())
    base_hr.var_names = source_adata.var_names.copy()
    base_hr.uns = copy.deepcopy(source_adata.uns)
    base_hr.uns["adaptive_interpolation"] = {
        "method": "adaptive target-spacing search + nearest/bilinear interpolation",
        "requested_min_spots": int(min_spots),
        "requested_max_spots": int(max_spots),
        "requested_target_spots": int(target_spots),
        "source_spots": int(source_adata.n_obs),
        "geometric_spots_before_reduction": int(geometry.valid_count),
        "final_spots": int(final_count),
        "target_spacing": float(geometry.target_spacing),
        "source_median_spacing": float(scaler.original_median_spacing),
        "normalization_scale_factor": float(scaler.scale_factor),
        "spatial_reduction_applied": bool(reduction_applied),
        "seed": int(seed),
    }

    image_outputs = save_crop_and_resize_images(
        slice_id=slice_id,
        he_image=he_image,
        label_image=label_image,
        tissue_image=tissue_image,
        transform=transform,
        cropped_output_dir=cropped_output_dir,
        resized_output_dir=resized_output_dir,
    )

    original_adata = create_coordinate_variant(
        hr_adata=base_hr,
        primary_coords=coords_original,
        coordinate_system="original_full_resolution",
        transform=transform,
        coords_original=coords_original,
    )
    original_adata.write_h5ad(original_h5ad, compression="gzip")
    del original_adata

    cropped_adata = create_coordinate_variant(
        hr_adata=base_hr,
        primary_coords=coords_cropped,
        coordinate_system="cropped",
        transform=transform,
        coords_original=coords_original,
        coords_cropped=coords_cropped,
    )
    cropped_adata.write_h5ad(cropped_h5ad, compression="gzip")
    del cropped_adata

    resized_adata = create_coordinate_variant(
        hr_adata=base_hr,
        primary_coords=coords_resized,
        coordinate_system="resized",
        transform=transform,
        coords_original=coords_original,
        coords_cropped=coords_cropped,
        coords_resized=coords_resized,
    )
    resized_adata.write_h5ad(resized_h5ad, compression="gzip")
    del resized_adata

    with open(scalefactors_file, "w", encoding="utf-8") as file:
        file.write(f"scale_factor: {float(transform['scale']):.6f}\n")
        file.write(f"resized: {'yes' if float(transform['scale']) < 1.0 else 'no'}\n")
        file.write(
            f"original_image_size_wh: {image_width}x{image_height}\n"
        )
        file.write(
            f"crop_box_xyxy_exclusive: {transform['start_x']},{transform['start_y']},"
            f"{transform['end_x']},{transform['end_y']}\n"
        )
        file.write(
            f"cropped_size_wh: {transform['crop_width']}x{transform['crop_height']}\n"
        )
        file.write(
            f"resized_size_wh: {transform['resized_width']}x{transform['resized_height']}\n"
        )

    metadata = {
        "slice": slice_id,
        "input_h5ad": str(adata_path),
        "input_he": str(he_path),
        "input_label": str(label_path),
        "input_tissue_mask": str(tissue_path),
        "source_spots": int(source_adata.n_obs),
        "output_spots_each_file": int(final_count),
        "genes": int(source_adata.n_vars),
        "requested_spot_range": [int(min_spots), int(max_spots)],
        "requested_target_spots": int(target_spots),
        "geometric_spots_before_reduction": int(geometry.valid_count),
        "target_spacing": float(geometry.target_spacing),
        "source_median_spacing": float(scaler.original_median_spacing),
        "spatial_reduction_applied": bool(reduction_applied),
        "search_history": search_history,
        "crop": {
            "padding": int(padding),
            "start_xy": [int(transform["start_x"]), int(transform["start_y"])],
            "end_xy_exclusive": [int(transform["end_x"]), int(transform["end_y"])],
            "size_wh": [int(transform["crop_width"]), int(transform["crop_height"])],
        },
        "resize": {
            "max_size": int(max_size),
            "scale": float(transform["scale"]),
            "size_wh": [
                int(transform["resized_width"]),
                int(transform["resized_height"]),
            ],
        },
        "outputs": {
            "original_h5ad": original_h5ad,
            "cropped_h5ad": cropped_h5ad,
            "resized_h5ad": resized_h5ad,
            **image_outputs,
            "scalefactors": scalefactors_file,
        },
    }
    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"  original h5ad: {original_h5ad}")
    print(f"  cropped h5ad : {cropped_h5ad}")
    print(f"  resized h5ad : {resized_h5ad}")
    print(
        f"  crop={transform['crop_width']}x{transform['crop_height']}, "
        f"scale={float(transform['scale']):.6f}, "
        f"resized={transform['resized_width']}x{transform['resized_height']}"
    )

    return {
        "slice": slice_id,
        "input": adata_path,
        "source_spots": int(source_adata.n_obs),
        "output_spots": int(final_count),
        "target_spacing": float(geometry.target_spacing),
        "crop_width": int(transform["crop_width"]),
        "crop_height": int(transform["crop_height"]),
        "resize_scale": float(transform["scale"]),
        "resized_width": int(transform["resized_width"]),
        "resized_height": int(transform["resized_height"]),
        "original_h5ad": original_h5ad,
        "cropped_h5ad": cropped_h5ad,
        "resized_h5ad": resized_h5ad,
        "status": "completed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptively interpolate STHBC to 3500-4500 spots, crop/resize the "
            "images, and save original/cropped/resized coordinate h5ad files."
        )
    )
    parser.add_argument(
        "--base-data-dir",
        default="./data/STHBC/her2st",
        help="Source h5ad root: {root}/{slice}/{slice}.h5ad",
    )
    parser.add_argument(
        "--he-dir",
        default="./data/STHBC/her2st",
        help="Raw H&E root: {root}/{slice}/he-raw.jpg",
    )
    parser.add_argument(
        "--label-dir",
        default="./data/STHBC/label_img",
        help="Label root: {root}/{slice}_label.png",
    )
    parser.add_argument(
        "--tissue-mask-dir",
        default="./data/STHBC/mask",
        help="Tissue-mask root: {root}/{slice}.png",
    )
    parser.add_argument(
        "--original-output-dir",
        default="./data/STHBC/ST",
        help="Original-coordinate HR h5ad root",
    )
    parser.add_argument(
        "--cropped-output-dir",
        default="./data/STHBC/cropped_data",
    )
    parser.add_argument(
        "--resized-output-dir",
        default="./data/STHBC/resized_data",
    )
    parser.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    parser.add_argument("--min-spots", type=int, default=3500)
    parser.add_argument("--max-spots", type=int, default=4500)
    parser.add_argument("--target-spots", type=int, default=4000)
    parser.add_argument("--padding", type=int, default=200)
    parser.add_argument("--max-size", type=int, default=6000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.min_spots <= args.target_spots <= args.max_spots):
        raise ValueError("Require 0 < min_spots <= target_spots <= max_spots")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if args.padding < 0:
        raise ValueError("padding must be non-negative")
    if args.max_size <= 0:
        raise ValueError("max_size must be positive")

    summaries: List[Dict[str, object]] = []
    for slice_id in args.slices:
        print("\n" + "=" * 100)
        print(f"Processing Slice: {slice_id}")
        print("=" * 100)

        adata_path = os.path.join(args.base_data_dir, slice_id, f"{slice_id}.h5ad")
        he_path = os.path.join(args.he_dir, slice_id, "he-raw.jpg")
        label_path = os.path.join(args.label_dir, f"{slice_id}_label.png")
        tissue_path = os.path.join(args.tissue_mask_dir, f"{slice_id}.png")

        missing = [
            path
            for path in [adata_path, he_path, label_path, tissue_path]
            if not os.path.exists(path)
        ]
        if missing:
            print(f"Warning: missing input(s), skip {slice_id}: {missing}")
            summaries.append(
                {"slice": slice_id, "status": "missing_input", "missing": ";".join(missing)}
            )
            continue

        try:
            result = run_interpolation_crop_resize(
                slice_id=slice_id,
                adata_path=adata_path,
                he_path=he_path,
                label_path=label_path,
                tissue_path=tissue_path,
                original_output_dir=args.original_output_dir,
                cropped_output_dir=args.cropped_output_dir,
                resized_output_dir=args.resized_output_dir,
                min_spots=args.min_spots,
                max_spots=args.max_spots,
                target_spots=args.target_spots,
                padding=args.padding,
                max_size=args.max_size,
                seed=args.seed,
                chunk_size=args.chunk_size,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"ERROR [{slice_id}]: {type(exc).__name__}: {exc}")
            result = {
                "slice": slice_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        summaries.append(result)

    os.makedirs(args.original_output_dir, exist_ok=True)
    summary_path = os.path.join(
        args.original_output_dir, "STHBC_HR_crop_resize_summary.csv"
    )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print(f"Summary saved to: {summary_path}")
    if len(summary_frame):
        print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
    """Use the same coordinate convention as the original script."""
    if {
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    }.issubset(set(adata.obs.columns)):
        coords = np.stack(
            [
                np.asarray(adata.obs["pxl_col_in_fullres"], dtype=np.float64) / 16.0,
                np.asarray(adata.obs["pxl_row_in_fullres"], dtype=np.float64) / 16.0,
            ],
            axis=1,
        )
    elif "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    else:
        raise KeyError(
            "No coordinates found. Expected obs['pxl_col_in_fullres'], "
            "obs['pxl_row_in_fullres'] or obsm['spatial']."
        )

    if coords.shape != (adata.n_obs, 2):
        raise ValueError(
            f"Coordinate shape {coords.shape} does not match AnnData observations "
            f"({adata.n_obs}, 2)"
        )
    if not np.isfinite(coords).all():
        raise ValueError("Coordinates contain NaN or Inf")
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

    # Preserve the original staggered-grid convention.
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

    # Build a bracket around the desired target count.
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
            # We can still use this geometry and reduce it spatially later.
            print(
                f"  [warn] minimum spacing still gives {low_count} spots; "
                "a deterministic spatial reduction will be applied."
            )

    # Binary search toward the center of the requested range.
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
        # Prefer a count above the target because it can be reduced without
        # inventing duplicate coordinates.
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

    # Stratify the plane into approximately target_count cells and choose the
    # point nearest to each occupied cell center. Remaining places are filled
    # reproducibly from unselected points.
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

        # Nearest-neighbor projection to the coarse grid, equivalent to the
        # original per-gene griddata(..., method='nearest'), but done once per chunk.
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

        # Very small interpolation noise should remain exactly zero.
        chunk_values[np.abs(chunk_values) < 1e-8] = 0.0
        output_chunks.append(sp.csr_matrix(chunk_values, dtype=np.float32))
        print(f"  genes {start + 1}-{end}/{n_genes}")

    return sp.hstack(output_chunks, format="csr", dtype=np.float32)


def run_interpolation_to_h5ad(
    adata_path: str,
    output_dir: str,
    min_spots: int,
    max_spots: int,
    target_spots: int,
    seed: int,
    chunk_size: int,
    overwrite: bool,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "HR_processed.h5ad")
    metadata_file = os.path.join(output_dir, "interpolation_metadata.json")

    if os.path.exists(output_file) and not overwrite:
        print(f"Skip existing output: {output_file}")
        existing = sc.read_h5ad(output_file, backed="r")
        result = {
            "input": adata_path,
            "output": output_file,
            "original_spots": None,
            "output_spots": int(existing.n_obs),
            "status": "skipped_existing",
        }
        existing.file.close()
        return result

    print(f"Loading AnnData from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    raw_coords = extract_raw_coordinates(adata)

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
        print(
            f"  spatial reduction: {geometry.valid_count} -> {len(final_norm_coords)}"
        )
    else:
        final_norm_coords = valid_norm_coords
        reduction_applied = False

    final_count = len(final_norm_coords)
    if not min_spots <= final_count <= max_spots:
        raise RuntimeError(
            f"Final spot count {final_count} is outside [{min_spots}, {max_spots}]."
        )

    sparse_data = interpolate_expression_in_chunks(
        expression=adata.X,
        geometry=geometry,
        selected_valid_indices=selected_indices,
        chunk_size=chunk_size,
    )
    if sparse_data.shape != (final_count, adata.n_vars):
        raise RuntimeError(
            f"Interpolated matrix shape {sparse_data.shape} does not match "
            f"expected {(final_count, adata.n_vars)}"
        )

    final_coords = scaler.inverse_transform(final_norm_coords).astype(np.float32)

    obs = pd.DataFrame(index=[f"interp_{i:05d}" for i in range(final_count)])
    obs["spatial_x"] = final_coords[:, 0]
    obs["spatial_y"] = final_coords[:, 1]
    obs["is_interpolated"] = True

    hr_adata = sc.AnnData(
        X=sparse_data,
        obs=obs,
        var=adata.var.copy(),
    )
    hr_adata.var_names = adata.var_names.copy()
    hr_adata.obsm["spatial"] = final_coords
    hr_adata.uns = copy.deepcopy(adata.uns)
    hr_adata.uns["adaptive_interpolation"] = {
        "method": "adaptive target-spacing search + nearest/bilinear interpolation",
        "requested_min_spots": int(min_spots),
        "requested_max_spots": int(max_spots),
        "requested_target_spots": int(target_spots),
        "original_spots": int(adata.n_obs),
        "geometric_spots_before_reduction": int(geometry.valid_count),
        "final_spots": int(final_count),
        "target_spacing": float(geometry.target_spacing),
        "original_median_spacing": float(scaler.original_median_spacing),
        "scale_factor": float(scaler.scale_factor),
        "background_radius_normalized": float(geometry.background_radius),
        "valid_radius_normalized": float(geometry.valid_radius),
        "spatial_reduction_applied": bool(reduction_applied),
        "seed": int(seed),
    }

    hr_adata.write(output_file, compression="gzip")

    metadata = {
        "input_file": str(adata_path),
        "output_file": str(output_file),
        "original_spots": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "requested_range": [int(min_spots), int(max_spots)],
        "requested_target": int(target_spots),
        "geometric_spots_before_reduction": int(geometry.valid_count),
        "final_spots": int(final_count),
        "target_spacing": float(geometry.target_spacing),
        "original_median_spacing": float(scaler.original_median_spacing),
        "scale_factor": float(scaler.scale_factor),
        "spatial_reduction_applied": bool(reduction_applied),
        "search_history": search_history,
    }
    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Success! High-resolution data saved to: {output_file}")
    print(
        f"Details - Original Spots: {adata.n_obs}, "
        f"High-Res Spots: {hr_adata.n_obs}, Genes: {hr_adata.n_vars}"
    )

    return {
        "input": adata_path,
        "output": output_file,
        "original_spots": int(adata.n_obs),
        "output_spots": int(hr_adata.n_obs),
        "target_spacing": float(geometry.target_spacing),
        "reduction_applied": bool(reduction_applied),
        "status": "completed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate STHBC slices while adaptively controlling each output "
            "slice to 3500-4500 spots."
        )
    )
    parser.add_argument(
        "--base-data-dir",
        default="D:/DNM_project(cancer detect)/data/HB/her2st",
    )
    parser.add_argument(
        "--base-output-dir",
        default=r"D:\Project_2\Pixel_Seg\data\STHBC\ST",
    )
    parser.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    parser.add_argument("--min-spots", type=int, default=3500)
    parser.add_argument("--max-spots", type=int, default=4500)
    parser.add_argument("--target-spots", type=int, default=4000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.min_spots <= args.target_spots <= args.max_spots):
        raise ValueError(
            "Require 0 < --min-spots <= --target-spots <= --max-spots"
        )
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    summaries: List[Dict[str, object]] = []
    for slice_id in args.slices:
        print("\n" + "=" * 100)
        print(f"Processing Slice: {slice_id}")
        print("=" * 100)
        adata_path = os.path.join(
            args.base_data_dir, slice_id, f"{slice_id}.h5ad"
        )
        output_dir = os.path.join(args.base_output_dir, slice_id)

        if not os.path.exists(adata_path):
            print(f"Warning: File not found at {adata_path}. Skipping.")
            summaries.append(
                {
                    "slice": slice_id,
                    "input": adata_path,
                    "output": None,
                    "status": "missing_input",
                }
            )
            continue

        result = run_interpolation_to_h5ad(
            adata_path=adata_path,
            output_dir=output_dir,
            min_spots=args.min_spots,
            max_spots=args.max_spots,
            target_spots=args.target_spots,
            seed=args.seed,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )
        result["slice"] = slice_id
        summaries.append(result)

    os.makedirs(args.base_output_dir, exist_ok=True)
    summary_path = os.path.join(
        args.base_output_dir, "STHBC_interpolation_summary.csv"
    )
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print(f"Summary saved to: {summary_path}")
    if summaries:
        print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
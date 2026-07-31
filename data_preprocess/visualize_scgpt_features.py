from __future__ import annotations

"""
Visualize 512-dimensional scGPT-spatial features generated from all available genes.

Expected cache
--------------
{context_cache_root}/{cache_tag}/{slice_id}_ctx.npz

Default cache tag:
    all_genes_scgpt512

The cache must contain:
    thumb           H x W x 3 RGB thumbnail
    tissue_thumb    H x W tissue mask (optional)
    coords_thumb    N x 2 thumbnail coordinates
    features        N x 512 scGPT embeddings

Visualization strategy
----------------------
1. Global PCA, global color scale
   - One slice-balanced PCA is fitted across all requested slices.
   - The same PCA axes and the same percentile color limits are used for every
     slice, so colors are comparable across slices.

2. Global PCA, local color scale
   - The PCA axes remain globally fixed.
   - Each slice uses its own percentile limits, revealing weak within-slice
     spatial variation that may be hidden by global scaling.

3. Per-slice local PCA
   - A separate PCA is fitted within each slice.
   - This is useful only for visual inspection; colors are NOT comparable
     between slices.

4. Raw 512-channel inspection
   - The 64 most variable embedding dimensions in each slice are shown in an
     8 x 8 grid.

The PCA operations in this script are only for visualization. They do not
modify the saved 512-dimensional features used by the downstream ST encoder.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


DEFAULT_SLICES = [
    "A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1",
    "ViHBC", "DuCIS", "XeHBC",
]


# =============================================================================
# Generic I/O
# =============================================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML root: {path}")

    return cfg


def resolve_config_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()

    local = Path(__file__).resolve().parent / value
    if local.exists():
        return local.resolve()

    raise FileNotFoundError(value)


def cv_imwrite(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    suffix = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(suffix, image)

    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")

    encoded.tofile(path)


# =============================================================================
# Cache registry
# =============================================================================

def build_cache_registry(
    cfg: Mapping[str, Any],
    cache_tag: str,
    slices: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    datasets = cfg.get("datasets", {})
    if not isinstance(datasets, Mapping):
        raise KeyError("cfg.datasets is required")

    registry: Dict[str, Dict[str, str]] = {}

    for dataset_name, dataset_cfg_obj in datasets.items():
        dataset_cfg = dict(dataset_cfg_obj)
        context_root = str(dataset_cfg.get("context_cache_root", ""))

        if not context_root:
            continue

        for slice_id in map(str, dataset_cfg.get("slices", [])):
            if slice_id not in slices:
                continue

            registry[slice_id] = {
                "dataset": str(dataset_name),
                "context_path": os.path.join(
                    context_root,
                    cache_tag,
                    f"{slice_id}_ctx.npz",
                ),
            }

    missing = sorted(set(slices) - set(registry))
    if missing:
        raise KeyError(
            f"Slices missing from YAML context-cache registry: {missing}"
        )

    return registry


# =============================================================================
# Geometry and feature maps
# =============================================================================

def robust_limits(
    values: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)

    low = np.nanpercentile(
        values,
        low_percentile,
        axis=0,
    )
    high = np.nanpercentile(
        values,
        high_percentile,
        axis=0,
    )

    high = np.where(high > low, high, low + 1.0)

    return low.astype(np.float32), high.astype(np.float32)


def scale_to_rgb(
    values: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    scaled = (
        np.asarray(values, dtype=np.float32)
        - np.asarray(low, dtype=np.float32)
    ) / (
        np.asarray(high, dtype=np.float32)
        - np.asarray(low, dtype=np.float32)
    )

    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def median_nearest_neighbor_distance(coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=np.float64)

    if len(coords) < 2:
        return 1.0

    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    result = float(np.median(distances[:, 1]))

    if not np.isfinite(result) or result <= 0:
        return 1.0

    return result


def nearest_spot_assignment(
    coords: np.ndarray,
    height: int,
    width: int,
    tissue_mask: Optional[np.ndarray],
    radius_factor: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    coords = np.asarray(coords, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Invalid coordinate shape: {coords.shape}")
    if len(coords) == 0:
        raise ValueError("No spots available for rasterization")

    yy, xx = np.mgrid[0:height, 0:width]
    pixels = np.column_stack((xx.ravel(), yy.ravel()))

    tree = cKDTree(coords)
    distances, indices = tree.query(pixels, k=1)

    spacing = median_nearest_neighbor_distance(coords)
    radius = max(1.0, spacing * float(radius_factor))

    valid = distances <= radius

    if tissue_mask is not None:
        mask = np.asarray(tissue_mask)

        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        valid &= mask.ravel() > 0

    return (
        indices.reshape(height, width),
        valid.reshape(height, width),
        radius,
    )


def make_rgb_feature_map(
    spot_rgb: np.ndarray,
    assignment: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    height, width = assignment.shape
    output = np.full((height, width, 3), 255, dtype=np.uint8)
    output[valid_mask] = spot_rgb[assignment[valid_mask]]
    return output


def overlay_rgb_feature_map(
    thumb_rgb: np.ndarray,
    feature_rgb: np.ndarray,
    valid_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    output = thumb_rgb.astype(np.float32).copy()

    output[valid_mask] = (
        (1.0 - alpha) * output[valid_mask]
        + alpha * feature_rgb[valid_mask].astype(np.float32)
    )

    return np.clip(output, 0, 255).astype(np.uint8)


# =============================================================================
# Plotting
# =============================================================================

def save_pc_triplet(
    output_path: str,
    pcs: np.ndarray,
    assignment: np.ndarray,
    valid_mask: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    title_prefix: str,
    scale_name: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for component, axis in enumerate(axes):
        dense = np.full(
            assignment.shape,
            np.nan,
            dtype=np.float32,
        )
        dense[valid_mask] = pcs[
            assignment[valid_mask],
            component,
        ]

        image = axis.imshow(
            dense,
            cmap="turbo",
            vmin=float(low[component]),
            vmax=float(high[component]),
        )

        axis.set_title(
            f"{title_prefix} PC{component + 1} — {scale_name}"
        )
        axis.axis("off")
        fig.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.02,
        )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_comparison_panel(
    output_path: str,
    thumb_rgb: np.ndarray,
    global_overlay: np.ndarray,
    local_scale_overlay: np.ndarray,
    local_pca_overlay: np.ndarray,
    slice_id: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    panels = [
        (thumb_rgb, f"{slice_id}: H&E"),
        (
            global_overlay,
            "Global PCA\nGlobal scale",
        ),
        (
            local_scale_overlay,
            "Global PCA\nSlice-local scale",
        ),
        (
            local_pca_overlay,
            "Slice-local PCA\nSlice-local scale",
        ),
    ]

    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_top_channel_grid(
    output_path: str,
    features: np.ndarray,
    assignment: np.ndarray,
    valid_mask: np.ndarray,
    top_n: int,
    slice_id: str,
    dpi: int,
) -> List[int]:
    channel_variance = np.var(features, axis=0)
    selected = np.argsort(-channel_variance)[
        : min(int(top_n), features.shape[1])
    ]

    count = len(selected)
    columns = 8
    rows = int(np.ceil(count / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(24, 3 * rows),
        squeeze=False,
    )

    for panel_index, axis in enumerate(axes.ravel()):
        if panel_index >= count:
            axis.axis("off")
            continue

        channel = int(selected[panel_index])
        values = features[:, channel]

        low, high = np.percentile(values, [1.0, 99.0])
        if not np.isfinite(low):
            low = float(np.min(values))
        if not np.isfinite(high) or high <= low:
            high = low + 1.0

        dense = np.full(
            assignment.shape,
            np.nan,
            dtype=np.float32,
        )
        dense[valid_mask] = values[
            assignment[valid_mask]
        ]

        axis.imshow(
            dense,
            cmap="viridis",
            vmin=float(low),
            vmax=float(high),
        )
        axis.set_title(
            f"F{channel + 1:03d}\n"
            f"Var={channel_variance[channel]:.3g}",
            fontsize=8,
        )
        axis.axis("off")

    fig.suptitle(
        f"{slice_id}: top {count} variable dimensions "
        "of the 512-D all-gene scGPT embedding",
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return selected.astype(int).tolist()


# =============================================================================
# Feature diagnostics
# =============================================================================

def median_knn_cosine_similarity(
    features: np.ndarray,
    coords: np.ndarray,
    neighbors: int = 6,
) -> float:
    if len(features) <= 1:
        return float("nan")

    k = min(int(neighbors) + 1, len(features))
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k)

    if indices.ndim == 1:
        return float("nan")

    indices = indices[:, 1:]

    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True,
    )
    normalized = features / np.maximum(norms, 1e-12)

    all_values: List[np.ndarray] = []

    for center in range(len(features)):
        similarity = (
            normalized[indices[center]]
            @ normalized[center]
        )
        all_values.append(similarity)

    return float(
        np.median(np.concatenate(all_values))
    )


def feature_summary(features: np.ndarray) -> Dict[str, Any]:
    dimension_std = np.std(features, axis=0)
    spot_norm = np.linalg.norm(features, axis=1)

    return {
        "feature_shape": list(features.shape),
        "feature_global_mean": float(np.mean(features)),
        "feature_global_std": float(np.std(features)),
        "median_dimension_std": float(
            np.median(dimension_std)
        ),
        "minimum_dimension_std": float(
            np.min(dimension_std)
        ),
        "maximum_dimension_std": float(
            np.max(dimension_std)
        ),
        "zero_variance_dimensions": int(
            np.count_nonzero(dimension_std < 1e-8)
        ),
        "median_spot_l2_norm": float(
            np.median(spot_norm)
        ),
    }


# =============================================================================
# Loading and PCA
# =============================================================================

def load_context_cache(
    slice_id: str,
    dataset_name: str,
    path: str,
) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as cache:
        required = {
            "thumb",
            "coords_thumb",
            "features",
        }
        missing = sorted(required - set(cache.files))

        if missing:
            raise KeyError(
                f"[{slice_id}] cache is missing keys: {missing}"
            )

        thumb = np.asarray(
            cache["thumb"],
            dtype=np.uint8,
        )
        tissue = (
            np.asarray(cache["tissue_thumb"], dtype=np.uint8)
            if "tissue_thumb" in cache.files
            else None
        )
        coords = np.asarray(
            cache["coords_thumb"],
            dtype=np.float32,
        )
        features = np.asarray(
            cache["features"],
            dtype=np.float32,
        )

        optional: Dict[str, Any] = {}

        for key in (
            "input_gene_count",
            "max_length",
            "spots_exceeding_context_capacity",
            "feature_dim",
            "full_h",
            "full_w",
            "ds",
        ):
            if key in cache.files:
                optional[key] = int(
                    np.asarray(cache[key]).item()
                )

        for key in (
            "feature_type",
            "preprocessing_mode",
            "source_h5ad",
            "cache_tag",
        ):
            if key in cache.files:
                optional[key] = str(
                    np.asarray(cache[key]).item()
                )

    if features.ndim != 2:
        raise ValueError(
            f"[{slice_id}] feature matrix must be 2-D, "
            f"got {features.shape}"
        )
    if features.shape[1] != 512:
        raise ValueError(
            f"[{slice_id}] expected 512-D features, "
            f"got {features.shape[1]}"
        )
    if len(coords) != len(features):
        raise ValueError(
            f"[{slice_id}] coords={len(coords)} but "
            f"features={len(features)}"
        )
    if not np.isfinite(features).all():
        raise ValueError(
            f"[{slice_id}] features contain NaN/Inf"
        )
    if not np.isfinite(coords).all():
        raise ValueError(
            f"[{slice_id}] coordinates contain NaN/Inf"
        )

    height, width = thumb.shape[:2]

    in_bounds = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < width)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < height)
    )

    return {
        "slice": slice_id,
        "dataset": dataset_name,
        "path": path,
        "thumb": thumb,
        "tissue": tissue,
        "coords": coords,
        "features": features,
        "thumb_height": height,
        "thumb_width": width,
        "coords_in_bounds": in_bounds,
        **optional,
    }


def fit_slice_balanced_global_pca(
    bundles: Mapping[str, Mapping[str, Any]],
    max_spots_per_slice: int,
    seed: int,
) -> Tuple[PCA, Dict[str, np.ndarray], Dict[str, Any]]:
    counts = {
        slice_id: len(bundle["features"])
        for slice_id, bundle in bundles.items()
    }

    sample_count = min(counts.values())

    if max_spots_per_slice > 0:
        sample_count = min(
            sample_count,
            int(max_spots_per_slice),
        )

    rng = np.random.default_rng(seed)
    samples: List[np.ndarray] = []
    sampled_counts: Dict[str, int] = {}

    for slice_id, bundle in bundles.items():
        features = np.asarray(
            bundle["features"],
            dtype=np.float32,
        )

        if len(features) > sample_count:
            indices = rng.choice(
                len(features),
                size=sample_count,
                replace=False,
            )
            sample = features[indices]
        else:
            sample = features

        samples.append(sample)
        sampled_counts[slice_id] = len(sample)

    fit_matrix = np.concatenate(samples, axis=0)

    pca = PCA(
        n_components=3,
        random_state=seed,
    )
    pca.fit(fit_matrix)

    transformed = {
        slice_id: pca.transform(
            np.asarray(bundle["features"], dtype=np.float32)
        ).astype(np.float32)
        for slice_id, bundle in bundles.items()
    }

    metadata = {
        "fit_shape": list(fit_matrix.shape),
        "sampled_spots_by_slice": sampled_counts,
        "explained_variance_ratio": (
            pca.explained_variance_ratio_.tolist()
        ),
        "explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
    }

    return pca, transformed, metadata


def fit_local_pca(
    features: np.ndarray,
    seed: int,
) -> Tuple[PCA, np.ndarray]:
    pca = PCA(
        n_components=3,
        random_state=seed,
    )
    transformed = pca.fit_transform(
        np.asarray(features, dtype=np.float32)
    ).astype(np.float32)

    return pca, transformed


# =============================================================================
# CLI and main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize 512-D scGPT features generated from all "
            "vocabulary-compatible genes."
        )
    )

    parser.add_argument(
        "--config",
        default="../engine/cross_dataset.yaml",
    )
    parser.add_argument(
        "--cache-tag",
        default="all_genes_scgpt512_ds4",
    )
    parser.add_argument(
        "--slices",
        nargs="+",
        default=DEFAULT_SLICES,
    )
    parser.add_argument(
        "--output-dir",
        default=(
            " "
            "all_genes_scgpt512_visualization"
        ),
    )
    parser.add_argument(
        "--global-pca-spots-per-slice",
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.68,
    )
    parser.add_argument(
        "--raster-radius-factor",
        type=float,
        default=1.20,
    )
    parser.add_argument(
        "--top-channels",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
    )
    parser.add_argument(
        "--skip-channel-grid",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= float(args.overlay_alpha) <= 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1]")

    if not (
        0.0
        <= float(args.percentile_low)
        < float(args.percentile_high)
        <= 100.0
    ):
        raise ValueError(
            "Require 0 <= percentile-low < percentile-high <= 100"
        )

    config_path = resolve_config_path(args.config)
    cfg = load_yaml(config_path)

    slices = [str(value) for value in args.slices]

    registry = build_cache_registry(
        cfg=cfg,
        cache_tag=args.cache_tag,
        slices=slices,
    )

    print("=" * 100)
    print("Loading all-gene scGPT 512-D context caches")
    print("=" * 100)

    bundles: Dict[str, Dict[str, Any]] = {}

    for slice_id in slices:
        info = registry[slice_id]

        bundle = load_context_cache(
            slice_id=slice_id,
            dataset_name=info["dataset"],
            path=info["context_path"],
        )
        bundles[slice_id] = bundle

        input_gene_count = bundle.get(
            "input_gene_count",
            "unknown",
        )
        max_length = bundle.get(
            "max_length",
            "unknown",
        )
        truncated = bundle.get(
            "spots_exceeding_context_capacity",
            "unknown",
        )

        print(
            f"{slice_id:>6s} | dataset={bundle['dataset']:<6s} | "
            f"features={bundle['features'].shape} | "
            f"input_genes={input_gene_count} | "
            f"max_length={max_length} | "
            f"spots_over_capacity={truncated}"
        )

    print("\nFitting one slice-balanced global 512 -> 3 PCA ...")

    global_pca, global_pcs, global_pca_metadata = (
        fit_slice_balanced_global_pca(
            bundles=bundles,
            max_spots_per_slice=args.global_pca_spots_per_slice,
            seed=args.seed,
        )
    )

    all_global_pcs = np.concatenate(
        [global_pcs[slice_id] for slice_id in slices],
        axis=0,
    )

    global_low, global_high = robust_limits(
        all_global_pcs,
        args.percentile_low,
        args.percentile_high,
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_root / "global_visualization_pca_512_to_3.npz",
        components=global_pca.components_.astype(np.float32),
        mean=global_pca.mean_.astype(np.float32),
        explained_variance=global_pca.explained_variance_.astype(
            np.float32
        ),
        explained_variance_ratio=(
            global_pca.explained_variance_ratio_.astype(np.float32)
        ),
        global_pc_low=global_low,
        global_pc_high=global_high,
        slices=np.asarray(slices, dtype="U"),
    )

    summary_rows: List[Dict[str, Any]] = []

    for slice_id in slices:
        print("\n" + "-" * 100)
        print(f"Visualizing {slice_id}")

        bundle = bundles[slice_id]
        slice_dir = output_root / slice_id
        slice_dir.mkdir(parents=True, exist_ok=True)

        thumb = bundle["thumb"]
        tissue = bundle["tissue"]
        coords = bundle["coords"]
        features = bundle["features"]

        assignment, valid_mask, raster_radius = (
            nearest_spot_assignment(
                coords=coords,
                height=bundle["thumb_height"],
                width=bundle["thumb_width"],
                tissue_mask=tissue,
                radius_factor=args.raster_radius_factor,
            )
        )

        # ---------------------------------------------------------------------
        # Global PCA + global scale
        # ---------------------------------------------------------------------
        global_values = global_pcs[slice_id]

        global_rgb = scale_to_rgb(
            global_values,
            global_low,
            global_high,
        )
        global_map = make_rgb_feature_map(
            global_rgb,
            assignment,
            valid_mask,
        )
        global_overlay = overlay_rgb_feature_map(
            thumb,
            global_map,
            valid_mask,
            alpha=args.overlay_alpha,
        )

        cv_imwrite(
            str(slice_dir / "01_global_pca_rgb_map_global_scale.png"),
            cv2.cvtColor(global_map, cv2.COLOR_RGB2BGR),
        )
        cv_imwrite(
            str(slice_dir / "02_global_pca_rgb_overlay_global_scale.png"),
            cv2.cvtColor(global_overlay, cv2.COLOR_RGB2BGR),
        )

        save_pc_triplet(
            output_path=str(
                slice_dir / "03_global_pca_pc1_pc2_pc3_global_scale.png"
            ),
            pcs=global_values,
            assignment=assignment,
            valid_mask=valid_mask,
            low=global_low,
            high=global_high,
            title_prefix=slice_id,
            scale_name="global PCA / global scale",
            dpi=args.dpi,
        )

        # ---------------------------------------------------------------------
        # Global PCA + local slice scale
        # ---------------------------------------------------------------------
        local_global_low, local_global_high = robust_limits(
            global_values,
            args.percentile_low,
            args.percentile_high,
        )

        local_scale_rgb = scale_to_rgb(
            global_values,
            local_global_low,
            local_global_high,
        )
        local_scale_map = make_rgb_feature_map(
            local_scale_rgb,
            assignment,
            valid_mask,
        )
        local_scale_overlay = overlay_rgb_feature_map(
            thumb,
            local_scale_map,
            valid_mask,
            alpha=args.overlay_alpha,
        )

        cv_imwrite(
            str(slice_dir / "04_global_pca_rgb_map_local_scale.png"),
            cv2.cvtColor(local_scale_map, cv2.COLOR_RGB2BGR),
        )
        cv_imwrite(
            str(slice_dir / "05_global_pca_rgb_overlay_local_scale.png"),
            cv2.cvtColor(local_scale_overlay, cv2.COLOR_RGB2BGR),
        )

        save_pc_triplet(
            output_path=str(
                slice_dir / "06_global_pca_pc1_pc2_pc3_local_scale.png"
            ),
            pcs=global_values,
            assignment=assignment,
            valid_mask=valid_mask,
            low=local_global_low,
            high=local_global_high,
            title_prefix=slice_id,
            scale_name="global PCA / local scale",
            dpi=args.dpi,
        )

        # ---------------------------------------------------------------------
        # Per-slice PCA + local scale
        # ---------------------------------------------------------------------
        local_pca, local_pcs = fit_local_pca(
            features,
            seed=args.seed,
        )

        local_pca_low, local_pca_high = robust_limits(
            local_pcs,
            args.percentile_low,
            args.percentile_high,
        )

        local_pca_rgb = scale_to_rgb(
            local_pcs,
            local_pca_low,
            local_pca_high,
        )
        local_pca_map = make_rgb_feature_map(
            local_pca_rgb,
            assignment,
            valid_mask,
        )
        local_pca_overlay = overlay_rgb_feature_map(
            thumb,
            local_pca_map,
            valid_mask,
            alpha=args.overlay_alpha,
        )

        cv_imwrite(
            str(slice_dir / "07_local_pca_rgb_map.png"),
            cv2.cvtColor(local_pca_map, cv2.COLOR_RGB2BGR),
        )
        cv_imwrite(
            str(slice_dir / "08_local_pca_rgb_overlay.png"),
            cv2.cvtColor(local_pca_overlay, cv2.COLOR_RGB2BGR),
        )

        save_pc_triplet(
            output_path=str(
                slice_dir / "09_local_pca_pc1_pc2_pc3.png"
            ),
            pcs=local_pcs,
            assignment=assignment,
            valid_mask=valid_mask,
            low=local_pca_low,
            high=local_pca_high,
            title_prefix=slice_id,
            scale_name="slice-local PCA",
            dpi=args.dpi,
        )

        save_comparison_panel(
            output_path=str(
                slice_dir / "10_three_visualization_modes_comparison.png"
            ),
            thumb_rgb=thumb,
            global_overlay=global_overlay,
            local_scale_overlay=local_scale_overlay,
            local_pca_overlay=local_pca_overlay,
            slice_id=slice_id,
            dpi=args.dpi,
        )

        selected_channels: List[int] = []

        if not args.skip_channel_grid:
            selected_channels = save_top_channel_grid(
                output_path=str(
                    slice_dir / "11_top64_raw_embedding_channels.png"
                ),
                features=features,
                assignment=assignment,
                valid_mask=valid_mask,
                top_n=args.top_channels,
                slice_id=slice_id,
                dpi=args.dpi,
            )

        summary = feature_summary(features)
        knn_similarity = median_knn_cosine_similarity(
            features,
            coords,
        )

        report = {
            "slice": slice_id,
            "dataset": bundle["dataset"],
            "context_cache": bundle["path"],
            "input_gene_count": bundle.get("input_gene_count"),
            "max_length": bundle.get("max_length"),
            "spots_exceeding_context_capacity": bundle.get(
                "spots_exceeding_context_capacity"
            ),
            "preprocessing_mode": bundle.get("preprocessing_mode"),
            "feature_type": bundle.get("feature_type"),
            "coordinates_out_of_bounds": int(
                np.count_nonzero(~bundle["coords_in_bounds"])
            ),
            "raster_radius_thumb_px": float(raster_radius),
            **summary,
            "median_knn_cosine_similarity": knn_similarity,
            "global_pca_local_limits": {
                "low": local_global_low.tolist(),
                "high": local_global_high.tolist(),
            },
            "local_pca_explained_variance_ratio": (
                local_pca.explained_variance_ratio_.tolist()
            ),
            "local_pca_explained_variance_ratio_sum": float(
                local_pca.explained_variance_ratio_.sum()
            ),
            "top_variable_raw_channels_1_based": [
                int(index + 1) for index in selected_channels
            ],
            "interpretation_note": (
                "Global-PCA/global-scale maps are cross-slice comparable. "
                "Global-PCA/local-scale maps reveal within-slice variation "
                "but their colors are not quantitatively comparable across "
                "slices. Local-PCA colors are not comparable across slices."
            ),
        }

        with (
            slice_dir / "feature_visualization_report.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
            )

        summary_rows.append(
            {
                "slice": slice_id,
                "dataset": bundle["dataset"],
                "spots": features.shape[0],
                "feature_dim": features.shape[1],
                "input_gene_count": bundle.get("input_gene_count"),
                "max_length": bundle.get("max_length"),
                "spots_over_context_capacity": bundle.get(
                    "spots_exceeding_context_capacity"
                ),
                "coordinates_out_of_bounds": report[
                    "coordinates_out_of_bounds"
                ],
                "feature_global_std": summary[
                    "feature_global_std"
                ],
                "median_dimension_std": summary[
                    "median_dimension_std"
                ],
                "zero_variance_dimensions": summary[
                    "zero_variance_dimensions"
                ],
                "median_knn_cosine_similarity": knn_similarity,
                "local_pca_variance_ratio_sum": report[
                    "local_pca_explained_variance_ratio_sum"
                ],
            }
        )

        print(
            f"  shape={features.shape}, "
            f"global_std={summary['feature_global_std']:.6g}, "
            f"zero_var_dims={summary['zero_variance_dimensions']}, "
            f"median_kNN_cos={knn_similarity:.6f}"
        )
        print(f"  saved: {slice_dir}")

    summary_table = pd.DataFrame(summary_rows)

    summary_path = (
        output_root / "all_genes_scgpt512_visualization_QC.csv"
    )
    summary_table.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    global_metadata = {
        "config": str(config_path),
        "cache_tag": args.cache_tag,
        "slices": slices,
        "feature_dimension": 512,
        "global_visualization_pca": global_pca_metadata,
        "global_percentile_limits": [
            float(args.percentile_low),
            float(args.percentile_high),
        ],
        "important_note": (
            "All PCA transformations in this script are visualization-only. "
            "The downstream model still receives the original 512-D features."
        ),
    }

    with (
        output_root / "global_visualization_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            global_metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("Completed")
    print("=" * 100)
    print(summary_table.to_string(index=False))
    print(f"\nQC summary: {summary_path}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()

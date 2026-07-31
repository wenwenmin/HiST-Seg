from __future__ import annotations

"""Downstream Task 3: boundary-aware spatial transcriptomic transition analysis.

This standalone script evaluates whether the tumor boundary recovered by
HiST-Seg is accompanied by an ordered transition in spatial-transcriptomic
(ST) representations.

Default protocol
----------------
* Test sections: ViHBC, DuCIS, XeHBC.
* Model seed: 42.
* Checkpoints:
    engine/checkpoints_pathst_cross_p512_ctx4/seed_42/fold_{slice}/
* Existing ``prediction.npz`` files are reused. Missing predictions are inferred
  directly with ``engine/inference_cross_dataset.py``; no training is performed.
* ST coordinates are read from ``adata.obsm['spatial_resized']`` (fallback:
  ``adata.obsm['spatial']``).
* Direct 512-D scGPT-spatial features are read from each ``{slice}_ctx.npz``.
* Signed boundary distance is expressed in median spot-spacing units:
    tumor core          : d <= -2
    inner boundary      : -2 < d <= 0
    outer boundary      : 0 < d <= 2
    distant non-tumor   : d > 2
  Negative values are inside the tumor and positive values are outside.

Outputs
-------
* Spot-level CSV with probability, GT/predicted signed distance, boundary zones,
  molecular-transition score and PCA coordinates.
* Per-slice metrics and JSON summary.
* Spatial boundary-zone maps, probability/uncertainty transition curves,
  ST molecular-transition curves, PCA plots, zone confusion matrices and a
  one compact manuscript summary for each held-out section.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler


# Add the project root so sibling engine/model/utils packages can be imported
# when this file is executed from downstream_task/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from engine.inference_cross_dataset import (
        cv_imread,
        dataset_for_slice,
        infer_slide,
        load_checkpoint,
        postprocess,
    )
    from model.pathst_net import build_model
    from utils.config import load_config
except ImportError as exc:
    raise ImportError(
        "Cannot import the current HiST-Seg inference stack. Expected:\n"
        f"  {PROJECT_ROOT / 'engine' / 'inference_cross_dataset.py'}\n"
        f"  {PROJECT_ROOT / 'model' / 'pathst_net.py'}\n"
        f"  {PROJECT_ROOT / 'utils' / 'config.py'}\n"
        "Place this script in the project's downstream_task directory."
    ) from exc


SCRIPT_VERSION = "2026-07-17-TASK3-BOUNDARY-ST-FINAL-v3-all-slice-summaries"
MODEL_NAME = "HiST-Seg"
DEFAULT_FOLDS = ["ViHBC", "DuCIS", "XeHBC"]
DEFAULT_CONFIG_CANDIDATES = (
    "../engine/cross_dataset.yaml",
    "cross_dataset.yaml",
    "../engine/cross_dataset(2).yaml",
)
DEFAULT_CHECKPOINT_ROOT = "../engine/checkpoints_pathst_cross_p512_ctx4/seed_42"

ZONE_NAMES = [
    "Tumor core",
    "Inner boundary",
    "Outer boundary",
    "Distant non-tumor",
]
ZONE_SHORT = ["Core", "Inner", "Outer", "Distant"]
ZONE_COLORS = {
    0: "#d73027",
    1: "#fc8d59",
    2: "#91bfdb",
    3: "#4575b4",
}

# Publication-oriented typography.
FONT_BASE = 15
FONT_TICK = 13
FONT_AXIS = 16
FONT_TITLE = 18
FONT_SUPTITLE = 21
FONT_LEGEND = 12


# =============================================================================
# Basic utilities
# =============================================================================


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.size": FONT_BASE,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_AXIS,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "figure.titlesize": FONT_SUPTITLE,
        "axes.linewidth": 1.1,
        "savefig.dpi": 400,
    })


def resolve_config_path(value: Optional[str]) -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates: List[Path] = []
    if value:
        candidates.extend([Path(value), script_dir / value, script_dir.parent / value])
    else:
        for candidate in DEFAULT_CONFIG_CANDIDATES:
            candidates.extend([Path(candidate), script_dir / candidate, script_dir.parent / candidate])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    attempted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot locate cross-dataset YAML. Tried:\n  {attempted}")


def resolve_directory(value: str, description: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [Path(value), script_dir / value, script_dir.parent / value]
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    attempted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot locate {description}. Tried:\n  {attempted}")


def locate_checkpoint(checkpoint_root: Path, fold: str) -> Path:
    fold_dir = checkpoint_root / f"fold_{fold}"
    preferred = [
        fold_dir / f"pathst_{fold}_best.pth",
        fold_dir / f"hist_seg_{fold}_best.pth",
        fold_dir / f"histseg_{fold}_best.pth",
    ]
    for path in preferred:
        if path.is_file():
            return path
    if fold_dir.is_dir():
        matches = sorted(
            list(fold_dir.glob("*best*.pth"))
            + list(fold_dir.glob("*best*.pt"))
            + list(fold_dir.glob("*.pth"))
        )
        if matches:
            return matches[0]
    attempted = "\n  ".join(str(path) for path in preferred)
    raise FileNotFoundError(
        f"Checkpoint for {fold} was not found under {fold_dir}. Tried:\n  {attempted}"
    )


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


# =============================================================================
# Current project inference stack (no downstream_utils dependency)
# =============================================================================


def _load_slide_inputs(cfg, fold: str) -> Dict[str, object]:
    dataset_name, ds_cfg, base = dataset_for_slice(cfg, str(fold))
    he_bgr = cv_imread(os.path.join(base, "he_resized.png"), cv2.IMREAD_COLOR)
    label = cv_imread(os.path.join(base, "label_resized.png"), cv2.IMREAD_GRAYSCALE)
    tissue = cv_imread(os.path.join(base, "mask_resized.png"), cv2.IMREAD_GRAYSCALE)
    if he_bgr is None:
        raise FileNotFoundError(os.path.join(base, "he_resized.png"))
    if label is None:
        raise FileNotFoundError(os.path.join(base, "label_resized.png"))
    he = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2RGB)
    if tissue is None:
        tissue = np.full(he.shape[:2], 255, dtype=np.uint8)
    if he.shape[:2] != label.shape[:2] or he.shape[:2] != tissue.shape[:2]:
        raise ValueError(
            f"[{fold}] image shape mismatch: H&E={he.shape[:2]}, "
            f"label={label.shape[:2]}, tissue={tissue.shape[:2]}"
        )
    return {
        "dataset": str(dataset_name),
        "dataset_cfg": ds_cfg,
        "base": str(base),
        "he": he,
        "label": label,
        "tissue": tissue,
    }


def _read_cached_threshold(metrics_path: Path, cfg, override: Optional[float]) -> Tuple[float, str]:
    if override is not None:
        return float(override), "command line"
    if metrics_path.is_file():
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            value = payload.get("Threshold")
            if value is not None and np.isfinite(float(value)):
                return float(value), str(payload.get("Threshold_Source", "cached metrics"))
        except Exception as exc:
            print(f"[warning] Failed to read cached threshold {metrics_path}: {exc}")
    return float(cfg.inference.prob_threshold), "config fallback"


def load_prediction_or_infer_current(
    cfg,
    fold: str,
    checkpoint_root: Path,
    prediction_root: str | Path,
    *,
    batch_size: Optional[int],
    channels_last: bool,
    threshold: Optional[float],
    force_infer: bool,
    seed: int = 42,
) -> Dict[str, object]:
    fold = str(fold)
    slide = _load_slide_inputs(cfg, fold)
    fold_output = Path(prediction_root) / f"seed_{int(seed)}" / fold
    cache_path = fold_output / "prediction.npz"
    metrics_path = fold_output / "metrics.json"

    if cache_path.is_file() and not force_infer:
        with np.load(cache_path, allow_pickle=False) as archive:
            if "probability" not in archive.files:
                raise KeyError(
                    f"Cached prediction lacks 'probability': {cache_path}; fields={archive.files}"
                )
            probability = np.asarray(archive["probability"], dtype=np.float32)
            cached_prediction = (
                np.asarray(archive["prediction"], dtype=np.uint8)
                if "prediction" in archive.files else None
            )
        if probability.shape != np.asarray(slide["tissue"]).shape:
            raise ValueError(
                f"[{fold}] probability {probability.shape} != tissue "
                f"{np.asarray(slide['tissue']).shape}: {cache_path}"
            )
        used_threshold, threshold_source = _read_cached_threshold(metrics_path, cfg, threshold)
        if threshold is not None or cached_prediction is None:
            prediction = postprocess(
                probability, np.asarray(slide["tissue"]), used_threshold, cfg
            )
        else:
            prediction = (cached_prediction > 0).astype(np.uint8)
        print(f"Loaded cached prediction [{fold}]: {cache_path}")
        return {
            **slide,
            "probability": probability,
            "prediction": prediction,
            "threshold": float(used_threshold),
            "threshold_source": threshold_source,
            "from_cache": True,
            "prediction_cache": str(cache_path),
        }

    checkpoint_path = locate_checkpoint(checkpoint_root, fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_batch_size = int(batch_size or cfg.inference.batch_size)
    print(f"Device [{fold}]: {device}")
    print(f"Checkpoint [{fold}]: {checkpoint_path}")

    model = build_model(cfg, build_uni=True).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    checkpoint = load_checkpoint(
        model,
        str(checkpoint_path),
        fold,
        device,
        expected_seed=int(seed),
        strict_slices=True,
    )
    model.eval()

    used_threshold = (
        float(threshold)
        if threshold is not None
        else float(checkpoint.get("calibrated_threshold", cfg.inference.prob_threshold))
    )
    threshold_source = (
        "command line"
        if threshold is not None
        else str(checkpoint.get("threshold_source", "checkpoint/config"))
    )
    dataset_name, probability, label, tissue = infer_slide(
        model,
        cfg,
        fold,
        fold,
        device,
        effective_batch_size,
        bool(channels_last),
    )
    prediction = postprocess(probability, tissue, used_threshold, cfg)

    fold_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        probability=np.asarray(probability, dtype=np.float32),
        prediction=np.asarray(prediction, dtype=np.uint8),
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "Dataset": str(dataset_name),
                "Test_Slice": fold,
                "Seed": int(seed),
                "Threshold": float(used_threshold),
                "Threshold_Source": threshold_source,
                "Checkpoint": str(checkpoint_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    result = {
        **slide,
        "dataset": str(dataset_name),
        "probability": np.asarray(probability, dtype=np.float32),
        "prediction": np.asarray(prediction, dtype=np.uint8),
        "label": label,
        "tissue": tissue,
        "threshold": float(used_threshold),
        "threshold_source": threshold_source,
        "from_cache": False,
        "prediction_cache": str(cache_path),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Saved prediction cache [{fold}]: {cache_path}")
    return result


# =============================================================================
# ST coordinates and features
# =============================================================================


def read_h5ad_coordinates(path: Path) -> Tuple[np.ndarray, List[str], str]:
    try:
        import scanpy as sc
        adata = sc.read_h5ad(str(path), backed="r")
    except ImportError:
        import anndata as ad
        adata = ad.read_h5ad(str(path), backed="r")

    coordinate_key = None
    for key in ("spatial_resized", "spatial"):
        if key in adata.obsm:
            coordinate_key = key
            break
    if coordinate_key is None:
        raise KeyError(
            f"No spatial_resized/spatial coordinates in {path}; available={list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[coordinate_key], dtype=np.float32)
    spot_ids = [str(value) for value in adata.obs_names]
    if hasattr(adata, "file") and adata.file is not None:
        try:
            adata.file.close()
        except Exception:
            pass
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinates in {path}: {coords.shape}")
    return coords[:, :2], spot_ids, coordinate_key


def resolve_context_cache_path_direct(cfg, fold: str, ds_cfg) -> Path:
    context_root = Path(str(ds_cfg.context_cache_root))
    cache_tag = str(
        getattr(cfg.data, "context_cache_tag", getattr(cfg.preprocess, "cache_tag", ""))
    )
    preferred = context_root / cache_tag / f"{fold}_ctx.npz"
    if preferred.is_file():
        return preferred

    direct = context_root / f"{fold}_ctx.npz"
    if direct.is_file():
        return direct

    matches = sorted(context_root.rglob(f"{fold}_ctx.npz")) if context_root.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print("[warning] Multiple context caches found; selecting the first:")
        for item in matches:
            print(f"  {item}")
        return matches[0]
    raise FileNotFoundError(
        f"Context cache for {fold} not found. Expected:\n  {preferred}"
    )


def load_st_data(cfg, fold: str, slide: Mapping[str, object]) -> Dict[str, object]:
    ds_cfg = slide["dataset_cfg"]
    base = Path(str(slide["base"]))
    h5ad_name = str(getattr(ds_cfg, "h5ad_name", "adata_resized.h5ad"))
    h5ad_path = base / h5ad_name
    if not h5ad_path.is_file():
        candidates = [base / "adata_resized.h5ad", base / "adata_HR_resized.h5ad"]
        h5ad_path = next((path for path in candidates if path.is_file()), h5ad_path)
    if not h5ad_path.is_file():
        raise FileNotFoundError(h5ad_path)

    coords, spot_ids, coordinate_key = read_h5ad_coordinates(h5ad_path)
    context_path = resolve_context_cache_path_direct(cfg, fold, ds_cfg)
    with np.load(context_path, allow_pickle=False) as archive:
        fields = list(archive.files)
        if "features" not in fields:
            raise KeyError(f"'features' missing from {context_path}; fields={fields}")
        features = np.asarray(archive["features"], dtype=np.float32)
        coords_thumb = (
            np.asarray(archive["coords_thumb"], dtype=np.float32)
            if "coords_thumb" in fields else None
        )
        full_h = int(np.asarray(archive["full_h"]).item()) if "full_h" in fields else None
        full_w = int(np.asarray(archive["full_w"]).item()) if "full_w" in fields else None
        thumb_shape = archive["thumb"].shape[:2] if "thumb" in fields else None

    if features.ndim != 2 or features.shape[0] != len(coords):
        raise ValueError(
            f"[{fold}] feature/coordinate mismatch: features={features.shape}, coords={coords.shape}"
        )
    if len(spot_ids) != len(coords):
        raise ValueError(f"[{fold}] spot ID count {len(spot_ids)} != coordinates {len(coords)}")
    if not np.isfinite(features).all():
        raise ValueError(f"[{fold}] ST features contain NaN/Inf: {context_path}")

    # Diagnostic only: compare context-cache coordinates with authoritative h5ad coordinates.
    coordinate_mae = float("nan")
    if coords_thumb is not None and full_h and full_w and thumb_shape:
        thumb_h, thumb_w = thumb_shape
        reconstructed = coords_thumb.copy()
        reconstructed[:, 0] *= float(full_w) / float(thumb_w)
        reconstructed[:, 1] *= float(full_h) / float(thumb_h)
        if reconstructed.shape == coords.shape:
            coordinate_mae = float(np.mean(np.abs(reconstructed - coords)))

    print(f"ST h5ad [{fold}]: {h5ad_path}")
    print(f"ST context cache [{fold}]: {context_path}")
    print(f"Context fields [{fold}]: {fields}")
    print(
        f"ST data [{fold}]: spots={len(coords)}, features={features.shape}, "
        f"coordinate_key={coordinate_key}, cache-coordinate MAE={coordinate_mae:.4f}px"
    )
    return {
        "h5ad_path": str(h5ad_path),
        "context_path": str(context_path),
        "context_fields": fields,
        "coords": coords,
        "spot_ids": spot_ids,
        "features": features,
        "coordinate_key": coordinate_key,
        "cache_coordinate_mae_px": coordinate_mae,
    }


# =============================================================================
# Spot aggregation and boundary distances
# =============================================================================


def infer_spot_radius(
    coords_xy: np.ndarray,
    factor: float,
    min_px: float,
    max_px: float,
) -> Tuple[float, float]:
    if len(coords_xy) < 2:
        return float(min_px), float("nan")
    distances, _ = cKDTree(coords_xy.astype(np.float64)).query(coords_xy, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    median_nn = float(np.median(nearest)) if nearest.size else float(min_px * 2.0)
    radius = float(np.clip(median_nn * float(factor), float(min_px), float(max_px)))
    return radius, median_nn


class DiskSpotAggregator:
    """Vectorized circular aggregation over many spots."""

    def __init__(
        self,
        coords_xy: np.ndarray,
        tissue_mask: np.ndarray,
        radius_px: float,
        max_chunk_elements: int = 2_000_000,
    ):
        self.coords = np.rint(np.asarray(coords_xy, dtype=np.float32)).astype(np.int32)
        self.tissue = np.asarray(tissue_mask).astype(bool)
        self.height, self.width = self.tissue.shape[:2]
        radius_int = max(1, int(math.ceil(float(radius_px))))
        yy, xx = np.mgrid[-radius_int:radius_int + 1, -radius_int:radius_int + 1]
        keep = (xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2) <= float(radius_px) ** 2
        self.dy = yy[keep].astype(np.int32)
        self.dx = xx[keep].astype(np.int32)
        self.n_offsets = int(len(self.dx))
        self.chunk_size = max(1, int(max_chunk_elements) // max(1, self.n_offsets))

    def aggregate(self, score_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        score_map = np.asarray(score_map, dtype=np.float32)
        if score_map.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Score map {score_map.shape[:2]} != tissue {(self.height, self.width)}"
            )
        n = len(self.coords)
        means = np.full(n, np.nan, dtype=np.float32)
        tissue_fraction = np.zeros(n, dtype=np.float32)
        for start in range(0, n, self.chunk_size):
            stop = min(n, start + self.chunk_size)
            x = self.coords[start:stop, 0, None]
            y = self.coords[start:stop, 1, None]
            xx = x + self.dx[None, :]
            yy = y + self.dy[None, :]
            in_bounds = (xx >= 0) & (xx < self.width) & (yy >= 0) & (yy < self.height)
            xx_safe = np.clip(xx, 0, self.width - 1)
            yy_safe = np.clip(yy, 0, self.height - 1)
            tissue_valid = in_bounds & self.tissue[yy_safe, xx_safe]
            denominator = tissue_valid.sum(axis=1)
            disk_pixels = np.maximum(1, in_bounds.sum(axis=1))
            tissue_fraction[start:stop] = (
                denominator.astype(np.float32) / disk_pixels.astype(np.float32)
            )
            values = score_map[yy_safe, xx_safe]
            numerator = (values * tissue_valid).sum(axis=1, dtype=np.float64)
            valid = denominator > 0
            chunk_mean = np.full(stop - start, np.nan, dtype=np.float32)
            chunk_mean[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
            means[start:stop] = chunk_mean
        return means, tissue_fraction


def signed_distance_at_spots(
    tumor_mask: np.ndarray,
    tissue_mask: np.ndarray,
    coords_xy: np.ndarray,
    median_spot_spacing: float,
) -> np.ndarray:
    """Signed Euclidean distance at spot centers, normalized by spot spacing.

    Negative values are inside the tumor and positive values are outside.
    OpenCV's float32 distance transform avoids the very large float64 arrays
    produced by scipy.ndimage on whole-slide images.
    """
    tumor = np.asarray(tumor_mask).astype(bool) & np.asarray(tissue_mask).astype(bool)
    if not np.any(tumor):
        raise RuntimeError("Tumor mask is empty; signed distance is undefined.")
    if np.all(tumor[np.asarray(tissue_mask).astype(bool)]):
        raise RuntimeError("All tissue pixels are tumor; outside distance is undefined.")

    outside_input = (~tumor).astype(np.uint8)
    signed = cv2.distanceTransform(outside_input, cv2.DIST_L2, 5).astype(np.float32)
    inside = cv2.distanceTransform(tumor.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    signed[tumor] = -inside[tumor]
    del inside, outside_input

    coordinates = np.rint(coords_xy).astype(np.int64)
    x = np.clip(coordinates[:, 0], 0, signed.shape[1] - 1)
    y = np.clip(coordinates[:, 1], 0, signed.shape[0] - 1)
    values = signed[y, x].astype(np.float32)
    del signed
    spacing = max(float(median_spot_spacing), 1e-6)
    return values / spacing


def assign_zones(distance: np.ndarray, boundary_width: float) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    zones = np.full(len(distance), -1, dtype=np.int8)
    finite = np.isfinite(distance)
    zones[finite & (distance <= -float(boundary_width))] = 0
    zones[finite & (distance > -float(boundary_width)) & (distance <= 0.0)] = 1
    zones[finite & (distance > 0.0) & (distance <= float(boundary_width))] = 2
    zones[finite & (distance > float(boundary_width))] = 3
    return zones


# =============================================================================
# Molecular transition analysis
# =============================================================================


def standardize_features(features: np.ndarray) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    transformed = scaler.fit_transform(np.asarray(features, dtype=np.float32))
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
    return transformed.astype(np.float32)


def molecular_transition_axis(
    standardized_features: np.ndarray,
    gt_zones: np.ndarray,
    gt_distance: np.ndarray,
    *,
    min_anchor_spots: int = 20,
    anchor_fraction: float = 0.05,
    max_anchor_spots: int = 100,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build a robust molecular transition axis from GT distance extremes.

    A thin or fragmented tumor can contain very few spots in the strict
    ``Tumor core`` zone even when many tumor-side spots are available. To avoid
    unstable 512-D centroids and hard failures, anchors are chosen from the
    deepest GT-tumor-side spots and the farthest GT-non-tumor-side spots.
    """
    features = np.asarray(standardized_features, dtype=np.float32)
    zones = np.asarray(gt_zones)
    distance = np.asarray(gt_distance, dtype=np.float64)

    if features.ndim != 2 or features.shape[0] != distance.shape[0]:
        raise ValueError(
            f"Feature/distance mismatch: features={features.shape}, "
            f"distance={distance.shape}"
        )
    if min_anchor_spots < 5:
        raise ValueError("min_anchor_spots must be at least 5")
    if not 0.0 < float(anchor_fraction) <= 0.5:
        raise ValueError("anchor_fraction must be in (0, 0.5]")
    if max_anchor_spots < min_anchor_spots:
        raise ValueError("max_anchor_spots must be >= min_anchor_spots")

    finite = np.isfinite(distance)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size < 10:
        raise RuntimeError(
            f"Too few finite GT distances for molecular axis: {finite_indices.size}"
        )

    requested = int(np.ceil(float(anchor_fraction) * finite_indices.size))
    requested = max(int(min_anchor_spots), requested)
    requested = min(int(max_anchor_spots), requested)

    inside_indices = np.flatnonzero(finite & (distance <= 0.0))
    outside_indices = np.flatnonzero(finite & (distance > 0.0))

    if min(inside_indices.size, outside_indices.size) >= 5:
        anchor_count = min(requested, inside_indices.size, outside_indices.size)
        core_order = inside_indices[np.argsort(distance[inside_indices])]
        distant_order = outside_indices[np.argsort(distance[outside_indices])[::-1]]
        core_indices = core_order[:anchor_count]
        distant_indices = distant_order[:anchor_count]
        anchor_mode = "distance_extremes_within_GT_sides"
    else:
        anchor_count = min(requested, finite_indices.size // 2)
        if anchor_count < 5:
            raise RuntimeError(
                "Insufficient finite spots to construct molecular anchors: "
                f"finite={finite_indices.size}"
            )
        ordered = finite_indices[np.argsort(distance[finite_indices])]
        core_indices = ordered[:anchor_count]
        distant_indices = ordered[-anchor_count:]
        anchor_mode = "global_distance_extremes_fallback"

    core_centroid = features[core_indices].mean(axis=0)
    distant_centroid = features[distant_indices].mean(axis=0)
    direction = distant_centroid - core_centroid
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise RuntimeError("Molecular anchor centroids are indistinguishable.")
    direction /= norm
    midpoint = 0.5 * (core_centroid + distant_centroid)
    raw_score = (features - midpoint) @ direction

    median = float(np.median(raw_score))
    q25, q75 = np.percentile(raw_score, [25, 75])
    scale = max(float(q75 - q25), 1e-6)
    score = ((raw_score - median) / scale).astype(np.float32)

    zone_means = {
        ZONE_NAMES[zone]: (
            float(np.mean(score[zones == zone]))
            if np.any(zones == zone) else float("nan")
        )
        for zone in range(4)
    }
    ordered_values = np.asarray(
        [zone_means[name] for name in ZONE_NAMES], dtype=np.float64
    )
    adjacent = np.diff(ordered_values)
    monotonic_fraction = (
        float(np.mean(adjacent > 0))
        if np.all(np.isfinite(adjacent)) else float("nan")
    )
    centroid_distance = float(np.linalg.norm(distant_centroid - core_centroid))

    return score, {
        "Molecular_Anchor_Mode": anchor_mode,
        "Molecular_Anchor_Spots_Per_Side": int(anchor_count),
        "Molecular_Anchor_Min_Requested": int(min_anchor_spots),
        "Molecular_Anchor_Fraction": float(anchor_fraction),
        "Molecular_Anchor_Max": int(max_anchor_spots),
        "Strict_Core_Zone_Spots": int(np.sum(zones == 0)),
        "Strict_Distant_Zone_Spots": int(np.sum(zones == 3)),
        "Core_Spots": int(core_indices.size),
        "Distant_Spots": int(distant_indices.size),
        "Core_Anchor_Distance_Min": float(np.min(distance[core_indices])),
        "Core_Anchor_Distance_Max": float(np.max(distance[core_indices])),
        "Distant_Anchor_Distance_Min": float(np.min(distance[distant_indices])),
        "Distant_Anchor_Distance_Max": float(np.max(distance[distant_indices])),
        "Core_Distant_Centroid_Distance": centroid_distance,
        "Zone_Molecular_Means": zone_means,
        "Adjacent_Monotonic_Fraction": monotonic_fraction,
    }

def pca_embedding(standardized_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, svd_solver="auto")
    coordinates = pca.fit_transform(standardized_features).astype(np.float32)
    return coordinates, pca.explained_variance_ratio_.astype(np.float64)


# =============================================================================
# Per-slice analysis
# =============================================================================


def analyze_slice(
    cfg,
    fold: str,
    checkpoint_root: Path,
    prediction_root: Path,
    *,
    seed: int,
    batch_size: Optional[int],
    channels_last: bool,
    threshold: Optional[float],
    force_infer: bool,
    spot_radius_factor: float,
    spot_radius_min_px: float,
    spot_radius_max_px: float,
    min_spot_tissue_fraction: float,
    boundary_width: float,
    max_distance: float,
    min_molecular_anchor_spots: int,
    molecular_anchor_fraction: float,
    max_molecular_anchor_spots: int,
) -> Dict[str, object]:
    prediction_data = load_prediction_or_infer_current(
        cfg,
        fold,
        checkpoint_root,
        prediction_root,
        batch_size=batch_size,
        channels_last=channels_last,
        threshold=threshold,
        force_infer=force_infer,
        seed=seed,
    )
    st_data = load_st_data(cfg, fold, prediction_data)

    coords = np.asarray(st_data["coords"], dtype=np.float32)
    features = np.asarray(st_data["features"], dtype=np.float32)
    tissue = np.asarray(prediction_data["tissue"]) > 127
    label = (np.asarray(prediction_data["label"]) > 127) & tissue
    prediction = (np.asarray(prediction_data["prediction"]) > 0) & tissue
    probability = np.asarray(prediction_data["probability"], dtype=np.float32)
    probability[~tissue] = 0.0

    radius_px, median_spacing = infer_spot_radius(
        coords,
        factor=spot_radius_factor,
        min_px=spot_radius_min_px,
        max_px=spot_radius_max_px,
    )
    aggregator = DiskSpotAggregator(coords, tissue, radius_px)
    spot_probability, tissue_fraction = aggregator.aggregate(probability)
    gt_fraction, _ = aggregator.aggregate(label.astype(np.float32))

    gt_distance = signed_distance_at_spots(label, tissue, coords, median_spacing)
    pred_distance = signed_distance_at_spots(prediction, tissue, coords, median_spacing)
    gt_zones = assign_zones(gt_distance, boundary_width)
    pred_zones = assign_zones(pred_distance, boundary_width)

    valid = (
        np.isfinite(spot_probability)
        & np.isfinite(gt_fraction)
        & np.isfinite(gt_distance)
        & np.isfinite(pred_distance)
        & (tissue_fraction >= float(min_spot_tissue_fraction))
        & (gt_zones >= 0)
        & (pred_zones >= 0)
    )
    if int(valid.sum()) < 20:
        raise RuntimeError(f"[{fold}] only {int(valid.sum())} valid spots remain")

    coords_v = coords[valid]
    features_v = features[valid]
    spot_ids_v = [st_data["spot_ids"][index] for index in np.flatnonzero(valid)]
    spot_indices_v = np.flatnonzero(valid).astype(np.int32)
    probability_v = spot_probability[valid]
    gt_fraction_v = gt_fraction[valid]
    tissue_fraction_v = tissue_fraction[valid]
    gt_distance_v = gt_distance[valid]
    pred_distance_v = pred_distance[valid]
    gt_zones_v = gt_zones[valid]
    pred_zones_v = pred_zones[valid]
    entropy_v = binary_entropy(probability_v).astype(np.float32)

    standardized = standardize_features(features_v)
    molecular_score, molecular_meta = molecular_transition_axis(
        standardized,
        gt_zones_v,
        gt_distance_v,
        min_anchor_spots=min_molecular_anchor_spots,
        anchor_fraction=molecular_anchor_fraction,
        max_anchor_spots=max_molecular_anchor_spots,
    )
    print(
        f"Molecular anchors [{fold}]: "
        f"mode={molecular_meta['Molecular_Anchor_Mode']}, "
        f"per_side={molecular_meta['Molecular_Anchor_Spots_Per_Side']}, "
        f"strict_core={molecular_meta['Strict_Core_Zone_Spots']}, "
        f"strict_distant={molecular_meta['Strict_Distant_Zone_Spots']}"
    )
    pca_xy, pca_ratio = pca_embedding(standardized)

    clipped_gt = np.clip(gt_distance_v, -max_distance, max_distance)
    clipped_pred = np.clip(pred_distance_v, -max_distance, max_distance)
    boundary_gt = np.abs(gt_distance_v) <= boundary_width
    boundary_pred = np.abs(pred_distance_v) <= boundary_width

    zone_counts = {ZONE_NAMES[zone]: int(np.sum(gt_zones_v == zone)) for zone in range(4)}
    zone_accuracy = float(accuracy_score(gt_zones_v, pred_zones_v))
    zone_macro_f1 = float(
        f1_score(gt_zones_v, pred_zones_v, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    )
    boundary_f1 = float(f1_score(boundary_gt, boundary_pred, zero_division=0))
    distance_error = clipped_pred - clipped_gt

    boundary_entropy = float(np.mean(entropy_v[boundary_gt])) if np.any(boundary_gt) else float("nan")
    nonboundary_entropy = float(np.mean(entropy_v[~boundary_gt])) if np.any(~boundary_gt) else float("nan")

    metrics: Dict[str, object] = {
        "Dataset": str(prediction_data["dataset"]),
        "Slice": fold,
        "Seed": int(seed),
        "Valid_Spots": int(len(coords_v)),
        "Spot_Radius_Pixel": float(radius_px),
        "Median_Spot_Spacing_Pixel": float(median_spacing),
        "Boundary_Width_SpotUnits": float(boundary_width),
        "Distance_Clip_SpotUnits": float(max_distance),
        "Threshold": float(prediction_data["threshold"]),
        "Threshold_Source": str(prediction_data["threshold_source"]),
        "Prediction_From_Cache": bool(prediction_data["from_cache"]),
        "SignedDistance_MAE": float(np.mean(np.abs(distance_error))),
        "SignedDistance_RMSE": float(np.sqrt(np.mean(distance_error ** 2))),
        "SignedDistance_Pearson": safe_pearson(clipped_gt, clipped_pred),
        "SignedDistance_Spearman": safe_spearman(clipped_gt, clipped_pred),
        "Zone_Accuracy": zone_accuracy,
        "Zone_MacroF1": zone_macro_f1,
        "Boundary_F1": boundary_f1,
        "Probability_GTDistance_Spearman": safe_spearman(gt_distance_v, probability_v),
        "Probability_Transition_Monotonicity": -safe_spearman(gt_distance_v, probability_v),
        "Molecular_GTDistance_Spearman": safe_spearman(gt_distance_v, molecular_score),
        "Molecular_PredDistance_Spearman": safe_spearman(pred_distance_v, molecular_score),
        "Boundary_Entropy": boundary_entropy,
        "NonBoundary_Entropy": nonboundary_entropy,
        "Boundary_Entropy_Increase": boundary_entropy - nonboundary_entropy,
        "PCA1_Explained": float(pca_ratio[0]),
        "PCA2_Explained": float(pca_ratio[1]),
        "GT_Zone_Counts": zone_counts,
        **molecular_meta,
        "Prediction_Cache": str(prediction_data["prediction_cache"]),
        "H5AD_Path": str(st_data["h5ad_path"]),
        "Context_Cache_Path": str(st_data["context_path"]),
        "Coordinate_Key": str(st_data["coordinate_key"]),
        "Cache_Coordinate_MAE_Pixel": float(st_data["cache_coordinate_mae_px"]),
    }

    spot_rows: List[Dict[str, object]] = []
    for row_index in range(len(coords_v)):
        gt_zone = int(gt_zones_v[row_index])
        pred_zone = int(pred_zones_v[row_index])
        spot_rows.append({
            "Dataset": str(prediction_data["dataset"]),
            "Slice": fold,
            "Seed": int(seed),
            "Spot_Index": int(spot_indices_v[row_index]),
            "Spot_ID": spot_ids_v[row_index],
            "x_pixel": float(coords_v[row_index, 0]),
            "y_pixel": float(coords_v[row_index, 1]),
            "Tissue_Fraction": float(tissue_fraction_v[row_index]),
            "GT_Tumor_Fraction": float(gt_fraction_v[row_index]),
            "HiST_Seg_TumorScore": float(probability_v[row_index]),
            "Prediction_Entropy": float(entropy_v[row_index]),
            "GT_Signed_Distance_SpotUnits": float(gt_distance_v[row_index]),
            "Pred_Signed_Distance_SpotUnits": float(pred_distance_v[row_index]),
            "GT_Zone_ID": gt_zone,
            "GT_Zone": ZONE_NAMES[gt_zone],
            "Pred_Zone_ID": pred_zone,
            "Pred_Zone": ZONE_NAMES[pred_zone],
            "Molecular_Transition_Score": float(molecular_score[row_index]),
            "ST_PCA1": float(pca_xy[row_index, 0]),
            "ST_PCA2": float(pca_xy[row_index, 1]),
        })

    return {
        "fold": fold,
        "dataset": str(prediction_data["dataset"]),
        "he": np.asarray(prediction_data["he"]),
        "label": label,
        "prediction": prediction,
        "tissue": tissue,
        "probability_map": probability,
        "coords": coords_v,
        "spot_probability": probability_v,
        "entropy": entropy_v,
        "gt_distance": gt_distance_v,
        "pred_distance": pred_distance_v,
        "gt_zones": gt_zones_v,
        "pred_zones": pred_zones_v,
        "molecular_score": molecular_score,
        "pca_xy": pca_xy,
        "metrics": metrics,
        "spot_rows": spot_rows,
    }


# =============================================================================
# Plotting helpers
# =============================================================================


def resize_for_display(image: np.ndarray, max_size: int = 1500) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return image.copy(), 1.0
    scale = float(max_size) / float(longest)
    output = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if image.ndim == 3 else cv2.INTER_NEAREST,
    )
    return output, scale


def zone_legend_handles() -> List[Line2D]:
    return [
        Line2D(
            [0], [0], marker="o", linestyle="None", markersize=8,
            markerfacecolor=ZONE_COLORS[zone], markeredgecolor="white",
            label=ZONE_NAMES[zone],
        )
        for zone in range(4)
    ]


def scatter_zones(
    ax: plt.Axes,
    he_show: np.ndarray,
    coords: np.ndarray,
    zones: np.ndarray,
    scale: float,
    title: str,
    point_size: float = 11.0,
) -> None:
    ax.imshow(he_show)
    for zone in range(4):
        mask = zones == zone
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0] * scale,
            coords[mask, 1] * scale,
            s=point_size,
            color=ZONE_COLORS[zone],
            alpha=0.82,
            edgecolors="none",
            linewidths=0,
        )
    ax.set_title(title, fontweight="bold")
    ax.axis("off")


def contour_mask(ax: plt.Axes, mask: np.ndarray, scale: float, color: str, title: str) -> None:
    mask_show, _ = resize_for_display(mask.astype(np.uint8), max_size=max(mask.shape))
    # Resize to exactly match the displayed H&E if a scale was applied.
    if scale != 1.0:
        target_w = max(1, int(round(mask.shape[1] * scale)))
        target_h = max(1, int(round(mask.shape[0] * scale)))
        mask_show = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    if np.any(mask_show) and np.any(~mask_show.astype(bool)):
        ax.contour(mask_show.astype(np.float32), levels=[0.5], colors=[color], linewidths=2.8)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")


def save_spatial_boundary_figure(
    path: Path,
    result: Mapping[str, object],
    dpi: int,
) -> None:
    he_show, scale = resize_for_display(np.asarray(result["he"]), max_size=1450)
    coords = np.asarray(result["coords"])
    gt_zones = np.asarray(result["gt_zones"])
    pred_zones = np.asarray(result["pred_zones"])
    probability = np.asarray(result["spot_probability"])

    fig, axes = plt.subplots(2, 3, figsize=(17.2, 10.8), dpi=dpi)
    axes[0, 0].imshow(he_show)
    axes[0, 0].set_title(f"{result['fold']} H&E", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(he_show)
    contour_mask(axes[0, 1], np.asarray(result["label"]), scale, "#00b050", "GT tumor boundary")

    axes[0, 2].imshow(he_show)
    contour_mask(axes[0, 2], np.asarray(result["prediction"]), scale, "#ffd700", f"{MODEL_NAME} boundary")

    scatter_zones(axes[1, 0], he_show, coords, gt_zones, scale, "GT boundary zones")
    scatter_zones(axes[1, 1], he_show, coords, pred_zones, scale, f"{MODEL_NAME} boundary zones")

    axes[1, 2].imshow(he_show)
    axes[1, 2].scatter(
        coords[:, 0] * scale,
        coords[:, 1] * scale,
        c=probability,
        s=11,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        alpha=0.90,
        edgecolors="white",
        linewidths=0.16,
    )
    axes[1, 2].set_title(f"{MODEL_NAME} spot probability (0--1)", fontweight="bold")
    axes[1, 2].axis("off")

    fig.legend(
        handles=zone_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"{result['fold']}: boundary-aware spatial transition analysis",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def binned_summary(
    distance: np.ndarray,
    values: np.ndarray,
    max_distance: float,
    bin_width: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance = np.asarray(distance, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    edges = np.arange(-max_distance, max_distance + bin_width * 1.01, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(len(centers), np.nan, dtype=np.float64)
    sems = np.full(len(centers), np.nan, dtype=np.float64)
    counts = np.zeros(len(centers), dtype=np.int32)
    for index in range(len(centers)):
        if index == len(centers) - 1:
            mask = (distance >= edges[index]) & (distance <= edges[index + 1])
        else:
            mask = (distance >= edges[index]) & (distance < edges[index + 1])
        finite = mask & np.isfinite(values)
        counts[index] = int(finite.sum())
        if counts[index] > 0:
            means[index] = float(np.mean(values[finite]))
            sems[index] = (
                float(np.std(values[finite], ddof=1) / math.sqrt(counts[index]))
                if counts[index] > 1 else 0.0
            )
    return centers, means, sems, counts


def dataset_colors(results: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    cmap = plt.get_cmap("tab10")
    return {str(result["fold"]): cmap(index) for index, result in enumerate(results)}


def plot_transition_curve(
    ax: plt.Axes,
    results: Sequence[Mapping[str, object]],
    value_key: str,
    max_distance: float,
    bin_width: float,
    ylabel: str,
    title: str,
    y_limits: Optional[Tuple[float, float]] = None,
) -> None:
    colors = dataset_colors(results)
    for result in results:
        centers, means, sems, counts = binned_summary(
            np.asarray(result["gt_distance"]),
            np.asarray(result[value_key]),
            max_distance,
            bin_width,
        )
        valid = np.isfinite(means) & (counts >= 3)
        if not np.any(valid):
            continue
        fold = str(result["fold"])
        ax.plot(centers[valid], means[valid], linewidth=2.5, label=fold, color=colors[fold])
        ax.fill_between(
            centers[valid],
            means[valid] - sems[valid],
            means[valid] + sems[valid],
            alpha=0.18,
            color=colors[fold],
        )
    ax.axvline(0.0, linestyle="--", linewidth=1.4, color="black")
    ax.axvspan(-2.0, 2.0, alpha=0.08, color="gray")
    ax.set_xlim(-max_distance, max_distance)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_xlabel("GT signed distance (spot-spacing units)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)


def save_probability_uncertainty_transition(
    path: Path,
    results: Sequence[Mapping[str, object]],
    max_distance: float,
    bin_width: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.4), dpi=dpi)
    plot_transition_curve(
        axes[0], results, "spot_probability", max_distance, bin_width,
        f"{MODEL_NAME} cancer probability", "Probability across the tumor boundary", (0.0, 1.02),
    )
    plot_transition_curve(
        axes[1], results, "entropy", max_distance, bin_width,
        "Prediction entropy", "Uncertainty across the tumor boundary", (0.0, 0.72),
    )
    fig.suptitle("Boundary-dependent prediction transition", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_molecular_transition_curves(
    path: Path,
    results: Sequence[Mapping[str, object]],
    max_distance: float,
    bin_width: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.5), dpi=dpi)
    plot_transition_curve(
        axes[0], results, "molecular_score", max_distance, bin_width,
        "ST molecular-transition score", "Molecular transition across the GT boundary", None,
    )

    x = np.arange(4, dtype=np.float64)
    width = 0.24
    colors = dataset_colors(results)
    for index, result in enumerate(results):
        means = [
            float(np.mean(np.asarray(result["molecular_score"])[np.asarray(result["gt_zones"]) == zone]))
            if np.any(np.asarray(result["gt_zones"]) == zone) else np.nan
            for zone in range(4)
        ]
        axes[1].bar(
            x + (index - (len(results) - 1) / 2.0) * width,
            means,
            width=width,
            label=str(result["fold"]),
            color=colors[str(result["fold"])],
            alpha=0.85,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ZONE_SHORT, rotation=18)
    axes[1].set_ylabel("Mean molecular-transition score")
    axes[1].set_title("ST feature states across boundary zones", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False)

    fig.suptitle("Boundary-aware spatial-transcriptomic transition", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_pca_zones(
    path: Path,
    results: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 5.4), dpi=dpi)
    axes = np.atleast_1d(axes)
    for ax, result in zip(axes, results):
        pca_xy = np.asarray(result["pca_xy"])
        zones = np.asarray(result["gt_zones"])
        for zone in range(4):
            mask = zones == zone
            if not np.any(mask):
                continue
            ax.scatter(
                pca_xy[mask, 0], pca_xy[mask, 1],
                s=11, alpha=0.55, color=ZONE_COLORS[zone],
                edgecolors="none", label=ZONE_NAMES[zone],
            )
        metrics = result["metrics"]
        ax.set_xlabel(f"PC1 ({100 * float(metrics['PCA1_Explained']):.1f}%)")
        ax.set_ylabel(f"PC2 ({100 * float(metrics['PCA2_Explained']):.1f}%)")
        ax.set_title(str(result["fold"]), fontweight="bold")
        ax.grid(alpha=0.18)
    fig.legend(
        handles=zone_legend_handles(),
        loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False,
    )
    fig.suptitle("ST feature space stratified by GT boundary zones", fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_zone_confusions(
    path: Path,
    results: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.9), dpi=dpi)
    axes = np.atleast_1d(axes)
    image = None
    for ax, result in zip(axes, results):
        cm = confusion_matrix(
            np.asarray(result["gt_zones"]),
            np.asarray(result["pred_zones"]),
            labels=[0, 1, 2, 3],
            normalize="true",
        )
        image = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
        for row in range(4):
            for col in range(4):
                value = cm[row, col]
                ax.text(
                    col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=11, color="white" if value > 0.55 else "black",
                )
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(ZONE_SHORT, rotation=32, ha="right")
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(ZONE_SHORT)
        ax.set_xlabel("Predicted zone")
        ax.set_ylabel("GT zone")
        ax.set_title(str(result["fold"]), fontweight="bold")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.tolist(), fraction=0.022, pad=0.02)
        cbar.set_label("Row-normalized proportion")
    fig.suptitle("Boundary-zone classification consistency", fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.18, top=0.84, wspace=0.32)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_metric_bars(
    path: Path,
    results: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    metrics = [
        ("Zone_MacroF1", "Zone macro-F1"),
        ("Boundary_F1", "Boundary F1"),
        ("Probability_Transition_Monotonicity", "Probability transition"),
        ("Molecular_PredDistance_Spearman", "Molecular alignment"),
    ]
    folds = [str(result["fold"]) for result in results]
    x = np.arange(len(folds), dtype=np.float64)
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.5, 6.3), dpi=dpi)
    for index, (key, label) in enumerate(metrics):
        values = [float(result["metrics"][key]) for result in results]
        ax.bar(x + (index - 1.5) * width, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Boundary-transition performance", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_manuscript_summary(
    path: Path,
    result: Mapping[str, object],
    max_distance: float,
    bin_width: float,
    dpi: int,
) -> None:
    """Save one fully slice-specific manuscript summary figure.

    Every panel in the output corresponds to the same held-out section:
    spatial boundary maps, probability transition, molecular transition and
    quantitative boundary-transition metrics. This avoids generating three
    nearly identical figures whose spatial panels differ but whose statistics
    are pooled across all sections.
    """
    fold = str(result["fold"])
    single_result = [result]
    he_show, scale = resize_for_display(np.asarray(result["he"]), max_size=1200)
    coords = np.asarray(result["coords"])

    fig = plt.figure(figsize=(18.0, 10.5), dpi=dpi)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.92],
        hspace=0.30,
        wspace=0.24,
    )
    ax_he = fig.add_subplot(grid[0, 0])
    ax_gt = fig.add_subplot(grid[0, 1])
    ax_pred = fig.add_subplot(grid[0, 2])
    ax_prob = fig.add_subplot(grid[1, 0])
    ax_mol = fig.add_subplot(grid[1, 1])
    ax_metric = fig.add_subplot(grid[1, 2])

    ax_he.imshow(he_show)
    ax_he.set_title(f"{fold} H&E", fontweight="bold")
    ax_he.axis("off")
    scatter_zones(
        ax_gt,
        he_show,
        coords,
        np.asarray(result["gt_zones"]),
        scale,
        "GT boundary zones",
        point_size=7,
    )
    scatter_zones(
        ax_pred,
        he_show,
        coords,
        np.asarray(result["pred_zones"]),
        scale,
        f"{MODEL_NAME} boundary zones",
        point_size=7,
    )

    plot_transition_curve(
        ax_prob,
        single_result,
        "spot_probability",
        max_distance,
        bin_width,
        "Cancer probability",
        "Probability transition",
        (0.0, 1.02),
    )
    plot_transition_curve(
        ax_mol,
        single_result,
        "molecular_score",
        max_distance,
        bin_width,
        "ST transition score",
        "Molecular transition",
        None,
    )

    metrics = [
        ("Zone_MacroF1", "Zone F1"),
        ("Boundary_F1", "Boundary F1"),
        ("Probability_Transition_Monotonicity", "Prob.\ntransition"),
        ("Molecular_PredDistance_Spearman", "ST\nalignment"),
    ]
    metric_values = [float(result["metrics"][key]) for key, _ in metrics]
    metric_labels = [label for _, label in metrics]
    positions = np.arange(len(metrics), dtype=np.float64)
    bars = ax_metric.bar(positions, metric_values, width=0.64)
    ax_metric.set_xticks(positions)
    ax_metric.set_xticklabels(metric_labels, fontweight="bold")
    finite_values = [value for value in metric_values if np.isfinite(value)]
    lower = min([-0.05] + finite_values)
    upper = max([1.0] + finite_values)
    margin = max(0.08, 0.08 * max(1.0, upper - lower))
    ax_metric.set_ylim(min(-0.10, lower - margin), max(1.05, upper + margin))
    ax_metric.axhline(0.0, linewidth=0.8, alpha=0.35)
    ax_metric.set_ylabel("Score")
    ax_metric.set_title(f"{fold} boundary-transition metrics", fontweight="bold")
    ax_metric.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, metric_values):
        if not np.isfinite(value):
            continue
        vertical_offset = 3 if value >= 0 else -14
        va = "bottom" if value >= 0 else "top"
        ax_metric.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )

    panel_axes = [ax_he, ax_gt, ax_pred, ax_prob, ax_mol, ax_metric]
    panel_labels = ["(a)", "", "", "(b)", "(c)", "(d)"]
    for label, ax in zip(panel_labels, panel_axes):
        if label:
            ax.text(
                -0.10,
                1.06,
                label,
                transform=ax.transAxes,
                fontsize=20,
                fontweight="bold",
                va="top",
            )

    fig.legend(
        handles=zone_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.49),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"{fold}: boundary-aware spatial-transcriptomic transition analysis",
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.07, top=0.92)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI and main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream Task 3: boundary-aware ST transition analysis."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--folds", nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument(
        "--prediction_root",
        default=None,
        help="Existing prediction root; default uses cfg.paths.output_dir.",
    )
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--channels_last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--force_infer",
        action="store_true",
        help="Ignore existing prediction.npz files and rerun inference.",
    )
    parser.add_argument("--spot_radius_factor", type=float, default=0.5)
    parser.add_argument("--spot_radius_min_px", type=float, default=8.0)
    parser.add_argument("--spot_radius_max_px", type=float, default=80.0)
    parser.add_argument("--min_spot_tissue_fraction", type=float, default=0.25)
    parser.add_argument(
        "--boundary_width",
        type=float,
        default=2.0,
        help="Boundary-zone width in median spot-spacing units.",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=6.0,
        help="Signed-distance range used by curves and clipped-distance metrics.",
    )
    parser.add_argument(
        "--min_molecular_anchor_spots",
        type=int,
        default=20,
        help="Minimum molecular-anchor spots per side.",
    )
    parser.add_argument(
        "--molecular_anchor_fraction",
        type=float,
        default=0.05,
        help="Target fraction of valid spots per molecular-anchor side.",
    )
    parser.add_argument(
        "--max_molecular_anchor_spots",
        type=int,
        default=100,
        help="Maximum molecular-anchor spots per side.",
    )
    parser.add_argument("--distance_bin_width", type=float, default=0.5)
    parser.add_argument(
        "--representative_fold",
        default=None,
        help=(
            "Deprecated compatibility option. A manuscript summary is now "
            "generated independently for every requested fold."
        ),
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 100)
    print(f"Task 3 standalone version: {SCRIPT_VERSION}")
    print("Boundary-aware spatial transcriptomic transition analysis")
    print("=" * 100)

    if args.boundary_width <= 0:
        raise ValueError("--boundary_width must be positive")
    if args.max_distance <= args.boundary_width:
        raise ValueError("--max_distance must be larger than --boundary_width")
    if args.distance_bin_width <= 0:
        raise ValueError("--distance_bin_width must be positive")
    if args.min_molecular_anchor_spots < 5:
        raise ValueError("--min_molecular_anchor_spots must be at least 5")
    if not 0.0 < args.molecular_anchor_fraction <= 0.5:
        raise ValueError("--molecular_anchor_fraction must be in (0, 0.5]")
    if args.max_molecular_anchor_spots < args.min_molecular_anchor_spots:
        raise ValueError(
            "--max_molecular_anchor_spots must be >= --min_molecular_anchor_spots"
        )

    apply_publication_style()
    config_path = resolve_config_path(args.config)
    cfg = load_config(str(config_path))
    checkpoint_root = resolve_directory(args.checkpoint_root, "seed-42 checkpoint root")
    prediction_root = Path(args.prediction_root or str(cfg.paths.output_dir))
    output_root = ensure_dir(
        args.output_root
        or (Path(str(cfg.paths.output_dir)) / "downstream_task" / "task3_boundary_st_transition")
    )
    figure_dir = ensure_dir(output_root / "figures")
    spatial_dir = ensure_dir(figure_dir / "spatial_boundary_maps")

    print(f"Config: {config_path}")
    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Prediction root: {prediction_root}")
    print(f"Output root: {output_root}")
    print(f"Folds: {args.folds}")
    print(f"Boundary width: {args.boundary_width} spot-spacing units")

    results: List[Dict[str, object]] = []
    all_spot_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []

    for fold in [str(value) for value in args.folds]:
        print("\n" + "-" * 100)
        print(f"Processing {fold}")
        result = analyze_slice(
            cfg,
            fold,
            checkpoint_root,
            prediction_root,
            seed=int(args.seed),
            batch_size=args.batch_size,
            channels_last=bool(args.channels_last),
            threshold=args.threshold,
            force_infer=bool(args.force_infer),
            spot_radius_factor=float(args.spot_radius_factor),
            spot_radius_min_px=float(args.spot_radius_min_px),
            spot_radius_max_px=float(args.spot_radius_max_px),
            min_spot_tissue_fraction=float(args.min_spot_tissue_fraction),
            boundary_width=float(args.boundary_width),
            max_distance=float(args.max_distance),
            min_molecular_anchor_spots=int(args.min_molecular_anchor_spots),
            molecular_anchor_fraction=float(args.molecular_anchor_fraction),
            max_molecular_anchor_spots=int(args.max_molecular_anchor_spots),
        )
        results.append(result)
        all_spot_rows.extend(result["spot_rows"])
        # Flatten nested values for CSV while preserving full dictionaries in JSON.
        metrics = dict(result["metrics"])
        metrics_csv = {
            key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
            for key, value in metrics.items()
        }
        metric_rows.append(metrics_csv)

        save_spatial_boundary_figure(
            spatial_dir / f"{fold}_boundary_st_transition.png",
            result,
            int(args.dpi),
        )
        print(
            f"[{fold}] valid_spots={metrics['Valid_Spots']} | "
            f"distance_MAE={float(metrics['SignedDistance_MAE']):.3f} | "
            f"zone_macroF1={float(metrics['Zone_MacroF1']):.3f} | "
            f"boundary_F1={float(metrics['Boundary_F1']):.3f} | "
            f"molecular_alignment={float(metrics['Molecular_PredDistance_Spearman']):.3f}"
        )

    write_csv(output_root / "task3_boundary_st_spot_table.csv", all_spot_rows)
    write_csv(output_root / "task3_boundary_st_per_slice_metrics.csv", metric_rows)

    save_probability_uncertainty_transition(
        figure_dir / "01_probability_uncertainty_transition.png",
        results,
        float(args.max_distance),
        float(args.distance_bin_width),
        int(args.dpi),
    )
    save_molecular_transition_curves(
        figure_dir / "02_molecular_transition_curves.png",
        results,
        float(args.max_distance),
        float(args.distance_bin_width),
        int(args.dpi),
    )
    save_pca_zones(
        figure_dir / "03_st_pca_boundary_zones.png",
        results,
        int(args.dpi),
    )
    save_zone_confusions(
        figure_dir / "04_boundary_zone_confusion_matrices.png",
        results,
        int(args.dpi),
    )
    save_metric_bars(
        figure_dir / "05_boundary_transition_metrics.png",
        results,
        int(args.dpi),
    )
    manuscript_summary_paths: List[str] = []
    for result in results:
        fold = str(result["fold"])
        summary_path = figure_dir / f"06_task3_manuscript_summary_{fold}.png"
        save_manuscript_summary(
            summary_path,
            result,
            float(args.max_distance),
            float(args.distance_bin_width),
            int(args.dpi),
        )
        manuscript_summary_paths.append(str(summary_path))
        print(f"Saved manuscript summary [{fold}]: {summary_path.name}")

    summary = {
        "script_version": SCRIPT_VERSION,
        "task": "Boundary-aware spatial transcriptomic transition analysis",
        "model": MODEL_NAME,
        "seed": int(args.seed),
        "folds": [str(result["fold"]) for result in results],
        "boundary_definition": {
            "distance_units": "median spot spacing",
            "negative_direction": "inside tumor",
            "positive_direction": "outside tumor",
            "boundary_width": float(args.boundary_width),
            "zones": {
                "Tumor core": f"d <= -{args.boundary_width}",
                "Inner boundary": f"-{args.boundary_width} < d <= 0",
                "Outer boundary": f"0 < d <= {args.boundary_width}",
                "Distant non-tumor": f"d > {args.boundary_width}",
            },
        },
        "molecular_axis_note": (
            "The descriptive ST transition axis is defined in within-slice "
            "standardized 512-D scGPT-spatial feature space using the deepest "
            "GT tumor-side spots and the farthest GT non-tumor-side spots. "
            "This adaptive anchor protocol remains stable when the strict tumor-core "
            "zone contains very few spots."
        ),
        "molecular_anchor_protocol": {
            "minimum_spots_per_side": int(args.min_molecular_anchor_spots),
            "target_fraction_per_side": float(args.molecular_anchor_fraction),
            "maximum_spots_per_side": int(args.max_molecular_anchor_spots),
        },
        "per_slice_metrics": [result["metrics"] for result in results],
        "outputs": {
            "spot_table": str(output_root / "task3_boundary_st_spot_table.csv"),
            "metrics": str(output_root / "task3_boundary_st_per_slice_metrics.csv"),
            "figure_dir": str(figure_dir),
            "manuscript_summaries": manuscript_summary_paths,
        },
    }
    write_json(output_root / "task3_boundary_st_summary.json", summary)

    print("\n" + "=" * 100)
    print("Task 3 completed successfully")
    print("=" * 100)
    print(f"Results: {output_root}")
    print(f"Figures: {figure_dir}")
    print("Per-slice manuscript summaries:")
    for summary_path in manuscript_summary_paths:
        print(f"  - {Path(summary_path).name}")


if __name__ == "__main__":
    main()

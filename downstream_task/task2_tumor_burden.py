from __future__ import annotations

"""Downstream Task 2: regional and whole-section tumor-burden quantification.

The task evaluates whether HiST-Seg can convert pixel-level cancer probability
maps into reliable quantitative tumor-burden estimates.

Default protocol
----------------
* Test sections: ViHBC, DuCIS, XeHBC.
* Model seed: 42.
* Checkpoints:
    checkpoints_pathst_cross_p512_ctx4/seed_42/fold_{slice}/
* Regional analysis: non-overlapping 1024 x 1024 pixel ROIs.
* Keep ROIs with at least 30% tissue coverage.
* Binary burden is the primary area-based estimate.
* Soft burden (mean cancer probability) is reported as an auxiliary estimate.

This standalone script does not import downstream_utils.py. It directly uses
engine/inference_cross_dataset.py from the current HiST-Seg project. Existing
prediction caches are reused unless --force_infer is specified; no training is performed.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Add the project root so the sibling ``engine``, ``model`` and ``utils``
# packages can be imported when this script is executed from downstream_task/.
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
        "Cannot import the current project inference stack. Expected files:\n"
        f"  {PROJECT_ROOT / 'engine' / 'inference_cross_dataset.py'}\n"
        f"  {PROJECT_ROOT / 'model' / 'pathst_net.py'}\n"
        f"  {PROJECT_ROOT / 'utils' / 'config.py'}\n"
        "Run this script inside the HiST-Seg project shown in your directory tree."
    ) from exc


SCRIPT_VERSION = "2026-07-16-STANDALONE-FINAL-v2"
MODEL_NAME = "HiST-Seg"
DEFAULT_FOLDS = ["ViHBC", "DuCIS", "XeHBC"]
DEFAULT_CHECKPOINT_ROOT = "../engine/checkpoints_pathst_cross_p512_ctx4/seed_42"
DEFAULT_CONFIG_CANDIDATES = (
    "cross_dataset.yaml",
    "../engine/cross_dataset.yaml",
    "cross_dataset(2).yaml",
)

# Publication-oriented typography. All values can be changed here if needed.
FONT_BASE = 16
FONT_TICK = 14
FONT_AXIS = 17
FONT_TITLE = 19
FONT_SUPTITLE = 22
FONT_LEGEND = 13


# =============================================================================
# General utilities
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
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


def resolve_config_path(value: Optional[str]) -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates: List[Path] = []
    if value:
        candidates.extend([
            Path(value),
            script_dir / value,
            script_dir.parent / value,
        ])
    else:
        for candidate in DEFAULT_CONFIG_CANDIDATES:
            candidates.extend([
                Path(candidate),
                script_dir / candidate,
                script_dir.parent / candidate,
            ])
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
        f"Checkpoint for {fold} was not found under {fold_dir}. "
        f"Preferred names:\n  {attempted}"
    )


def _load_slide_inputs(cfg, fold: str) -> Dict[str, object]:
    """Load H&E, label and tissue mask for one external section."""
    dataset_name, _, base = dataset_for_slice(cfg, str(fold))
    he_bgr = cv_imread(os.path.join(base, "he_resized.png"), cv2.IMREAD_COLOR)
    label = cv_imread(os.path.join(base, "label_resized.png"), cv2.IMREAD_GRAYSCALE)
    tissue = cv_imread(os.path.join(base, "mask_resized.png"), cv2.IMREAD_GRAYSCALE)

    if he_bgr is None:
        raise FileNotFoundError(os.path.join(base, "he_resized.png"))
    he = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2RGB)

    if tissue is None:
        tissue = np.full(he.shape[:2], 255, dtype=np.uint8)
    if tissue.shape != he.shape[:2]:
        raise ValueError(
            f"[{fold}] tissue/H&E shape mismatch: {tissue.shape} vs {he.shape[:2]}"
        )
    if label is None:
        raise FileNotFoundError(os.path.join(base, "label_resized.png"))
    if label.shape != he.shape[:2]:
        raise ValueError(
            f"[{fold}] label/H&E shape mismatch: {label.shape} vs {he.shape[:2]}"
        )

    return {
        "dataset": str(dataset_name),
        "base": str(base),
        "he": he,
        "label": label,
        "tissue": tissue,
    }


def _read_cached_threshold(
    metrics_path: Path,
    cfg,
    threshold_override: Optional[float],
) -> Tuple[float, str]:
    if threshold_override is not None:
        return float(threshold_override), "command line"

    if metrics_path.is_file():
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            value = payload.get("Threshold")
            if value is not None and np.isfinite(float(value)):
                return float(value), str(payload.get("Threshold_Source", "cached metrics"))
        except Exception as exc:
            print(f"[warning] Could not read cached threshold from {metrics_path}: {exc}")

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
    """Reuse a seed-42 prediction cache or run current cross-dataset inference.

    This implementation intentionally calls ``engine/inference_cross_dataset.py``
    and does not depend on the obsolete downstream_utils import stack.
    """
    fold = str(fold)
    slide = _load_slide_inputs(cfg, fold)
    fold_output = Path(prediction_root) / f"seed_{int(seed)}" / fold
    cache_path = fold_output / "prediction.npz"
    metrics_path = fold_output / "metrics.json"

    if cache_path.is_file() and not force_infer:
        with np.load(cache_path, allow_pickle=False) as archive:
            if "probability" not in archive.files:
                raise KeyError(
                    f"Cached prediction lacks 'probability': {cache_path}; "
                    f"available={archive.files}"
                )
            probability = np.asarray(archive["probability"], dtype=np.float32)
            cached_prediction = (
                np.asarray(archive["prediction"], dtype=np.uint8)
                if "prediction" in archive.files
                else None
            )

        if probability.shape != np.asarray(slide["tissue"]).shape:
            raise ValueError(
                f"[{fold}] cached probability shape {probability.shape} does not "
                f"match tissue {np.asarray(slide['tissue']).shape}: {cache_path}"
            )

        used_threshold, threshold_source = _read_cached_threshold(
            metrics_path, cfg, threshold
        )
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

    # Use the arrays returned by inference; they are sourced from the same files
    # as ``slide`` but avoid unnecessary second reads for label/tissue.
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


def finite_vector(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def concordance_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (CCC)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    var_x, var_y = float(np.var(x)), float(np.var(y))
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if denominator <= 1e-15:
        return 1.0 if np.allclose(x, y) else float("nan")
    return float(2.0 * covariance / denominator)


def agreement_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float | int]:
    mask = np.isfinite(gt) & np.isfinite(pred)
    gt, pred = gt[mask], pred[mask]
    if len(gt) == 0:
        return {"N": 0}

    error = pred - gt
    abs_error = np.abs(error)
    squared_error = error ** 2
    bias = float(np.mean(error))
    error_sd = float(np.std(error, ddof=1)) if len(error) > 1 else 0.0
    gt_variance = float(np.sum((gt - np.mean(gt)) ** 2))
    r2 = (
        float(1.0 - np.sum(squared_error) / gt_variance)
        if gt_variance > 1e-15
        else float("nan")
    )

    return {
        "N": int(len(gt)),
        "MAE": float(np.mean(abs_error)),
        "Median_AE": float(np.median(abs_error)),
        "P95_AE": float(np.percentile(abs_error, 95)),
        "RMSE": float(math.sqrt(np.mean(squared_error))),
        "Mean_Bias": bias,
        "Error_SD": error_sd,
        "LoA_Lower": float(bias - 1.96 * error_sd),
        "LoA_Upper": float(bias + 1.96 * error_sd),
        "Pearson": safe_pearson(gt, pred),
        "Spearman": safe_spearman(gt, pred),
        "CCC": concordance_correlation_coefficient(gt, pred),
        "R2": r2,
    }


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


# =============================================================================
# Burden computation
# =============================================================================

def validate_slide_arrays(data: Mapping[str, object], fold: str) -> None:
    required = ("probability", "prediction", "tissue", "label", "he")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"[{fold}] inference output misses keys: {missing}")

    probability = np.asarray(data["probability"])
    prediction = np.asarray(data["prediction"])
    tissue = np.asarray(data["tissue"])
    label = data["label"]
    if label is None:
        raise RuntimeError(f"[{fold}] ground-truth label is required for burden analysis")
    label = np.asarray(label)

    shapes = {
        "probability": probability.shape[:2],
        "prediction": prediction.shape[:2],
        "tissue": tissue.shape[:2],
        "label": label.shape[:2],
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"[{fold}] spatial shape mismatch: {shapes}")


def whole_section_burden(data: Mapping[str, object], fold: str) -> Dict[str, object]:
    probability = np.asarray(data["probability"], dtype=np.float32)
    prediction = np.asarray(data["prediction"]) > 0
    tissue = np.asarray(data["tissue"]) > 127
    label = np.asarray(data["label"]) > 127

    if not np.any(tissue):
        raise RuntimeError(f"[{fold}] tissue mask contains no valid tissue pixels")

    gt = float(np.mean(label[tissue]))
    soft = float(np.mean(probability[tissue]))
    binary = float(np.mean(prediction[tissue]))

    return {
        "Dataset": str(data.get("dataset", fold)),
        "Slice": fold,
        "Model": MODEL_NAME,
        "Threshold": float(data.get("threshold", 0.5)),
        "Threshold_Source": str(data.get("threshold_source", "unknown")),
        "Prediction_From_Cache": bool(data.get("from_cache", False)),
        "Tissue_Pixels": int(np.count_nonzero(tissue)),
        "GT_Burden": gt,
        "Pred_Burden_Soft": soft,
        "Pred_Burden_Binary": binary,
        "Soft_Abs_Error": abs(soft - gt),
        "Binary_Abs_Error": abs(binary - gt),
        "Soft_Signed_Error": soft - gt,
        "Binary_Signed_Error": binary - gt,
    }


def grid_starts(length: int, roi_size: int, stride: int) -> List[int]:
    if length <= 0:
        return []
    return list(range(0, length, stride))


def extract_regional_burdens(
    data: Mapping[str, object],
    fold: str,
    roi_size: int,
    roi_stride: int,
    min_tissue_fraction: float,
    min_tissue_pixels: int,
) -> List[Dict[str, object]]:
    probability = np.asarray(data["probability"], dtype=np.float32)
    prediction = np.asarray(data["prediction"]) > 0
    tissue = np.asarray(data["tissue"]) > 127
    label = np.asarray(data["label"]) > 127
    height, width = tissue.shape

    rows: List[Dict[str, object]] = []
    roi_index = 0
    for y0 in grid_starts(height, roi_size, roi_stride):
        y1 = min(height, y0 + roi_size)
        for x0 in grid_starts(width, roi_size, roi_stride):
            x1 = min(width, x0 + roi_size)
            roi_area = max(1, (x1 - x0) * (y1 - y0))
            tissue_roi = tissue[y0:y1, x0:x1]
            tissue_pixels = int(np.count_nonzero(tissue_roi))
            tissue_fraction = float(tissue_pixels / roi_area)

            if tissue_fraction < min_tissue_fraction:
                continue
            if tissue_pixels < min_tissue_pixels:
                continue

            prob_roi = probability[y0:y1, x0:x1]
            pred_roi = prediction[y0:y1, x0:x1]
            gt_roi = label[y0:y1, x0:x1]

            gt_burden = float(np.mean(gt_roi[tissue_roi]))
            soft_burden = float(np.mean(prob_roi[tissue_roi]))
            binary_burden = float(np.mean(pred_roi[tissue_roi]))

            rows.append({
                "Dataset": str(data.get("dataset", fold)),
                "Slice": fold,
                "ROI_Index": int(roi_index),
                "x0": int(x0),
                "y0": int(y0),
                "x1": int(x1),
                "y1": int(y1),
                "ROI_Width": int(x1 - x0),
                "ROI_Height": int(y1 - y0),
                "ROI_Center_X": float((x0 + x1) / 2.0),
                "ROI_Center_Y": float((y0 + y1) / 2.0),
                "ROI_Area": int(roi_area),
                "Tissue_Pixels": tissue_pixels,
                "Tissue_Fraction": tissue_fraction,
                "GT_Burden": gt_burden,
                "Pred_Burden_Soft": soft_burden,
                "Pred_Burden_Binary": binary_burden,
                "Soft_Signed_Error": soft_burden - gt_burden,
                "Binary_Signed_Error": binary_burden - gt_burden,
                "Soft_Abs_Error": abs(soft_burden - gt_burden),
                "Binary_Abs_Error": abs(binary_burden - gt_burden),
            })
            roi_index += 1

    return rows


def regional_metric_table(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    slices = list(dict.fromkeys(str(row["Slice"]) for row in rows))

    for group_name, group_rows in [
        (slice_id, [row for row in rows if str(row["Slice"]) == slice_id])
        for slice_id in slices
    ] + [("Pooled", list(rows))]:
        gt = np.asarray([float(row["GT_Burden"]) for row in group_rows], dtype=np.float64)
        for mode, key in (
            ("Soft", "Pred_Burden_Soft"),
            ("Binary", "Pred_Burden_Binary"),
        ):
            pred = np.asarray([float(row[key]) for row in group_rows], dtype=np.float64)
            result.append({
                "Scope": group_name,
                "Burden_Type": mode,
                **agreement_metrics(gt, pred),
            })

    # Macro averages of per-slice error metrics. Correlations are intentionally
    # not macro-averaged because they are not additive statistics.
    for mode in ("Soft", "Binary"):
        per_slice = [
            row for row in result
            if row["Scope"] != "Pooled" and row["Burden_Type"] == mode
        ]
        if per_slice:
            result.append({
                "Scope": "MacroAverage",
                "Burden_Type": mode,
                "N": int(sum(int(row.get("N", 0)) for row in per_slice)),
                "MAE": float(np.nanmean([float(row.get("MAE", np.nan)) for row in per_slice])),
                "RMSE": float(np.nanmean([float(row.get("RMSE", np.nan)) for row in per_slice])),
                "Median_AE": float(np.nanmean([float(row.get("Median_AE", np.nan)) for row in per_slice])),
                "P95_AE": float(np.nanmean([float(row.get("P95_AE", np.nan)) for row in per_slice])),
                "Mean_Bias": float(np.nanmean([float(row.get("Mean_Bias", np.nan)) for row in per_slice])),
                "CCC": float(np.nanmean([float(row.get("CCC", np.nan)) for row in per_slice])),
            })

    return result


# =============================================================================
# Figures
# =============================================================================

def dataset_colors(datasets: Sequence[str]) -> Dict[str, object]:
    order = list(dict.fromkeys(datasets))
    cmap = plt.get_cmap("tab10")
    return {dataset: cmap(index % 10) for index, dataset in enumerate(order)}


def save_whole_section_bar(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    labels = [str(row["Slice"]) for row in rows]
    gt = [float(row["GT_Burden"]) for row in rows]
    soft = [float(row["Pred_Burden_Soft"]) for row in rows]
    binary = [float(row["Pred_Burden_Binary"]) for row in rows]

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 6.8), dpi=dpi)
    ax.bar(x - width, gt, width=width, label="Ground truth")
    ax.bar(x, soft, width=width, label=f"{MODEL_NAME} soft burden")
    ax.bar(x + width, binary, width=width, label=f"{MODEL_NAME} binary burden")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Whole-section tumor burden")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Whole-section tumor-burden quantification", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=1, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scatter_panel(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    prediction_key: str,
    burden_name: str,
    colors: Mapping[str, object],
) -> None:
    gt = np.asarray([float(row["GT_Burden"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row[prediction_key]) for row in rows], dtype=np.float64)
    datasets = [str(row["Dataset"]) for row in rows]

    for dataset in dict.fromkeys(datasets):
        mask = np.asarray([value == dataset for value in datasets], dtype=bool)
        ax.scatter(
            gt[mask], pred[mask], s=58, alpha=0.82,
            edgecolors="white", linewidths=0.6,
            color=colors[dataset], label=dataset,
        )

    metrics = agreement_metrics(gt, pred)
    ax.plot([0, 1], [0, 1], "--", linewidth=1.4, color="gray", label="Ideal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ground-truth regional burden")
    ax.set_ylabel("Predicted regional burden")
    ax.set_title(
        f"{burden_name}\n"
        f"MAE={metrics.get('MAE', np.nan):.3f}, "
        f"CCC={metrics.get('CCC', np.nan):.3f}, "
        f"r={metrics.get('Pearson', np.nan):.3f}",
        fontweight="bold",
    )
    ax.grid(alpha=0.23)


def save_regional_agreement(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    colors = dataset_colors([str(row["Dataset"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.0), dpi=dpi)
    scatter_panel(axes[0], rows, "Pred_Burden_Soft", "Soft burden", colors)
    scatter_panel(axes[1], rows, "Pred_Burden_Binary", "Binary burden", colors)

    handles, labels = axes[1].get_legend_handles_labels()
    unique: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(), unique.keys(), frameon=False,
        loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=len(unique),
    )
    fig.suptitle("Regional tumor-burden agreement", fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bland_altman_panel(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    prediction_key: str,
    burden_name: str,
    colors: Mapping[str, object],
) -> None:
    gt = np.asarray([float(row["GT_Burden"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row[prediction_key]) for row in rows], dtype=np.float64)
    datasets = [str(row["Dataset"]) for row in rows]
    mean_value = (gt + pred) / 2.0
    difference = pred - gt
    bias = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0
    lower, upper = bias - 1.96 * sd, bias + 1.96 * sd

    for dataset in dict.fromkeys(datasets):
        mask = np.asarray([value == dataset for value in datasets], dtype=bool)
        ax.scatter(
            mean_value[mask], difference[mask], s=54, alpha=0.82,
            edgecolors="white", linewidths=0.6, color=colors[dataset], label=dataset,
        )

    ax.axhline(bias, linewidth=1.8, color="black", label=f"Bias={bias:.3f}")
    ax.axhline(lower, linestyle="--", linewidth=1.4, color="gray", label=f"Lower LoA={lower:.3f}")
    ax.axhline(upper, linestyle="--", linewidth=1.4, color="gray", label=f"Upper LoA={upper:.3f}")
    ax.set_xlabel("Mean of predicted and GT burden")
    ax.set_ylabel("Predicted $-$ GT burden")
    ax.set_title(burden_name, fontweight="bold")
    ax.grid(alpha=0.23)


def save_bland_altman(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    colors = dataset_colors([str(row["Dataset"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.6), dpi=dpi)
    bland_altman_panel(axes[0], rows, "Pred_Burden_Soft", "Soft burden", colors)
    bland_altman_panel(axes[1], rows, "Pred_Burden_Binary", "Binary burden", colors)
    axes[1].legend(frameon=False, fontsize=11, loc="best")
    fig.suptitle("Bland--Altman analysis of regional tumor burden", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_error_boxplot(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    slices = list(dict.fromkeys(str(row["Slice"]) for row in rows))
    soft_data = [
        [float(row["Soft_Abs_Error"]) for row in rows if str(row["Slice"]) == slide]
        for slide in slices
    ]
    binary_data = [
        [float(row["Binary_Abs_Error"]) for row in rows if str(row["Slice"]) == slide]
        for slide in slices
    ]

    positions = np.arange(len(slices), dtype=np.float64)
    offset = 0.18
    fig, ax = plt.subplots(figsize=(10.2, 6.8), dpi=dpi)
    soft_plot = ax.boxplot(
        soft_data, positions=positions - offset, widths=0.28,
        patch_artist=True, showfliers=False,
    )
    binary_plot = ax.boxplot(
        binary_data, positions=positions + offset, widths=0.28,
        patch_artist=True, showfliers=False,
    )
    for patch in soft_plot["boxes"]:
        patch.set_facecolor(plt.get_cmap("tab10")(0))
        patch.set_alpha(0.72)
    for patch in binary_plot["boxes"]:
        patch.set_facecolor(plt.get_cmap("tab10")(1))
        patch.set_alpha(0.72)

    ax.set_xticks(positions)
    ax.set_xticklabels(slices, fontweight="bold")
    ax.set_ylabel("Regional absolute burden error")
    ax.set_title("Regional tumor-burden error distributions", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        [soft_plot["boxes"][0], binary_plot["boxes"][0]],
        ["Soft burden", "Binary burden"], frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def resize_rgb_for_display(image: np.ndarray, max_size: int = 1600) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return image.copy(), 1.0
    scale = float(max_size) / float(longest)
    output = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return output, scale


def draw_roi_overlay(
    ax: plt.Axes,
    he_rgb: np.ndarray,
    scale: float,
    rows: Sequence[Mapping[str, object]],
    value_key: str,
    cmap_name: str,
    title: str,
    vmin: float,
    vmax: float,
) -> ScalarMappable:
    ax.imshow(he_rgb)
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    for row in rows:
        value = float(row[value_key])
        rectangle = patches.Rectangle(
            (float(row["x0"]) * scale, float(row["y0"]) * scale),
            (float(row["x1"]) - float(row["x0"])) * scale,
            (float(row["y1"]) - float(row["y0"])) * scale,
            linewidth=0.65,
            edgecolor="white",
            facecolor=cmap(norm(value)),
            alpha=0.76,
        )
        ax.add_patch(rectangle)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    return ScalarMappable(norm=norm, cmap=cmap)


def save_spatial_burden_maps(
    path: Path,
    fold: str,
    he: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    he_array = np.asarray(he)
    # downstream_utils may return RGB or BGR depending on project version.
    # The existing downstream scripts display ``he`` directly, so preserve it.
    he_show, scale = resize_rgb_for_display(he_array, max_size=1600)

    max_error = max(
        0.05,
        float(max(
            max(float(row["Soft_Abs_Error"]) for row in rows),
            max(float(row["Binary_Abs_Error"]) for row in rows),
        )),
    )

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.5), dpi=dpi)
    axes[0, 0].imshow(he_show)
    axes[0, 0].set_title(f"{fold} H&E", fontweight="bold")
    axes[0, 0].axis("off")

    burden_maps = []
    burden_maps.append(draw_roi_overlay(
        axes[0, 1], he_show, scale, rows,
        "GT_Burden", "magma", "GT regional burden", 0.0, 1.0,
    ))
    burden_maps.append(draw_roi_overlay(
        axes[0, 2], he_show, scale, rows,
        "Pred_Burden_Soft", "magma", f"{MODEL_NAME} soft burden", 0.0, 1.0,
    ))
    burden_maps.append(draw_roi_overlay(
        axes[1, 0], he_show, scale, rows,
        "Pred_Burden_Binary", "magma", f"{MODEL_NAME} binary burden", 0.0, 1.0,
    ))
    soft_error_map = draw_roi_overlay(
        axes[1, 1], he_show, scale, rows,
        "Soft_Abs_Error", "Reds", "Soft-burden absolute error", 0.0, max_error,
    )
    binary_error_map = draw_roi_overlay(
        axes[1, 2], he_show, scale, rows,
        "Binary_Abs_Error", "Reds", "Binary-burden absolute error", 0.0, max_error,
    )

    burden_colorbar = fig.colorbar(
        burden_maps[0], ax=[axes[0, 1], axes[0, 2], axes[1, 0]],
        fraction=0.025, pad=0.018,
    )
    burden_colorbar.set_label("Tumor burden", fontsize=FONT_AXIS)
    burden_colorbar.ax.tick_params(labelsize=FONT_TICK)

    error_colorbar = fig.colorbar(
        soft_error_map, ax=[axes[1, 1], axes[1, 2]],
        fraction=0.025, pad=0.018,
    )
    error_colorbar.set_label("Absolute error", fontsize=FONT_AXIS)
    error_colorbar.ax.tick_params(labelsize=FONT_TICK)

    fig.suptitle(f"{fold}: regional tumor-burden quantification", fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.84, bottom=0.03, top=0.92, wspace=0.06, hspace=0.13)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_summary_panel(
    path: Path,
    whole_rows: Sequence[Mapping[str, object]],
    regional_rows: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    """Create one compact, manuscript-ready four-panel summary."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.0), dpi=dpi)

    # (a) Whole-section burden.
    labels = [str(row["Slice"]) for row in whole_rows]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    axes[0, 0].bar(x - width, [float(row["GT_Burden"]) for row in whole_rows], width, label="GT")
    axes[0, 0].bar(x, [float(row["Pred_Burden_Soft"]) for row in whole_rows], width, label="Soft")
    axes[0, 0].bar(x + width, [float(row["Pred_Burden_Binary"]) for row in whole_rows], width, label="Binary")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, fontweight="bold")
    axes[0, 0].set_ylabel("Whole-section burden")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("(a) Whole-section burden", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=11)
    axes[0, 0].grid(axis="y", alpha=0.23)

    # (b) Binary regional agreement.
    colors = dataset_colors([str(row["Dataset"]) for row in regional_rows])
    scatter_panel(
        axes[0, 1], regional_rows,
        "Pred_Burden_Binary", "(b) Binary regional agreement", colors,
    )

    # (c) Binary Bland-Altman.
    bland_altman_panel(
        axes[1, 0], regional_rows,
        "Pred_Burden_Binary", "(c) Binary Bland--Altman", colors,
    )

    # (d) Regional absolute errors.
    slices = list(dict.fromkeys(str(row["Slice"]) for row in regional_rows))
    positions = np.arange(len(slices), dtype=np.float64)
    data = [
        [float(row["Binary_Abs_Error"]) for row in regional_rows if str(row["Slice"]) == slide]
        for slide in slices
    ]
    box = axes[1, 1].boxplot(data, positions=positions, widths=0.48, patch_artist=True, showfliers=False)
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(plt.get_cmap("tab10")(index % 10))
        patch.set_alpha(0.72)
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(slices, fontweight="bold")
    axes[1, 1].set_ylabel("Binary-burden absolute error")
    axes[1, 1].set_title("(d) Regional error distributions", loc="left", fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.23)

    # Avoid duplicated legends in the summary figure.
    if axes[0, 1].get_legend() is not None:
        axes[0, 1].get_legend().remove()
    if axes[1, 0].get_legend() is not None:
        axes[1, 0].get_legend().remove()

    fig.suptitle(
        "Regional and whole-section tumor-burden quantification",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downstream Task 2: regional and whole-section tumor-burden "
            "quantification using trained HiST-Seg seed-42 models."
        )
    )
    parser.add_argument("--config", default=None, help="Cross-dataset YAML path.")
    parser.add_argument("--folds", nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument(
        "--checkpoint_root",
        default=DEFAULT_CHECKPOINT_ROOT,
        help=(
            "Directory containing fold_ViHBC, fold_DuCIS and fold_XeHBC. "
            "Default: ../engine/checkpoints_pathst_cross_p512_ctx4/seed_42"
        ),
    )
    parser.add_argument(
        "--prediction_root",
        default=None,
        help=(
            "Existing/inference prediction root. By default cfg.paths.output_dir "
            "is used. Existing prediction.npz files are reused."
        ),
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
        help="Ignore prediction caches and rerun whole-section inference.",
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        default=512,
        help="Regional ROI side length in full-resolution pixels.",
    )
    parser.add_argument(
        "--roi_stride",
        type=int,
        default=512,
        help="ROI stride in full-resolution pixels. Default is non-overlapping.",
    )
    parser.add_argument(
        "--min_tissue_fraction",
        type=float,
        default=0.30,
        help="Minimum tissue coverage required to retain an ROI.",
    )
    parser.add_argument(
        "--min_tissue_pixels",
        type=int,
        default=1,
        help="Additional minimum number of tissue pixels in a retained ROI.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    print(f"[Task2 script version] {SCRIPT_VERSION}")
    print(f"[Running file] {Path(__file__).resolve()}")
    args = parse_args()
    apply_publication_style()

    if args.roi_size <= 0 or args.roi_stride <= 0:
        raise ValueError("--roi_size and --roi_stride must be positive")
    if not 0.0 <= args.min_tissue_fraction <= 1.0:
        raise ValueError("--min_tissue_fraction must be in [0, 1]")
    if args.roi_stride < args.roi_size:
        print(
            "[warning] roi_stride < roi_size: ROIs overlap. The default "
            "non-overlapping protocol uses roi_stride == roi_size."
        )

    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    checkpoint_root = resolve_directory(args.checkpoint_root, "seed-42 checkpoint root")

    print("=" * 100)
    print("Downstream Task 2: regional and whole-section tumor-burden quantification")
    print("=" * 100)
    print(f"Config          : {config_path}")
    print(f"Checkpoint root : {checkpoint_root}")
    print(f"Folds           : {args.folds}")
    print(f"ROI protocol    : size={args.roi_size}, stride={args.roi_stride}, "
          f"min_tissue_fraction={args.min_tissue_fraction}")

    for fold in args.folds:
        checkpoint = locate_checkpoint(checkpoint_root, fold)
        print(f"Checkpoint [{fold}]: {checkpoint}")

    prediction_root = args.prediction_root or str(cfg.paths.output_dir)
    default_output = (
        Path(str(cfg.paths.output_dir))
        / "downstream_task"
        / "task2_regional_whole_section_tumor_burden"
    )
    output_root = ensure_dir(args.output_root or default_output)
    figure_dir = ensure_dir(output_root / "figures")
    spatial_dir = ensure_dir(figure_dir / "spatial_maps")

    whole_rows: List[Dict[str, object]] = []
    regional_rows: List[Dict[str, object]] = []

    for fold in args.folds:
        print("\n" + "-" * 100)
        print(f"Processing {fold}")

        data = load_prediction_or_infer_current(
            cfg,
            fold,
            checkpoint_root,
            prediction_root,
            batch_size=args.batch_size,
            channels_last=args.channels_last,
            threshold=args.threshold,
            force_infer=args.force_infer,
            seed=42,
        )
        validate_slide_arrays(data, fold)

        whole_row = whole_section_burden(data, fold)
        whole_rows.append(whole_row)

        fold_regions = extract_regional_burdens(
            data=data,
            fold=fold,
            roi_size=args.roi_size,
            roi_stride=args.roi_stride,
            min_tissue_fraction=args.min_tissue_fraction,
            min_tissue_pixels=args.min_tissue_pixels,
        )
        if not fold_regions:
            raise RuntimeError(
                f"[{fold}] no ROI passed the tissue filter. Reduce "
                "--min_tissue_fraction or inspect the tissue mask."
            )
        regional_rows.extend(fold_regions)

        save_spatial_burden_maps(
            spatial_dir / f"{fold}_regional_tumor_burden_maps.png",
            fold,
            np.asarray(data["he"]),
            fold_regions,
            args.dpi,
        )

        print(
            f"[{fold}] regions={len(fold_regions)} | "
            f"GT={whole_row['GT_Burden']:.4f} | "
            f"soft={whole_row['Pred_Burden_Soft']:.4f} | "
            f"binary={whole_row['Pred_Burden_Binary']:.4f} | "
            f"from_cache={whole_row['Prediction_From_Cache']}"
        )

    metric_rows = regional_metric_table(regional_rows)

    write_csv(output_root / "task2_whole_section_burden.csv", whole_rows)
    write_csv(output_root / "task2_regional_burden_rois.csv", regional_rows)
    write_csv(output_root / "task2_regional_burden_metrics.csv", metric_rows)

    save_whole_section_bar(
        figure_dir / "01_whole_section_burden_grouped_bar.png",
        whole_rows,
        args.dpi,
    )
    save_regional_agreement(
        figure_dir / "02_regional_burden_agreement.png",
        regional_rows,
        args.dpi,
    )
    save_bland_altman(
        figure_dir / "03_regional_bland_altman.png",
        regional_rows,
        args.dpi,
    )
    save_error_boxplot(
        figure_dir / "04_regional_absolute_error_boxplot.png",
        regional_rows,
        args.dpi,
    )
    save_summary_panel(
        figure_dir / "05_task2_manuscript_summary.png",
        whole_rows,
        regional_rows,
        args.dpi,
    )

    pooled_metrics = {
        str(row["Burden_Type"]): row
        for row in metric_rows
        if str(row["Scope"]) == "Pooled"
    }
    summary = {
        "task": "regional_and_whole_section_tumor_burden_quantification",
        "model": MODEL_NAME,
        "seed": 42,
        "config": str(config_path),
        "checkpoint_root": str(checkpoint_root),
        "prediction_root": str(prediction_root),
        "folds": list(args.folds),
        "roi_protocol": {
            "roi_size": int(args.roi_size),
            "roi_stride": int(args.roi_stride),
            "non_overlapping": bool(args.roi_stride >= args.roi_size),
            "min_tissue_fraction": float(args.min_tissue_fraction),
            "min_tissue_pixels": int(args.min_tissue_pixels),
        },
        "num_regions": int(len(regional_rows)),
        "num_regions_per_slice": {
            fold: int(sum(str(row["Slice"]) == fold for row in regional_rows))
            for fold in args.folds
        },
        "whole_section_results": whole_rows,
        "pooled_regional_metrics": pooled_metrics,
        "all_regional_metric_rows": metric_rows,
        "figures": sorted(path.name for path in figure_dir.glob("*.png")),
        "spatial_figures": sorted(path.name for path in spatial_dir.glob("*.png")),
    }
    write_json(output_root / "task2_tumor_burden_summary.json", summary)

    print("\n" + "=" * 100)
    print("Task 2 completed")
    print("=" * 100)
    print(f"Output directory: {output_root}")
    print(f"Total retained regional ROIs: {len(regional_rows)}")
    for mode in ("Binary", "Soft"):
        row = pooled_metrics.get(mode)
        if row:
            print(
                f"{mode:6s} pooled regional burden | "
                f"MAE={float(row.get('MAE', np.nan)):.4f} | "
                f"RMSE={float(row.get('RMSE', np.nan)):.4f} | "
                f"CCC={float(row.get('CCC', np.nan)):.4f} | "
                f"Pearson={float(row.get('Pearson', np.nan)):.4f}"
            )


if __name__ == "__main__":
    main()

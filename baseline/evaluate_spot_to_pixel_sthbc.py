# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
# Evaluation protocol:
# Predictions must first be generated from the native, original-scale spot data.
# This script uses resized_data and adata.obsm['spatial_resized'] only to map
# those predictions into the common aligned pixel-level evaluation space.
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import scanpy as sc
from scipy import ndimage
from scipy.spatial import cKDTree


# =============================================================================
# Basic I/O
# =============================================================================

def cv_imread(path: str, flags=cv2.IMREAD_COLOR):
    """Read images from paths that may contain non-ASCII characters."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found:{path}")
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, flags)
    if img is None:
        raise RuntimeError(f"Image decoding failed:{path}")
    return img


def cv_imwrite(path: str, img: np.ndarray):
    """Write images to paths that may contain non-ASCII characters."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Image encoding failed:{path}")
    buf.tofile(path)


def resolve_tissue_mask(
    slice_dir: str,
    target_shape: Tuple[int, int],
) -> Tuple[np.ndarray, str]:
    """
    Read the fixed tissue mask used by the project.

    Expected file:
        {slice_dir}/mask_resized.png

    Convention:
        pixel > 127 : tissue
        pixel <= 127: background

    The tissue mask must already be aligned with label_resized.png.
    No automatic resizing is performed during evaluation.
    """
    path = os.path.join(slice_dir, "mask_resized.png")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tissue mask not found: {path}\n"
            "Each section directory must contain mask_resized.png."
        )

    mask = cv_imread(path, cv2.IMREAD_GRAYSCALE)

    if mask.shape != target_shape:
        raise ValueError(
            'Tissue area mask is inconsistent with label dimensions:'
            f"  mask_resized.png: {mask.shape}\n"
            f"  label_resized.png: {target_shape}\n"
            'The evaluation phase does not scale automatically, please check the spatial alignment after preprocessing.'
        )

    tissue_mask = mask > 127
    if not np.any(tissue_mask):
        raise ValueError(
            f"Tissue mask is empty: {path}\n"
            "Confirm that tissue pixels in mask_resized.png are white or high-valued."
        )

    return tissue_mask, path


# =============================================================================
# Metrics
# =============================================================================

def compute_hd95(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """
    Symmetric 95th-percentile Hausdorff distance in pixels.

    The calculation is restricted to the common valid/tissue evaluation mask.
    """
    p = pred.astype(bool) & valid
    g = gt.astype(bool) & valid

    if not p.any() or not g.any():
        return float("nan")

    structure = ndimage.generate_binary_structure(2, 1)
    p_surface = p ^ ndimage.binary_erosion(p, structure=structure, border_value=0)
    g_surface = g ^ ndimage.binary_erosion(g, structure=structure, border_value=0)

    if not p_surface.any() or not g_surface.any():
        return float("nan")

    dist_to_g = ndimage.distance_transform_edt(~g_surface)
    dist_to_p = ndimage.distance_transform_edt(~p_surface)

    distances = np.concatenate(
        [
            dist_to_g[p_surface],
            dist_to_p[g_surface],
        ]
    )
    return float(np.percentile(distances, 95.0))


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, float]:
    """
    Pixel-level metrics used in the main segmentation table:
    Target IoU, Dice, Recall, Precision, and HD95.
    """
    eps = 1e-7
    valid = valid.astype(bool)

    p = (pred == 1) & valid
    g = (gt == 1) & valid

    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g & valid))
    fn = int(np.sum(~p & g & valid))
    tn = int(np.sum(~p & ~g & valid))

    target_iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    hd95 = compute_hd95(p, g, valid)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "Target_IoU": float(target_iou),
        "Dice": float(dice),
        "Recall": float(recall),
        "Precision": float(precision),
        "HD95": float(hd95),
    }


# =============================================================================
# Spot prediction loading
# =============================================================================

def load_spot_outputs(
    pred_path: str,
    model_name: str,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Return:
      spot_scores       : values in [0, 1]
      spot_predictions  : binary labels
      has_continuous_scores

    BLEEP normally provides continuous probabilities.
    MCGAE / MEATRD / SpaCRD may provide only binary labels.
    """
    raw = np.asarray(np.loadtxt(pred_path, dtype=float))

    if raw.ndim == 0:
        raw = raw.reshape(1)
    elif raw.ndim == 2:
        # If a two-column probability matrix is provided, use the positive class.
        raw = raw[:, -1]

    raw = raw.reshape(-1)
    if raw.size == 0:
        raise ValueError(f"Prediction file is empty:{pred_path}")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"Prediction file contains NaN or Inf:{pred_path}")

    spot_scores = np.clip(raw, 0.0, 1.0)
    is_binary_input = bool(
        np.all(np.isclose(spot_scores, 0.0) | np.isclose(spot_scores, 1.0))
    )

    # Use one fixed threshold for continuous predictions.
    # For binary inputs, preserve their native labels.
    if is_binary_input:
        spot_predictions = (spot_scores > 0.0).astype(np.uint8)
    else:
        spot_predictions = (spot_scores >= float(threshold)).astype(np.uint8)

    return spot_scores, spot_predictions, (not is_binary_input)


# =============================================================================
# Parameter-free nearest-spot Voronoi rasterization
# =============================================================================

def build_nearest_spot_index_map(
    coordinates: np.ndarray,
    tissue_mask: np.ndarray,
    row_chunk_size: int = 256,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Assign every tissue pixel to its nearest ST spot by Euclidean distance.

    This is a tissue-clipped nearest-neighbor Voronoi rasterization:
      - no spot radius;
      - no morphology;
      - no Gaussian sigma;
      - no ground-truth-dependent parameter.

    The returned map stores the ORIGINAL spot index for every tissue pixel.
    Background pixels are -1.
    """
    coords = np.asarray(coordinates, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coordinates should be [N,2+], current shape={coords.shape}")
    coords = coords[:, :2]

    h, w = tissue_mask.shape

    in_bounds = (
        np.isfinite(coords[:, 0])
        & np.isfinite(coords[:, 1])
        & (coords[:, 0] >= 0)
        & (coords[:, 0] < w)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < h)
    )
    valid_spot_ids = np.flatnonzero(in_bounds)
    if valid_spot_ids.size == 0:
        raise ValueError('There are no valid ST spot coordinates within the image extent.')

    tree = cKDTree(coords[valid_spot_ids])
    nearest_index = np.full((h, w), -1, dtype=np.int32)

    tissue_pixels = int(tissue_mask.sum())
    assigned_pixels = 0

    for y0 in range(0, h, int(row_chunk_size)):
        y1 = min(h, y0 + int(row_chunk_size))
        tissue_chunk = tissue_mask[y0:y1]
        yy, xx = np.nonzero(tissue_chunk)
        if xx.size == 0:
            continue

        query_xy = np.column_stack(
            [
                xx.astype(np.float32),
                (yy + y0).astype(np.float32),
            ]
        )

        try:
            _, local_nn = tree.query(query_xy, k=1, workers=-1)
        except TypeError:
            # Compatibility with older SciPy versions.
            _, local_nn = tree.query(query_xy, k=1)

        global_spot_ids = valid_spot_ids[np.asarray(local_nn, dtype=np.int64)]
        nearest_index[yy + y0, xx] = global_spot_ids.astype(np.int32)
        assigned_pixels += int(xx.size)

    diagnostics = {
        "Num_Spots_Total": int(len(coords)),
        "Num_Spots_InBounds": int(valid_spot_ids.size),
        "Tissue_Pixels": int(tissue_pixels),
        "Assigned_Tissue_Pixels": int(assigned_pixels),
        "Assigned_Fraction": float(assigned_pixels / max(1, tissue_pixels)),
    }
    return nearest_index, diagnostics


def rasterize_binary_prediction(
    nearest_spot_index: np.ndarray,
    spot_predictions: np.ndarray,
) -> np.ndarray:
    """
    Convert spot-level binary predictions to a dense pixel mask using the
    precomputed nearest-spot Voronoi assignment.
    """
    pred = np.zeros(nearest_spot_index.shape, dtype=np.uint8)
    valid = nearest_spot_index >= 0
    pred[valid] = spot_predictions[nearest_spot_index[valid]]
    return pred


def apply_cancer_mask_overlay(
    img_bgr: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.45,
    color=(0, 0, 255),
) -> np.ndarray:
    overlay = img_bgr.copy().astype(np.float32)
    mask = pred_mask.astype(bool)
    color_arr = np.asarray(color, dtype=np.float32)
    overlay[mask] = (
        overlay[mask] * (1.0 - alpha)
        + color_arr * alpha
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


# =============================================================================
# Statistics and output
# =============================================================================

METRIC_KEYS = ["Target_IoU", "Dice", "Recall", "Precision", "HD95"]


def sample_std(values: List[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def metric_mean_std(seed_metrics: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    values = np.asarray(
        [float(m[key]) for m in seed_metrics if np.isfinite(float(m[key]))],
        dtype=np.float64,
    )
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), sample_std(values.tolist())


def fmt(mean: float, std: float, hd95: bool = False) -> str:
    if not np.isfinite(mean):
        return "NaN"
    if hd95:
        return f"{mean:.2f}±{std:.2f}"
    return f"{mean:.4f}±{std:.4f}"


def write_csv(path: str, rows: List[Dict[str, object]]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate_all_baselines(test_slices: List[str]):
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    PRED_ROOT_DIR = r"outputs"
    RESIZED_BASE_DIR = r"data/STHBC/resized_data"
    OUTPUT_ROOT_DIR = (
        r"data/STHBC"
        r"\predictions_Baseline_Spot_Voronoi"
    )

    BASELINES = ["BLEEP", "MCGAE", "MEATRD", "SpaCRD"]
    SEEDS = [42, 123, 567]
    CONTINUOUS_THRESHOLD = 0.5
    VORONOI_ROW_CHUNK = 256

    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    # metrics[method][slice][seed] = metric dictionary
    metrics_all: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {
        method: {} for method in BASELINES
    }
    per_seed_rows: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []

    print("\n" + "=" * 100)
    print("STHBC spot-level baseline -> pixel-level evaluation")
    print("Rasterization: tissue-clipped nearest-spot Voronoi")
    print("Post-processing: NONE")
    print("Metrics: Target IoU, Dice, Recall, Precision, HD95")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Slice-first loop:
    # the expensive Voronoi assignment is built only once per slice.
    # -------------------------------------------------------------------------
    for slice_name in test_slices:
        print(f"\n{'=' * 90}")
        print(f"Slice: {slice_name}")
        print(f"{'=' * 90}")

        slice_dir = os.path.join(RESIZED_BASE_DIR, slice_name)
        label_path = os.path.join(slice_dir, "label_resized.png")
        tissue_path_expected = os.path.join(slice_dir, "mask_resized.png")
        adata_path = os.path.join(slice_dir, "adata_resized.h5ad")
        he_path = os.path.join(slice_dir, "he_resized.png")

        required = [label_path, tissue_path_expected, adata_path, he_path]
        if not all(os.path.exists(p) for p in required):
            print(f"[Skip] Basic files are missing:{slice_name}")
            for p in required:
                if not os.path.exists(p):
                    print(f"  missing: {p}")
            continue

        true_mask = (
            cv_imread(label_path, cv2.IMREAD_GRAYSCALE) > 127
        ).astype(np.uint8)
        h, w = true_mask.shape

        tissue_mask, tissue_path = resolve_tissue_mask(
            slice_dir,
            target_shape=(h, w),
        )

        he_img = cv_imread(he_path, cv2.IMREAD_COLOR)
        if he_img.shape[:2] != (h, w):
            he_img = cv2.resize(
                he_img,
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )

        adata = sc.read_h5ad(adata_path)
        if "spatial_resized" not in adata.obsm:
            raise KeyError(
                f"{slice_name}: 'spatial_resized' does not exist in adata.obsm;"
                f"Available keys:{list(adata.obsm.keys())}"
            )
        coords = np.asarray(
            adata.obsm["spatial_resized"],
            dtype=np.float32,
        )[:, :2]

        # Build the parameter-free pixel-to-spot assignment once.
        nearest_index, geometry_info = build_nearest_spot_index_map(
            coordinates=coords,
            tissue_mask=tissue_mask,
            row_chunk_size=VORONOI_ROW_CHUNK,
        )

        geometry_row = {
            "Slice": slice_name,
            "Tissue_Mask_Path": tissue_path,
            **geometry_info,
        }
        geometry_rows.append(geometry_row)

        print(
            f"  spots={geometry_info['Num_Spots_Total']} | "
            f"in-bounds={geometry_info['Num_Spots_InBounds']} | "
            f"tissue pixels={geometry_info['Tissue_Pixels']} | "
            f"assigned={geometry_info['Assigned_Fraction']:.4f}"
        )

        for model_name in BASELINES:
            metrics_all[model_name].setdefault(slice_name, {})
            pred_base_dir = os.path.join(
                PRED_ROOT_DIR,
                model_name,
                "STHBC",
            )
            model_output_dir = os.path.join(
                OUTPUT_ROOT_DIR,
                model_name,
                slice_name,
            )

            print(f"\n  [{model_name}]")

            for seed in SEEDS:
                if model_name == "BLEEP":
                    pred_path = os.path.join(
                        pred_base_dir,
                        f"probs_{slice_name}_seed={seed}.txt",
                    )
                else:
                    pred_path = os.path.join(
                        pred_base_dir,
                        f"{slice_name}_pred_seed={seed}.txt",
                    )

                if not os.path.exists(pred_path):
                    print(f"    seed={seed}: missing -> {pred_path}")
                    continue

                try:
                    spot_scores, spot_predictions, has_continuous = load_spot_outputs(
                        pred_path=pred_path,
                        model_name=model_name,
                        threshold=CONTINUOUS_THRESHOLD,
                    )
                except Exception as exc:
                    print(f"    seed={seed}: Failed to read:{exc}")
                    continue

                if len(spot_predictions) != len(coords):
                    print(
                        f"    seed={seed}: number of coordinates({len(coords)}) and predicted numbers"
                        f"({len(spot_predictions)}) is inconsistent and skipped"
                    )
                    continue

                # -----------------------------------------------------------------
                # Main protocol:
                # each tissue pixel directly inherits the nearest spot prediction.
                # NO circle drawing, NO dilation/closing, NO Gaussian smoothing.
                # -----------------------------------------------------------------
                pred_mask = rasterize_binary_prediction(
                    nearest_index,
                    spot_predictions,
                )

                result = compute_metrics(
                    pred=pred_mask,
                    gt=true_mask,
                    valid=tissue_mask,
                )

                metrics_all[model_name][slice_name][seed] = result

                row = {
                    "Method": model_name,
                    "Slice": slice_name,
                    "Seed": seed,
                    "Rasterization": "Nearest-spot Voronoi",
                    "Continuous_Spot_Scores": bool(has_continuous),
                    **result,
                }
                per_seed_rows.append(row)

                seed_dir = os.path.join(
                    model_output_dir,
                    f"seed_{seed}",
                )
                os.makedirs(seed_dir, exist_ok=True)

                mask_255 = (pred_mask * 255).astype(np.uint8)
                cv_imwrite(
                    os.path.join(
                        seed_dir,
                        f"{slice_name}_{model_name}_voronoi_mask.png",
                    ),
                    mask_255,
                )

                overlay = apply_cancer_mask_overlay(
                    he_img,
                    pred_mask,
                    alpha=0.45,
                )
                cv_imwrite(
                    os.path.join(
                        seed_dir,
                        f"{slice_name}_{model_name}_voronoi_overlay.jpg",
                    ),
                    overlay,
                )

                hd_text = (
                    f"{result['HD95']:.2f}"
                    if np.isfinite(result["HD95"])
                    else "NaN"
                )
                print(
                    f"    seed={seed} | "
                    f"Target_IoU={result['Target_IoU']:.4f} | "
                    f"Dice={result['Dice']:.4f} | "
                    f"Recall={result['Recall']:.4f} | "
                    f"Precision={result['Precision']:.4f} | "
                    f"HD95={hd_text}"
                )

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------
    per_slice_rows: List[Dict[str, object]] = []
    model_summary_rows: List[Dict[str, object]] = []

    for model_name in BASELINES:
        print(f"\n\n{'=' * 120}")
        print(f"{model_name} | slice-level mean ± sample SD over seeds")
        print(f"{'=' * 120}")

        slice_mean_values = {key: [] for key in METRIC_KEYS}

        for slice_name in test_slices:
            seed_dict = metrics_all.get(model_name, {}).get(slice_name, {})
            if not seed_dict:
                print(f"{slice_name:<8}: N/A")
                continue

            seed_metrics = list(seed_dict.values())
            row: Dict[str, object] = {
                "Method": model_name,
                "Slice": slice_name,
                "Num_Seeds": len(seed_metrics),
            }

            display_parts = []
            for key in METRIC_KEYS:
                mean, std = metric_mean_std(seed_metrics, key)
                row[f"{key}_Mean"] = mean
                row[f"{key}_Std"] = std
                if np.isfinite(mean):
                    slice_mean_values[key].append(mean)
                display_parts.append(
                    f"{key}={fmt(mean, std, hd95=(key == 'HD95'))}"
                )

            per_slice_rows.append(row)
            print(f"{slice_name:<8}: " + " | ".join(display_parts))

        summary: Dict[str, object] = {"Method": model_name}
        for key in METRIC_KEYS:
            vals = np.asarray(slice_mean_values[key], dtype=np.float64)
            if vals.size == 0:
                mean = float("nan")
                std = float("nan")
            else:
                mean = float(np.mean(vals))
                std = sample_std(vals.tolist())

            summary[f"Macro_{key}_Mean"] = mean
            summary[f"Macro_{key}_Std_Across_Slices"] = std

        model_summary_rows.append(summary)

        print("\nMacro average across held-out slices:")
        print(
            "  "
            + " | ".join(
                [
                    f"{key}="
                    f"{fmt(summary[f'Macro_{key}_Mean'], summary[f'Macro_{key}_Std_Across_Slices'], hd95=(key == 'HD95'))}"
                    for key in METRIC_KEYS
                ]
            )
        )

    # -------------------------------------------------------------------------
    # Save reproducible tables
    # -------------------------------------------------------------------------
    write_csv(
        os.path.join(OUTPUT_ROOT_DIR, "pixel_metrics_per_seed.csv"),
        per_seed_rows,
    )
    write_csv(
        os.path.join(OUTPUT_ROOT_DIR, "pixel_metrics_per_slice_mean_std.csv"),
        per_slice_rows,
    )
    write_csv(
        os.path.join(OUTPUT_ROOT_DIR, "pixel_metrics_model_macro_summary.csv"),
        model_summary_rows,
    )
    write_csv(
        os.path.join(OUTPUT_ROOT_DIR, "voronoi_geometry_diagnostics.csv"),
        geometry_rows,
    )

    protocol = {
        "task": "Pixel-level CTR segmentation comparison",
        "spot_prediction_space": (
            "Method-native original-scale H&E/ST data; one value per original spot"
        ),
        "pixel_evaluation_space": (
            "STHBC resized_data with adata.obsm['spatial_resized']"
        ),
        "spot_to_pixel_conversion": "Tissue-clipped nearest-spot Voronoi rasterization",
        "postprocessing": "None",
        "continuous_threshold": CONTINUOUS_THRESHOLD,
        "metrics": METRIC_KEYS,
        "seed_std": "Sample standard deviation (ddof=1)",
        "important_note": (
            "Use the same tissue evaluation mask and the same HD95 implementation "
            "for HiST-Seg and all baselines."
        ),
    }
    with open(
        os.path.join(OUTPUT_ROOT_DIR, "evaluation_protocol.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:\n{OUTPUT_ROOT_DIR}")


if __name__ == "__main__":
    TEST_SLICES = [
        "A1", "B1", "C1", "D1",
        "E1", "F1", "G2", "H1",
    ]
    evaluate_all_baselines(TEST_SLICES)

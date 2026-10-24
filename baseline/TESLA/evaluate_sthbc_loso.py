# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
'Public baseline experiment module. Configure file-system paths as described in readme_baseline.txt.'

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from medpy.metric.binary import hd95 as _medpy_hd95
except ImportError:
    _medpy_hd95 = None

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
except ImportError as exc:
    if _medpy_hd95 is None:
        raise ImportError(
            'Calculating HD95 requires at least medpy or scipy. Please install one of these:'
            'pip install medpy or pip install scipy'
        ) from exc


# =============================================================================
# Configure this path for your environment; see readme_baseline.txt.
# =============================================================================
SLICES: List[str] = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]
RANDOM_SEEDS: List[int] = [42, 123, 567]

DATA_ROOT = Path(r"data/STHBC/resized_data")
TESLA_WORKSPACE = Path(r"workspaces/TESLA/tesla_workspace_multiseed")

PRED_MASK_ROOT = TESLA_WORKSPACE / "pred_masks"
OUTPUT_DIR = TESLA_WORKSPACE / "metrics_5metrics_from_saved_masks"


PRED_FILENAME = "{slice_name}_seed{seed}_tesla_pred_mask.png"


BINARY_THRESHOLD = 127



AUTO_RESIZE_IF_NEEDED = True


EMPTY_HD95_VALUE = np.nan


STD_DDOF = 0


# =============================================================================

# =============================================================================
def read_gray(path: Path, name: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"{name}Read failed or file does not exist:{path}")
    return image


def to_binary(image: np.ndarray, threshold: int = BINARY_THRESHOLD) -> np.ndarray:
    'Implementation of to_binary for this baseline experiment.'
    return (image > threshold).astype(np.uint8)


def resize_binary(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    'Implementation of resize_binary for this baseline experiment.'
    target_h, target_w = target_hw
    if mask.shape[:2] == (target_h, target_w):
        return (mask > 0).astype(np.uint8)

    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8)


def crop_union_for_hd95(pred: np.ndarray, gt: np.ndarray, padding: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    'Implementation of crop_union_for_hd95 for this baseline experiment.'
    union = (pred > 0) | (gt > 0)
    ys, xs = np.where(union)
    if ys.size == 0:
        return pred, gt

    h, w = union.shape
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(h, int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(w, int(xs.max()) + padding + 1)
    return pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]


# =============================================================================

# =============================================================================
def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    'Implementation of compute_hd95 for this baseline experiment.'
    pred_bool = np.asarray(pred, dtype=bool)
    gt_bool = np.asarray(gt, dtype=bool)

    if not np.any(pred_bool) or not np.any(gt_bool):
        return float(EMPTY_HD95_VALUE)

    if _medpy_hd95 is not None:
        return float(_medpy_hd95(pred_bool, gt_bool))

    footprint = generate_binary_structure(pred_bool.ndim, 1)
    pred_border = pred_bool ^ binary_erosion(
        pred_bool, structure=footprint, iterations=1, border_value=0
    )
    gt_border = gt_bool ^ binary_erosion(
        gt_bool, structure=footprint, iterations=1, border_value=0
    )

    
    dt_to_gt = distance_transform_edt(~gt_border)
    dt_to_pred = distance_transform_edt(~pred_border)
    distances = np.concatenate([dt_to_gt[pred_border], dt_to_pred[gt_border]])

    if distances.size == 0:
        return float(EMPTY_HD95_VALUE)
    return float(np.percentile(distances, 95))


def calculate_five_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tissue_mask: np.ndarray,
) -> Dict[str, float]:
    'Implementation of calculate_five_metrics for this baseline experiment.'
    if pred_mask.shape != gt_mask.shape or tissue_mask.shape != gt_mask.shape:
        raise ValueError(
            f"The input sizes must be consistent, pred={pred_mask.shape}, gt={gt_mask.shape}, "
            f"tissue={tissue_mask.shape}"
        )

    valid = tissue_mask > 0
    pred = (pred_mask > 0) & valid
    gt = (gt_mask > 0) & valid

    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(np.count_nonzero(~pred & ~gt & valid))

    eps = 1e-8
    target_iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)

    pred_count = int(np.count_nonzero(pred))
    gt_count = int(np.count_nonzero(gt))

    if pred_count == 0 or gt_count == 0:
        hd95_value = float(EMPTY_HD95_VALUE)
    else:
        try:
            pred_crop, gt_crop = crop_union_for_hd95(
                pred.astype(np.uint8), gt.astype(np.uint8), padding=1
            )
            hd95_value = compute_hd95(pred_crop, gt_crop)
        except Exception as exc:
            print(f"[Warning] HD95 calculation failed and will be recorded as NaN:{exc}")
            hd95_value = float("nan")

    return {
        "Target_IoU": float(target_iou),
        "Dice": float(dice),
        "Recall": float(recall),
        "Precision": float(precision),
        "HD95": hd95_value,
        
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Pred_Positive_Pixels": pred_count,
        "GT_Positive_Pixels": gt_count,
        "Tissue_Pixels": int(np.count_nonzero(valid)),
    }


# =============================================================================

# =============================================================================
METRIC_COLUMNS = ["Target_IoU", "Dice", "Recall", "Precision", "HD95"]


def safe_nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def safe_nanstd(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanstd(arr, ddof=STD_DDOF))


def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    'Implementation of build_summary for this baseline experiment.'
    rows: List[Dict[str, float]] = []

    for slice_name in SLICES:
        part = raw_df[raw_df["Slice"] == slice_name]
        if part.empty:
            continue

        row: Dict[str, float] = {
            "Slice": slice_name,
            "Num_Seeds": int(part["Seed"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            values = part[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = safe_nanmean(values)
            row[f"{metric}_std"] = safe_nanstd(values)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    avg_row: Dict[str, float] = {
        "Slice": "Average",
        "Num_Seeds": int(raw_df["Seed"].nunique()),
    }
    for metric in METRIC_COLUMNS:
        slice_means = summary_df[f"{metric}_mean"].astype(float).to_numpy()
        avg_row[f"{metric}_mean"] = safe_nanmean(slice_means)
        avg_row[f"{metric}_std"] = safe_nanstd(slice_means)

    return pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)


def format_mean_std(mean_value: float, std_value: float, digits: int) -> str:
    if pd.isna(mean_value):
        return "NaN"
    if pd.isna(std_value):
        return f"{mean_value:.{digits}f}±NaN"
    return f"{mean_value:.{digits}f}±{std_value:.{digits}f}"


def make_display_dataframe(summary_df: pd.DataFrame) -> pd.DataFrame:
    display_rows = []
    for _, row in summary_df.iterrows():
        display_rows.append(
            {
                "Slice": row["Slice"],
                "Seeds": int(row["Num_Seeds"]),
                "Target_IoU": format_mean_std(
                    row["Target_IoU_mean"], row["Target_IoU_std"], 4
                ),
                "Dice": format_mean_std(row["Dice_mean"], row["Dice_std"], 4),
                "Recall": format_mean_std(row["Recall_mean"], row["Recall_std"], 4),
                "Precision": format_mean_std(
                    row["Precision_mean"], row["Precision_std"], 4
                ),
                "HD95 (px)": format_mean_std(row["HD95_mean"], row["HD95_std"], 2),
            }
        )
    return pd.DataFrame(display_rows)


# =============================================================================

# =============================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, float]] = []
    missing_files: List[Dict[str, object]] = []

    print("=" * 110)
    print('TESLA Independent 5-metric evaluation of saved prediction masks')
    print('Imputation, annotation, or any training/inference process will not be re-run.')
    print(f"Forecast Catalog:{PRED_MASK_ROOT}")
    print(f"Result directory:{OUTPUT_DIR}")
    print("=" * 110)

    for seed in RANDOM_SEEDS:
        for slice_name in SLICES:
            pred_path = (
                PRED_MASK_ROOT
                / f"seed_{seed}"
                / PRED_FILENAME.format(slice_name=slice_name, seed=seed)
            )
            gt_path = DATA_ROOT / slice_name / "label_resized.png"
            tissue_path = DATA_ROOT / slice_name / "mask_resized.png"

            print(f"\n[{slice_name} | seed={seed}]")

            required = {
                "Prediction": pred_path,
                "Ground truth": gt_path,
                "Tissue mask": tissue_path,
            }
            missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
            if missing:
                print('[Skip] Missing files:')
                for item in missing:
                    print(f"    {item}")
                missing_files.append(
                    {
                        "Slice": slice_name,
                        "Seed": seed,
                        "Missing": " | ".join(missing),
                    }
                )
                continue

            try:
                pred = to_binary(read_gray(pred_path, "Prediction"))
                gt = to_binary(read_gray(gt_path, "Ground truth"))
                tissue = to_binary(read_gray(tissue_path, "Tissue mask"))

                if pred.shape != gt.shape:
                    if not AUTO_RESIZE_IF_NEEDED:
                        raise ValueError(
                            f"Predicted size{pred.shape}with GT size{gt.shape}inconsistent"
                        )
                    print(f"  [Resize] prediction: {pred.shape} -> {gt.shape}")
                    pred = resize_binary(pred, gt.shape)

                if tissue.shape != gt.shape:
                    if not AUTO_RESIZE_IF_NEEDED:
                        raise ValueError(
                            f"Organization mask size{tissue.shape}with GT size{gt.shape}inconsistent"
                        )
                    print(f"  [Resize] tissue mask: {tissue.shape} -> {gt.shape}")
                    tissue = resize_binary(tissue, gt.shape)

                metrics = calculate_five_metrics(pred, gt, tissue)
            except Exception as exc:
                print(f"[Skip] Evaluation failed:{exc}")
                missing_files.append(
                    {
                        "Slice": slice_name,
                        "Seed": seed,
                        "Missing": f"Evaluation error: {exc}",
                    }
                )
                continue

            record = {
                "Slice": slice_name,
                "Seed": seed,
                "Prediction_Path": str(pred_path),
                **metrics,
            }
            records.append(record)

            hd_text = "NaN" if np.isnan(metrics["HD95"]) else f"{metrics['HD95']:.2f}"
            print(
                f"  Target_IoU={metrics['Target_IoU']:.4f} | "
                f"Dice={metrics['Dice']:.4f} | "
                f"Recall={metrics['Recall']:.4f} | "
                f"Precision={metrics['Precision']:.4f} | "
                f"HD95={hd_text}px"
            )

    if not records:
        print('[Error] No evaluable predictions found, please check path and filename.')
        if missing_files:
            pd.DataFrame(missing_files).to_csv(
                OUTPUT_DIR / "tesla_5metrics_missing_or_failed.csv",
                index=False,
                encoding="utf-8-sig",
            )
        return

    raw_df = pd.DataFrame(records)
    raw_df = raw_df.sort_values(["Slice", "Seed"]).reset_index(drop=True)
    summary_df = build_summary(raw_df)
    display_df = make_display_dataframe(summary_df)

    raw_path = OUTPUT_DIR / "tesla_5metrics_raw.csv"
    summary_path = OUTPUT_DIR / "tesla_5metrics_summary_numeric.csv"
    formatted_path = OUTPUT_DIR / "tesla_5metrics_summary_mean_std.csv"
    table_path = OUTPUT_DIR / "tesla_5metrics_summary_table.txt"
    missing_path = OUTPUT_DIR / "tesla_5metrics_missing_or_failed.csv"

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    display_df.to_csv(formatted_path, index=False, encoding="utf-8-sig")

    table_text = display_df.to_string(index=False)
    with open(table_path, "w", encoding="utf-8") as file:
        file.write(table_text + "\n")

    if missing_files:
        pd.DataFrame(missing_files).to_csv(missing_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print('TESLA - 5 indicator slice-level statistical summary (mean ± population standard deviation)')
    print("=" * 110)
    print(table_text)
    print("=" * 110)

    print('Results saved:')
    print(f"Slice by slice and seed value ->{raw_path}")
    print(f"Numeric mean/std summary ->{summary_path}")
    print(f"mean±std display table ->{formatted_path}")
    print(f"Text summary table ->{table_path}")
    if missing_files:
        print(f"Missing/failed records ->{missing_path}")

    print('Note: HD95 records NaN when either the predicted or true label is empty; the other four terms use a smooth denominator consistent with the original code.')


if __name__ == "__main__":
    main()

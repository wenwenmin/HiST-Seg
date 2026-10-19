"""
仅使用 TESLA 已保存的最终二值预测掩码，重新计算 5 个像素级指标：
    Target_IoU, Dice, Recall, Precision, HD95(px)

本脚本不会运行 TESLA imputation / annotation，也不会重新训练或推理。

默认读取：
D:\\Project_2\\TESLA\\tesla_workspace_multiseed\\pred_masks\\seed_{seed}\\{slice}_seed{seed}_tesla_pred_mask.png

标签与组织掩码：
D:\\Project_2\\Pixel_Seg\\data\\STHBC\\resized_data\\{slice}\\label_resized.png
D:\\Project_2\\Pixel_Seg\\data\\STHBC\\resized_data\\{slice}\\mask_resized.png
"""

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
            "计算 HD95 至少需要 medpy 或 scipy。请安装其中一个："
            "pip install medpy 或 pip install scipy"
        ) from exc


# =============================================================================
# 配置：通常只需要检查下面几项路径
# =============================================================================
SLICES: List[str] = [
    "SN048_1", "SN048_2", "SN048_3", "SN048_4",
    "SN84_1", "SN84_2", "SN123_1", "SN123_2",
    "SN123_4", "SN124_1", "SN124_2", "SN124_4",
]
RANDOM_SEEDS: List[int] = [42, 123, 567]

LABEL_DIR = Path(r"D:\Project_2\Pixel_Seg\data\CRC\cropped_data\label_cropped")
TISSUE_DIR = Path(r"D:\Project_2\Pixel_Seg\data\CRC\cropped_data\tissue_mask_cropped")
TESLA_WORKSPACE = Path(r"D:\Project_2\TESLA\tesla_workspace_crc_multiseed")

PRED_MASK_ROOT = TESLA_WORKSPACE / "pred_masks"
OUTPUT_DIR = TESLA_WORKSPACE / "metrics_5metrics_from_saved_masks"

# 原脚本保存的预测文件名格式。
PRED_FILENAME = "{slice_name}_seed{seed}_tesla_crc_pred_mask.png"

# 二值化阈值。源代码保存为 0/255，因此使用 >127。
BINARY_THRESHOLD = 127

# True：预测图尺寸与标签不一致时，用最近邻插值自动对齐。
# False：尺寸不一致时直接报错并跳过该结果。
AUTO_RESIZE_IF_NEEDED = True

# HD95 在预测或 GT 为空时没有正常定义，保持为 NaN，避免人为设置巨大惩罚值。
EMPTY_HD95_VALUE = np.nan

# mean±std 中使用总体标准差，与原脚本保持一致。
STD_DDOF = 0


# =============================================================================
# 基础工具
# =============================================================================
def read_gray(path: Path, name: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"{name}读取失败或文件不存在: {path}")
    return image


def to_binary(image: np.ndarray, threshold: int = BINARY_THRESHOLD) -> np.ndarray:
    """将 0/255 或灰度图安全转换为 uint8 的 0/1 掩码。"""
    return (image > threshold).astype(np.uint8)


def resize_binary(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """使用最近邻插值将二值掩码对齐到目标 H,W。"""
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
    """
    将 HD95 的计算区域裁剪到 pred/gt 前景并集的包围框。

    这不会改变前景表面间的像素距离，同时可显著减少大切片上的内存占用。
    """
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
# 指标计算
# =============================================================================
def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """优先使用 MedPy；未安装 MedPy 时使用与其定义一致的 SciPy 实现。"""
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

    # 到对方表面的欧氏距离，与 MedPy binary.hd95 的双向表面距离定义一致。
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
    """
    仅在 tissue_mask > 0 的组织区域内计算：
      - Target_IoU = TP / (TP + FP + FN)
      - Dice       = 2TP / (2TP + FP + FN)
      - Recall     = TP / (TP + FN)
      - Precision  = TP / (TP + FP)
      - HD95       = 双向表面距离的第 95 百分位数，单位为像素
    """
    if pred_mask.shape != gt_mask.shape or tissue_mask.shape != gt_mask.shape:
        raise ValueError(
            f"输入尺寸必须一致，pred={pred_mask.shape}, gt={gt_mask.shape}, "
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
            print(f"    [Warning] HD95 计算失败，将记录为 NaN: {exc}")
            hd95_value = float("nan")

    return {
        "Target_IoU": float(target_iou),
        "Dice": float(dice),
        "Recall": float(recall),
        "Precision": float(precision),
        "HD95": hd95_value,
        # 额外保存混淆矩阵计数，便于后续核查；汇总表不展示这些列。
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Pred_Positive_Pixels": pred_count,
        "GT_Positive_Pixels": gt_count,
        "Tissue_Pixels": int(np.count_nonzero(valid)),
    }


# =============================================================================
# 汇总与格式化
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
    """
    每个切片：对可用随机种子的指标计算 mean±std。

    Average 行：
      mean = 各切片 mean 的平均；
      std  = 各切片 mean 之间的总体标准差。

    该口径与用户原脚本保持一致。
    """
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
# 主流程：只读取保存的预测 mask
# =============================================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, float]] = []
    missing_files: List[Dict[str, object]] = []

    print("=" * 110)
    print("TESLA CRC 已保存预测掩码的独立 5 指标评估")
    print("不会重新运行 imputation、annotation 或任何训练/推理过程。")
    print(f"预测目录: {PRED_MASK_ROOT}")
    print(f"结果目录: {OUTPUT_DIR}")
    print("=" * 110)

    for seed in RANDOM_SEEDS:
        for slice_name in SLICES:
            pred_path = (
                PRED_MASK_ROOT
                / f"seed_{seed}"
                / PRED_FILENAME.format(slice_name=slice_name, seed=seed)
            )
            gt_path = LABEL_DIR / f"{slice_name}.png"
            tissue_path = TISSUE_DIR / f"{slice_name}.png"

            print(f"\n[{slice_name} | seed={seed}]")

            required = {
                "Prediction": pred_path,
                "Ground truth": gt_path,
                "Tissue mask": tissue_path,
            }
            missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
            if missing:
                print("  [Skip] 缺少文件：")
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
                            f"预测尺寸 {pred.shape} 与 GT 尺寸 {gt.shape} 不一致"
                        )
                    print(f"  [Resize] prediction: {pred.shape} -> {gt.shape}")
                    pred = resize_binary(pred, gt.shape)

                if tissue.shape != gt.shape:
                    if not AUTO_RESIZE_IF_NEEDED:
                        raise ValueError(
                            f"组织掩码尺寸 {tissue.shape} 与 GT 尺寸 {gt.shape} 不一致"
                        )
                    print(f"  [Resize] tissue mask: {tissue.shape} -> {gt.shape}")
                    tissue = resize_binary(tissue, gt.shape)

                metrics = calculate_five_metrics(pred, gt, tissue)
            except Exception as exc:
                print(f"  [Skip] 评估失败: {exc}")
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
        print("\n[Error] 没有找到任何可评估的预测结果，请检查路径和文件名。")
        if missing_files:
            pd.DataFrame(missing_files).to_csv(
                OUTPUT_DIR / "tesla_crc_5metrics_missing_or_failed.csv",
                index=False,
                encoding="utf-8-sig",
            )
        return

    raw_df = pd.DataFrame(records)
    raw_df = raw_df.sort_values(["Slice", "Seed"]).reset_index(drop=True)
    summary_df = build_summary(raw_df)
    display_df = make_display_dataframe(summary_df)

    raw_path = OUTPUT_DIR / "tesla_crc_5metrics_raw.csv"
    summary_path = OUTPUT_DIR / "tesla_crc_5metrics_summary_numeric.csv"
    formatted_path = OUTPUT_DIR / "tesla_crc_5metrics_summary_mean_std.csv"
    table_path = OUTPUT_DIR / "tesla_crc_5metrics_summary_table.txt"
    missing_path = OUTPUT_DIR / "tesla_crc_5metrics_missing_or_failed.csv"

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    display_df.to_csv(formatted_path, index=False, encoding="utf-8-sig")

    table_text = display_df.to_string(index=False)
    with open(table_path, "w", encoding="utf-8") as file:
        file.write(table_text + "\n")

    if missing_files:
        pd.DataFrame(missing_files).to_csv(missing_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("TESLA CRC - 5 指标切片级统计汇总（均值±总体标准差）")
    print("=" * 110)
    print(table_text)
    print("=" * 110)

    print("\n结果已保存：")
    print(f"  逐切片逐种子数值      -> {raw_path}")
    print(f"  数值型 mean/std 汇总  -> {summary_path}")
    print(f"  mean±std 展示表       -> {formatted_path}")
    print(f"  文本汇总表            -> {table_path}")
    if missing_files:
        print(f"  缺失/失败记录         -> {missing_path}")

    print("\n说明：HD95 在预测或真实标签任一为空时记为 NaN；其他四项采用与原代码一致的平滑分母。")


if __name__ == "__main__":
    main()

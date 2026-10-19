from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


SEEDS = [42, 123, 567]
THRESHOLD = 0.5


def load_vector(path: Path, dtype: type) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    values = np.loadtxt(path, dtype=dtype)
    values = np.asarray(values).reshape(-1)

    if values.size == 0:
        raise ValueError(f"文件为空：{path}")

    return values


def validate_probability_and_prediction(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    probability_path: Path,
    prediction_path: Path,
) -> None:
    if len(probabilities) != len(predictions):
        raise ValueError(
            "概率和预测数量不一致："
            f"{len(probabilities)} != {len(predictions)}\n"
            f"概率文件：{probability_path}\n"
            f"预测文件：{prediction_path}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(f"概率文件包含 NaN 或 Inf：{probability_path}")

    if np.any(probabilities < -1e-8) or np.any(probabilities > 1 + 1e-8):
        raise ValueError(
            f"概率不在 [0, 1] 范围内：{probability_path}，"
            f"min={probabilities.min()}, max={probabilities.max()}"
        )

    unique_predictions = set(np.unique(predictions).tolist())
    if not unique_predictions.issubset({0, 1}):
        raise ValueError(
            f"预测文件必须只包含 0/1：{prediction_path}，"
            f"实际值={sorted(unique_predictions)}"
        )


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    include_accuracy: bool,
    single_class_default: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if not (len(y_true) == len(y_prob) == len(y_pred)):
        raise ValueError(
            "标签、概率和预测数量不一致："
            f"{len(y_true)}, {len(y_prob)}, {len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("过滤后没有可用于评估的 spot。")

    unique_labels = set(np.unique(y_true).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"二值标签必须只包含 0/1，实际值={sorted(unique_labels)}"
        )

    if len(unique_labels) > 1:
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            "AUC": float(auc),
            "AP": float(ap),
            "F1": float(f1),
            "Sensitivity": float(sensitivity),
            "Specificity": float(specificity),
        }

        if include_accuracy:
            metrics["Accuracy"] = float(
                accuracy_score(y_true, y_pred)
            )

    else:
        # 与用户提供的原始评估脚本保持一致：
        # 单类别切片使用固定回退值，而不是抛出 AUC 异常。
        metrics = {
            "AUC": single_class_default,
            "AP": single_class_default,
            "F1": single_class_default,
            "Sensitivity": single_class_default,
            "Specificity": single_class_default,
        }

        if include_accuracy:
            metrics["Accuracy"] = single_class_default

    return metrics


def build_and_save_summary(
    detail_df: pd.DataFrame,
    sample_order: list[str],
    metric_columns: list[str],
    output_dir: Path,
    file_prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / f"{file_prefix}_per_seed_metrics.csv"
    summary_numeric_path = output_dir / f"{file_prefix}_summary_numeric.csv"
    summary_formatted_path = output_dir / f"{file_prefix}_summary_mean_std.csv"

    detail_df.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    numeric_rows: list[dict[str, object]] = []
    formatted_rows: list[dict[str, object]] = []

    for sample in sample_order:
        group = detail_df.loc[detail_df["Slice"] == sample]

        if len(group) != len(SEEDS):
            raise RuntimeError(
                f"{sample} 应包含 {len(SEEDS)} 个随机种子，"
                f"实际只有 {len(group)} 个。"
            )

        numeric_row: dict[str, object] = {
            "Slice": sample,
            "N_seeds": len(group),
        }
        formatted_row: dict[str, object] = {
            "Slice": sample,
        }

        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)

            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            variance = float(np.var(values, ddof=0))

            numeric_row[f"{metric}_mean"] = mean
            numeric_row[f"{metric}_std"] = std
            numeric_row[f"{metric}_variance"] = variance

            # 原始评估脚本采用 np.mean ± np.std。
            formatted_row[metric] = f"{mean:.4f} ± {std:.4f}"

        numeric_rows.append(numeric_row)
        formatted_rows.append(formatted_row)

    # Overall 与原始脚本一致：对所有切片 × 所有随机种子的结果展开后统计。
    numeric_overall: dict[str, object] = {
        "Slice": "Overall",
        "N_seeds": len(detail_df),
    }
    formatted_overall: dict[str, object] = {
        "Slice": "Overall",
    }

    for metric in metric_columns:
        values = detail_df[metric].to_numpy(dtype=np.float64)

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        variance = float(np.var(values, ddof=0))

        numeric_overall[f"{metric}_mean"] = mean
        numeric_overall[f"{metric}_std"] = std
        numeric_overall[f"{metric}_variance"] = variance
        formatted_overall[metric] = f"{mean:.4f} ± {std:.4f}"

    numeric_rows.append(numeric_overall)
    formatted_rows.append(formatted_overall)

    numeric_df = pd.DataFrame(numeric_rows)
    formatted_df = pd.DataFrame(formatted_rows)

    numeric_df.to_csv(
        summary_numeric_path,
        index=False,
        encoding="utf-8-sig",
    )
    formatted_df.to_csv(
        summary_formatted_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 120)
    print("汇总表：Mean ± Std（与原始评估脚本一致）")
    print("=" * 120)
    print(formatted_df.to_string(index=False))
    print("=" * 120)
    print(f"逐种子结果：{detail_path}")
    print(f"数值汇总：  {summary_numeric_path}")
    print(f"格式化汇总：{summary_formatted_path}")

SLICES = [
    "SN048_1",
    "SN048_2",
    "SN048_3",
    "SN048_4",
    "SN84_1",
    "SN84_2",
    "SN123_1",
    "SN123_2",
    "SN123_4",
    "SN124_1",
    "SN124_2",
    "SN124_4",
]

PREDICTION_ROOT = Path(
    r"D:\DNM_project(cancer detect)\predict_files\TESLA\CRC_spot_level"
)

# 按 main_CRC.py 中 ../data/CRC/label3 的结构设置。
LABEL_ROOT_CANDIDATES = [
    Path(r"D:\DNM_project(cancer detect)\data\CRC\label3"),
    Path(r"D:\Project_2\Pixel_Seg\data\CRC\label3"),
]

OUTPUT_DIR = PREDICTION_ROOT / "evaluation_metrics"


def resolve_label_root(candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "未找到 CRC 标签目录。检查或修改 LABEL_ROOT_CANDIDATES：\n"
        f"{candidate_text}"
    )


def align_crc_labels_and_predictions(
    raw_labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    sample: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    原始 CRC 评估规则：
      1. 删除标签为 -2 的无效/背景 spot；
      2. 标签 2 视为癌症阳性，其余有效标签视为阴性。

    同时兼容预测文件包含全部 spot 或已过滤 spot 两种情况。
    """
    raw_labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
    valid_mask = raw_labels != -2
    valid_labels = np.where(
        raw_labels[valid_mask] == 2,
        1,
        0,
    ).astype(np.int64)

    if len(probabilities) == len(raw_labels):
        probabilities = probabilities[valid_mask]
        predictions = predictions[valid_mask]
    elif len(probabilities) == int(valid_mask.sum()):
        pass
    else:
        raise ValueError(
            f"{sample} 的标签与预测数量无法对齐："
            f"raw_labels={len(raw_labels)}, "
            f"valid_labels={valid_mask.sum()}, "
            f"predictions={len(probabilities)}"
        )

    return valid_labels, probabilities, predictions


def main() -> None:
    label_root = resolve_label_root(LABEL_ROOT_CANDIDATES)
    rows: list[dict[str, object]] = []

    print("=" * 100)
    print("TESLA CRC spot-level evaluation")
    print(f"标签目录：{label_root}")
    print(f"预测目录：{PREDICTION_ROOT}")
    print(f"随机种子：{SEEDS}")
    print("=" * 100)

    for sample in SLICES:
        label_path = label_root / f"{sample}-labels3.txt"
        raw_labels = load_vector(label_path, int)

        for seed in SEEDS:
            probability_path = (
                PREDICTION_ROOT
                / f"{sample}_probs_seed{seed}.txt"
            )
            prediction_path = (
                PREDICTION_ROOT
                / f"{sample}_pred_seed{seed}.txt"
            )

            probabilities = load_vector(
                probability_path,
                float,
            )
            predictions = load_vector(
                prediction_path,
                int,
            )

            validate_probability_and_prediction(
                probabilities,
                predictions,
                probability_path,
                prediction_path,
            )

            y_true, probabilities, predictions = (
                align_crc_labels_and_predictions(
                    raw_labels,
                    probabilities,
                    predictions,
                    sample,
                )
            )

            metrics = compute_binary_metrics(
                y_true=y_true,
                y_prob=probabilities,
                y_pred=predictions,
                # main_CRC.py 的测试评估没有计算 Accuracy，
                # 因此这里严格保持一致。
                include_accuracy=False,
                single_class_default=0.5,
            )

            row = {
                "Slice": sample,
                "Seed": seed,
                "N_spots": len(y_true),
                "N_positive": int(y_true.sum()),
                **metrics,
            }
            rows.append(row)

            print(
                f"[完成] {sample}, seed={seed}: "
                + ", ".join(
                    f"{name}={value:.4f}"
                    for name, value in metrics.items()
                )
            )

    detail_df = pd.DataFrame(rows)
    metric_columns = [
        "AUC",
        "AP",
        "F1",
        "Sensitivity",
        "Specificity",
    ]

    build_and_save_summary(
        detail_df=detail_df,
        sample_order=SLICES,
        metric_columns=metric_columns,
        output_dir=OUTPUT_DIR,
        file_prefix="TESLA_CRC",
    )


if __name__ == "__main__":
    main()

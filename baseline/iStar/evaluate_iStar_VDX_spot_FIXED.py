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

SLICES = ["ViHBC", "DuCIS", "XeHBC"]

PREDICTION_ROOT = Path(
    r"D:\DNM_project(cancer detect)\predict_files\iStar"
)

# 标签路径严格沿用三个原始跨数据集评估文件中的目录结构。
# 每个列表会自动选取第一个真实存在的路径。
LABEL_PATH_CANDIDATES: dict[str, list[Path]] = {
    "ViHBC": [
        Path(
            r"D:\DNM_project(cancer detect)\data\HB"
            r"\Visium_HBC\spot_labels.txt"
        ),
        Path(
            r"D:\Project_2\Pixel_Seg\data\ViHBC"
            r"\spot_labels.txt"
        ),
        Path(
            r"D:\Project_2\Pixel_Seg\data\ViHBC"
            r"\resized_data\spot_labels.txt"
        ),
    ],
    "DuCIS": [
        Path(
            r"D:\DNM_project(cancer detect)\data\HB"
            r"\Ductal_CIS\spot_labels.txt"
        ),
        Path(
            r"D:\Project_2\Pixel_Seg\data\DuCIS"
            r"\spot_labels.txt"
        ),
        Path(
            r"D:\Project_2\Pixel_Seg\data\DuCIS"
            r"\resized_data\spot_labels.txt"
        ),
    ],
    "XeHBC": [
        Path(
            r"D:\DNM_project(cancer detect)\data\HB"
            r"\Xenium_HBC\spot_labels.txt"
        ),
        Path(
            r"D:\Project_2\Pixel_Seg\data\XeHBC"
            r"\spot_labels.txt"
        ),
        Path(
            r"D:\Project_2\Pixel_Seg\data\XeHBC"
            r"\resized_data\spot_labels.txt"
        ),
    ],
}

OUTPUT_DIR = PREDICTION_ROOT / "three_slices_evaluation_metrics"


def resolve_label_path(
    sample: str,
    candidates: Iterable[Path],
) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"未找到 {sample} 的标签文件。检查或修改 "
        f"LABEL_PATH_CANDIDATES['{sample}']：\n"
        f"{candidate_text}"
    )


def main() -> None:
    rows: list[dict[str, object]] = []

    print("=" * 100)
    print("iStar ViHBC + DuCIS + XeHBC spot-level evaluation")
    print(f"预测根目录：{PREDICTION_ROOT}")
    print(f"随机种子：{SEEDS}")
    print("=" * 100)

    for sample in SLICES:
        label_path = resolve_label_path(
            sample,
            LABEL_PATH_CANDIDATES[sample],
        )
        y_true = load_vector(label_path, int).astype(np.int64)

        unique_labels = set(np.unique(y_true).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"{sample} 的原始跨数据集评估代码直接使用 spot_labels.txt "
                f"作为二值标签，因此文件应只包含 0/1；实际值为 "
                f"{sorted(unique_labels)}。文件：{label_path}"
            )

        print(f"\n{sample} 标签：{label_path}")

        prediction_dir = (
            PREDICTION_ROOT
            / f"{sample}_spot_level"
        )

        for seed in SEEDS:
            probability_path = (
                prediction_dir
                / f"probs_seed{seed}.txt"
            )
            prediction_path = (
                prediction_dir
                / f"pred_seed{seed}.txt"
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

            if not (
                len(y_true)
                == len(probabilities)
                == len(predictions)
            ):
                raise ValueError(
                    f"{sample}, seed={seed} 的标签与预测数量不一致："
                    f"labels={len(y_true)}, "
                    f"probs={len(probabilities)}, "
                    f"preds={len(predictions)}"
                )

            metrics = compute_binary_metrics(
                y_true=y_true,
                y_prob=probabilities,
                y_pred=predictions,
                include_accuracy=True,
                # 三个原始跨数据集脚本在单类别回退时，
                # Sens/Spec/Acc 设为 0；AUC/AP/F1 设为 0.5。
                # 实际数据通常包含两个类别，因此不会触发。
                single_class_default=0.5,
            )

            # 精确复现外部脚本的单类别回退差异。
            if len(np.unique(y_true)) == 1:
                metrics["Sensitivity"] = 0.0
                metrics["Specificity"] = 0.0
                metrics["Accuracy"] = 0.0

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
        "Accuracy",
    ]

    build_and_save_summary(
        detail_df=detail_df,
        sample_order=SLICES,
        metric_columns=metric_columns,
        output_dir=OUTPUT_DIR,
        file_prefix="iStar_ThreeSlices",
    )


if __name__ == "__main__":
    main()

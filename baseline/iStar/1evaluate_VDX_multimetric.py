import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from medpy.metric.binary import hd95
from sklearn.metrics import average_precision_score, roc_auc_score


# ============================================================
# 基础配置
# ============================================================
SLIDES = ["ViHBC", "DuCIS", "XeHBC"]
SEEDS = [42, 123, 567]

BASE_DATA_DIR = Path(r"D:\Project_2\Pixel_Seg\data")
ISTAR_WORKSPACE_DIR = Path(r"D:\Project_2\iStar\istar_workspace_three_slices")
VIS_SAVE_DIR = ISTAR_WORKSPACE_DIR / "visualizations"
METRIC_SAVE_DIR = ISTAR_WORKSPACE_DIR / "evaluation_results"
PRED_MASK_SAVE_DIR = ISTAR_WORKSPACE_DIR / "pred_masks"
PROBABILITY_SAVE_DIR = ISTAR_WORKSPACE_DIR / "probability_maps"

# 三张切片均为乳腺癌相关数据，沿用 STHBC 的零样本 marker。
BRCA_MARKERS = ["ERBB2", "EPCAM", "MKI67", "CDH1"]

# 按 marker 得分从高到低选取前 50% 的聚类作为癌症区域。
CANCER_CLUSTER_RATIO = 0.5

# 预测可视化透明度。
OVERLAY_ALPHA = 0.60


# ============================================================
# 文件与图像辅助函数
# ============================================================
def find_existing_file(directory: Path, candidates, description: str) -> Path:
    """
    按顺序查找候选文件名，兼容不同数据预处理脚本的命名差异。
    """
    for name in candidates:
        path = directory / name
        if path.exists():
            return path

    candidate_text = ", ".join(str(directory / name) for name in candidates)
    raise FileNotFoundError(
        f"找不到{description}。已检查以下路径：\n{candidate_text}"
    )


def read_gray_image(path: Path, description: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取{description}: {path}")
    return image


def resize_nearest(array: np.ndarray, target_wh) -> np.ndarray:
    """
    以最近邻方式缩放离散标签图。
    OpenCV 不支持直接缩放 int32，因此先转换为 float32，缩放后再还原。
    """
    target_w, target_h = target_wh
    resized = cv2.resize(
        array.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.rint(resized).astype(np.int32)


# ============================================================
# 零样本癌症聚类识别与连续概率图
# ============================================================
def get_zero_shot_cluster_scores(
    adata,
    istar_clusters_2d: np.ndarray,
    marker_genes,
    spatial_key: str,
):
    """
    根据 marker 基因表达计算每个 iStar 聚类的癌症得分。

    返回
    ----
    selected_clusters : list[int]
        marker 得分排名前 CANCER_CLUSTER_RATIO 的聚类。
    cluster_scores : dict[int, float]
        每个聚类的原始 marker 平均表达得分。
    valid_markers : list[str]
        当前切片中实际存在的 marker。
    """
    adata.var_names_make_unique()
    valid_markers = [g for g in marker_genes if g in adata.var_names]

    if not valid_markers:
        print("  [!] 当前表达矩阵中不存在任何指定 marker，无法进行零样本聚类筛选。")
        return [], {}, []

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm 中不存在坐标键 '{spatial_key}'，"
            f"当前可用键为: {list(adata.obsm.keys())}"
        )

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    height, width = istar_clusters_2d.shape

    spot_clusters = np.full(adata.n_obs, -1, dtype=np.int32)

    # 坐标格式沿用原始代码：[x, y]。
    xs = np.rint(coords[:, 0]).astype(np.int64)
    ys = np.rint(coords[:, 1]).astype(np.int64)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    spot_clusters[valid] = istar_clusters_2d[ys[valid], xs[valid]]

    # 避免覆盖原 adata，使用 copy 后增加临时列。
    adata_eval = adata.copy()
    adata_eval.obs["pred_cluster"] = spot_clusters.astype(str)

    cluster_scores = {}
    cluster_ids = sorted(
        int(cid)
        for cid in np.unique(spot_clusters)
        if int(cid) >= 0
    )

    for cid in cluster_ids:
        spot_mask = spot_clusters == cid
        if not np.any(spot_mask):
            continue

        expr_matrix = adata_eval[spot_mask, valid_markers].X
        if hasattr(expr_matrix, "toarray"):
            expr_matrix = expr_matrix.toarray()
        else:
            expr_matrix = np.asarray(expr_matrix)

        score = float(np.nanmean(expr_matrix))
        if np.isfinite(score):
            cluster_scores[cid] = score

    if not cluster_scores:
        print("  [!] 未能为任何有效聚类计算 marker 得分。")
        return [], {}, valid_markers

    sorted_clusters = sorted(
        cluster_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    num_to_select = max(
        1,
        int(np.ceil(len(sorted_clusters) * CANCER_CLUSTER_RATIO)),
    )
    selected_clusters = [
        int(cid) for cid, _ in sorted_clusters[:num_to_select]
    ]

    return selected_clusters, cluster_scores, valid_markers


def build_probability_map(
    cluster_map: np.ndarray,
    cluster_scores,
    tissue_mask: np.ndarray,
) -> np.ndarray:
    """
    将聚类级 marker 得分映射为像素级连续概率图。

    使用聚类得分的 min-max 归一化。AUPRC/AUROC 主要依赖排序，
    因而保留了 marker 得分的相对顺序。
    """
    probability = np.zeros(cluster_map.shape, dtype=np.float32)

    if not cluster_scores:
        return probability

    scores = np.asarray(list(cluster_scores.values()), dtype=np.float64)
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))

    if score_max > score_min:
        normalized_scores = {
            int(cid): float((score - score_min) / (score_max - score_min))
            for cid, score in cluster_scores.items()
        }
    else:
        # 所有聚类分数完全相同时，没有可用于排序的信息。
        normalized_scores = {
            int(cid): 0.5 for cid in cluster_scores
        }

    for cid, score in normalized_scores.items():
        probability[cluster_map == cid] = score

    probability[tissue_mask == 0] = 0.0
    return probability



# ============================================================
# 保存纯预测结果（供后续 spot 级分析直接读取）
# ============================================================
def save_prediction_outputs(
    slide_name: str,
    seed: int,
    pred_mask: np.ndarray,
    probability_map: np.ndarray,
):
    """保存不含 H&E 背景的纯二值 mask 和连续概率图。

    目录结构
    --------
    pred_masks/seed_42/ViHBC_seed42_istar_pred_mask.png
    probability_maps/seed_42/ViHBC_seed42_istar_probability.npz
    probability_maps/seed_42/ViHBC_seed42_istar_probability.png

    ``pred_mask.png`` 中：0=非癌症，255=癌症。
    ``probability.npz`` 中保存 float32 的 [0, 1] 连续 marker 概率图。
    """
    pred_dir = PRED_MASK_SAVE_DIR / f"seed_{seed}"
    prob_dir = PROBABILITY_SAVE_DIR / f"seed_{seed}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f"{slide_name}_seed{seed}_istar_pred_mask.png"
    pred_u8 = ((np.asarray(pred_mask) > 0).astype(np.uint8) * 255)
    if not cv2.imwrite(str(pred_path), pred_u8):
        raise IOError(f"无法保存 iStar pred_mask: {pred_path}")

    probability = np.clip(
        np.nan_to_num(
            np.asarray(probability_map, dtype=np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    prob_npz_path = prob_dir / f"{slide_name}_seed{seed}_istar_probability.npz"
    np.savez_compressed(
        prob_npz_path,
        probability=probability,
        slide=np.asarray(slide_name),
        seed=np.asarray(seed, dtype=np.int32),
    )

    prob_png_path = prob_dir / f"{slide_name}_seed{seed}_istar_probability.png"
    prob_u8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(prob_png_path), prob_u8):
        raise IOError(f"无法保存 iStar probability PNG: {prob_png_path}")

    print(f"             iStar pred_mask 已保存: {pred_path}")
    print(f"             iStar probability 已保存: {prob_npz_path}")
    return pred_path, prob_npz_path


# ============================================================
# 指标计算
# ============================================================
def calculate_metrics(
    pred_mask: np.ndarray,
    probability_map: np.ndarray,
    gt_mask: np.ndarray,
    tissue_mask: np.ndarray,
):
    """
    在组织区域内计算：
    Target IoU、Dice、Recall、Precision、AUPRC、P-AUROC 和 HD95。

    P-AUROC 表示 pixel-level AUROC。
    """
    valid_pixels = tissue_mask > 0

    pred = ((pred_mask > 0) & valid_pixels).astype(np.uint8)
    gt = ((gt_mask > 0) & valid_pixels).astype(np.uint8)

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))

    smooth = 1e-8
    target_iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)
    recall = tp / (tp + fn + smooth)
    precision = tp / (tp + fp + smooth)

    y_true = gt_mask[valid_pixels].astype(np.uint8).reshape(-1)
    y_score = probability_map[valid_pixels].astype(np.float64).reshape(-1)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)

    # AUPRC 在有正样本时有定义；P-AUROC 要求同时存在正、负两类。
    if y_true.size > 0 and np.any(y_true == 1):
        try:
            auprc = float(average_precision_score(y_true, y_score))
        except Exception:
            auprc = np.nan
    else:
        auprc = np.nan

    if y_true.size > 0 and np.unique(y_true).size == 2:
        try:
            pixel_auroc = float(roc_auc_score(y_true, y_score))
        except Exception:
            pixel_auroc = np.nan
    else:
        pixel_auroc = np.nan

    try:
        if np.any(pred == 1) and np.any(gt == 1):
            hd95_value = float(hd95(pred, gt))
        else:
            hd95_value = np.nan
    except Exception:
        hd95_value = np.nan

    return {
        "Target_IoU": float(target_iou),
        "Dice": float(dice),
        "Recall": float(recall),
        "Precision": float(precision),
        "AUPRC": auprc,
        "P-AUROC": pixel_auroc,
        "HD95": hd95_value,
    }


# ============================================================
# 可视化
# ============================================================
def save_visualizations(
    slide_name: str,
    seed: int,
    he_path: Path,
    pred_mask: np.ndarray,
    probability_map: np.ndarray,
    tissue_mask: np.ndarray,
    save_dir: Path,
):
    he_image = cv2.imread(str(he_path), cv2.IMREAD_COLOR)
    if he_image is None:
        print(f"  [!] 无法读取 H&E，跳过可视化: {he_path}")
        return

    target_h, target_w = pred_mask.shape
    he_image = cv2.resize(
        he_image,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    # 预测区域叠加图：红色表示癌症预测。
    overlay_layer = he_image.copy()
    overlay_layer[pred_mask > 0] = (0, 0, 255)
    prediction_overlay = cv2.addWeighted(
        overlay_layer,
        OVERLAY_ALPHA,
        he_image,
        1.0 - OVERLAY_ALPHA,
        0,
    )
    prediction_overlay[tissue_mask == 0] = he_image[tissue_mask == 0]

    overlay_path = save_dir / f"{slide_name}_seed_{seed}_cancer_prediction.jpg"
    cv2.imwrite(str(overlay_path), prediction_overlay)

    # 连续 marker 概率热图，便于核对 AUPRC/P-AUROC 的输入。
    prob_uint8 = np.clip(probability_map * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    heatmap[tissue_mask == 0] = 0
    heatmap_overlay = cv2.addWeighted(
        heatmap,
        0.55,
        he_image,
        0.45,
        0,
    )
    heatmap_overlay[tissue_mask == 0] = he_image[tissue_mask == 0]

    heatmap_path = save_dir / f"{slide_name}_seed_{seed}_marker_probability.jpg"
    cv2.imwrite(str(heatmap_path), heatmap_overlay)


# ============================================================
# 单切片、单随机种子评估
# ============================================================
def evaluate_slide(
    slide_name: str,
    seed: int,
    istar_cluster_path: Path,
    label_path: Path,
    tissue_mask_path: Path,
    he_path: Path,
    adata_path: Path,
    marker_genes,
    vis_save_dir: Path,
):
    gt_image = read_gray_image(label_path, "癌症标签")
    tissue_image = read_gray_image(tissue_mask_path, "组织掩码")

    gt_mask = (gt_image > 127).astype(np.uint8)

    if tissue_image.shape != gt_mask.shape:
        tissue_image = cv2.resize(
            tissue_image,
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    tissue_mask = (tissue_image > 127).astype(np.uint8)

    with open(istar_cluster_path, "rb") as file:
        cluster_object = pickle.load(file)

    cluster_map = np.asarray(cluster_object)
    if cluster_map.ndim != 2:
        raise ValueError(
            f"iStar 聚类结果必须是二维数组，实际 shape={cluster_map.shape}: "
            f"{istar_cluster_path}"
        )

    cluster_map = resize_nearest(
        cluster_map,
        (gt_mask.shape[1], gt_mask.shape[0]),
    )

    adata = sc.read_h5ad(str(adata_path))

    spatial_key = (
        "spatial_resized"
        if "spatial_resized" in adata.obsm
        else "spatial"
    )

    selected_clusters, cluster_scores, valid_markers = (
        get_zero_shot_cluster_scores(
            adata=adata,
            istar_clusters_2d=cluster_map,
            marker_genes=marker_genes,
            spatial_key=spatial_key,
        )
    )

    pred_mask = np.isin(cluster_map, selected_clusters).astype(np.uint8)
    pred_mask[tissue_mask == 0] = 0

    probability_map = build_probability_map(
        cluster_map=cluster_map,
        cluster_scores=cluster_scores,
        tissue_mask=tissue_mask,
    )

    # 保存纯二值预测 mask。后续像素到 spot 的转换必须读取这个文件，
    # 不能再读取带 H&E 背景的红色 overlay 图。
    save_prediction_outputs(
        slide_name=slide_name,
        seed=seed,
        pred_mask=pred_mask,
        probability_map=probability_map,
    )

    metrics = calculate_metrics(
        pred_mask=pred_mask,
        probability_map=probability_map,
        gt_mask=gt_mask,
        tissue_mask=tissue_mask,
    )

    hd95_text = (
        f"{metrics['HD95']:.2f}"
        if np.isfinite(metrics["HD95"])
        else "NaN"
    )
    auprc_text = (
        f"{metrics['AUPRC']:.4f}"
        if np.isfinite(metrics["AUPRC"])
        else "NaN"
    )
    auroc_text = (
        f"{metrics['P-AUROC']:.4f}"
        if np.isfinite(metrics["P-AUROC"])
        else "NaN"
    )

    print(
        f"  [Seed {seed}] "
        f"Target_IoU: {metrics['Target_IoU']:.4f} | "
        f"Dice: {metrics['Dice']:.4f} | "
        f"Recall: {metrics['Recall']:.4f} | "
        f"Precision: {metrics['Precision']:.4f} | "
        f"AUPRC: {auprc_text} | "
        f"P-AUROC: {auroc_text} | "
        f"HD95: {hd95_text}"
    )
    print(
        f"             spatial_key={spatial_key} | "
        f"valid_markers={valid_markers} | "
        f"selected_clusters={selected_clusters}"
    )

    save_visualizations(
        slide_name=slide_name,
        seed=seed,
        he_path=he_path,
        pred_mask=pred_mask,
        probability_map=probability_map,
        tissue_mask=tissue_mask,
        save_dir=vis_save_dir,
    )

    return metrics


# ============================================================
# 汇总与保存
# ============================================================
METRIC_NAMES = [
    "Target_IoU",
    "Dice",
    "Recall",
    "Precision",
    "AUPRC",
    "P-AUROC",
    "HD95",
]


def format_mean_std(values, decimals=4):
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return "NaN", np.nan, np.nan

    mean_value = float(np.mean(valid))
    std_value = float(np.std(valid))
    return (
        f"{mean_value:.{decimals}f}±{std_value:.{decimals}f}",
        mean_value,
        std_value,
    )


def print_summary_table(all_results):
    separator = "=" * 174
    row_separator = (
        "|----------|-------------------|-------------------|-------------------|"
        "-------------------|-------------------|-------------------|----------------|"
    )

    print("\n\n" + separator)
    print("📋 iStar - 三切片多随机种子统计汇总表（均值±标准差）")
    print(separator)
    print(
        "| Slice    | Target_IoU        | Dice              | Recall            | "
        "Precision         | AUPRC             | P-AUROC           | HD95 (px)      |"
    )
    print(row_separator)

    slide_means = {metric: [] for metric in METRIC_NAMES}
    summary_rows = []

    for slide in SLIDES:
        seeds_data = all_results[slide]
        if not seeds_data["Target_IoU"]:
            continue

        formatted = {}
        means = {}
        stds = {}

        for metric in METRIC_NAMES:
            decimals = 2 if metric == "HD95" else 4
            text, mean_value, std_value = format_mean_std(
                seeds_data[metric],
                decimals=decimals,
            )
            formatted[metric] = text
            means[metric] = mean_value
            stds[metric] = std_value

            if np.isfinite(mean_value):
                slide_means[metric].append(mean_value)

        print(
            f"| {slide:<8} | "
            f"{formatted['Target_IoU']:<17} | "
            f"{formatted['Dice']:<17} | "
            f"{formatted['Recall']:<17} | "
            f"{formatted['Precision']:<17} | "
            f"{formatted['AUPRC']:<17} | "
            f"{formatted['P-AUROC']:<17} | "
            f"{formatted['HD95']:<14} |"
        )

        row = {"Slice": slide}
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = means[metric]
            row[f"{metric}_std"] = stds[metric]
        summary_rows.append(row)

    print(row_separator)

    average_formatted = {}
    average_row = {"Slice": "Average"}

    for metric in METRIC_NAMES:
        decimals = 2 if metric == "HD95" else 4
        text, mean_value, std_value = format_mean_std(
            slide_means[metric],
            decimals=decimals,
        )
        average_formatted[metric] = text
        average_row[f"{metric}_mean"] = mean_value
        average_row[f"{metric}_std"] = std_value

    print(
        f"| {'Average':<8} | "
        f"{average_formatted['Target_IoU']:<17} | "
        f"{average_formatted['Dice']:<17} | "
        f"{average_formatted['Recall']:<17} | "
        f"{average_formatted['Precision']:<17} | "
        f"{average_formatted['AUPRC']:<17} | "
        f"{average_formatted['P-AUROC']:<17} | "
        f"{average_formatted['HD95']:<14} |"
    )
    print(separator + "\n")

    summary_rows.append(average_row)
    return pd.DataFrame(summary_rows)


def save_metric_tables(per_seed_rows, summary_df):
    METRIC_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = METRIC_SAVE_DIR / "istar_three_slices_per_seed_metrics.csv"
    summary_path = METRIC_SAVE_DIR / "istar_three_slices_summary_metrics.csv"

    per_seed_df.to_csv(per_seed_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"逐种子指标已保存: {per_seed_path}")
    print(f"汇总指标已保存:   {summary_path}")


# ============================================================
# 主程序
# ============================================================
def main():
    VIS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        slide: {metric: [] for metric in METRIC_NAMES}
        for slide in SLIDES
    }
    per_seed_rows = []

    for slide in SLIDES:
        print(f"\n================ Processing Slide: {slide} ================")

        resized_dir = BASE_DATA_DIR / slide / "resized_data"

        try:
            label_path = find_existing_file(
                resized_dir,
                [
                    "label_resized.png",
                    "cancer_label_resized.png",
                    "gt_resized.png",
                    "label.png",
                ],
                "癌症标签",
            )
            tissue_mask_path = find_existing_file(
                resized_dir,
                [
                    "mask_resized.png",
                    "tissue_mask_resized.png",
                    "mask.png",
                ],
                "组织掩码",
            )
            he_path = find_existing_file(
                resized_dir,
                [
                    "he_resized.png",
                    "he.png",
                ],
                "H&E 图像",
            )
            adata_path = find_existing_file(
                resized_dir,
                [
                    "adata_resized.h5ad",
                    f"{slide}_resized.h5ad",
                    f"{slide}.h5ad",
                ],
                "AnnData 文件",
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"  [ERROR] {error}")
            continue

        for seed in SEEDS:
            cluster_path = (
                ISTAR_WORKSPACE_DIR
                / f"{slide}_seed_{seed}"
                / "clusters-gene"
                / "labels.pickle"
            )

            if not cluster_path.exists():
                print(f"  [!] 找不到 Seed {seed} 的聚类结果: {cluster_path}")
                continue

            try:
                metrics = evaluate_slide(
                    slide_name=slide,
                    seed=seed,
                    istar_cluster_path=cluster_path,
                    label_path=label_path,
                    tissue_mask_path=tissue_mask_path,
                    he_path=he_path,
                    adata_path=adata_path,
                    marker_genes=BRCA_MARKERS,
                    vis_save_dir=VIS_SAVE_DIR,
                )
            except Exception as error:
                print(f"  [ERROR] {slide}, Seed {seed} 评估失败: {error}")
                continue

            for metric in METRIC_NAMES:
                all_results[slide][metric].append(metrics[metric])

            row = {
                "Slice": slide,
                "Seed": seed,
                **metrics,
            }
            per_seed_rows.append(row)

    summary_df = print_summary_table(all_results)
    save_metric_tables(per_seed_rows, summary_df)


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os
from pathlib import Path
import pickle
import scanpy as sc
import pandas as pd
from medpy.metric.binary import hd95
from sklearn.metrics import average_precision_score, roc_auc_score


# 纯像素级二值预测掩码保存目录（与 TESLA 的目录结构保持一致）
PRED_MASK_SAVE_DIR = Path(
    r"D:\Project_2\iStar\istar_workspace_multiseed\pred_masks"
)


def calculate_metrics(pred_mask, probability_map, gt_mask, tissue_mask):
    """
    在组织区域内统一计算 7 个指标：
    Target_IoU、Dice、Recall、Precision、AUPRC、P-AUROC、HD95。
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

    if y_true.size > 0 and np.any(y_true == 1):
        try:
            auprc = float(average_precision_score(y_true, y_score))
        except Exception:
            auprc = np.nan
    else:
        auprc = np.nan

    if y_true.size > 0 and np.unique(y_true).size == 2:
        try:
            p_auroc = float(roc_auc_score(y_true, y_score))
        except Exception:
            p_auroc = np.nan
    else:
        p_auroc = np.nan

    try:
        if np.sum(pred) > 0 and np.sum(gt) > 0:
            hd95_val = float(hd95(pred, gt))
        else:
            hd95_val = np.nan
    except Exception:
        hd95_val = np.nan

    return {
        "Target_IoU": float(target_iou),
        "Dice": float(dice),
        "Recall": float(recall),
        "Precision": float(precision),
        "AUPRC": auprc,
        "P-AUROC": p_auroc,
        "HD95": hd95_val,
    }


def get_zero_shot_cancer_clusters(adata, istar_clusters_2d, marker_genes):
    """
    通过提取共有基因交集，计算各簇在 Marker 基因上的平均表达量，
    选取表达量排名前 50% 的簇作为癌症区域。
    """
    valid_markers = list(set(marker_genes).intersection(set(adata.var_names)))

    if not valid_markers:
        print("  [!] Warning: 提供的 Marker 基因在当前表达矩阵中均不存在。")
        return [], {}

    # 提取空间坐标并映射到 2D 聚类掩码上
    coords = adata.obsm['spatial'].astype(int)
    spot_clusters = []

    for coord in coords:
        x, y = coord[0], coord[1]
        try:
            spot_clusters.append(istar_clusters_2d[y, x])
        except IndexError:
            spot_clusters.append(-1)

    adata.obs['pred_cluster'] = spot_clusters

    cluster_scores = {}
    unique_clusters = np.unique(istar_clusters_2d)

    for cid in unique_clusters:
        if cid == -1: continue

        spots_in_cluster = adata[adata.obs['pred_cluster'] == cid]
        if len(spots_in_cluster) == 0:
            continue

        expr_matrix = spots_in_cluster[:, valid_markers].X
        if hasattr(expr_matrix, "toarray"):
            expr_matrix = expr_matrix.toarray()

        cluster_scores[cid] = np.mean(expr_matrix)

    # 按得分降序排序，选取排名前 50% 的簇
    if cluster_scores:
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        total_clusters = len(sorted_clusters)
        num_to_select = max(1, total_clusters // 2)
        selected_clusters = [item[0] for item in sorted_clusters[:num_to_select]]
    else:
        selected_clusters = []

    return selected_clusters, cluster_scores


def build_probability_map(cluster_map, cluster_scores, tissue_mask):
    """
    将聚类级 marker 得分映射为像素级连续癌症分数图。
    仅用于 AUPRC/P-AUROC，不改变原有 top-50% 癌症簇选择逻辑。
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
        normalized_scores = {int(cid): 0.5 for cid in cluster_scores}

    for cid, score in normalized_scores.items():
        probability[cluster_map == cid] = score
    probability[tissue_mask == 0] = 0.0
    return probability



def save_pred_mask(slide_name, seed, pred_mask):
    r"""
    保存纯二值 iStar 预测 mask，供后续像素到 spot 的转换直接读取。

    输出：
    D:\Project_2\iStar\istar_workspace_multiseed\pred_masks\
    seed_{seed}\{slide}_seed{seed}_istar_pred_mask.png

    PNG 中：
        0   = 非癌症
        255 = 癌症
    """
    pred_dir = PRED_MASK_SAVE_DIR / f"seed_{seed}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    pred_path = (
        pred_dir
        / f"{slide_name}_seed{seed}_istar_pred_mask.png"
    )

    pred_u8 = (
        (np.asarray(pred_mask) > 0).astype(np.uint8)
        * 255
    )

    if not cv2.imwrite(str(pred_path), pred_u8):
        raise IOError(f"无法保存 iStar pred_mask：{pred_path}")

    print(f"             iStar pred_mask 已保存：{pred_path}")
    return pred_path


def evaluate_slide(slide_name, seed, istar_cluster_path, label_path, tissue_mask_path, he_path, adata_path,
                   marker_genes, vis_save_dir):
    # 1. 读取标签与掩码
    gt_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    tissue_mask = cv2.imread(tissue_mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_img > 127).astype(np.uint8)
    tissue_mask = (tissue_mask > 127).astype(np.uint8)

    # 2. 读取 iStar 预测结果并对齐尺寸
    with open(istar_cluster_path, 'rb') as f:
        istar_clusters = pickle.load(f)
    istar_clusters = istar_clusters.astype(np.uint8)
    istar_clusters = cv2.resize(istar_clusters, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # ================= 零样本 Marker 筛选 =================
    if os.path.exists(adata_path):
        adata = sc.read_h5ad(adata_path)
        selected_clusters, cluster_scores = get_zero_shot_cancer_clusters(
            adata, istar_clusters, marker_genes
        )
    else:
        print(f"  [!] Warning: 找不到 {adata_path}，跳过基因筛选。")
        selected_clusters = []
        cluster_scores = {}

    # 根据生物学先验选出的簇生成预测掩码
    current_pred = np.zeros_like(gt_mask)
    for cid in selected_clusters:
        current_pred[istar_clusters == cid] = 1

    # 与像素级评估保持一致：组织外区域强制置零。
    current_pred[tissue_mask == 0] = 0

    # 保存纯黑白预测 mask。后续像素转 spot 必须读取该文件，
    # 不能读取带 H&E 背景的红色 overlay。
    save_pred_mask(
        slide_name=slide_name,
        seed=seed,
        pred_mask=current_pred,
    )

    probability_map = build_probability_map(
        cluster_map=istar_clusters,
        cluster_scores=cluster_scores,
        tissue_mask=tissue_mask,
    )

    metrics = calculate_metrics(
        current_pred,
        probability_map,
        gt_mask,
        tissue_mask,
    )

    hd_str = f"{metrics['HD95']:.2f}" if np.isfinite(metrics['HD95']) else "NaN"
    auprc_str = f"{metrics['AUPRC']:.4f}" if np.isfinite(metrics['AUPRC']) else "NaN"
    pauroc_str = f"{metrics['P-AUROC']:.4f}" if np.isfinite(metrics['P-AUROC']) else "NaN"
    print(
        f"  [Seed {seed}] "
        f"Target_IoU: {metrics['Target_IoU']:.4f} | "
        f"Dice: {metrics['Dice']:.4f} | "
        f"Recall: {metrics['Recall']:.4f} | "
        f"Precision: {metrics['Precision']:.4f} | "
        f"AUPRC: {auprc_str} | "
        f"P-AUROC: {pauroc_str} | "
        f"HD95: {hd_str}"
    )

    # ================= 可视化 (增加 Seed 标识) =================
    he_img = cv2.imread(he_path)
    if he_img is not None:
        he_img = cv2.resize(he_img, (gt_mask.shape[1], gt_mask.shape[0]))
        overlay = he_img.copy()

        overlay[current_pred == 1] = [0, 0, 255]
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, he_img, 1 - alpha, 0, he_img)

        os.makedirs(vis_save_dir, exist_ok=True)
        vis_path = os.path.join(vis_save_dir, f"{slide_name}_seed_{seed}_cancer_prediction.jpg")
        cv2.imwrite(vis_path, he_img)

    return metrics


METRIC_NAMES = [
    "Target_IoU",
    "Dice",
    "Recall",
    "Precision",
    "AUPRC",
    "P-AUROC",
    "HD95",
]


def _format_mean_std(values, metric):
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return "NaN", np.nan, np.nan

    mean_value = float(np.mean(valid))
    std_value = float(np.std(valid))
    if metric == "HD95":
        return f"{mean_value:.2f}±{std_value:.2f}", mean_value, std_value
    return f"{mean_value:.4f}±{std_value:.4f}", mean_value, std_value


def print_summary_table(all_results):
    separator = "=" * 178
    row_separator = (
        "|----------|-------------------|-------------------|-------------------|"
        "-------------------|-------------------|-------------------|----------------|"
    )

    print("\n\n" + separator)
    print("📋 STHBC Baseline - 切片级多种子统计汇总表 (均值±标准差)(iStar)")
    print(separator)
    print(
        "| Slice    | Target_IoU        | Dice              | Recall            | "
        "Precision         | AUPRC             | P-AUROC           | HD95 (px)      |"
    )
    print(row_separator)

    overall_means = {metric: [] for metric in METRIC_NAMES}
    summary_rows = []

    for slide, seeds_data in all_results.items():
        if not seeds_data["Target_IoU"]:
            continue

        formatted = {}
        means = {}
        stds = {}

        for metric in METRIC_NAMES:
            text_value, mean_value, std_value = _format_mean_std(
                seeds_data[metric], metric
            )
            formatted[metric] = text_value
            means[metric] = mean_value
            stds[metric] = std_value
            if np.isfinite(mean_value):
                overall_means[metric].append(mean_value)

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
        text_value, mean_value, std_value = _format_mean_std(
            overall_means[metric], metric
        )
        average_formatted[metric] = text_value
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


if __name__ == "__main__":
    slides = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
    seeds = [42, 123, 567]

    # 基础路径配置
    base_data_dir = r"D:\Project_2\Pixel_Seg\data\STHBC"
    raw_he_base_dir = r"D:\DNM_project(cancer detect)\data\HB\her2st"
    istar_workspace_dir = r"D:\Project_2\iStar\istar_workspace"

    # 定义乳腺癌的先验 Marker 基因列表
    brca_markers = ['ERBB2', 'EPCAM', 'MKI67', 'CDH1']

    vis_save_dir = os.path.join(istar_workspace_dir, "visualizations")

    # 初始化统计字典
    all_results = {
        slide: {metric: [] for metric in METRIC_NAMES}
        for slide in slides
    }
    per_seed_rows = []

    for slide in slides:
        print(f"\n================ Processing Slide: {slide} ================")
        for seed in seeds:
            # 构建各类文件的完整路径
            label_path = os.path.join(base_data_dir, "label_img", f"{slide}_label.png")
            tissue_mask_path = os.path.join(base_data_dir, "mask", f"{slide}_hest.png")
            he_path = os.path.join(raw_he_base_dir, slide, "he-raw.jpg")

            # 匹配包含随机种子的文件夹命名规范
            istar_cluster_path = os.path.join(istar_workspace_dir, f"{slide}_seed_{seed}", "clusters-gene", "labels.pickle")
            adata_path = os.path.join(raw_he_base_dir, slide, f"{slide}.h5ad")

            if os.path.exists(istar_cluster_path) and os.path.exists(label_path):
                metrics = evaluate_slide(slide, seed, istar_cluster_path, label_path, tissue_mask_path, he_path,
                                         adata_path, brca_markers, vis_save_dir)

                # 收集每次 seed 的结果
                for metric in METRIC_NAMES:
                    all_results[slide][metric].append(metrics[metric])

                per_seed_rows.append({
                    "Slice": slide,
                    "Seed": seed,
                    **metrics,
                })
            else:
                print(f"  [!] Warning: 找不到 Seed {seed} 的结果路径 {istar_cluster_path}")

    # 所有评估结束，输出汇总表
    print_summary_table(all_results)
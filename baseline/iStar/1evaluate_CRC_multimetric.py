import cv2
import numpy as np
import os
import pickle
import warnings
import scanpy as sc
import pandas as pd
from medpy.metric.binary import hd95
from sklearn.metrics import average_precision_score, roc_auc_score

# 屏蔽 AnnData 和 Scanpy 的常规警告，保持控制台清爽
warnings.filterwarnings("ignore", category=UserWarning, module="anndata")
warnings.filterwarnings("ignore", category=FutureWarning)


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
    使用动态均值阈值过滤，选出真正的癌症区域。
    """
    valid_markers = list(set(marker_genes).intersection(set(adata.var_names)))

    if not valid_markers:
        print("  [!] Warning: 提供的 Marker 基因在当前表达矩阵中均不存在。")
        return [], {}

    # 🌟 修复 1：优先尝试读取裁剪后的对齐坐标
    if 'spatial_cropped' in adata.obsm:
        coords = adata.obsm['spatial_cropped'].astype(int)
    else:
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
        if cid == -1 or cid == 255:
            continue

        spots_in_cluster = adata[adata.obs['pred_cluster'] == cid]
        if len(spots_in_cluster) == 0:
            continue

        expr_matrix = spots_in_cluster[:, valid_markers].X
        if hasattr(expr_matrix, "toarray"):
            expr_matrix = expr_matrix.toarray()

        # 计算该簇所有 Spot 在 Marker 基因上的平均表达量
        cluster_scores[cid] = np.mean(expr_matrix)

    # 🌟 修复 2：废弃前 50% 逻辑，改用“动态均值阈值法”
    if cluster_scores:
        # 算出所有簇的平均表达得分
        mean_score = np.mean(list(cluster_scores.values()))

        # 只有得分严格高于平均值（甚至你可以乘以一个系数如 1.2 来提高严谨度）的簇，才被判定为癌症
        selected_clusters = [cid for cid, score in cluster_scores.items() if score > mean_score]

        # 兜底：如果都没超过均值，就选得分最高的那一个
        if not selected_clusters:
            selected_clusters = [max(cluster_scores, key=cluster_scores.get)]
    else:
        selected_clusters = []

    return selected_clusters, cluster_scores


def build_probability_map(cluster_map, cluster_scores, tissue_mask):
    """
    将 iStar 聚类级 marker 得分映射为像素级连续癌症分数图。
    使用 min-max 归一化，仅用于 AUPRC 和 P-AUROC，不改变原有二值癌症簇筛选逻辑。
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
        normalized_scores = {
            int(cid): 0.5
            for cid in cluster_scores
        }

    for cid, score in normalized_scores.items():
        probability[cluster_map == cid] = score

    probability[tissue_mask == 0] = 0.0
    return probability


# def get_zero_shot_cancer_clusters(adata, istar_clusters_2d, marker_genes):
#     """
#     [纯暴力调试版]
#     完全废弃 Marker 基因测算逻辑。
#     直接读取聚类结果中的所有类标号，并无脑选取前 5 个类视为癌症区域。
#     """
#     # 提取这张图上所有存在的聚类 ID，例如 [0, 1, 2, 3, 4, 5, 6, 255]
#     unique_clusters = np.unique(istar_clusters_2d)
#
#     # 过滤掉图像背景边缘 (通常是 255) 和越界值 (-1)
#     valid_clusters = [c for c in unique_clusters if c not in [-1, 255]]
#
#     # 强行取前 5 个类。如果总共只有 4 个类，就全取。
#     top_n = 5
#     selected_clusters = valid_clusters[:top_n]
#
#     return selected_clusters


def evaluate_slide(slide_name, seed, istar_cluster_path, label_path, tissue_mask_path, he_path, visium_dir,
                   marker_genes,
                   vis_save_dir):
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
    if os.path.exists(visium_dir):
        try:
            adata = sc.read_visium(visium_dir)
            adata.var_names_make_unique()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            selected_clusters, cluster_scores = get_zero_shot_cancer_clusters(
                adata, istar_clusters, marker_genes
            )
        except Exception as e:
            print(f"  [!] Error reading Visium data at {visium_dir}: {e}")
            selected_clusters = []
            cluster_scores = {}
    else:
        print(f"  [!] Warning: 找不到 Visium 目录 {visium_dir}，跳过基因筛选。")
        selected_clusters = []
        cluster_scores = {}

    # 根据生物学先验选出的簇生成预测掩码
    current_pred = np.zeros_like(gt_mask)
    for cid in selected_clusters:
        current_pred[istar_clusters == cid] = 1

    probability_map = build_probability_map(
        cluster_map=istar_clusters,
        cluster_scores=cluster_scores,
        tissue_mask=tissue_mask,
    )

    # 计算最终的 7 个评估指标
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
        f"  [Seed {seed:<3}] "
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

        # 红色预测区域叠加
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
    print("📋 CRC Baseline - 切片级多种子统计汇总表 (均值±标准差)(iStar)")
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


def precheck_marker_genes(samples, st_dir, markers):
    """全局预检：检查所有候选 Marker 基因在所有切片的表达矩阵中是否存在"""
    print("\n" + "🚀" * 20)
    print("🔍 [预检阶段] 验证 Marker 基因的存在情况")
    print("🚀" * 20)

    gene_presence = {g: [] for g in markers}

    for slide in samples:
        visium_dir = os.path.join(st_dir, slide)
        if not os.path.exists(visium_dir):
            print(f"  [!] 跳过切片 {slide}: 找不到 Visium 目录 {visium_dir}")
            continue

        try:
            adata = sc.read_visium(visium_dir)
            adata.var_names_make_unique()
            genes_in_slide = set(adata.var_names)

            for g in markers:
                if g in genes_in_slide:
                    gene_presence[g].append(slide)
        except Exception as e:
            print(f"  [!] 读取 {slide} 数据出错: {e}")

    total_checked = len(samples)
    print("\n📊 基因检查报告:")
    print("-" * 50)

    # 筛选出至少存在于一个切片中的有效基因
    valid_markers = []

    for g in markers:
        present_in = len(gene_presence[g])
        if present_in == total_checked:
            print(f"  ✅ {g:<8}: 完美！存在于全部 {present_in}/{total_checked} 个切片中。")
            valid_markers.append(g)
        elif present_in > 0:
            print(f"  ⚠️ {g:<8}: 部分存在！仅在 {present_in}/{total_checked} 个切片中找到。")
            valid_markers.append(g)  # 部分存在也可以留着，计算时会自动交集
        else:
            print(f"  ❌ {g:<8}: 警告！在所有 {total_checked} 个切片中均未找到（可能已在预处理中被过滤）。")

    print("-" * 50 + "\n")
    return valid_markers


if __name__ == "__main__":
    # 1. 完整样本集
    all_samples = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4',
                   'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2',
                   'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']
    seeds = [42, 123, 567]

    # 2. 路径配置
    base_dir = r"D:\Project_2\Pixel_Seg\data\CRC\cropped_data"
    label_base_dir = os.path.join(base_dir, "label_cropped")
    mask_base_dir = os.path.join(base_dir, "tissue_mask_cropped")
    he_base_dir = os.path.join(base_dir, "he_cropped")
    raw_st_base_dir = r"D:\DNM_project(cancer detect)\data\CRC"
    istar_workspace_dir = r"D:\Project_2\iStar\istar_workspace\CRC"
    vis_save_dir = os.path.join(istar_workspace_dir, "visualizations")

    # 3. 扩充的 CRC 先验 Marker 基因列表
    candidate_markers =  ['RPS24', 'PPBP', 'PCCA',]

    # ================= 🚀 新增：预先检查基因有效性 =================
    # 这一步会扫描所有数据，输出一份存在性报告，并返回至少存在于一部分切片中的基因列表
    crc_markers = precheck_marker_genes(all_samples, raw_st_base_dir, candidate_markers)

    if not crc_markers:
        raise ValueError("所有提供的 Marker 基因都不在数据集中！请检查基因名大小写或过滤参数。")
    # ==============================================================

    # 初始化统计字典
    all_results = {
        slide: {metric: [] for metric in METRIC_NAMES}
        for slide in all_samples
    }
    per_seed_rows = []

    for slide in all_samples:
        print(f"\n================ Processing Slide: {slide} ================")
        for seed in seeds:
            label_path = os.path.join(label_base_dir, f"{slide}.png")
            tissue_mask_path = os.path.join(mask_base_dir, f"{slide}.png")
            he_path = os.path.join(he_base_dir, f"{slide}.png")

            # 指向包含 Seed 信息的预测结果路径
            istar_cluster_path = os.path.join(istar_workspace_dir, f"{slide}_seed_{seed}", "clusters-gene",
                                              "labels.pickle")
            visium_dir = os.path.join(raw_st_base_dir, slide)

            if os.path.exists(istar_cluster_path) and os.path.exists(label_path):
                metrics = evaluate_slide(slide, seed, istar_cluster_path, label_path, tissue_mask_path, he_path,
                                         visium_dir, crc_markers, vis_save_dir)

                # 收集每次 seed 的结果
                for metric in METRIC_NAMES:
                    all_results[slide][metric].append(metrics[metric])

                per_seed_rows.append({
                    "Slice": slide,
                    "Seed": seed,
                    **metrics,
                })
            else:
                print(f"  [!] Warning: 找不到 Seed {seed} 的结果路径 {istar_cluster_path} 或标签文件。")

    # 所有评估结束，输出汇总表并保存逐种子/汇总 CSV
    summary_df = print_summary_table(all_results)
    metric_save_dir = os.path.join(istar_workspace_dir, "evaluation_results")
    os.makedirs(metric_save_dir, exist_ok=True)
    pd.DataFrame(per_seed_rows).to_csv(
        os.path.join(metric_save_dir, "istar_crc_per_seed_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        os.path.join(metric_save_dir, "istar_crc_summary_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )



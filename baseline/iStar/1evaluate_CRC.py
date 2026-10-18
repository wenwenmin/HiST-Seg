import cv2
import numpy as np
import os
import pickle
import warnings
import scanpy as sc
import pandas as pd
from medpy.metric.binary import hd95

# 屏蔽 AnnData 和 Scanpy 的常规警告，保持控制台清爽
warnings.filterwarnings("ignore", category=UserWarning, module="anndata")
warnings.filterwarnings("ignore", category=FutureWarning)


def calculate_metrics(pred_mask, gt_mask, tissue_mask):
    # 严格限定在组织区域内，并转换为 uint8 格式，防止 medpy 报错
    pred = ((pred_mask > 0) & (tissue_mask > 0)).astype(np.uint8)
    gt = ((gt_mask > 0) & (tissue_mask > 0)).astype(np.uint8)

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    smooth = 1e-8
    iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)
    recall = tp / (tp + fn + smooth)

    try:
        # 只有在预测和金标准都存在正样本时，HD95 才有意义
        if np.sum(pred) > 0 and np.sum(gt) > 0:
            hd95_val = hd95(pred, gt)
        else:
            hd95_val = np.nan
    except Exception:
        hd95_val = np.nan

    return {'Target_IoU': iou, 'Dice': dice, 'Recall': recall, 'HD95': hd95_val}


def get_zero_shot_cancer_clusters(adata, istar_clusters_2d, marker_genes):
    """
    通过提取共有基因交集，计算各簇在 Marker 基因上的平均表达量，
    使用动态均值阈值过滤，选出真正的癌症区域。
    """
    valid_markers = list(set(marker_genes).intersection(set(adata.var_names)))

    if not valid_markers:
        print("  [!] Warning: 提供的 Marker 基因在当前表达矩阵中均不存在。")
        return []

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

    return selected_clusters

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

            selected_clusters = get_zero_shot_cancer_clusters(adata, istar_clusters, marker_genes)
        except Exception as e:
            print(f"  [!] Error reading Visium data at {visium_dir}: {e}")
            selected_clusters = []
    else:
        print(f"  [!] Warning: 找不到 Visium 目录 {visium_dir}，跳过基因筛选。")
        selected_clusters = []

    # 根据生物学先验选出的簇生成预测掩码
    current_pred = np.zeros_like(gt_mask)
    for cid in selected_clusters:
        current_pred[istar_clusters == cid] = 1

    # 计算最终的分割指标
    metrics = calculate_metrics(current_pred, gt_mask, tissue_mask)

    # 终端精简输出
    hd_str = f"{metrics['HD95']:.2f}" if not np.isnan(metrics['HD95']) else "NaN"
    print(
        f"  [Seed {seed:<3}] Target_IoU: {metrics['Target_IoU']:.4f} | Dice: {metrics['Dice']:.4f} | Recall: {metrics['Recall']:.4f} | HD95: {hd_str}")

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


def print_summary_table(all_results):
    print(
        "\n\n=========================================================================================================")
    print("📋 CRC Baseline - 切片级多种子统计汇总表 (均值±标准差)(iStar)")
    print("=========================================================================================================")
    print("| Slice    | Target_IoU        | Dice              | Recall            | HD95 (px)      |")
    print("|----------|-------------------|-------------------|-------------------|----------------|")

    overall_means = {'Target_IoU': [], 'Dice': [], 'Recall': [], 'HD95': []}

    for slide, seeds_data in all_results.items():
        if not seeds_data['Target_IoU']:
            continue

        iou_mean, iou_std = np.mean(seeds_data['Target_IoU']), np.std(seeds_data['Target_IoU'])
        dice_mean, dice_std = np.mean(seeds_data['Dice']), np.std(seeds_data['Dice'])
        rec_mean, rec_std = np.mean(seeds_data['Recall']), np.std(seeds_data['Recall'])

        # 过滤掉 NaN 的 HD95 值再计算
        valid_hd95 = [val for val in seeds_data['HD95'] if not np.isnan(val)]
        if valid_hd95:
            hd_mean, hd_std = np.mean(valid_hd95), np.std(valid_hd95)
            hd_str = f"{hd_mean:.2f}±{hd_std:.2f}"
            overall_means['HD95'].append(hd_mean)
        else:
            hd_str = "NaN"

        # 保存均值用于最后计算整体 Average
        overall_means['Target_IoU'].append(iou_mean)
        overall_means['Dice'].append(dice_mean)
        overall_means['Recall'].append(rec_mean)

        iou_str = f"{iou_mean:.4f}±{iou_std:.4f}"
        dice_str = f"{dice_mean:.4f}±{dice_std:.4f}"
        rec_str = f"{rec_mean:.4f}±{rec_std:.4f}"

        print(f"| {slide:<8} | {iou_str:<17} | {dice_str:<17} | {rec_str:<17} | {hd_str:<14} |")

    print("|----------|-------------------|-------------------|-------------------|----------------|")

    # 计算整体 Average
    if overall_means['Target_IoU']:
        avg_iou_mean, avg_iou_std = np.mean(overall_means['Target_IoU']), np.std(overall_means['Target_IoU'])
        avg_dice_mean, avg_dice_std = np.mean(overall_means['Dice']), np.std(overall_means['Dice'])
        avg_rec_mean, avg_rec_std = np.mean(overall_means['Recall']), np.std(overall_means['Recall'])

        if overall_means['HD95']:
            avg_hd_mean, avg_hd_std = np.mean(overall_means['HD95']), np.std(overall_means['HD95'])
            avg_hd_str = f"{avg_hd_mean:.2f}±{avg_hd_std:.2f}"
        else:
            avg_hd_str = "NaN"

        avg_iou_str = f"{avg_iou_mean:.4f}±{avg_iou_std:.4f}"
        avg_dice_str = f"{avg_dice_mean:.4f}±{avg_dice_std:.4f}"
        avg_rec_str = f"{avg_rec_mean:.4f}±{avg_rec_std:.4f}"

        print(f"| Average  | {avg_iou_str:<17} | {avg_dice_str:<17} | {avg_rec_str:<17} | {avg_hd_str:<14} |")

    print("=========================================================================================================\n")


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
    all_results = {slide: {'Target_IoU': [], 'Dice': [], 'Recall': [], 'HD95': []} for slide in all_samples}

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
                all_results[slide]['Target_IoU'].append(metrics['Target_IoU'])
                all_results[slide]['Dice'].append(metrics['Dice'])
                all_results[slide]['Recall'].append(metrics['Recall'])
                all_results[slide]['HD95'].append(metrics['HD95'])
            else:
                print(f"  [!] Warning: 找不到 Seed {seed} 的结果路径 {istar_cluster_path} 或标签文件。")

    # 所有评估结束，输出汇总表
    print_summary_table(all_results)



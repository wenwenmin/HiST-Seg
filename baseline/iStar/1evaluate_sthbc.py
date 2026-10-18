import cv2
import numpy as np
import os
import pickle
import scanpy as sc
import pandas as pd
from medpy.metric.binary import hd95


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
    选取表达量排名前 50% 的簇作为癌症区域。
    """
    valid_markers = list(set(marker_genes).intersection(set(adata.var_names)))

    if not valid_markers:
        print("  [!] Warning: 提供的 Marker 基因在当前表达矩阵中均不存在。")
        return []

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

    return selected_clusters


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
        selected_clusters = get_zero_shot_cancer_clusters(adata, istar_clusters, marker_genes)
    else:
        print(f"  [!] Warning: 找不到 {adata_path}，跳过基因筛选。")
        selected_clusters = []

    # 根据生物学先验选出的簇生成预测掩码
    current_pred = np.zeros_like(gt_mask)
    for cid in selected_clusters:
        current_pred[istar_clusters == cid] = 1

    # 计算最终的分割指标 (只保留指定的4个核心指标)
    metrics = calculate_metrics(current_pred, gt_mask, tissue_mask)

    # 终端精简输出
    hd_str = f"{metrics['HD95']:.2f}" if not np.isnan(metrics['HD95']) else "NaN"
    print(f"  [Seed {seed}] Target_IoU: {metrics['Target_IoU']:.4f} | Dice: {metrics['Dice']:.4f} | Recall: {metrics['Recall']:.4f} | HD95: {hd_str}")

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


def print_summary_table(all_results):
    print("\n\n=========================================================================================================")
    print("📋 Baseline - 切片级多种子统计汇总表 (均值±标准差)(iStar)")
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
    all_results = {slide: {'Target_IoU': [], 'Dice': [], 'Recall': [], 'HD95': []} for slide in slides}

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
                all_results[slide]['Target_IoU'].append(metrics['Target_IoU'])
                all_results[slide]['Dice'].append(metrics['Dice'])
                all_results[slide]['Recall'].append(metrics['Recall'])
                all_results[slide]['HD95'].append(metrics['HD95'])
            else:
                print(f"  [!] Warning: 找不到 Seed {seed} 的结果路径 {istar_cluster_path}")

    # 所有评估结束，输出汇总表
    print_summary_table(all_results)
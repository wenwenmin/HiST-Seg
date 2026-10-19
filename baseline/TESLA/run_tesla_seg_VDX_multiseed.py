import os
import sys
import random
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from medpy.metric.binary import hd95
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score

try:
    import torch
except Exception:
    torch = None

# ==========================================
# 动态添加 TESLA 包的环境变量路径
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
tesla_pkg_path = os.path.join(current_dir, "TESLA_package")
sys.path.insert(0, tesla_pkg_path)

from TESLA.imputation import imputation as run_imputation
from TESLA.annotation import annotation as run_annotation

# ==========================================
# 全局参数
# ==========================================
RESIZE_FACTOR = 0.2
TESLA_RES = 50

# 默认关闭 Otsu。
USE_OTSU_CLUSTER_SELECTION = False

# 是否自动判断并修正 obsm['spatial'] 的 x/y 顺序。
AUTO_FIX_SPATIAL_XY = True

# 多随机种子实验设置
RANDOM_SEEDS = [42, 123, 567]


# ==========================================
# 统计与辅助工具函数
# ==========================================
def set_global_seed(seed):
    """尽可能固定 Python / NumPy / PyTorch 随机性。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def _safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def _safe_nanvar(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    # 使用总体方差 ddof=0
    return float(np.nanvar(values, ddof=0))


def _format_mean_var(mean_val, var_val, digits=4):
    """格式化输出：均值±方差"""
    if np.isnan(mean_val):
        return "NaN"
    if np.isnan(var_val):
        return f"{mean_val:.{digits}f}±NaN"
    return f"{mean_val:.{digits}f}±{var_val:.{digits}f}"


# ==========================================
# 图像与矩阵工具函数
# ==========================================
def resize_for_display(img, dsize, is_label=False):
    if is_label:
        return cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)

    src_h, src_w = img.shape[:2]
    dst_w, dst_h = dsize
    shrinking = (dst_w * dst_h) < (src_w * src_h)
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(img, dsize, interpolation=interp)


def resize_binary_mask(mask_bin, target_shape_hw):
    target_h, target_w = target_shape_hw
    if mask_bin.shape[:2] == (target_h, target_w):
        out = mask_bin.copy()
    else:
        out = cv2.resize(mask_bin.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8)


def calculate_metrics(pred_mask, gt_mask, tissue_mask):
    """计算 7 个评估指标：只在 tissue_mask 区域内计算。"""
    pred = (pred_mask > 0) & (tissue_mask > 0)
    gt = (gt_mask > 0) & (tissue_mask > 0)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)

    smooth = 1e-8
    iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)
    recall = tp / (tp + fn + smooth)
    precision = tp / (tp + fp + smooth)

    # 提取有效区域展平为一维，用于计算 AUPRC 和 AUROC
    pred_flat = pred[tissue_mask > 0].astype(np.uint8)
    gt_flat = gt[tissue_mask > 0].astype(np.uint8)

    try:
        auprc = float(average_precision_score(gt_flat, pred_flat))
    except Exception:
        auprc = np.nan

    try:
        p_auroc = float(roc_auc_score(gt_flat, pred_flat))
    except Exception:
        p_auroc = np.nan

    try:
        if np.sum(pred) > 0 and np.sum(gt) > 0:
            hd95_val = float(hd95(pred, gt))
            hd_str = f"{hd95_val:.4f}"
        else:
            hd95_val = np.nan
            hd_str = "NaN"
    except Exception:
        hd95_val = np.nan
        hd_str = "NaN"

    return {
        "Target_IoU": float(iou),
        "Dice": float(dice),
        "Recall": float(recall),
        "Precision": float(precision),
        "AUPRC": float(auprc),
        "P-AUROC": float(p_auroc),
        "HD95": hd95_val,
        "HD95_str": hd_str,
    }


def ensure_bgr_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"图像读取失败: {img_path}")
    return np.ascontiguousarray(img)


def ensure_gray_image(img_path, name="image"):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"{name} 读取失败: {img_path}")
    return img


def set_tesla_pixel_coordinates(adata, spatial_coords, img_shape_hw, auto_fix_xy=True):
    img_h, img_w = img_shape_hw
    c0 = np.asarray(spatial_coords[:, 0], dtype=float)
    c1 = np.asarray(spatial_coords[:, 1], dtype=float)

    def in_range(vals, upper):
        return np.nanmin(vals) >= -5 and np.nanmax(vals) <= upper + 5

    if auto_fix_xy:
        looks_like_row_col = in_range(c0, img_h) and in_range(c1, img_w)
        looks_like_col_row = in_range(c0, img_w) and in_range(c1, img_h)

        if looks_like_col_row and not looks_like_row_col:
            adata.obs["pixel_x"] = c1
            adata.obs["pixel_y"] = c0
            print("  -> spatial 判断为常规 (x,y)=(col,row)，已交换为 TESLA 的 pixel_x=row, pixel_y=col")
        else:
            adata.obs["pixel_x"] = c0
            adata.obs["pixel_y"] = c1
            if looks_like_col_row and looks_like_row_col:
                print("  -> spatial 范围较难区分 x/y，暂保持原顺序；若结果整体错位，请手动交换 pixel_x/pixel_y")
            else:
                print("  -> spatial 按原顺序写入 TESLA: pixel_x=spatial[:,0], pixel_y=spatial[:,1]")
    else:
        adata.obs["pixel_x"] = c0
        adata.obs["pixel_y"] = c1

    return adata


def pred_refined_to_2d_map(pred_refined, img_shape_hw, resize_factor):
    img_h, img_w = img_shape_hw
    resize_w = int(img_w * resize_factor)
    resize_h = int(img_h * resize_factor)

    pred_arr = np.asarray(pred_refined).astype(np.int32)
    expected_size = resize_h * resize_w

    if pred_arr.ndim == 2:
        pred_map = pred_arr
    elif pred_arr.ndim == 1:
        if pred_arr.size != expected_size:
            raise ValueError(f"pred_refined 长度不匹配: got {pred_arr.size}, expected {expected_size}")
        pred_map = pred_arr.reshape(resize_h, resize_w)
    else:
        raise ValueError(f"pred_refined 维度异常: ndim={pred_arr.ndim}, shape={pred_arr.shape}")

    return pred_map.astype(np.int32)


def select_target_clusters(tesla_target_clusters, c_m, use_otsu=False):
    if tesla_target_clusters is None:
        tesla_target_clusters = []

    if not use_otsu:
        return [int(x) for x in tesla_target_clusters]

    if c_m is None or len(c_m) == 0:
        return [int(x) for x in tesla_target_clusters]

    c_m_arr = np.array(c_m)
    if c_m_arr.ndim == 1:
        cluster_ids = np.arange(len(c_m_arr)).astype(int)
        scores = c_m_arr.astype(float)
    elif c_m_arr.ndim == 2 and c_m_arr.shape[1] >= 2:
        cluster_ids = c_m_arr[:, 0].astype(int)
        scores = c_m_arr[:, 1].astype(float)
    else:
        return [int(x) for x in tesla_target_clusters]

    if scores.max() <= scores.min():
        return [int(x) for x in tesla_target_clusters]

    scores_norm = np.uint8(255 * (scores - scores.min()) / (scores.max() - scores.min()))
    thresh_norm, _ = cv2.threshold(scores_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    real_thresh = scores.min() + (float(thresh_norm) / 255.0) * (scores.max() - scores.min())

    selected = cluster_ids[scores >= real_thresh].tolist()
    return [int(x) for x in selected]


def build_three_slice_configs():
    base = r"D:\Project_2\Pixel_Seg\data"
    slice_names = ["ViHBC", "DuCIS", "XeHBC"]

    configs = []
    for s in slice_names:
        resized_dir = os.path.join(base, s, "resized_data")
        configs.append(
            {
                "name": s,
                "img_path": os.path.join(resized_dir, "he_resized.png"),
                "adata_path": os.path.join(resized_dir, "adata_resized.h5ad"),
                "gt_path": os.path.join(resized_dir, "label_resized.png"),
                "mask_path": os.path.join(resized_dir, "mask_resized.png"),
            }
        )
    return configs


# ==========================================
# 汇总表格构建函数
# ==========================================
def build_summary_dataframe(records, slice_order):
    metric_cols = ["Target_IoU", "Dice", "Recall", "Precision", "AUPRC", "P-AUROC", "HD95"]
    raw_df = pd.DataFrame(records)

    if raw_df.empty:
        return raw_df, pd.DataFrame()

    rows = []
    for s in slice_order:
        part = raw_df[raw_df["Slice"] == s]
        if part.empty:
            continue

        row = {"Slice": s, "Num_Seeds": int(part["Seed"].nunique()) if "Seed" in part.columns else len(part)}
        for m in metric_cols:
            vals = part[m].astype(float).to_numpy()
            row[f"{m}_mean"] = _safe_nanmean(vals)
            row[f"{m}_var"] = _safe_nanvar(vals)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        avg_row = {"Slice": "Average",
                   "Num_Seeds": int(raw_df["Seed"].nunique()) if "Seed" in raw_df.columns else np.nan}
        for m in metric_cols:
            slice_means = summary_df[f"{m}_mean"].astype(float).to_numpy()
            avg_row[f"{m}_mean"] = _safe_nanmean(slice_means)
            avg_row[f"{m}_var"] = _safe_nanvar(slice_means)
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)

    return raw_df, summary_df


def summary_dataframe_to_table(summary_df, method_name="TESLA_Three_Slices"):
    if summary_df.empty:
        return "\n[Summary] 没有成功完成的切片，无法生成汇总表。\n"

    display_rows = []
    for _, row in summary_df.iterrows():
        display_rows.append({
            "Slice": str(row["Slice"]),
            "Target_IoU": _format_mean_var(row["Target_IoU_mean"], row["Target_IoU_var"], digits=4),
            "Dice": _format_mean_var(row["Dice_mean"], row["Dice_var"], digits=4),
            "Recall": _format_mean_var(row["Recall_mean"], row["Recall_var"], digits=4),
            "Precision": _format_mean_var(row["Precision_mean"], row["Precision_var"], digits=4),
            "AUPRC": _format_mean_var(row["AUPRC_mean"], row["AUPRC_var"], digits=4),
            "P-AUROC": _format_mean_var(row["P-AUROC_mean"], row["P-AUROC_var"], digits=4),
            "HD95 (px)": _format_mean_var(row["HD95_mean"], row["HD95_var"], digits=2),
        })

    widths = {
        "Slice": max(8, max(len(r["Slice"]) for r in display_rows)),
        "Target_IoU": max(15, max(len(r["Target_IoU"]) for r in display_rows)),
        "Dice": max(15, max(len(r["Dice"]) for r in display_rows)),
        "Recall": max(15, max(len(r["Recall"]) for r in display_rows)),
        "Precision": max(15, max(len(r["Precision"]) for r in display_rows)),
        "AUPRC": max(15, max(len(r["AUPRC"]) for r in display_rows)),
        "P-AUROC": max(15, max(len(r["P-AUROC"]) for r in display_rows)),
        "HD95 (px)": max(13, max(len(r["HD95 (px)"]) for r in display_rows)),
    }

    def border_line():
        return (
                "|" + "-" * (widths["Slice"] + 2) + "|" + "-" * (widths["Target_IoU"] + 2) +
                "|" + "-" * (widths["Dice"] + 2) + "|" + "-" * (widths["Recall"] + 2) +
                "|" + "-" * (widths["Precision"] + 2) + "|" + "-" * (widths["AUPRC"] + 2) +
                "|" + "-" * (widths["P-AUROC"] + 2) + "|" + "-" * (widths["HD95 (px)"] + 2) + "|"
        )

    def row_line(row):
        return (
            f"| {row['Slice']:<{widths['Slice']}} | {row['Target_IoU']:<{widths['Target_IoU']}} "
            f"| {row['Dice']:<{widths['Dice']}} | {row['Recall']:<{widths['Recall']}} "
            f"| {row['Precision']:<{widths['Precision']}} | {row['AUPRC']:<{widths['AUPRC']}} "
            f"| {row['P-AUROC']:<{widths['P-AUROC']}} | {row['HD95 (px)']:<{widths['HD95 (px)']}} |"
        )

    title = f"📋 {method_name} - 切片级多种子统计汇总表 (均值±方差)"
    table_width = len(border_line())
    sep = "=" * table_width

    lines = [
        "", sep, title, sep,
        (
            f"| {'Slice':<{widths['Slice']}} | {'Target_IoU':<{widths['Target_IoU']}} "
            f"| {'Dice':<{widths['Dice']}} | {'Recall':<{widths['Recall']}} "
            f"| {'Precision':<{widths['Precision']}} | {'AUPRC':<{widths['AUPRC']}} "
            f"| {'P-AUROC':<{widths['P-AUROC']}} | {'HD95 (px)':<{widths['HD95 (px)']}} |"
        ),
        border_line(),
    ]

    for r in display_rows:
        if r["Slice"] == "Average":
            lines.append(border_line())
        lines.append(row_line(r))

    lines.append(sep)
    return "\n".join(lines)


def process_one_slice(cfg, marker_genes, output_dirs, seed):
    s = cfg["name"]
    print(f"\n========== 开始处理切片: {s} | Seed: {seed} ==========")

    try:
        img_raw = ensure_bgr_image(cfg["img_path"])
        img_tesla = img_raw.copy()
        img_vis = img_raw.copy()

        adata = sc.read_h5ad(cfg["adata_path"])
        adata.var_names_make_unique()
        gt_img = ensure_gray_image(cfg["gt_path"], name="GT label")
        tissue_mask_raw = ensure_gray_image(cfg["mask_path"], name="Tissue mask")
    except Exception as e:
        print(f"[{s} | seed={seed}] 文件读取失败: {e}")
        return None

    img_h, img_w = img_tesla.shape[:2]
    gt_mask = (gt_img > 127).astype(np.uint8)
    tissue_mask_raw_bin = (tissue_mask_raw > 127).astype(np.uint8)

    tissue_mask_tesla = resize_binary_mask(tissue_mask_raw_bin, (img_h, img_w))
    tissue_mask_eval = resize_binary_mask(tissue_mask_raw_bin, gt_mask.shape[:2])

    valid_genes = [g for g in marker_genes if g in adata.var_names]
    if not valid_genes:
        print(f"[{s} | seed={seed}] 未找到有效 marker genes，跳过。")
        return None

    adata_sub = adata[:, valid_genes].copy()
    if hasattr(adata_sub.X, "toarray"):
        adata_sub.X = adata_sub.X.toarray()

    really_valid_genes = []
    for idx, g in enumerate(valid_genes):
        vals = np.asarray(adata_sub.X[:, idx]).reshape(-1)
        if np.nanmax(vals) > np.nanmin(vals):
            really_valid_genes.append(g)

    if not really_valid_genes:
        print(f"[{s} | seed={seed}] marker genes 在该切片中均为常数表达，跳过。")
        return None

    valid_genes = really_valid_genes
    adata_sub = adata_sub[:, valid_genes].copy()

    if "spatial_resized" in adata_sub.obsm:
        spatial_key = "spatial_resized"
    elif "spatial" in adata_sub.obsm:
        spatial_key = "spatial"
    else:
        print(f"[{s} | seed={seed}] adata.obsm 中没有对应空间坐标键，跳过。")
        return None

    spatial_coords = np.asarray(adata_sub.obsm[spatial_key])
    adata_sub = set_tesla_pixel_coordinates(
        adata=adata_sub,
        spatial_coords=spatial_coords,
        img_shape_hw=(img_h, img_w),
        auto_fix_xy=AUTO_FIX_SPATIAL_XY,
    )

    contours, _ = cv2.findContours(tissue_mask_tesla, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"[{s} | seed={seed}] tissue mask 中没有找到轮廓，跳过。")
        return None
    cnt = max(contours, key=cv2.contourArea)

    # TESLA 计算
    print(f"[{s} | seed={seed}] 正在进行基因表达超像素增强 (imputation)...")
    try:
        set_global_seed(seed)
        sudo_adata = run_imputation(
            img=img_tesla,
            raw=adata_sub,
            cnt=cnt,
            genes=valid_genes,
            res=TESLA_RES,
        )
    except Exception as e:
        print(f"[{s} | seed={seed}] TESLA imputation 失败: {e}")
        return None

    valid_imputed_genes = []
    for g in valid_genes:
        if g in sudo_adata.var_names:
            g_idx = sudo_adata.var_names.get_loc(g)
            g_vals = np.asarray(sudo_adata.X[:, g_idx]).reshape(-1)
            if np.nanmax(g_vals) > np.nanmin(g_vals):
                valid_imputed_genes.append(g)

    if not valid_imputed_genes:
        print(f"[{s} | seed={seed}] imputation 后没有有效 marker genes，跳过。")
        return None

    print(f"[{s} | seed={seed}] 正在使用 CNN 进行联合分割注释 (annotation)...")
    try:
        set_global_seed(seed)
        pred_refined, tesla_target_clusters, c_m = run_annotation(
            img=img_tesla,
            sudo_adata=sudo_adata,
            genes=valid_imputed_genes,
            resize_factor=RESIZE_FACTOR,
            binary=tissue_mask_tesla,
            res=TESLA_RES,
            min_UMI=0,
            rseed=seed,
            tseed=seed,
            nseed=seed,
        )
    except Exception as e:
        print(f"[{s} | seed={seed}] TESLA annotation 失败: {e}")
        return None

    try:
        pred_refined_map = pred_refined_to_2d_map(
            pred_refined=pred_refined,
            img_shape_hw=(img_h, img_w),
            resize_factor=RESIZE_FACTOR,
        )
    except Exception as e:
        print(f"[{s} | seed={seed}] pred_refined reshape 失败: {e}")
        return None

    pred_refined_resized = resize_for_display(
        pred_refined_map,
        (gt_mask.shape[1], gt_mask.shape[0]),
        is_label=True,
    ).astype(np.int32)

    dynamic_target_clusters = select_target_clusters(
        tesla_target_clusters=tesla_target_clusters,
        c_m=c_m,
        use_otsu=USE_OTSU_CLUSTER_SELECTION,
    )

    current_pred = np.zeros_like(gt_mask, dtype=np.uint8)
    for cid in dynamic_target_clusters:
        current_pred[pred_refined_resized == cid] = 1

    current_pred = (current_pred & tissue_mask_eval).astype(np.uint8)

    valid_pred_raw = pred_refined_resized[tissue_mask_eval > 0]
    valid_gt = gt_mask[tissue_mask_eval > 0]
    if valid_pred_raw.size > 0 and valid_gt.size > 0:
        ari_score = adjusted_rand_score(valid_gt, valid_pred_raw)
    else:
        ari_score = float("nan")

    # 获取 7 项指标
    metrics = calculate_metrics(current_pred, gt_mask, tissue_mask_eval)

    print(f"[{s} | seed={seed}] Evaluation Results:")
    print(
        f"  Target_IoU: {metrics['Target_IoU']:.4f} | "
        f"Dice: {metrics['Dice']:.4f} | "
        f"Recall: {metrics['Recall']:.4f} | "
        f"Precision: {metrics['Precision']:.4f}\n"
        f"  AUPRC: {metrics['AUPRC']:.4f} | "
        f"P-AUROC: {metrics['P-AUROC']:.4f} | "
        f"HD95: {metrics['HD95_str']}px\n"
    )

    # 保存文件
    pred_mask_path = os.path.join(output_dirs["mask"], f"{s}_seed{seed}_tesla_pred_mask.png")
    cv2.imwrite(pred_mask_path, (current_pred * 255).astype(np.uint8))

    cluster_map_path = os.path.join(output_dirs["cluster"], f"{s}_seed{seed}_tesla_cluster_map.png")
    cv2.imwrite(cluster_map_path, pred_refined_resized.astype(np.uint16))

    he_img = resize_for_display(img_vis, (gt_mask.shape[1], gt_mask.shape[0]))
    overlay = he_img.copy()
    cancer_display_mask = (current_pred == 1) & (tissue_mask_eval > 0)
    overlay[cancer_display_mask] = [0, 0, 255]

    alpha = 0.6
    vis_img = cv2.addWeighted(overlay, alpha, he_img, 1 - alpha, 0)
    vis_path = os.path.join(output_dirs["vis"], f"{s}_seed{seed}_tesla_cancer_prediction.jpg")
    cv2.imwrite(vis_path, vis_img)

    return {
        "Slice": s,
        "Seed": seed,
        "ARI": float(ari_score) if not np.isnan(ari_score) else np.nan,
        "Target_IoU": float(metrics["Target_IoU"]),
        "Dice": float(metrics["Dice"]),
        "Recall": float(metrics["Recall"]),
        "Precision": float(metrics["Precision"]),
        "AUPRC": float(metrics["AUPRC"]),
        "P-AUROC": float(metrics["P-AUROC"]),
        "HD95": float(metrics["HD95"]) if not np.isnan(metrics["HD95"]) else np.nan,
    }


def main():
    marker_genes = ["ERBB2", "GATA3", "PIP", "SCGB2A2", "KRT5", "KRT7", "KRT14", "KRT18"]
    dataset_configs = build_three_slice_configs()

    tesla_workspace_dir = r"D:\Project_2\TESLA\tesla_workspace_three_slices"
    vis_root_dir = os.path.join(tesla_workspace_dir, "visualizations")
    mask_root_dir = os.path.join(tesla_workspace_dir, "pred_masks")
    cluster_root_dir = os.path.join(tesla_workspace_dir, "cluster_maps")
    metrics_save_dir = os.path.join(tesla_workspace_dir, "metrics")

    os.makedirs(vis_root_dir, exist_ok=True)
    os.makedirs(mask_root_dir, exist_ok=True)
    os.makedirs(cluster_root_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)

    metric_records = []
    slice_names = [cfg["name"] for cfg in dataset_configs]

    for seed in RANDOM_SEEDS:
        print("\n" + "#" * 100)
        print(f"# 开始随机种子实验: seed={seed}")
        print("#" * 100)

        output_dirs = {
            "vis": os.path.join(vis_root_dir, f"seed_{seed}"),
            "mask": os.path.join(mask_root_dir, f"seed_{seed}"),
            "cluster": os.path.join(cluster_root_dir, f"seed_{seed}"),
            "metrics": metrics_save_dir,
        }
        for d in ["vis", "mask", "cluster"]:
            os.makedirs(output_dirs[d], exist_ok=True)

        for cfg in dataset_configs:
            result = process_one_slice(cfg, marker_genes, output_dirs, seed)
            if result is not None:
                metric_records.append(result)

    if metric_records:
        raw_df, summary_df = build_summary_dataframe(metric_records, slice_names)

        raw_csv = os.path.join(metrics_save_dir, "tesla_three_slices_raw_metrics.csv")
        summary_csv = os.path.join(metrics_save_dir, "tesla_three_slices_summary_metrics.csv")
        table_txt = os.path.join(metrics_save_dir, "tesla_three_slices_summary_table.txt")

        raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

        table_str = summary_dataframe_to_table(summary_df)
        print(table_str)

        with open(table_txt, "w", encoding="utf-8") as f:
            f.write(table_str + "\n")

        print("\n[Summary] 汇总文件已保存:")
        print(f"  raw metrics   -> {raw_csv}")
        print(f"  summary csv   -> {summary_csv}")
        print(f"  summary table -> {table_txt}")
        print(f"[Note] 随机种子列表: {RANDOM_SEEDS}")
    else:
        print("\n没有任何切片成功完成预测，请检查输入路径。")


if __name__ == "__main__":
    main()
import os
import sys
import random
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from medpy.metric.binary import hd95
from sklearn.metrics import adjusted_rand_score

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

# 重要：先默认关闭 Otsu。先确认 TESLA 原始预测可以正确还原为二维区域；
# 如果后续需要测试自适应筛选癌症簇，再改成 True。
USE_OTSU_CLUSTER_SELECTION = False

# 是否自动判断并修正 obsm['spatial'] 的 x/y 顺序。
# TESLA 内部的 pixel_x 表示图像行坐标 row / height，pixel_y 表示列坐标 col / width。
AUTO_FIX_SPATIAL_XY = True

# 多随机种子实验设置。
RANDOM_SEEDS = [42, 123, 567]


# ==========================================
# 工具函数
# ==========================================
def resize_for_display(img, dsize, is_label=False):
    """
    图像缩放工具。
    - 连续图像：缩小时用 INTER_AREA，放大时用 INTER_CUBIC。
    - 标签图 / mask：必须用 INTER_NEAREST，避免类别值被插值混合。
    """
    if is_label:
        return cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)

    src_h, src_w = img.shape[:2]
    dst_w, dst_h = dsize
    shrinking = (dst_w * dst_h) < (src_w * src_h)
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(img, dsize, interpolation=interp)


def resize_binary_mask(mask_bin, target_shape_hw):
    """把二值 mask resize 到指定 H,W，并保持 0/1。"""
    target_h, target_w = target_shape_hw
    if mask_bin.shape[:2] == (target_h, target_w):
        out = mask_bin.copy()
    else:
        out = cv2.resize(mask_bin.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8)



def _probability_histograms(probability_map, gt_mask, tissue_mask, row_chunk_size=256):
    """
    将连续像素分数图量化为 uint16，并分块统计正/负像素分数直方图。
    避免在大尺寸切片上直接展开并排序全部像素造成过高内存占用。
    """
    if probability_map.shape != gt_mask.shape or tissue_mask.shape != gt_mask.shape:
        raise ValueError(
            f"尺寸不一致: probability={probability_map.shape}, "
            f"gt={gt_mask.shape}, tissue={tissue_mask.shape}"
        )

    score_u16 = np.rint(
        np.clip(probability_map, 0.0, 1.0) * 65535.0
    ).astype(np.uint16)

    positive_hist = np.zeros(65536, dtype=np.int64)
    negative_hist = np.zeros(65536, dtype=np.int64)

    valid = tissue_mask > 0
    height = gt_mask.shape[0]

    for row_start in range(0, height, row_chunk_size):
        row_end = min(height, row_start + row_chunk_size)

        scores = score_u16[row_start:row_end].reshape(-1)
        labels = (gt_mask[row_start:row_end] > 0).reshape(-1)
        valid_chunk = valid[row_start:row_end].reshape(-1)

        scores = scores[valid_chunk]
        labels = labels[valid_chunk]

        if scores.size == 0:
            continue

        if np.any(labels):
            positive_hist += np.bincount(
                scores[labels], minlength=65536
            ).astype(np.int64, copy=False)

        if np.any(~labels):
            negative_hist += np.bincount(
                scores[~labels], minlength=65536
            ).astype(np.int64, copy=False)

    return positive_hist, negative_hist


def compute_pixel_auroc(probability_map, gt_mask, tissue_mask):
    """计算 pixel-level AUROC（P-AUROC）。"""
    positive_hist, negative_hist = _probability_histograms(
        probability_map,
        gt_mask,
        tissue_mask,
    )

    positive_count = int(positive_hist.sum())
    negative_count = int(negative_hist.sum())

    if positive_count == 0 or negative_count == 0:
        return np.nan

    # Mann-Whitney U：正样本分数更高记 1 分，相同记 0.5 分。
    negative_below = (
        np.cumsum(negative_hist, dtype=np.int64)
        - negative_hist
    )

    concordant_pairs = np.sum(
        positive_hist.astype(np.float64)
        * (
            negative_below.astype(np.float64)
            + 0.5 * negative_hist.astype(np.float64)
        )
    )

    return float(
        concordant_pairs
        / (positive_count * negative_count)
    )


def compute_pixel_auprc(probability_map, gt_mask, tissue_mask):
    """计算 pixel-level AUPRC / Average Precision。"""
    positive_hist, negative_hist = _probability_histograms(
        probability_map,
        gt_mask,
        tissue_mask,
    )

    positive_count = int(positive_hist.sum())
    negative_count = int(negative_hist.sum())

    if positive_count == 0 or negative_count == 0:
        return np.nan

    # 从高分到低分逐阈值累计 TP/FP。
    tp = np.cumsum(
        positive_hist[::-1],
        dtype=np.int64,
    )
    fp = np.cumsum(
        negative_hist[::-1],
        dtype=np.int64,
    )

    precision = tp.astype(np.float64) / np.maximum(
        tp.astype(np.float64) + fp.astype(np.float64),
        1.0,
    )
    recall = (
        tp.astype(np.float64)
        / float(positive_count)
    )

    previous_recall = np.concatenate(
        ([0.0], recall[:-1])
    )

    return float(
        np.sum(
            (recall - previous_recall)
            * precision
        )
    )


def build_tesla_probability_map(
    cluster_map,
    c_m,
    tissue_mask,
    fallback_mask=None,
):
    """
    根据 TESLA annotation() 返回的 c_m 构建连续像素级癌症分数图。

    说明：
    - 不改变原有 target_clusters / Otsu 的二值癌症簇选择逻辑；
    - 仅把 c_m 中的聚类级得分 min-max 归一化后映射回 cluster_map，
      用于计算 AUPRC 和 P-AUROC；
    - 若 c_m 无法解析或与 cluster id 无法对齐，则退化为二值预测分数图。
    """
    probability = np.zeros(
        cluster_map.shape,
        dtype=np.float32,
    )

    cluster_scores = {}

    try:
        if c_m is not None:
            arr = np.asarray(c_m)

            if arr.ndim == 1:
                values = arr.astype(np.float64)
                ids = np.arange(
                    len(values),
                    dtype=np.int64,
                )

            elif arr.ndim == 2 and arr.shape[1] >= 2:
                ids = np.rint(
                    arr[:, 0].astype(np.float64)
                ).astype(np.int64)
                values = arr[:, 1].astype(np.float64)

            elif arr.ndim == 2 and arr.shape[1] == 1:
                values = arr[:, 0].astype(np.float64)
                ids = np.arange(
                    len(values),
                    dtype=np.int64,
                )

            else:
                ids = np.asarray([], dtype=np.int64)
                values = np.asarray([], dtype=np.float64)

            for cid, score in zip(ids, values):
                if np.isfinite(score):
                    cluster_scores[int(cid)] = float(score)

    except Exception as error:
        print(f"  -> c_m 连续得分解析失败，将退化为二值分数图: {error}")
        cluster_scores = {}

    if cluster_scores:
        score_values = np.asarray(
            list(cluster_scores.values()),
            dtype=np.float64,
        )
        score_min = float(np.min(score_values))
        score_max = float(np.max(score_values))

        if score_max > score_min:
            normalized_scores = {
                cid: float(
                    (score - score_min)
                    / (score_max - score_min)
                )
                for cid, score in cluster_scores.items()
            }
        else:
            normalized_scores = {
                cid: 0.5
                for cid in cluster_scores
            }

        mapped_pixel_count = 0
        for cid, score in normalized_scores.items():
            cluster_pixels = cluster_map == cid
            count = int(np.sum(cluster_pixels))
            if count > 0:
                probability[cluster_pixels] = score
                mapped_pixel_count += count

        if mapped_pixel_count == 0:
            print(
                "  -> c_m 中的 cluster id 与 pred_refined cluster map "
                "未成功对齐，AUPRC/P-AUROC 将退化为二值分数图。"
            )
            cluster_scores = {}

    if not cluster_scores:
        if fallback_mask is not None:
            probability = (
                fallback_mask > 0
            ).astype(np.float32)

    probability[tissue_mask == 0] = 0.0
    return probability


def calculate_metrics(
    pred_mask,
    probability_map,
    gt_mask,
    tissue_mask,
):
    """
    统一计算 7 个指标：
    Target_IoU、Dice、Recall、Precision、AUPRC、P-AUROC、HD95。
    """
    valid = tissue_mask > 0
    pred = (pred_mask > 0) & valid
    gt = (gt_mask > 0) & valid

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))

    smooth = 1e-8
    target_iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)
    recall = tp / (tp + fn + smooth)
    precision = tp / (tp + fp + smooth)

    auprc = compute_pixel_auprc(
        probability_map,
        gt_mask,
        tissue_mask,
    )
    p_auroc = compute_pixel_auroc(
        probability_map,
        gt_mask,
        tissue_mask,
    )

    try:
        if np.sum(pred) > 0 and np.sum(gt) > 0:
            hd95_val = float(hd95(pred, gt))
        else:
            hd95_val = np.nan
    except Exception:
        hd95_val = np.nan

    hd_str = (
        "NaN"
        if np.isnan(hd95_val)
        else f"{hd95_val:.4f}"
    )

    return {
        "Target_IoU": float(target_iou),
        "Dice": float(dice),
        "Recall": float(recall),
        "Precision": float(precision),
        "AUPRC": float(auprc),
        "P-AUROC": float(p_auroc),
        "HD95": hd95_val,
        "HD95_str": hd_str,
    }


def ensure_bgr_image(img_path):
    """读取 3 通道 BGR 图像，并保证内存连续。"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"图像读取失败: {img_path}")
    return np.ascontiguousarray(img)


def ensure_gray_image(img_path, name="image"):
    """读取灰度图。"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"{name} 读取失败: {img_path}")
    return img


def set_tesla_pixel_coordinates(adata, spatial_coords, img_shape_hw, auto_fix_xy=True):
    """
    设置 TESLA 需要的 pixel_x / pixel_y。

    注意：TESLA 的 imputation.py 中会使用 image[x_pixel, y_pixel] 访问像素，
    因此 TESLA 语义是：
        pixel_x = row = height 方向坐标
        pixel_y = col = width 方向坐标

    但多数 AnnData 的 obsm['spatial'] 是常规图像坐标：
        spatial[:, 0] = x = col = width 方向
        spatial[:, 1] = y = row = height 方向

    所以这里默认自动判断是否需要交换。
    """
    img_h, img_w = img_shape_hw
    c0 = np.asarray(spatial_coords[:, 0], dtype=float)
    c1 = np.asarray(spatial_coords[:, 1], dtype=float)

    def in_range(vals, upper):
        return np.nanmin(vals) >= -5 and np.nanmax(vals) <= upper + 5

    if auto_fix_xy:
        # 情况 A：spatial[:,0] 像 row，spatial[:,1] 像 col，说明已经是 TESLA 需要的顺序。
        looks_like_row_col = in_range(c0, img_h) and in_range(c1, img_w)
        # 情况 B：spatial[:,0] 像 col，spatial[:,1] 像 row，说明需要交换。
        looks_like_col_row = in_range(c0, img_w) and in_range(c1, img_h)

        if looks_like_col_row and not looks_like_row_col:
            adata.obs["pixel_x"] = c1  # row / y / height
            adata.obs["pixel_y"] = c0  # col / x / width
            print("  -> spatial 坐标判断为常规 (x,y)=(col,row)，已交换为 TESLA 的 pixel_x=row, pixel_y=col")
        else:
            # 如果两种判断都成立，说明图像接近正方形或坐标范围不够区分；保持原顺序并打印提示。
            adata.obs["pixel_x"] = c0
            adata.obs["pixel_y"] = c1
            if looks_like_col_row and looks_like_row_col:
                print("  -> spatial 坐标范围较难区分 x/y，暂保持原顺序；若结果整体旋转/错位，请手动交换 pixel_x/pixel_y")
            else:
                print("  -> spatial 坐标按原顺序写入 TESLA: pixel_x=spatial[:,0], pixel_y=spatial[:,1]")
    else:
        adata.obs["pixel_x"] = c0
        adata.obs["pixel_y"] = c1

    print(
        f"  -> image H,W=({img_h},{img_w}); "
        f"pixel_x range=({adata.obs['pixel_x'].min():.1f},{adata.obs['pixel_x'].max():.1f}); "
        f"pixel_y range=({adata.obs['pixel_y'].min():.1f},{adata.obs['pixel_y'].max():.1f})"
    )

    return adata


def pred_refined_to_2d_map(pred_refined, img_shape_hw, resize_factor):
    """
    将 TESLA annotation() 返回的 pred_refined 转成二维标签图。

    这是本脚本最关键的修复点：
    TESLA.annotation() 内部返回的 pred_refined 是 flatten 后的一维数组，长度为：
        int(img_h * resize_factor) * int(img_w * resize_factor)
    不能直接 cv2.resize 一维数组，否则 OpenCV 会把它当成单列/单行图像，产生条纹状预测。
    """
    img_h, img_w = img_shape_hw
    resize_w = int(img_w * resize_factor)
    resize_h = int(img_h * resize_factor)

    pred_arr = np.asarray(pred_refined).astype(np.int32)
    expected_size = resize_h * resize_w

    if pred_arr.ndim == 2:
        pred_map = pred_arr
    elif pred_arr.ndim == 1:
        if pred_arr.size != expected_size:
            raise ValueError(
                f"pred_refined 长度不匹配，无法 reshape: got {pred_arr.size}, "
                f"expected {expected_size} = {resize_h} * {resize_w}. "
                f"请检查 annotation 的 resize_factor 是否与这里一致。"
            )
        pred_map = pred_arr.reshape(resize_h, resize_w)
    else:
        raise ValueError(f"pred_refined 维度异常: ndim={pred_arr.ndim}, shape={pred_arr.shape}")

    return pred_map.astype(np.int32)


def select_target_clusters(tesla_target_clusters, c_m, use_otsu=False):
    """
    选择最终癌症簇。

    默认直接使用 TESLA.annotation() 根据 marker 表达返回的 target_clusters。
    Otsu 是额外后处理，可能把条带状高表达噪声簇也选进去，因此默认关闭。
    """
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
    print(f"  -> Otsu 自适应表达阈值: {real_thresh:.4f}")

    selected = cluster_ids[scores >= real_thresh].tolist()
    return [int(x) for x in selected]


def _safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def _safe_nanstd(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    # 与示例表格一致：这里使用总体标准差 ddof=0，而不是样本标准差 ddof=1。
    return float(np.nanstd(values, ddof=0))


def _format_mean_std(mean_val, std_val, digits=4):
    if np.isnan(mean_val):
        return "NaN"
    if np.isnan(std_val):
        return f"{mean_val:.{digits}f}±NaN"
    return f"{mean_val:.{digits}f}±{std_val:.{digits}f}"


def build_summary_dataframe(records, slice_order):
    """
    根据逐切片/逐运行记录生成最终汇总表。

    - 如果每个切片只运行 1 次，则每个切片的 std 为 0。
    - Average 行的 mean 是所有切片均值的平均。
    - Average 行的 std 是所有切片均值之间的总体标准差，与用户示例表格一致。
    """
    metric_cols = [
        "Target_IoU",
        "Dice",
        "Recall",
        "Precision",
        "AUPRC",
        "P-AUROC",
        "HD95",
    ]
    raw_df = pd.DataFrame(records)

    if raw_df.empty:
        return raw_df, pd.DataFrame()

    rows = []
    for s in slice_order:
        part = raw_df[raw_df["Slice"] == s]
        if part.empty:
            continue

        row = {"Slice": s}
        for m in metric_cols:
            vals = part[m].astype(float).to_numpy()
            row[f"{m}_mean"] = _safe_nanmean(vals)
            row[f"{m}_std"] = _safe_nanstd(vals)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        avg_row = {"Slice": "Average"}
        for m in metric_cols:
            slice_means = summary_df[f"{m}_mean"].astype(float).to_numpy()
            avg_row[f"{m}_mean"] = _safe_nanmean(slice_means)
            avg_row[f"{m}_std"] = _safe_nanstd(slice_means)
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)

    return raw_df, summary_df



def summary_dataframe_to_table(summary_df, method_name="TESLA"):
    """把 7 个指标的多随机种子汇总结果转换为控制台表格。"""
    if summary_df.empty:
        return "\n[Summary] 没有成功完成的切片，无法生成汇总表。\n"

    metric_specs = [
        ("Target_IoU", "Target_IoU", 4),
        ("Dice", "Dice", 4),
        ("Recall", "Recall", 4),
        ("Precision", "Precision", 4),
        ("AUPRC", "AUPRC", 4),
        ("P-AUROC", "P-AUROC", 4),
        ("HD95", "HD95 (px)", 2),
    ]

    display_rows = []

    for _, row in summary_df.iterrows():
        display_row = {
            "Slice": str(row["Slice"])
        }

        for metric_key, display_name, digits in metric_specs:
            display_row[display_name] = _format_mean_std(
                row[f"{metric_key}_mean"],
                row[f"{metric_key}_std"],
                digits=digits,
            )

        display_rows.append(display_row)

    widths = {
        "Slice": max(
            8,
            max(len(r["Slice"]) for r in display_rows),
        )
    }

    for _, display_name, _ in metric_specs:
        widths[display_name] = max(
            14 if display_name == "HD95 (px)" else 17,
            max(
                len(r[display_name])
                for r in display_rows
            ),
        )

    def border_line():
        parts = [
            "-" * (widths["Slice"] + 2)
        ]
        for _, display_name, _ in metric_specs:
            parts.append(
                "-" * (widths[display_name] + 2)
            )
        return "|" + "|".join(parts) + "|"

    def row_line(row):
        parts = [
            f" {row['Slice']:<{widths['Slice']}} "
        ]
        for _, display_name, _ in metric_specs:
            parts.append(
                f" {row[display_name]:<{widths[display_name]}} "
            )
        return "|" + "|".join(parts) + "|"

    title = (
        f"📋 {method_name} - 切片级多随机种子统计汇总表 "
        f"(均值±标准差)"
    )
    table_width = max(
        175,
        len(title) + 4,
    )
    sep = "=" * table_width

    header_row = {
        "Slice": "Slice"
    }
    for _, display_name, _ in metric_specs:
        header_row[display_name] = display_name

    lines = [
        "",
        sep,
        title,
        sep,
        row_line(header_row),
        border_line(),
    ]

    for row in display_rows:
        if row["Slice"] == "Average":
            lines.append(
                border_line()
            )
        lines.append(
            row_line(row)
        )

    lines.append(sep)
    return "\n".join(lines)


def save_and_print_summary(records, slice_order, metrics_save_dir, method_name="TESLA"):
    """保存 raw metrics、summary csv、summary txt，并打印最终汇总表。"""
    os.makedirs(metrics_save_dir, exist_ok=True)

    raw_df, summary_df = build_summary_dataframe(records, slice_order)

    raw_csv = os.path.join(metrics_save_dir, f"{method_name.lower()}_raw_metrics.csv")
    summary_csv = os.path.join(metrics_save_dir, f"{method_name.lower()}_summary_metrics.csv")
    table_txt = os.path.join(metrics_save_dir, f"{method_name.lower()}_summary_table.txt")

    if not raw_df.empty:
        raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    if not summary_df.empty:
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    table_str = summary_dataframe_to_table(summary_df, method_name=method_name)
    print(table_str)

    with open(table_txt, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    print("\n[Summary] 汇总文件已保存:")
    print(f"  raw metrics   -> {raw_csv}")
    print(f"  summary csv   -> {summary_csv}")
    print(f"  summary table -> {table_txt}")




# ==========================================
# 多随机种子统计辅助函数
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


def _safe_nanvar(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    # 使用总体方差 ddof=0，与表格中的总体标准差对应。
    return float(np.nanvar(values, ddof=0))


def build_summary_dataframe(records, slice_order):
    """
    根据逐切片/逐随机种子记录生成最终汇总表。

    - 每个切片的 mean/std/var 来自该切片在 RANDOM_SEEDS 下的多次运行。
    - Average 行的 mean 是所有切片 mean 的平均。
    - Average 行的 std/var 是所有切片 mean 之间的总体标准差/总体方差，
      与用户给出的 iStar 示例表格中的 Average 计算方式一致。
    """
    metric_cols = [
        "Target_IoU",
        "Dice",
        "Recall",
        "Precision",
        "AUPRC",
        "P-AUROC",
        "HD95",
    ]
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
            row[f"{m}_std"] = _safe_nanstd(vals)
            row[f"{m}_var"] = _safe_nanvar(vals)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        avg_row = {"Slice": "Average", "Num_Seeds": int(raw_df["Seed"].nunique()) if "Seed" in raw_df.columns else np.nan}
        for m in metric_cols:
            slice_means = summary_df[f"{m}_mean"].astype(float).to_numpy()
            avg_row[f"{m}_mean"] = _safe_nanmean(slice_means)
            avg_row[f"{m}_std"] = _safe_nanstd(slice_means)
            avg_row[f"{m}_var"] = _safe_nanvar(slice_means)
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)

    return raw_df, summary_df


def main():
    slices = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]
    marker_genes = ["ERBB2", "GATA3", "PIP", "SCGB2A2", "KRT5", "KRT7", "KRT14", "KRT18"]

    img_base = r"D:\Project_2\Pixel_Seg\data\STHBC\resized_data"
    adata_base = r"D:\Project_2\Pixel_Seg\data\STHBC\resized_data"
    gt_base = r"D:\Project_2\Pixel_Seg\data\STHBC\resized_data"
    mask_base = r"D:\Project_2\Pixel_Seg\data\STHBC\resized_data"

    tesla_workspace_dir = r"D:\Project_2\TESLA\tesla_workspace_multiseed"
    vis_root_dir = os.path.join(tesla_workspace_dir, "visualizations")
    mask_root_dir = os.path.join(tesla_workspace_dir, "pred_masks")
    cluster_root_dir = os.path.join(tesla_workspace_dir, "cluster_maps")
    metrics_save_dir = os.path.join(tesla_workspace_dir, "metrics")
    os.makedirs(vis_root_dir, exist_ok=True)
    os.makedirs(mask_root_dir, exist_ok=True)
    os.makedirs(cluster_root_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)

    metric_records = []

    for seed in RANDOM_SEEDS:
        set_global_seed(seed)
        print("\n" + "#" * 100)
        print(f"# 开始随机种子实验: seed={seed}")
        print("#" * 100)

        vis_save_dir = os.path.join(vis_root_dir, f"seed_{seed}")
        mask_save_dir = os.path.join(mask_root_dir, f"seed_{seed}")
        cluster_save_dir = os.path.join(cluster_root_dir, f"seed_{seed}")
        os.makedirs(vis_save_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)
        os.makedirs(cluster_save_dir, exist_ok=True)

        for s in slices:
            print(f"\n========== 开始处理切片: {s} | Seed: {seed} ==========")
            img_path = os.path.join(img_base, s, "he_resized.png")
            adata_path = os.path.join(adata_base, s, "adata_resized.h5ad")
            gt_path = os.path.join(gt_base, s, "label_resized.png")
            mask_path = os.path.join(mask_base, s, "mask_resized.png")

            try:
                img_raw = ensure_bgr_image(img_path)
                img_tesla = img_raw.copy()
                img_vis = img_raw.copy()

                adata = sc.read_h5ad(adata_path)
                adata.var_names_make_unique()
                gt_img = ensure_gray_image(gt_path, name="GT label")
                tissue_mask_raw = ensure_gray_image(mask_path, name="Tissue mask")
            except Exception as e:
                print(f"[{s} | seed={seed}] 文件读取失败: {e}")
                continue

            img_h, img_w = img_tesla.shape[:2]
            gt_mask = (gt_img > 127).astype(np.uint8)
            tissue_mask_raw_bin = (tissue_mask_raw > 127).astype(np.uint8)

            # TESLA 计算需要和 H&E 原图同尺寸的 tissue mask。
            tissue_mask_tesla = resize_binary_mask(tissue_mask_raw_bin, (img_h, img_w))
            # 评估需要和 gt 同尺寸的 tissue mask。
            tissue_mask_eval = resize_binary_mask(tissue_mask_raw_bin, gt_mask.shape[:2])

            valid_genes = [g for g in marker_genes if g in adata.var_names]
            if not valid_genes:
                print(f"[{s} | seed={seed}] 未找到有效 marker genes，跳过。")
                continue

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
                continue

            valid_genes = really_valid_genes
            adata_sub = adata_sub[:, valid_genes].copy()

            if "spatial_resized" in adata_sub.obsm:
                spatial_key = "spatial_resized"
            elif "spatial" in adata_sub.obsm:
                spatial_key = "spatial"
            else:
                print(f"[{s} | seed={seed}] adata.obsm 中没有 'spatial_resized' 或 'spatial'，跳过。")
                continue

            spatial_coords = np.asarray(adata_sub.obsm[spatial_key])
            print(f"  -> 使用空间坐标: adata.obsm['{spatial_key}']")
            adata_sub = set_tesla_pixel_coordinates(
                adata=adata_sub,
                spatial_coords=spatial_coords,
                img_shape_hw=(img_h, img_w),
                auto_fix_xy=AUTO_FIX_SPATIAL_XY,
            )

            # 用 H&E 同尺寸的二值 tissue mask 找轮廓。
            contours, _ = cv2.findContours(tissue_mask_tesla, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print(f"[{s} | seed={seed}] tissue mask 中没有找到轮廓，跳过。")
                continue
            cnt = max(contours, key=cv2.contourArea)

            # ---------------------------------------------------------
            # TESLA 核心推理流程
            # ---------------------------------------------------------
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
                continue

            valid_imputed_genes = []
            for g in valid_genes:
                if g in sudo_adata.var_names:
                    g_idx = sudo_adata.var_names.get_loc(g)
                    g_vals = np.asarray(sudo_adata.X[:, g_idx]).reshape(-1)
                    if np.nanmax(g_vals) > np.nanmin(g_vals):
                        valid_imputed_genes.append(g)

            if not valid_imputed_genes:
                print(f"[{s} | seed={seed}] imputation 后没有有效 marker genes，跳过。")
                continue

            print(f"[{s} | seed={seed}] 正在使用 CNN 进行联合分割注释 (annotation)...")
            try:
                # TESLA.annotation() 返回顺序为：pred_refined, target_clusters, c_m。
                # 这里显式传入 rseed/tseed/nseed，使三次实验真正由 [42,123,567] 控制。
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
                continue

            # ---------------------------------------------------------
            # 关键修复：pred_refined 一维 flatten 标签 -> 二维标签图 -> 上采样到 GT 尺寸
            # ---------------------------------------------------------
            try:
                pred_refined_map = pred_refined_to_2d_map(
                    pred_refined=pred_refined,
                    img_shape_hw=(img_h, img_w),
                    resize_factor=RESIZE_FACTOR,
                )
            except Exception as e:
                print(f"[{s} | seed={seed}] pred_refined reshape 失败: {e}")
                continue

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
            print(f"  -> TESLA 原始 target clusters: {tesla_target_clusters}")
            print(f"  -> 最终选定癌症簇 IDs: {dynamic_target_clusters}")

            # ---------------------------------------------------------
            # 生成最终二值预测 mask
            # ---------------------------------------------------------
            current_pred = np.zeros_like(gt_mask, dtype=np.uint8)
            for cid in dynamic_target_clusters:
                current_pred[pred_refined_resized == cid] = 1

            # 背景区域清零。
            current_pred = (current_pred & tissue_mask_eval).astype(np.uint8)

            # ---------------------------------------------------------
            # 使用 TESLA 的 c_m 聚类得分构造连续像素级分数图。
            # 仅用于 AUPRC / P-AUROC，不改变原有二值癌症簇选择。
            # ---------------------------------------------------------
            probability_map = build_tesla_probability_map(
                cluster_map=pred_refined_resized,
                c_m=c_m,
                tissue_mask=tissue_mask_eval,
                fallback_mask=current_pred,
            )

            # 聚类 ARI：注意这里是 raw cluster label 与二值 GT 的一致性，仅用于观察边界质量。
            valid_pred_raw = pred_refined_resized[tissue_mask_eval > 0]
            valid_gt = gt_mask[tissue_mask_eval > 0]
            if valid_pred_raw.size > 0 and valid_gt.size > 0:
                ari_score = adjusted_rand_score(valid_gt, valid_pred_raw)
            else:
                ari_score = float("nan")

            metrics = calculate_metrics(
                current_pred,
                probability_map,
                gt_mask,
                tissue_mask_eval,
            )
            hd_str = metrics["HD95_str"]
            auprc_str = (
                f"{metrics['AUPRC']:.4f}"
                if np.isfinite(metrics['AUPRC'])
                else "NaN"
            )
            pauroc_str = (
                f"{metrics['P-AUROC']:.4f}"
                if np.isfinite(metrics['P-AUROC'])
                else "NaN"
            )

            print(f"[{s} | seed={seed}] Evaluation Results:")
            print(f"  Unsupervised ARI: {ari_score:.4f} (聚类边界质量，仅供参考)")
            print(
                f"  Target_IoU: {metrics['Target_IoU']:.4f} | "
                f"Dice: {metrics['Dice']:.4f} | "
                f"Recall: {metrics['Recall']:.4f} | "
                f"Precision: {metrics['Precision']:.4f} | "
                f"AUPRC: {auprc_str} | "
                f"P-AUROC: {pauroc_str} | "
                f"HD95: {hd_str}px\n"
            )

            # 收集逐切片/逐随机种子结果，用于最后打印整个数据集汇总表。
            metric_records.append({
                "Slice": s,
                "Seed": seed,
                "ARI": float(ari_score) if not np.isnan(ari_score) else np.nan,
                "Target_IoU": float(metrics["Target_IoU"]),
                "Dice": float(metrics["Dice"]),
                "Recall": float(metrics["Recall"]),
                "Precision": float(metrics["Precision"]),
                "AUPRC": float(metrics["AUPRC"]) if np.isfinite(metrics["AUPRC"]) else np.nan,
                "P-AUROC": float(metrics["P-AUROC"]) if np.isfinite(metrics["P-AUROC"]) else np.nan,
                "HD95": float(metrics["HD95"]) if not np.isnan(metrics["HD95"]) else np.nan,
            })

            # ---------------------------------------------------------
            # 保存预测 mask、cluster map 和红色叠加可视化
            # ---------------------------------------------------------
            pred_mask_path = os.path.join(mask_save_dir, f"{s}_seed{seed}_tesla_pred_mask.png")
            cv2.imwrite(pred_mask_path, (current_pred * 255).astype(np.uint8))

            # cluster map 保存为 16-bit，避免簇 ID 被截断；用于后续 debug。
            cluster_map_path = os.path.join(cluster_save_dir, f"{s}_seed{seed}_tesla_cluster_map.png")
            cv2.imwrite(cluster_map_path, pred_refined_resized.astype(np.uint16))

            he_img = resize_for_display(img_vis, (gt_mask.shape[1], gt_mask.shape[0]))
            overlay = he_img.copy()
            cancer_display_mask = (current_pred == 1) & (tissue_mask_eval > 0)
            overlay[cancer_display_mask] = [0, 0, 255]

            alpha = 0.6
            vis_img = cv2.addWeighted(overlay, alpha, he_img, 1 - alpha, 0)
            vis_path = os.path.join(vis_save_dir, f"{s}_seed{seed}_tesla_cancer_prediction.jpg")
            cv2.imwrite(vis_path, vis_img)

            print(f"[{s} | seed={seed}] 已保存:")
            print(f"  mask       -> {pred_mask_path}")
            print(f"  cluster map-> {cluster_map_path}")
            print(f"  overlay    -> {vis_path}")

    # ---------------------------------------------------------
    # 整个数据集最终汇总表
    # ---------------------------------------------------------
    save_and_print_summary(
        records=metric_records,
        slice_order=slices,
        metrics_save_dir=metrics_save_dir,
        method_name="TESLA",
    )

    print("\n[Note] 控制台表格显示的是 mean±std；summary CSV 额外包含 *_var 方差列。")
    print(f"[Note] 随机种子列表: {RANDOM_SEEDS}")


if __name__ == "__main__":
    main()

import os
import sys
import cv2
import numpy as np
import scanpy as sc
from medpy.metric.binary import hd95
from sklearn.metrics import adjusted_rand_score

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


def calculate_metrics(pred_mask, gt_mask, tissue_mask):
    """计算评估指标：只在 tissue_mask 区域内计算。"""
    pred = (pred_mask > 0) & (tissue_mask > 0)
    gt = (gt_mask > 0) & (tissue_mask > 0)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)

    smooth = 1e-8
    iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)
    recall = tp / (tp + fn + smooth)

    try:
        if np.sum(pred) > 0 and np.sum(gt) > 0:
            hd95_val = hd95(pred, gt)
            hd_str = f"{hd95_val:.4f}"
        else:
            hd_str = "NaN"
    except Exception:
        hd_str = "NaN"

    return {"Target_IoU": iou, "Dice": dice, "Recall": recall, "HD95_str": hd_str}


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


def main():
    slices = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]
    marker_genes = ["ERBB2", "GATA3", "PIP", "SCGB2A2", "KRT5", "KRT7", "KRT14", "KRT18"]

    img_base = r"D:\Project_2\Pixel_Seg\data\STHBC\resized_data"
    adata_base = r"D:\Project_2\Pixel_Seg\data\STHBC\resized_data"
    gt_base = r"D:\Project_2\Pixel_Seg\data\STHBC\resized_data"
    mask_base = r"D:\Project_2\Pixel_Seg\data\STHBC\resized_data"

    tesla_workspace_dir = r"D:\Project_2\TESLA\tesla_workspace"
    vis_save_dir = os.path.join(tesla_workspace_dir, "visualizations")
    mask_save_dir = os.path.join(tesla_workspace_dir, "pred_masks")
    cluster_save_dir = os.path.join(tesla_workspace_dir, "cluster_maps")
    os.makedirs(vis_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(cluster_save_dir, exist_ok=True)

    for s in slices:
        print(f"\n========== 开始处理切片: {s} ==========")
        img_path = os.path.join(img_base, s, "he_resized.png")
        adata_path = os.path.join(adata_base, s, f"adata_resized.h5ad")
        gt_path = os.path.join(gt_base, s, f"label_resized.png")
        mask_path = os.path.join(mask_base, s, f"mask_resized.png")



        try:
            img_raw = ensure_bgr_image(img_path)
            img_tesla = img_raw.copy()
            img_vis = img_raw.copy()

            adata = sc.read_h5ad(adata_path)
            adata.var_names_make_unique()
            gt_img = ensure_gray_image(gt_path, name="GT label")
            tissue_mask_raw = ensure_gray_image(mask_path, name="Tissue mask")
        except Exception as e:
            print(f"[{s}] 文件读取失败: {e}")
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
            print(f"[{s}] 未找到有效 marker genes，跳过。")
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
            print(f"[{s}] marker genes 在该切片中均为常数表达，跳过。")
            continue

        valid_genes = really_valid_genes
        adata_sub = adata_sub[:, valid_genes].copy()

        if "spatial" not in adata_sub.obsm:
            print(f"[{s}] adata.obsm 中没有 'spatial'，跳过。")
            continue

        spatial_coords = np.asarray(adata_sub.obsm["spatial_resized"])
        adata_sub = set_tesla_pixel_coordinates(
            adata=adata_sub,
            spatial_coords=spatial_coords,
            img_shape_hw=(img_h, img_w),
            auto_fix_xy=AUTO_FIX_SPATIAL_XY,
        )

        # 用 H&E 同尺寸的二值 tissue mask 找轮廓。
        contours, _ = cv2.findContours(tissue_mask_tesla, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f"[{s}] tissue mask 中没有找到轮廓，跳过。")
            continue
        cnt = max(contours, key=cv2.contourArea)

        # ---------------------------------------------------------
        # TESLA 核心推理流程
        # ---------------------------------------------------------
        print(f"[{s}] 正在进行基因表达超像素增强 (imputation)...")
        try:
            sudo_adata = run_imputation(
                img=img_tesla,
                raw=adata_sub,
                cnt=cnt,
                genes=valid_genes,
                res=TESLA_RES,
            )
        except Exception as e:
            print(f"[{s}] TESLA imputation 失败: {e}")
            continue

        valid_imputed_genes = []
        for g in valid_genes:
            if g in sudo_adata.var_names:
                g_idx = sudo_adata.var_names.get_loc(g)
                g_vals = np.asarray(sudo_adata.X[:, g_idx]).reshape(-1)
                if np.nanmax(g_vals) > np.nanmin(g_vals):
                    valid_imputed_genes.append(g)

        if not valid_imputed_genes:
            print(f"[{s}] imputation 后没有有效 marker genes，跳过。")
            continue

        print(f"[{s}] 正在使用 CNN 进行联合分割注释 (annotation)...")
        try:
            # TESLA.annotation() 返回顺序为：pred_refined, target_clusters, c_m。
            pred_refined, tesla_target_clusters, c_m = run_annotation(
                img=img_tesla,
                sudo_adata=sudo_adata,
                genes=valid_imputed_genes,
                resize_factor=RESIZE_FACTOR,
                binary=tissue_mask_tesla,
                res=TESLA_RES,
                min_UMI=0,
            )
        except Exception as e:
            print(f"[{s}] TESLA annotation 失败: {e}")
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
            print(f"[{s}] pred_refined reshape 失败: {e}")
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

        # 聚类 ARI：注意这里是 raw cluster label 与二值 GT 的一致性，仅用于观察边界质量。
        valid_pred_raw = pred_refined_resized[tissue_mask_eval > 0]
        valid_gt = gt_mask[tissue_mask_eval > 0]
        if valid_pred_raw.size > 0 and valid_gt.size > 0:
            ari_score = adjusted_rand_score(valid_gt, valid_pred_raw)
        else:
            ari_score = float("nan")

        metrics = calculate_metrics(current_pred, gt_mask, tissue_mask_eval)
        hd_str = metrics["HD95_str"]

        print(f"[{s}] Evaluation Results:")
        print(f"  Unsupervised ARI: {ari_score:.4f} (聚类边界质量，仅供参考)")
        print(
            f"  Target_IoU: {metrics['Target_IoU']:.4f} | "
            f"Dice: {metrics['Dice']:.4f} | "
            f"Recall: {metrics['Recall']:.4f} | "
            f"HD95: {hd_str}px\n"
        )

        # ---------------------------------------------------------
        # 保存预测 mask、cluster map 和红色叠加可视化
        # ---------------------------------------------------------
        pred_mask_path = os.path.join(mask_save_dir, f"{s}_tesla_pred_mask.png")
        cv2.imwrite(pred_mask_path, (current_pred * 255).astype(np.uint8))

        # cluster map 保存为 16-bit，避免簇 ID 被截断；用于后续 debug。
        cluster_map_path = os.path.join(cluster_save_dir, f"{s}_tesla_cluster_map.png")
        cv2.imwrite(cluster_map_path, pred_refined_resized.astype(np.uint16))

        he_img = resize_for_display(img_vis, (gt_mask.shape[1], gt_mask.shape[0]))
        overlay = he_img.copy()
        cancer_display_mask = (current_pred == 1) & (tissue_mask_eval > 0)
        overlay[cancer_display_mask] = [0, 0, 255]

        alpha = 0.6
        vis_img = cv2.addWeighted(overlay, alpha, he_img, 1 - alpha, 0)
        vis_path = os.path.join(vis_save_dir, f"{s}_tesla_cancer_prediction.jpg")
        cv2.imwrite(vis_path, vis_img)

        print(f"[{s}] 已保存:")
        print(f"  mask       -> {pred_mask_path}")
        print(f"  cluster map-> {cluster_map_path}")
        print(f"  overlay    -> {vis_path}")


if __name__ == "__main__":
    main()
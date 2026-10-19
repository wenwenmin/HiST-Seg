import os
import cv2
import pickle
import numpy as np
from medpy.metric.binary import hd95
from sklearn.metrics import adjusted_rand_score


def calculate_metrics(pred_mask, gt_mask, tissue_mask):
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
        hd95_val = hd95(pred, gt) if (np.sum(pred) > 0 and np.sum(gt) > 0) else np.nan
    except Exception:
        hd95_val = np.nan

    return {'Target_IoU': iou, 'Dice': dice, 'Recall': recall, 'HD95': hd95_val}


def main():
    slides = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    img_base = r"D:\DNM_project(cancer detect)\data\HB\her2st"
    gt_base = r"D:\Project_2\Pixel_Seg\data\STHBC\label_img"
    mask_base = r"D:\Project_2\Pixel_Seg\data\STHBC\mask"

    workspace_dir = r"D:\Project_2\TESLA\tesla_workspace"
    vis_save_dir = os.path.join(workspace_dir, "visualizations")
    os.makedirs(vis_save_dir, exist_ok=True)

    all_results = {slide: {} for slide in slides}

    for s in slides:
        print(f"\n================ Processing Slide: {s} ================")
        prefix = os.path.join(workspace_dir, f"{s}_tesla")
        pred_save_path = os.path.join(prefix, "pred_refined.npy")
        cm_save_path = os.path.join(prefix, "c_m.pickle")

        if not (os.path.exists(pred_save_path) and os.path.exists(cm_save_path)):
            print(f"  [!] 找不到 {s} 的 TESLA 缓存结果，跳过。")
            continue

        # 1. 读取干净的原图和标签
        img_path = os.path.join(img_base, s, "he-raw.jpg")
        gt_path = os.path.join(gt_base, f"{s}_label.png")
        mask_path = os.path.join(mask_base, f"{s}_hest.png")

        he_img = cv2.imread(img_path)  # 此时读取的图像绝对干净
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        tissue_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        gt_mask = (gt_img > 127).astype(np.uint8)
        tissue_mask_bin = (tissue_mask > 127).astype(np.uint8)

        # 2. 读取 TESLA 离线聚类结果
        pred_refined = np.load(pred_save_path).astype(np.int32)
        with open(cm_save_path, 'rb') as f:
            c_m = pickle.load(f)

        pred_refined_resized = cv2.resize(pred_refined, (gt_mask.shape[1], gt_mask.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

        # 3. 防御性解析 c_m 并进行 Otsu 自适应阈值筛选
        dynamic_target_clusters = []
        if c_m is not None and len(c_m) > 0:
            c_m_arr = np.array(c_m)
            if c_m_arr.ndim == 1:
                cluster_ids, scores = np.arange(len(c_m_arr)), c_m_arr
            elif c_m_arr.ndim == 2:
                cluster_ids, scores = c_m_arr[:, 0].astype(int), c_m_arr[:, 1]
            else:
                cluster_ids, scores = np.array([]), np.array([])

            if len(scores) > 0 and scores.max() > scores.min():
                scores_norm = np.uint8(255 * (scores - scores.min()) / (scores.max() - scores.min()))
                thresh_norm, _ = cv2.threshold(scores_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                real_thresh = scores.min() + (thresh_norm / 255.0) * (scores.max() - scores.min())
                dynamic_target_clusters = cluster_ids[scores >= real_thresh].tolist()

        print(f"  -> 最终选定癌症簇 IDs: {dynamic_target_clusters}")

        # 4. 生成二值化预测掩码
        current_pred = np.zeros_like(gt_mask)
        for cid in dynamic_target_clusters:
            current_pred[pred_refined_resized == cid] = 1
        current_pred = current_pred & tissue_mask_bin

        # 5. 计算指标
        metrics = calculate_metrics(current_pred, gt_mask, tissue_mask_bin)
        all_results[s] = metrics

        hd_str = f"{metrics['HD95']:.2f}" if not np.isnan(metrics['HD95']) else "NaN"
        print(
            f"  Target_IoU: {metrics['Target_IoU']:.4f} | Dice: {metrics['Dice']:.4f} | Recall: {metrics['Recall']:.4f} | HD95: {hd_str}")

        # 6. 纯净可视化
        if he_img is not None:
            he_img = cv2.resize(he_img, (gt_mask.shape[1], gt_mask.shape[0]))
            overlay = he_img.copy()
            overlay[(current_pred == 1) & (tissue_mask_bin > 0)] = [0, 0, 255]

            alpha = 0.6
            cv2.addWeighted(overlay, alpha, he_img, 1 - alpha, 0, he_img)

            vis_path = os.path.join(vis_save_dir, f"{s}_tesla_cancer_prediction.jpg")
            cv2.imwrite(vis_path, he_img)


if __name__ == "__main__":
    main()
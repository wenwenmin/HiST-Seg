# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.

import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast

from ahf_unet_baseline import AHFUnet_Baseline


METRIC_KEYS = [
    "Target_IoU",
    "Dice",
    "Recall",
    "Precision",
    "AUPRC",
    "P_AUROC",
    "HD95",
]


def cv_imread(path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found:{path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"OpenCV cannot decode image:{path}")
    return image


def cv_imwrite(path, img):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Image encoding failed:{path}")
    buf.tofile(path)


def apply_red_heatmap(img_bgr, prob_map):
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255
    p = np.expand_dims(np.clip(prob_map, 0.0, 1.0), 2)
    return (img_bgr * (1.0 - p) + red * p).astype(np.uint8)


def apply_cancer_mask_overlay(img_bgr, pred_mask, alpha=0.4, color=(0, 0, 255)):
    mask_color = np.zeros_like(img_bgr)
    mask_color[:] = color
    mask_3d = np.expand_dims(pred_mask, 2)
    return np.where(
        mask_3d == 1,
        (img_bgr * (1.0 - alpha) + mask_color * alpha).astype(np.uint8),
        img_bgr,
    )


def get_gaussian_window(size, sigma_factor=0.125):
    sigma = max(1.0, size * sigma_factor)
    x = np.arange(size, dtype=np.float32) - size // 2
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return np.outer(g, g).astype(np.float32)


def _probability_histograms(prob_map, gt, valid, row_chunk_size=256):
    if prob_map.shape != gt.shape or valid.shape != gt.shape:
        raise ValueError(
            f"Inconsistent sizes: prob={prob_map.shape}, gt={gt.shape}, valid={valid.shape}"
        )

    positive_hist = np.zeros(65536, dtype=np.int64)
    negative_hist = np.zeros(65536, dtype=np.int64)

    score_u16 = np.rint(
        np.clip(prob_map, 0.0, 1.0) * 65535.0
    ).astype(np.uint16)

    valid = valid.astype(bool)
    height = gt.shape[0]

    for row_start in range(0, height, row_chunk_size):
        row_end = min(height, row_start + row_chunk_size)

        scores = score_u16[row_start:row_end].reshape(-1)
        labels = (gt[row_start:row_end] == 1).reshape(-1)
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


def compute_pixel_auroc(prob_map, gt, valid):
    positive_hist, negative_hist = _probability_histograms(
        prob_map, gt, valid
    )

    positive_count = int(positive_hist.sum())
    negative_count = int(negative_hist.sum())
    if positive_count == 0 or negative_count == 0:
        return np.nan

    negative_below = (
        np.cumsum(negative_hist, dtype=np.int64) - negative_hist
    )
    concordant_pairs = np.sum(
        positive_hist.astype(np.float64)
        * (
            negative_below.astype(np.float64)
            + 0.5 * negative_hist.astype(np.float64)
        )
    )

    return float(
        concordant_pairs / (positive_count * negative_count)
    )


def compute_pixel_auprc(prob_map, gt, valid):
    positive_hist, negative_hist = _probability_histograms(
        prob_map, gt, valid
    )

    positive_count = int(positive_hist.sum())
    negative_count = int(negative_hist.sum())
    if positive_count == 0 or negative_count == 0:
        return np.nan

    
    tp = np.cumsum(positive_hist[::-1], dtype=np.int64)
    fp = np.cumsum(negative_hist[::-1], dtype=np.int64)

    precision = tp.astype(np.float64) / np.maximum(
        tp.astype(np.float64) + fp.astype(np.float64),
        1.0,
    )
    recall = tp.astype(np.float64) / float(positive_count)

    previous_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous_recall) * precision))


def compute_metrics(pred, prob_map, gt, valid):
    'Implementation of compute_metrics for this baseline experiment.'
    eps = 1e-7
    valid = valid.astype(bool)

    p = (pred == 1) & valid
    g = (gt == 1) & valid

    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    tn = int(np.sum(~p & ~g & valid))

    target_iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)

    auprc = compute_pixel_auprc(prob_map, gt, valid)
    p_auroc = compute_pixel_auroc(prob_map, gt, valid)

    if np.sum(p) == 0 or np.sum(g) == 0:
        hd_95 = np.nan
    else:
        try:
            from medpy.metric.binary import hd95
        except ImportError as exc:
            raise ImportError(
                'Calculating HD95 requires installing MedPy, please run: pip install MedPy'
            ) from exc
        hd_95 = float(hd95(p, g))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "Target_IoU": float(target_iou),
        "Dice": float(dice),
        "Recall": float(recall),
        "Precision": float(precision),
        "AUPRC": float(auprc),
        "P_AUROC": float(p_auroc),
        "HD95": float(hd_95),
    }


def sliding_window_inference(
    model,
    he_img,
    device,
    patch_size,
    overlap,
):
    h, w = he_img.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(
            f"slice size{w}x{h}Less than patch_size={patch_size}"
        )

    transform = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])

    img_rgb = cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB)

    gauss = torch.from_numpy(
        get_gaussian_window(patch_size)
    ).float().to(device)

    stride = max(1, int(round(patch_size * (1.0 - overlap))))

    y_steps = list(range(0, h - patch_size + 1, stride))
    x_steps = list(range(0, w - patch_size + 1, stride))

    if y_steps[-1] != h - patch_size:
        y_steps.append(h - patch_size)
    if x_steps[-1] != w - patch_size:
        x_steps.append(w - patch_size)

    prob_map_gpu = torch.zeros((h, w), device=device)
    weight_map_gpu = torch.zeros((h, w), device=device)

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                patch = img_rgb[
                    y:y + patch_size,
                    x:x + patch_size,
                ]
                tensor = transform(image=patch)["image"]
                tensor = tensor.unsqueeze(0).to(device)

                with autocast(
                    "cuda",
                    enabled=(device.type == "cuda"),
                ):
                    logits = model(tensor)
                    probs = torch.sigmoid(logits).squeeze()

                prob_map_gpu[
                    y:y + patch_size,
                    x:x + patch_size,
                ] += probs * gauss

                weight_map_gpu[
                    y:y + patch_size,
                    x:x + patch_size,
                ] += gauss

    prob_map = (
        prob_map_gpu
        / torch.clamp(weight_map_gpu, min=1e-8)
    ).float().cpu().numpy()

    return prob_map


def format_metric_value(mean, std, metric_name):
    if not np.isfinite(mean):
        return "NaN"
    if metric_name == "HD95":
        return f"{mean:.2f}±{std:.2f}"
    return f"{mean:.4f}±{std:.4f}"


def summarize_multiseed(
    all_seed_rows,
    slice_names,
    output_dir,
    prefix,
):
    'Implementation of summarize_multiseed for this baseline experiment.'
    os.makedirs(output_dir, exist_ok=True)

    raw_df = pd.DataFrame(all_seed_rows)
    raw_csv = os.path.join(
        output_dir,
        f"{prefix}_all_seed_metrics.csv",
    )
    raw_df.to_csv(raw_csv, index=False)

    numeric_rows = []
    display_rows = []

    for slice_name in slice_names:
        slice_df = raw_df[raw_df["Slice"] == slice_name]
        if slice_df.empty:
            continue

        numeric_row = {"Slice": slice_name}
        display_row = {"Slice": slice_name}

        for metric in METRIC_KEYS:
            values = slice_df[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if values.size == 0:
                mean_val = np.nan
                std_val = np.nan
            else:
                mean_val = float(np.mean(values))
                std_val = (
                    float(np.std(values, ddof=1))
                    if values.size > 1
                    else 0.0
                )

            numeric_row[f"{metric}_mean"] = mean_val
            numeric_row[f"{metric}_std"] = std_val
            display_row[
                "HD95 (px)" if metric == "HD95" else metric
            ] = format_metric_value(
                mean_val,
                std_val,
                metric,
            )

        numeric_rows.append(numeric_row)
        display_rows.append(display_row)

    numeric_df = pd.DataFrame(numeric_rows)
    display_df = pd.DataFrame(display_rows)

    display_columns = [
        "Slice",
        "Target_IoU",
        "Dice",
        "Recall",
        "Precision",
        "AUPRC",
        "P_AUROC",
        "HD95 (px)",
    ]
    display_df = display_df[display_columns]

    
    
    average_display = {"Slice": "Average"}
    for metric in METRIC_KEYS:
        mean_col = f"{metric}_mean"
        values = numeric_df[mean_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if values.size == 0:
            mean_val = np.nan
            std_val = np.nan
        else:
            mean_val = float(np.mean(values))
            std_val = (
                float(np.std(values, ddof=1))
                if values.size > 1
                else 0.0
            )

        average_display[
            "HD95 (px)" if metric == "HD95" else metric
        ] = format_metric_value(
            mean_val,
            std_val,
            metric,
        )

    display_df = pd.concat(
        [display_df, pd.DataFrame([average_display])],
        ignore_index=True,
    )

    numeric_csv = os.path.join(
        output_dir,
        f"{prefix}_multiseed_summary_numeric.csv",
    )
    display_csv = os.path.join(
        output_dir,
        f"{prefix}_multiseed_summary_display.csv",
    )

    numeric_df.to_csv(
        numeric_csv,
        index=False,
        float_format="%.6f",
    )
    display_df.to_csv(
        display_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 180)
    print(
        f"{prefix}- Multiple random seed evaluation summary"
        '(mean ± standard deviation)'
    )
    print("=" * 180)
    print(display_df.to_string(index=False))
    print("=" * 180)

    print(f"\n original seed indicator:{raw_csv}")
    print(f"Numerical version summary:{numeric_csv}")
    print(f"Display version summary:{display_csv}")

    return raw_df, numeric_df, display_df

def run_inference_single_seed(
    he_path,
    mask_path,
    tissue_path,
    weight_path,
    output_dir,
    patch_size=512,
    overlap=0.5,
    prob_threshold=0.50,
    save_visualization=True,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = AHFUnet_Baseline().to(device)
    model.load_state_dict(
        torch.load(
            weight_path,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    he_img = cv_imread(he_path)
    true_mask = (
        cv_imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE,
        ) > 127
    ).astype(np.uint8)

    h, w = he_img.shape[:2]

    if os.path.exists(tissue_path):
        tissue_img = cv_imread(
            tissue_path,
            cv2.IMREAD_GRAYSCALE,
        )
        tissue_mask = (
            tissue_img > 127
        ).astype(np.uint8)
        if tissue_mask.shape != (h, w):
            tissue_mask = cv2.resize(
                tissue_mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
    else:
        print('⚠️ The organization mask is not found, and the entire image is valid by default.')
        tissue_mask = np.ones(
            (h, w),
            dtype=np.uint8,
        )

    prob_map = sliding_window_inference(
        model=model,
        he_img=he_img,
        device=device,
        patch_size=patch_size,
        overlap=overlap,
    )
    prob_map = prob_map * tissue_mask
    pred_mask = (
        prob_map > prob_threshold
    ).astype(np.uint8)

    metrics = compute_metrics(
        pred=pred_mask,
        prob_map=prob_map,
        gt=true_mask,
        valid=(tissue_mask == 1),
    )

    if save_visualization:
        cv_imwrite(
            os.path.join(output_dir, "heatmap.jpg"),
            apply_red_heatmap(he_img, prob_map),
        )
        cv_imwrite(
            os.path.join(
                output_dir,
                "cancer_overlay.jpg",
            ),
            apply_cancer_mask_overlay(
                he_img,
                pred_mask,
                alpha=0.4,
            ),
        )
        cv_imwrite(
            os.path.join(
                output_dir,
                "pred_mask.png",
            ),
            pred_mask.astype(np.uint8) * 255,
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


if __name__ == "__main__":
    SLICE_NAMES = ["ViHBC", "DuCIS", "XeHBC"]
    SEEDS = [42, 123, 567]

    DATA_ROOT = (
        r"data"
        r"\{slice_name}\resized_data"
    )
    WEIGHT_TEMPLATE = (
        r"outputs/AHFUNet"
        r"\checkpoints_baseline\pure_image"
        r"\best_model_A1_baseline_seed{seed}.pth"
    )
    OUTPUT_ROOT = (
        r"outputs/AHFUNet"
        r"\predictions_baseline_wsi_{slice_name}"
    )
    SUMMARY_ROOT = (
        r"outputs/AHFUNet"
    )

    PATCH_SIZE = 512
    OVERLAP = 0.5
    PROB_THRESHOLD = 0.50

    all_seed_rows = []

    for slice_name in SLICE_NAMES:
        print("\n" + "=" * 70)
        print(f"Handle external slices:{slice_name}")
        print("=" * 70)

        data_dir = DATA_ROOT.format(
            slice_name=slice_name
        )
        he_path = os.path.join(
            data_dir,
            "he_resized.png",
        )
        mask_path = os.path.join(
            data_dir,
            "label_resized.png",
        )
        tissue_path = os.path.join(
            data_dir,
            "mask_resized.png",
        )

        for path in [
            he_path,
            mask_path,
            tissue_path,
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing files:{path}"
                )

        for seed in SEEDS:
            print(f"\n  ▶ Seed={seed}")

            weight_path = WEIGHT_TEMPLATE.format(
                seed=seed
            )
            if not os.path.exists(weight_path):
                raise FileNotFoundError(
                    f"Weight file does not exist:{weight_path}"
                )

            out_dir = os.path.join(
                OUTPUT_ROOT.format(
                    slice_name=slice_name
                ),
                f"seed_{seed}",
            )

            metrics = run_inference_single_seed(
                he_path=he_path,
                mask_path=mask_path,
                tissue_path=tissue_path,
                weight_path=weight_path,
                output_dir=out_dir,
                patch_size=PATCH_SIZE,
                overlap=OVERLAP,
                prob_threshold=PROB_THRESHOLD,
                save_visualization=True,
            )

            row = {
                "Slice": slice_name,
                "Seed": seed,
                **{
                    key: metrics[key]
                    for key in METRIC_KEYS
                },
            }
            all_seed_rows.append(row)

            hd_text = (
                f"{metrics['HD95']:.2f}"
                if np.isfinite(metrics["HD95"])
                else "NaN"
            )
            print(
                f"    Target_IoU={metrics['Target_IoU']:.4f} | "
                f"Dice={metrics['Dice']:.4f} | "
                f"Recall={metrics['Recall']:.4f} | "
                f"Precision={metrics['Precision']:.4f} | "
                f"AUPRC={metrics['AUPRC']:.4f} | "
                f"P-AUROC={metrics['P_AUROC']:.4f} | "
                f"HD95={hd_text}"
            )

    summarize_multiseed(
        all_seed_rows=all_seed_rows,
        slice_names=SLICE_NAMES,
        output_dir=SUMMARY_ROOT,
        prefix="AHFUNet_external_three_slices",
    )

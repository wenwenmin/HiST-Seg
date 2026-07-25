
from __future__ import annotations

import argparse
import csv
import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.st_feature_utils import load_runtime_context_cache, resolve_context_cache_path
from data.context_inputs import extract_aligned_context
from model.pathst_net import build_model
from utils.config import load_config


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def cv_imread(path: str, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(path):
        return None
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def cv_imwrite(path: str, image: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise IOError(path)
    encoded.tofile(path)


def dataset_for_slice(cfg, slice_id: str):
    for dataset_name, ds_cfg in cfg.datasets.items():
        if slice_id in list(ds_cfg.slices):
            base = os.path.join(str(ds_cfg.resized_dir), slice_id) if bool(ds_cfg.per_slice_subdir) else str(ds_cfg.resized_dir)
            return str(dataset_name), ds_cfg, base
    raise KeyError(slice_id)


def resize_for_display(arr: np.ndarray, max_size: int = 1400):
    """Resize an RGB image or a 2D map for visualization only."""
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return arr.copy(), 1.0
    scale = float(max_size) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if arr.ndim == 3 else cv2.INTER_NEAREST
    return cv2.resize(arr, (new_w, new_h), interpolation=interp), scale


def overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
                        color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Overlay a binary mask on an RGB image."""
    out = rgb.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return np.clip(out, 0, 255).astype(np.uint8)
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boundary_on_rgb(rgb: np.ndarray, mask: np.ndarray,
                         color: Tuple[int, int, int] = (255, 230, 0), thickness: int = 3) -> np.ndarray:
    """Draw a thick, bright boundary around a binary mask."""
    out = rgb.copy()
    mask_u8 = (mask.astype(bool).astype(np.uint8) * 255)
    if mask_u8.max() == 0:
        return out
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = cv2.bitwise_xor(mask_u8, eroded)
    if thickness > 1:
        boundary = cv2.dilate(boundary, kernel, iterations=max(1, int(thickness)))
    out[boundary > 0] = np.asarray(color, dtype=np.uint8)
    return out


def save_probability_heatmap(out_path: str, fold: str, probability: np.ndarray, tissue: Optional[np.ndarray] = None,
                             prediction: Optional[np.ndarray] = None):
    """Save a magma-style cancer probability heatmap.

    If ``prediction`` is provided, probability outside the final post-processed
    prediction mask is suppressed so the heatmap, overlay, and metrics all refer
    to the same final prediction.
    """
    prob_vis = probability.astype(np.float32).copy()
    if tissue is not None:
        prob_vis[tissue <= 127] = 0.0
    if prediction is not None:
        prob_vis[prediction.astype(bool) == 0] = 0.0
    prob_show, _ = resize_for_display(prob_vis, max_size=1400)
    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=220)
    im = ax.imshow(prob_show, cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"{fold} Cancer probability")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_label_overlay(out_path: str, fold: str, he: np.ndarray, label: Optional[np.ndarray], tissue: Optional[np.ndarray] = None):
    """Save ground-truth tumor label overlay in green."""
    he_show, scale = resize_for_display(he, max_size=1400)
    if label is None:
        overlay = he_show.copy()
    else:
        label_mask = (label > 127)
        if tissue is not None:
            label_mask &= (tissue > 127)
        label_show = resize_for_display(label_mask.astype(np.uint8), max_size=1400)[0]
        label_show = label_show.astype(bool)
        overlay = overlay_mask_on_rgb(he_show, label_show, alpha=0.48, color=(0, 220, 0))
        overlay = draw_boundary_on_rgb(
            overlay, label_show, color=(0, 255, 0), thickness=max(2, int(round(3 * scale)))
        )
    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=220)
    ax.imshow(overlay)
    ax.set_title(f"{fold} Label overlay")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_prediction_overlay(out_path: str, fold: str, he: np.ndarray, prediction: np.ndarray,
                            label: Optional[np.ndarray] = None, tissue: Optional[np.ndarray] = None):
    """Save a TP/FP/FN prediction-quality overlay.

    Colors:
      red   = true positive tumor region;
      cyan  = false positive prediction;
      green = false negative missed tumor;
      bright yellow = final predicted tumor boundary.
    """
    he_show, scale = resize_for_display(he, max_size=1400)
    pred_mask = prediction.astype(bool)
    if tissue is not None:
        pred_mask &= (tissue > 127)

    if label is not None:
        gt_mask = (label > 127)
        if tissue is not None:
            gt_mask &= (tissue > 127)
        tp = pred_mask & gt_mask
        fp = pred_mask & (~gt_mask)
        fn = (~pred_mask) & gt_mask
    else:
        tp = pred_mask
        fp = np.zeros_like(pred_mask, dtype=bool)
        fn = np.zeros_like(pred_mask, dtype=bool)

    tp_show = resize_for_display(tp.astype(np.uint8), max_size=1400)[0].astype(bool)
    fp_show = resize_for_display(fp.astype(np.uint8), max_size=1400)[0].astype(bool)
    fn_show = resize_for_display(fn.astype(np.uint8), max_size=1400)[0].astype(bool)
    pred_show = resize_for_display(pred_mask.astype(np.uint8), max_size=1400)[0].astype(bool)

    overlay = he_show.copy()
    overlay = overlay_mask_on_rgb(overlay, tp_show, alpha=0.48, color=(255, 0, 0))
    overlay = overlay_mask_on_rgb(overlay, fp_show, alpha=0.58, color=(0, 180, 255))
    overlay = overlay_mask_on_rgb(overlay, fn_show, alpha=0.62, color=(0, 255, 0))
    overlay = draw_boundary_on_rgb(
        overlay, pred_show, color=(255, 230, 0), thickness=max(3, int(round(5 * scale)))
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.9), dpi=220)
    ax.imshow(overlay)
    if label is not None:
        ax.set_title(f"{fold} Prediction overlay: TP red | FP cyan | FN green")
    else:
        ax.set_title(f"{fold} Prediction overlay")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_prediction_only_overlay(out_path: str, fold: str, he: np.ndarray, prediction: np.ndarray,
                                 tissue: Optional[np.ndarray] = None):
    """Save an overlay that shows only the final prediction mask and its boundary."""
    he_show, scale = resize_for_display(he, max_size=1400)
    pred_mask = prediction.astype(bool)
    if tissue is not None:
        pred_mask &= (tissue > 127)
    pred_show = resize_for_display(pred_mask.astype(np.uint8), max_size=1400)[0].astype(bool)

    overlay = he_show.copy()
    overlay = overlay_mask_on_rgb(overlay, pred_show, alpha=0.52, color=(255, 0, 0))
    overlay = draw_boundary_on_rgb(
        overlay, pred_show, color=(255, 230, 0), thickness=max(3, int(round(5 * scale)))
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=220)
    ax.imshow(overlay)
    ax.set_title(f"{fold} Prediction-only overlay")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def save_prediction_only_overlay_no_boundary(out_path: str, fold: str, he: np.ndarray, prediction: np.ndarray,
                                             tissue: Optional[np.ndarray] = None):
    """Save the final prediction mask overlaid on H&E without drawing a boundary."""
    he_show, _ = resize_for_display(he, max_size=1400)
    pred_mask = prediction.astype(bool)
    if tissue is not None:
        pred_mask &= (tissue > 127)
    pred_show = resize_for_display(pred_mask.astype(np.uint8), max_size=1400)[0].astype(bool)

    overlay = overlay_mask_on_rgb(
        he_show, pred_show, alpha=0.52, color=(255, 0, 0)
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=220)
    ax.imshow(overlay)
    ax.set_title(f"{fold} Prediction-only overlay")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def safe_crop(arr: np.ndarray, x0: int, y0: int, size: int, pad_mode: str = "constant"):
    h, w = arr.shape[:2]
    x1, y1 = x0 + size, y0 + size
    xs, xe = max(0, x0), min(w, x1)
    ys, ye = max(0, y0), min(h, y1)
    crop = arr[ys:ye, xs:xe]
    pads = [(max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w))]
    if arr.ndim == 3:
        pads.append((0, 0))
    if any(left or right for left, right in pads):
        try:
            crop = np.pad(crop, pads, mode=pad_mode)
        except ValueError:
            crop = np.pad(crop, pads, mode="constant")
    return crop


def safe_center_window(arr: np.ndarray, cx: int, cy: int, size: int):
    x0, y0 = cx - size // 2, cy - size // 2
    return safe_crop(arr, x0, y0, size), (x0, y0)


def positions(height: int, width: int, patch: int, stride: int):
    if height < patch or width < patch:
        raise ValueError(f"Image {width}x{height} smaller than patch {patch}")
    xs = list(range(0, width - patch + 1, stride))
    ys = list(range(0, height - patch + 1, stride))
    if xs[-1] != width - patch:
        xs.append(width - patch)
    if ys[-1] != height - patch:
        ys.append(height - patch)
    return [(x, y) for y in ys for x in xs]


def gaussian_window(size: int, sigma_factor: float):
    sigma = max(1.0, size * sigma_factor)
    coords = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    one = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    window = np.outer(one, one).astype(np.float32)
    return window / max(float(window.max()), 1e-8)


def normalize_hr(rgb_u8: np.ndarray):
    image = rgb_u8.astype(np.float32) / 255.0
    return ((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def load_fold_cache(cfg, ds_cfg, fold: str, slice_id: str):
    path = resolve_context_cache_path(cfg, ds_cfg, slice_id, fold)
    runtime_cache = load_runtime_context_cache(
        path,
        cfg=cfg,
        expected_dim=int(cfg.dims.st_dim),
        expected_fold=fold,
    )
    print(
        f"Context[inference] {slice_id}: spots={len(runtime_cache['features'])} "
        f"HEx={runtime_cache['thumb_native'].shape[:2]} "
        f"grid={int(cfg.geometry.context_grid_size)}"
    )
    return runtime_cache


def load_slide(cfg, fold: str, slice_id: str):
    dataset_name, ds_cfg, base = dataset_for_slice(cfg, slice_id)
    he = cv_imread(os.path.join(base, "he_resized.png"), cv2.IMREAD_COLOR)
    label = cv_imread(os.path.join(base, "label_resized.png"), cv2.IMREAD_GRAYSCALE)
    tissue = cv_imread(os.path.join(base, "mask_resized.png"), cv2.IMREAD_GRAYSCALE)
    if he is None:
        raise FileNotFoundError(os.path.join(base, "he_resized.png"))
    he = cv2.cvtColor(he, cv2.COLOR_BGR2RGB)
    if tissue is None:
        tissue = np.full(he.shape[:2], 255, np.uint8)
    if label is not None and label.shape != he.shape[:2]:
        raise ValueError("Label and H&E dimensions differ")
    return dataset_name, he, label, tissue, load_fold_cache(cfg, ds_cfg, fold, slice_id)



def resolve_checkpoint_path(checkpoint_root: str, fold: str) -> str:
    """Return the single best checkpoint for one external fold."""
    path = os.path.join(checkpoint_root, f"fold_{fold}", f"pathst_{fold}_best.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Best checkpoint not found: {path}. Train this fold first with train_cross_dataset.py."
        )
    return path


def load_checkpoint(model, path: str, fold: str, device, expected_seed: Optional[int] = None, strict_slices: bool = True):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if str(checkpoint.get("test_slice")) != fold:
        raise RuntimeError(f"Checkpoint fold mismatch: {checkpoint.get('test_slice')} vs {fold}")
    stored_seed = checkpoint.get("random_seed")
    if expected_seed is not None and stored_seed is not None and int(stored_seed) != int(expected_seed):
        raise RuntimeError(
            f"Checkpoint seed mismatch: stored={stored_seed}, requested={expected_seed}, path={path}"
        )

    all_slices = set(str(s) for s in checkpoint.get("all_slices", []))
    domain_val_slices = set(str(s) for s in checkpoint.get("domain_val_slices", []))
    expected_train = all_slices - {str(fold)} - domain_val_slices
    actual_train = set(str(s) for s in checkpoint.get("train_slices", []))
    if actual_train and expected_train and actual_train != expected_train:
        msg = (
            f"Checkpoint train_slices mismatch. Expected {sorted(expected_train)} "
            f"after excluding test={fold} and domain_val={sorted(domain_val_slices)}, "
            f"got {sorted(actual_train)}."
        )
        if strict_slices:
            raise RuntimeError(msg)
        print(f"⚠️ Warning: {msg}")

    state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    if state is None:
        raise RuntimeError(f"Checkpoint has no 'ema' or 'model' state: {path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [key for key in missing if not key.startswith("detail.vit.")]
    if invalid_missing or unexpected:
        raise RuntimeError(f"State mismatch: missing={invalid_missing}, unexpected={unexpected}")
    print(f"Loaded best checkpoint: {path}")
    return checkpoint


def build_context_inputs(cfg, cache, x: int, y: int):
    return extract_aligned_context(
        cache,
        x,
        y,
        cfg,
        is_train=False,
        keep_probability=1.0,
    )


def infer_slide(model, cfg, fold: str, slice_id: str, device, batch_size: int,
                channels_last: bool):
    dataset_name, he, label, tissue, cache = load_slide(cfg, fold, slice_id)
    height, width = he.shape[:2]
    patch = int(cfg.geometry.patch_size)
    stride = max(1, int(round(patch * (1.0 - float(cfg.inference.overlap)))))
    coords = positions(height, width, patch, stride)
    blend = gaussian_window(patch, float(cfg.inference.gaussian_sigma_factor))
    probability_sum = np.zeros((height, width), np.float32)
    weight_sum = np.zeros((height, width), np.float32)

    for start_index in tqdm(range(0, len(coords), batch_size), desc=f"Infer {slice_id}", ncols=110):
        chunk = coords[start_index:start_index + batch_size]
        hr_batch = []
        he_context_batch = []
        context_tissue_batch = []
        spot_features_batch = []
        spot_positions_batch = []
        spot_mask_batch = []

        for x, y in chunk:
            hr = safe_crop(he, x, y, patch, pad_mode="edge")
            tissue_patch = safe_crop(tissue, x, y, patch, pad_mode="constant")
            context_data = build_context_inputs(cfg, cache, x, y)
            he_context = context_data["he_context"]
            context_tissue = context_data["context_tissue"]


            hr_batch.append(normalize_hr(hr))
            he_context_batch.append(normalize_hr(he_context))
            context_tissue_batch.append(
                (context_tissue > 127).astype(np.float32)[None]
            )
            spot_features_batch.append(context_data["spot_features"])
            spot_positions_batch.append(context_data["spot_positions"])
            spot_mask_batch.append(context_data["spot_mask"])

        hr_tensor = torch.from_numpy(np.stack(hr_batch)).to(device)
        he_context_tensor = torch.from_numpy(np.stack(he_context_batch)).to(device)
        context_tissue_tensor = torch.from_numpy(np.stack(context_tissue_batch)).to(device)
        spot_features_tensor = torch.from_numpy(np.stack(spot_features_batch)).to(device)
        spot_positions_tensor = torch.from_numpy(np.stack(spot_positions_batch)).to(device)
        spot_mask_tensor = torch.from_numpy(np.stack(spot_mask_batch)).to(device)

        if channels_last:
            hr_tensor = hr_tensor.contiguous(memory_format=torch.channels_last)
            he_context_tensor = he_context_tensor.contiguous(memory_format=torch.channels_last)
            context_tissue_tensor = context_tissue_tensor.contiguous(memory_format=torch.channels_last)

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=bool(cfg.train.amp and device.type == "cuda")
        ):
            probability = model.predict(
                hr_tensor,
                he_context=he_context_tensor,
                spot_features=spot_features_tensor,
                spot_positions=spot_positions_tensor,
                spot_mask=spot_mask_tensor,
                context_tissue=context_tissue_tensor,
            ).float().cpu().numpy()[:, 0]

        for (x, y), patch_probability in zip(chunk, probability):
            probability_sum[y:y + patch, x:x + patch] += patch_probability * blend
            weight_sum[y:y + patch, x:x + patch] += blend

    probability = probability_sum / np.maximum(weight_sum, 1e-8)
    probability[tissue <= 127] = 0.0
    return dataset_name, probability, label, tissue


def postprocess(probability, tissue, threshold: float, cfg):
    pred = (probability >= threshold).astype(np.uint8)
    pred[tissue <= 127] = 0
    pp = cfg.inference.postprocess
    if not bool(pp.enable):
        return pred
    if int(pp.open_kernel) > 1:
        size = int(pp.open_kernel) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
    if int(pp.close_kernel) > 1:
        size = int(pp.close_kernel) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    if int(pp.min_area) > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(pred, 8)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] < int(pp.min_area):
                pred[labels == component] = 0
    return pred


def boundary(mask: np.ndarray):
    mask = mask.astype(bool)
    return mask ^ ndimage.binary_erosion(mask)


def hd95(pred: np.ndarray, target: np.ndarray):
    pred_b, target_b = boundary(pred), boundary(target)
    if not pred_b.any() and not target_b.any():
        return 0.0
    if not pred_b.any() or not target_b.any():
        return float(math.hypot(*pred.shape))
    dist_target = ndimage.distance_transform_edt(~target_b)
    dist_pred = ndimage.distance_transform_edt(~pred_b)
    distances = np.concatenate([dist_target[pred_b], dist_pred[target_b]])
    return float(np.percentile(distances, 95))


def metrics(probability, pred, label, tissue):
    if label is None:
        return {}
    valid = tissue > 127
    target = (label > 127) & valid
    prediction = pred.astype(bool) & valid
    tp = int((prediction & target).sum())
    fp = int((prediction & (~target) & valid).sum())
    fn = int(((~prediction) & target).sum())
    tn = int(((~prediction) & (~target) & valid).sum())
    dice = 2.0 * tp / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    y_true = target[valid].astype(np.uint8)
    y_score = probability[valid].astype(np.float32)

    # AUPRC 和 P-AUROC 都基于组织区域内的像素级连续概率图计算。
    # 当标签中只存在一个类别时，两项指标均无法定义，返回 NaN。
    if np.unique(y_true).size > 1:
        auprc = float(average_precision_score(y_true, y_score))
        p_auroc = float(roc_auc_score(y_true, y_score))
    else:
        auprc = float("nan")
        p_auroc = float("nan")

    return {
        "Dice": float(dice),
        "Target_IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "HD95": hd95(prediction, target),
        "AUPRC": auprc,
        "P_AUROC": p_auroc,
        "Predicted_Fraction": float(prediction.sum() / max(1, valid.sum())),
        "GT_Fraction": float(target.sum() / max(1, valid.sum())),
    }


def append_csv(path: str, row: Dict[str, Any]):
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)



MULTISEED_METRICS = (
    ("Target_IoU", "Target_IoU", 4),
    ("Dice", "Dice", 4),
    ("Recall", "Recall", 4),
    ("Precision", "Precision", 4),
    ("AUPRC", "AUPRC", 4),
    ("P-AUROC", "P_AUROC", 4),
    ("HD95 (px)", "HD95", 2),
)


def resolve_run_seeds(cfg, cli_seeds):
    configured = list(getattr(cfg.experiment, "seeds", [42, 123, 567]))
    seeds = cli_seeds if cli_seeds is not None else configured
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("At least one inference seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return seeds


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _mean_std(values):
    """Compute mean and sample standard deviation across random seeds.

    ``ddof=1`` gives the sample standard deviation across repeated runs.
    """
    arr = np.asarray([float(v) for v in values], dtype=np.float64)

    if arr.size == 0:
        return float("nan"), float("nan")

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def build_multiseed_summary(rows, slice_order, seeds):
    """Return per-slice mean/std and a seed-wise macro Average row."""
    by_pair = {
        (str(row["Test_Slice"]), int(row["Seed"])): row
        for row in rows
    }

    summary = {}

    # Per-slice statistics across random seeds.
    for slice_id in slice_order:
        missing = [
            seed
            for seed in seeds
            if (str(slice_id), int(seed)) not in by_pair
        ]
        if missing:
            raise RuntimeError(
                f"Missing inference results for slice={slice_id}, seeds={missing}"
            )

        metric_summary = {}

        for _, key, _ in MULTISEED_METRICS:
            values = [
                _finite_float(
                    by_pair[(str(slice_id), int(seed))].get(key)
                )
                for seed in seeds
            ]
            values = [v for v in values if v is not None]

            mean, std = _mean_std(values)

            metric_summary[key] = {
                "mean": mean,
                "std": std,
                "n": len(values),
            }

        summary[str(slice_id)] = metric_summary

    # Average row:
    # 1) compute one macro average across slices for each seed;
    # 2) compute mean/std across those seed-wise macro averages.
    average_summary = {}

    for _, key, _ in MULTISEED_METRICS:
        seed_macro_values = []

        for seed in seeds:
            values = [
                _finite_float(
                    by_pair[(str(slice_id), int(seed))].get(key)
                )
                for slice_id in slice_order
            ]
            values = [v for v in values if v is not None]

            if values:
                seed_macro_values.append(float(np.mean(values)))

        mean, std = _mean_std(seed_macro_values)

        average_summary[key] = {
            "mean": mean,
            "std": std,
            "n": len(seed_macro_values),
            "per_seed_macro": seed_macro_values,
        }

    summary["Average"] = average_summary
    return summary


def _format_mean_std(stat, digits):
    mean = float(stat.get("mean", float("nan")))
    std = float(stat.get("std", float("nan")))

    if not math.isfinite(mean) or not math.isfinite(std):
        return "NaN"

    return f"{mean:.{digits}f}±{std:.{digits}f}"


def print_multiseed_table(model_name, table_name, summary, slice_order):
    headers = ["Slice"] + [label for label, _, _ in MULTISEED_METRICS]
    table_rows = []

    for slice_id in list(slice_order) + ["Average"]:
        values = [str(slice_id)]

        for _, key, digits in MULTISEED_METRICS:
            values.append(
                _format_mean_std(
                    summary[str(slice_id)][key],
                    digits,
                )
            )

        table_rows.append(values)

    widths = []

    for col in range(len(headers)):
        widths.append(
            max(
                len(headers[col]),
                *(len(row[col]) for row in table_rows),
            )
        )

    separator = (
        "|"
        + "|".join("-" * (width + 2) for width in widths)
        + "|"
    )
    line_len = len(separator)

    print()
    print(f"👉 当前正在评估模型: 【{model_name}】")
    print("=" * line_len)
    print(
        f"📋 {table_name} - 切片级多种子统计汇总表 "
        f"(均值±标准差)"
    )
    print("=" * line_len)

    print(
        "| "
        + " | ".join(
            headers[i].ljust(widths[i])
            for i in range(len(headers))
        )
        + " |"
    )

    print(separator)

    for row in table_rows:
        if row[0] == "Average":
            print(separator)

        print(
            "| "
            + " | ".join(
                row[i].ljust(widths[i])
                for i in range(len(row))
            )
            + " |"
        )

    print("=" * line_len)


def write_multiseed_outputs(
    output_root,
    prefix,
    rows,
    summary,
    slice_order,
    seeds,
):
    """Write raw per-seed metrics and mean±std summary outputs."""
    os.makedirs(output_root, exist_ok=True)

    # Raw metrics for every slice and every seed.
    raw_csv = os.path.join(
        output_root,
        f"{prefix}_all_seed_metrics.csv",
    )

    if rows:
        fieldnames = list(rows[0].keys())

        with open(
            raw_csv,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(rows)

    # Compact mean ± std table.
    table_csv = os.path.join(
        output_root,
        f"{prefix}_multiseed_mean_std.csv",
    )

    headers = ["Slice"] + [label for label, _, _ in MULTISEED_METRICS]

    with open(
        table_csv,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for slice_id in list(slice_order) + ["Average"]:
            writer.writerow(
                [slice_id]
                + [
                    _format_mean_std(
                        summary[str(slice_id)][key],
                        digits,
                    )
                    for _, key, digits in MULTISEED_METRICS
                ]
            )

    # JSON summary containing mean/std only.
    summary_json = os.path.join(
        output_root,
        f"{prefix}_multiseed_mean_std.json",
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "seeds": list(seeds),
                "statistic": "mean ± sample standard deviation (ddof=1)",
                "summary": summary,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Excel version of the mean ± std table.
    xlsx_path = os.path.join(
        output_root,
        f"{prefix}_multiseed_mean_std.xlsx",
    )

    try:
        from openpyxl import Workbook
        from openpyxl.styles import (
            Alignment,
            Border,
            Font,
            PatternFill,
            Side,
        )

        wb = Workbook()
        ws = wb.active
        ws.title = "mean_std"
        ws.append(headers)

        for slice_id in list(slice_order) + ["Average"]:
            ws.append(
                [slice_id]
                + [
                    _format_mean_std(
                        summary[str(slice_id)][key],
                        digits,
                    )
                    for _, key, digits in MULTISEED_METRICS
                ]
            )

        thin = Side(style="thin", color="000000")

        for row in ws.iter_rows():
            for cell in row:
                cell.border = Border(
                    left=thin,
                    right=thin,
                    top=thin,
                    bottom=thin,
                )
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                )

        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(
                "solid",
                fgColor="D9EAF7",
            )

        for cell in ws[ws.max_row]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(
                "solid",
                fgColor="E2F0D9",
            )

        widths = [
            16, 19, 19, 19,
            19, 19, 19, 20,
        ]

        for index, width in enumerate(widths, start=1):
            ws.column_dimensions[
                chr(64 + index)
            ].width = width

        wb.save(xlsx_path)

    except Exception as exc:
        xlsx_path = None
        print(
            f"[warn] Could not write XLSX multi-seed table: {exc}"
        )

    print(f"All-seed raw metrics CSV: {raw_csv}")
    print(f"Mean±std table CSV: {table_csv}")
    print(f"Mean±std summary JSON: {summary_json}")

    if xlsx_path:
        print(f"Mean±std table XLSX: {xlsx_path}")

    return raw_csv, table_csv, summary_json, xlsx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cross_dataset.yaml")
    parser.add_argument("--folds", nargs="+", default=['DuCIS'],
                        help="External test folds. Omit to evaluate ViHBC, DuCIS and XeHBC.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Model seeds. Omit to use experiment.seeds from YAML.")
    parser.add_argument("--checkpoint_root", default=None)
    parser.add_argument("--no_strict_checkpoint_slices", action="store_true")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--channels_last", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / args.config
    cfg = load_config(config_path)
    folds = [str(fold) for fold in (args.folds or list(cfg.experiment.external_folds))]
    unknown = sorted(set(folds) - set(str(x) for x in cfg.experiment.external_folds))
    if unknown:
        raise ValueError(f"Unsupported external folds: {unknown}")
    seeds = resolve_run_seeds(cfg, args.seeds)
    checkpoint_root = args.checkpoint_root or str(cfg.paths.checkpoint_dir)
    output_root = args.output_root or str(cfg.paths.output_dir)
    batch_size = args.batch_size or int(cfg.inference.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []

    print(f"Cross-dataset inference seeds={seeds}, folds={folds}, device={device}")
    for seed in seeds:
        seed_checkpoint_root = os.path.join(checkpoint_root, f"seed_{seed}")
        seed_output_root = os.path.join(output_root, f"seed_{seed}")
        os.makedirs(seed_output_root, exist_ok=True)
        seed_summary_path = os.path.join(seed_output_root, f"cross_dataset_metrics_seed{seed}.csv")
        if os.path.exists(seed_summary_path):
            os.remove(seed_summary_path)

        for fold in folds:
            model = build_model(cfg, build_uni=True).to(device)
            if args.channels_last:
                model = model.to(memory_format=torch.channels_last)
            checkpoint_path = resolve_checkpoint_path(seed_checkpoint_root, str(fold))
            checkpoint = load_checkpoint(
                model, checkpoint_path, str(fold), device, expected_seed=seed,
                strict_slices=not bool(args.no_strict_checkpoint_slices),
            )
            model.eval()
            threshold = float(args.threshold) if args.threshold is not None else float(
                checkpoint.get("calibrated_threshold", cfg.inference.prob_threshold)
            )
            dataset_name, probability, label, tissue = infer_slide(
                model, cfg, str(fold), str(fold), device, batch_size, args.channels_last
            )
            pred = postprocess(probability, tissue, threshold, cfg)
            result = metrics(probability, pred, label, tissue)
            fold_dir = os.path.join(seed_output_root, str(fold))
            os.makedirs(fold_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(fold_dir, "prediction.npz"), probability=probability, prediction=pred,
            )
            cv_imwrite(os.path.join(fold_dir, "probability.png"), np.clip(probability * 255, 0, 255).astype(np.uint8))
            cv_imwrite(os.path.join(fold_dir, "prediction.png"), pred.astype(np.uint8) * 255)
            _, _, base = dataset_for_slice(cfg, str(fold))
            he_bgr = cv_imread(os.path.join(base, "he_resized.png"), cv2.IMREAD_COLOR)
            if he_bgr is None:
                raise FileNotFoundError(os.path.join(base, "he_resized.png"))
            he = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2RGB)

            save_probability_heatmap(
                os.path.join(fold_dir, "cancer_probability_heatmap.png"),
                f"{fold} seed={seed}", probability, tissue, prediction=pred,
            )
            save_label_overlay(
                os.path.join(fold_dir, "label_overlay.png"),
                f"{fold} seed={seed}", he, label, tissue,
            )
            save_prediction_overlay(
                os.path.join(fold_dir, "prediction_overlay.png"),
                f"{fold} seed={seed}", he, pred, label, tissue,
            )
            save_prediction_only_overlay(
                os.path.join(fold_dir, "prediction_only_overlay.png"),
                f"{fold} seed={seed}", he, pred, tissue,
            )
            save_prediction_only_overlay_no_boundary(
                os.path.join(fold_dir, "prediction_only_overlay_no_boundary.png"),
                f"{fold} seed={seed}", he, pred, tissue,
            )
            payload = {
                "Dataset": dataset_name, "Test_Slice": str(fold), "Seed": int(seed),
                "Threshold": threshold, **result,
                "Threshold_Source": "command line" if args.threshold is not None else checkpoint.get("threshold_source"),
                "Checkpoint": checkpoint_path,
                "Checkpoint_Kind": "best",
            }
            with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            append_csv(seed_summary_path, payload)
            all_rows.append(payload)
            print(payload)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"Seed {seed} metrics: {seed_summary_path}")

    summary = build_multiseed_summary(all_rows, folds, seeds)
    print_multiseed_table("PathST-Seg", "PathST-Seg_Three_Slices", summary, folds)
    write_multiseed_outputs(output_root, "cross_dataset", all_rows, summary, folds, seeds)


if __name__ == "__main__":
    main()

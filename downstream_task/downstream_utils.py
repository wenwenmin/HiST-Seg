from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.inference_cross_dataset import (  # noqa: E402
    cv_imread,
    cv_imwrite,
    dataset_for_slice,
    infer_slide,
    load_checkpoint,
    load_fold_cache,
    load_slide,
    postprocess,
)
from model.pathst_net import build_model  # noqa: E402
from utils.config import load_config  # noqa: E402

DEFAULT_FOLDS = ["ViHBC"]


def resolve_config_path(config: str | os.PathLike[str]) -> Path:
    path = Path(config)
    candidates = [
        path,
        ROOT / path,
        ROOT / "engine" / path,
        ROOT / "engine" / "pathst_net_improved_residual_st_stable_v5.yaml",
        ROOT / "engine" / "pathst_net_improved_residual_st_v5.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find config. Tried:\n" + "\n".join(str(c) for c in candidates)
    )


def load_downstream_config(config: str | os.PathLike[str]):
    cfg = load_config(resolve_config_path(config))
    if not hasattr(cfg, "data"):
        cfg.data = {}
    if not hasattr(cfg.data, "min_tissue_ratio"):
        cfg.data.min_tissue_ratio = 0.0
    if not hasattr(cfg.data, "strict_fold_context_cache"):
        cfg.data.strict_fold_context_cache = True
    if not hasattr(cfg.train, "threshold_fallback_value"):
        cfg.train.threshold_fallback_value = 0.5
    return cfg


def ensure_dir(path: str | os.PathLike[str]) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)


def write_csv(path: str | os.PathLike[str], rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | os.PathLike[str], payload: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_slide_images(cfg, slice_id: str):
    dataset_name, ds_cfg, base = dataset_for_slice(cfg, slice_id)
    he = cv_imread(os.path.join(base, "he_resized.png"), cv2.IMREAD_COLOR)
    label = cv_imread(os.path.join(base, "label_resized.png"), cv2.IMREAD_GRAYSCALE)
    tissue = cv_imread(os.path.join(base, "mask_resized.png"), cv2.IMREAD_GRAYSCALE)
    if he is None:
        raise FileNotFoundError(os.path.join(base, "he_resized.png"))
    he = cv2.cvtColor(he, cv2.COLOR_BGR2RGB)
    if tissue is None:
        tissue = np.full(he.shape[:2], 255, np.uint8)
    return dataset_name, he, label, tissue


def checkpoint_path(checkpoint_root: str | os.PathLike[str], fold: str) -> str:
    return str(Path(checkpoint_root) / f"fold_{fold}" / f"pathst_improved_{fold}_best.pth")


def load_fold_checkpoint_meta(checkpoint_root: str | os.PathLike[str], fold: str) -> Dict[str, Any]:
    path = checkpoint_path(checkpoint_root, fold)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt if isinstance(ckpt, dict) else {}


def build_and_load_model(cfg, fold: str, checkpoint_root: str | os.PathLike[str], device, channels_last: bool):
    model = build_model(cfg, build_uni=True).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    ckpt_path = checkpoint_path(checkpoint_root, fold)
    ckpt = load_checkpoint(model, ckpt_path, str(fold), device)
    model.eval()
    return model, ckpt


def load_prediction_or_infer(
    cfg,
    fold: str,
    checkpoint_root: str | os.PathLike[str],
    prediction_root: str | os.PathLike[str],
    batch_size: Optional[int] = None,
    channels_last: bool = True,
    threshold: Optional[float] = None,
    force_infer: bool = False,
):
    fold_dir = Path(prediction_root) / str(fold)
    npz_path = fold_dir / "prediction.npz"
    metrics_path = fold_dir / "metrics.json"

    dataset_name, he, label, tissue = load_slide_images(cfg, str(fold))
    ckpt_meta: Dict[str, Any] = {}
    used_threshold = float(threshold) if threshold is not None else float(getattr(cfg.inference, "prob_threshold", 0.5))
    threshold_source = "default/config"

    if npz_path.exists() and not force_infer:
        data = np.load(npz_path, allow_pickle=False)
        probability = data["probability"].astype(np.float32)
        prediction = data["prediction"].astype(np.uint8) if "prediction" in data.files else None
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            if threshold is None and "Threshold" in metrics:
                used_threshold = float(metrics["Threshold"])
                threshold_source = str(metrics.get("Threshold_Source", "metrics.json"))
        else:
            try:
                ckpt_meta = load_fold_checkpoint_meta(checkpoint_root, fold)
                if threshold is None:
                    used_threshold = float(ckpt_meta.get("calibrated_threshold", used_threshold))
                    threshold_source = str(ckpt_meta.get("threshold_source", "checkpoint"))
            except FileNotFoundError:
                pass
        if prediction is None or threshold is not None:
            prediction = postprocess(probability, tissue, used_threshold, cfg).astype(np.uint8)
        return {
            "dataset": dataset_name,
            "fold": str(fold),
            "he": he,
            "label": label,
            "tissue": tissue,
            "probability": probability,
            "prediction": prediction,
            "threshold": used_threshold,
            "threshold_source": threshold_source,
            "prediction_path": str(npz_path),
            "from_cache": True,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = build_and_load_model(cfg, str(fold), checkpoint_root, device, channels_last)
    used_threshold = float(threshold) if threshold is not None else float(
        ckpt.get("calibrated_threshold", getattr(cfg.inference, "prob_threshold", 0.5))
    )
    threshold_source = "command line" if threshold is not None else str(ckpt.get("threshold_source", "checkpoint"))
    batch_size = int(batch_size or cfg.inference.batch_size)
    dataset_name, probability, label, tissue = infer_slide(
        model, cfg, str(fold), str(fold), device, batch_size, channels_last
    )
    prediction = postprocess(probability, tissue, used_threshold, cfg).astype(np.uint8)
    ensure_dir(fold_dir)
    np.savez_compressed(npz_path, probability=probability.astype(np.float32), prediction=prediction.astype(np.uint8))
    cv_imwrite(str(fold_dir / "probability.png"), np.clip(probability * 255, 0, 255).astype(np.uint8))
    cv_imwrite(str(fold_dir / "prediction.png"), prediction.astype(np.uint8) * 255)
    write_json(metrics_path, {
        "Dataset": dataset_name,
        "Test_Slice": str(fold),
        "Threshold": used_threshold,
        "Threshold_Source": threshold_source,
        "Prediction_Path": str(npz_path),
    })
    _, he, _, _ = load_slide_images(cfg, str(fold))
    return {
        "dataset": dataset_name,
        "fold": str(fold),
        "he": he,
        "label": label,
        "tissue": tissue,
        "probability": probability.astype(np.float32),
        "prediction": prediction.astype(np.uint8),
        "threshold": used_threshold,
        "threshold_source": threshold_source,
        "prediction_path": str(npz_path),
        "from_cache": False,
    }


def binary_metrics(probability: np.ndarray, prediction: np.ndarray, label: Optional[np.ndarray], tissue: np.ndarray) -> Dict[str, float]:
    if label is None:
        return {}
    valid = tissue > 127
    y = (label > 127) & valid
    p = prediction.astype(bool) & valid
    tp = float((p & y).sum())
    fp = float((p & (~y) & valid).sum())
    fn = float(((~p) & y).sum())
    tn = float(((~p) & (~y) & valid).sum())
    return {
        "Dice": 2.0 * tp / max(1.0, 2.0 * tp + fp + fn),
        "IoU": tp / max(1.0, tp + fp + fn),
        "Precision": tp / max(1.0, tp + fp),
        "Recall": tp / max(1.0, tp + fn),
        "Specificity": tn / max(1.0, tn + fp),
    }


def pearsonr_np(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x, y = x[mask], y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearmanr_np(x: Sequence[float], y: Sequence[float]) -> float:
    def rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        unique, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        for k, c in enumerate(counts):
            if c > 1:
                ranks[inv == k] = ranks[inv == k].mean()
        return ranks

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return pearsonr_np(rankdata(x[mask]), rankdata(y[mask]))


def resize_for_display(image: np.ndarray, max_size: int = 1600):
    h, w = image.shape[:2]
    scale = min(1.0, float(max_size) / max(h, w))
    if scale >= 0.999:
        return image.copy(), 1.0
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if image.ndim == 2:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45, color=(255, 0, 0)) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    c = np.asarray(color, dtype=np.float32)
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boundary_on_rgb(rgb: np.ndarray, mask: np.ndarray, color=(255, 255, 0), thickness: int = 2) -> np.ndarray:
    out = rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def sample_array_at_xy(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray, default=np.nan) -> np.ndarray:
    h, w = arr.shape[:2]
    xi = np.rint(xs).astype(np.int64)
    yi = np.rint(ys).astype(np.int64)
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full(xs.shape, default, dtype=np.float32)
    out[valid] = arr[yi[valid], xi[valid]].astype(np.float32)
    return out


def load_st_cache_for_fold(cfg, fold: str):
    dataset_name, ds_cfg, _ = dataset_for_slice(cfg, fold)
    cache = load_fold_cache(cfg, ds_cfg, str(fold), str(fold))
    return dataset_name, ds_cfg, cache


def disk_mean(arr: np.ndarray, cx: float, cy: float, radius: float, valid_mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    h, w = arr.shape[:2]
    r = max(1, int(round(radius)))
    x0, x1 = max(0, int(round(cx)) - r), min(w, int(round(cx)) + r + 1)
    y0, y1 = max(0, int(round(cy)) - r), min(h, int(round(cy)) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return float("nan"), 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 <= float(radius) ** 2
    if valid_mask is not None:
        disk = disk & valid_mask[y0:y1, x0:x1].astype(bool)
    denom = int(disk.sum())
    if denom <= 0:
        return float("nan"), 0.0
    return float(np.nanmean(arr[y0:y1, x0:x1][disk])), float(denom / max(1, disk.size))
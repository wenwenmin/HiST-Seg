from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required. Install with: pip install scipy") from exc

try:
    from skimage.morphology import disk, remove_small_holes, remove_small_objects
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-image is required. Install with: pip install scikit-image") from exc

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def add_project_to_path(project_root: str | Path) -> None:
    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def read_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def read_binary_mask(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > 0).astype(np.uint8)


@dataclass
class SlideData:
    slide_id: str
    he: np.ndarray      # H, W, 3 uint8
    label: np.ndarray   # H, W uint8 {0,1}
    tissue: np.ndarray  # H, W uint8 {0,1}

    @property
    def height(self) -> int:
        return int(self.he.shape[0])

    @property
    def width(self) -> int:
        return int(self.he.shape[1])


def resolve_slide_paths(data_root: str | Path, slide_id: str, files_cfg: dict) -> Tuple[Path, Path, Path]:
    """Resolve either per-slide folders or flat, pattern-based dataset layouts."""
    layout = files_cfg.get("layout", "per_slide")
    base = Path(data_root) if layout == "flat" else Path(data_root) / slide_id

    def resolve(name: str, default: str) -> Path:
        relative = str(files_cfg.get(name, default)).format(slide_id=slide_id)
        return base / relative

    return (
        resolve("he", "he_resized.png"),
        resolve("label", "label_resized.png"),
        resolve("tissue", "mask_resized.png"),
    )


def slide_source(cfg: dict, slide_id: str) -> Tuple[str | Path, dict]:
    """Return the root and file layout for one slide, allowing per-slide overrides."""
    source = cfg.get("slide_sources", {}).get(slide_id, {})
    files_cfg = dict(cfg.get("files", {}))
    files_cfg.update(source.get("files", {}))
    return source.get("data_root", cfg["data_root"]), files_cfg


def load_slide(data_root: str | Path, slide_id: str, files_cfg: dict) -> SlideData:
    he_path, label_path, tissue_path = resolve_slide_paths(data_root, slide_id, files_cfg)
    missing = [str(p) for p in [he_path, label_path, tissue_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {slide_id}:\n" + "\n".join(missing))
    he = read_rgb(he_path)
    label = read_binary_mask(label_path)
    tissue = read_binary_mask(tissue_path)
    if he.shape[:2] != label.shape or he.shape[:2] != tissue.shape:
        raise ValueError(
            f"Shape mismatch for {slide_id}: he={he.shape[:2]}, label={label.shape}, tissue={tissue.shape}"
        )
    # Keep labels in valid tissue only. This protects metrics and sampling from annotation spillover.
    label = (label & tissue).astype(np.uint8)
    return SlideData(slide_id=slide_id, he=he, label=label, tissue=tissue)


def load_all_slides(cfg: dict) -> Dict[str, SlideData]:
    slides: Dict[str, SlideData] = {}
    for sid in cfg["slices"]:
        data_root, files_cfg = slide_source(cfg, sid)
        slides[sid] = load_slide(data_root, sid, files_cfg)
    return slides


def check_dataset(cfg: dict, out_dir: Optional[str | Path] = None) -> List[dict]:
    rows: List[dict] = []
    for sid in cfg["slices"]:
        data_root, files_cfg = slide_source(cfg, sid)
        _, label_path, _ = resolve_slide_paths(data_root, sid, files_cfg)
        sl = load_slide(data_root, sid, files_cfg)
        h, w = sl.label.shape
        tissue_px = int(sl.tissue.sum())
        tumor_px = int(sl.label.sum())
        spill_px = int(((read_binary_mask(label_path) == 1) & (sl.tissue == 0)).sum())
        rows.append({
            "slice": sid,
            "height": h,
            "width": w,
            "tissue_px": tissue_px,
            "tumor_px": tumor_px,
            "tumor_ratio_in_tissue": float(tumor_px / max(tissue_px, 1)),
            "label_outside_tissue_px_before_clip": spill_px,
        })
    if out_dir is not None:
        ensure_dir(out_dir)
        csv_path = Path(out_dir) / "dataset_check.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved dataset check: {csv_path}")
    return rows


def integral_image(mask: np.ndarray) -> np.ndarray:
    # integral image with leading zero row/col for fast clipped rectangle sum.
    return np.pad(mask.astype(np.float64).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")


def rect_sum(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def clipped_patch_ratio(ii: np.ndarray, h: int, w: int, center_x: int, center_y: int, size: int) -> float:
    half = size // 2
    x0 = max(0, center_x - half)
    y0 = max(0, center_y - half)
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    # Correct if clipping at left/top altered the size.
    x0 = max(0, x1 - size) if x1 == w else x0
    y0 = max(0, y1 - size) if y1 == h else y0
    area = max((x1 - x0) * (y1 - y0), 1)
    return rect_sum(ii, x0, y0, x1, y1) / area


def crop_center_with_pad(arr: np.ndarray, center_x: int, center_y: int, size: int, pad_value: int | Sequence[int]) -> np.ndarray:
    half = size // 2
    x0, y0 = center_x - half, center_y - half
    x1, y1 = x0 + size, y0 + size
    h, w = arr.shape[:2]
    if arr.ndim == 3:
        out = np.empty((size, size, arr.shape[2]), dtype=arr.dtype)
        out[...] = np.array(pad_value, dtype=arr.dtype).reshape(1, 1, -1)
    else:
        out = np.empty((size, size), dtype=arr.dtype)
        out[...] = np.array(pad_value, dtype=arr.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    if sx1 > sx0 and sy1 > sy0:
        out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out


def resize_img(img: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def make_detail_context(slide: SlideData, center_x: int, center_y: int, patch_size: int, context_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    context_size = int(round(patch_size * context_scale))
    detail_img = crop_center_with_pad(slide.he, center_x, center_y, patch_size, [255, 255, 255])
    detail_mask = crop_center_with_pad(slide.label, center_x, center_y, patch_size, 0)
    context_img_raw = crop_center_with_pad(slide.he, center_x, center_y, context_size, [255, 255, 255])
    context_mask_raw = crop_center_with_pad(slide.label, center_x, center_y, context_size, 0)
    context_img = resize_img(context_img_raw, patch_size, is_mask=False)
    context_mask = resize_img(context_mask_raw, patch_size, is_mask=True)
    return detail_img, context_img, detail_mask.astype(np.uint8), context_mask.astype(np.uint8)


def normalize_to_tensor(img: np.ndarray) -> torch.Tensor:
    arr = img.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((mask > 0).astype(np.float32))


def _apply_geom(imgs: List[np.ndarray], masks: List[np.ndarray], rng: np.random.Generator, cfg: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if cfg.get("random_flip", True) and rng.random() < 0.5:
        imgs = [np.ascontiguousarray(np.fliplr(x)) for x in imgs]
        masks = [np.ascontiguousarray(np.fliplr(x)) for x in masks]
    if cfg.get("random_flip", True) and rng.random() < 0.5:
        imgs = [np.ascontiguousarray(np.flipud(x)) for x in imgs]
        masks = [np.ascontiguousarray(np.flipud(x)) for x in masks]
    if cfg.get("random_rot90", True):
        k = int(rng.integers(0, 4))
        if k:
            imgs = [np.ascontiguousarray(np.rot90(x, k)) for x in imgs]
            masks = [np.ascontiguousarray(np.rot90(x, k)) for x in masks]
    return imgs, masks


def _apply_color(img: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    out = img.astype(np.float32)
    if cfg.get("color_jitter", True):
        # brightness/contrast, mild and stable for H&E.
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-15, 15))
        out = out * alpha + beta
        if rng.random() < 0.35:
            hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= float(rng.uniform(0.85, 1.20))
            hsv[..., 2] *= float(rng.uniform(0.90, 1.10))
            out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if cfg.get("gaussian_blur", True) and rng.random() < 0.15:
        out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (3, 3), 0).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_single(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, aug_cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    if not aug_cfg.get("enabled", True):
        return img, mask
    imgs, masks = _apply_geom([img], [mask], rng, aug_cfg)
    return _apply_color(imgs[0], rng, aug_cfg), masks[0]


def augment_dual(detail: np.ndarray, context: np.ndarray, mask: np.ndarray, rng: np.random.Generator, aug_cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not aug_cfg.get("enabled", True):
        return detail, context, mask
    imgs, masks = _apply_geom([detail, context], [mask], rng, aug_cfg)
    # Use the same RNG stream but color jitter is applied to each input separately; this is acceptable because
    # the two inputs represent different magnification appearances.
    return _apply_color(imgs[0], rng, aug_cfg), _apply_color(imgs[1], rng, aug_cfg), masks[0]


def sample_patch_manifest(
    slides: Dict[str, SlideData],
    slide_ids: Sequence[str],
    cfg: dict,
    counts: dict,
    seed: int,
) -> List[dict]:
    patch_size = int(cfg["model"]["patch_size"])
    samp = cfg["sampling"]
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for sid in slide_ids:
        sl = slides[sid]
        tissue_y, tissue_x = np.where(sl.tissue > 0)
        if len(tissue_x) == 0:
            raise ValueError(f"No tissue pixels found in {sid}")
        x_min, x_max = int(tissue_x.min()), int(tissue_x.max())
        y_min, y_max = int(tissue_y.min()), int(tissue_y.max())
        label_ii = integral_image(sl.label)
        tissue_ii = integral_image(sl.tissue)
        buckets = {"normal": [], "tumor": [], "boundary": []}
        needed = {k: int(counts.get(k, 0)) for k in buckets}
        max_needed = max(needed.values()) if needed else 0
        attempts = 0
        max_attempts = int(samp.get("max_attempts_per_class", 80000))
        while attempts < max_attempts and any(len(buckets[k]) < needed[k] for k in buckets):
            attempts += 1
            x = int(rng.integers(x_min, x_max + 1))
            y = int(rng.integers(y_min, y_max + 1))
            tissue_ratio = clipped_patch_ratio(tissue_ii, sl.height, sl.width, x, y, patch_size)
            if tissue_ratio < float(samp["min_tissue_ratio"]):
                continue
            tumor_ratio = clipped_patch_ratio(label_ii, sl.height, sl.width, x, y, patch_size)
            cand = {
                "slide_id": sid,
                "center_x": x,
                "center_y": y,
                "tissue_ratio": tissue_ratio,
                "tumor_ratio": tumor_ratio,
            }
            if (float(samp["boundary_min_tumor_ratio"]) <= tumor_ratio <= float(samp["boundary_max_tumor_ratio"])) and len(buckets["boundary"]) < needed["boundary"]:
                buckets["boundary"].append({**cand, "patch_type": "boundary"})
            elif tumor_ratio >= float(samp["tumor_min_tumor_ratio"]) and len(buckets["tumor"]) < needed["tumor"]:
                buckets["tumor"].append({**cand, "patch_type": "tumor"})
            elif tumor_ratio <= float(samp["normal_max_tumor_ratio"]) and len(buckets["normal"]) < needed["normal"]:
                buckets["normal"].append({**cand, "patch_type": "normal"})
        for k in buckets:
            if len(buckets[k]) < needed[k]:
                print(f"[WARN] {sid}: requested {needed[k]} {k} patches, got {len(buckets[k])}. Try lowering thresholds or increasing attempts.")
            rows.extend(buckets[k])
        print(f"[sample] {sid}: " + ", ".join(f"{k}={len(buckets[k])}" for k in buckets))
    rng.shuffle(rows)
    return rows


def save_manifest(rows: List[dict], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    if not rows:
        raise ValueError("Empty manifest")
    fields = ["slide_id", "center_x", "center_y", "patch_type", "tissue_ratio", "tumor_ratio"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: str | Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["center_x"] = int(float(r["center_x"]))
            r["center_y"] = int(float(r["center_y"]))
            r["tissue_ratio"] = float(r["tissue_ratio"])
            r["tumor_ratio"] = float(r["tumor_ratio"])
            rows.append(r)
    return rows


class SinglePatchDataset(Dataset):
    def __init__(self, rows: List[dict], slides: Dict[str, SlideData], cfg: dict, input_level: str, train: bool):
        assert input_level in {"detail", "context"}
        self.rows = rows
        self.slides = slides
        self.cfg = cfg
        self.input_level = input_level
        self.train = train
        self.patch_size = int(cfg["model"]["patch_size"])
        self.context_scale = float(cfg["model"]["context_scale"])
        self.aug_cfg = cfg.get("augmentation", {})
        self.seed = int(cfg["train"].get("seed", 42))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        sl = self.slides[r["slide_id"]]
        detail, context, detail_mask, context_mask = make_detail_context(sl, r["center_x"], r["center_y"], self.patch_size, self.context_scale)
        if self.input_level == "detail":
            img, mask = detail, detail_mask
        else:
            img, mask = context, context_mask
        if self.train:
            rng = np.random.default_rng(self.seed + idx + int(torch.randint(0, 1000000, (1,)).item()))
            img, mask = augment_single(img, mask, rng, self.aug_cfg)
        return normalize_to_tensor(img), mask_to_tensor(mask)


class DualPatchDataset(Dataset):
    def __init__(self, rows: List[dict], slides: Dict[str, SlideData], cfg: dict, train: bool):
        self.rows = rows
        self.slides = slides
        self.cfg = cfg
        self.train = train
        self.patch_size = int(cfg["model"]["patch_size"])
        self.context_scale = float(cfg["model"]["context_scale"])
        self.aug_cfg = cfg.get("augmentation", {})
        self.seed = int(cfg["train"].get("seed", 42))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        sl = self.slides[r["slide_id"]]
        detail, context, detail_mask, _ = make_detail_context(sl, r["center_x"], r["center_y"], self.patch_size, self.context_scale)
        if self.train:
            rng = np.random.default_rng(self.seed + idx + int(torch.randint(0, 1000000, (1,)).item()))
            detail, context, detail_mask = augment_dual(detail, context, detail_mask, rng, self.aug_cfg)
        return normalize_to_tensor(detail), normalize_to_tensor(context), mask_to_tensor(detail_mask)


def make_single_model(cfg: dict, freeze_encoder: bool = True) -> nn.Module:
    add_project_to_path(cfg["project_root"])
    from dlp.models.model_archs import SingleModelMiTB1

    model = SingleModelMiTB1(feature=95, out_channels=1, dropout=float(cfg["model"].get("dropout", 0.15)))
    weight_path = Path(cfg["weights_dir"]) / cfg["model"].get("pretrained_imagenet", "single_mit_b1_imagenet.pth")
    if not weight_path.exists():
        raise FileNotFoundError(f"ImageNet MiT-B1 weight not found: {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu")
    state_dict.pop("head.weight", None)
    state_dict.pop("head.bias", None)
    load = model.encoder.load_state_dict(state_dict, strict=False)
    if load.unexpected_keys:
        print("[WARN] unexpected ImageNet keys:", load.unexpected_keys[:10])
    set_requires_grad(model.encoder, not freeze_encoder)
    return model


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def init_mca_as_paper(model: nn.Module) -> None:
    crosses = ["first_cross", "second_cross", "third_cross", "fourth_cross"]
    for cross in crosses:
        mod = getattr(model.decoder.mca, cross)
        mod.queries.weight = nn.Parameter(torch.zeros_like(mod.queries.weight))
        mod.keys.weight = nn.Parameter(torch.eye(mod.keys.weight.shape[0], dtype=mod.keys.weight.dtype))
        mod.values.weight = nn.Parameter(torch.eye(mod.values.weight.shape[0], dtype=mod.values.weight.dtype))
        mod.projection.weight = nn.Parameter(torch.eye(mod.projection.weight.shape[0], dtype=mod.projection.weight.dtype))
        mod.queries.bias = nn.Parameter(torch.zeros_like(mod.queries.bias))
        mod.keys.bias = nn.Parameter(torch.zeros_like(mod.keys.bias))
        mod.values.bias = nn.Parameter(torch.zeros_like(mod.values.bias))
        mod.projection.bias = nn.Parameter(torch.zeros_like(mod.projection.bias))


def make_dual_model_from_singles(cfg: dict, detail_path: str | Path, context_path: str | Path, freeze_base: bool = True) -> nn.Module:
    add_project_to_path(cfg["project_root"])
    from dlp.models.model_archs import DualMiTB1CrossAtt

    model = DualMiTB1CrossAtt(heads=8, feature=95, out_channels=1, dropout=float(cfg["model"].get("dropout", 0.15)))
    detail_sd = torch.load(detail_path, map_location="cpu")
    context_sd = torch.load(context_path, map_location="cpu")
    detail_enc = {k.replace("encoder.", "", 1): v for k, v in detail_sd.items() if k.startswith("encoder.")}
    context_enc = {k.replace("encoder.", "", 1): v for k, v in context_sd.items() if k.startswith("encoder.")}
    detail_dec = {k.replace("decoder.", "", 1): v for k, v in detail_sd.items() if k.startswith("decoder.")}
    model.encoder.encoder_detail.load_state_dict(detail_enc, strict=True)
    model.encoder.encoder_context.load_state_dict(context_enc, strict=True)
    model.decoder.lv2decoder.load_state_dict(detail_dec, strict=True)
    init_mca_as_paper(model)
    if freeze_base:
        set_requires_grad(model.encoder.encoder_detail, False)
        set_requires_grad(model.encoder.encoder_context, False)
        set_requires_grad(model.decoder.lv2decoder, False)
        set_requires_grad(model.decoder.mca, True)
    return model


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).mean()


def soft_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(inputs)
    num = 2.0 * (probs * targets).sum(dim=(1, 2, 3))
    den = (probs + targets).sum(dim=(1, 2, 3)) + 1e-7
    return (1.0 - num / den).mean()


def focal_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.95 * focal_loss(inputs, targets) + 0.05 * soft_dice_loss(inputs, targets)


def get_device(cfg: dict) -> torch.device:
    want = str(cfg["train"].get("device", "cuda")).lower()
    if want.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer_single(model: nn.Module, cfg: dict, backbone_trainable: bool) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    groups = [{"params": model.decoder.parameters(), "lr": float(train_cfg["lr_decoder"])}]
    if backbone_trainable:
        groups.append({"params": model.encoder.parameters(), "lr": float(train_cfg["lr_backbone"])})
    return torch.optim.AdamW(groups, weight_decay=float(train_cfg.get("weight_decay", 0.0)))


def build_optimizer_dual(model: nn.Module, cfg: dict, base_trainable: bool) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    groups = [{"params": model.decoder.mca.parameters(), "lr": float(train_cfg["lr_mca"])}]
    if base_trainable:
        groups.extend([
            {"params": model.encoder.encoder_detail.parameters(), "lr": float(train_cfg["lr_backbone"])},
            {"params": model.encoder.encoder_context.parameters(), "lr": float(train_cfg["lr_backbone"])},
            {"params": model.decoder.lv2decoder.parameters(), "lr": float(train_cfg["lr_decoder"])},
        ])
    return torch.optim.AdamW(groups, weight_decay=float(train_cfg.get("weight_decay", 0.0)))


def _make_grad_scaler(device: torch.device, amp: bool):
    enabled = bool(amp and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # compatible with older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device: torch.device, amp: bool):
    enabled = bool(amp and device.type == "cuda")
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except TypeError:  # compatible with older PyTorch
        return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch_single(model, loader, optimizer, device, amp: bool) -> float:
    model.train()
    scaler = _make_grad_scaler(device, amp)
    total = 0.0
    n = 0
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            logits = model(imgs)
            loss = focal_dice_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu()) * imgs.size(0)
        n += imgs.size(0)
    return total / max(n, 1)


def train_one_epoch_dual(model, loader, optimizer, device, amp: bool) -> float:
    model.train()
    scaler = _make_grad_scaler(device, amp)
    total = 0.0
    n = 0
    for detail, context, masks in loader:
        detail = detail.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        masks = masks.unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            logits = model(detail, context)
            loss = focal_dice_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu()) * detail.size(0)
        n += detail.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_patch_single(model, loader, device) -> dict:
    model.eval()
    losses = []
    probs_all = []
    labels_all = []
    for imgs, masks in loader:
        imgs = imgs.to(device)
        y = masks.unsqueeze(1).to(device)
        logits = model(imgs)
        losses.append(float(focal_dice_loss(logits, y).cpu()))
        probs_all.append(torch.sigmoid(logits).cpu().numpy().ravel())
        labels_all.append(y.cpu().numpy().ravel())
    return compute_binary_metrics(np.concatenate(labels_all), np.concatenate(probs_all), threshold=0.5, hd95=False, prefix="") | {"loss": float(np.mean(losses))}


@torch.no_grad()
def eval_patch_dual(model, loader, device) -> dict:
    model.eval()
    losses = []
    probs_all = []
    labels_all = []
    for detail, context, masks in loader:
        detail = detail.to(device)
        context = context.to(device)
        y = masks.unsqueeze(1).to(device)
        logits = model(detail, context)
        losses.append(float(focal_dice_loss(logits, y).cpu()))
        probs_all.append(torch.sigmoid(logits).cpu().numpy().ravel())
        labels_all.append(y.cpu().numpy().ravel())
    return compute_binary_metrics(np.concatenate(labels_all), np.concatenate(probs_all), threshold=0.5, hd95=False, prefix="") | {"loss": float(np.mean(losses))}


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, hd95: bool = False, prefix: str = "", mask_shape: Optional[Tuple[int, int]] = None) -> dict:
    y_true = (y_true.astype(np.float32) > 0.5).astype(np.uint8)
    y_prob = y_prob.astype(np.float32)
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        ap = float("nan")
    out = {
        prefix + "dice": float(dice), prefix + "iou": float(iou), prefix + "precision": float(precision),
        prefix + "recall": float(recall), prefix + "auc": auc, prefix + "ap": ap,
        prefix + "tp": tp, prefix + "fp": fp, prefix + "tn": tn, prefix + "fn": fn,
    }
    if hd95 and mask_shape is not None:
        out[prefix + "hd95"] = float(hd95_distance(y_pred.reshape(mask_shape), y_true.reshape(mask_shape)))
    return out


def hd95_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    pred_border = pred ^ binary_erosion_safe(pred)
    gt_border = gt ^ binary_erosion_safe(gt)
    dt_pred = distance_transform_edt(~pred_border)
    dt_gt = distance_transform_edt(~gt_border)
    d1 = dt_gt[pred_border]
    d2 = dt_pred[gt_border]
    if d1.size == 0 or d2.size == 0:
        return float("nan")
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def binary_erosion_safe(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def postprocess_mask(mask: np.ndarray, cfg: dict) -> np.ndarray:
    infer = cfg.get("inference", {})
    if not infer.get("postprocess", True):
        return mask.astype(np.uint8)
    out = mask.astype(bool)
    closing_radius = int(infer.get("closing_radius", 0))
    if closing_radius > 0:
        out = binary_closing(out, structure=disk(closing_radius))
    min_obj = int(infer.get("remove_small_objects", 0))
    if min_obj > 0:
        out = remove_small_objects(out, min_size=min_obj)
    min_hole = int(infer.get("remove_small_holes", 0))
    if min_hole > 0:
        out = remove_small_holes(out, area_threshold=min_hole)
    out = binary_fill_holes(out)
    return out.astype(np.uint8)


def make_hann_weight(size: int) -> np.ndarray:
    w = np.hanning(size).astype(np.float32)
    ww = np.outer(w, w)
    ww = ww / (ww.max() + 1e-8)
    return np.maximum(ww, 1e-3).astype(np.float32)


def make_positions(length: int, patch: int, stride: int) -> List[int]:
    if length <= patch:
        return [0]
    pos = list(range(0, length - patch + 1, stride))
    last = length - patch
    if not pos or pos[-1] != last:
        pos.append(last)
    return pos


@torch.no_grad()
def infer_whole_slide(model: nn.Module, slide: SlideData, cfg: dict, device: torch.device, mode: str = "dual") -> np.ndarray:
    assert mode in {"dual", "single_detail"}
    model.eval()
    ps = int(cfg["model"]["patch_size"])
    cs = float(cfg["model"]["context_scale"])
    stride = int(cfg["inference"].get("stride", ps // 2))
    min_tissue = float(cfg["inference"].get("min_tissue_ratio", 0.05))
    batch_size = int(cfg["train"].get("eval_batch_size", cfg["train"].get("batch_size", 8)))
    weight = make_hann_weight(ps)
    prob_sum = np.zeros((slide.height, slide.width), dtype=np.float32)
    weight_sum = np.zeros((slide.height, slide.width), dtype=np.float32)
    xs = make_positions(slide.width, ps, stride)
    ys = make_positions(slide.height, ps, stride)
    batch_detail: List[torch.Tensor] = []
    batch_context: List[torch.Tensor] = []
    batch_pos: List[Tuple[int, int]] = []

    def flush():
        if not batch_detail:
            return
        detail_t = torch.stack(batch_detail, dim=0).to(device)
        if mode == "dual":
            context_t = torch.stack(batch_context, dim=0).to(device)
            logits = model(detail_t, context_t)
        else:
            logits = model(detail_t)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
        for prob, (x0, y0) in zip(probs, batch_pos):
            x1 = min(x0 + ps, slide.width)
            y1 = min(y0 + ps, slide.height)
            ph, pw = y1 - y0, x1 - x0
            prob_sum[y0:y1, x0:x1] += prob[:ph, :pw] * weight[:ph, :pw]
            weight_sum[y0:y1, x0:x1] += weight[:ph, :pw]
        batch_detail.clear(); batch_context.clear(); batch_pos.clear()

    for y0 in ys:
        for x0 in xs:
            tissue_ratio = slide.tissue[y0:min(y0 + ps, slide.height), x0:min(x0 + ps, slide.width)].mean()
            if tissue_ratio < min_tissue:
                continue
            cx, cy = int(x0 + ps // 2), int(y0 + ps // 2)
            detail, context, _, _ = make_detail_context(slide, cx, cy, ps, cs)
            batch_detail.append(normalize_to_tensor(detail))
            if mode == "dual":
                batch_context.append(normalize_to_tensor(context))
            batch_pos.append((x0, y0))
            if len(batch_detail) >= batch_size:
                flush()
    flush()
    prob = np.zeros_like(prob_sum, dtype=np.float32)
    valid = weight_sum > 0
    prob[valid] = prob_sum[valid] / np.maximum(weight_sum[valid], 1e-8)
    prob[slide.tissue == 0] = 0.0
    return prob


def evaluate_whole_slide(model: nn.Module, slide: SlideData, cfg: dict, device: torch.device, mode: str = "dual", save_dir: Optional[str | Path] = None) -> dict:
    prob = infer_whole_slide(model, slide, cfg, device, mode=mode)
    threshold = float(cfg["inference"].get("threshold", cfg["train"].get("threshold", 0.5)))
    pred_raw = (prob >= threshold).astype(np.uint8)
    pred = postprocess_mask(pred_raw, cfg)
    pred[slide.tissue == 0] = 0
    valid = slide.tissue.astype(bool)
    metrics = compute_binary_metrics(slide.label[valid].ravel(), prob[valid].ravel(), threshold=threshold, hd95=False)

    # Cleaned-mask metrics are computed only inside the tissue area for Dice/IoU/etc.
    # HD95, however, must be computed on full 2D masks.  Do not reshape the
    # tissue-filtered 1D vectors back to the whole-slide shape.
    metrics_clean = compute_binary_metrics(
        slide.label[valid].ravel(),
        pred[valid].astype(np.float32).ravel(),
        threshold=0.5,
        hd95=False,
    )
    label_2d = slide.label.astype(np.uint8).copy()
    pred_2d = pred.astype(np.uint8).copy()
    label_2d[slide.tissue == 0] = 0
    pred_2d[slide.tissue == 0] = 0
    metrics_clean["hd95"] = float(hd95_distance(pred_2d, label_2d))

    # Use cleaned mask for threshold-dependent metrics and probability map for AUC/AP.
    for k in ["dice", "iou", "precision", "recall", "tp", "fp", "tn", "fn", "hd95"]:
        if k in metrics_clean:
            metrics[k] = metrics_clean[k]
    metrics["selection_score"] = float(0.6 * metrics.get("dice", 0.0) + 0.4 * (0.0 if math.isnan(metrics.get("ap", float("nan"))) else metrics.get("ap", 0.0)))
    if save_dir is not None:
        save_inference_outputs(slide, prob, pred, metrics, save_dir, cfg)
    return metrics


def save_inference_outputs(slide: SlideData, prob: np.ndarray, pred: np.ndarray, metrics: dict, save_dir: str | Path, cfg: dict) -> None:
    save_dir = ensure_dir(save_dir)
    if cfg.get("inference", {}).get("save_prob_npy", True):
        np.save(save_dir / f"{slide.slide_id}_prob.npy", prob.astype(np.float32))
    Image.fromarray((pred * 255).astype(np.uint8)).save(save_dir / f"{slide.slide_id}_pred_mask.png")
    Image.fromarray((slide.label * 255).astype(np.uint8)).save(save_dir / f"{slide.slide_id}_label_mask.png")
    if cfg.get("inference", {}).get("save_visualization", True):
        heat = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        Image.fromarray(heat).save(save_dir / f"{slide.slide_id}_prob_heatmap.png")
        Image.fromarray(make_overlay(slide.he, slide.label, color=(0, 255, 0), alpha=0.42)).save(save_dir / f"{slide.slide_id}_label_overlay.png")
        Image.fromarray(make_overlay(slide.he, pred, color=(255, 220, 0), alpha=0.50)).save(save_dir / f"{slide.slide_id}_pred_overlay.png")
        Image.fromarray(make_pred_only_overlay(slide.he, pred)).save(save_dir / f"{slide.slide_id}_prediction_only_overlay.png")
    with open(save_dir / f"{slide.slide_id}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()}, f, indent=2)


def make_overlay(he: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    out = he.copy().astype(np.float32)
    m = mask.astype(bool)
    c = np.array(color, dtype=np.float32)
    out[m] = out[m] * (1 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_pred_only_overlay(he: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = he.copy().astype(np.float32)
    contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 220, 0), thickness=3)
    return np.clip(out, 0, 255).astype(np.uint8)


def append_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_rows_csv(path: str | Path, rows: List[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    ensure_dir(path.parent)
    fields = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(rows: List[dict], metric_names: Sequence[str]) -> List[dict]:
    out = []
    for name in metric_names:
        vals = []
        for r in rows:
            v = r.get(name)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if not math.isnan(fv):
                vals.append(fv)
        if vals:
            out.append({"metric": name, "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)), "n": len(vals)})
    return out

from __future__ import annotations

"""Prepare ViHBC seed=42 pixel-to-spot prediction cache.

This script does NOT train a model and does NOT require a modified YAML file.
It directly loads the existing checkpoint:

    checkpoints_pathst_cross_p512_ctx4_patchval_overlap_safe/
        seed_42/fold_ViHBC/pathst_ViHBC_best.pth

Workflow
--------
1. Load the existing ViHBC seed=42 checkpoint.
2. Run whole-slide inference only when the PathST-Seg probability cache is absent.
3. Convert PathST-Seg and pixel-level baselines to spot-level tumor scores once.
4. Save a compact NPZ/CSV cache for the downstream plotting script.

After this cache is generated, ROC/PR and all other plots no longer reopen or
rescan the whole-slide prediction masks.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Locate the PathST-Seg project without assuming where this script is placed.
# -----------------------------------------------------------------------------


def discover_project_root() -> Path:
    starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    checked = set()
    for start in starts:
        for candidate in (start, *start.parents):
            candidate = candidate.resolve()
            if candidate in checked:
                continue
            checked.add(candidate)
            if (
                (candidate / "model" / "pathst_net.py").is_file()
                and (candidate / "data" / "st_feature_utils.py").is_file()
                and (candidate / "utils" / "config.py").is_file()
            ):
                return candidate
    raise RuntimeError(
        "Cannot locate the PathST-Seg project root. Place this script inside the "
        "project, or run it from a directory whose parent contains model/, data/, "
        "and utils/."
    )


PROJECT_ROOT = discover_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.context_inputs import extract_aligned_context
from data.st_feature_utils import load_runtime_context_cache, resolve_context_cache_path
from model.pathst_net import build_model
from utils.config import load_config


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
FOLD = "ViHBC"
SEED = 42
SCRIPT_VERSION = "2026-07-16-FINAL-h5ad-coordinate-source-v4"


# Pixel-level baseline files used by the ViHBC seed=42 downstream task.
# iStar must point to the clean binary mask produced by
# ``evaluate_istar_with_pred_mask.py`` rather than to the red H&E overlay.
DEFAULT_TESLA_TEMPLATE = (
    r"D:/Project_2/TESLA/tesla_workspace_three_slices/pred_masks/seed_42/"
    r"ViHBC_seed42_tesla_pred_mask.png"
)
DEFAULT_ISTAR_TEMPLATE = (
    r"D:/Project_2/iStar/istar_workspace_three_slices/pred_masks/seed_42/"
    r"ViHBC_seed42_istar_pred_mask.png"
)
DEFAULT_UNET_TEMPLATE = (
    r"D:/Project_2/Pixel_Seg/baseline/ResNet50UNet/"
    r"predictions_resnet50_wsi_ViHBC/seed_42/pred_mask.png"
)
DEFAULT_AHFUNET_TEMPLATE = (
    r"D:/Project_2/Pixel_Seg/baseline/AHFUNet/"
    r"predictions_baseline_wsi_{fold}/seed_{seed}/pred_mask.png"
)

# -----------------------------------------------------------------------------
# Basic I/O and path resolution
# -----------------------------------------------------------------------------


def resolve_config_path(value: str) -> Path:
    requested = Path(value)
    candidates = [
        requested,
        Path.cwd() / requested,
        Path(__file__).resolve().parent / requested,
        PROJECT_ROOT / requested,
        PROJECT_ROOT / "engine" / requested,
    ]
    # The uploaded file may still carry the automatic "(2)" suffix.
    if requested.name == "cross_dataset.yaml":
        candidates.extend(
            [
                Path.cwd() / "cross_dataset(2).yaml",
                Path(__file__).resolve().parent / "cross_dataset(2).yaml",
                PROJECT_ROOT / "engine" / "cross_dataset(2).yaml",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Cannot find cross-dataset configuration. Checked:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def resolve_relative_existing(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        config_path.parent / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "engine" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Return the most likely engine-relative location for a useful error.
    return (config_path.parent / path).resolve()


def cv_imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, flags)
    if image is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return image


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_for_slice(cfg, slice_id: str):
    for dataset_name, ds_cfg in cfg.datasets.items():
        if slice_id in [str(item) for item in ds_cfg.slices]:
            base = Path(str(ds_cfg.resized_dir))
            if bool(ds_cfg.per_slice_subdir):
                base = base / slice_id
            return str(dataset_name), ds_cfg, base
    raise KeyError(f"Slice {slice_id!r} is not present in cfg.datasets")


def resolve_checkpoint_path(cfg, config_path: Path, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        checkpoint = Path(explicit_path)
        if not checkpoint.is_absolute():
            checkpoint = (Path.cwd() / checkpoint).resolve()
    else:
        root = resolve_relative_existing(str(cfg.paths.checkpoint_dir), config_path)
        checkpoint = root / f"seed_{SEED}" / f"fold_{FOLD}" / f"pathst_{FOLD}_best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Existing seed=42 ViHBC checkpoint was not found:\n"
            f"  {checkpoint}\n"
            "Expected folder structure:\n"
            "  checkpoints_pathst_cross_p512_ctx4_patchval_overlap_safe/"
            "seed_42/fold_ViHBC/pathst_ViHBC_best.pth"
        )
    return checkpoint.resolve()


# -----------------------------------------------------------------------------
# Whole-slide inference. This follows inference_cross_dataset.py but only runs
# ViHBC seed=42 and only produces the continuous probability map needed here.
# -----------------------------------------------------------------------------


def safe_crop(array: np.ndarray, x0: int, y0: int, size: int, pad_mode: str = "constant"):
    height, width = array.shape[:2]
    x1, y1 = x0 + size, y0 + size
    xs, xe = max(0, x0), min(width, x1)
    ys, ye = max(0, y0), min(height, y1)
    crop = array[ys:ye, xs:xe]
    pads = [(max(0, -y0), max(0, y1 - height)), (max(0, -x0), max(0, x1 - width))]
    if array.ndim == 3:
        pads.append((0, 0))
    if any(left or right for left, right in pads):
        try:
            crop = np.pad(crop, pads, mode=pad_mode)
        except ValueError:
            crop = np.pad(crop, pads, mode="constant")
    return crop


def sliding_positions(height: int, width: int, patch: int, stride: int):
    if height < patch or width < patch:
        raise ValueError(f"Image {width}x{height} is smaller than patch size {patch}")
    xs = list(range(0, width - patch + 1, stride))
    ys = list(range(0, height - patch + 1, stride))
    if xs[-1] != width - patch:
        xs.append(width - patch)
    if ys[-1] != height - patch:
        ys.append(height - patch)
    return [(x, y) for y in ys for x in xs]


def gaussian_window(size: int, sigma_factor: float) -> np.ndarray:
    sigma = max(1.0, size * sigma_factor)
    coordinates = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    one_dimensional = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    window = np.outer(one_dimensional, one_dimensional).astype(np.float32)
    return window / max(float(window.max()), 1e-8)


def normalize_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    image = rgb_u8.astype(np.float32) / 255.0
    return ((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def load_runtime_cache(cfg, ds_cfg, fold: str, slice_id: str):
    """Load the model-facing runtime cache and retain its source NPZ path.

    ``load_runtime_context_cache`` constructs a runtime dictionary for model
    inference.  That dictionary is not guaranteed to expose the original
    ``coords_thumb`` field, so downstream spot aggregation must read coordinates
    from the source context-cache NPZ instead of assuming ``cache["coords"]``.
    """
    cache_path = Path(resolve_context_cache_path(cfg, ds_cfg, slice_id, fold)).resolve()
    cache = load_runtime_context_cache(
        str(cache_path),
        cfg=cfg,
        expected_dim=int(cfg.dims.st_dim),
        expected_fold=fold,
    )
    print(
        f"Context cache: spots={len(cache['features'])}, "
        f"thumbnail={cache['thumb_native'].shape[:2]}, "
        f"grid={int(cfg.geometry.context_grid_size)}"
    )
    print(f"Context cache file: {cache_path}")
    print(f"Runtime cache keys: {sorted(cache.keys())}")
    return cache, cache_path


def resolve_vihbc_h5ad_path(ds_cfg, base: Path) -> Path:
    """Resolve the exact AnnData file used by the scGPT preprocessing script."""
    configured = str(getattr(ds_cfg, "h5ad_name", "adata_resized.h5ad"))
    candidates = [
        base / configured,
        base / "adata_resized.h5ad",
        base / "adata_HR_resized.h5ad",
        base / f"{FOLD}_resized.h5ad",
        base / f"{FOLD}.h5ad",
    ]
    seen = set()
    unique = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    for candidate in unique:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot locate the ViHBC AnnData file. Checked:\n  "
        + "\n  ".join(str(path) for path in unique)
    )


def print_context_cache_diagnostics(cfg, ds_cfg, context_cache_path: Path) -> Path:
    """Print the exact context-cache path implied by YAML and by the resolver."""
    context_root = Path(str(ds_cfg.context_cache_root)).resolve()
    cache_tag = str(getattr(cfg.data, "context_cache_tag", ""))
    yaml_expected = (context_root / cache_tag / f"{FOLD}_ctx.npz").resolve()

    print("\n" + "=" * 96)
    print("ViHBC context-cache diagnostics")
    print("=" * 96)
    print(f"YAML context_cache_root : {context_root}")
    print(f"YAML context_cache_tag  : {cache_tag}")
    print(f"YAML-derived ctx path   : {yaml_expected}")
    print(f"Resolver-selected path  : {context_cache_path}")
    print(f"YAML path exists        : {yaml_expected.is_file()}")
    print(f"Resolver path exists    : {context_cache_path.is_file()}")

    if yaml_expected != context_cache_path:
        print("[WARNING] YAML-derived path and resolver-selected path differ.")
        print("The resolver-selected path is the file used by model inference.")

    if context_cache_path.is_file():
        with np.load(context_cache_path, allow_pickle=False) as archive:
            print(f"Context NPZ fields      : {sorted(archive.files)}")
            for key in ("features", "coords_thumb", "coords", "thumb", "tissue_thumb"):
                if key in archive.files:
                    print(f"  {key:<18}: shape={np.asarray(archive[key]).shape}")
            for key in ("full_h", "full_w", "ds", "cache_tag", "feature_dim"):
                if key in archive.files:
                    value = np.asarray(archive[key])
                    print(f"  {key:<18}: {value.item() if value.ndim == 0 else value}")
    print("=" * 96 + "\n")
    return yaml_expected


def load_authoritative_spot_coordinates(
    *,
    h5ad_path: Path,
    context_cache_path: Path,
    runtime_cache,
    full_image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load spot coordinates from AnnData and audit them against the context NPZ.

    ``preprocess_scgpt_breast_external.py`` reads coordinates from
    ``adata.obsm['spatial_resized']`` (falling back to ``'spatial'``), preserves
    the AnnData row order, and only derives ``coords_thumb`` from those
    full-resolution coordinates.  Therefore the AnnData coordinates are the
    authoritative and simplest source for downstream spot aggregation.

    The source context NPZ is still opened and audited so the user can verify
    exactly which file the model loaded and whether its coordinates agree.
    """
    import scanpy as sc

    full_h, full_w = map(int, full_image_shape)
    expected_spots = int(len(runtime_cache["features"]))

    print(f"Authoritative AnnData   : {h5ad_path}")
    adata = sc.read_h5ad(str(h5ad_path))
    coordinate_key = next(
        (key for key in ("spatial_resized", "spatial") if key in adata.obsm),
        None,
    )
    if coordinate_key is None:
        raise KeyError(
            f"{h5ad_path} has no adata.obsm['spatial_resized'] or ['spatial']; "
            f"available keys={list(adata.obsm.keys())}"
        )

    coords_full = np.asarray(adata.obsm[coordinate_key], dtype=np.float32)
    if coords_full.ndim != 2 or coords_full.shape[1] != 2:
        raise ValueError(
            f"Invalid AnnData coordinate shape {coords_full.shape} from {coordinate_key}"
        )
    if coords_full.shape[0] != expected_spots:
        raise ValueError(
            "Spot count mismatch between AnnData coordinates and runtime cache: "
            f"adata={coords_full.shape[0]}, runtime_features={expected_spots}. "
            "This means the context cache and AnnData were not generated from the same spot order."
        )
    if not np.isfinite(coords_full).all():
        raise ValueError(f"AnnData {coordinate_key} contains NaN/Inf")

    thumb_h, thumb_w = map(int, runtime_cache["thumb_native"].shape[:2])
    coords_thumb = coords_full.copy()
    coords_thumb[:, 0] *= float(thumb_w) / float(full_w)
    coords_thumb[:, 1] *= float(thumb_h) / float(full_h)

    audit_text = "context NPZ has no coordinate field; AnnData coordinates used"
    if context_cache_path.is_file():
        with np.load(context_cache_path, allow_pickle=False) as archive:
            files = set(archive.files)
            if "coords_thumb" in files:
                npz_thumb = np.asarray(archive["coords_thumb"], dtype=np.float32)
                if npz_thumb.shape != coords_thumb.shape:
                    raise ValueError(
                        f"coords_thumb shape mismatch: NPZ={npz_thumb.shape}, "
                        f"AnnData-derived={coords_thumb.shape}"
                    )
                max_error = float(np.max(np.abs(npz_thumb - coords_thumb)))
                mean_error = float(np.mean(np.abs(npz_thumb - coords_thumb)))
                print(
                    "Coordinate audit       : context NPZ coords_thumb agrees with "
                    f"AnnData; mean_abs_error={mean_error:.6g}, max_abs_error={max_error:.6g}"
                )
                if max_error > 1e-2:
                    raise ValueError(
                        "Context-cache coords_thumb does not match coordinates derived "
                        f"from {h5ad_path}; max_abs_error={max_error:.6g}"
                    )
                audit_text = (
                    f"AnnData obsm[{coordinate_key!r}] (audited against context NPZ coords_thumb; "
                    f"max_abs_error={max_error:.6g})"
                )
            elif "coords" in files:
                npz_full = np.asarray(archive["coords"], dtype=np.float32)
                if npz_full.shape != coords_full.shape:
                    raise ValueError(
                        f"coords shape mismatch: NPZ={npz_full.shape}, AnnData={coords_full.shape}"
                    )
                max_error = float(np.max(np.abs(npz_full - coords_full)))
                print(
                    "Coordinate audit       : context NPZ coords agrees with AnnData; "
                    f"max_abs_error={max_error:.6g}"
                )
                if max_error > 1e-2:
                    raise ValueError(
                        "Context-cache coords does not match AnnData coordinates; "
                        f"max_abs_error={max_error:.6g}"
                    )
                audit_text = (
                    f"AnnData obsm[{coordinate_key!r}] (audited against context NPZ coords; "
                    f"max_abs_error={max_error:.6g})"
                )
            else:
                print(
                    "Coordinate audit       : selected context NPZ exposes no coordinate key; "
                    "this is acceptable because coordinates are read from the exact AnnData "
                    "used to generate the cache."
                )

    tolerance = 2.0
    x, y = coords_full[:, 0], coords_full[:, 1]
    in_bounds = (
        (x >= -tolerance)
        & (x <= full_w - 1 + tolerance)
        & (y >= -tolerance)
        & (y <= full_h - 1 + tolerance)
    )
    if not bool(np.all(in_bounds)):
        bad = np.flatnonzero(~in_bounds)[:5]
        examples = [coords_full[index].tolist() for index in bad]
        raise ValueError(
            f"AnnData coordinates fall outside H&E bounds {(full_h, full_w)}. "
            f"Examples: {examples}"
        )

    print(f"Coordinate key         : adata.obsm[{coordinate_key!r}]")
    print(f"Coordinate rows        : {coords_full.shape[0]}")
    print(f"Full H&E shape         : {(full_h, full_w)}")
    print(f"Runtime thumbnail      : {(thumb_h, thumb_w)}")
    return coords_full, coords_thumb, audit_text

def load_vihbc_inputs(cfg):
    dataset_name, ds_cfg, base = dataset_for_slice(cfg, FOLD)
    base = base.resolve()
    he_bgr = cv_imread_unicode(str(base / "he_resized.png"), cv2.IMREAD_COLOR)
    he_rgb = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2RGB)

    tissue_path = base / "mask_resized.png"
    if tissue_path.is_file():
        tissue = cv_imread_unicode(str(tissue_path), cv2.IMREAD_GRAYSCALE)
    else:
        tissue = np.full(he_rgb.shape[:2], 255, dtype=np.uint8)

    label_path = base / "label_resized.png"
    label = (
        cv_imread_unicode(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label_path.is_file()
        else None
    )
    if label is None:
        raise FileNotFoundError(f"ViHBC label is required for this downstream task: {label_path}")
    if label.shape != he_rgb.shape[:2] or tissue.shape != he_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: HE={he_rgb.shape[:2]}, label={label.shape}, tissue={tissue.shape}"
        )

    h5ad_path = resolve_vihbc_h5ad_path(ds_cfg, base)
    cache, cache_path = load_runtime_cache(cfg, ds_cfg, FOLD, FOLD)
    print_context_cache_diagnostics(cfg, ds_cfg, cache_path)
    return dataset_name, ds_cfg, base, h5ad_path, he_rgb, label, tissue, cache, cache_path


def load_existing_checkpoint(model, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if str(checkpoint.get("test_slice")) != FOLD:
        raise RuntimeError(
            f"Checkpoint fold mismatch: stored={checkpoint.get('test_slice')}, expected={FOLD}"
        )
    stored_seed = checkpoint.get("random_seed")
    if stored_seed is not None and int(stored_seed) != SEED:
        raise RuntimeError(
            f"Checkpoint seed mismatch: stored={stored_seed}, expected={SEED}"
        )

    state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    if state is None:
        raise RuntimeError(f"Checkpoint has no model state: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [key for key in missing if not key.startswith("detail.vit.")]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={invalid_missing}, unexpected={unexpected}"
        )
    print(f"Loaded checkpoint: {checkpoint_path}")
    return checkpoint


def infer_vihbc_probability(
    model,
    cfg,
    he_rgb: np.ndarray,
    tissue: np.ndarray,
    runtime_cache,
    device: torch.device,
    batch_size: int,
    channels_last: bool,
) -> np.ndarray:
    height, width = he_rgb.shape[:2]
    patch = int(cfg.geometry.patch_size)
    stride = max(1, int(round(patch * (1.0 - float(cfg.inference.overlap)))))
    coordinates = sliding_positions(height, width, patch, stride)
    blend = gaussian_window(patch, float(cfg.inference.gaussian_sigma_factor))
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for start in tqdm(
        range(0, len(coordinates), batch_size),
        desc="Infer ViHBC seed=42",
        ncols=110,
    ):
        chunk = coordinates[start : start + batch_size]
        hr_batch = []
        he_context_batch = []
        context_tissue_batch = []
        spot_features_batch = []
        spot_positions_batch = []
        spot_mask_batch = []

        for x, y in chunk:
            hr = safe_crop(he_rgb, x, y, patch, pad_mode="edge")
            context_data = extract_aligned_context(
                runtime_cache,
                x,
                y,
                cfg,
                is_train=False,
                keep_probability=1.0,
            )
            hr_batch.append(normalize_rgb(hr))
            he_context_batch.append(normalize_rgb(context_data["he_context"]))
            context_tissue_batch.append(
                (context_data["context_tissue"] > 127).astype(np.float32)[None]
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
            context_tissue_tensor = context_tissue_tensor.contiguous(
                memory_format=torch.channels_last
            )

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=bool(cfg.train.amp and device.type == "cuda")
        ):
            batch_probability = model.predict(
                hr_tensor,
                he_context=he_context_tensor,
                spot_features=spot_features_tensor,
                spot_positions=spot_positions_tensor,
                spot_mask=spot_mask_tensor,
                context_tissue=context_tissue_tensor,
            ).float().cpu().numpy()[:, 0]

        for (x, y), patch_probability in zip(chunk, batch_probability):
            probability_sum[y : y + patch, x : x + patch] += patch_probability * blend
            weight_sum[y : y + patch, x : x + patch] += blend

    probability = probability_sum / np.maximum(weight_sum, 1e-8)
    probability[tissue <= 127] = 0.0
    return probability.astype(np.float32)


# -----------------------------------------------------------------------------
# Pixel prediction loading and vectorized spot aggregation
# -----------------------------------------------------------------------------


def resolve_prediction_path(
    path_template: str,
    fold: str,
    seed: Optional[int],
) -> Optional[Path]:
    try:
        raw = str(path_template).format(
            fold=fold,
            seed="" if seed is None else int(seed),
        )
    except Exception:
        raw = str(path_template)
    path = Path(raw)

    if path.is_file():
        return path
    if path.is_dir():
        preferred = (
            "pred_mask.png",
            "prediction.png",
            "mask.png",
            "probability.png",
            "prediction.npz",
        )
        for name in preferred:
            candidate = path / name
            if candidate.is_file():
                return candidate
        for pattern in ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.npy", "*.npz"):
            matches = sorted(path.glob(pattern))
            if matches:
                return matches[0]
        return None

    if path.suffix == "":
        for suffix in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".npy", ".npz"):
            candidate = Path(raw + suffix)
            if candidate.is_file():
                return candidate
    return None


def load_array_from_prediction(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        with np.load(path) as archive:
            for key in ("probability", "prediction", "pred", "mask", "arr_0"):
                if key in archive:
                    return np.asarray(archive[key])
            raise KeyError(f"No supported array key in {path}; keys={list(archive.keys())}")
    return cv_imread_unicode(str(path), cv2.IMREAD_UNCHANGED)


def load_score_map(
    path: Path,
    target_shape: Tuple[int, int],
    score_mode: str,
) -> np.ndarray:
    image = load_array_from_prediction(path)
    image = np.squeeze(np.asarray(image))
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported prediction shape {image.shape}: {path}")
    if image.ndim != 2:
        raise ValueError(f"Prediction must be 2D after squeeze, got {image.shape}: {path}")

    mode = str(score_mode).lower().strip()
    if mode in {"binary", "mask", "label"}:
        array = image.astype(np.float32)
        threshold = 0.5 if float(np.nanmax(array)) <= 1.0 else 127.0
        score = (array > threshold).astype(np.float32)
        interpolation = cv2.INTER_NEAREST
    elif mode in {"probability", "continuous", "score"}:
        score = image.astype(np.float32)
        maximum = float(np.nanmax(score)) if score.size else 1.0
        if maximum > 1.0:
            score = score / (255.0 if maximum <= 255.0 else maximum)
        score = np.clip(score, 0.0, 1.0)
        interpolation = cv2.INTER_LINEAR
    else:
        raise ValueError(f"Unsupported score mode {score_mode!r}")

    target_height, target_width = target_shape
    if score.shape != (target_height, target_width):
        print(
            f"Resize prediction {path.name}: {score.shape} -> "
            f"{(target_height, target_width)}"
        )
        score = cv2.resize(
            score,
            (target_width, target_height),
            interpolation=interpolation,
        )
    return score.astype(np.float32, copy=False)


def mean_prediction_map(
    method: str,
    path_template: str,
    seeds: Sequence[Optional[int]],
    score_mode: str,
    target_shape: Tuple[int, int],
):
    accumulated: Optional[np.ndarray] = None
    sources: List[dict] = []
    for seed in seeds:
        path = resolve_prediction_path(path_template, FOLD, seed)
        if path is None:
            print(
                f"[missing pixel baseline] {method}, seed={seed}: "
                f"{path_template}"
            )
            continue
        try:
            score_map = load_score_map(path, target_shape, score_mode)
        except Exception as exc:
            print(f"[warning] Failed to load {method}, seed={seed}: {exc}")
            continue
        if accumulated is None:
            accumulated = np.zeros(target_shape, dtype=np.float32)
        accumulated += score_map
        sources.append(
            {
                "seed": "single" if seed is None else int(seed),
                "path": str(path),
                "score_mode": score_mode,
            }
        )
        print(f"Loaded pixel baseline: {method}, seed={seed}, path={path}")

    if accumulated is None or not sources:
        return None, sources
    accumulated /= float(len(sources))
    return accumulated, sources


def infer_spot_radius(
    coordinates_xy: np.ndarray,
    factor: float,
    minimum: float,
    maximum: float,
) -> Tuple[float, float]:
    if len(coordinates_xy) < 2:
        return float(minimum), float("nan")
    distances, _ = cKDTree(coordinates_xy.astype(np.float64)).query(
        coordinates_xy,
        k=2,
    )
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    median_nn = float(np.median(nearest)) if nearest.size else float(minimum * 2.0)
    radius = float(np.clip(median_nn * factor, minimum, maximum))
    return radius, median_nn


class DiskSpotAggregator:
    """Fast circular mean aggregation using chunked NumPy indexing."""

    def __init__(
        self,
        coordinates_xy: np.ndarray,
        tissue_mask: np.ndarray,
        radius_px: float,
        max_chunk_elements: int,
    ):
        self.coordinates = np.rint(coordinates_xy).astype(np.int32)
        self.tissue = np.asarray(tissue_mask).astype(bool)
        self.height, self.width = self.tissue.shape

        radius_int = max(1, int(math.ceil(radius_px)))
        yy, xx = np.mgrid[
            -radius_int : radius_int + 1,
            -radius_int : radius_int + 1,
        ]
        inside = (xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2) <= radius_px**2
        self.dx = xx[inside].astype(np.int32)
        self.dy = yy[inside].astype(np.int32)
        self.offset_count = int(len(self.dx))
        self.chunk_size = max(
            1,
            int(max_chunk_elements) // max(1, self.offset_count),
        )

    def aggregate(self, score_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        score_map = np.asarray(score_map, dtype=np.float32)
        if score_map.shape != (self.height, self.width):
            raise ValueError(
                f"Score map shape {score_map.shape} does not match "
                f"tissue shape {(self.height, self.width)}"
            )

        count = len(self.coordinates)
        means = np.full(count, np.nan, dtype=np.float32)
        tissue_fractions = np.zeros(count, dtype=np.float32)

        for start in range(0, count, self.chunk_size):
            stop = min(count, start + self.chunk_size)
            center_x = self.coordinates[start:stop, 0, None]
            center_y = self.coordinates[start:stop, 1, None]
            xx = center_x + self.dx[None, :]
            yy = center_y + self.dy[None, :]

            in_bounds = (
                (xx >= 0)
                & (xx < self.width)
                & (yy >= 0)
                & (yy < self.height)
            )
            safe_x = np.clip(xx, 0, self.width - 1)
            safe_y = np.clip(yy, 0, self.height - 1)
            valid_tissue = in_bounds & self.tissue[safe_y, safe_x]

            tissue_count = valid_tissue.sum(axis=1)
            in_bounds_count = np.maximum(1, in_bounds.sum(axis=1))
            tissue_fractions[start:stop] = (
                tissue_count.astype(np.float32) / in_bounds_count.astype(np.float32)
            )

            values = score_map[safe_y, safe_x]
            numerator = (values * valid_tissue).sum(axis=1, dtype=np.float64)
            valid = tissue_count > 0
            chunk_means = np.full(stop - start, np.nan, dtype=np.float32)
            chunk_means[valid] = (
                numerator[valid] / tissue_count[valid]
            ).astype(np.float32)
            means[start:stop] = chunk_means

        return means, tissue_fractions


# -----------------------------------------------------------------------------
# Existing cache inspection
# -----------------------------------------------------------------------------


def read_cached_method_names(path: Path) -> List[str]:
    if not path.is_file():
        return []
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "methods" not in archive:
                return []
            return [str(value) for value in archive["methods"].tolist()]
    except Exception as exc:
        print(f"[warning] 无法检查旧 spot cache，将重新生成: {exc}")
        return []


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer ViHBC seed=42 from an existing checkpoint and cache spot scores."
    )
    parser.add_argument("--config", default="cross_dataset.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional full checkpoint path. By default it is resolved from "
            "cfg.paths.checkpoint_dir/seed_42/fold_ViHBC/pathst_ViHBC_best.pth."
        ),
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--channels_last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--force_infer",
        action="store_true",
        help="Re-run PathST-Seg whole-slide inference even when its probability cache exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the compact pixel-to-spot cache.",
    )

    parser.add_argument("--tumor_fraction_threshold", type=float, default=0.5)
    parser.add_argument("--min_spot_tissue_fraction", type=float, default=0.25)
    parser.add_argument("--spot_radius_factor", type=float, default=0.50)
    parser.add_argument("--spot_radius_min_px", type=float, default=8.0)
    parser.add_argument("--spot_radius_max_px", type=float, default=48.0)
    parser.add_argument("--aggregation_chunk_elements", type=int, default=2_000_000)

    parser.add_argument("--tesla_template", default=DEFAULT_TESLA_TEMPLATE)
    parser.add_argument("--istar_template", default=DEFAULT_ISTAR_TEMPLATE)
    parser.add_argument(
        "--unet_template", "--cgsnet_template",
        dest="unet_template",
        default=DEFAULT_UNET_TEMPLATE,
        help="ResNet50-UNet binary pred_mask. --cgsnet_template is retained as an alias.",
    )
    parser.add_argument("--ahfunet_template", default=DEFAULT_AHFUNET_TEMPLATE)
    parser.add_argument("--istar_seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--unet_seeds", "--cgsnet_seeds",
        dest="unet_seeds",
        nargs="+", type=int, default=[42],
    )
    parser.add_argument("--ahfunet_seeds", nargs="+", type=int, default=[42, 123, 567])
    parser.add_argument("--tesla_mode", choices=["binary", "probability"], default="binary")
    parser.add_argument("--istar_mode", choices=["binary", "probability"], default="binary")
    parser.add_argument(
        "--unet_mode", "--cgsnet_mode",
        dest="unet_mode",
        choices=["binary", "probability"], default="binary",
    )
    parser.add_argument("--ahfunet_mode", choices=["binary", "probability"], default="binary")
    args = parser.parse_args()
    print(f"prepare_vihbc_seed42_spot_cache version: {SCRIPT_VERSION}")

    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    checkpoint_path = resolve_checkpoint_path(cfg, config_path, args.checkpoint)

    default_cache_dir = (
        Path(str(cfg.paths.output_dir))
        / "downstream_task2"
        / "ViHBC_seed42_cache"
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    probability_cache = cache_dir / "PathST_Seg_ViHBC_seed42_probability.npz"
    spot_cache = cache_dir / "ViHBC_seed42_pixel_to_spot_cache.npz"
    spot_csv = cache_dir / "ViHBC_seed42_pixel_to_spot_cache.csv"
    metadata_json = cache_dir / "ViHBC_seed42_pixel_to_spot_metadata.json"

    if spot_cache.exists() and not args.overwrite and not args.force_infer:
        cached_methods = set(read_cached_method_names(spot_cache))
        required_methods = {"Ours", "TESLA", "iStar", "UNet"}
        missing_methods = sorted(required_methods - cached_methods)
        if not missing_methods:
            print(f"Spot cache already contains all required methods: {spot_cache}")
            print(f"Methods: {sorted(cached_methods)}")
            print("No model inference or pixel-mask aggregation was repeated.")
            return
        print(
            f"Existing spot cache is missing {missing_methods}; it will be rebuilt. "
            "The saved PathST-Seg probability cache will still be reused."
        )

    (
        dataset_name,
        ds_cfg,
        base,
        h5ad_path,
        he_rgb,
        label,
        tissue,
        runtime_cache,
        context_cache_path,
    ) = load_vihbc_inputs(cfg)

    if probability_cache.is_file() and not args.force_infer:
        with np.load(probability_cache) as archive:
            probability = archive["probability"].astype(np.float32)
        print(f"Loaded cached PathST-Seg probability: {probability_cache}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        model = build_model(cfg, build_uni=True).to(device)
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
        checkpoint = load_existing_checkpoint(model, checkpoint_path, device)
        model.eval()
        probability = infer_vihbc_probability(
            model=model,
            cfg=cfg,
            he_rgb=he_rgb,
            tissue=tissue,
            runtime_cache=runtime_cache,
            device=device,
            batch_size=args.batch_size or int(cfg.inference.batch_size),
            channels_last=bool(args.channels_last),
        )
        np.savez_compressed(
            probability_cache,
            probability=probability,
            fold=np.asarray(FOLD),
            seed=np.asarray(SEED, dtype=np.int32),
            checkpoint=np.asarray(str(checkpoint_path)),
            calibrated_threshold=np.asarray(
                float(checkpoint.get("calibrated_threshold", cfg.inference.prob_threshold)),
                dtype=np.float32,
            ),
        )
        print(f"Saved PathST-Seg probability cache: {probability_cache}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if probability.shape != tissue.shape:
        raise ValueError(
            f"Probability shape {probability.shape} does not match ViHBC tissue shape {tissue.shape}"
        )

    # Read authoritative full-resolution coordinates from the exact AnnData
    # used by preprocessing. The context NPZ is opened only for consistency audit;
    # no runtime-cache ``coords``/``coords_thumb`` key is assumed.
    coordinates_xy, coordinates_thumb, coordinate_source = (
        load_authoritative_spot_coordinates(
            h5ad_path=h5ad_path,
            context_cache_path=context_cache_path,
            runtime_cache=runtime_cache,
            full_image_shape=he_rgb.shape[:2],
        )
    )
    spot_radius, median_nn = infer_spot_radius(
        coordinates_xy,
        factor=float(args.spot_radius_factor),
        minimum=float(args.spot_radius_min_px),
        maximum=float(args.spot_radius_max_px),
    )
    print(
        f"ViHBC spots={len(coordinates_xy)}, median nearest-neighbor="
        f"{median_nn:.3f}px, radius={spot_radius:.3f}px"
    )

    aggregator = DiskSpotAggregator(
        coordinates_xy,
        tissue > 127,
        spot_radius,
        max_chunk_elements=int(args.aggregation_chunk_elements),
    )
    gt_fraction, tissue_fraction = aggregator.aggregate((label > 127).astype(np.float32))
    valid = (
        np.isfinite(gt_fraction)
        & (tissue_fraction >= float(args.min_spot_tissue_fraction))
    )
    spot_indices = np.flatnonzero(valid).astype(np.int32)
    gt_fraction_valid = gt_fraction[valid]
    gt_label_valid = (
        gt_fraction_valid >= float(args.tumor_fraction_threshold)
    ).astype(np.uint8)

    method_names: List[str] = []
    method_scores_all: List[np.ndarray] = []
    sources: Dict[str, object] = {}

    ours_spot_score, _ = aggregator.aggregate(probability)
    method_names.append("Ours")
    method_scores_all.append(ours_spot_score.astype(np.float32))
    sources["Ours"] = {
        "seed": SEED,
        "checkpoint": str(checkpoint_path),
        "probability_cache": str(probability_cache),
        "score_mode": "continuous probability",
    }

    pixel_specs = [
        ("TESLA", args.tesla_template, [42], args.tesla_mode),
        ("iStar", args.istar_template, list(args.istar_seeds), args.istar_mode),
        ("UNet", args.unet_template, list(args.unet_seeds), args.unet_mode),
        ("AHFUNet", args.ahfunet_template, list(args.ahfunet_seeds), args.ahfunet_mode),
    ]

    for method, template, seeds, mode in pixel_specs:
        mean_map, loaded_sources = mean_prediction_map(
            method=method,
            path_template=template,
            seeds=seeds,
            score_mode=mode,
            target_shape=tissue.shape,
        )
        if mean_map is None:
            print(f"[skip] No available pixel prediction for {method}")
            continue
        score, _ = aggregator.aggregate(mean_map)
        method_names.append(method)
        method_scores_all.append(score.astype(np.float32))
        sources[method] = {
            "available_prediction_count": len(loaded_sources),
            "sources": loaded_sources,
        }
        del mean_map

    score_matrix_all = np.stack(method_scores_all, axis=1).astype(np.float32)
    score_matrix = score_matrix_all[valid].astype(np.float32)
    np.savez_compressed(
        spot_cache,
        fold=np.asarray(FOLD),
        seed=np.asarray(SEED, dtype=np.int32),
        dataset=np.asarray(dataset_name),
        methods=np.asarray(method_names, dtype="U64"),
        spot_indices=spot_indices,
        coords_xy=coordinates_xy[valid].astype(np.float32),
        coords_thumb=coordinates_thumb[valid].astype(np.float32),
        radius_px=np.asarray(spot_radius, dtype=np.float32),
        median_nn_px=np.asarray(median_nn, dtype=np.float32),
        tissue_fraction=tissue_fraction[valid].astype(np.float32),
        gt_fraction=gt_fraction_valid.astype(np.float32),
        gt_label=gt_label_valid,
        scores=score_matrix,
        all_scores=score_matrix_all,
        valid_mask=valid.astype(np.uint8),
        num_original_spots=np.asarray(len(coordinates_xy), dtype=np.int32),
    )

    rows = []
    for position, original_index in enumerate(spot_indices):
        row = {
            "Dataset": dataset_name,
            "Fold": FOLD,
            "Seed": SEED,
            "Spot_Index": int(original_index),
            "x_pixel": float(coordinates_xy[original_index, 0]),
            "y_pixel": float(coordinates_xy[original_index, 1]),
            "Spot_Radius_Pixel": float(spot_radius),
            "Tissue_Fraction": float(tissue_fraction[original_index]),
            "GT_TumorFraction": float(gt_fraction[original_index]),
            "GT_TumorLabel": int(gt_label_valid[position]),
        }
        for method_index, method in enumerate(method_names):
            row[f"Score_{method}"] = float(score_matrix[position, method_index])
        rows.append(row)
    write_csv(spot_csv, rows)

    metadata = {
        "dataset": dataset_name,
        "fold": FOLD,
        "seed": SEED,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "vihbc_data_directory": str(base),
        "context_cache_path": str(context_cache_path),
        "h5ad_path": str(h5ad_path),
        "coordinate_source": coordinate_source,
        "num_original_spots": int(len(coordinates_xy)),
        "num_valid_spots": int(valid.sum()),
        "spot_radius_px": float(spot_radius),
        "median_nearest_neighbor_px": float(median_nn),
        "min_spot_tissue_fraction": float(args.min_spot_tissue_fraction),
        "tumor_fraction_threshold": float(args.tumor_fraction_threshold),
        "methods": method_names,
        "sources": sources,
        "speed_note": (
            "Whole-slide masks were converted to spot scores once. Subsequent "
            "ROC/PR and plotting operate only on this compact cache."
        ),
    }
    with open(metadata_json, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("\nCache preparation completed.")
    print(f"PathST-Seg full probability cache: {probability_cache}")
    print(f"Pixel-to-spot cache: {spot_cache}")
    print(f"Readable spot table: {spot_csv}")
    print(f"Metadata: {metadata_json}")
    print(f"Methods cached: {method_names}")


if __name__ == "__main__":
    main()

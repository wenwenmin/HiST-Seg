from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.context_inputs import extract_aligned_context, transform_relative_positions
from data.st_feature_utils import load_runtime_context_cache
from utils.stain_aug import hed_jitter, strong_color_augment

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
_IMAGE_EXT_PRIORITY = {".jpg": 1, ".jpeg": 1, ".png": 2}


def _cfg_get(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default)


def _imread(path: str, flags: int) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def _parse_name(filename: str) -> Tuple[str, int, int]:
    match = re.match(r"(.+)_x(\d+)_y(\d+)\.(?:jpg|jpeg|png)$", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse patch coordinates from: {filename}")
    return match.group(1), int(match.group(2)), int(match.group(3))


class PathSTDataset(Dataset):
    """Aligned detail/H&E-context/ST-spot dataset.

    Each sample contains a configurable detail patch (512x512 in the supplied
    configs), an H&E context crop covering exactly the same physical field as the
    selected ST spots, padded raw 512-D spot embeddings, relative spot
    coordinates, and masks. The learnable spot encoder and rasterizer run inside
    the model; no dense 512-channel ST map is constructed in the data loader.
    """

    def __init__(
        self,
        patch_dir: str,
        ctx_cache_dir: str,
        allowed_slices: Optional[Sequence[str]],
        cfg: Any,
        is_train: bool = True,
        return_meta: bool = False,
        metadata_cache_name: str = "patch_metadata.csv",
        expected_fold: Optional[str] = None,
    ):
        self.cfg = cfg
        # Optional fold identifier used only to validate legacy fold-aware caches.
        # Current all-gene scGPT caches are label-free and normally omit this field.
        self.expected_fold = None if expected_fold is None else str(expected_fold)
        self.is_train = bool(is_train)
        self.return_meta = bool(return_meta)
        self.P = int(cfg.geometry.patch_size)
        self.context_image_size = int(cfg.geometry.context_image_size)
        self.context_grid_size = int(cfg.geometry.context_grid_size)
        self.cache_ds = int(cfg.geometry.downsample)
        self.st_dim = int(cfg.dims.st_dim)
        self.max_context_spots = int(getattr(cfg.data, "max_context_spots", 1024))
        self.min_tissue_ratio = float(_cfg_get(cfg.data, "min_tissue_ratio", 0.0))
        self.strict_cache = bool(_cfg_get(cfg.data, "strict_context_cache", True))

        self.patch_dir = patch_dir
        self.img_folder = os.path.join(patch_dir, "images")
        self.msk_folder = os.path.join(patch_dir, "masks")
        self.tis_folder = os.path.join(patch_dir, "tissue_masks")
        self.metadata_path = os.path.join(patch_dir, metadata_cache_name)
        for required in (self.img_folder, self.msk_folder):
            if not os.path.isdir(required):
                raise NotADirectoryError(required)


        allowed = set(allowed_slices or [])
        preferred: Dict[str, str] = {}
        for filename in os.listdir(self.img_folder):
            stem, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext not in _IMAGE_EXT_PRIORITY:
                continue
            try:
                sid, _, _ = _parse_name(filename)
            except ValueError:
                continue
            if allowed and sid not in allowed:
                continue
            previous = preferred.get(stem)
            if previous is None or _IMAGE_EXT_PRIORITY[ext] > _IMAGE_EXT_PRIORITY[os.path.splitext(previous)[1].lower()]:
                preferred[stem] = filename

        cached = self._load_metadata_cache()
        rows: List[Dict[str, Any]] = []
        changed = False
        for stem in sorted(preferred):
            filename = preferred[stem]
            sid, x, y = _parse_name(filename)
            row = cached.get(stem)
            if row is None:
                row = self._compute_patch_stats(stem, filename, sid, x, y)
                changed = True
            else:
                row = dict(row)
                row.update(filename=filename, slice_id=sid, x=x, y=y)
            if float(row["tissue_ratio"]) >= self.min_tissue_ratio:
                rows.append(row)
        self.items = rows
        if changed:
            merged = dict(cached)
            merged.update({row["stem"]: row for row in rows})
            self._write_metadata_cache(merged.values())

        self.ctx: Dict[str, Dict[str, np.ndarray]] = {}
        context_slices = sorted({str(row["slice_id"]) for row in rows})
        for sid in context_slices:
            path = os.path.join(ctx_cache_dir, f"{sid}_ctx.npz")
            if not os.path.exists(path):
                if self.strict_cache:
                    raise FileNotFoundError(
                        f"Missing ST cache: {path}. Run the matching scGPT preprocessing script first."
                    )
                continue
            runtime_cache = load_runtime_context_cache(
                path,
                cfg=self.cfg,
                expected_dim=self.st_dim,
                expected_fold=self.expected_fold,
            )
            features = runtime_cache["features"]
            norm_stats = runtime_cache["normalization_stats"]
            self.ctx[sid] = {
                "thumb_native": runtime_cache["thumb_native"],
                "tissue_thumb_native": runtime_cache["tissue_thumb_native"],
                "coords_thumb_native": runtime_cache["coords_thumb_native"],
                "features": features,
                "cache_downsample": np.int64(runtime_cache["cache_downsample"]),
                "full_height": np.int64(runtime_cache["full_height"]),
                "full_width": np.int64(runtime_cache["full_width"]),
            }
            print(
                f"  ST-cache {sid}: spots={len(features)} "
                f"context={self.context_image_size}px grid={self.context_grid_size} "
                f"cache_ds={int(runtime_cache['cache_downsample'])} "
                f"norm_std={norm_stats['after_std']:.4f}"
            )

        print(
            f"Dataset patches={len(self.items)} slices={context_slices} "
            f"fold={self.expected_fold} UNI=online-frozen"
        )

    def _load_metadata_cache(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.metadata_path):
            return {}
        result: Dict[str, Dict[str, Any]] = {}
        with open(self.metadata_path, "r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    result[row["stem"]] = {
                        "stem": row["stem"], "filename": row["filename"],
                        "slice_id": row["slice_id"], "x": int(row["x"]), "y": int(row["y"]),
                        "tissue_pixels": int(float(row["tissue_pixels"])),
                        "cancer_pixels": int(float(row["cancer_pixels"])),
                        "tissue_ratio": float(row["tissue_ratio"]),
                        "cancer_ratio": float(row["cancer_ratio"]),
                    }
                except Exception:
                    continue
        return result

    def _write_metadata_cache(self, rows: Iterable[Dict[str, Any]]) -> None:
        fields = [
            "stem", "filename", "slice_id", "x", "y", "tissue_pixels", "cancer_pixels",
            "tissue_ratio", "cancer_ratio",
        ]
        with open(self.metadata_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in sorted(rows, key=lambda z: (str(z["slice_id"]), int(z["y"]), int(z["x"]))):
                writer.writerow({key: row[key] for key in fields})

    def _compute_patch_stats(self, stem: str, filename: str, sid: str, x: int, y: int):
        mask = _imread(os.path.join(self.msk_folder, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        tissue = _imread(os.path.join(self.tis_folder, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(os.path.join(self.msk_folder, stem + ".png"))
        tissue_bool = np.ones(mask.shape, bool) if tissue is None else tissue > 127
        cancer_bool = (mask > 127) & tissue_bool
        tissue_pixels = int(tissue_bool.sum())
        cancer_pixels = int(cancer_bool.sum())
        return {
            "stem": stem, "filename": filename, "slice_id": sid, "x": x, "y": y,
            "tissue_pixels": tissue_pixels, "cancer_pixels": cancer_pixels,
            "tissue_ratio": tissue_pixels / max(1, mask.size),
            "cancer_ratio": cancer_pixels / max(1, tissue_pixels),
        }

    def __len__(self):
        return len(self.items)

    def get_meta(self, index: int) -> Dict[str, Any]:
        return dict(self.items[index])

    def _sample_geom(self):
        if not self.is_train:
            return False, False, 0
        return np.random.rand() < 0.5, np.random.rand() < 0.5, int(np.random.randint(4))

    @staticmethod
    def _apply_geom(arrays: List[np.ndarray], params):
        hflip, vflip, k = params
        if hflip:
            arrays = [array[:, ::-1].copy() for array in arrays]
        if vflip:
            arrays = [array[::-1, :].copy() for array in arrays]
        if k:
            arrays = [np.rot90(array, k).copy() for array in arrays]
        return arrays

    @staticmethod
    def _apply_geom_tokens(tokens: np.ndarray, params):
        hflip, vflip, k = params
        if hflip:
            tokens = tokens[..., ::-1].copy()
        if vflip:
            tokens = tokens[..., ::-1, :].copy()
        if k:
            tokens = np.rot90(tokens, k, axes=(-2, -1)).copy()
        return tokens

    def _photometric(self, image: np.ndarray) -> np.ndarray:
        if not self.is_train:
            return image

        aug_cfg = getattr(getattr(self.cfg, "train", None), "strong_color_aug", None)
        strong_enabled = bool(_cfg_get(aug_cfg, "enabled", False))

        if strong_enabled:
            strength_scale = 1.0
            image = strong_color_augment(
                image,
                hed_p=float(_cfg_get(aug_cfg, "hed_p", 0.90)),
                hed_sigma=float(_cfg_get(aug_cfg, "hed_sigma", 0.06)) * strength_scale,
                hed_bias=float(_cfg_get(aug_cfg, "hed_bias", 0.06)) * strength_scale,
                bc_p=float(_cfg_get(aug_cfg, "bc_p", 0.85)),
                brightness=float(_cfg_get(aug_cfg, "brightness", 0.22)) * strength_scale,
                contrast=float(_cfg_get(aug_cfg, "contrast", 0.22)) * strength_scale,
                hsv_p=float(_cfg_get(aug_cfg, "hsv_p", 0.55)),
                saturation=float(_cfg_get(aug_cfg, "saturation", 0.18)) * strength_scale,
                hue=float(_cfg_get(aug_cfg, "hue", 0.035)) * strength_scale,
                gamma_p=float(_cfg_get(aug_cfg, "gamma_p", 0.45)),
                gamma=float(_cfg_get(aug_cfg, "gamma", 0.20)) * strength_scale,
                blur_p=float(_cfg_get(aug_cfg, "blur_p", 0.15)),
                noise_p=float(_cfg_get(aug_cfg, "noise_p", 0.15)),
                noise_std=float(_cfg_get(aug_cfg, "noise_std", 0.012)) * strength_scale,
            )
            return image.astype(np.float32)

        image = hed_jitter(image, sigma=0.03, bias=0.03, p=0.7)
        amplitude = 0.12
        if np.random.rand() < 0.5:
            brightness = 1.0 + np.random.uniform(-amplitude, amplitude)
            contrast = 1.0 + np.random.uniform(-amplitude, amplitude)
            mean = float(image.mean())
            image = np.clip((image - mean) * contrast + mean * brightness, 0.0, 1.0)
        return image.astype(np.float32)

    @staticmethod
    def _image_tensor(image: np.ndarray) -> torch.Tensor:
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float()

    def __getitem__(self, index: int):
        item = self.items[index]
        stem, filename, sid = str(item["stem"]), str(item["filename"]), str(item["slice_id"])
        x0, y0 = int(item["x"]), int(item["y"])
        image = _imread(os.path.join(self.img_folder, filename), cv2.IMREAD_COLOR)
        mask = _imread(os.path.join(self.msk_folder, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        tissue = _imread(os.path.join(self.tis_folder, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(stem)
        if tissue is None:
            tissue = np.full(mask.shape, 255, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cache = self.ctx.get(sid)
        keep_probability = 1.0

        if cache is None:
            context_data = {
                "he_context": np.zeros(
                    (self.context_image_size, self.context_image_size, 3), dtype=np.uint8
                ),
                "context_tissue": np.zeros(
                    (self.context_image_size, self.context_image_size), dtype=np.uint8
                ),
                "spot_features": np.zeros(
                    (self.max_context_spots, self.st_dim), dtype=np.float32
                ),
                "spot_positions": np.zeros(
                    (self.max_context_spots, 2), dtype=np.float32
                ),
                "spot_mask": np.zeros((self.max_context_spots,), dtype=np.float32),
                "spot_count": np.asarray(0, dtype=np.int64),
            }
        else:
            context_data = extract_aligned_context(
                cache,
                x0,
                y0,
                self.cfg,
                is_train=self.is_train,
                keep_probability=keep_probability,
            )

        context_image = context_data["he_context"]
        context_tissue = context_data["context_tissue"]
        spot_features = context_data["spot_features"]
        spot_positions = context_data["spot_positions"]
        spot_mask = context_data["spot_mask"]

        image = image.astype(np.float32) / 255.0
        context_image = context_image.astype(np.float32) / 255.0
        mask_f = (mask > 127).astype(np.float32)[..., None]
        tissue_f = (tissue > 127).astype(np.float32)[..., None]
        context_tissue_f = (context_tissue > 127).astype(np.float32)[..., None]

        geom = self._sample_geom()
        image, mask_f, tissue_f, context_image, context_tissue_f = self._apply_geom(
            [image, mask_f, tissue_f, context_image, context_tissue_f], geom
        )
        spot_positions = transform_relative_positions(
            spot_positions,
            hflip=geom[0],
            vflip=geom[1],
            rotations_ccw=geom[2],
        )

        # Reuse the same random photometric parameters for detail and context.
        # Spatial noise remains resolution-specific, which is desirable.
        if self.is_train:
            random_state = np.random.get_state()
            image = self._photometric(image)
            np.random.set_state(random_state)
            context_image = self._photometric(context_image)
        else:
            image = self._photometric(image)
            context_image = self._photometric(context_image)

        sample: Dict[str, Any] = {
            "hr": self._image_tensor(image),
            "he_context": self._image_tensor(context_image),
            "context_tissue": torch.from_numpy(
                context_tissue_f.transpose(2, 0, 1).copy()
            ).float(),
            "spot_features": torch.from_numpy(spot_features.copy()).float(),
            "spot_positions": torch.from_numpy(spot_positions.copy()).float(),
            "spot_mask": torch.from_numpy(spot_mask.copy()).float(),
            "spot_count": torch.as_tensor(int(context_data["spot_count"]), dtype=torch.long),
            "mask": torch.from_numpy(mask_f.transpose(2, 0, 1).copy()).float(),
            "tissue": torch.from_numpy(tissue_f.transpose(2, 0, 1).copy()).float(),
        }
        if self.return_meta:
            sample["slice_id"] = sid
            sample["x"] = x0
            sample["y"] = y0
        return sample

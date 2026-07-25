"""Extract lossless 512x512 patches for STHBC and the three external datasets.

All paths and slice lists come from ``engine/cross_dataset.yaml`` so patch
preprocessing, training, and inference share one source of truth.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

DATASET_NAMES = ("STHBC", "ViHBC", "DuCIS", "XeHBC")


def cv_imread(path: str, flags: int):
    if not os.path.exists(path):
        return None
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def cv_imwrite(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    suffix = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise IOError(f"Failed to encode image: {path}")
    encoded.tofile(path)


def resolve_config_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    candidate = Path(__file__).resolve().parent / value
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(value)


def collect_positions(height: int, width: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    """Match whole-slide inference coordinates exactly, including right/bottom edges."""
    if height < patch_size or width < patch_size:
        raise ValueError(f"Image {width}x{height} is smaller than patch_size={patch_size}")
    xs = list(range(0, width - patch_size + 1, stride))
    ys = list(range(0, height - patch_size + 1, stride))
    if xs[-1] != width - patch_size:
        xs.append(width - patch_size)
    if ys[-1] != height - patch_size:
        ys.append(height - patch_size)
    return [(x, y) for y in ys for x in xs]


def extract_slice(
    *,
    slice_id: str,
    image_path: str,
    label_path: str,
    tissue_path: str,
    output_dir: str,
    patch_size: int,
    overlap: float,
    min_tissue_ratio: float,
    overwrite: bool,
) -> None:
    image = cv_imread(image_path, cv2.IMREAD_COLOR)
    label = cv_imread(label_path, cv2.IMREAD_GRAYSCALE)
    tissue = cv_imread(tissue_path, cv2.IMREAD_GRAYSCALE)
    if image is None or label is None:
        raise FileNotFoundError(f"[{slice_id}] missing image/label: {image_path}, {label_path}")
    if tissue is None:
        tissue = np.full(label.shape, 255, np.uint8)
    if image.shape[:2] != label.shape[:2] or image.shape[:2] != tissue.shape[:2]:
        raise ValueError(
            f"[{slice_id}] shape mismatch: image={image.shape[:2]}, "
            f"label={label.shape[:2]}, tissue={tissue.shape[:2]}"
        )

    image_out = os.path.join(output_dir, "images")
    mask_out = os.path.join(output_dir, "masks")
    tissue_out = os.path.join(output_dir, "tissue_masks")
    for directory in (image_out, mask_out, tissue_out):
        os.makedirs(directory, exist_ok=True)

    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    rows: List[Dict[str, object]] = []
    skipped = 0
    for x, y in tqdm(
        collect_positions(image.shape[0], image.shape[1], patch_size, stride),
        desc=f"patch {slice_id}",
        ncols=100,
    ):
        image_patch = image[y:y + patch_size, x:x + patch_size]
        label_patch = label[y:y + patch_size, x:x + patch_size]
        tissue_patch = tissue[y:y + patch_size, x:x + patch_size]
        tissue_bool = tissue_patch > 127
        tissue_pixels = int(tissue_bool.sum())
        tissue_ratio = tissue_pixels / float(patch_size * patch_size)
        if tissue_ratio < min_tissue_ratio:
            skipped += 1
            continue

        cancer_pixels = int(((label_patch > 127) & tissue_bool).sum())
        cancer_ratio = cancer_pixels / max(1, tissue_pixels)
        stem = f"{slice_id}_x{x}_y{y}"
        cv_imwrite(os.path.join(image_out, stem + ".png"), image_patch)
        cv_imwrite(os.path.join(mask_out, stem + ".png"), label_patch)
        cv_imwrite(os.path.join(tissue_out, stem + ".png"), tissue_patch)
        rows.append({
            "stem": stem,
            "filename": stem + ".png",
            "slice_id": slice_id,
            "x": x,
            "y": y,
            "tissue_pixels": tissue_pixels,
            "cancer_pixels": cancer_pixels,
            "tissue_ratio": f"{tissue_ratio:.8f}",
            "cancer_ratio": f"{cancer_ratio:.8f}",
        })

    metadata_path = os.path.join(output_dir, "patch_metadata.csv")
    existing = []
    if os.path.exists(metadata_path) and not overwrite:
        with open(metadata_path, "r", newline="", encoding="utf-8-sig") as handle:
            existing = [row for row in csv.DictReader(handle) if row.get("slice_id") != slice_id]
    fields = [
        "stem", "filename", "slice_id", "x", "y",
        "tissue_pixels", "cancer_pixels", "tissue_ratio", "cancer_ratio",
    ]
    with open(metadata_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + rows)

    print(
        f"[{slice_id}] kept={len(rows)} skipped={skipped} "
        f"image={image.shape[1]}x{image.shape[0]} patch={patch_size} stride={stride}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../engine/cross_dataset.yaml")
    parser.add_argument("--dataset", choices=(*DATASET_NAMES, "all"), default="all")
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min_tissue_ratio", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    patch_size = int(args.patch_size or cfg["geometry"]["patch_size"])
    selected = DATASET_NAMES if args.dataset == "all" else (args.dataset,)

    for dataset_name in selected:
        dataset_cfg = cfg["datasets"][dataset_name]
        output_dir = str(dataset_cfg["patch_dir"])
        if args.overwrite and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        print(f"\n====== {dataset_name} ======")
        for slice_id in map(str, dataset_cfg["slices"]):
            base = (
                os.path.join(str(dataset_cfg["resized_dir"]), slice_id)
                if bool(dataset_cfg.get("per_slice_subdir", False))
                else str(dataset_cfg["resized_dir"])
            )
            extract_slice(
                slice_id=slice_id,
                image_path=os.path.join(base, "he_resized.png"),
                label_path=os.path.join(base, "label_resized.png"),
                tissue_path=os.path.join(base, "mask_resized.png"),
                output_dir=output_dir,
                patch_size=patch_size,
                overlap=float(args.overlap),
                min_tissue_ratio=float(args.min_tissue_ratio),
                overwrite=bool(args.overwrite),
            )

    print("Done.")


if __name__ == "__main__":
    main()

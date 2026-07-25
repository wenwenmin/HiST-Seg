"""Extract lossless 512x512 CRC patches using inference-consistent coordinates."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


CRC_SLICES = (
    "SN048_1", "SN048_2", "SN048_3", "SN048_4",
    "SN84_1", "SN84_2", "SN123_1", "SN123_2",
    "SN123_4", "SN124_1", "SN124_2", "SN124_4",
)


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


def collect_positions(height: int, width: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
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
        raise FileNotFoundError(f"[{slice_id}] missing image/label")
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
    positions = collect_positions(image.shape[0], image.shape[1], patch_size, stride)
    rows = []
    skipped = 0
    for x, y in tqdm(positions, desc=f"patch {slice_id}", ncols=100):
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
    parser.add_argument("--config", default="../engine/crc_loso.yaml")
    parser.add_argument("--slices", nargs="+", default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min_tissue_ratio", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    dataset_cfg = cfg["datasets"]["CRC"]
    patch_size = int(args.patch_size or cfg["geometry"]["patch_size"])
    slices = [str(value) for value in (args.slices or dataset_cfg["slices"])]
    unknown = sorted(set(slices) - set(CRC_SLICES))
    if unknown:
        raise ValueError(f"Unknown CRC slices: {unknown}")

    output_dir = str(dataset_cfg["patch_dir"])
    if args.overwrite and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    for slice_id in slices:
        extract_slice(
            slice_id=slice_id,
            image_path=os.path.join(str(dataset_cfg["image_dir"]), f"{slice_id}.png"),
            label_path=os.path.join(str(dataset_cfg["label_dir"]), f"{slice_id}.png"),
            tissue_path=os.path.join(str(dataset_cfg["tissue_dir"]), f"{slice_id}.png"),
            output_dir=output_dir,
            patch_size=patch_size,
            overlap=float(args.overlap),
            min_tissue_ratio=float(args.min_tissue_ratio),
            overwrite=bool(args.overwrite),
        )

    print(f"Done. Patches saved to: {output_dir}")


if __name__ == "__main__":
    main()

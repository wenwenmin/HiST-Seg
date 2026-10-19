r"""
TESLA spot-level prediction visualization for STHBC, VDX and CRC.

Only seed 42 is visualized.

Probability rendering:
    probability 0 -> white
    probability 1 -> deep red
    intermediate values -> linear white-to-red transition

The script generates:
    1. probability overlay image
    2. binary prediction overlay image
    3. optional side-by-side image: original H&E | probability overlay

Output root:
D:\DNM_project(cancer detect)\predict_files\TESLA\spot_level\visualize
    ├── STHBC
    ├── VDX
    └── CRC
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import cv2
import numpy as np
from tqdm import tqdm


# ============================================================================
# Global settings
# ============================================================================

SEED = 42
PRED_THRESHOLD = 0.5

# Longest side of the saved image. Large images are downsampled only for display.
MAX_OUTPUT_DIMENSION = 4000

# 1.0 reproduces opaque circles used in the supplied visualization scripts.
# A smaller value, such as 0.80, keeps more H&E morphology visible.
SPOT_ALPHA = 0.90

# Draw a thin black outline around every spot.
DRAW_SPOT_BORDER = False
SPOT_BORDER_THICKNESS = 1

# Save original H&E and probability overlay side by side.
SAVE_SIDE_BY_SIDE = True

# Save binary red/white spot maps in addition to probability maps.
SAVE_BINARY_MAP = True

# If more than this proportion of spot centers falls outside the image,
# the script stops for that slice because the selected image is likely misaligned.
MAX_OUTSIDE_FRACTION = 0.05

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


# ============================================================================
# Dataset paths
# ============================================================================

STHBC_SLICES = [
    "A1",
    "B1",
    "C1",
    "D1",
    "E1",
    "F1",
    "G2",
    "H1",
]

VDX_SLICES = [
    "ViHBC",
    "DuCIS",
    "XeHBC",
]

CRC_SLICES = [
    "SN048_1",
    "SN048_2",
    "SN048_3",
    "SN048_4",
    "SN84_1",
    "SN84_2",
    "SN123_1",
    "SN123_2",
    "SN123_4",
    "SN124_1",
    "SN124_2",
    "SN124_4",
]

PIXEL_SEG_DATA_ROOT = Path(
    r"D:\Project_2\Pixel_Seg\data"
)

TESLA_PREDICTION_ROOT = Path(
    r"D:\DNM_project(cancer detect)\predict_files\TESLA"
)

OUTPUT_ROOT = (
    TESLA_PREDICTION_ROOT
    / "spot_level"
    / "visualize"
)


# ============================================================================
# Optional manual image-path overrides
# ============================================================================

# Normally the script searches for a suitable image automatically.
# If automatic discovery selects the wrong image or cannot find one,
# fill in the exact H&E image path here.
#
# Example:
# IMAGE_PATH_OVERRIDES["STHBC"]["A1"] = Path(
#     r"D:\...\A1\he_resized.jpg"
# )
#
# Example:
# IMAGE_PATH_OVERRIDES["CRC"]["SN048_1"] = Path(
#     r"D:\...\SN048_1_cropped.png"
# )

IMAGE_PATH_OVERRIDES: dict[str, dict[str, Path]] = {
    "STHBC": {},
    "VDX": {},
    "CRC": {},
}


# ============================================================================
# Slice configuration
# ============================================================================

@dataclass(frozen=True)
class SliceConfig:
    dataset: str
    slice_name: str
    adata_path: Path
    coordinate_key: str
    probability_path: Path
    prediction_path: Path
    output_dir: Path
    spot_radius: float
    image_search_roots: tuple[Path, ...]
    image_name_keywords: tuple[str, ...]


def read_scale_factor(scale_path: Path) -> float:
    """
    Read `scale_factor` from a text file.

    Supported examples:
        scale_factor: 0.324834
        scale_factor = 0.324834
        "scale_factor": 0.324834
    """
    if not scale_path.exists():
        raise FileNotFoundError(
            f"找不到缩放文件：{scale_path}"
        )

    text = scale_path.read_text(
        encoding="utf-8-sig",
        errors="strict",
    )

    number_pattern = (
        r"[-+]?"
        r"(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?"
    )

    match = re.search(
        rf"""["']?scale_factor["']?\s*[:=]\s*
             (?P<value>{number_pattern})""",
        text,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    if match is None:
        raise ValueError(
            f"未能在 {scale_path} 中读取 scale_factor"
        )

    scale_factor = float(
        match.group("value")
    )

    if (
        not np.isfinite(scale_factor)
        or scale_factor <= 0
    ):
        raise ValueError(
            f"scale_factor 必须为正数，读取值为 {scale_factor}"
        )

    return scale_factor


def build_sthbc_configs() -> list[SliceConfig]:
    configs: list[SliceConfig] = []

    for slice_name in STHBC_SLICES:
        slice_dir = (
            PIXEL_SEG_DATA_ROOT
            / "STHBC"
            / "resized_data"
            / slice_name
        )

        configs.append(
            SliceConfig(
                dataset="STHBC",
                slice_name=slice_name,
                adata_path=(
                    slice_dir
                    / "adata_resized.h5ad"
                ),
                coordinate_key="spatial_resized",
                probability_path=(
                    TESLA_PREDICTION_ROOT
                    / "STHBC_spot_level"
                    / f"{slice_name}_probs_seed{SEED}.txt"
                ),
                prediction_path=(
                    TESLA_PREDICTION_ROOT
                    / "STHBC_spot_level"
                    / f"{slice_name}_pred_seed{SEED}.txt"
                ),
                output_dir=(
                    OUTPUT_ROOT
                    / "STHBC"
                ),
                # Spot diameter = 224.
                spot_radius=112.0,
                image_search_roots=(
                    slice_dir,
                    (
                        PIXEL_SEG_DATA_ROOT
                        / "STHBC"
                    ),
                ),
                image_name_keywords=(
                    "resized",
                    "he",
                    "image",
                    "tissue",
                    slice_name.lower(),
                ),
            )
        )

    return configs


def build_vdx_configs() -> list[SliceConfig]:
    configs: list[SliceConfig] = []

    for slice_name in VDX_SLICES:
        slice_dir = (
            PIXEL_SEG_DATA_ROOT
            / slice_name
            / "resized_data"
        )

        scale_factor = read_scale_factor(
            slice_dir
            / "scalefactors.txt"
        )

        # Spot diameter = 256 * scale_factor.
        spot_radius = (
            256.0
            * scale_factor
            / 2.0
        )

        configs.append(
            SliceConfig(
                dataset="VDX",
                slice_name=slice_name,
                adata_path=(
                    slice_dir
                    / "adata_resized.h5ad"
                ),
                coordinate_key="spatial_resized",
                probability_path=(
                    TESLA_PREDICTION_ROOT
                    / f"{slice_name}_spot_level"
                    / f"probs_seed{SEED}.txt"
                ),
                prediction_path=(
                    TESLA_PREDICTION_ROOT
                    / f"{slice_name}_spot_level"
                    / f"pred_seed{SEED}.txt"
                ),
                output_dir=(
                    OUTPUT_ROOT
                    / "VDX"
                ),
                spot_radius=spot_radius,
                image_search_roots=(
                    slice_dir,
                    (
                        PIXEL_SEG_DATA_ROOT
                        / slice_name
                    ),
                ),
                image_name_keywords=(
                    "resized",
                    "he",
                    "image",
                    "tissue",
                    slice_name.lower(),
                ),
            )
        )

    return configs


def build_crc_configs() -> list[SliceConfig]:
    configs: list[SliceConfig] = []

    crc_data_root = (
        PIXEL_SEG_DATA_ROOT
        / "CRC"
        / "cropped_data"
    )

    adata_root = (
        crc_data_root
        / "h5ad_cropped"
    )

    for slice_name in CRC_SLICES:
        configs.append(
            SliceConfig(
                dataset="CRC",
                slice_name=slice_name,
                adata_path=(
                    adata_root
                    / f"{slice_name}_cropped.h5ad"
                ),
                coordinate_key="spatial_cropped",
                probability_path=(
                    TESLA_PREDICTION_ROOT
                    / "CRC_spot_level"
                    / f"{slice_name}_probs_seed{SEED}.txt"
                ),
                prediction_path=(
                    TESLA_PREDICTION_ROOT
                    / "CRC_spot_level"
                    / f"{slice_name}_pred_seed{SEED}.txt"
                ),
                output_dir=(
                    OUTPUT_ROOT
                    / "CRC"
                ),
                # Spot diameter = 64.
                spot_radius=32.0,
                image_search_roots=(
                    crc_data_root,
                    (
                        PIXEL_SEG_DATA_ROOT
                        / "CRC"
                    ),
                ),
                image_name_keywords=(
                    "cropped",
                    "he",
                    "image",
                    "tissue",
                    slice_name.lower(),
                ),
            )
        )

    return configs


# ============================================================================
# Data loading
# ============================================================================

def load_coordinates(
    adata_path: Path,
    coordinate_key: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Load spot coordinates and compatible embedded H&E images, if present.

    Embedded images are used only when their dimensions already cover
    the supplied resized/cropped coordinates.
    """
    if not adata_path.exists():
        raise FileNotFoundError(
            f"找不到 AnnData 文件：{adata_path}"
        )

    adata = ad.read_h5ad(
        adata_path,
        backed="r",
    )

    try:
        if coordinate_key not in adata.obsm:
            raise KeyError(
                f"{adata_path} 中不存在 "
                f"adata.obsm['{coordinate_key}']"
            )

        spatial = np.asarray(
            adata.obsm[coordinate_key],
            dtype=np.float64,
        )

        if (
            spatial.ndim != 2
            or spatial.shape[1] < 2
        ):
            raise ValueError(
                f"{coordinate_key} 应为 [N, >=2]，"
                f"实际形状为 {spatial.shape}"
            )

        coordinates = spatial[:, :2].copy()

        embedded_images: list[np.ndarray] = []

        spatial_uns = adata.uns.get(
            "spatial",
            None,
        )

        if isinstance(spatial_uns, dict):
            for library_data in spatial_uns.values():
                if not isinstance(library_data, dict):
                    continue

                images = library_data.get(
                    "images",
                    {},
                )

                if not isinstance(images, dict):
                    continue

                # Prefer higher-resolution variants.
                for image_key in (
                    "fullres",
                    "hires",
                    "lowres",
                ):
                    if image_key not in images:
                        continue

                    image = np.asarray(
                        images[image_key]
                    )

                    if image.ndim != 3:
                        continue

                    if np.issubdtype(
                        image.dtype,
                        np.floating,
                    ):
                        image = np.clip(
                            image,
                            0.0,
                            1.0,
                        )
                        image = (
                            image
                            * 255.0
                        ).round().astype(np.uint8)
                    else:
                        image = np.clip(
                            image,
                            0,
                            255,
                        ).astype(np.uint8)

                    if image.shape[2] == 4:
                        image = cv2.cvtColor(
                            image,
                            cv2.COLOR_RGBA2BGR,
                        )
                    elif image.shape[2] == 3:
                        image = cv2.cvtColor(
                            image,
                            cv2.COLOR_RGB2BGR,
                        )
                    else:
                        continue

                    embedded_images.append(
                        image
                    )

    finally:
        if adata.file is not None:
            adata.file.close()

    if not np.isfinite(coordinates).all():
        raise ValueError(
            f"{coordinate_key} 中包含 NaN 或 Inf：{adata_path}"
        )

    return coordinates, embedded_images


def load_vector(
    path: Path,
    dtype: type,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{path}"
        )

    values = np.loadtxt(
        path,
        dtype=dtype,
    )

    values = np.asarray(
        values
    ).reshape(-1)

    if values.size == 0:
        raise ValueError(
            f"文件为空：{path}"
        )

    return values


# ============================================================================
# H&E image discovery
# ============================================================================

def should_ignore_image(
    path: Path,
) -> bool:
    """
    Exclude masks, annotations and generated visualizations.
    """
    lower = str(path).lower()

    excluded_terms = (
        "mask",
        "label",
        "annotation",
        "ground_truth",
        "groundtruth",
        "_gt",
        "pred",
        "prob",
        "visual",
        "heatmap",
        "segmentation",
    )

    return any(
        term in lower
        for term in excluded_terms
    )


def collect_image_candidates(
    roots: Iterable[Path],
    slice_name: str,
) -> list[Path]:
    """
    Collect likely H&E image files.

    For broad dataset roots, paths containing the slice name are preferred.
    """
    candidates: list[Path] = []
    seen: set[Path] = set()

    slice_lower = slice_name.lower()

    for root_index, root in enumerate(roots):
        if not root.exists():
            continue

        if root.is_file():
            paths = [root]
        else:
            try:
                paths = root.rglob("*")
            except OSError:
                continue

        for path in paths:
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if should_ignore_image(path):
                continue

            # The first root is normally the slice-specific folder.
            # In broader roots, require the slice name in the path.
            if (
                root_index > 0
                and slice_lower not in str(path).lower()
            ):
                continue

            resolved = path.resolve()

            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)

    return candidates


def image_covers_coordinates(
    image: np.ndarray,
    coordinates: np.ndarray,
) -> bool:
    """
    Determine whether spot centers fit within the image.
    """
    if image.ndim != 3:
        return False

    height, width = image.shape[:2]

    min_x = float(np.min(coordinates[:, 0]))
    min_y = float(np.min(coordinates[:, 1]))
    max_x = float(np.max(coordinates[:, 0]))
    max_y = float(np.max(coordinates[:, 1]))

    return (
        min_x >= -1
        and min_y >= -1
        and max_x < width + 1
        and max_y < height + 1
    )


def score_image_candidate(
    path: Path,
    image: np.ndarray,
    coordinates: np.ndarray,
    slice_name: str,
    keywords: tuple[str, ...],
    adata_parent: Path,
) -> float:
    """
    Rank candidate H&E images using path keywords and coordinate coverage.
    """
    if not image_covers_coordinates(
        image,
        coordinates,
    ):
        return -math.inf

    text = str(path).lower()
    name = path.name.lower()

    score = 0.0

    if slice_name.lower() in text:
        score += 100.0

    if path.parent.resolve() == adata_parent.resolve():
        score += 35.0

    for keyword in keywords:
        if keyword.lower() in name:
            score += 12.0

    if "he" in name or "h&e" in name:
        score += 25.0

    if "resized" in name or "cropped" in name:
        score += 20.0

    if "image" in name or "tissue" in name:
        score += 10.0

    height, width = image.shape[:2]

    target_width = max(
        float(np.max(coordinates[:, 0])) + 1.0,
        1.0,
    )
    target_height = max(
        float(np.max(coordinates[:, 1])) + 1.0,
        1.0,
    )

    # Prefer an image close to the coordinate canvas size,
    # but allow a reasonable empty border around the tissue.
    width_ratio = width / target_width
    height_ratio = height / target_height

    closeness_penalty = (
        abs(math.log(max(width_ratio, 1e-8)))
        + abs(math.log(max(height_ratio, 1e-8)))
    )

    score -= 10.0 * closeness_penalty

    return score


def read_image_unicode(
    path: Path,
) -> np.ndarray | None:
    """
    Read an image safely even when a Windows path contains non-ASCII text.
    """
    try:
        encoded = np.fromfile(
            str(path),
            dtype=np.uint8,
        )

        if encoded.size == 0:
            return None

        image = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        return image

    except Exception:
        return None


def load_best_he_image(
    config: SliceConfig,
    coordinates: np.ndarray,
    embedded_images: list[np.ndarray],
) -> tuple[np.ndarray, str]:
    """
    Load an explicitly configured image, or automatically select the best one.
    """
    override = IMAGE_PATH_OVERRIDES.get(
        config.dataset,
        {},
    ).get(
        config.slice_name,
        None,
    )

    if override is not None:
        image = read_image_unicode(
            override
        )

        if image is None:
            raise OSError(
                f"无法读取手动指定的 H&E 图像：{override}"
            )

        if not image_covers_coordinates(
            image,
            coordinates,
        ):
            raise ValueError(
                f"手动指定图像尺寸 {image.shape[1]}x{image.shape[0]} "
                f"无法覆盖 {config.coordinate_key} 坐标范围。"
            )

        return image, str(override)

    candidate_paths = collect_image_candidates(
        roots=config.image_search_roots,
        slice_name=config.slice_name,
    )

    ranked: list[
        tuple[float, Path, np.ndarray]
    ] = []

    for path in candidate_paths:
        image = read_image_unicode(
            path
        )

        if image is None:
            continue

        score = score_image_candidate(
            path=path,
            image=image,
            coordinates=coordinates,
            slice_name=config.slice_name,
            keywords=config.image_name_keywords,
            adata_parent=config.adata_path.parent,
        )

        if np.isfinite(score):
            ranked.append(
                (score, path, image)
            )

    if ranked:
        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, best_path, best_image = ranked[0]

        print(
            f"  自动选择 H&E：{best_path} "
            f"(score={best_score:.2f})"
        )

        if len(ranked) > 1:
            print("  其他可用候选：")
            for score, path, image in ranked[1:4]:
                print(
                    f"    - {path} | "
                    f"{image.shape[1]}x{image.shape[0]} | "
                    f"score={score:.2f}"
                )

        return best_image, str(best_path)

    # Embedded images are only accepted when their dimensions already
    # match the resized/cropped coordinate system.
    compatible_embedded = [
        image
        for image in embedded_images
        if image_covers_coordinates(
            image,
            coordinates,
        )
    ]

    if compatible_embedded:
        compatible_embedded.sort(
            key=lambda image: (
                image.shape[0]
                * image.shape[1]
            ),
            reverse=True,
        )

        image = compatible_embedded[0]

        print(
            "  未找到外部 H&E 文件，使用 adata.uns['spatial'] "
            f"中的图像：{image.shape[1]}x{image.shape[0]}"
        )

        return image, "adata.uns['spatial']"

    searched = "\n".join(
        f"    - {root}"
        for root in config.image_search_roots
    )

    raise FileNotFoundError(
        f"无法为 {config.dataset}/{config.slice_name} 找到与坐标匹配的 "
        "H&E 图像。\n"
        f"已搜索：\n{searched}\n"
        "请在 IMAGE_PATH_OVERRIDES 中填入准确路径。"
    )


# ============================================================================
# Visualization
# ============================================================================

def resize_for_display(
    image: np.ndarray,
    max_dimension: int,
) -> tuple[np.ndarray, float]:
    """
    Downsample large images while preserving aspect ratio.
    """
    height, width = image.shape[:2]
    longest = max(height, width)

    if longest <= max_dimension:
        return image.copy(), 1.0

    scale = (
        max_dimension
        / longest
    )

    new_width = max(
        1,
        int(round(width * scale)),
    )
    new_height = max(
        1,
        int(round(height * scale)),
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def probability_to_bgr(
    probability: float,
) -> tuple[int, int, int]:
    """
    White-to-red probability color in OpenCV BGR order.
    """
    probability = float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )

    blue_green = int(
        round(
            255.0
            * (1.0 - probability)
        )
    )

    return (
        blue_green,
        blue_green,
        255,
    )


def draw_probability_map(
    display_image: np.ndarray,
    coordinates: np.ndarray,
    probabilities: np.ndarray,
    original_radius: float,
    display_scale: float,
) -> tuple[np.ndarray, int]:
    """
    Draw spot-level cancer probabilities on the H&E image.
    """
    overlay = display_image.copy()
    height, width = display_image.shape[:2]

    scaled_coordinates = np.floor(
        coordinates
        * display_scale
        + 0.5
    ).astype(np.int64)

    scaled_radius = max(
        1,
        int(round(
            original_radius
            * display_scale
        )),
    )

    outside = 0

    for index in tqdm(
        range(len(probabilities)),
        desc="绘制概率 spots",
        leave=False,
    ):
        x, y = scaled_coordinates[index]

        if (
            x < 0
            or x >= width
            or y < 0
            or y >= height
        ):
            outside += 1
            continue

        color = probability_to_bgr(
            probabilities[index]
        )

        cv2.circle(
            overlay,
            (int(x), int(y)),
            scaled_radius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        if DRAW_SPOT_BORDER:
            cv2.circle(
                overlay,
                (int(x), int(y)),
                scaled_radius,
                (0, 0, 0),
                thickness=SPOT_BORDER_THICKNESS,
                lineType=cv2.LINE_AA,
            )

    if SPOT_ALPHA >= 1.0:
        result = overlay
    else:
        result = cv2.addWeighted(
            overlay,
            SPOT_ALPHA,
            display_image,
            1.0 - SPOT_ALPHA,
            0.0,
        )

    return result, outside


def draw_binary_map(
    display_image: np.ndarray,
    coordinates: np.ndarray,
    predictions: np.ndarray,
    original_radius: float,
    display_scale: float,
) -> tuple[np.ndarray, int]:
    """
    Draw binary spot predictions: cancer=red, non-cancer=white.
    """
    overlay = display_image.copy()
    height, width = display_image.shape[:2]

    scaled_coordinates = np.floor(
        coordinates
        * display_scale
        + 0.5
    ).astype(np.int64)

    scaled_radius = max(
        1,
        int(round(
            original_radius
            * display_scale
        )),
    )

    outside = 0

    for index in tqdm(
        range(len(predictions)),
        desc="绘制二值 spots",
        leave=False,
    ):
        x, y = scaled_coordinates[index]

        if (
            x < 0
            or x >= width
            or y < 0
            or y >= height
        ):
            outside += 1
            continue

        if predictions[index] >= 1:
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)

        cv2.circle(
            overlay,
            (int(x), int(y)),
            scaled_radius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        if DRAW_SPOT_BORDER:
            cv2.circle(
                overlay,
                (int(x), int(y)),
                scaled_radius,
                (0, 0, 0),
                thickness=SPOT_BORDER_THICKNESS,
                lineType=cv2.LINE_AA,
            )

    if SPOT_ALPHA >= 1.0:
        result = overlay
    else:
        result = cv2.addWeighted(
            overlay,
            SPOT_ALPHA,
            display_image,
            1.0 - SPOT_ALPHA,
            0.0,
        )

    return result, outside


def add_information_panel(
    image: np.ndarray,
    dataset: str,
    slice_name: str,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    radius_original: float,
    image_source: str,
) -> np.ndarray:
    """
    Add a compact panel above the visualization.
    """
    height, width = image.shape[:2]

    panel_height = max(
        74,
        int(round(height * 0.035)),
    )

    panel = np.full(
        (panel_height, width, 3),
        245,
        dtype=np.uint8,
    )

    font_scale = max(
        0.55,
        min(1.2, width / 2500.0),
    )

    thickness = max(
        1,
        int(round(font_scale * 1.5)),
    )

    title = (
        f"TESLA | {dataset} | {slice_name} | seed {SEED}"
    )

    positive_count = int(
        np.sum(predictions >= 1)
    )

    stats = (
        f"spots={len(probabilities)}  "
        f"positive={positive_count}  "
        f"mean_prob={np.mean(probabilities):.3f}  "
        f"radius={radius_original:.2f}px"
    )

    cv2.putText(
        panel,
        title,
        (18, int(panel_height * 0.42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (30, 30, 30),
        thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        stats,
        (18, int(panel_height * 0.80)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.78,
        (70, 70, 70),
        max(1, thickness - 1),
        cv2.LINE_AA,
    )

    # White-to-red legend.
    legend_width = min(
        380,
        max(150, width // 6),
    )
    legend_height = max(
        12,
        panel_height // 6,
    )

    start_x = max(
        18,
        width - legend_width - 70,
    )
    start_y = max(
        8,
        panel_height // 2 - legend_height // 2,
    )

    for x_offset in range(legend_width):
        prob = (
            x_offset
            / max(legend_width - 1, 1)
        )

        color = probability_to_bgr(
            prob
        )

        cv2.line(
            panel,
            (start_x + x_offset, start_y),
            (
                start_x + x_offset,
                start_y + legend_height,
            ),
            color,
            1,
        )

    cv2.rectangle(
        panel,
        (start_x, start_y),
        (
            start_x + legend_width,
            start_y + legend_height,
        ),
        (80, 80, 80),
        1,
    )

    cv2.putText(
        panel,
        "0",
        (
            start_x - 5,
            min(
                panel_height - 5,
                start_y + legend_height + 20,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.62,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        "1",
        (
            start_x + legend_width - 5,
            min(
                panel_height - 5,
                start_y + legend_height + 20,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.62,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )

    return np.vstack(
        [panel, image]
    )


def save_image_unicode(
    path: Path,
    image: np.ndarray,
) -> None:
    """
    Save images safely to Windows paths containing non-ASCII characters.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extension = path.suffix.lower()

    success, encoded = cv2.imencode(
        extension,
        image,
    )

    if not success:
        raise OSError(
            f"图像编码失败：{path}"
        )

    encoded.tofile(
        str(path)
    )


def validate_inputs(
    config: SliceConfig,
    coordinates: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> None:
    if not (
        len(coordinates)
        == len(probabilities)
        == len(predictions)
    ):
        raise ValueError(
            f"{config.dataset}/{config.slice_name} 数量不一致："
            f"coordinates={len(coordinates)}, "
            f"probabilities={len(probabilities)}, "
            f"predictions={len(predictions)}"
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            f"概率中包含 NaN 或 Inf：{config.probability_path}"
        )

    if (
        np.any(probabilities < -1e-8)
        or np.any(probabilities > 1.0 + 1e-8)
    ):
        raise ValueError(
            f"概率必须位于 [0,1]："
            f"min={probabilities.min()}, "
            f"max={probabilities.max()}"
        )

    unique_predictions = set(
        np.unique(predictions).tolist()
    )

    if not unique_predictions.issubset(
        {0, 1}
    ):
        raise ValueError(
            f"预测文件必须只包含 0/1，实际为 "
            f"{sorted(unique_predictions)}"
        )


def visualize_one_slice(
    config: SliceConfig,
) -> None:
    print("\n" + "=" * 110)
    print(
        f"处理：{config.dataset}/{config.slice_name}"
    )
    print(
        f"坐标：{config.adata_path} -> "
        f"obsm['{config.coordinate_key}']"
    )
    print(
        f"概率：{config.probability_path}"
    )
    print(
        f"预测：{config.prediction_path}"
    )
    print(
        f"原始 spot 半径：{config.spot_radius:.6f}px"
    )

    coordinates, embedded_images = load_coordinates(
        adata_path=config.adata_path,
        coordinate_key=config.coordinate_key,
    )

    probabilities = load_vector(
        config.probability_path,
        float,
    ).astype(np.float64)

    predictions = load_vector(
        config.prediction_path,
        int,
    ).astype(np.uint8)

    validate_inputs(
        config=config,
        coordinates=coordinates,
        probabilities=probabilities,
        predictions=predictions,
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    # As a consistency check, compare saved predictions with probability >= 0.5.
    threshold_predictions = (
        probabilities
        >= PRED_THRESHOLD
    ).astype(np.uint8)

    mismatch_count = int(
        np.sum(
            threshold_predictions
            != predictions
        )
    )

    if mismatch_count > 0:
        print(
            f"  警告：预测文件与 probability >= {PRED_THRESHOLD} "
            f"有 {mismatch_count} 个 spot 不一致；"
            "可视化仍使用已保存的预测文件。"
        )

    he_image, image_source = load_best_he_image(
        config=config,
        coordinates=coordinates,
        embedded_images=embedded_images,
    )

    print(
        f"  H&E 尺寸："
        f"{he_image.shape[1]}x{he_image.shape[0]}"
    )

    display_image, display_scale = resize_for_display(
        image=he_image,
        max_dimension=MAX_OUTPUT_DIMENSION,
    )

    print(
        f"  显示缩放比例：{display_scale:.6f}；"
        f"输出底图：{display_image.shape[1]}x{display_image.shape[0]}"
    )

    probability_map, outside_count = draw_probability_map(
        display_image=display_image,
        coordinates=coordinates,
        probabilities=probabilities,
        original_radius=config.spot_radius,
        display_scale=display_scale,
    )

    outside_fraction = (
        outside_count
        / len(coordinates)
    )

    if outside_fraction > MAX_OUTSIDE_FRACTION:
        raise ValueError(
            f"{outside_count}/{len(coordinates)} "
            f"({outside_fraction:.2%}) 个 spot 中心位于图像外。"
            "自动选择的 H&E 图像可能与坐标不匹配，"
            "请在 IMAGE_PATH_OVERRIDES 中手动指定。"
        )

    probability_with_panel = add_information_panel(
        image=probability_map,
        dataset=config.dataset,
        slice_name=config.slice_name,
        probabilities=probabilities,
        predictions=predictions,
        radius_original=config.spot_radius,
        image_source=image_source,
    )

    probability_output = (
        config.output_dir
        / (
            f"{config.slice_name}_TESLA_"
            f"prob_seed{SEED}.png"
        )
    )

    save_image_unicode(
        probability_output,
        probability_with_panel,
    )

    print(
        f"  概率图已保存：{probability_output}"
    )

    if SAVE_BINARY_MAP:
        binary_map, binary_outside = draw_binary_map(
            display_image=display_image,
            coordinates=coordinates,
            predictions=predictions,
            original_radius=config.spot_radius,
            display_scale=display_scale,
        )

        if binary_outside != outside_count:
            raise RuntimeError(
                "概率图和二值图的越界 spot 数量不一致。"
            )

        binary_with_panel = add_information_panel(
            image=binary_map,
            dataset=config.dataset,
            slice_name=config.slice_name,
            probabilities=probabilities,
            predictions=predictions,
            radius_original=config.spot_radius,
            image_source=image_source,
        )

        binary_output = (
            config.output_dir
            / (
                f"{config.slice_name}_TESLA_"
                f"pred_seed{SEED}.png"
            )
        )

        save_image_unicode(
            binary_output,
            binary_with_panel,
        )

        print(
            f"  二值图已保存：{binary_output}"
        )

    if SAVE_SIDE_BY_SIDE:
        separator = np.full(
            (
                display_image.shape[0],
                8,
                3,
            ),
            235,
            dtype=np.uint8,
        )

        side_by_side = np.hstack(
            [
                display_image,
                separator,
                probability_map,
            ]
        )

        # Keep the combined image within a practical width.
        side_by_side, _ = resize_for_display(
            image=side_by_side,
            max_dimension=MAX_OUTPUT_DIMENSION * 2,
        )

        side_by_side_with_panel = add_information_panel(
            image=side_by_side,
            dataset=config.dataset,
            slice_name=config.slice_name,
            probabilities=probabilities,
            predictions=predictions,
            radius_original=config.spot_radius,
            image_source=image_source,
        )

        comparison_output = (
            config.output_dir
            / (
                f"{config.slice_name}_TESLA_"
                f"original_vs_prob_seed{SEED}.png"
            )
        )

        save_image_unicode(
            comparison_output,
            side_by_side_with_panel,
        )

        print(
            f"  对比图已保存：{comparison_output}"
        )

    if outside_count > 0:
        print(
            f"  提示：{outside_count} 个 spot 中心位于显示图像外，"
            "绘制时已跳过。"
        )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 110)
    print("TESLA spot-level visualization")
    print(f"Seed：{SEED}")
    print(f"输出根目录：{OUTPUT_ROOT}")
    print(
        "数据集：STHBC、VDX（ViHBC/DuCIS/XeHBC）、CRC"
    )
    print("=" * 110)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    configs: list[SliceConfig] = []

    # Build independently so a missing VDX scalefactors file produces
    # a clear error before visualization starts.
    configs.extend(
        build_sthbc_configs()
    )
    configs.extend(
        build_vdx_configs()
    )
    configs.extend(
        build_crc_configs()
    )

    completed: list[
        tuple[str, str]
    ] = []

    failed: list[
        tuple[str, str, str]
    ] = []

    for config in configs:
        try:
            visualize_one_slice(
                config
            )

            completed.append(
                (
                    config.dataset,
                    config.slice_name,
                )
            )

        except Exception as exc:
            failed.append(
                (
                    config.dataset,
                    config.slice_name,
                    str(exc),
                )
            )

            print(
                f"\n[失败] "
                f"{config.dataset}/{config.slice_name}"
            )
            print(
                f"原因：{exc}"
            )

    print("\n" + "=" * 110)
    print(
        f"处理结束：成功 {len(completed)}/{len(configs)}，"
        f"失败 {len(failed)}/{len(configs)}"
    )

    if completed:
        print("\n成功切片：")
        for dataset, slice_name in completed:
            print(
                f"  - {dataset}/{slice_name}"
            )

    if failed:
        print("\n失败切片：")
        for dataset, slice_name, error in failed:
            print(
                f"  - {dataset}/{slice_name}: {error}"
            )

        print(
            "\n若失败原因是 H&E 图像未找到或自动选择错误，"
            "请在脚本顶部的 IMAGE_PATH_OVERRIDES 中填写准确路径。"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()

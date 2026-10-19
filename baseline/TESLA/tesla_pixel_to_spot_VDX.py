"""
将 TESLA 在 ViHBC、DuCIS、XeHBC 三个切片上的像素级二值预测，
转换为 spot 级预测概率与二分类结果。

计算规则
--------
1. 白色像素表示癌症区域，黑色像素表示非癌症区域。
2. 从 adata.obsm["spatial_resized"] 读取 spot 中心坐标，坐标顺序为 (x, y)。
3. 从 scalefactors.txt 读取 scale_factor。
4. 缩放后的 spot 直径 = 256 * scale_factor。
5. spot 半径 = 128 * scale_factor。
6. spot 级概率 =
       spot 圆形有效区域内的癌症像素数
       --------------------------------
       spot 圆形有效区域内的像素总数
7. spot 级预测阈值为 0.5。
8. 输出顺序与 adata.obsm["spatial_resized"] / adata.obs_names 的原始顺序一致。
9. 输出 txt 不包含表头和索引，每行对应一个 spot。
"""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import cv2
import numpy as np


# ============================================================================
# 配置
# ============================================================================

SLIDES = ["ViHBC", "DuCIS", "XeHBC"]
SEEDS = [42, 123, 567]

MASK_ROOT = Path(
    r"D:\Project_2\TESLA\tesla_workspace_three_slices\pred_masks"
)

DATA_ROOT = Path(
    r"D:\Project_2\Pixel_Seg\data"
)

OUTPUT_ROOT = Path(
    r"D:\DNM_project(cancer detect)\predict_files\TESLA"
)

# 缩放前的 spot 直径。
BASE_SPOT_DIAMETER = 256.0

# spot 级二分类阈值。
PRED_THRESHOLD = 0.5


# ============================================================================
# 文件读取
# ============================================================================

def read_binary_mask(mask_path: Path) -> np.ndarray:
    """
    读取 TESLA 预测 mask，并转换为 0/1 二值数组。

    支持常见的：
    - 0/255 二值 mask；
    - 0/1 二值 mask；
    - 可能存在少量非严格黑白值的灰度 mask。
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"找不到 TESLA 预测 mask：{mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise OSError(f"OpenCV 无法读取预测 mask：{mask_path}")

    if mask.size == 0:
        raise ValueError(f"预测 mask 为空：{mask_path}")

    if mask.max() <= 1:
        binary_mask = mask > 0
    else:
        binary_mask = mask >= 128

    return binary_mask.astype(np.uint8)


def read_spatial_coordinates(adata_path: Path) -> np.ndarray:
    """
    从 adata.obsm["spatial_resized"] 读取 spot 中心坐标。

    默认约定：
        spatial_resized[:, 0] 为 x；
        spatial_resized[:, 1] 为 y。
    """
    if not adata_path.exists():
        raise FileNotFoundError(f"找不到 AnnData 文件：{adata_path}")

    # 使用 backed="r" 避免将完整表达矩阵载入内存。
    adata = ad.read_h5ad(adata_path, backed="r")

    try:
        if "spatial_resized" not in adata.obsm:
            raise KeyError(
                f"{adata_path} 中不存在 "
                "adata.obsm['spatial_resized']"
            )

        spatial = np.asarray(
            adata.obsm["spatial_resized"],
            dtype=np.float64,
        )

        if spatial.ndim != 2 or spatial.shape[1] < 2:
            raise ValueError(
                "adata.obsm['spatial_resized'] 应为 "
                f"[spot数, 至少2列]，实际形状为 {spatial.shape}"
            )

        spatial_xy = spatial[:, :2].copy()

        if len(spatial_xy) != adata.n_obs:
            raise ValueError(
                "spatial_resized 的 spot 数量与 adata.n_obs 不一致："
                f"{len(spatial_xy)} != {adata.n_obs}"
            )

    finally:
        if adata.file is not None:
            adata.file.close()

    if not np.isfinite(spatial_xy).all():
        raise ValueError(
            f"{adata_path} 的 spatial_resized 中包含 NaN 或 Inf"
        )

    return spatial_xy


def read_scale_factor(scale_path: Path) -> float:
    """
    从 scalefactors.txt 中读取键 scale_factor。

    支持如下格式：
        scale_factor: 0.324834
        scale_factor = 0.324834
        "scale_factor": 0.324834

    文件内可以包含其他键值，本函数只读取 scale_factor。
    """
    if not scale_path.exists():
        raise FileNotFoundError(f"找不到缩放文件：{scale_path}")

    text = scale_path.read_text(
        encoding="utf-8-sig",
        errors="strict",
    )

    # 支持整数、小数和科学计数法。
    number_pattern = (
        r"[-+]?"
        r"(?:"
        r"(?:\d+(?:\.\d*)?)"
        r"|"
        r"(?:\.\d+)"
        r")"
        r"(?:[eE][-+]?\d+)?"
    )

    pattern = re.compile(
        rf"""["']?scale_factor["']?\s*[:=]\s*
             (?P<value>{number_pattern})""",
        flags=re.IGNORECASE | re.VERBOSE,
    )

    match = pattern.search(text)

    if match is None:
        raise ValueError(
            f"未能在 {scale_path} 中找到类似 "
            "'scale_factor: 0.324834' 的键值"
        )

    scale_factor = float(match.group("value"))

    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError(
            f"{scale_path} 中的 scale_factor 必须为正数，"
            f"实际读取值为 {scale_factor}"
        )

    return scale_factor


# ============================================================================
# 像素级 mask 转换为 spot 级概率
# ============================================================================

def pixel_mask_to_spot_probabilities(
    binary_mask: np.ndarray,
    spatial_xy: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    计算每个圆形 spot 内癌症像素所占比例。

    radius 允许为浮点数。例如：
        scale_factor = 0.324834
        radius = 128 * 0.324834 = 41.578752

    对每个整数纵向偏移 dy，根据
        dx^2 + dy^2 <= radius^2
    计算允许的最大整数横向偏移。

    对于靠近图像边界的 spot，只统计实际落入图像内的圆形区域，
    分子和分母均按有效图像区域计算。
    """
    if binary_mask.ndim != 2:
        raise ValueError(
            f"binary_mask 必须为二维数组，实际形状为 {binary_mask.shape}"
        )

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(
            f"spot 半径必须为正数，实际值为 {radius}"
        )

    spatial_xy = np.asarray(spatial_xy, dtype=np.float64)

    if spatial_xy.ndim != 2 or spatial_xy.shape[1] != 2:
        raise ValueError(
            "spatial_xy 必须为 [spot数, 2]，"
            f"实际形状为 {spatial_xy.shape}"
        )

    height, width = binary_mask.shape

    # 每一行建立前缀和，快速统计任意水平线段中的癌症像素数。
    row_prefix = np.pad(
        np.cumsum(binary_mask, axis=1, dtype=np.int64),
        pad_width=((0, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )

    # 浮点半径对应的全部可能整数纵向偏移。
    max_dy = int(np.floor(radius))
    dy_all = np.arange(
        -max_dy,
        max_dy + 1,
        dtype=np.int64,
    )

    remaining = np.maximum(
        radius * radius - dy_all.astype(np.float64) ** 2,
        0.0,
    )

    dx_half_all = np.floor(
        np.sqrt(remaining)
    ).astype(np.int64)

    # 将 spot 中心映射到最近的像素中心。
    centers = np.floor(
        spatial_xy + 0.5
    ).astype(np.int64)

    probabilities = np.empty(
        len(centers),
        dtype=np.float64,
    )

    invalid_spots: list[int] = []

    for spot_idx, (center_x, center_y) in enumerate(centers):
        ys = center_y + dy_all

        # 删除超出图像上下边界的行。
        valid_rows = (
            (ys >= 0)
            & (ys < height)
        )

        if not np.any(valid_rows):
            probabilities[spot_idx] = np.nan
            invalid_spots.append(spot_idx)
            continue

        ys = ys[valid_rows]
        dx_half = dx_half_all[valid_rows]

        # 每个有效行上圆形范围的左右端点。
        left_unclipped = center_x - dx_half
        right_unclipped = center_x + dx_half

        # 先判断该线段是否与图像水平方向有交集。
        intersects_image = (
            (right_unclipped >= 0)
            & (left_unclipped < width)
        )

        if not np.any(intersects_image):
            probabilities[spot_idx] = np.nan
            invalid_spots.append(spot_idx)
            continue

        ys = ys[intersects_image]
        left_unclipped = left_unclipped[intersects_image]
        right_unclipped = right_unclipped[intersects_image]

        # 裁剪至图像范围。
        left = np.maximum(
            left_unclipped,
            0,
        )

        right = np.minimum(
            right_unclipped,
            width - 1,
        )

        # 使用行前缀和统计癌症像素。
        cancer_pixels = (
            row_prefix[ys, right + 1]
            - row_prefix[ys, left]
        ).sum(dtype=np.int64)

        valid_pixels = (
            right - left + 1
        ).sum(dtype=np.int64)

        if valid_pixels <= 0:
            probabilities[spot_idx] = np.nan
            invalid_spots.append(spot_idx)
            continue

        probabilities[spot_idx] = (
            cancer_pixels / valid_pixels
        )

    if invalid_spots:
        preview = invalid_spots[:10]

        raise ValueError(
            f"有 {len(invalid_spots)} 个 spot 与预测 mask "
            f"完全没有交集，前几个 spot 索引为 {preview}。"
            "请检查 spatial_resized 坐标、mask 尺寸以及 "
            "scale_factor 是否属于同一缩放坐标系。"
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "计算得到的 spot 概率中包含 NaN 或 Inf"
        )

    # 理论上概率应位于 [0, 1]，clip 用于抵御极小浮点误差。
    return np.clip(
        probabilities,
        0.0,
        1.0,
    )


# ============================================================================
# 单个切片处理
# ============================================================================

def process_one_slide(slide: str) -> tuple[int, list[tuple[int, str]]]:
    """
    读取单个切片的 adata 和 scale_factor，并处理三个随机种子。

    返回：
        成功任务数；
        失败任务列表 [(seed, error), ...]。
    """
    resized_data_dir = (
        DATA_ROOT
        / slide
        / "resized_data"
    )

    adata_path = (
        resized_data_dir
        / "adata_resized.h5ad"
    )

    scale_path = (
        resized_data_dir
        / "scalefactors.txt"
    )

    output_dir = (
        OUTPUT_ROOT
        / f"{slide}_spot_level"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spatial_xy = read_spatial_coordinates(
        adata_path
    )

    scale_factor = read_scale_factor(
        scale_path
    )

    spot_diameter = (
        BASE_SPOT_DIAMETER
        * scale_factor
    )

    spot_radius = (
        spot_diameter
        / 2.0
    )

    print("-" * 100)
    print(f"切片：{slide}")
    print(f"spot 数量：{len(spatial_xy)}")
    print(f"scale_factor：{scale_factor:.10f}")
    print(f"缩放后 spot 直径：{spot_diameter:.6f} px")
    print(f"缩放后 spot 半径：{spot_radius:.6f} px")
    print(f"输出目录：{output_dir}")

    successful = 0
    failures: list[tuple[int, str]] = []

    for seed in SEEDS:
        try:
            mask_path = (
                MASK_ROOT
                / f"seed_{seed}"
                / f"{slide}_seed{seed}_tesla_pred_mask.png"
            )

            probability_output_path = (
                output_dir
                / f"probs_seed{seed}.txt"
            )

            prediction_output_path = (
                output_dir
                / f"pred_seed{seed}.txt"
            )

            binary_mask = read_binary_mask(
                mask_path
            )

            probabilities = pixel_mask_to_spot_probabilities(
                binary_mask=binary_mask,
                spatial_xy=spatial_xy,
                radius=spot_radius,
            )

            predictions = (
                probabilities >= PRED_THRESHOLD
            ).astype(np.uint8)

            if len(probabilities) != len(spatial_xy):
                raise RuntimeError(
                    f"输出 spot 数量异常："
                    f"{len(probabilities)} != {len(spatial_xy)}"
                )

            # 每行一个概率，无表头、无索引。
            np.savetxt(
                probability_output_path,
                probabilities,
                fmt="%.8f",
            )

            # 每行一个 0 或 1，无表头、无索引。
            np.savetxt(
                prediction_output_path,
                predictions,
                fmt="%d",
            )

            successful += 1

            print(
                f"[完成] seed={seed} | "
                f"mask={binary_mask.shape[1]}x{binary_mask.shape[0]} | "
                f"positive={int(predictions.sum())}/{len(predictions)} | "
                f"prob_mean={probabilities.mean():.6f}"
            )
            print(
                f"       概率文件：{probability_output_path}"
            )
            print(
                f"       预测文件：{prediction_output_path}"
            )

        except Exception as exc:
            failures.append(
                (seed, str(exc))
            )

            print(
                f"[失败] slide={slide}, seed={seed}"
            )
            print(
                f"       原因：{exc}"
            )

    return successful, failures


# ============================================================================
# 主程序
# ============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_tasks = (
        len(SLIDES)
        * len(SEEDS)
    )

    completed_tasks = 0

    failed_tasks: list[
        tuple[str, int, str]
    ] = []

    print("=" * 100)
    print("TESLA 三个外部切片：像素级预测转换为 spot 级预测")
    print(f"切片：{SLIDES}")
    print(f"随机种子：{SEEDS}")
    print(
        "spot 直径计算："
        f"{BASE_SPOT_DIAMETER:g} * scale_factor"
    )
    print(f"预测阈值：{PRED_THRESHOLD}")
    print(f"mask 根目录：{MASK_ROOT}")
    print(f"数据根目录：{DATA_ROOT}")
    print(f"输出根目录：{OUTPUT_ROOT}")
    print("=" * 100)

    for slide in SLIDES:
        try:
            successful, failures = process_one_slide(
                slide
            )

            completed_tasks += successful

            for seed, error in failures:
                failed_tasks.append(
                    (slide, seed, error)
                )

        except Exception as exc:
            # 若 adata 或 scale_factor 读取失败，
            # 该切片的三个种子都记为失败。
            for seed in SEEDS:
                failed_tasks.append(
                    (slide, seed, str(exc))
                )

            print("-" * 100)
            print(
                f"[切片处理失败] slide={slide}"
            )
            print(
                f"原因：{exc}"
            )

    print("=" * 100)
    print(
        f"处理结束：成功 {completed_tasks}/{total_tasks}，"
        f"失败 {len(failed_tasks)}/{total_tasks}"
    )

    if failed_tasks:
        print("\n失败任务汇总：")

        for slide, seed, error in failed_tasks:
            print(
                f"  - slide={slide}, seed={seed}: {error}"
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()

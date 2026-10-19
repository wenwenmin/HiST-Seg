"""
将 TESLA 像素级二值预测 mask 转换为 spot 级预测概率与二分类结果。

定义：
1. 白色像素为癌症区域，黑色像素为非癌症区域。
2. spot 直径视为 224 像素，半径为 112 像素。
3. spot 级概率 = spot 圆形范围内癌症像素数 / 有效像素总数。
4. spot 级预测阈值为 0.5。
5. 输出顺序与 adata.obs_names / adata.obsm["spatial_resized"] 的原始顺序一致。
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import cv2
import numpy as np


# ============================================================================
# 配置
# ============================================================================

SLIDES = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]
SEEDS = [42, 123, 567]

MASK_ROOT = Path(
    r"D:\Project_2\TESLA\tesla_workspace_multiseed\pred_masks"
)

ADATA_ROOT = Path(
    r"D:\Project_2\Pixel_Seg\data\STHBC\resized_data"
)

OUTPUT_ROOT = Path(
    r"D:\DNM_project(cancer detect)\predict_files\TESLA\STHBC_spot_level"
)

SPOT_DIAMETER = 224
SPOT_RADIUS = SPOT_DIAMETER // 2

PRED_THRESHOLD = 0.5


# ============================================================================
# 数据读取
# ============================================================================

def read_binary_mask(mask_path: Path) -> np.ndarray:
    """
    读取 TESLA 预测 mask，并转换为 0/1 二值数组。

    支持：
    - 黑白值为 0/255；
    - 黑白值为 0/1。
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"无法读取预测 mask：{mask_path}")

    if mask.max() <= 1:
        binary_mask = mask > 0
    else:
        binary_mask = mask >= 128

    return binary_mask.astype(np.uint8)


def read_spatial_coordinates(adata_path: Path) -> np.ndarray:
    """
    从 adata.obsm['spatial_resized'] 中读取 spot 坐标。

    默认坐标顺序：
        spatial_resized[:, 0] -> x
        spatial_resized[:, 1] -> y
    """
    if not adata_path.exists():
        raise FileNotFoundError(f"找不到 AnnData 文件：{adata_path}")

    # backed='r' 可以避免将较大的表达矩阵完整载入内存。
    adata = ad.read_h5ad(adata_path, backed="r")

    try:
        if "spatial_resized" not in adata.obsm:
            raise KeyError(
                f"{adata_path} 中不存在 "
                "adata.obsm['spatial_resized']"
            )

        spatial_xy = np.asarray(
            adata.obsm["spatial_resized"][:, :2],
            dtype=np.float64,
        ).copy()

    finally:
        if adata.file is not None:
            adata.file.close()

    if spatial_xy.ndim != 2 or spatial_xy.shape[1] != 2:
        raise ValueError(
            "spatial_resized 的形状应为 [spot数, 2]，"
            f"实际形状为 {spatial_xy.shape}"
        )

    if not np.isfinite(spatial_xy).all():
        raise ValueError(
            f"{adata_path} 的 spatial_resized 中包含 NaN 或 Inf"
        )

    return spatial_xy


# ============================================================================
# 像素级结果转换为 spot 级结果
# ============================================================================

def pixel_mask_to_spot_probabilities(
    binary_mask: np.ndarray,
    spatial_xy: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    计算每个 spot 圆形范围内的癌症像素比例。

    为避免逐个 spot 创建 224x224 的临时图像，本函数对每一行建立前缀和，
    然后逐行统计圆形区域中的像素数。该方法计算的是精确离散圆形区域。

    对于靠近图像边界的 spot：
    仅统计实际落在图像范围内的圆形像素，分母也只使用有效图像像素。
    """
    if binary_mask.ndim != 2:
        raise ValueError(
            f"binary_mask 必须为二维数组，实际形状为 {binary_mask.shape}"
        )

    if radius <= 0:
        raise ValueError(f"radius 必须大于 0，实际值为 {radius}")

    height, width = binary_mask.shape

    # 对 mask 的每一行建立前缀和。
    # row_prefix[y, right + 1] - row_prefix[y, left]
    # 可得到第 y 行 [left, right] 范围内的癌症像素数。
    row_prefix = np.pad(
        np.cumsum(binary_mask, axis=1, dtype=np.int64),
        pad_width=((0, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )

    # 圆内每个纵向偏移 dy 所对应的最大横向偏移。
    dy_all = np.arange(-radius, radius + 1, dtype=np.int64)

    dx_half_all = np.floor(
        np.sqrt(radius * radius - dy_all * dy_all)
    ).astype(np.int64)

    # spatial_resized 通常为整数或接近整数。
    # 这里采用四舍五入映射到最近的像素中心。
    centers = np.floor(spatial_xy + 0.5).astype(np.int64)

    probabilities = np.empty(len(centers), dtype=np.float64)
    invalid_spots: list[int] = []

    for spot_idx, (center_x, center_y) in enumerate(centers):
        # 当前圆覆盖的所有图像行。
        ys = center_y + dy_all

        # 删除位于图像上下边界以外的行。
        valid_rows = (ys >= 0) & (ys < height)

        if not np.any(valid_rows):
            probabilities[spot_idx] = np.nan
            invalid_spots.append(spot_idx)
            continue

        ys = ys[valid_rows]
        dx_half = dx_half_all[valid_rows]

        # 每一行中属于圆形区域的左右边界。
        left = np.maximum(center_x - dx_half, 0)
        right = np.minimum(center_x + dx_half, width - 1)

        # 删除完全位于图像左右边界之外的行段。
        valid_segments = left <= right

        if not np.any(valid_segments):
            probabilities[spot_idx] = np.nan
            invalid_spots.append(spot_idx)
            continue

        ys = ys[valid_segments]
        left = left[valid_segments]
        right = right[valid_segments]

        # 统计圆形区域内的癌症像素数。
        cancer_pixels = (
            row_prefix[ys, right + 1]
            - row_prefix[ys, left]
        ).sum(dtype=np.int64)

        # 统计圆形区域内实际有效的像素总数。
        valid_pixels = (
            right - left + 1
        ).sum(dtype=np.int64)

        probabilities[spot_idx] = cancer_pixels / valid_pixels

    # 如果某些 spot 与 mask 完全没有交集，通常说明坐标尺度不一致。
    if invalid_spots:
        preview = invalid_spots[:10]

        raise ValueError(
            f"有 {len(invalid_spots)} 个 spot 的圆形区域完全位于 mask 外。"
            f"前几个 spot 索引为 {preview}。请检查 "
            "adata.obsm['spatial_resized'] 与预测 mask "
            "是否处于相同分辨率和坐标系。"
        )

    return probabilities


# ============================================================================
# 单个切片、单个随机种子的处理
# ============================================================================

def process_one_slide_seed(slide: str, seed: int) -> None:
    mask_path = (
        MASK_ROOT
        / f"seed_{seed}"
        / f"{slide}_seed{seed}_tesla_pred_mask.png"
    )

    adata_path = (
        ADATA_ROOT
        / slide
        / "adata_resized.h5ad"
    )

    probability_output_path = (
        OUTPUT_ROOT
        / f"{slide}_probs_seed{seed}.txt"
    )

    prediction_output_path = (
        OUTPUT_ROOT
        / f"{slide}_pred_seed{seed}.txt"
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"找不到 TESLA 预测 mask：{mask_path}"
        )

    binary_mask = read_binary_mask(mask_path)
    spatial_xy = read_spatial_coordinates(adata_path)

    probabilities = pixel_mask_to_spot_probabilities(
        binary_mask=binary_mask,
        spatial_xy=spatial_xy,
        radius=SPOT_RADIUS,
    )

    predictions = (
        probabilities >= PRED_THRESHOLD
    ).astype(np.uint8)

    # 输出数量必须与 spot 数完全一致。
    if len(probabilities) != len(spatial_xy):
        raise RuntimeError(
            f"{slide}, seed={seed} 的输出数量异常："
            f"{len(probabilities)} != {len(spatial_xy)}"
        )

    # 每行保存一个概率，不保存表头和索引。
    np.savetxt(
        probability_output_path,
        probabilities,
        fmt="%.8f",
    )

    # 每行保存一个 0 或 1，不保存表头和索引。
    np.savetxt(
        prediction_output_path,
        predictions,
        fmt="%d",
    )

    print(
        f"[完成] slide={slide}, seed={seed} | "
        f"mask={binary_mask.shape[1]}x{binary_mask.shape[0]} | "
        f"spots={len(probabilities)} | "
        f"positive={int(predictions.sum())} | "
        f"prob_mean={probabilities.mean():.6f}"
    )

    print(f"       概率文件：{probability_output_path}")
    print(f"       预测文件：{prediction_output_path}")


# ============================================================================
# 主程序
# ============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_tasks = len(SLIDES) * len(SEEDS)
    completed_tasks = 0

    failed_tasks: list[tuple[str, int, str]] = []

    print("=" * 100)
    print("TESLA 像素级预测转换为 spot 级预测")
    print(f"切片：{SLIDES}")
    print(f"随机种子：{SEEDS}")
    print(f"spot 直径：{SPOT_DIAMETER}")
    print(f"spot 半径：{SPOT_RADIUS}")
    print(f"预测阈值：{PRED_THRESHOLD}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print("=" * 100)

    for seed in SEEDS:
        for slide in SLIDES:
            try:
                process_one_slide_seed(
                    slide=slide,
                    seed=seed,
                )

                completed_tasks += 1

            except Exception as exc:
                failed_tasks.append(
                    (slide, seed, str(exc))
                )

                print(
                    f"[失败] slide={slide}, seed={seed}"
                )
                print(f"       原因：{exc}")

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

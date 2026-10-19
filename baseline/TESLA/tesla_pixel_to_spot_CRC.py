"""
将 TESLA 在 CRC 12 张切片上的像素级二值预测 mask
转换为 spot 级预测概率与二分类结果。

规则
----
1. 白色像素表示癌症区域，黑色像素表示非癌症区域。
2. spot 中心坐标从 adata.obsm["spatial_cropped"] 读取。
3. 坐标顺序默认是：
       spatial_cropped[:, 0] -> x
       spatial_cropped[:, 1] -> y
4. spot 直径固定为 64 像素，因此半径为 32 像素。
5. spot 级概率 =
       spot 圆形有效区域内的癌症像素数
       --------------------------------
       spot 圆形有效区域内的总像素数
6. spot 级预测阈值为 0.5。
7. 输出顺序与 adata.obsm["spatial_cropped"] 和 adata.obs_names
   的原始顺序一致。
8. 输出 txt 不包含表头和索引，每行对应一个 spot。
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import cv2
import numpy as np


# ============================================================================
# 配置
# ============================================================================

REF_NAMES = [
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

SEEDS = [42, 123, 567]

MASK_ROOT = Path(
    r"D:\Project_2\TESLA\tesla_workspace_crc_multiseed\pred_masks"
)

ADATA_ROOT = Path(
    r"D:\Project_2\Pixel_Seg\data\CRC\cropped_data\h5ad_cropped"
)

OUTPUT_ROOT = Path(
    r"D:\DNM_project(cancer detect)\predict_files\TESLA\CRC_spot_level"
)

SPOT_DIAMETER = 64
SPOT_RADIUS = SPOT_DIAMETER // 2

PRED_THRESHOLD = 0.5


# ============================================================================
# 文件读取
# ============================================================================

def read_binary_mask(mask_path: Path) -> np.ndarray:
    """
    读取 TESLA 预测 mask，并转换为 0/1 二值数组。

    支持：
    - 0/255 二值 mask；
    - 0/1 二值 mask；
    - 可能存在少量非严格黑白值的灰度 mask。
    """
    if not mask_path.exists():
        raise FileNotFoundError(
            f"找不到 TESLA 预测 mask：{mask_path}"
        )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise OSError(
            f"OpenCV 无法读取预测 mask：{mask_path}"
        )

    if mask.size == 0:
        raise ValueError(
            f"预测 mask 为空：{mask_path}"
        )

    if mask.max() <= 1:
        binary_mask = mask > 0
    else:
        binary_mask = mask >= 128

    return binary_mask.astype(np.uint8)


def read_spatial_coordinates(
    adata_path: Path,
) -> np.ndarray:
    """
    从 adata.obsm["spatial_cropped"] 中读取 spot 中心坐标。

    默认约定：
        spatial_cropped[:, 0] 为 x；
        spatial_cropped[:, 1] 为 y。
    """
    if not adata_path.exists():
        raise FileNotFoundError(
            f"找不到 AnnData 文件：{adata_path}"
        )

    # 使用 backed="r" 避免将完整表达矩阵加载到内存。
    adata = ad.read_h5ad(
        adata_path,
        backed="r",
    )

    try:
        if "spatial_cropped" not in adata.obsm:
            raise KeyError(
                f"{adata_path} 中不存在 "
                "adata.obsm['spatial_cropped']"
            )

        spatial = np.asarray(
            adata.obsm["spatial_cropped"],
            dtype=np.float64,
        )

        if spatial.ndim != 2 or spatial.shape[1] < 2:
            raise ValueError(
                "adata.obsm['spatial_cropped'] 应为 "
                f"[spot数, 至少2列]，实际形状为 {spatial.shape}"
            )

        spatial_xy = spatial[:, :2].copy()

        if len(spatial_xy) != adata.n_obs:
            raise ValueError(
                "spatial_cropped 的 spot 数量与 adata.n_obs 不一致："
                f"{len(spatial_xy)} != {adata.n_obs}"
            )

    finally:
        if adata.file is not None:
            adata.file.close()

    if not np.isfinite(spatial_xy).all():
        raise ValueError(
            f"{adata_path} 的 spatial_cropped 中包含 NaN 或 Inf"
        )

    return spatial_xy


# ============================================================================
# 像素级 mask 转换为 spot 级概率
# ============================================================================

def pixel_mask_to_spot_probabilities(
    binary_mask: np.ndarray,
    spatial_xy: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    计算每个圆形 spot 内癌症像素所占比例。

    本函数对 mask 的每一行建立前缀和，从而快速统计每个圆形区域
    中的癌症像素数。

    对于靠近图像边界的 spot：
    仅统计圆形区域中实际位于图像范围内的像素，分母也只统计
    有效图像像素。
    """
    if binary_mask.ndim != 2:
        raise ValueError(
            f"binary_mask 必须为二维数组，实际形状为 {binary_mask.shape}"
        )

    if radius <= 0:
        raise ValueError(
            f"spot 半径必须大于 0，实际值为 {radius}"
        )

    spatial_xy = np.asarray(
        spatial_xy,
        dtype=np.float64,
    )

    if spatial_xy.ndim != 2 or spatial_xy.shape[1] != 2:
        raise ValueError(
            "spatial_xy 必须为 [spot数, 2]，"
            f"实际形状为 {spatial_xy.shape}"
        )

    if not np.isfinite(spatial_xy).all():
        raise ValueError(
            "spatial_xy 中包含 NaN 或 Inf"
        )

    height, width = binary_mask.shape

    # 每一行建立前缀和：
    # row_prefix[y, right + 1] - row_prefix[y, left]
    # 等于第 y 行 [left, right] 内癌症像素数。
    row_prefix = np.pad(
        np.cumsum(
            binary_mask,
            axis=1,
            dtype=np.int64,
        ),
        pad_width=((0, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )

    # 圆内所有整数纵向偏移。
    dy_all = np.arange(
        -radius,
        radius + 1,
        dtype=np.int64,
    )

    # 对应每个 dy 的最大整数横向偏移。
    dx_half_all = np.floor(
        np.sqrt(
            radius * radius
            - dy_all.astype(np.float64) ** 2
        )
    ).astype(np.int64)

    # 将 spot 坐标映射到最近的整数像素中心。
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

        left_unclipped = center_x - dx_half
        right_unclipped = center_x + dx_half

        # 判断当前水平线段是否与图像有交集。
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

        # 裁剪到图像边界。
        left = np.maximum(
            left_unclipped,
            0,
        )

        right = np.minimum(
            right_unclipped,
            width - 1,
        )

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
            "请检查 adata.obsm['spatial_cropped'] 与预测 mask "
            "是否处于相同坐标系和分辨率。"
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "计算得到的 spot 概率中包含 NaN 或 Inf"
        )

    return np.clip(
        probabilities,
        0.0,
        1.0,
    )


# ============================================================================
# 单个切片和随机种子的处理
# ============================================================================

def process_one_slice_seed(
    slice_name: str,
    seed: int,
    spatial_xy: np.ndarray,
) -> None:
    """
    处理一个切片在一个随机种子下的预测结果。
    """
    mask_path = (
        MASK_ROOT
        / f"seed_{seed}"
        / f"{slice_name}_seed{seed}_tesla_crc_pred_mask.png"
    )

    probability_output_path = (
        OUTPUT_ROOT
        / f"{slice_name}_probs_seed{seed}.txt"
    )

    prediction_output_path = (
        OUTPUT_ROOT
        / f"{slice_name}_pred_seed{seed}.txt"
    )

    binary_mask = read_binary_mask(
        mask_path
    )

    probabilities = pixel_mask_to_spot_probabilities(
        binary_mask=binary_mask,
        spatial_xy=spatial_xy,
        radius=SPOT_RADIUS,
    )

    predictions = (
        probabilities >= PRED_THRESHOLD
    ).astype(np.uint8)

    if len(probabilities) != len(spatial_xy):
        raise RuntimeError(
            f"{slice_name}, seed={seed} 的输出数量异常："
            f"{len(probabilities)} != {len(spatial_xy)}"
        )

    # 每行保存一个概率，无表头和索引。
    np.savetxt(
        probability_output_path,
        probabilities,
        fmt="%.8f",
    )

    # 每行保存一个 0 或 1，无表头和索引。
    np.savetxt(
        prediction_output_path,
        predictions,
        fmt="%d",
    )

    print(
        f"[完成] slice={slice_name}, seed={seed} | "
        f"mask={binary_mask.shape[1]}x{binary_mask.shape[0]} | "
        f"spots={len(probabilities)} | "
        f"positive={int(predictions.sum())} | "
        f"prob_mean={probabilities.mean():.6f}"
    )

    print(
        f"       概率文件：{probability_output_path}"
    )

    print(
        f"       预测文件：{prediction_output_path}"
    )


# ============================================================================
# 主程序
# ============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_tasks = (
        len(REF_NAMES)
        * len(SEEDS)
    )

    completed_tasks = 0

    failed_tasks: list[
        tuple[str, int, str]
    ] = []

    print("=" * 110)
    print("TESLA CRC：像素级预测转换为 spot 级预测")
    print(f"切片数量：{len(REF_NAMES)}")
    print(f"切片：{REF_NAMES}")
    print(f"随机种子：{SEEDS}")
    print(f"spot 直径：{SPOT_DIAMETER} px")
    print(f"spot 半径：{SPOT_RADIUS} px")
    print(f"预测阈值：{PRED_THRESHOLD}")
    print(f"mask 根目录：{MASK_ROOT}")
    print(f"adata 根目录：{ADATA_ROOT}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print("=" * 110)

    for slice_name in REF_NAMES:
        adata_path = (
            ADATA_ROOT
            / f"{slice_name}_cropped.h5ad"
        )

        try:
            # 同一切片的三个随机种子共享同一组 spot 坐标，
            # 因此只读取一次 adata。
            spatial_xy = read_spatial_coordinates(
                adata_path
            )

            print("-" * 110)
            print(
                f"切片：{slice_name} | "
                f"spot 数量：{len(spatial_xy)}"
            )

        except Exception as exc:
            print("-" * 110)
            print(
                f"[切片读取失败] slice={slice_name}"
            )
            print(
                f"原因：{exc}"
            )

            for seed in SEEDS:
                failed_tasks.append(
                    (slice_name, seed, str(exc))
                )

            continue

        for seed in SEEDS:
            try:
                process_one_slice_seed(
                    slice_name=slice_name,
                    seed=seed,
                    spatial_xy=spatial_xy,
                )

                completed_tasks += 1

            except Exception as exc:
                failed_tasks.append(
                    (slice_name, seed, str(exc))
                )

                print(
                    f"[失败] slice={slice_name}, seed={seed}"
                )
                print(
                    f"       原因：{exc}"
                )

    print("=" * 110)
    print(
        f"处理结束：成功 {completed_tasks}/{total_tasks}，"
        f"失败 {len(failed_tasks)}/{total_tasks}"
    )

    if failed_tasks:
        print("\n失败任务汇总：")

        for slice_name, seed, error in failed_tasks:
            print(
                f"  - slice={slice_name}, seed={seed}: {error}"
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()

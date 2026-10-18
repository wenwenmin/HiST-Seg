import os
import subprocess
import numpy as np
import cv2
from PIL import Image

# 消除 joblib 寻找物理核心的警告
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 为了能处理可能非常大的病理图像，解除 PIL 的像素限制
Image.MAX_IMAGE_PIXELS = None

# ==========================================
# 1. 配置文件路径 (单切片模式)
# ==========================================
# 输入的原图路径
IMG_PATH = r"D:\DNM_project(cancer detect)\data\HB\Visium_HBC\he-raw.jpg"

# 输出目录：我们将生成的特征文件和 Mask 都保存在这里，方便管理
BASE_OUT_DIR = r"D:\Project_2\像素级分割\data\ViHBC\mask"
os.makedirs(BASE_OUT_DIR, exist_ok=True)

# 提取特征的命名前缀和存放路径
OUT_PREFIX = os.path.join(BASE_OUT_DIR, "ViHBC_features")
PKL_PATH = OUT_PREFIX + "_embeddings-hist.pickle"

# 最终生成的 Mask 存放路径
MASK_OUT_PATH = os.path.join(BASE_OUT_DIR, "ViHBC_mask.png")


def refine_mask(mask_array, img_path, bg_threshold=235, min_hole_size=5000):
    """
    利用原图物理边界精修 Mask 边缘
    同时：填补组织内部微小的假空洞（浅染区/脂肪滴），但保留真正的大空洞（管腔/大撕裂）

    参数:
    - min_hole_size: 判定为“假空洞”的最大面积阈值（像素个数）。
                     病理图通常很大，你可以根据实际原图的分辨率调整这个值。
                     如果原图是几万x几万像素，这个值可以调到 10000 甚至 50000。
    """
    # 1. 确保输入的放大版 AI Mask 是标准的 0 和 1 二值图格式
    if mask_array.dtype == bool:
        ai_mask = mask_array.astype(np.uint8)
    else:
        ai_mask = (mask_array > 0).astype(np.uint8)

    # 2. 读取高分辨率原图并转为灰度图
    with Image.open(img_path) as img:
        gray_img = np.array(img.convert('L'))

    # 3. 生成高精度“物理边界 Mask”
    physical_mask = (gray_img < bg_threshold).astype(np.uint8)

    # 4. 取交集：限制在真实物理边界内，但这会产生内部浅染区的细小空洞
    final_mask = ai_mask & physical_mask

    # 5. 形态学开运算：去除外围游离的细小白色噪点
    kernel_open = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)

    # ==========================================
    # 6. 新增：基于面积过滤内部空洞
    # ==========================================
    # 将 mask 反转：此时黑色背景和空洞变成了白色 (1)，组织变成了黑色 (0)
    inverted_mask = (1 - final_mask).astype(np.uint8)

    # 寻找所有空洞和背景的轮廓 (使用 RETR_LIST 找出所有独立的块)
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 计算该黑色区域的面积
        area = cv2.contourArea(cnt)

        # 如果黑色区域的面积小于阈值，说明它是组织内部的“假空洞”/裂隙
        # 我们将其用白色(1)实心填补回 final_mask 中
        if area < min_hole_size:
            cv2.drawContours(final_mask, [cnt], -1, 1, thickness=cv2.FILLED)

        # 注意：切片外部真正的巨大背景，以及组织中间真正的大空洞，
        # 它们的 area 都会远大于 min_hole_size，因此会被完美保留（不被填补）。

    # 7. 返回 0 和 255 的标准灰度图格式
    return final_mask * 255


def main():
    print(f"\n{'=' * 20} 正在处理单切片数据 {'=' * 20}")
    print(f"输入图片: {IMG_PATH}")
    print(f"输出目录: {BASE_OUT_DIR}")

    # 检查原图是否存在
    if not os.path.exists(IMG_PATH):
        print(f"❌ 错误: 找不到输入图片，请检查路径是否正确。")
        return

    try:
        # ---------------------------------------------------------
        # 步骤 1: 提取特征
        # ---------------------------------------------------------
        if not os.path.exists(PKL_PATH):
            print("\n>>> [1/3] 正在运行深度学习模型提取特征...")
            cmd_extract = [
                "python", "extract_features.py",
                "--img_path", IMG_PATH,
                "--out_prefix", OUT_PREFIX
            ]
            subprocess.run(cmd_extract, check=True)
        else:
            print("\n>>> [1/3] 特征文件已存在，跳过提取阶段。")

        # ---------------------------------------------------------
        # 步骤 2: 生成初始 Mask (小尺寸)
        # ---------------------------------------------------------
        print(">>> [2/3] 正在进行聚类分割并生成初始 Mask...")
        cmd_mask = [
            "python", "get_mask.py",
            PKL_PATH,
            MASK_OUT_PATH
        ]
        subprocess.run(cmd_mask, check=True)

        # ---------------------------------------------------------
        # 步骤 3: 还原分辨率并精修物理边缘
        # ---------------------------------------------------------
        print(">>> [3/3] 正在将 Mask 还原至原图分辨率并裁切边缘溢出...")

        # 获取原图尺寸
        with Image.open(IMG_PATH) as img:
            target_size = img.size

        # 获取小图 Mask 并放大
        with Image.open(MASK_OUT_PATH) as mask:
            resized_mask = mask.resize(target_size, resample=Image.Resampling.NEAREST)
            mask_array = np.array(resized_mask)

        # 使用精修逻辑去除溢出部分
        final_mask_array = refine_mask(mask_array, IMG_PATH)

        # 覆盖保存为高精度大图
        final_mask = Image.fromarray(final_mask_array)
        final_mask.save(MASK_OUT_PATH)

        print(f"\n✅ 单切片处理彻底完成！高精度 Mask 已保存至: {MASK_OUT_PATH}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 执行子脚本时发生错误，处理中断: {e}")
    except Exception as e:
        print(f"\n❌ 处理切片时发生意外错误: {e}")


if __name__ == '__main__':
    main()
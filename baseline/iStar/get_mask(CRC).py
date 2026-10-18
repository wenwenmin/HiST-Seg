import os
import subprocess
from PIL import Image

# 消除 joblib 寻找物理核心的警告
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 为了能处理可能非常大的病理图像，解除 PIL 的像素限制
Image.MAX_IMAGE_PIXELS = None

# 1. 填入 CRC 数据集的切片名称
slice_names = [
    'SN048_1', 'SN048_2', 'SN048_3', 'SN048_4',
    'SN84_1', 'SN84_2',
    'SN123_1', 'SN123_2', 'SN123_3', 'SN123_4',
    'SN124_1', 'SN124_2', 'SN124_3', 'SN124_4'
]

# 2. 配置 CRC 数据集的基础路径
BASE_IN_DIR = r"D:\DNM_project(cancer detect)\data\CRC\slices_image"
BASE_OUT_DIR = os.path.abspath(r"../Pixel_Seg/data/CRC/mask")

# 新增：由于现在所有原图都在同一个文件夹，我们需要一个独立目录来存放生成的中间特征文件 (.pickle)
BASE_FEAT_DIR = r"D:\DNM_project(cancer detect)\data\CRC\features"

# 确保输出目录和特征存放目录存在
os.makedirs(BASE_OUT_DIR, exist_ok=True)
os.makedirs(BASE_FEAT_DIR, exist_ok=True)


def main():
    for slice_name in slice_names:
        print(f"\n{'=' * 20} 正在处理切片: {slice_name} {'=' * 20}")

        # 根据新数据集的结构修改路径拼接方式
        img_path = os.path.join(BASE_IN_DIR, f"{slice_name}.png")

        # 特征提取的输出前缀（例如：.../features/SN048_1）
        out_prefix = os.path.join(BASE_FEAT_DIR, slice_name)
        pkl_path = out_prefix + "_embeddings-hist.pickle"

        # 最终 Mask 的保存路径
        mask_out_path = os.path.join(BASE_OUT_DIR, f"{slice_name}.png")

        # 检查原图是否存在（容错：如果在 png 找不到，尝试找 jpg）
        if not os.path.exists(img_path):
            img_path_jpg = os.path.join(BASE_IN_DIR, f"{slice_name}.jpg")
            if os.path.exists(img_path_jpg):
                img_path = img_path_jpg
            else:
                print(f"❌ 警告: 找不到输入图片 {slice_name}，已跳过。")
                continue

        try:
            # ---------------------------------------------------------
            # 步骤 1: 提取特征
            # ---------------------------------------------------------
            if not os.path.exists(pkl_path):
                print(">>> [1/3] 正在运行深度学习模型提取特征...")
                cmd_extract = [
                    "python", "extract_features.py",
                    "--img_path", img_path,
                    "--out_prefix", out_prefix
                ]
                subprocess.run(cmd_extract, check=True)
            else:
                print(">>> [1/3] 特征文件已存在，跳过提取阶段。")

            # ---------------------------------------------------------
            # 步骤 2: 生成 Mask (小尺寸)
            # ---------------------------------------------------------
            print(">>> [2/3] 正在进行聚类分割并生成初始 Mask...")
            cmd_mask = [
                "python", "get_mask.py",
                pkl_path,
                mask_out_path
            ]
            subprocess.run(cmd_mask, check=True)

            # ---------------------------------------------------------
            # 步骤 3: 将 Mask 放大至原图绝对分辨率
            # ---------------------------------------------------------
            print(">>> [3/3] 正在将 Mask 还原至原图分辨率...")

            # 读取原图获取目标尺寸 (宽, 高)
            with Image.open(img_path) as img:
                target_size = img.size

            # 读取刚刚生成的小 Mask，并放大覆盖
            with Image.open(mask_out_path) as mask:
                # 使用最近邻插值 (NEAREST) 保证掩膜边界清晰
                resized_mask = mask.resize(target_size, resample=Image.Resampling.NEAREST)
                resized_mask.save(mask_out_path)

            print(f"✅ 切片 {slice_name} 处理彻底完成！高分辨率 Mask 已保存至: {mask_out_path}")

        except subprocess.CalledProcessError as e:
            print(f"❌ 处理切片 {slice_name} 时发生错误，外部脚本执行中断: {e}")
        except Exception as e:
            print(f"❌ 处理切片 {slice_name} 时发生 Python 层级错误: {e}")


if __name__ == '__main__':
    main()
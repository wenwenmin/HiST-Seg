
import os
import subprocess
from PIL import Image

# 消除 joblib 寻找物理核心的警告
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 为了能处理可能非常大的病理图像，解除 PIL 的像素限制
Image.MAX_IMAGE_PIXELS = None

# 1. 填入你真实的 8 个切片文件夹名称
slice_names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

# 2. 配置你的基础路径
BASE_IN_DIR = r"D:\DNM_project(cancer detect)\data\HB\her2st"
BASE_OUT_DIR = os.path.abspath(r"../Pixel_Seg/data/STHBC/mask")

os.makedirs(BASE_OUT_DIR, exist_ok=True)


def main():
    for slice_name in slice_names:
        print(f"\n{'=' * 20} 正在处理切片: {slice_name} {'=' * 20}")

        img_path = os.path.join(BASE_IN_DIR, slice_name, "he-raw.png")
        out_prefix = os.path.join(BASE_IN_DIR, slice_name, "features")
        pkl_path = out_prefix + "_embeddings-hist.pickle"
        mask_out_path = os.path.join(BASE_OUT_DIR, f"{slice_name}.png")

        # 检查原图是否存在（注意后缀，如果是 .jpg 请自行修改）
        if not os.path.exists(img_path):
            # 尝试找找 jpg 后缀
            img_path_jpg = os.path.join(BASE_IN_DIR, slice_name, "he-raw.jpg")
            if os.path.exists(img_path_jpg):
                img_path = img_path_jpg
            else:
                print(f"❌ 警告: 找不到输入图片，已跳过。")
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

            # 读取原图获取目标尺寸 (宽, 高)，这步不会占用大量内存
            with Image.open(img_path) as img:
                target_size = img.size

            # 读取刚刚生成的小 Mask
            with Image.open(mask_out_path) as mask:
                # 使用最近邻插值 (NEAREST) 放大，保证黑白不产生过渡灰阶
                resized_mask = mask.resize(target_size, resample=Image.Resampling.NEAREST)

                # 直接覆盖保存原来的小图
                resized_mask.save(mask_out_path)

            print(f"✅ 切片 {slice_name} 处理彻底完成！高分辨率 Mask 已保存至: {mask_out_path}")

        except subprocess.CalledProcessError as e:
            print(f"❌ 处理切片 {slice_name} 时发生错误，执行中断: {e}")
        except Exception as e:
            print(f"❌ 缩放切片 {slice_name} Mask 时发生错误: {e}")


if __name__ == '__main__':
    main()


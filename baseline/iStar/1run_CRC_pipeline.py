import os
import shutil
import scanpy as sc
import pandas as pd
import numpy as np
import subprocess
import sys
import cv2

# CRC 数据集的基础路径
base_dir = r"D:\Project_2\Pixel_Seg\data\CRC\cropped_data"
he_base_dir = os.path.join(base_dir, "he_cropped")
mask_base_dir = os.path.join(base_dir, "tissue_mask_cropped")
h5ad_base_dir = os.path.join(base_dir, "h5ad_cropped")

# 专门为 CRC 数据集建立的工作空间
workspace_dir = r"D:\Project_2\iStar\istar_workspace\CRC"

# all_samples = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4',
#                'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2',
#                'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']

all_samples = ['SN048_2', 'SN048_3', 'SN048_4','SN84_1']

# 定义需要遍历的随机种子列表
seeds = [42, 123, 567]


def run_cmd(cmd_list):
    print(f"Running: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True)


def prepare_and_run(slide, seed):
    print(f"\n========== Processing Slide: {slide} | Seed: {seed} ==========")

    # 结合样本名和种子生成独立的工作空间前缀
    prefix = os.path.join(workspace_dir, f"{slide}_seed_{seed}\\")
    os.makedirs(prefix, exist_ok=True)

    # 1. 转换并拷贝 H&E 图像 (将 .png 转换为 iStar 默认读取的 .jpg)
    raw_he_png = os.path.join(he_base_dir, f"{slide}.png")
    workspace_raw_he_jpg = os.path.join(prefix, "he-raw.jpg")
    if not os.path.exists(workspace_raw_he_jpg):
        # 使用 OpenCV 读取 PNG 并保存为 JPG
        img = cv2.imread(raw_he_png)
        if img is not None:
            cv2.imwrite(workspace_raw_he_jpg, img)
        else:
            raise FileNotFoundError(f"Cannot read H&E image: {raw_he_png}")

    # 2. 拷贝并重命名 Tissue Mask
    raw_mask = os.path.join(mask_base_dir, f"{slide}.png")
    scaled_mask = os.path.join(prefix, "mask-scaled.png")
    if not os.path.exists(scaled_mask):
        shutil.copy(raw_mask, scaled_mask)

    # 3. 解析 h5ad 提取基因表达和空间坐标 (适配新命名和坐标键)
    cnts_path = os.path.join(prefix, "cnts.tsv")
    locs_raw_path = os.path.join(prefix, "locs-raw.tsv")
    radius_raw_path = os.path.join(prefix, "radius-raw.txt")
    pixel_size_path = os.path.join(prefix, "pixel-size.txt")
    pixel_size_raw_path = os.path.join(prefix, "pixel-size-raw.txt")

    if not os.path.exists(cnts_path) or not os.path.exists(locs_raw_path):
        print("Extracting data from .h5ad...")
        h5ad_file = os.path.join(h5ad_base_dir, f"{slide}_cropped.h5ad")
        adata = sc.read_h5ad(h5ad_file)

        expr_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                               index=adata.obs_names, columns=adata.var_names)
        expr_df.to_csv(cnts_path, sep='\t')

        # 核心修改：使用 spatial_cropped 读取坐标
        locs_df = pd.DataFrame(adata.obsm['spatial_cropped'], index=adata.obs_names, columns=['x', 'y'])
        locs_df.to_csv(locs_raw_path, sep='\t')

        with open(radius_raw_path, "w") as f:
            f.write("50\n")

    with open(pixel_size_path, "w") as f:
        f.write("0.5\n")
    with open(pixel_size_raw_path, "w") as f:
        f.write("0.5\n")

    # 4. 执行 iStar 核心 Pipeline
    device = "cuda"
    python_exe = sys.executable

    try:
        print("\n>>> 4.1 Preprocessing images...")
        run_cmd([python_exe, "rescale.py", prefix, "--image"])
        run_cmd([python_exe, "preprocess.py", prefix, "--image", "--mask"])

        print("\n>>> 4.2 Extracting histology features (HViT)...")
        # 顺应最新版本代码，显式指定图像路径和输出前缀
        img_path = os.path.join(prefix, "he.jpg")
        run_cmd([
            python_exe, "extract_features.py",
            "--img_path", img_path,
            "--out_prefix", prefix,
            f"--device={device}"
        ])

        print("\n>>> 4.3 Selecting top variable genes...")
        run_cmd([python_exe, "select_genes.py", "--n-top=1000", cnts_path, os.path.join(prefix, "gene-names.txt")])

        print("\n>>> 4.4 Rescaling coordinates...")
        run_cmd([python_exe, "rescale.py", prefix, "--locs", "--radius"])
        # ====== 【新增补丁】：自动修复下划线命名错误 ======
        wrong_emb_path = os.path.join(prefix, "_embeddings-hist.pickle")
        correct_emb_path = os.path.join(prefix, "embeddings-hist.pickle")

        # 也有可能后缀生成了其他形态，我们做一个双重保险查找
        if os.path.exists(wrong_emb_path):
            print(f"Auto-fixing filename: renaming {wrong_emb_path} -> embeddings-hist.pickle")
            os.rename(wrong_emb_path, correct_emb_path)
        # ==================================================

        print("\n>>> 4.5 Training Super-resolution Model (This will take a few minutes)...")
        # 传入种子参数
        run_cmd([python_exe, "impute.py", prefix, "--epochs=400", f"--device={device}", f"--seed={seed}"])

        print("\n>>> 4.6 Clustering Super-resolution genes...")
        # 传入种子参数
        run_cmd([
            python_exe, "cluster.py",
            "--filter-size=8",
            "--min-cluster-size=20",
            "--n-clusters=10",
            f"--mask={os.path.join(prefix, 'mask-small.png')}",
            os.path.join(prefix, "embeddings-gene.pickle"),
            os.path.join(prefix, "clusters-gene/"),
            f"--seed={seed}"
        ])

        # =========================================================
        # 4.7 磁盘空间清理 (Garbage Collection)
        # =========================================================
        print("\n>>> 4.7 Cleaning up large intermediate files to save disk space...")

        # 需要删除的超大文件列表
        files_to_remove = [
            os.path.join(prefix, "cnts.tsv"),  # 原始基因表达矩阵
            os.path.join(prefix, "embeddings-hist.pickle"),  # H&E 视觉特征 (G级大小)
            os.path.join(prefix, "embeddings-gene.pickle"),  # 超分辨率基因特征
            os.path.join(prefix, "states", "model.pt")  # 深度学习模型权重
        ]

        # 需要删除的超大文件夹 (包含数千张高变基因的高清分布图)
        dirs_to_remove = [
            os.path.join(prefix, "cnts-super")
        ]

        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)
                print(f"  [Deleted File] {f}")

        for d in dirs_to_remove:
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"  [Deleted Dir] {d}")

        print(f"\n========== Slide {slide} (Seed {seed}) Pipeline Finished Successfully! ==========\n")

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline failed at slide {slide} (Seed {seed}). Command returned non-zero exit status.")
        sys.exit(1)


if __name__ == "__main__":
    # 嵌套循环：为每个 slide 执行 3 个不同种子的独立实验
    for slide in all_samples:
        for seed in seeds:
            prepare_and_run(slide, seed)
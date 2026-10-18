import os
import shutil
import scanpy as sc
import pandas as pd
import numpy as np
import subprocess
import sys
import cv2

# ==========================================
# 基础路径与数据集配置
# ==========================================
base_data_dir = r"D:\Project_2\Pixel_Seg\data"
workspace_dir = r"D:\Project_2\iStar\istar_workspace_three_slices"

# 目标处理切片与多随机种子
slides = ["ViHBC", "DuCIS", "XeHBC"]
seeds = [42, 123, 567]


def run_cmd(cmd_list):
    """使用 subprocess 执行命令，完美避开空格路径和环境跳出的问题"""
    print(f"Running: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True)


def prepare_and_run(slide, seed):
    print(f"\n========== Processing Slide: {slide} | Seed: {seed} ==========")

    # 使得每个切片的每个种子对应独立的保存文件夹
    prefix = os.path.join(workspace_dir, f"{slide}_seed_{seed}\\")
    os.makedirs(prefix, exist_ok=True)

    # 获取切片对应的 resized_data 目录
    resized_dir = os.path.join(base_data_dir, slide, "resized_data")

    # 1. 拷贝并转换 H&E 图像 (iStar 管道通常需要 he-raw.jpg)
    raw_he_png = os.path.join(resized_dir, "he_resized.png")
    workspace_raw_he = os.path.join(prefix, "he-raw.jpg")
    if not os.path.exists(workspace_raw_he):
        # 使用 cv2 转换 png 到 jpg
        img = cv2.imread(raw_he_png)
        if img is None:
            raise ValueError(f"读取 H&E 图像失败: {raw_he_png}")
        cv2.imwrite(workspace_raw_he, img)

    # 2. 拷贝并重命名 Tissue Mask
    raw_mask = os.path.join(resized_dir, "mask_resized.png")
    scaled_mask = os.path.join(prefix, "mask-scaled.png")
    if not os.path.exists(scaled_mask):
        shutil.copy(raw_mask, scaled_mask)

    # 3. 解析 h5ad 提取基因表达和空间坐标
    cnts_path = os.path.join(prefix, "cnts.tsv")
    locs_raw_path = os.path.join(prefix, "locs-raw.tsv")
    radius_raw_path = os.path.join(prefix, "radius-raw.txt")
    pixel_size_path = os.path.join(prefix, "pixel-size.txt")

    if not os.path.exists(cnts_path) or not os.path.exists(locs_raw_path):
        print("Extracting data from .h5ad...")
        h5ad_path = os.path.join(resized_dir, "adata_resized.h5ad")
        adata = sc.read_h5ad(h5ad_path)
        adata.var_names_make_unique()

        # 提取表达矩阵
        expr_df = pd.DataFrame(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )
        expr_df.to_csv(cnts_path, sep='\t')

        # 优先使用 resized 的空间坐标
        spatial_key = "spatial_resized"
        locs_df = pd.DataFrame(adata.obsm[spatial_key], index=adata.obs_names, columns=['x', 'y'])
        locs_df.to_csv(locs_raw_path, sep='\t')

        with open(radius_raw_path, "w") as f:
            f.write("10\n")

    # 预设目标像素尺寸与原始尺寸
    with open(pixel_size_path, "w") as f:
        f.write("0.5\n")
    pixel_size_raw_path = os.path.join(prefix, "pixel-size-raw.txt")
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

        # 兼容处理提取脚本命名可能存在下划线的小 Bug
        wrong_emb_path = os.path.join(prefix, "_embeddings-hist.pickle")
        correct_emb_path = os.path.join(prefix, "embeddings-hist.pickle")
        if os.path.exists(wrong_emb_path):
            print(f"Auto-fixing filename: renaming {wrong_emb_path} -> embeddings-hist.pickle")
            os.rename(wrong_emb_path, correct_emb_path)

        print("\n>>> 4.5 Training Super-resolution Model (This will take a few minutes)...")
        run_cmd([python_exe, "impute.py", prefix, "--epochs=400", f"--device={device}", f"--seed={str(seed)}"])

        print("\n>>> 4.6 Clustering Super-resolution genes...")
        run_cmd([
            python_exe, "cluster.py",
            "--filter-size=8",
            "--min-cluster-size=20",
            "--n-clusters=10",
            f"--mask={os.path.join(prefix, 'mask-small.png')}",
            os.path.join(prefix, "embeddings-gene.pickle"),
            os.path.join(prefix, "clusters-gene/"),
            f"--seed={str(seed)}"
        ])

        print("\n>>> 4.7 Cleaning up large intermediate files to save disk space...")
        files_to_remove = [
            os.path.join(prefix, "cnts.tsv"),
            os.path.join(prefix, "embeddings-hist.pickle"),
            os.path.join(prefix, "embeddings-gene.pickle"),
            os.path.join(prefix, "states", "model.pt")
        ]

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

    except subprocess.CalledProcessError:
        print(f"\n[ERROR] Pipeline failed at slide {slide} (Seed {seed}). Command returned non-zero exit status.")
        sys.exit(1)


if __name__ == "__main__":
    for slide in slides:
        for seed in seeds:
            prepare_and_run(slide, seed)
import os
import shutil
import scanpy as sc
import pandas as pd
import numpy as np
import subprocess
import sys

# 你的数据基础路径
base_data_dir = r"D:\DNM_project(cancer detect)\data\HB\her2st"
mask_base_dir = r"D:\Project_2\Pixel_Seg\data\STHBC\mask"
workspace_dir = r"D:\Project_2\iStar\istar_workspace"

slides = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
# 【新增】：定义需要遍历的随机种子列表
seeds = [42, 123, 567]


def run_cmd(cmd_list):
    """使用 subprocess 执行命令，完美避开空格路径和环境跳出的问题"""
    print(f"Running: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True)


def prepare_and_run(slide, seed):
    print(f"\n========== Processing Slide: {slide} | Seed: {seed} ==========")

    # 【修改 1】：修改输出前缀路径，使得每个种子对应独立的保存文件夹
    prefix = os.path.join(workspace_dir, f"{slide}_seed_{seed}\\")
    os.makedirs(prefix, exist_ok=True)

    # 1. 拷贝 H&E 图像 (必须保持 he-raw.jpg 命名，供 rescale.py 读取)
    raw_he = os.path.join(base_data_dir, slide, "he-raw.jpg")
    workspace_raw_he = os.path.join(prefix, "he-raw.jpg")
    if not os.path.exists(workspace_raw_he):
        shutil.copy(raw_he, workspace_raw_he)

    # 2. 拷贝并重命名 Tissue Mask
    raw_mask = os.path.join(mask_base_dir, f"{slide}_hest.png")
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
        adata = sc.read_h5ad(os.path.join(base_data_dir, slide, f"{slide}.h5ad"))

        expr_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                               index=adata.obs_names, columns=adata.var_names)
        expr_df.to_csv(cnts_path, sep='\t')

        locs_df = pd.DataFrame(adata.obsm['spatial'], index=adata.obs_names, columns=['x', 'y'])
        locs_df.to_csv(locs_raw_path, sep='\t')

        with open(radius_raw_path, "w") as f:
            f.write("50\n")

    # 预设目标像素尺寸
    with open(pixel_size_path, "w") as f:
        f.write("0.5\n")

    # 预设原始图像的像素尺寸
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

        wrong_emb_path = os.path.join(prefix, "_embeddings-hist.pickle")
        correct_emb_path = os.path.join(prefix, "embeddings-hist.pickle")
        if os.path.exists(wrong_emb_path):
            print(f"Auto-fixing filename: renaming {wrong_emb_path} -> embeddings-hist.pickle")
            os.rename(wrong_emb_path, correct_emb_path)

        print("\n>>> 4.5 Training Super-resolution Model (This will take a few minutes)...")
        # 【修改 2】：向模型训练脚本传递 seed 参数
        run_cmd([python_exe, "impute.py", prefix, "--epochs=400", f"--device={device}", f"--seed={seed}"])

        print("\n>>> 4.6 Clustering Super-resolution genes...")
        # 【修改 2】：向聚类脚本传递 seed 参数
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
        # 【修改 3】：移植 1run_CRC_pipeline.py 中的磁盘空间清理逻辑
        # =========================================================
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

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline failed at slide {slide} (Seed {seed}). Command returned non-zero exit status.")
        sys.exit(1)


if __name__ == "__main__":
    # 嵌套循环：为每个 slide 执行 3 个不同种子的独立实验
    for slide in slides:
        for seed in seeds:
            prepare_and_run(slide, seed)
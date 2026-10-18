import pandas as pd
import scanpy as sc
import anndata as ad
from math import e
from typing import Literal, Optional, List
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from PIL import ImageFile
import gc  # 添加垃圾回收模块

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from scipy.sparse import coo_matrix
import torch
import numpy as np
import dgl

import mygene

def ensembl_to_symbol(ensembl_ids):
    """将ENSEMBL ID转换为基因符号"""
    mg = mygene.MyGeneInfo()

    # 批量查询，最多1000个基因
    results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')

    symbol_map = {}
    for result in results:
        if 'symbol' in result:
            symbol_map[result['query']] = result['symbol']
        else:
            # 如果找不到符号，保留原始ENSEMBL ID
            symbol_map[result['query']] = result['query']

    return symbol_map

class Build_multi_graph:
    def __init__(self, adata, image,
                 position, n_neighbors: int = [0, 1, 2, 3, 4, 5][4],  # 4,
                 patch_size: int = 48, train_mode: bool = True):
        self.adata = adata
        self.adata_raw = adata
        self.image = image
        self.position = position
        self.n_dataset = len(adata)
        self.n_neighbors = n_neighbors
        self.patch_size = patch_size
        self.train_mode = train_mode

        self.batch = self.get_batch()
        u, v = self.get_edge()
        self.g = dgl.to_bidirected(dgl.graph((u, v)))
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['batch'] = self.batch
        self.g.ndata['gene'] = self.get_gene()
        self.g.ndata['patch'] = self.get_patch()
        # 清理内存
        gc.collect()

    def get_batch(self):
        adata_list = []
        for i in range(self.n_dataset):
            # 使用copy()避免视图问题
            a = self.adata[i].copy()
            a.obs['batch'] = i
            adata_list.append(a)
            # 每处理一个数据集就清理一次内存
            if i % 2 == 0:
                gc.collect()

        self.adata = ad.concat(adata_list, merge='same')
        self.adata.obs_names_make_unique(join=',')
        batch = np.array(pd.get_dummies(self.adata.obs['batch']), dtype=np.float32)
        return torch.Tensor(batch)

    def get_edge(self):
        self.adata.obs['idx'] = range(self.adata.n_obs)
        u_list, v_list = [], []
        for i in range(self.n_dataset):
            adata = self.adata[self.adata.obs['batch'] == i]
            position = self.position[i]
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1)  # 4
            nbrs = nbrs.fit(position)
            _, indices = nbrs.kneighbors(position)
            u = adata.obs['idx'][indices[:, 0].repeat(self.n_neighbors)]
            v = adata.obs['idx'][indices[:, 1:].flatten()]
            u_list = u_list + u.tolist()
            v_list = v_list + v.tolist()
            # 清理内存
            if i % 2 == 0:
                gc.collect()
        return u_list, v_list

    def get_patch(self):
        self.patch = self.image[0]
        for i in range(self.n_dataset - 1):
            self.patch = torch.concat([self.patch, self.image[i + 1]], dim=0)
            # 清理内存
            if i % 2 == 0:
                gc.collect()
        return self.patch

    def get_gene(self):
        A = coo_matrix(self.adata.X).tocsr()
        Gene = A.todense()
        return torch.Tensor(Gene)  # torch.Tensor(self.adata.X)


def preprocess_data(adata: ad.AnnData):
    adata = adata[:, adata.var_names.notnull()]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=e)
    return adata


def cut_patch(path, position,patch_size=224, train_mode=False):
    patch_size = patch_size  # 128#32
    # 使用更高效的方式加载图像
    img = Image.open(path)
    img = img.convert('RGB')  # 确保是RGB格式

    p_list = []
    r = np.ceil(patch_size / 2).astype(int)

    # 简化预处理流程
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    for i in range(len(position)):
        x, y = position[i, :]
        # 确保坐标在图像范围内
        x = max(r, min(img.width - r, x))
        y = max(r, min(img.height - r, y))

        p = img.crop((x - r, y - r, x + r, y + r))
        p = preprocess(p)
        p_list.append(p)

        # 每处理100个patch清理一次内存
        if i % 100 == 0:
            gc.collect()

    return torch.stack(p_list)


def read(ref_dir, tgt_dir, ref_name, n_genes=3000, overlap=None, preprocess=False):
    """修复版本：确保基因名一致性"""
    ref, ref_img, ref_pos = [], [], []
    tgt, tgt_img, tgt_pos = [], [], []
    ref_g_list = []
    tgt_g_list = []

    # 1. 首先统一处理所有数据集的基因名
    print("Step 1: Loading and standardizing gene names...")

    # 处理参考数据集
    all_ref_genes = []
    for r in ref_name:
        print(f"Loading reference: {r}")
        adata = sc.read(ref_dir + r + '.h5ad')
        mask = adata.obs['in_tissue'] == 1
        adata = adata[mask].copy()

        # 统一基因名处理
        if 'feature_name' in adata.var.columns:
            adata.var_names = adata.var['feature_name'].str.split('_').str[0]
        adata.var_names = [name.upper().strip() for name in adata.var_names]  # 统一大写和去除空格
        adata.var_names_make_unique()

        all_ref_genes.append(set(adata.var_names))
        ref.append(adata)

        # 加载图像和位置
        position = adata.obsm['spatial']
        image = cut_patch(ref_dir + r + '.jpg', position, patch_size=32, train_mode=False)
        ref_img.append(image)
        ref_pos.append(position)

    # 处理目标数据集
    print("Loading target dataset...")
    adata_tgt = sc.read_visium(tgt_dir)
    adata_tgt.var_names_make_unique()

    # 转换目标基因名（更保守的方式）
    print("Converting target gene names...")
    ensembl_ids = list(adata_tgt.var_names)

    # 先尝试简单的ENSEMBL ID处理（去掉版本号）
    simple_tgt_genes = [gene_id.split('.')[0].upper() for gene_id in ensembl_ids]
    adata_tgt.var_names = simple_tgt_genes
    adata_tgt.var_names_make_unique()

    mask = adata_tgt.obs['in_tissue'] == 1
    adata_tgt = adata_tgt[mask].copy()

    tgt.append(adata_tgt)
    position = adata_tgt.obsm['spatial']
    image = cut_patch(tgt_dir + 'he-raw.png', position, patch_size=32, train_mode=False)
    tgt_img.append(image)
    tgt_pos.append(position)

    # 2. 找到真正的公共基因
    print("Step 2: Finding common genes...")

    # 参考数据集的公共基因
    common_ref_genes = set.intersection(*all_ref_genes) if all_ref_genes else set()
    print(f"Common genes in reference datasets: {len(common_ref_genes)}")

    # 目标数据集的基因
    tgt_genes = set(tgt[0].var_names)
    print(f"Target dataset genes: {len(tgt_genes)}")

    # 找到真正的公共基因
    common_genes = common_ref_genes & tgt_genes
    print(f"Final common genes: {len(common_genes)}")

    if len(common_genes) < 100:
        raise ValueError(f"Too few common genes: {len(common_genes)}")

    # 3. 应用公共基因筛选（在所有预处理之前）
    print("Step 3: Applying common gene filter...")
    common_genes_list = list(common_genes)

    for i in range(len(ref)):
        # 确保只保留公共基因中确实存在于当前数据集的基因
        available_genes = [g for g in common_genes_list if g in ref[i].var_names]
        ref[i] = ref[i][:, available_genes]
        print(f"Reference {i}: {ref[i].n_vars} genes after filtering")

    for i in range(len(tgt)):
        available_genes = [g for g in common_genes_list if g in tgt[i].var_names]
        tgt[i] = tgt[i][:, available_genes]
        print(f"Target {i}: {tgt[i].n_vars} genes after filtering")

    # 4. 预处理数据
    print("Step 4: Preprocessing data...")
    ref = [preprocess_data(d) for d in ref if d.n_vars > 0]
    tgt = [preprocess_data(d) for d in tgt if d.n_vars > 0]

    # 5. 高变基因选择（更安全的方式）
    print("Step 5: Selecting highly variable genes...")

    if len(ref) > 0 and ref[0].n_vars > n_genes:
        # 创建参考数据集的副本进行高变基因选择
        ref_for_hvg = ref[0].copy()

        # 确保有足够的基因
        actual_n_genes = min(n_genes, ref_for_hvg.n_vars)
        sc.pp.highly_variable_genes(ref_for_hvg, n_top_genes=actual_n_genes, subset=True)

        hvg_genes = ref_for_hvg.var_names
        print(f"Selected {len(hvg_genes)} HVG")

        # 应用高变基因筛选（确保基因存在）
        for i in range(len(ref)):
            available_hvg = [g for g in hvg_genes if g in ref[i].var_names]
            ref[i] = ref[i][:, available_hvg]

        for i in range(len(tgt)):
            available_hvg = [g for g in hvg_genes if g in tgt[i].var_names]
            tgt[i] = tgt[i][:, available_hvg]

    # 6. 最终检查基因一致性
    print("Step 6: Final gene consistency check...")
    if len(ref) > 0 and len(tgt) > 0:
        ref_genes = set(ref[0].var_names)
        tgt_genes = set(tgt[0].var_names)

        if ref_genes != tgt_genes:
            print("Warning: Gene sets are not identical after all processing")
            print(f"Reference genes: {len(ref_genes)}, Target genes: {len(tgt_genes)}")
            print(f"Intersection: {len(ref_genes & tgt_genes)}")

            # 强制使用交集
            final_common = list(ref_genes & tgt_genes)
            if len(final_common) > 0:
                for i in range(len(ref)):
                    ref[i] = ref[i][:, final_common]
                for i in range(len(tgt)):
                    tgt[i] = tgt[i][:, final_common]

    # 7. 构建图
    print("Step 7: Building graphs...")
    patch_size = 32

    # 构建参考图
    ref_g = Build_multi_graph(ref, ref_img, ref_pos, patch_size=patch_size, train_mode=False).g

    # 构建单个参考图
    for i in range(len(ref)):
        ref_g_list.append(Build_multi_graph([ref[i]], [ref_img[i]], [ref_pos[i]],
                                            patch_size=patch_size, train_mode=False).g)

    # 构建目标图
    for i in range(len(tgt)):
        tgt_g_list.append(Build_multi_graph([tgt[i]], [tgt_img[i]], [tgt_pos[i]],
                                            patch_size=patch_size, train_mode=False).g)

    print('Graph building completed successfully!')
    return ref_g, ref_g_list, tgt_g_list, ref[0].var_names if len(ref) > 0 else []


# 主程序
if __name__ == "__main__":
    ref_dir = "../../data/HB/healthy/"
    tgt_dir = "../../data/HB/Ductal_CIS/"
    ref_name = [ 'v04', 'v05', 'v06', 'v07']#, 'v08', 'v09', 'v10']

    try:
        ref_g, ref_g_list, tgt_g_list, select_gene = read(ref_dir, tgt_dir, ref_name)

        dgl.save_graphs("dataset/DuctalCIS/ref.bin", [ref_g])
        dgl.save_graphs("dataset/DuctalCIS/DuctalCIS_tgt.bin", [tgt_g_list[0]])

        print("数据处理完成！")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
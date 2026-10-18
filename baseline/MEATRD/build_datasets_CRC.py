import dgl
import torch
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from math import e
from typing import Literal, Optional, List
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from PIL import ImageFile
from utils import seed_everything
import warnings
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from scipy.sparse import coo_matrix

class Build_multi_graph:
    def __init__(self, adata, image,
                 position, n_neighbors: int = [0,1,2,3,4,5][4],#4,
                 patch_size: int = 48, train_mode: bool = True):
        seed_everything(0)
        self.adata = adata
        self.adata_raw = adata
        self.image = image
        self.position = position
        self.n_dataset = len(adata)  # 切片数量
        self.n_neighbors = n_neighbors
        self.patch_size = patch_size
        self.train_mode = train_mode

        self.batch = self.get_batch()  # 一个矩阵 表示每个spot属于哪个切片
        u, v = self.get_edge()  # 两个列表 第一个是起点 第二个是终点 起点的话每四个都一样 因为四个邻居 -所有切片的
        # 双向边 挂自环
        self.g = dgl.to_bidirected(dgl.graph((u, v)))
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['batch'] = self.batch  # 一个矩阵 表示每个spot属于哪个切片
        self.g.ndata['gene'] = self.get_gene()  # 基因特征
        self.g.ndata['patch'] = self.get_patch()  # 图像特征
        #if self.image is not None:
            #self.g.ndata['patch'] = self.get_patch()

    def get_batch(self):
        adata = []
        for i in range(self.n_dataset):
            a = self.adata[i]
            a.obs['batch'] = i
            adata.append(a)
        self.adata = ad.concat(adata, merge='same')
        self.adata.obs_names_make_unique(join=',')  # spot名字有区别性
        batch = np.array(pd.get_dummies(self.adata.obs['batch']), dtype=np.float32)  # batch变为独热
        return torch.Tensor(batch)  # 一个矩阵 表示每个spot属于哪个切片

    def get_edge(self):
        self.adata.obs['idx'] = range(self.adata.n_obs)
        u_list, v_list = [], []
        for i in range(self.n_dataset):
            adata = self.adata[self.adata.obs['batch'] == i]
            position = self.position[i]
            # 每个点的邻居索引列表，indices形状是 (num_spots, n_neighbors+1) 一个切片的
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1)
            nbrs = nbrs.fit(position)
            _, indices = nbrs.kneighbors(position)
            u = adata.obs['idx'][indices[:, 0].repeat(self.n_neighbors)]
            v = adata.obs['idx'][indices[:, 1:].flatten()]
            u_list = u_list + u.tolist()
            v_list = v_list + v.tolist()
        return u_list, v_list

    def get_patch(self):
        self.patch = self.image[0] 
        for i in range(self.n_dataset-1):
            self.patch = torch.concat([self.patch, self.image[i+1]], dim=0)
        return self.patch
    
    def get_gene(self):
        A = coo_matrix(self.adata.X).tocsr()
        Gene = A.todense()
        return torch.Tensor(Gene)#torch.Tensor(self.adata.X)
    
def preprocess_data(adata: ad.AnnData):
    seed_everything(0)
    adata = adata[:, adata.var_names.notnull()]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=e)
    return adata
def cut_patch(path, position, train_mode=False,slice_name=" "):
    patch_size = 32#128#32
    img = np.array(Image.open(path))[:, :, :3]
    p_list = []
    #if not isinstance(img[0, 0, 0], np.uint8):
    img = np.uint8(img * 255)
    img = Image.fromarray(img)
    r = np.ceil(patch_size/2).astype(int)
    trans = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                #transforms.ToTensor()
            ])
    preprocess = transforms.Compose([
            transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    def get_coordinate_transform(s_name):
        if s_name == "SN048_1":
            return lambda x, y: (int(x * 2.18) + 480, int(y * 2.18) + 230)
        elif s_name == "SN048_2":
            return lambda x, y: (int(x * 2.18) + 300, int(y * 2.18) + 160)
        elif s_name == "SN048_3":
            return lambda x, y: (int(x * 2.18) + 420, int(y * 2.18) + 80)
        elif s_name == "SN048_4":
            return lambda x, y: (int(x * 2.18) + 200, int(y * 2.18) + 80)
        elif s_name == "SN84_1":
            return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 200)
        elif s_name in ["SN84_2", "SN124_4"]:
            return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 240)
        elif s_name in ["SN123_1", "SN123_2", "SN123_4"]:
            return lambda x, y: (int(x * 2.18) + 270, int(y * 2.18) + 140)
        elif s_name in ["SN124_1", "SN124_2"]:
            return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 280)
        else:
            # 默认转换（如果没有匹配的切片名）
            return lambda x, y: (int(x * 2.18+200), int(y * 2.18+200))
    coord_transform = get_coordinate_transform(slice_name)

    for i in range(len(position)):
        x, y = position[i, :]
        x, y = coord_transform(x, y)
        p = img.crop((x - r, y - r, x + r, y + r))
        #if train_mode:
            #p = trans(p)
        p = preprocess(p)
        p_list.append(p.reshape(3, 2*r, 2*r))
    return torch.stack(p_list)

def read(ref_dir, tgt_dir, ref_name, tgt_name, n_genes=3000, overlap=None, preprocess=False):
    seed_everything(0)
    ref, ref_img, ref_pos = [], [], []
    tgt, tgt_img, tgt_pos = [], [], []
    ref_g_list = []
    tgt_g_list = []
    for r in ref_name:
        adata = sc.read_visium(ref_dir + r)
        adata.var_names_make_unique()
        # 清理不在组织上的spot
        mask = adata.obs['in_tissue'] == 1
        adata._inplace_subset_obs(mask)
        position = adata.obsm['spatial']
        image = cut_patch(f"../../data/CRC/slices_image/{r}.png", position, train_mode=False,slice_name=r)
        ref.append(adata)
        ref_img.append(image)
        ref_pos.append(position)
    for t in tgt_name:
        adata = sc.read_visium(tgt_dir + t)
        adata.var_names_make_unique()
        # 清理不在组织上的spot
        mask = adata.obs['in_tissue'] == 1
        adata._inplace_subset_obs(mask)
        position = adata.obsm['spatial']
        #image = np.array(Image.open(tgt_dir + r + '.png'))
        image = cut_patch(f"../../data/CRC/slices_image/{t}.png", position, train_mode=False,slice_name=t)
        tgt.append(adata)
        tgt_img.append(image)
        tgt_pos.append(position)
    # 找出参考数据和测试数据的共有基因集
    overlap_gene = list(set(ref[0].var_names))
    for i in range(len(ref)-1):    
        overlap_gene=list(set(overlap_gene) & set(ref[i+1].var_names))
    for i in range(len(tgt)):
        overlap_gene=list(set(overlap_gene) & set(tgt[i].var_names))
    ref = [i[:, overlap_gene] for i in ref]
    tgt = [i[:, overlap_gene] for i in tgt]
    ref = [preprocess_data(d) for d in ref]  # 标准化基因
    tgt = [preprocess_data(d) for d in tgt]  # 标准化基因
    # 同一个组织 基因交集还是很多的 再进一步选择高变基因
    sc.pp.highly_variable_genes(ref[0], n_top_genes=n_genes, subset=True)
    #ref[0] = ref[0][:, overlap]
    ref = [d[:, ref[0].var_names] for d in ref]
    tgt = [d[:, ref[0].var_names] for d in tgt]
    patch_size = 32#set_patch(ref[0])#!!!
    print('read over')
    ref_g = Build_multi_graph(ref, ref_img, ref_pos, patch_size=patch_size, train_mode=False).g  # 基因 图像 坐标
    for i in range(len(ref)):
        ref_g_list.append(Build_multi_graph([ref[i]], [ref_img[i]], [ref_pos[i]], patch_size=patch_size, train_mode=False).g)#True).g
    print('ref over')
    for i in range(len(tgt)):
        tgt_g_list.append(Build_multi_graph([tgt[i]], [tgt_img[i]], [tgt_pos[i]], patch_size=patch_size, train_mode=False).g)
    print('bulid graph over')
    # 每一个对象都是一个图结构 节点特征有基因和图像两个模态 这个可以存为bin
    return ref_g, ref_g_list, tgt_g_list, ref[0].var_names

import torch
import numpy as np
import random
import dgl
ref_dir = "../../data/CRC/"
tgt_dir = "../../data/CRC/"
ref_name = ['SN123_3', 'SN124_3']
tgt_name = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4', 'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2', 'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']
ref_g, ref_g_list, tgt_g_list, select_gene = read(ref_dir, tgt_dir, ref_name, tgt_name)#
dgl.save_graphs(f"dataset/CRC/ref.bin", [ref_g])
for i, name in enumerate(tgt_name):
    dgl.save_graphs(f"dataset/CRC/{name}_tgt.bin", [tgt_g_list[i]])

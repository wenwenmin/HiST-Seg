import scanpy as sc
import numpy as np
import scipy.sparse
from torch.utils.data import Dataset
from PIL import Image
from UNI_histology_extractor import UNIExtractor
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()


def row_min_max_normalize(features: np.ndarray) -> np.ndarray:
    row_min = features.min(axis=1, keepdims=True)
    row_max = features.max(axis=1, keepdims=True)
    normalized = (features - row_min) / (row_max - row_min + 1e-8)
    return normalized


class ROIDataset(Dataset):
    def __init__(self, img_list, transform):
        super().__init__()
        self.images_lst = img_list
        self.transform = transform

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        pil_image = Image.fromarray(self.images_lst[idx].astype('uint8'))
        image = self.transform(pil_image)
        return image


def process_her2st_and_visiumhbc(ref_dir, her2st_slices, visiumhbc_path, n_genes=3000):
    # 处理HER2ST切片
    her2st_ref, her2st_img, her2st_pos = [], [], []
    UNI = UNIExtractor()

    # 处理HER2ST的8个切片
    for r in her2st_slices:
        adata = sc.read_h5ad(ref_dir + r + '/' + r + '.h5ad')
        adata.var_names_make_unique()
        position = adata.obsm['spatial']

        image = UNI.extract(ref_dir + r + '/he-raw.jpg', position, crop_size=256)

        her2st_ref.append(adata)
        her2st_img.append(image)
        her2st_pos.append(position)
        print(f'HER2ST Section {r} is processed...')

    # 处理Visium_HBC切片
    print('Processing Visium_HBC section...')
    visiumhbc_adata = sc.read_visium(visiumhbc_path)
    visiumhbc_adata.var_names_make_unique()
    visiumhbc_position = visiumhbc_adata.obsm['spatial']

    # 使用新的图像路径
    visiumhbc_image_path = "../../data/HB/Visium_HBC/he-raw.jpg"
    visiumhbc_image = UNI.extract(visiumhbc_image_path, visiumhbc_position, crop_size=224)

    # 找到所有切片的公共基因（包括HER2ST和Visium_HBC）
    all_genes = set(her2st_ref[0].var_names)
    for r in her2st_ref[1:]:
        all_genes &= set(r.var_names)
    all_genes &= set(visiumhbc_adata.var_names)
    common_genes = list(all_genes)
    print(f'Found {len(common_genes)} common genes across all sections')

    # 只保留公共基因
    her2st_ref = [r[:, common_genes] for r in her2st_ref]
    visiumhbc_adata = visiumhbc_adata[:, common_genes]

    # 对HER2ST切片进行归一化和高变基因筛选
    for d in her2st_ref:
        sc.pp.normalize_total(d, target_sum=1e4)
        sc.pp.log1p(d)

    sc.pp.highly_variable_genes(her2st_ref[0], flavor="seurat_v3", n_top_genes=n_genes, subset=True)
    hvg_genes = her2st_ref[0].var_names

    # 所有HER2ST切片只保留高变基因
    her2st_ref = [d[:, hvg_genes] for d in her2st_ref]

    # Visium_HBC切片也使用相同的高变基因
    visiumhbc_adata = visiumhbc_adata[:, hvg_genes]

    # 对Visium_HBC切片进行相同的归一化处理
    sc.pp.normalize_total(visiumhbc_adata, target_sum=1e4)
    sc.pp.log1p(visiumhbc_adata)

    # 提取基因表达数据
    her2st_expr = [adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X for adata in her2st_ref]
    visiumhbc_expr = visiumhbc_adata.X.toarray() if scipy.sparse.issparse(visiumhbc_adata.X) else visiumhbc_adata.X

    # 保存训练数据集（HER2ST切片）
    np.savez_compressed(
        "data/Visium_HBC/train_VisiumHBC_dataset.npz",
        img_feats=np.array(her2st_img, dtype=object),
        gene_exprs=np.array(her2st_expr, dtype=object),
        coords=np.array(her2st_pos, dtype=object)
    )

    # 保存测试数据集（Visium_HBC切片）
    np.savez_compressed(
        "data/Visium_HBC/test_VisiumHBC_dataset.npz",
        img_feats=np.array([visiumhbc_image], dtype=object),
        gene_exprs=np.array([visiumhbc_expr], dtype=object),
        coords=np.array([visiumhbc_position], dtype=object)
    )

    print("Training and test datasets saved successfully!")


# HER2ST数据集的8个切片
her2st_slices = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
her2st_dir = "../../data/HB/her2st/"
visiumhbc_path = "../../data/HB/Visium_HBC"

process_her2st_and_visiumhbc(her2st_dir, her2st_slices, visiumhbc_path)
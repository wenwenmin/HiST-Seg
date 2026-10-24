# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
# Spot-level protocol:
# Generate spot predictions from the method-native, original-scale H&E/ST data
# (or features prepared from those data). Do not use resized_data/cropped_data
# as model inputs. Those aligned files are used only for pixel-level evaluation.
# Preserve the original AnnData observation order in every prediction file.
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


def process_her2st_and_ducis(ref_dir, her2st_slices, ducis_path, n_genes=3000):
    
    her2st_ref, her2st_img, her2st_pos = [], [], []
    UNI = UNIExtractor()

    
    for r in her2st_slices:
        adata = sc.read_h5ad(ref_dir + r + '/' + r + '.h5ad')
        adata.var_names_make_unique()
        position = adata.obsm['spatial']

        image = UNI.extract(ref_dir + r + '/he-raw.jpg', position, crop_size=256)

        her2st_ref.append(adata)
        her2st_img.append(image)
        her2st_pos.append(position)
        print(f'HER2ST Section {r} is processed...')

    
    print('Processing DuCIS section...')
    ducis_adata = sc.read_visium(ducis_path)
    ducis_adata.var_names_make_unique()
    ducis_position = ducis_adata.obsm['spatial']

    # Configure this path for your environment; see readme_baseline.txt.
    ducis_image_path = "../../data/HB/Ductal_CIS/he-raw.jpg"
    ducis_image = UNI.extract(ducis_image_path, ducis_position, crop_size=224)

    
    all_genes = set(her2st_ref[0].var_names)
    for r in her2st_ref[1:]:
        all_genes &= set(r.var_names)
    all_genes &= set(ducis_adata.var_names)
    common_genes = list(all_genes)
    print(f'Found {len(common_genes)} common genes across all sections')

    
    her2st_ref = [r[:, common_genes] for r in her2st_ref]
    ducis_adata = ducis_adata[:, common_genes]

    
    for d in her2st_ref:
        sc.pp.normalize_total(d, target_sum=1e4)
        sc.pp.log1p(d)

    sc.pp.highly_variable_genes(her2st_ref[0], flavor="seurat_v3", n_top_genes=n_genes, subset=True)
    hvg_genes = her2st_ref[0].var_names

    
    her2st_ref = [d[:, hvg_genes] for d in her2st_ref]

    
    ducis_adata = ducis_adata[:, hvg_genes]

    
    sc.pp.normalize_total(ducis_adata, target_sum=1e4)
    sc.pp.log1p(ducis_adata)

    
    her2st_expr = [adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X for adata in her2st_ref]
    ducis_expr = ducis_adata.X.toarray() if scipy.sparse.issparse(ducis_adata.X) else ducis_adata.X

    
    np.savez_compressed(
        "data/DuCIS/train_DuCIS_dataset.npz",
        img_feats=np.array(her2st_img, dtype=object),
        gene_exprs=np.array(her2st_expr, dtype=object),
        coords=np.array(her2st_pos, dtype=object)
    )

    
    np.savez_compressed(
        "data/DuCIS/test_DuCIS_dataset.npz",
        img_feats=np.array([ducis_image], dtype=object),
        gene_exprs=np.array([ducis_expr], dtype=object),
        coords=np.array([ducis_position], dtype=object)
    )

    print("Training and test datasets saved successfully!")



her2st_slices = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
her2st_dir = "../../data/HB/her2st/"
ducis_path = "../../data/HB/Ductal_CIS"

process_her2st_and_ducis(her2st_dir, her2st_slices, ducis_path)

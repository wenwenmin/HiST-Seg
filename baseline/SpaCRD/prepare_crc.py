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


def get_coordinate_transform(slice_name):
    'Implementation of get_coordinate_transform for this baseline experiment.'
    if slice_name == "SN048_1":
        return lambda x, y: (int(x * 2.18) + 480, int(y * 2.18) + 230)
    elif slice_name == "SN048_2":
        return lambda x, y: (int(x * 2.18) + 300, int(y * 2.18) + 160)
    elif slice_name == "SN048_3":
        return lambda x, y: (int(x * 2.18) + 420, int(y * 2.18) + 80)
    elif slice_name == "SN048_4":
        return lambda x, y: (int(x * 2.18) + 200, int(y * 2.18) + 80)
    elif slice_name == "SN84_1":
        return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 200)
    elif slice_name in ["SN84_2", "SN124_4"]:
        return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 240)
    elif slice_name in ["SN123_1", "SN123_2", "SN123_4"]:
        return lambda x, y: (int(x * 2.18) + 270, int(y * 2.18) + 140)
    elif slice_name in ["SN124_1", "SN124_2"]:
        return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 280)
    else:
        
        return lambda x, y: (int(x * 2.18), int(y * 2.18))


def process(ref_dir, ref_name, n_genes=3000, overlap=None, preprocess=False):
    ref, ref_img, ref_pos = [], [], []
    UNI = UNIExtractor()

    for r in ref_name:
        adata = sc.read_visium(ref_dir + r)
        adata.var_names_make_unique()

        mask = adata.obs['in_tissue'] == 1
        adata._inplace_subset_obs(mask)
        position = adata.obsm['spatial']

        # Configure this path for your environment; see readme_baseline.txt.
        image_path = f"../../data/CRC/slices_image/{r}.png"

        
        coord_transform = get_coordinate_transform(r)
        transformed_positions = np.array([coord_transform(x, y) for x, y in position])

        # Configure this path for your environment; see readme_baseline.txt.
        image = UNI.extract(image_path, transformed_positions, crop_size=52)

        ref.append(adata)
        ref_img.append(image)
        ref_pos.append(position)  
        print(f'Section {r} is processed...')

    
    overlap_genes = set(ref[0].var_names)
    for r in ref[1:]:
        overlap_genes &= set(r.var_names)
    overlap_genes = list(overlap_genes)

    ref = [r[:, overlap_genes] for r in ref]

    for d in ref:
        sc.pp.normalize_total(d, target_sum=1e4)
        sc.pp.log1p(d)

    sc.pp.highly_variable_genes(ref[0], flavor="seurat_v3", n_top_genes=n_genes, subset=True)

    ref = [d[:, ref[0].var_names] for d in ref]

    ref_expr = [adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X for adata in ref]
    del ref

    np.savez_compressed(
        f"data/CRC/train_dataset(new).npz",
        img_feats=np.array(ref_img, dtype=object),
        gene_exprs=np.array(ref_expr, dtype=object),
        coords=np.array(ref_pos, dtype=object)
    )


ref_dir = "../../data/CRC/"
ref_name = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4', 'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2', 'SN123_3', 'SN123_4',
            'SN124_1', 'SN124_2', 'SN124_3', 'SN124_4']

process(ref_dir, ref_name)

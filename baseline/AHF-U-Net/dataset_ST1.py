import os


os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import re
import cv2
import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HEDataset(Dataset):
    def __init__(self, patch_dir, h5ad_base_dir, allowed_slices=None, transform=None, max_spots=16, st_dim=1000,
                 patch_size=256, min_tissue_ratio=0.01):
        'Implementation of __init__ for this baseline experiment.'
        self.patch_dir = patch_dir
        self.transform = transform
        self.max_spots = max_spots
        self.st_dim = st_dim
        self.patch_size = patch_size
        self.min_tissue_ratio = min_tissue_ratio

        # Configure this path for your environment; see readme_baseline.txt.
        self.image_paths = []
        img_folder = os.path.join(patch_dir, "images")
        tissue_folder = os.path.join(patch_dir, "tissue_masks")

        print(f"⏳ Scanning patches and filtering pure background areas (threshold:{min_tissue_ratio * 100}%)...")
        valid_count = 0
        skipped_count = 0

        for file in os.listdir(img_folder):
            if not file.endswith('.jpg'):
                continue

            
            match = re.match(r"(.+)_x\d+_y\d+\.jpg", file)
            if match:
                slice_name = match.group(1)
            else:
                slice_name = file.split('_')[0]  

            if allowed_slices and slice_name not in allowed_slices:
                continue

            
            if self.min_tissue_ratio > 0:
                tissue_mask_path = os.path.join(tissue_folder, file.replace('.jpg', '.png'))
                if os.path.exists(tissue_mask_path):
                    tissue_img = cv2.imdecode(np.fromfile(tissue_mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if tissue_img is not None:
                        tissue_ratio = np.sum(tissue_img > 127) / tissue_img.size
                        if tissue_ratio < self.min_tissue_ratio:
                            skipped_count += 1
                            continue

            self.image_paths.append(file)
            valid_count += 1

        print(f"✅ Scanning completed! reserve{valid_count}valid patches, physically filtered{skipped_count}invalid background patches.")

        
        self.st_cache = {}
        if allowed_slices:
            print('⏳ Caching spatial transcriptome feature matrix...')
            for s_name in allowed_slices:
                h5ad_path = os.path.join(h5ad_base_dir, f"{s_name}_aligned_256.h5ad")

                if os.path.exists(h5ad_path):
                    adata = sc.read_h5ad(h5ad_path)
                    feats = adata.obsm['aligned_features']
                    feats_dense = feats.toarray() if hasattr(feats, 'toarray') else feats

                    self.st_cache[s_name] = {
                        'coords': adata.obsm['spatial_resized'],
                        'features': feats_dense.astype(np.float32)
                    }
                else:
                    print(f"⚠️ WARNING: Feature file not found{h5ad_path}")
            print(f"✅ Successfully cached{len(self.st_cache)}ST data of each slice.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_name = self.image_paths[idx]
        img_path = os.path.join(self.patch_dir, "images", file_name)
        mask_path = os.path.join(self.patch_dir, "masks", file_name.replace('.jpg', '.png'))
        tissue_mask_path = os.path.join(self.patch_dir, "tissue_masks", file_name.replace('.jpg', '.png'))

        
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        tissue_mask = cv2.imdecode(np.fromfile(tissue_mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        tissue_mask = (tissue_mask > 127).astype(np.float32)

        
        if self.transform:
            combined_masks = np.stack([mask, tissue_mask], axis=-1)
            augmented = self.transform(image=image, mask=combined_masks)
            image = augmented['image']
            aug_mask_tensor = augmented['mask']

            mask = aug_mask_tensor[:, :, 0].unsqueeze(0)
            tissue_mask_tensor = aug_mask_tensor[:, :, 1].unsqueeze(0)
        else:
            mask = torch.tensor(mask).unsqueeze(0)
            tissue_mask_tensor = torch.tensor(tissue_mask).unsqueeze(0)

        
        
        match = re.match(r"(.+)_x\d+_y\d+\.jpg", file_name)
        if match:
            slice_name = match.group(1)
        else:
            slice_name = file_name.split('_')[0]

        try:
            x_min = int(re.search(r'_x(\d+)_', file_name).group(1))
            y_min = int(re.search(r'_y(\d+)\.', file_name).group(1))
        except AttributeError:
            raise ValueError(f"Unable to retrieve file name{file_name}The coordinates are parsed in, please check!")

        x_max = x_min + self.patch_size
        y_max = y_min + self.patch_size

        
        grid_size = 4
        spot_physical_size = 64

        st_grid = np.zeros((grid_size, grid_size, self.st_dim), dtype=np.float32)
        count_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        
        if slice_name in self.st_cache:
            coords = self.st_cache[slice_name]['coords']
            features = self.st_cache[slice_name]['features']

            valid_idx = np.where(
                (coords[:, 0] >= x_min) & (coords[:, 0] < x_max) &
                (coords[:, 1] >= y_min) & (coords[:, 1] < y_max)
            )[0]

            for idx in valid_idx:
                local_x = coords[idx, 0] - x_min
                local_y = coords[idx, 1] - y_min

                col = int(local_x // spot_physical_size)
                row = int(local_y // spot_physical_size)

                col = min(col, grid_size - 1)
                row = min(row, grid_size - 1)

                st_grid[row, col] += features[idx]
                count_grid[row, col] += 1

        
        valid_cells = count_grid > 0
        st_grid[valid_cells] = st_grid[valid_cells] / count_grid[valid_cells][..., np.newaxis]

        
        grid_mask = (count_grid == 0)

        st_tensor = torch.tensor(st_grid).permute(2, 0, 1)
        mask_tensor = torch.tensor(grid_mask)

        return image, mask, tissue_mask_tensor, st_tensor, mask_tensor


# ==============================================================================

# ==============================================================================
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.05, p=0.5),
            
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=2, min_height=8, min_width=8,
                            fill_value=255, mask_fill_value=0, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
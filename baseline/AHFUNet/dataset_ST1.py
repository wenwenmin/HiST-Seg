import os

# 禁用 Albumentations 的联网版本检查 (消除握手超时警告)
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
        """
        patch_dir: 存放 Patch 的根目录 (包含 images, masks, 和 tissue_masks 子文件夹)
        h5ad_base_dir: 包含所有切片 HR_aligned_xxx.h5ad 的父目录
        allowed_slices: 当前 Fold 允许读取的切片列表
        max_spots: 每个 Patch 允许容纳的最大 Spot 数量
        st_dim: 基因特征的固定维度 (1000 或由 Fold-Aware 动态确定)
        patch_size: 切块大小 (256)
        min_tissue_ratio: 最小组织占比阈值。默认为 0.01 (1%)，低于此阈值的 Patch 将被直接丢弃。
        """
        self.patch_dir = patch_dir
        self.transform = transform
        self.max_spots = max_spots
        self.st_dim = st_dim
        self.patch_size = patch_size
        self.min_tissue_ratio = min_tissue_ratio

        # 1. 搜集图像路径并进行背景过滤
        self.image_paths = []
        img_folder = os.path.join(patch_dir, "images")
        tissue_folder = os.path.join(patch_dir, "tissue_masks")

        print(f"⏳ 正在扫描 Patch 并过滤纯背景区域 (阈值: {min_tissue_ratio * 100}%)...")
        valid_count = 0
        skipped_count = 0

        for file in os.listdir(img_folder):
            if not file.endswith('.jpg'):
                continue

            # 🌟 修复 1：使用正则表达式精准提取包含下划线的完整切片名 (例如 SN048_1)
            match = re.match(r"(.+)_x\d+_y\d+\.jpg", file)
            if match:
                slice_name = match.group(1)
            else:
                slice_name = file.split('_')[0]  # 兼容没带坐标的旧格式

            if allowed_slices and slice_name not in allowed_slices:
                continue

            # --- 背景过滤逻辑 ---
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

        print(f"✅ 扫描完毕！保留 {valid_count} 个有效 Patch，物理过滤了 {skipped_count} 个无效背景 Patch。")

        # 2. 预缓存空间转录组 (ST) 数据到内存
        self.st_cache = {}
        if allowed_slices:
            print("⏳ 正在缓存空间转录组特征矩阵...")
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
                    print(f"⚠️ 警告: 找不到特征文件 {h5ad_path}")
            print(f"✅ 成功缓存 {len(self.st_cache)} 个切片的 ST 数据。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_name = self.image_paths[idx]
        img_path = os.path.join(self.patch_dir, "images", file_name)
        mask_path = os.path.join(self.patch_dir, "masks", file_name.replace('.jpg', '.png'))
        tissue_mask_path = os.path.join(self.patch_dir, "tissue_masks", file_name.replace('.jpg', '.png'))

        # 1. 读取图像与标签
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        tissue_mask = cv2.imdecode(np.fromfile(tissue_mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        tissue_mask = (tissue_mask > 127).astype(np.float32)

        # 2. 图像数据增强
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

        # 3. 解析当前 Patch 的左上角物理坐标
        # 🌟 修复 2：在获取单个数据时，同样精准提取切片名
        match = re.match(r"(.+)_x\d+_y\d+\.jpg", file_name)
        if match:
            slice_name = match.group(1)
        else:
            slice_name = file_name.split('_')[0]

        try:
            x_min = int(re.search(r'_x(\d+)_', file_name).group(1))
            y_min = int(re.search(r'_y(\d+)\.', file_name).group(1))
        except AttributeError:
            raise ValueError(f"无法从文件名 {file_name} 中解析出坐标，请检查！")

        x_max = x_min + self.patch_size
        y_max = y_min + self.patch_size

        # 4. 初始化 8x8 的 ST 空间网格和【Spot计数器】
        grid_size = 4
        spot_physical_size = 64

        st_grid = np.zeros((grid_size, grid_size, self.st_dim), dtype=np.float32)
        count_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        # 5. 精准网格映射与均值池化
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

        # 对具有 Spot 的网格执行均值池化，防止数据丢失
        valid_cells = count_grid > 0
        st_grid[valid_cells] = st_grid[valid_cells] / count_grid[valid_cells][..., np.newaxis]

        # 直接利用计数器生成精确的注意力掩码
        grid_mask = (count_grid == 0)

        st_tensor = torch.tensor(st_grid).permute(2, 0, 1)
        mask_tensor = torch.tensor(grid_mask)

        return image, mask, tissue_mask_tensor, st_tensor, mask_tensor


# ==============================================================================
# --- 高级病理图像数据增强 ---
# ==============================================================================
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.05, p=0.5),
            # 🌟 修复 3：适配最新版 Albumentations，将 mask_fill 改为 mask_fill_value，将 num_holes_range 改为新版支持的参数格式
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
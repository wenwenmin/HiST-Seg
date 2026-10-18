import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import time
from datetime import timedelta

# 复用已有的多模态数据加载器，但只取图像与掩码
from dataset_ST1 import HEDataset, get_transforms
from ahf_unet_baseline import AHFUnet_Baseline


def set_seed(seed):
    """设置全局随机种子以保证实验可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_fold(test_slice, all_slices, patch_dir, h5ad_base_dir, seed, epochs=20, batch_size=8, lr=1e-4,
                   save_root="checkpoints_baseline"):
    # 应用当前轮次的随机种子
    set_seed(seed)

    fold_start_time = time.time()
    save_dir = os.path.join(save_root, "pure_image_CRC")
    os.makedirs(save_dir, exist_ok=True)

    # 权重文件名加入 seed 标识，防止不同种子互相覆盖
    save_path = os.path.join(save_dir, f"best_model_{test_slice}_baseline_seed{seed}.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_slices = [s for s in all_slices if s != test_slice]

    print(f"\n" + "=" * 50)
    print(f"📊 [Baseline] 留一法交叉验证 (CRC) | 独立测试切片: {test_slice} | 种子: {seed}")
    print(f"训练资源池 ({len(train_slices)}个切片): {train_slices}")
    print("=" * 50)

    # 数据集复用加载 (保留 CRC 专属的 st_dim 和 patch_size 设置)
    full_train_pool_ds = HEDataset(patch_dir, h5ad_base_dir, allowed_slices=train_slices,
                                   transform=get_transforms(is_train=True), st_dim=256, patch_size=256)
    full_val_pool_ds = HEDataset(patch_dir, h5ad_base_dir, allowed_slices=train_slices,
                                 transform=get_transforms(is_train=False), st_dim=256, patch_size=256)
    test_ds = HEDataset(patch_dir, h5ad_base_dir, allowed_slices=[test_slice], transform=get_transforms(is_train=False),
                        st_dim=256, patch_size=256)

    num_total_patches = len(full_train_pool_ds)
    val_size = int(num_total_patches * 0.2)
    train_size = num_total_patches - val_size

    indices = np.arange(num_total_patches)
    # 此处的随机打乱会受到最开头 set_seed(seed) 的严格控制
    np.random.shuffle(indices)

    train_loader = DataLoader(Subset(full_train_pool_ds, indices[:train_size]), batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(Subset(full_val_pool_ds, indices[train_size:]), batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化 Baseline 模型（全网从头训练，无冻结逻辑）
    model = AHFUnet_Baseline().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for images, masks, tissue_masks, _, _ in pbar:  # 忽略后两个 ST 变量
            images, masks, tissue_masks = images.to(device), masks.to(device).float(), tissue_masks.to(device).float()

            optimizer.zero_grad()
            with autocast('cuda'):
                # 传入占位参数保持底层签名兼容
                logits, loss, loss_dict = model(x=images, targets=masks, tissue_masks=tissue_masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(Focal=f"{loss_dict['focal']:.3f}", Dice=f"{loss_dict['dice']:.3f}")

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks, tissue_masks, _, _ in valid_loader:
                images, masks, tissue_masks = images.to(device), masks.to(device).float(), tissue_masks.to(
                    device).float()
                with autocast('cuda'):
                    _, loss, _ = model(x=images, targets=masks, tissue_masks=tissue_masks)
                    val_loss += loss.item()

        avg_val = val_loss / len(valid_loader)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} 结束 | Val Loss: {avg_val:.4f} | 耗时: {timedelta(seconds=int(epoch_duration))}")

        scheduler.step(avg_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"  -> 🌟 权重已更新至: {os.path.basename(save_path)}")

    # 测试集评估
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_dice_list = []
    with torch.no_grad():
        for images, masks, tissue_masks, _, _ in test_loader:
            images, masks, tissue_masks = images.to(device), masks.to(device).float(), tissue_masks.to(device).float()
            logits = model(images)
            pred = (torch.sigmoid(logits) > 0.5).float()
            intersection = (pred * masks * tissue_masks).sum()
            union = (pred * tissue_masks).sum() + (masks * tissue_masks).sum()
            test_dice_list.append(((2. * intersection + 1e-6) / (union + 1e-6)).item())

    fold_duration = time.time() - fold_start_time
    print(
        f"\n✅ [{test_slice} | Seed {seed}] 测试平均 Dice: {np.mean(test_dice_list):.4f} | 总耗时: {timedelta(seconds=int(fold_duration))}")
    return fold_duration


if __name__ == "__main__":
    # CRC 数据集路径和切片
    PATCH_DIR = r"D:\Project_2\Pixel_Seg\data\CRC\patches_256(xy)"
    H5AD_BASE_DIR = r"D:\Project_2\Pixel_Seg\data\CRC\ST_256D"
    SLICES = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4',
              'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2',
              'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']

    # 实验设定的三个随机种子
    SEEDS = [42, 123, 567]

    total_start = time.time()
    all_durations = []

    # 双重循环：遍历种子和切片
    for seed in SEEDS:
        print(f"\n" + "🌟" * 25)
        print(f"🚀 开始全新实验轮次 | 随机种子: {seed}")
        print("🌟" * 25)

        for ts in SLICES:
            # 传入当前循环的 seed
            duration = train_one_fold(ts, SLICES, PATCH_DIR, H5AD_BASE_DIR, seed=seed, epochs=20, batch_size=8)
            all_durations.append(duration)

    print("\n" + "图像单模态 Baseline 统计报告 (CRC 数据集)".center(50, "="))
    print(f"总计训练 Fold 数: {len(SLICES) * len(SEEDS)} (切片: {len(SLICES)} x 种子: {len(SEEDS)})")
    print(f"全过程总耗时: {timedelta(seconds=int(time.time() - total_start))}")
    print(f"平均每个 Fold 耗时: {timedelta(seconds=int(np.mean(all_durations)))}")
    print("=" * 50)



# import os
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# from torch.amp import autocast, GradScaler
# from tqdm import tqdm
# import numpy as np
# import time
# from datetime import timedelta
#
# # 复用已有的多模态数据加载器，但只取图像与掩码
# from dataset_ST1 import HEDataset, get_transforms
# from ahf_unet_baseline import AHFUnet_Baseline
#
# def train_one_fold(test_slice, all_slices, patch_dir, h5ad_base_dir, epochs=20, batch_size=8, lr=1e-4, save_root="checkpoints_baseline"):
#     fold_start_time = time.time()
#     save_dir = os.path.join(save_root, "pure_image_CRC") # 建议稍微修改下保存路径名称以区分数据集
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"best_model_{test_slice}_baseline.pth")
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train_slices = [s for s in all_slices if s != test_slice]
#
#     print(f"\n" + "=" * 50)
#     print(f"📊 [Baseline] 留一法交叉验证 | 独立测试切片: {test_slice}")
#     print(f"训练资源池 ({len(train_slices)}个切片): {train_slices}")
#     print("=" * 50)
#
#     # 数据集复用加载
#     full_train_pool_ds = HEDataset(patch_dir, h5ad_base_dir, allowed_slices=train_slices, transform=get_transforms(is_train=True),st_dim=256,patch_size=256)
#     full_val_pool_ds = HEDataset(patch_dir, h5ad_base_dir, allowed_slices=train_slices, transform=get_transforms(is_train=False),st_dim=256,patch_size=256)
#     test_ds = HEDataset(patch_dir, h5ad_base_dir, allowed_slices=[test_slice], transform=get_transforms(is_train=False),st_dim=256,patch_size=256)
#
#     num_total_patches = len(full_train_pool_ds)
#     val_size = int(num_total_patches * 0.2)
#     train_size = num_total_patches - val_size
#
#     indices = np.arange(num_total_patches)
#     np.random.seed(42)
#     np.random.shuffle(indices)
#
#     train_loader = DataLoader(Subset(full_train_pool_ds, indices[:train_size]), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
#     valid_loader = DataLoader(Subset(full_val_pool_ds, indices[train_size:]), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
#
#     # 初始化 Baseline 模型（全网从头训练，无冻结逻辑）
#     model = AHFUnet_Baseline().to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
#     scaler = GradScaler('cuda')
#
#     best_val_loss = float('inf')
#
#     for epoch in range(epochs):
#         epoch_start_time = time.time()
#         model.train()
#         train_loss = 0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
#
#         for images, masks, tissue_masks, _, _ in pbar:  # 忽略后两个 ST 变量
#             images, masks, tissue_masks = images.to(device), masks.to(device).float(), tissue_masks.to(device).float()
#
#             optimizer.zero_grad()
#             with autocast('cuda'):
#                 # 传入占位参数保持底层签名兼容
#                 logits, loss, loss_dict = model(x=images, targets=masks, tissue_masks=tissue_masks)
#
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#             train_loss += loss.item()
#             pbar.set_postfix(Focal=f"{loss_dict['focal']:.3f}", Dice=f"{loss_dict['dice']:.3f}")
#
#         # 验证阶段
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for images, masks, tissue_masks, _, _ in valid_loader:
#                 images, masks, tissue_masks = images.to(device), masks.to(device).float(), tissue_masks.to(device).float()
#                 with autocast('cuda'):
#                     _, loss, _ = model(x=images, targets=masks, tissue_masks=tissue_masks)
#                     val_loss += loss.item()
#
#         avg_val = val_loss / len(valid_loader)
#         epoch_duration = time.time() - epoch_start_time
#         print(f"Epoch {epoch + 1} 结束 | Val Loss: {avg_val:.4f} | 耗时: {timedelta(seconds=int(epoch_duration))}")
#
#         scheduler.step(avg_val)
#         if avg_val < best_val_loss:
#             best_val_loss = avg_val
#             torch.save(model.state_dict(), save_path)
#             print(f"  -> 🌟 权重已更新")
#
#     # 测试集评估
#     model.load_state_dict(torch.load(save_path))
#     model.eval()
#     test_dice_list = []
#     with torch.no_grad():
#         for images, masks, tissue_masks, _, _ in test_loader:
#             images, masks, tissue_masks = images.to(device), masks.to(device).float(), tissue_masks.to(device).float()
#             logits = model(images)
#             pred = (torch.sigmoid(logits) > 0.5).float()
#             intersection = (pred * masks * tissue_masks).sum()
#             union = (pred * tissue_masks).sum() + (masks * tissue_masks).sum()
#             test_dice_list.append(((2. * intersection + 1e-6) / (union + 1e-6)).item())
#
#     fold_duration = time.time() - fold_start_time
#     print(f"\n✅ [{test_slice}] 测试平均 Dice: {np.mean(test_dice_list):.4f} | 总耗时: {timedelta(seconds=int(fold_duration))}")
#     return fold_duration
#
#
# if __name__ == "__main__":
#     # 更新为 CRC 数据集路径和切片
#     PATCH_DIR = r"D:\Project_2\Pixel_Seg\data\CRC\patches_256(xy)"
#     H5AD_BASE_DIR = r"D:\Project_2\Pixel_Seg\data\CRC\ST_256D"
#     SLICES = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4',
#               'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2',
#               'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']
#
#     total_start = time.time()
#     durations = [train_one_fold(ts, SLICES, PATCH_DIR, H5AD_BASE_DIR, epochs=20, batch_size=8) for ts in SLICES]
#
#     print("\n" + "图像单模态 Baseline 统计报告 (CRC 数据集)".center(50, "="))
#     print(f"总计训练 Fold 数: {len(SLICES)}")
#     print(f"全过程总耗时: {timedelta(seconds=int(time.time() - total_start))}")
#     print(f"平均每个 Fold 耗时: {timedelta(seconds=int(np.mean(durations)))}")
#     print("=" * 50)
# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
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


from dataset_ST1 import HEDataset, get_transforms
from ahf_unet_baseline import AHFUnet_Baseline


def set_seed(seed):
    'Implementation of set_seed for this baseline experiment.'
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
    
    set_seed(seed)

    fold_start_time = time.time()
    save_dir = os.path.join(save_root, "pure_image")
    os.makedirs(save_dir, exist_ok=True)

    
    save_path = os.path.join(save_dir, f"best_model_{test_slice}_baseline_seed{seed}.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_slices = [s for s in all_slices if s != test_slice]

    print(f"\n" + "=" * 50)
    print(f"📊 [Baseline] Leave-one-out cross-validation | Independent test slices:{test_slice}| Seed:{seed}")
    print(f"Training resource pool (7 slices):{train_slices}")
    print("=" * 50)

    
    full_train_pool_ds = HEDataset(patch_dir, h5ad_base_dir, allowed_slices=train_slices,
                                   transform=get_transforms(is_train=True))
    full_val_pool_ds = HEDataset(patch_dir, h5ad_base_dir, allowed_slices=train_slices,
                                 transform=get_transforms(is_train=False))
    test_ds = HEDataset(patch_dir, h5ad_base_dir, allowed_slices=[test_slice], transform=get_transforms(is_train=False))

    num_total_patches = len(full_train_pool_ds)
    val_size = int(num_total_patches * 0.2)
    train_size = num_total_patches - val_size

    indices = np.arange(num_total_patches)
    
    np.random.shuffle(indices)

    train_loader = DataLoader(Subset(full_train_pool_ds, indices[:train_size]), batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(Subset(full_val_pool_ds, indices[train_size:]), batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    
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

        for images, masks, tissue_masks, _, _ in pbar:  
            images, masks, tissue_masks = images.to(device), masks.to(device).float(), tissue_masks.to(device).float()

            optimizer.zero_grad()
            with autocast('cuda'):
                
                logits, loss, loss_dict = model(x=images, targets=masks, tissue_masks=tissue_masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(Focal=f"{loss_dict['focal']:.3f}", Dice=f"{loss_dict['dice']:.3f}")

        
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
        print(f"Epoch {epoch + 1}End | Val Loss:{avg_val:.4f}| Time taken:{timedelta(seconds=int(epoch_duration))}")

        scheduler.step(avg_val)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"-> 🌟 The weight has been updated to:{os.path.basename(save_path)}")

    
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
        f"\n✅ [{test_slice} | Seed {seed}] Test average Dice:{np.mean(test_dice_list):.4f}| Time taken:{timedelta(seconds=int(fold_duration))}")
    return fold_duration


if __name__ == "__main__":
    PATCH_DIR = r"data/STHBC/patches(xy)"
    H5AD_BASE_DIR = r"data/STHBC/ST"
    SLICES = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    
    SEEDS = [567]

    total_start = time.time()
    all_durations = []

    
    for seed in SEEDS:
        print(f"\n" + "🌟" * 25)
        print(f"🚀 Start a new experimental round | Random seed:{seed}")
        print("🌟" * 25)

        for ts in SLICES:
            duration = train_one_fold(ts, SLICES, PATCH_DIR, H5AD_BASE_DIR, seed=seed, epochs=10, batch_size=8)
            all_durations.append(duration)

    print("\n" + 'Image single-modality Baseline statistical report'.center(50, "="))
    print(f"Total number of training Folds:{len(SLICES) * len(SEEDS)}(slice:{len(SLICES)}x seed:{len(SEEDS)})")
    print(f"Total time taken for the whole process:{timedelta(seconds=int(time.time() - total_start))}")
    print(f"Average time spent per Fold:{timedelta(seconds=int(np.mean(all_durations)))}")
    print("=" * 50)
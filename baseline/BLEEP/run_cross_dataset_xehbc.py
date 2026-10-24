# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
# Spot-level protocol:
# Generate spot predictions from the method-native, original-scale H&E/ST data
# (or features prepared from those data). Do not use resized_data/cropped_data
# as model inputs. Those aligned files are used only for pixel-level evaluation.
# Preserve the original AnnData observation order in every prediction file.
###============================ HER2ST to XeHBC =========================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import random
import time  
from datetime import datetime
import pandas as pd


from models import CLIPModelWithClassifier
from modules import ProjectionHead


def set_all_seeds(seed=42):
    'Implementation of set_all_seeds for this baseline experiment.'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def count_parameters(model):
    'Implementation of count_parameters for this baseline experiment.'
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_peak_gpu_memory():
    'Implementation of get_peak_gpu_memory for this baseline experiment.'
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


class SimpleSpatialDataset(Dataset):
    'SimpleSpatialDataset used by this baseline experiment.'

    def __init__(self, img_feats, gene_exprs, labels, sample_name=""):
        self.img_feats = img_feats
        self.gene_exprs = gene_exprs
        self.labels = labels
        self.sample_name = sample_name

        
        assert len(img_feats) == len(gene_exprs) == len(labels), 'Number of features and labels do not match'

    def __len__(self):
        return len(self.img_feats)

    def __getitem__(self, idx):
        return {
            "image_features": torch.tensor(self.img_feats[idx], dtype=torch.float32),
            "spot_features": torch.tensor(self.gene_exprs[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def load_training_slices(npz_path="../dataset/train_dataset_XeniumHBC(new).npz",
                         label_dir="../data/HB/her2st/label3",
                         all_samples=['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']):
    'Implementation of load_training_slices for this baseline experiment.'
    data = np.load(npz_path, allow_pickle=True)
    slices = []

    for i, sample in enumerate(all_samples):
        img_feats = data['img_resnet'][i]
        gene_exprs = data['gene_expr'][i]

        
        labels_path = f"{label_dir}/{sample}-labels3.txt"
        labels = np.loadtxt(labels_path, dtype=int)

        
        valid_indices = labels != -1
        img_feats = img_feats[valid_indices]
        gene_exprs = gene_exprs[valid_indices]
        labels = labels[valid_indices]

        
        labels = np.where(labels == 2, 1, 0)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        
        dataset = SimpleSpatialDataset(
            img_feats=img_feats,
            gene_exprs=gene_exprs,
            labels=labels,
            sample_name=sample
        )

        slices.append({
            'sample': sample,
            'dataset': dataset,
            'idx': i,
            'img_dim': img_feats.shape[1],
            'gene_dim': gene_exprs.shape[1]
        })

    return slices, all_samples


def load_test_dataset(npz_path="../dataset/test_dataset_XeniumHBC(new).npz",
                      label_path="../data/HB/Xenium_HBC/spot level/label.txt"):
    'Implementation of load_test_dataset for this baseline experiment.'
    test_data = np.load(npz_path, allow_pickle=True)
    test_img = test_data['img_resnet']
    test_expr = test_data['gene_expr']
    test_label = np.loadtxt(label_path, dtype=int)

    if test_label.ndim == 1:
        test_label = test_label.reshape(-1, 1)

    
    test_dataset = SimpleSpatialDataset(
        img_feats=test_img,
        gene_exprs=test_expr,
        labels=test_label,
        sample_name="test"
    )

    return test_dataset


def simple_collate_fn(batch):
    'Implementation of simple_collate_fn for this baseline experiment.'
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = torch.stack([item[key] for item in batch])
    return batch_dict


def train_clip_model(model, train_loader, val_loader, device, epochs=30, lr=1e-4):
    'Implementation of train_clip_model for this baseline experiment.'
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels'].squeeze(1)  

            optimizer.zero_grad()

            
            outputs = model({
                "image": batch["image_features"],
                "reduced_expression": batch["spot_features"],
                "labels": labels
            })

            loss = outputs["total_loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch["image_features"].size(0)
            batch_count += batch["image_features"].size(0)

        
        avg_train_loss = total_loss / batch_count

        
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch['labels'].cpu().numpy()

                outputs = model({
                    "image": batch["image_features"],
                    "reduced_expression": batch["spot_features"],
                    "labels": batch["labels"].squeeze(1)
                })

                
                preds = torch.sigmoid(outputs["class_logits"]).cpu().numpy()
                val_preds.append(preds)
                val_labels.append(labels)

        val_preds = np.concatenate(val_preds).flatten()
        val_labels = np.concatenate(val_labels).flatten()

        if len(np.unique(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_preds)
            val_ap = average_precision_score(val_labels, val_preds)
        else:
            val_auc = 0.5
            val_ap = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "temp_best_clip_XeniumHBC.pth")
            

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"train_loss: {avg_train_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

    
    model.load_state_dict(torch.load("temp_best_clip_XeniumHBC.pth"))
    return model


def evaluate_clip_model(model, seed, test_loader, device):
    'Implementation of evaluate_clip_model for this baseline experiment.'
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels'].cpu().numpy()

            outputs = model({
                "image": batch["image_features"],
                "reduced_expression": batch["spot_features"],
                "labels": batch["labels"].squeeze(1)
            })

            probs = torch.sigmoid(outputs["class_logits"]).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(labels)

    y_pred = np.concatenate(all_preds).flatten()
    y_probs = np.concatenate(all_probs).flatten()
    y_true = np.concatenate(all_labels).flatten()

    
    auc = roc_auc_score(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)

    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    
    file_name = f"../../predict_files/BLEEP/Xenium_HBC/From_HER2ST/predict_probs_{seed}.txt"
    directory = os.path.dirname(file_name)
    os.makedirs(directory, exist_ok=True)
    np.savetxt(file_name, y_probs, fmt='%s')

    return {
        'auc': auc,
        'ap': ap,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs
    }


def run_clip_experiment(seed, batch_size, epochs, projection_dim=256, temperature=0.07):
    'Implementation of run_clip_experiment for this baseline experiment.'
    set_all_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'!' * 20}CLIP model seed={seed}Experiment begins{'!' * 20}")
    print(f"Equipment used:{device}")

    
    train_npz = "../../dataset/train_dataset_XeniumHBC(new).npz"
    test_npz = "../../dataset/test_dataset_XeniumHBC(new).npz"
    train_label_dir = "../../data/HB/her2st/label3"
    test_label_path = "../../data/HB/Xenium_HBC/spot level/label.txt"

    
    print('Load training slice data...')
    train_slices, all_samples = load_training_slices(
        npz_path=train_npz,
        label_dir=train_label_dir
    )

    
    all_train_data = [slice_data['dataset'] for slice_data in train_slices]
    full_train_dataset = ConcatDataset(all_train_data)

    
    total_train_spots = sum(len(dataset) for dataset in all_train_data)
    print(f"The training set contains a total of{total_train_spots}spots")

    
    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_set, val_set = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    
    print('Load test set data...')
    test_dataset = load_test_dataset(
        npz_path=test_npz,
        label_path=test_label_path
    )
    print(f"The test set contains{len(test_dataset)}spots")

    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=simple_collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=simple_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=simple_collate_fn
    )

    
    img_dim = train_slices[0]['img_dim']
    gene_dim = train_slices[0]['gene_dim']

    
    model = CLIPModelWithClassifier(
        temperature=temperature,
        image_embedding=img_dim,
        spot_embedding=gene_dim,
        projection_dim=projection_dim
    ).to(device)

    # --- METRIC 1: Parameters ---
    num_params = count_parameters(model)
    print(f"Model parameters: image_dim={img_dim}, gene_dim={gene_dim}, projection_dim={projection_dim}")
    print(f"[METRICS] Total Trainable Parameters: {num_params:,}")

    # --- METRIC 2 & 4: Training Time & Memory ---
    print('Start training the CLIP model...')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_train = time.time()

    model = train_clip_model(model, train_loader, val_loader, device, epochs=epochs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_train = time.time()
    train_time = end_train - start_train
    peak_train_mem = get_peak_gpu_memory()

    print(f"[METRICS] Training Time: {train_time:.4f} seconds")
    print(f"[METRICS] Peak Training GPU Memory: {peak_train_mem:.2f} MB")

    # --- METRIC 3 & 4: Inference Time & Memory ---
    print('Evaluating the CLIP model on the test set...')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_infer = time.time()

    test_results = evaluate_clip_model(model, seed, test_loader, device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_infer = time.time()
    infer_time = end_infer - start_infer
    peak_infer_mem = get_peak_gpu_memory()

    print(f"[METRICS] Inference Time (Total): {infer_time:.4f} seconds")
    print(f"[METRICS] Peak Inference GPU Memory: {peak_infer_mem:.2f} MB")

    # Store resource metrics
    test_results['params'] = num_params
    test_results['train_time'] = train_time
    test_results['infer_time'] = infer_time
    test_results['train_mem'] = peak_train_mem
    test_results['infer_mem'] = peak_infer_mem

    
    print("\n" + "=" * 60)
    print(f"CLIP model seed{seed}Test results:")
    print("=" * 60)
    print(f"AUC: {test_results['auc']:.4f}")
    print(f"AP:  {test_results['ap']:.4f}")
    print(f"F1:  {test_results['f1']:.4f}")
    print(f"Accuracy:{test_results['accuracy']:.4f}")
    print(f"Sensitivity:{test_results['sensitivity']:.4f}")
    print(f"Specificity:{test_results['specificity']:.4f}")

    return test_results


def main():
    'Implementation of main for this baseline experiment.'
    
    seeds = [42]#[24, 42, 123, 345, 567]
    batch_size = 32  
    epochs = 100
    projection_dim = 256
    temperature = 0.07

    print(f"\n{'=' * 60}")
    print(f"{'CLIP multimodal model training (XeHBC)':^60}")
    print(f"{'=' * 60}\n")
    print(f"Experimental configuration:")
    print(f"- Random seed:{seeds}")
    print(f"- Batch size:{batch_size}")
    print(f"- Number of training rounds:{epochs}")
    print(f"- Projection dimensions:{projection_dim}")
    print(f"- Temperature parameters:{temperature}")

    
    all_results = []
    for seed in seeds:
        results = run_clip_experiment(seed, batch_size, epochs, projection_dim, temperature)
        all_results.append(results)

    
    aucs = [r['auc'] for r in all_results]
    aps = [r['ap'] for r in all_results]
    f1s = [r['f1'] for r in all_results]
    sensitivities = [r['sensitivity'] for r in all_results]
    specificities = [r['specificity'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]

    # Resource metrics
    params_mean = np.mean([r['params'] for r in all_results])
    train_time_mean = np.mean([r['train_time'] for r in all_results])
    infer_time_mean = np.mean([r['infer_time'] for r in all_results])
    train_mem_mean = np.mean([r['train_mem'] for r in all_results])
    infer_mem_mean = np.mean([r['infer_mem'] for r in all_results])

    
    print("\n" + "=" * 60)
    print('Detailed results for each random seed:')
    print("=" * 60)
    for i, seed in enumerate(seeds):
        print(f"seed{seed:4d}: "
              f"AUC={aucs[i]:.4f}, "
              f"AP={aps[i]:.4f}, "
              f"F1={f1s[i]:.4f}, "
              f"Sensitivity={sensitivities[i]:.4f}, "
              f"Specificity ={specificities[i]:.4f}, "
              f"Accuracy ={accuracies[i]:.4f}")

    
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)
    ap_mean, ap_std = np.mean(aps), np.std(aps)
    f1_mean, f1_std = np.mean(f1s), np.std(f1s)
    sens_mean, sens_std = np.mean(sensitivities), np.std(sensitivities)
    spec_mean, spec_std = np.mean(specificities), np.std(specificities)
    acc_mean, acc_std = np.mean(accuracies), np.std(accuracies)

    print("\n" + "=" * 80)
    print(f"{'CLIP model final performance and resource consumption (mean ± standard deviation)':^80}")
    print("=" * 80)

    
    print(f"{'Resource Metrics':<30} | {'Value (Mean)':<20}")
    print("-" * 80)
    print(f"{'Parameters':<30} | {int(params_mean):,}")
    print(f"{'Training Time (s)':<30} | {train_time_mean:.4f}")
    print(f"{'Inference Time (s)':<30} | {infer_time_mean:.4f}")
    print(f"{'Train Memory (MB)':<30} | {train_mem_mean:.2f}")
    print(f"{'Infer Memory (MB)':<30} | {infer_mem_mean:.2f}")

    print("-" * 80)
    print(f"{'Performance Metrics':<30} | {'Mean ± Std':<20}")
    print("-" * 80)
    print(f"{'AUC':<30} | {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"{'AP':<30} | {ap_mean:.4f} ± {ap_std:.4f}")
    print(f"{'F1':<30} | {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"{'Sensitivity':<30} | {sens_mean:.4f} ± {sens_std:.4f}")
    print(f"{'Specificity':<30} | {spec_mean:.4f} ± {spec_std:.4f}")
    print(f"{'Accuracy':<30} | {acc_mean:.4f} ± {acc_std:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

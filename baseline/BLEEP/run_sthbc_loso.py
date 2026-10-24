# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
# Spot-level protocol:
# Generate spot predictions from the method-native, original-scale H&E/ST data
# (or features prepared from those data). Do not use resized_data/cropped_data
# as model inputs. Those aligned files are used only for pixel-level evaluation.
# Preserve the original AnnData observation order in every prediction file.
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import config as CFG
from modules import ProjectionHead
from models import CLIPModelWithClassifier  # Configure this path for your environment; see readme_baseline.txt.

# Configure this path for your environment; see readme_baseline.txt.
os.makedirs("../../predict_files/BLEEP/HER2ST", exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



class HER2STDataset(Dataset):
    def __init__(self, img_feats, gene_exprs, labels):
        self.img_feats = img_feats  
        self.gene_exprs = gene_exprs  
        self.labels = labels  

    def __len__(self):
        return len(self.img_feats)

    def __getitem__(self, idx):
        return {
            "image_features": torch.tensor(self.img_feats[idx], dtype=torch.float32),
            "spot_features": torch.tensor(self.gene_exprs[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }



def load_all_data(npz_path, label_dir, all_samples):
    data = np.load(npz_path, allow_pickle=True)
    dataset_dict = {}

    for i, sample in enumerate(all_samples):
        
        img_feats = data['img_resnet'][i]  
        gene_exprs = data['gene_expr'][i]  
        coords = data['coords'][i]  

        
        labels_path = f"{label_dir}/{sample}_labels.txt"
        labels = np.loadtxt(labels_path, dtype=int)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        
        assert len(img_feats) == len(gene_exprs) == len(labels),\
            f"sample{sample}The number of features and labels does not match"

        dataset_dict[sample] = {
            "img_feats": img_feats,
            "gene_exprs": gene_exprs,
            "coords": coords,
            "labels": labels
        }
        print(f"Load slices{sample}: {len(img_feats)}spots")

    return dataset_dict



def train_model(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        
        batch = {
            "image": batch["image_features"].to(device),  
            "reduced_expression": batch["spot_features"].to(device),
            "labels": batch["labels"].to(device).squeeze(1)  
        }

        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["total_loss"]

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["image"].size(0)

    return total_loss / len(train_loader.dataset)



def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {
                "image": batch["image_features"].to(device),
                "reduced_expression": batch["spot_features"].to(device),
                "labels": batch["labels"].to(device).squeeze(1)
            }

            outputs = model(batch)
            
            preds = torch.sigmoid(outputs["class_logits"]).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())

    
    preds_binary = (np.array(all_preds) > 0.5).astype(int)  
    acc = accuracy_score(all_labels, preds_binary)
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)  
    f1 = f1_score(all_labels, preds_binary)

    return acc, auc, ap, f1, np.array(all_preds)



def leave_one_out_cv(dataset_dict, all_samples, seed):
    
    metrics = {
        "accuracy": {},
        "auc": {},
        "ap": {},
        "f1": {},
        "predictions": {}  
    }

    for sample in all_samples:
        metrics["accuracy"][sample] = []
        metrics["auc"][sample] = []
        metrics["ap"][sample] = []
        metrics["f1"][sample] = []

    for test_sample in all_samples:
        print(f"\n====== Leave-one-out cross-validation: test slices{test_sample} (seed={seed}) =====")

        
        train_samples = [s for s in all_samples if s != test_sample]

        
        train_img = np.concatenate([dataset_dict[s]["img_feats"] for s in train_samples], axis=0)
        train_gene = np.concatenate([dataset_dict[s]["gene_exprs"] for s in train_samples], axis=0)
        train_labels = np.concatenate([dataset_dict[s]["labels"] for s in train_samples], axis=0)

        
        test_img = dataset_dict[test_sample]["img_feats"]
        test_gene = dataset_dict[test_sample]["gene_exprs"]
        test_labels = dataset_dict[test_sample]["labels"]

        
        train_dataset = HER2STDataset(train_img, train_gene, train_labels)
        test_dataset = HER2STDataset(test_img, test_gene, test_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers
        )

        
        model = CLIPModelWithClassifier(
            temperature=CFG.temperature,
            image_embedding=train_img.shape[1],  
            spot_embedding=train_gene.shape[1],  
            projection_dim=CFG.projection_dim,
            #classification_weight=CFG.cl_weight
        ).to(device)

        
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

        
        best_auc = 0.0
        for epoch in range(CFG.epochs):
            train_loss = train_model(model, train_loader, optimizer, epoch)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

        
        acc, auc, ap, f1, preds = evaluate_model(model, test_loader)
        print(f"Test results - Accuracy:{acc:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")

        
        output_dir="../../predict_files/BLEEP/HER2ST/"
        os.makedirs(output_dir, exist_ok=True)
        pred_path = f"../../predict_files/BLEEP/HER2ST/probs_{test_sample}_seed={seed}.txt"
        np.savetxt(pred_path, preds)

        
        metrics["accuracy"][test_sample].append(acc)
        metrics["auc"][test_sample].append(auc)
        metrics["ap"][test_sample].append(ap)
        metrics["f1"][test_sample].append(f1)
        metrics["predictions"][test_sample] = preds

    return metrics



if __name__ == "__main__":
    
    CFG.batch_size = 512
    CFG.epochs = 15
    CFG.lr = 1e-4
    CFG.weight_decay = 1e-5
    CFG.projection_dim = 256
    CFG.temperature = 0.07
    #CFG.cl_weight = 1.0
    CFG.log_interval = 5
    CFG.num_workers = 4

    # Configure this path for your environment; see readme_baseline.txt.
    npz_path = "../../dataset/HER2ST_dataset(new).npz"
    label_dir = "../../data/HB/her2st/label"
    all_samples = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
    seeds = [24, 42, 123, 345, 567]  

    
    dataset_dict = load_all_data(npz_path, label_dir, all_samples)

    
    all_metrics = {
        "accuracy": {s: [] for s in all_samples},
        "auc": {s: [] for s in all_samples},
        "ap": {s: [] for s in all_samples},
        "f1": {s: [] for s in all_samples}
    }

    
    for seed in seeds:
        print(f"\n\n======================================")
        print(f"========== Random seed:{seed}Start ==========")
        print(f"======================================")

        
        set_seed(seed)

        
        metrics = leave_one_out_cv(dataset_dict, all_samples, seed)

        
        for sample in all_samples:
            all_metrics["accuracy"][sample].extend(metrics["accuracy"][sample])
            all_metrics["auc"][sample].extend(metrics["auc"][sample])
            all_metrics["ap"][sample].extend(metrics["ap"][sample])
            all_metrics["f1"][sample].extend(metrics["f1"][sample])

        
        print(f"\n====== Random seed{seed}Test results of each slice =====")
        for sample in all_samples:
            print(
                f"{sample}: AUC={metrics['auc'][sample][0]:.4f}, AP={metrics['ap'][sample][0]:.4f}, F1={metrics['f1'][sample][0]:.4f}")

    
    print("\n" + "=" * 60)
    print('Final result (Mean ± Std per Sample):')
    print("=" * 60)

    
    overall_metrics = {
        "auc": [],
        "ap": [],
        "f1": []
    }

    for sample in all_samples:
        auc_mean = np.mean(all_metrics["auc"][sample])
        auc_std = np.std(all_metrics["auc"][sample])
        ap_mean = np.mean(all_metrics["ap"][sample])
        ap_std = np.std(all_metrics["ap"][sample])
        f1_mean = np.mean(all_metrics["f1"][sample])
        f1_std = np.std(all_metrics["f1"][sample])

        print(
            f"{sample}: AUC={auc_mean:.4f}±{auc_std:.4f}, AP={ap_mean:.4f}±{ap_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}")

        
        overall_metrics["auc"].extend(all_metrics["auc"][sample])
        overall_metrics["ap"].extend(all_metrics["ap"][sample])
        overall_metrics["f1"].extend(all_metrics["f1"][sample])

    
    overall_auc_mean = np.mean(overall_metrics["auc"])
    overall_auc_std = np.std(overall_metrics["auc"])
    overall_ap_mean = np.mean(overall_metrics["ap"])
    overall_ap_std = np.std(overall_metrics["ap"])
    overall_f1_mean = np.mean(overall_metrics["f1"])
    overall_f1_std = np.std(overall_metrics["f1"])

    print("\n" + "=" * 60)
    print('Overall results:')
    print("=" * 60)
    print(f"AUC: {overall_auc_mean:.4f} ± {overall_auc_std:.4f}")
    print(f"AP:  {overall_ap_mean:.4f} ± {overall_ap_std:.4f}")
    print(f"F1:  {overall_f1_mean:.4f} ± {overall_f1_std:.4f}")

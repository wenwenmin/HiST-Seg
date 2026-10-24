import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import config as CFG
from modules import  ProjectionHead
from models import CLIPModelWithClassifier  # Configure this path for your environment; see readme_baseline.txt.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    f1 = f1_score(all_labels, preds_binary)

    return acc, auc, f1



def leave_one_out_cv(dataset_dict, all_samples):
    
    metrics = {
        "accuracy": [],
        "auc": [],
        "f1": []
    }

    for test_sample in all_samples:
        print(f"\n====== Leave-one-out cross-validation: test slices{test_sample} =====")

        
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
            classification_weight=CFG.classification_weight
        ).to(device)

        
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

        
        best_auc = 0.0
        for epoch in range(CFG.epochs):
            train_loss = train_model(model, train_loader, optimizer, epoch)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
            # if (epoch + 1) % CFG.log_interval == 0:
            #     print(f"Epoch [{epoch + 1}/{CFG.epochs}], Train Loss: {train_loss:.4f}")

        
        acc, auc, f1 = evaluate_model(model, test_loader)
        print(f"Test results - Accuracy:{acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

        
        metrics["accuracy"].append(acc)
        metrics["auc"].append(auc)
        metrics["f1"].append(f1)

    
    print('===== Leave-one-out cross-validation final result =====')
    print(f"Average accuracy:{np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
    print(f"Average AUC:{np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}")
    print(f"Average F1:{np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")

    return metrics



if __name__ == "__main__":
    
    CFG.batch_size = 128
    CFG.epochs = 2
    CFG.lr = 1e-4
    CFG.weight_decay = 1e-5
    CFG.projection_dim = 256
    CFG.temperature = 0.07
    CFG.classification_weight = 1.0
    CFG.log_interval = 10
    CFG.num_workers = 4

    # Configure this path for your environment; see readme_baseline.txt.
    npz_path = "../../dataset/HER2ST_dataset(new).npz"
    label_dir = "../../data/HB/her2st/label"
    all_samples = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    
    dataset_dict = load_all_data(npz_path, label_dir, all_samples)

    
    leave_one_out_cv(dataset_dict, all_samples)
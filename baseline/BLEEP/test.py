import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import config as CFG
from modules import  ProjectionHead
from models import CLIPModelWithClassifier  # 替换为模型定义路径

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据加载类
class HER2STDataset(Dataset):
    def __init__(self, img_feats, gene_exprs, labels):
        self.img_feats = img_feats  # 已提取的图像特征 (n_spots, img_embedding_dim)
        self.gene_exprs = gene_exprs  # 基因特征 (n_spots, spot_embedding_dim)
        self.labels = labels  # 标签 (n_spots, 1)

    def __len__(self):
        return len(self.img_feats)

    def __getitem__(self, idx):
        return {
            "image_features": torch.tensor(self.img_feats[idx], dtype=torch.float32),
            "spot_features": torch.tensor(self.gene_exprs[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# 加载所有数据
def load_all_data(npz_path, label_dir, all_samples):
    data = np.load(npz_path, allow_pickle=True)
    dataset_dict = {}

    for i, sample in enumerate(all_samples):
        # 加载特征和坐标（坐标暂未使用，如需使用可在后续扩展）
        img_feats = data['img_resnet'][i]  # 已提取的图像特征
        gene_exprs = data['gene_expr'][i]  # 基因特征
        coords = data['coords'][i]  # 坐标信息（备用）

        # 加载标签
        labels_path = f"{label_dir}/{sample}_labels.txt"
        labels = np.loadtxt(labels_path, dtype=int)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        # 检查特征和标签数量是否匹配
        assert len(img_feats) == len(gene_exprs) == len(labels), \
            f"样本{sample}的特征和标签数量不匹配"

        dataset_dict[sample] = {
            "img_feats": img_feats,
            "gene_exprs": gene_exprs,
            "coords": coords,
            "labels": labels
        }
        print(f"加载切片{sample}: {len(img_feats)}个spots")

    return dataset_dict


# 训练函数
def train_model(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        # 数据移至设备
        batch = {
            "image": batch["image_features"].to(device),  # 直接使用预提取的图像特征
            "reduced_expression": batch["spot_features"].to(device),
            "labels": batch["labels"].to(device).squeeze(1)  # BCE损失需要1D标签
        }

        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["total_loss"]

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["image"].size(0)

    return total_loss / len(train_loader.dataset)


# 评估函数
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
            # 计算预测概率（sigmoid输出）
            preds = torch.sigmoid(outputs["class_logits"]).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())

    # 计算评估指标
    preds_binary = (np.array(all_preds) > 0.5).astype(int)  # 二值化预测
    acc = accuracy_score(all_labels, preds_binary)
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, preds_binary)

    return acc, auc, f1


# 留一交叉验证主函数
def leave_one_out_cv(dataset_dict, all_samples):
    # 存储每个fold的评估结果
    metrics = {
        "accuracy": [],
        "auc": [],
        "f1": []
    }

    for test_sample in all_samples:
        print(f"\n===== 留一交叉验证: 测试切片 {test_sample} =====")

        # 划分训练集和测试集
        train_samples = [s for s in all_samples if s != test_sample]

        # 合并训练集数据
        train_img = np.concatenate([dataset_dict[s]["img_feats"] for s in train_samples], axis=0)
        train_gene = np.concatenate([dataset_dict[s]["gene_exprs"] for s in train_samples], axis=0)
        train_labels = np.concatenate([dataset_dict[s]["labels"] for s in train_samples], axis=0)

        # 测试集数据
        test_img = dataset_dict[test_sample]["img_feats"]
        test_gene = dataset_dict[test_sample]["gene_exprs"]
        test_labels = dataset_dict[test_sample]["labels"]

        # 创建数据集和数据加载器
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

        # 初始化模型（注意：使用预提取的图像特征，无需图像编码器）
        model = CLIPModelWithClassifier(
            temperature=CFG.temperature,
            image_embedding=train_img.shape[1],  # 图像特征维度（从数据中获取）
            spot_embedding=train_gene.shape[1],  # 基因特征维度
            projection_dim=CFG.projection_dim,
            classification_weight=CFG.classification_weight
        ).to(device)

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

        # 训练模型
        best_auc = 0.0
        for epoch in range(CFG.epochs):
            train_loss = train_model(model, train_loader, optimizer, epoch)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
            # if (epoch + 1) % CFG.log_interval == 0:
            #     print(f"Epoch [{epoch + 1}/{CFG.epochs}], Train Loss: {train_loss:.4f}")

        # 在测试集上评估
        acc, auc, f1 = evaluate_model(model, test_loader)
        print(f"测试结果 - 准确率: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")

        # 保存指标
        metrics["accuracy"].append(acc)
        metrics["auc"].append(auc)
        metrics["f1"].append(f1)

    # 计算平均指标
    print("\n===== 留一交叉验证最终结果 =====")
    print(f"平均准确率: {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
    print(f"平均AUC: {np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}")
    print(f"平均F1: {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")

    return metrics


# 主函数
if __name__ == "__main__":
    # 配置参数（可在config.py中定义）
    CFG.batch_size = 128
    CFG.epochs = 2
    CFG.lr = 1e-4
    CFG.weight_decay = 1e-5
    CFG.projection_dim = 256
    CFG.temperature = 0.07
    CFG.classification_weight = 1.0
    CFG.log_interval = 10
    CFG.num_workers = 4

    # 数据路径和样本列表
    npz_path = "../../dataset/HER2ST_dataset(new).npz"
    label_dir = "../../data/HB/her2st/label"
    all_samples = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    # 加载数据
    dataset_dict = load_all_data(npz_path, label_dir, all_samples)

    # 执行留一交叉验证
    leave_one_out_cv(dataset_dict, all_samples)
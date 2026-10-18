# ###============================ HER2ST to XeniumHBC =====================================
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
# import os
# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
# import random
# from datetime import datetime
# import pandas as pd
#
# # 导入CLIP模型相关组件
# from models import CLIPModelWithClassifier
# from modules import ProjectionHead
#
#
# def set_all_seeds(seed=42):
#     """设置随机种子"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#
# class SimpleSpatialDataset(Dataset):
#     """
#     简化的空间数据集类，适配CLIP模型输入
#     类似于文档1的数据结构，但适配文档2的模型输入格式
#     """
#
#     def __init__(self, img_feats, gene_exprs, labels, sample_name=""):
#         self.img_feats = img_feats
#         self.gene_exprs = gene_exprs
#         self.labels = labels
#         self.sample_name = sample_name
#
#         # 验证数据形状
#         assert len(img_feats) == len(gene_exprs) == len(labels), "特征和标签数量不匹配"
#
#     def __len__(self):
#         return len(self.img_feats)
#
#     def __getitem__(self, idx):
#         return {
#             "image_features": torch.tensor(self.img_feats[idx], dtype=torch.float32),
#             "spot_features": torch.tensor(self.gene_exprs[idx], dtype=torch.float32),
#             "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
#         }
#
#
# def load_training_slices(npz_path=" ",
#                          label_dir=" ",
#                          all_samples=['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']):
#     """加载训练切片数据 - 适配CLIP模型"""
#     data = np.load(npz_path, allow_pickle=True)
#     slices = []
#
#     for i, sample in enumerate(all_samples):
#         img_feats = data['img_resnet'][i]
#         gene_exprs = data['gene_expr'][i]
#
#         # 加载标签
#         labels_path = f"{label_dir}/{sample}-labels3.txt"
#         labels = np.loadtxt(labels_path, dtype=int)
#
#         # 应用训练集标签处理：过滤-1
#         valid_indices = labels != -1
#         img_feats = img_feats[valid_indices]
#         gene_exprs = gene_exprs[valid_indices]
#         labels = labels[valid_indices]
#
#         # 训练集标签转换：0/1 -> 0 (正常区域), 2 -> 1 (癌症区域)
#         labels = np.where(labels == 2, 1, 0)
#         if labels.ndim == 1:
#             labels = labels.reshape(-1, 1)
#
#         # 创建简化数据集实例
#         dataset = SimpleSpatialDataset(
#             img_feats=img_feats,
#             gene_exprs=gene_exprs,
#             labels=labels,
#             sample_name=sample
#         )
#
#         slices.append({
#             'sample': sample,
#             'dataset': dataset,
#             'idx': i,
#             'img_dim': img_feats.shape[1],
#             'gene_dim': gene_exprs.shape[1]
#         })
#
#     return slices, all_samples
#
#
# def load_test_dataset(npz_path=" ",
#                       label_path=" "):
#     """加载测试集数据 - 适配CLIP模型"""
#     test_data = np.load(npz_path, allow_pickle=True)
#     test_img = test_data['img_resnet']
#     test_expr = test_data['gene_expr']
#     test_label = np.loadtxt(label_path, dtype=int)
#
#     if test_label.ndim == 1:
#         test_label = test_label.reshape(-1, 1)
#
#     # 测试集标签处理：0->0 (正常), 1->1 (癌症)
#     test_dataset = SimpleSpatialDataset(
#         img_feats=test_img,
#         gene_exprs=test_expr,
#         labels=test_label,
#         sample_name="test"
#     )
#
#     return test_dataset
#
#
# def simple_collate_fn(batch):
#     """简单的批处理函数"""
#     batch_dict = {}
#     for key in batch[0].keys():
#         batch_dict[key] = torch.stack([item[key] for item in batch])
#     return batch_dict
#
#
# def train_clip_model(model, train_loader, val_loader, device, epochs=30, lr=1e-4):
#     """训练CLIP模型"""
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#     best_val_auc = 0.0
#
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
#         batch_count = 0
#
#         for batch in train_loader:
#             # 移动到设备
#             batch = {k: v.to(device) for k, v in batch.items()}
#             labels = batch['labels'].squeeze(1)  # BCE损失需要1D标签
#
#             optimizer.zero_grad()
#
#             # 前向传播 - CLIP模型返回字典
#             outputs = model({
#                 "image": batch["image_features"],
#                 "reduced_expression": batch["spot_features"],
#                 "labels": labels
#             })
#
#             loss = outputs["total_loss"]
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item() * batch["image_features"].size(0)
#             batch_count += batch["image_features"].size(0)
#
#         # 计算平均训练损失
#         avg_train_loss = total_loss / batch_count
#
#         # 验证循环
#         model.eval()
#         val_preds = []
#         val_labels = []
#         with torch.no_grad():
#             for batch in val_loader:
#                 batch = {k: v.to(device) for k, v in batch.items()}
#                 labels = batch['labels'].cpu().numpy()
#
#                 outputs = model({
#                     "image": batch["image_features"],
#                     "reduced_expression": batch["spot_features"],
#                     "labels": batch["labels"].squeeze(1)
#                 })
#
#                 # 使用sigmoid得到概率
#                 preds = torch.sigmoid(outputs["class_logits"]).cpu().numpy()
#                 val_preds.append(preds)
#                 val_labels.append(labels)
#
#         val_preds = np.concatenate(val_preds).flatten()
#         val_labels = np.concatenate(val_labels).flatten()
#
#         if len(np.unique(val_labels)) > 1:
#             val_auc = roc_auc_score(val_labels, val_preds)
#             val_ap = average_precision_score(val_labels, val_preds)
#         else:
#             val_auc = 0.5
#             val_ap = 0.5
#
#         if val_auc > best_val_auc:
#             best_val_auc = val_auc
#             torch.save(model.state_dict(), "temp_best_clip_XeniumHBC.pth")
#             print(f"保存最佳模型 (Val AUC={val_auc:.4f})")
#
#         print(f"Epoch {epoch + 1}/{epochs} | "
#               f"train_loss: {avg_train_loss:.4f} | "
#               f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")
#
#     # 加载最佳模型
#     model.load_state_dict(torch.load("temp_best_clip_XeniumHBC.pth"))
#     return model
#
#
# def evaluate_clip_model(model, seed, test_loader, device):
#     """评估CLIP模型"""
#     model.eval()
#     all_preds = []
#     all_labels = []
#     all_probs = []
#
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             labels = batch['labels'].cpu().numpy()
#
#             outputs = model({
#                 "image": batch["image_features"],
#                 "reduced_expression": batch["spot_features"],
#                 "labels": batch["labels"].squeeze(1)
#             })
#
#             probs = torch.sigmoid(outputs["class_logits"]).cpu().numpy()
#             preds = (probs > 0.5).astype(int)
#
#             all_preds.append(preds)
#             all_probs.append(probs)
#             all_labels.append(labels)
#
#     y_pred = np.concatenate(all_preds).flatten()
#     y_probs = np.concatenate(all_probs).flatten()
#     y_true = np.concatenate(all_labels).flatten()
#
#     # 计算评估指标
#     auc = roc_auc_score(y_true, y_probs)
#     ap = average_precision_score(y_true, y_probs)
#     f1 = f1_score(y_true, y_pred)
#
#     # 计算混淆矩阵
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#     accuracy = (tp + tn) / (tp + tn + fp + fn)
#
#     # 保存预测概率
#     file_name = f"../../predict_files/BLEEP/Xenium_HBC/From_HER2ST/predict_probs_{seed}.txt"
#     directory = os.path.dirname(file_name)
#     os.makedirs(directory, exist_ok=True)
#     np.savetxt(file_name, y_probs, fmt='%s')
#
#     return {
#         'auc': auc,
#         'ap': ap,
#         'f1': f1,
#         'sensitivity': sensitivity,
#         'specificity': specificity,
#         'accuracy': accuracy,
#         'y_true': y_true,
#         'y_pred': y_pred,
#         'y_probs': y_probs
#     }
#
#
# def run_clip_experiment(seed, batch_size, epochs, projection_dim=256, temperature=0.07):
#     """运行单个CLIP模型实验"""
#     set_all_seeds(seed)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"\n{'!' * 20} CLIP模型 种子={seed} 实验开始 {'!' * 20}")
#     print(f"使用设备: {device}")
#
#     # 配置参数
#     train_npz = "../../dataset/train_dataset_XeniumHBC(new).npz"
#     test_npz = "../../dataset/test_dataset_XeniumHBC(new).npz"
#     train_label_dir = "../../data/HB/her2st/label3"
#     test_label_path = "../../data/HB/Xenium_HBC/spot level/label.txt"
#
#     # 1. 加载训练数据切片
#     print("加载训练切片数据...")
#     train_slices, all_samples = load_training_slices(
#         npz_path=train_npz,
#         label_dir=train_label_dir
#     )
#
#     # 合并所有训练切片
#     all_train_data = [slice_data['dataset'] for slice_data in train_slices]
#     full_train_dataset = ConcatDataset(all_train_data)
#
#     # 打印训练数据统计
#     total_train_spots = sum(len(dataset) for dataset in all_train_data)
#     print(f"训练集总共包含 {total_train_spots} 个spots")
#
#     # 划分训练集和验证集 (80%/20%)
#     val_size = int(0.2 * len(full_train_dataset))
#     train_size = len(full_train_dataset) - val_size
#     train_set, val_set = random_split(
#         full_train_dataset,
#         [train_size, val_size],
#         generator=torch.Generator().manual_seed(seed)
#     )
#
#     # 2. 加载测试集
#     print("加载测试集数据...")
#     test_dataset = load_test_dataset(
#         npz_path=test_npz,
#         label_path=test_label_path
#     )
#     print(f"测试集包含 {len(test_dataset)} 个spots")
#
#     # 创建数据加载器
#     train_loader = DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=simple_collate_fn
#     )
#     val_loader = DataLoader(
#         val_set,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=simple_collate_fn
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=simple_collate_fn
#     )
#
#     # 获取特征维度
#     img_dim = train_slices[0]['img_dim']
#     gene_dim = train_slices[0]['gene_dim']
#
#     # 初始化CLIP模型
#     model = CLIPModelWithClassifier(
#         temperature=temperature,
#         image_embedding=img_dim,
#         spot_embedding=gene_dim,
#         projection_dim=projection_dim
#     ).to(device)
#
#     print(f"模型参数: image_dim={img_dim}, gene_dim={gene_dim}, projection_dim={projection_dim}")
#
#     # 训练模型
#     print("\n开始训练CLIP模型...")
#     model = train_clip_model(model, train_loader, val_loader, device, epochs=epochs)
#
#     # 在测试集上评估
#     print("\n在测试集上评估CLIP模型...")
#     test_results = evaluate_clip_model(model, seed, test_loader, device)
#
#     # 打印测试结果
#     print("\n" + "=" * 60)
#     print(f"CLIP模型 种子 {seed} 测试结果:")
#     print("=" * 60)
#     print(f"AUC: {test_results['auc']:.4f}")
#     print(f"AP:  {test_results['ap']:.4f}")
#     print(f"F1:  {test_results['f1']:.4f}")
#     print(f"准确率: {test_results['accuracy']:.4f}")
#     print(f"敏感度: {test_results['sensitivity']:.4f}")
#     print(f"特异度: {test_results['specificity']:.4f}")
#
#     return test_results
#
#
# def main():
#     """主函数"""
#     # 配置参数
#     seeds = [24, 42, 123, 345, 567]
#     batch_size = 512  # CLIP模型可以使用更大的batch size
#     epochs = 15
#     projection_dim = 256
#     temperature = 0.07
#
#     print(f"\n{'=' * 60}")
#     print(f"{'CLIP多模态模型训练 (VisiumHBC)':^60}")
#     print(f"{'=' * 60}\n")
#     print(f"实验配置:")
#     print(f" - 随机种子: {seeds}")
#     print(f" - 批大小: {batch_size}")
#     print(f" - 训练轮数: {epochs}")
#     print(f" - 投影维度: {projection_dim}")
#     print(f" - 温度参数: {temperature}")
#
#     # 运行所有实验
#     all_results = []
#     for seed in seeds:
#         results = run_clip_experiment(seed, batch_size, epochs, projection_dim, temperature)
#         all_results.append(results)
#
#     # 计算平均结果
#     aucs = [r['auc'] for r in all_results]
#     aps = [r['ap'] for r in all_results]
#     f1s = [r['f1'] for r in all_results]
#     sensitivities = [r['sensitivity'] for r in all_results]
#     specificities = [r['specificity'] for r in all_results]
#     accuracies = [r['accuracy'] for r in all_results]
#
#     # 输出详细结果
#     print("\n" + "=" * 60)
#     print("各随机种子详细结果:")
#     print("=" * 60)
#     for i, seed in enumerate(seeds):
#         print(f"种子 {seed:4d}: "
#               f"AUC={aucs[i]:.4f}, "
#               f"AP={aps[i]:.4f}, "
#               f"F1={f1s[i]:.4f}, "
#               f"敏感度={sensitivities[i]:.4f}, "
#               f"特异度={specificities[i]:.4f}, "
#               f"准确率={accuracies[i]:.4f}")
#
#     # 计算统计量
#     auc_mean, auc_std = np.mean(aucs), np.std(aucs)
#     ap_mean, ap_std = np.mean(aps), np.std(aps)
#     f1_mean, f1_std = np.mean(f1s), np.std(f1s)
#     sens_mean, sens_std = np.mean(sensitivities), np.std(sensitivities)
#     spec_mean, spec_std = np.mean(specificities), np.std(specificities)
#     acc_mean, acc_std = np.mean(accuracies), np.std(accuracies)
#
#     print("\n" + "=" * 60)
#     print("CLIP模型最终性能 (均值±标准差):")
#     print("=" * 60)
#     print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
#     print(f"AP:  {ap_mean:.4f} ± {ap_std:.4f}")
#     print(f"F1:  {f1_mean:.4f} ± {f1_std:.4f}")
#     print(f"敏感度: {sens_mean:.4f} ± {sens_std:.4f}")
#     print(f"特异度: {spec_mean:.4f} ± {spec_std:.4f}")
#     print(f"准确率: {acc_mean:.4f} ± {acc_std:.4f}")
#     print("=" * 60)
#
#
# if __name__ == "__main__":
#     main()


###============================ XeniumHBC to HER2ST =====================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import random
from datetime import datetime
import pandas as pd

# 导入CLIP模型相关组件
from models import CLIPModelWithClassifier
from modules import ProjectionHead


def set_all_seeds(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class SimpleSpatialDataset(Dataset):
    """
    简化的空间数据集类，适配CLIP模型输入
    类似于文档1的数据结构，但适配文档2的模型输入格式
    """

    def __init__(self, img_feats, gene_exprs, labels, sample_name=""):
        self.img_feats = img_feats
        self.gene_exprs = gene_exprs
        self.labels = labels
        self.sample_name = sample_name

        # 验证数据形状
        assert len(img_feats) == len(gene_exprs) == len(labels), "特征和标签数量不匹配"

    def __len__(self):
        return len(self.img_feats)

    def __getitem__(self, idx):
        return {
            "image_features": torch.tensor(self.img_feats[idx], dtype=torch.float32),
            "spot_features": torch.tensor(self.gene_exprs[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def load_training_slices(npz_path=" ",
                         label_dir=" ",
                         all_samples=['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']):
    """加载训练切片数据 - 适配CLIP模型"""
    data = np.load(npz_path, allow_pickle=True)
    slices = []

    for i, sample in enumerate(all_samples):
        img_feats = data['img_resnet'][i]
        gene_exprs = data['gene_expr'][i]

        # 加载标签
        labels_path = f"{label_dir}/{sample}-labels3.txt"
        labels = np.loadtxt(labels_path, dtype=int)

        # 应用训练集标签处理：过滤-1
        valid_indices = labels != -2
        img_feats = img_feats[valid_indices]
        gene_exprs = gene_exprs[valid_indices]
        labels = labels[valid_indices]

        # 训练集标签转换：0/1 -> 0 (正常区域), 2 -> 1 (癌症区域)
        labels = np.where(labels == 2, 1, 0)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        # 创建简化数据集实例
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


def load_test_dataset(npz_path=" ",
                      label_path=" "):
    """加载测试集数据 - 适配CLIP模型"""
    test_data = np.load(npz_path, allow_pickle=True)
    test_img = test_data['img_resnet']
    test_expr = test_data['gene_expr']
    test_label = np.loadtxt(label_path, dtype=int)

    if test_label.ndim == 1:
        test_label = test_label.reshape(-1, 1)

    # 测试集标签处理：0->0 (正常), 1->1 (癌症)
    test_dataset = SimpleSpatialDataset(
        img_feats=test_img,
        gene_exprs=test_expr,
        labels=test_label,
        sample_name="test"
    )

    return test_dataset


def simple_collate_fn(batch):
    """简单的批处理函数"""
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = torch.stack([item[key] for item in batch])
    return batch_dict


def train_clip_model(model, train_loader, val_loader, device, epochs=30, lr=1e-4):
    """训练CLIP模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            # 移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels'].squeeze(1)  # BCE损失需要1D标签

            optimizer.zero_grad()

            # 前向传播 - CLIP模型返回字典
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

        # 计算平均训练损失
        avg_train_loss = total_loss / batch_count

        # 验证循环
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

                # 使用sigmoid得到概率
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
            print(f"保存最佳模型 (Val AUC={val_auc:.4f})")

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"train_loss: {avg_train_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

    # 加载最佳模型
    model.load_state_dict(torch.load("temp_best_clip_XeniumHBC.pth"))
    return model


def evaluate_clip_model(model, seed, test_loader, device, save_dir="../../predict_files/BLEEP/HER2ST/From_XeniumHBC"):
    """评估CLIP模型"""
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

    # 计算评估指标
    auc = roc_auc_score(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 保存预测概率 - 修改保存路径
    file_name = f"{save_dir}/predict_probs_{seed}.txt"
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
    """运行单个CLIP模型实验 - 修改为训练集和测试集互换"""
    set_all_seeds(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'!' * 20} CLIP模型 种子={seed} 实验开始 (训练集和测试集互换) {'!' * 20}")
    print(f"使用设备: {device}")

    # 配置参数 - 互换训练集和测试集路径
    # 原训练集现在作为测试集
    test_npz = "../../dataset/train_dataset_XeniumHBC(new).npz"  # 原训练集
    test_label_dir = "../../data/HB/her2st/label3"  # 原训练集标签目录

    # 原测试集现在作为训练集
    train_npz = "../../dataset/test_dataset_XeniumHBC(new).npz"  # 原测试集
    train_label_path = "../../data/HB/Xenium_HBC/spot level/label.txt"  # 原测试集标签路径

    # 训练集样本列表（原测试集只有一个样本）
    train_samples = ["XeniumHBC"]
    # 测试集样本列表（原训练集有多个样本）
    test_samples = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    # 1. 加载训练数据（原测试集）
    print("加载训练集数据（原测试集）...")
    # 修改load_test_dataset函数以适应训练集加载
    train_dataset = load_test_dataset(
        npz_path=train_npz,
        label_path=train_label_path
    )
    print(f"训练集包含 {len(train_dataset)} 个spots")

    # 2. 加载测试集数据（原训练集）
    print("加载测试集数据（原训练集）...")
    test_slices, _ = load_training_slices(
        npz_path=test_npz,
        label_dir=test_label_dir,
        all_samples=test_samples
    )

    # 合并所有测试切片
    all_test_data = [slice_data['dataset'] for slice_data in test_slices]
    full_test_dataset = ConcatDataset(all_test_data)

    # 打印测试数据统计
    total_test_spots = sum(len(dataset) for dataset in all_test_data)
    print(f"测试集总共包含 {total_test_spots} 个spots")

    # 划分训练集和验证集 (80%/20%)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_set, val_set = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 创建数据加载器
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
        full_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=simple_collate_fn
    )

    # 获取特征维度（从训练集获取）
    sample_item = train_dataset[0]
    img_dim = sample_item["image_features"].shape[0]
    gene_dim = sample_item["spot_features"].shape[0]

    # 初始化CLIP模型
    model = CLIPModelWithClassifier(
        temperature=temperature,
        image_embedding=img_dim,
        spot_embedding=gene_dim,
        projection_dim=projection_dim
    ).to(device)

    print(f"模型参数: image_dim={img_dim}, gene_dim={gene_dim}, projection_dim={projection_dim}")

    # 训练模型
    print("\n开始训练CLIP模型...")
    model = train_clip_model(model, train_loader, val_loader, device, epochs=epochs)

    # 在测试集上评估
    print("\n在测试集上评估CLIP模型...")
    test_results = evaluate_clip_model(model, seed, test_loader, device)

    # 打印测试结果
    print("\n" + "=" * 60)
    print(f"CLIP模型 种子 {seed} 测试结果 (训练集和测试集互换):")
    print("=" * 60)
    print(f"AUC: {test_results['auc']:.4f}")
    print(f"AP:  {test_results['ap']:.4f}")
    print(f"F1:  {test_results['f1']:.4f}")
    print(f"准确率: {test_results['accuracy']:.4f}")
    print(f"敏感度: {test_results['sensitivity']:.4f}")
    print(f"特异度: {test_results['specificity']:.4f}")

    return test_results


def main():
    """主函数"""
    # 配置参数
    seeds = [24, 42, 123, 345, 567]
    batch_size = 512  # CLIP模型可以使用更大的batch size
    epochs = 30
    projection_dim = 256
    temperature = 0.07

    print(f"\n{'=' * 60}")
    print(f"{'CLIP多模态模型训练 (训练集和测试集互换)':^60}")
    print(f"{'=' * 60}\n")
    print(f"实验配置:")
    print(f" - 随机种子: {seeds}")
    print(f" - 批大小: {batch_size}")
    print(f" - 训练轮数: {epochs}")
    print(f" - 投影维度: {projection_dim}")
    print(f" - 温度参数: {temperature}")
    print(f" - 训练集: 原测试集 (XeniumHBC)")
    print(f" - 测试集: 原训练集 (HER2ST切片)")

    # 运行所有实验
    all_results = []
    for seed in seeds:
        results = run_clip_experiment(seed, batch_size, epochs, projection_dim, temperature)
        all_results.append(results)

    # 计算平均结果
    aucs = [r['auc'] for r in all_results]
    aps = [r['ap'] for r in all_results]
    f1s = [r['f1'] for r in all_results]
    sensitivities = [r['sensitivity'] for r in all_results]
    specificities = [r['specificity'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]

    # 输出详细结果
    print("\n" + "=" * 60)
    print("各随机种子详细结果 (训练集和测试集互换):")
    print("=" * 60)
    for i, seed in enumerate(seeds):
        print(f"种子 {seed:4d}: "
              f"AUC={aucs[i]:.4f}, "
              f"AP={aps[i]:.4f}, "
              f"F1={f1s[i]:.4f}, "
              f"敏感度={sensitivities[i]:.4f}, "
              f"特异度={specificities[i]:.4f}, "
              f"准确率={accuracies[i]:.4f}")

    # 计算统计量
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)
    ap_mean, ap_std = np.mean(aps), np.std(aps)
    f1_mean, f1_std = np.mean(f1s), np.std(f1s)
    sens_mean, sens_std = np.mean(sensitivities), np.std(sensitivities)
    spec_mean, spec_std = np.mean(specificities), np.std(specificities)
    acc_mean, acc_std = np.mean(accuracies), np.std(accuracies)

    print("\n" + "=" * 60)
    print("CLIP模型最终性能 (均值±标准差) - 训练集和测试集互换:")
    print("=" * 60)
    print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"AP:  {ap_mean:.4f} ± {ap_std:.4f}")
    print(f"F1:  {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"敏感度: {sens_mean:.4f} ± {sens_std:.4f}")
    print(f"特异度: {spec_mean:.4f} ± {spec_std:.4f}")
    print(f"准确率: {acc_mean:.4f} ± {acc_std:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
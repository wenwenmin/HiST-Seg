import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import config as CFG
from modules import ProjectionHead
from models import CLIPModelWithClassifier  # 确保路径正确

# 确保保存预测文件的目录存在
os.makedirs("../../predict_files/BLEEP/CRC", exist_ok=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    target_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13]
    all_img = data['img_resnet'][target_indices]
    all_expr = data['gene_expr'][target_indices]
    all_pos = data['coords'][target_indices]
    dataset_dict = {}

    for i, sample in enumerate(all_samples):
        # 加载特征
        img_feats = all_img[i]
        gene_exprs = all_expr[i]
        coords = all_pos[i]

        # 加载标签
        labels_path = f"{label_dir}/{sample}.txt"
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
            "image": batch["image_features"].to(device),
            "reduced_expression": batch["spot_features"].to(device),
            "labels": batch["labels"].to(device).squeeze(1)  # BCE损失需要1D标签
        }

        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["total_loss"]

        # 检查 NaN
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected at epoch {epoch}")
            continue

        loss.backward()

        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * batch["image"].size(0)

    return total_loss / len(train_loader.dataset)


# 评估函数，返回指标和预测概率
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

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 二值化预测
    preds_binary = (all_preds > 0.5).astype(int)

    acc = accuracy_score(all_labels, preds_binary)
    f1 = f1_score(all_labels, preds_binary)

    # === 修改：处理单一类别导致的 AUC 计算错误 ===
    # 检查测试集中是否同时包含正负样本 (至少要有两个类别)
    if len(np.unique(all_labels)) < 2:
        # print("Warning: Only one class present in test set. AUC/AP set to NaN.")
        auc = np.nan
        ap = np.nan
    else:
        try:
            auc = roc_auc_score(all_labels, all_preds)
            ap = average_precision_score(all_labels, all_preds)
        except ValueError:
            auc = np.nan
            ap = np.nan
    # ==========================================

    return acc, auc, ap, f1, all_preds


# 留一交叉验证函数，接受随机种子参数
def leave_one_out_cv(dataset_dict, all_samples, seed):
    # 存储每个fold的评估结果
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
        print(f"\n===== 留一交叉验证: 测试切片 {test_sample} (seed={seed}) =====")

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

        # 初始化模型
        model = CLIPModelWithClassifier(
            temperature=CFG.temperature,
            image_embedding=train_img.shape[1],
            spot_embedding=train_gene.shape[1],
            projection_dim=CFG.projection_dim,
            # classification_weight=CFG.cl_weight
        ).to(device)

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

        # 训练模型
        for epoch in range(CFG.epochs):
            train_loss = train_model(model, train_loader, optimizer, epoch)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

        # 在测试集上评估
        acc, auc, ap, f1, preds = evaluate_model(model, test_loader)
        print(f"测试结果 - 准确率: {acc:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")

        # 保存预测概率
        output_dir = "../../predict_files/BLEEP/CRC/"
        os.makedirs(output_dir, exist_ok=True)
        pred_path = f"../../predict_files/BLEEP/CRC/probs_{test_sample}_seed={seed}.txt"
        np.savetxt(pred_path, preds)

        # 保存指标
        metrics["accuracy"][test_sample].append(acc)
        metrics["auc"][test_sample].append(auc)
        metrics["ap"][test_sample].append(ap)
        metrics["f1"][test_sample].append(f1)
        metrics["predictions"][test_sample] = preds

    return metrics


# 主函数
if __name__ == "__main__":
    # 配置参数
    CFG.batch_size = 512
    CFG.epochs = 10
    CFG.lr = 1e-4
    CFG.weight_decay = 1e-5
    CFG.projection_dim = 256
    CFG.temperature = 0.07
    CFG.log_interval = 5
    CFG.num_workers = 4

    # 数据路径和样本列表
    npz_path = "../../dataset/CRC_dataset(Final-77).npz"
    label_dir = "../../data/CRC/label/"
    all_samples = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4', 'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2', 'SN123_4',
                   'SN124_1', 'SN124_2', 'SN124_4']
    seeds = [24, 42, 123, 345, 567]  # 五个随机种子

    # 加载数据
    dataset_dict = load_all_data(npz_path, label_dir, all_samples)

    # 存储所有种子的结果
    all_metrics = {
        "accuracy": {s: [] for s in all_samples},
        "auc": {s: [] for s in all_samples},
        "ap": {s: [] for s in all_samples},
        "f1": {s: [] for s in all_samples}
    }

    # 遍历每个随机种子
    for seed in seeds:
        print(f"\n\n======================================")
        print(f"========== 随机种子: {seed} 开始 ==========")
        print(f"======================================")

        set_seed(seed)

        metrics = leave_one_out_cv(dataset_dict, all_samples, seed)

        for sample in all_samples:
            all_metrics["accuracy"][sample].extend(metrics["accuracy"][sample])
            all_metrics["auc"][sample].extend(metrics["auc"][sample])
            all_metrics["ap"][sample].extend(metrics["ap"][sample])
            all_metrics["f1"][sample].extend(metrics["f1"][sample])

        print(f"\n===== 随机种子 {seed} 各切片测试结果 =====")
        for sample in all_samples:
            # 这里也使用 nanmean 防止当前种子是 nan
            auc_val = np.nanmean(metrics['auc'][sample])
            ap_val = np.nanmean(metrics['ap'][sample])
            f1_val = np.nanmean(metrics['f1'][sample])
            print(f"{sample}: AUC={auc_val:.4f}, AP={ap_val:.4f}, F1={f1_val:.4f}")

    # 计算并输出最终结果
    print("\n" + "=" * 60)
    print("最终结果 (Mean ± Std per Sample):")
    print("=" * 60)

    # 存储所有样本的指标用于计算整体结果
    overall_metrics = {
        "auc": [],
        "ap": [],
        "f1": []
    }

    for sample in all_samples:
        # === 修改：使用 nanmean 和 nanstd 自动忽略 NaN 值 ===
        auc_mean = np.nanmean(all_metrics["auc"][sample])
        auc_std = np.nanstd(all_metrics["auc"][sample])
        ap_mean = np.nanmean(all_metrics["ap"][sample])
        ap_std = np.nanstd(all_metrics["ap"][sample])
        f1_mean = np.nanmean(all_metrics["f1"][sample])
        f1_std = np.nanstd(all_metrics["f1"][sample])

        print(
            f"{sample}: AUC={auc_mean:.4f}±{auc_std:.4f}, AP={ap_mean:.4f}±{ap_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}")

        # 收集用于整体结果计算 (只收集非NaN的值，或者在下一步统一处理)
        overall_metrics["auc"].extend(all_metrics["auc"][sample])
        overall_metrics["ap"].extend(all_metrics["ap"][sample])
        overall_metrics["f1"].extend(all_metrics["f1"][sample])

    # 计算整体结果
    # === 修改：使用 nanmean 和 nanstd ===
    overall_auc_mean = np.nanmean(overall_metrics["auc"])
    overall_auc_std = np.nanstd(overall_metrics["auc"])
    overall_ap_mean = np.nanmean(overall_metrics["ap"])
    overall_ap_std = np.nanstd(overall_metrics["ap"])
    overall_f1_mean = np.nanmean(overall_metrics["f1"])
    overall_f1_std = np.nanstd(overall_metrics["f1"])

    print("\n" + "=" * 60)
    print("整体结果:")
    print("=" * 60)
    print(f"AUC: {overall_auc_mean:.4f} ± {overall_auc_std:.4f}")
    print(f"AP:  {overall_ap_mean:.4f} ± {overall_ap_std:.4f}")
    print(f"F1:  {overall_f1_mean:.4f} ± {overall_f1_std:.4f}")
















# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
# import config as CFG
# from modules import ProjectionHead
# from models import CLIPModelWithClassifier  # 替换为模型定义路径
#
# # 确保保存预测文件的目录存在
# os.makedirs("../../predict_files/BLEEP/CRC", exist_ok=True)
#
# # 设备配置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # 设置随机种子
# def set_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#
# # 数据加载类
# class HER2STDataset(Dataset):
#     def __init__(self, img_feats, gene_exprs, labels):
#         self.img_feats = img_feats  # 已提取的图像特征 (n_spots, img_embedding_dim)
#         self.gene_exprs = gene_exprs  # 基因特征 (n_spots, spot_embedding_dim)
#         self.labels = labels  # 标签 (n_spots, 1)
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
# # 加载所有数据
# def load_all_data(npz_path, label_dir, all_samples):
#     data = np.load(npz_path, allow_pickle=True)
#     target_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13]
#     all_img = data['img_resnet'][target_indices]
#     all_expr = data['gene_expr'][target_indices]
#     all_pos = data['coords'][target_indices]
#     dataset_dict = {}
#
#     for i, sample in enumerate(all_samples):
#         # 加载特征和坐标（坐标暂未使用，如需使用可在后续扩展）
#         img_feats = all_img[i]  # 已提取的图像特征
#         gene_exprs = all_expr[i]  # 基因特征
#         coords = all_pos[i]  # 坐标信息（备用）
#
#         # 加载标签
#         labels_path = f"{label_dir}/{sample}.txt"
#         labels = np.loadtxt(labels_path, dtype=int)
#         if labels.ndim == 1:
#             labels = labels.reshape(-1, 1)
#
#         # 检查特征和标签数量是否匹配
#         assert len(img_feats) == len(gene_exprs) == len(labels), \
#             f"样本{sample}的特征和标签数量不匹配"
#
#         dataset_dict[sample] = {
#             "img_feats": img_feats,
#             "gene_exprs": gene_exprs,
#             "coords": coords,
#             "labels": labels
#         }
#         print(f"加载切片{sample}: {len(img_feats)}个spots")
#
#     return dataset_dict
#
#
# # 训练函数
# def train_model(model, train_loader, optimizer, epoch):
#     model.train()
#     total_loss = 0.0
#     for batch in train_loader:
#         # 数据移至设备
#         batch = {
#             "image": batch["image_features"].to(device),  # 直接使用预提取的图像特征
#             "reduced_expression": batch["spot_features"].to(device),
#             "labels": batch["labels"].to(device).squeeze(1)  # BCE损失需要1D标签
#         }
#
#         optimizer.zero_grad()
#         outputs = model(batch)
#         loss = outputs["total_loss"]
#
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * batch["image"].size(0)
#
#     return total_loss / len(train_loader.dataset)
#
#
# # 评估函数，返回指标和预测概率
# def evaluate_model(model, test_loader):
#     model.eval()
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = {
#                 "image": batch["image_features"].to(device),
#                 "reduced_expression": batch["spot_features"].to(device),
#                 "labels": batch["labels"].to(device).squeeze(1)
#             }
#
#             outputs = model(batch)
#             # 计算预测概率（sigmoid输出）
#             preds = torch.sigmoid(outputs["class_logits"]).cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(batch["labels"].cpu().numpy())
#
#     # 计算评估指标
#     preds_binary = (np.array(all_preds) > 0.5).astype(int)  # 二值化预测
#     acc = accuracy_score(all_labels, preds_binary)
#     auc = roc_auc_score(all_labels, all_preds)
#     ap = average_precision_score(all_labels, all_preds)  # 添加AP指标
#     f1 = f1_score(all_labels, preds_binary)
#
#     return acc, auc, ap, f1, np.array(all_preds)
#
#
# # 留一交叉验证函数，接受随机种子参数
# def leave_one_out_cv(dataset_dict, all_samples, seed):
#     # 存储每个fold的评估结果
#     metrics = {
#         "accuracy": {},
#         "auc": {},
#         "ap": {},
#         "f1": {},
#         "predictions": {}  # 存储预测概率
#     }
#
#     for sample in all_samples:
#         metrics["accuracy"][sample] = []
#         metrics["auc"][sample] = []
#         metrics["ap"][sample] = []
#         metrics["f1"][sample] = []
#
#     for test_sample in all_samples:
#         print(f"\n===== 留一交叉验证: 测试切片 {test_sample} (seed={seed}) =====")
#
#         # 划分训练集和测试集
#         train_samples = [s for s in all_samples if s != test_sample]
#
#         # 合并训练集数据
#         train_img = np.concatenate([dataset_dict[s]["img_feats"] for s in train_samples], axis=0)
#         train_gene = np.concatenate([dataset_dict[s]["gene_exprs"] for s in train_samples], axis=0)
#         train_labels = np.concatenate([dataset_dict[s]["labels"] for s in train_samples], axis=0)
#
#         # 测试集数据
#         test_img = dataset_dict[test_sample]["img_feats"]
#         test_gene = dataset_dict[test_sample]["gene_exprs"]
#         test_labels = dataset_dict[test_sample]["labels"]
#
#         # 创建数据集和数据加载器
#         train_dataset = HER2STDataset(train_img, train_gene, train_labels)
#         test_dataset = HER2STDataset(test_img, test_gene, test_labels)
#
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=CFG.batch_size,
#             shuffle=True,
#             num_workers=CFG.num_workers
#         )
#         test_loader = DataLoader(
#             test_dataset,
#             batch_size=CFG.batch_size,
#             shuffle=False,
#             num_workers=CFG.num_workers
#         )
#
#         # 初始化模型（注意：使用预提取的图像特征，无需图像编码器）
#         model = CLIPModelWithClassifier(
#             temperature=CFG.temperature,
#             image_embedding=train_img.shape[1],  # 图像特征维度（从数据中获取）
#             spot_embedding=train_gene.shape[1],  # 基因特征维度
#             projection_dim=CFG.projection_dim,
#             #classification_weight=CFG.cl_weight
#         ).to(device)
#
#         # 优化器
#         optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
#
#         # 训练模型
#         best_auc = 0.0
#         for epoch in range(CFG.epochs):
#             train_loss = train_model(model, train_loader, optimizer, epoch)
#             print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
#
#         # 在测试集上评估
#         acc, auc, ap, f1, preds = evaluate_model(model, test_loader)
#         print(f"测试结果 - 准确率: {acc:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")
#
#         # 保存预测概率
#         output_dir="../../predict_files/BLEEP/CRC/"
#         os.makedirs(output_dir, exist_ok=True)
#         pred_path = f"../../predict_files/BLEEP/CRC/probs_{test_sample}_seed={seed}.txt"
#         np.savetxt(pred_path, preds)
#
#         # 保存指标
#         metrics["accuracy"][test_sample].append(acc)
#         metrics["auc"][test_sample].append(auc)
#         metrics["ap"][test_sample].append(ap)
#         metrics["f1"][test_sample].append(f1)
#         metrics["predictions"][test_sample] = preds
#
#     return metrics
#
#
# # 主函数
# if __name__ == "__main__":
#     # 配置参数（可在config.py中定义）
#     CFG.batch_size = 512
#     CFG.epochs = 10
#     CFG.lr = 1e-4
#     CFG.weight_decay = 1e-5
#     CFG.projection_dim = 256
#     CFG.temperature = 0.07
#     #CFG.cl_weight = 1.0
#     CFG.log_interval = 5
#     CFG.num_workers = 4
#
#     # 数据路径和样本列表
#     npz_path = "../../dataset/CRC_dataset(Final-77).npz"
#     label_dir = "../../data/CRC/label/"
#     all_samples = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4', 'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2', 'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']
#     seeds = [24, 42, 123, 345, 567]  # 五个随机种子
#
#     # 加载数据
#     dataset_dict = load_all_data(npz_path, label_dir, all_samples)
#
#     # 存储所有种子的结果
#     all_metrics = {
#         "accuracy": {s: [] for s in all_samples},
#         "auc": {s: [] for s in all_samples},
#         "ap": {s: [] for s in all_samples},
#         "f1": {s: [] for s in all_samples}
#     }
#
#     # 遍历每个随机种子
#     for seed in seeds:
#         print(f"\n\n======================================")
#         print(f"========== 随机种子: {seed} 开始 ==========")
#         print(f"======================================")
#
#         # 设置随机种子
#         set_seed(seed)
#
#         # 执行留一交叉验证
#         metrics = leave_one_out_cv(dataset_dict, all_samples, seed)
#
#         # 收集结果
#         for sample in all_samples:
#             all_metrics["accuracy"][sample].extend(metrics["accuracy"][sample])
#             all_metrics["auc"][sample].extend(metrics["auc"][sample])
#             all_metrics["ap"][sample].extend(metrics["ap"][sample])
#             all_metrics["f1"][sample].extend(metrics["f1"][sample])
#
#         # 输出当前种子下各切片的测试结果
#         print(f"\n===== 随机种子 {seed} 各切片测试结果 =====")
#         for sample in all_samples:
#             print(
#                 f"{sample}: AUC={metrics['auc'][sample][0]:.4f}, AP={metrics['ap'][sample][0]:.4f}, F1={metrics['f1'][sample][0]:.4f}")
#
#     # 计算并输出最终结果
#     print("\n" + "=" * 60)
#     print("最终结果 (Mean ± Std per Sample):")
#     print("=" * 60)
#
#     # 存储所有样本的指标用于计算整体结果
#     overall_metrics = {
#         "auc": [],
#         "ap": [],
#         "f1": []
#     }
#
#     for sample in all_samples:
#         auc_mean = np.mean(all_metrics["auc"][sample])
#         auc_std = np.std(all_metrics["auc"][sample])
#         ap_mean = np.mean(all_metrics["ap"][sample])
#         ap_std = np.std(all_metrics["ap"][sample])
#         f1_mean = np.mean(all_metrics["f1"][sample])
#         f1_std = np.std(all_metrics["f1"][sample])
#
#         print(
#             f"{sample}: AUC={auc_mean:.4f}±{auc_std:.4f}, AP={ap_mean:.4f}±{ap_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}")
#
#         # 收集用于整体结果计算
#         overall_metrics["auc"].extend(all_metrics["auc"][sample])
#         overall_metrics["ap"].extend(all_metrics["ap"][sample])
#         overall_metrics["f1"].extend(all_metrics["f1"][sample])
#
#     # 计算整体结果
#     overall_auc_mean = np.mean(overall_metrics["auc"])
#     overall_auc_std = np.std(overall_metrics["auc"])
#     overall_ap_mean = np.mean(overall_metrics["ap"])
#     overall_ap_std = np.std(overall_metrics["ap"])
#     overall_f1_mean = np.mean(overall_metrics["f1"])
#     overall_f1_std = np.std(overall_metrics["f1"])
#
#     print("\n" + "=" * 60)
#     print("整体结果:")
#     print("=" * 60)
#     print(f"AUC: {overall_auc_mean:.4f} ± {overall_auc_std:.4f}")
#     print(f"AP:  {overall_ap_mean:.4f} ± {overall_ap_std:.4f}")
#     print(f"F1:  {overall_f1_mean:.4f} ± {overall_f1_std:.4f}")
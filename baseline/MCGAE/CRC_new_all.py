import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from MCGAE.model import MCGAE  # 导入MCGAE模型
from MCGAE.utils import set_seed  # 假设utils中有随机种子设置函数
import warnings

warnings.filterwarnings("ignore")

# --------------------------
# 1. 配置参数与路径
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 切片列表
slices = ['ST-colon1', 'ST-colon2', 'ST-colon3', 'ST-colon4']

# 基础数据路径
base_data_path = "../../data/CRC_new/"

# 图像处理参数
patch_size = 256  # 裁剪的图像块大小
image_resize = None  # 若图像过大可调整分辨率，如(2000, 2000)

# 模型参数
n_latent = 50  # 潜在特征维度
n_clusters = 2  # 聚类数（正常/癌症）
fusion_mode = "holistic"  # 多视图融合方式：holistic/fractional/vanilla
epochs = 100  # 训练轮次
lr = 1e-3  # 学习率
weight_decay = 1e-5  # 权重衰减
n_components = 50  # PCA降维维度（与模型参数一致）

# 测试种子
test_seeds = [24, 42, 123, 345, 567]


# --------------------------
# 2. 数据加载与预处理
# --------------------------
def load_data(slice_name):
    """加载指定切片的基因表达数据、spot坐标、图像和标签"""
    # 构建切片特定路径
    slice_path = os.path.join(base_data_path, slice_name)
    image_path = os.path.join(slice_path, "tissue_hires_image.png")
    adata_path = os.path.join(slice_path, "transformed.h5ad")
    label_path = os.path.join(slice_path, "spot_labels.txt")

    print(f"\nLoading data for {slice_name}...")
    print(f"Image path: {image_path}")
    print(f"Adata path: {adata_path}")
    print(f"Label path: {label_path}")

    # 2.1 加载空间转录组数据（基因表达+坐标）
    adata = sc.read_h5ad(adata_path)  # 自动读取count矩阵和坐标
    adata.var_names_make_unique()
    print(f"Loaded adata: {adata.shape} (spots × genes)")
    n_spots = adata.shape[0]  # 获取spot总数

    # 2.2 加载真实标签（0=正常，1=癌症），按顺序与spot对应
    labels = np.loadtxt(label_path, dtype=int)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    if len(labels) != n_spots:
        raise ValueError(f"标签数量（{len(labels)}）与spot数量（{n_spots}）不匹配！")
    y_true = labels
    adata.obs["label"] = y_true  # 存储真实标签到obs

    # 过滤无标签的spot
    adata = adata[~adata.obs["label"].isna()]
    y_true = adata.obs["label"].astype(int).values
    print(f"Valid spots with labels: {len(y_true)}")

    # 2.3 提取spot坐标（转换为图像像素坐标）
    spot_coords = adata.obsm['spatial']  # 这直接就是坐标数组
    print(f"空间坐标形状: {spot_coords.shape}")

    # 使用与你的正确代码相同的坐标变换
    pixel_coords = []
    for (x, y) in spot_coords:
        # 应用相同的坐标变换
        x_transformed = int(x * 3.3 + 20)
        y_transformed = int(y * 3.3 + 20)
        pixel_coords.append([x_transformed, y_transformed])
    pixel_coords = np.array(pixel_coords)
    adata.obsm["pixel_coords"] = pixel_coords

    # 2.4 加载组织切片图像并裁剪每个spot的图像块
    img = Image.open(image_path).convert("RGB")
    if image_resize is not None:
        img = img.resize(image_resize)
    img_array = np.array(img)
    h, w, _ = img_array.shape
    print(f"Loaded image: {h}×{w} pixels")

    # 裁剪图像块（确保与spot顺序一致）
    patch_list = []
    for (x, y) in pixel_coords:
        x1 = max(0, x - patch_size // 2)
        x2 = min(w, x + patch_size // 2)
        y1 = max(0, y - patch_size // 2)
        y2 = min(h, y + patch_size // 2)
        patch = img_array[y1:y2, x1:x2]
        # 填充边界以保证固定大小
        pad_h = patch_size - (y2 - y1)
        pad_w = patch_size - (x2 - x1)
        patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        patch_list.append(patch)
    img_features = np.array(patch_list)
    print(f"Extracted {len(patch_list)} image patches")

    return adata, y_true, img_features


# --------------------------
# 3. 特征预处理（补充多视图特征）
# --------------------------
def preprocess_features(adata, img_features, seed):
    """预处理基因表达特征（多视图）、图像特征、邻接矩阵，存入adata.obsm"""
    # 3.1 基因表达特征预处理（生成多视图：原始、扰动、VAE增强）
    # 3.1.1 原始特征（feat_orig）
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    expr_orig = adata.X.toarray()
    expr_orig = StandardScaler().fit_transform(expr_orig)
    adata.obsm['feat_orig'] = expr_orig  # 原始特征

    # 3.1.2 扰动特征（feat_corr：添加高斯噪声，增强鲁棒性）
    np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=0.1, size=expr_orig.shape)  # 小噪声
    expr_corr = expr_orig + noise
    expr_corr = StandardScaler().fit_transform(expr_corr)
    adata.obsm['feat_corr'] = expr_corr  # 扰动特征

    # 3.1.3 VAE增强特征（feat_vae：简单VAE去噪，可选）
    class SimpleVAE(torch.nn.Module):
        def __init__(self, input_dim, latent_dim=100):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 200),
                torch.nn.ReLU(),
                torch.nn.Linear(200, latent_dim * 2)  # 均值+方差
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 200),
                torch.nn.ReLU(),
                torch.nn.Linear(200, input_dim)
            )
            self.latent_dim = latent_dim

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            z_params = self.encoder(x)
            mu, logvar = z_params[:, :self.latent_dim], z_params[:, self.latent_dim:]
            z = self.reparameterize(mu, logvar)
            x_rec = self.decoder(z)
            return x_rec, mu, logvar

    # 训练简单VAE生成增强特征
    input_dim = expr_orig.shape[1]
    vae = SimpleVAE(input_dim=input_dim, latent_dim=100).to(device)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)
    x_tensor = torch.FloatTensor(expr_orig).to(device)
    vae.train()
    for epoch in range(50):  # 少量迭代即可
        x_rec, mu, logvar = vae(x_tensor)
        # VAE损失：重构损失 + KL散度
        recon_loss = F.mse_loss(x_rec, x_tensor)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        loss = recon_loss + 0.001 * kl_loss
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer_vae.step()
    # 生成VAE增强特征（用重构结果）
    vae.eval()
    with torch.no_grad():
        x_rec, _, _ = vae(x_tensor)
    expr_vae = x_rec.cpu().numpy()
    expr_vae = StandardScaler().fit_transform(expr_vae)
    adata.obsm['feat_vae'] = expr_vae  # VAE增强特征

    # 3.1.4 PCA降维特征（满足use_pca=True的需求）
    pca = PCA(n_components=n_components, random_state=seed)
    adata.obsm['feat_pca_orig'] = pca.fit_transform(expr_orig)  # 原始特征PCA
    adata.obsm['feat_pca_corr'] = pca.transform(expr_corr)  # 扰动特征PCA
    adata.obsm['feat_pca_vae'] = pca.transform(expr_vae)  # VAE特征PCA

    # 3.2 图像特征提取（X_morphology）
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    cnn = torch.nn.Sequential(*list(cnn.children())[:-1])  # 移除最后一层全连接
    cnn.to(device)
    cnn.eval()

    img_feats = []
    with torch.no_grad():
        for patch in img_features:
            x = transform(patch).unsqueeze(0).to(device)
            feat = cnn(x).squeeze().cpu().numpy()
            img_feats.append(feat)
    img_features = np.array(img_feats)
    img_features = StandardScaler().fit_transform(img_features)
    adata.obsm['X_morphology'] = img_features  # 图像形态特征

    # 3.3 空间邻接矩阵（adj_orig, adj_aug, graph_orig）
    adj_orig = build_spatial_adj(adata, k=10)  # 原始邻接矩阵（KNN）
    adata.obsm['adj_orig'] = adj_orig
    # 增强邻接矩阵（可选，用余弦相似度）
    adj_aug = 1 - cdist(expr_orig, expr_orig, metric='cosine')  # 特征相似度邻接
    adj_aug[adj_aug < 0.5] = 0  # 阈值过滤
    adata.obsm['adj_aug'] = adj_aug.astype(np.float32)
    # graph_orig可直接复用adj_orig（模型可能需要的图结构）
    adata.obsm['graph_orig'] = adj_orig

    # 3.4 标签信息（label_cls，模型训练中可能用于中间评估）
    labels = adata.obs["label"].values.astype(int)  # 原始标签：(N,)
    # 转换为独热编码（N个样本，2个类别）
    one_hot_labels = np.zeros((len(labels), n_clusters), dtype=np.float32)
    one_hot_labels[np.arange(len(labels)), labels] = 1  # 第i个样本的labels[i]位置设为1
    adata.obsm['label_cls'] = one_hot_labels  # 存储为(N, 2)的独热矩阵

    # 打印特征形状
    print(f"基因特征（原始/扰动/VAE）：{expr_orig.shape}")
    print(f"PCA特征：{adata.obsm['feat_pca_orig'].shape}")
    print(f"图像特征：{img_features.shape}")
    print(f"邻接矩阵：{adj_orig.shape}")
    print(f"label_cls形状：{adata.obsm['label_cls'].shape}")
    return adata


# --------------------------
# 4. 构建空间邻接矩阵
# --------------------------
def build_spatial_adj(adata, k=10):
    """基于像素坐标构建KNN邻接矩阵（无向图）"""
    coords = adata.obsm["pixel_coords"]
    dist_matrix = cdist(coords, coords, metric="euclidean")
    adj = np.zeros_like(dist_matrix)
    for i in range(len(coords)):
        nearest_idx = np.argsort(dist_matrix[i])[1:k + 1]  # 排除自身
        adj[i, nearest_idx] = 1
    adj = (adj + adj.T) / 2  # 对称化
    return adj.astype(np.float32)


# --------------------------
# 5. 模型训练与聚类
# --------------------------
def train_mcgae(adata, y_true, seed):
    """训练MCGAE模型并输出聚类结果"""
    # 初始化模型
    model = MCGAE(
        adata=adata,
        n_latent=n_latent,
        n_components=n_components,
        use_pca=True,
        fusion_mode=fusion_mode,
        random_seed=seed,  # 使用当前种子
        w_morph=0.5  # 图像特征权重
    )

    # 训练模型
    model.train(
        max_epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        cluster_method="kmeans",  # 二分类适合kmeans
        n_clusters=n_clusters,
        cl_start_epoch=50,  # 中期开始引入聚类损失
        loss_print_interval=10,
        device=device  # 显式指定设备
    )

    # 获取模型输出（包含聚类概率q）
    model_output = model.get_model_output()
    y_pred_cluster = model_output['y_pred']

    # 关键修改：将q从GPU转移到CPU并转换为NumPy数组
    if 'q' in model_output:
        q_tensor = model_output['q']  # 获取q（可能是GPU张量）
        # 判断是否为PyTorch张量，若是则转移到CPU并转NumPy
        if isinstance(q_tensor, torch.Tensor):
            q = q_tensor.cpu().numpy()  # 先移到CPU，再转NumPy
        else:
            q = q_tensor  # 若已是NumPy数组则直接使用
    else:
        raise ValueError("模型输出model_output中未包含'q'，请检查模型输出逻辑")

    return y_pred_cluster, q


# --------------------------
# 6. 标签映射与评估
# --------------------------
def evaluate(y_true, y_pred_cluster, q, n_clusters):
    """将聚类结果映射到真实标签并计算指标"""
    from scipy.optimize import linear_sum_assignment

    # 计算混淆矩阵
    confusion = np.zeros((n_clusters, n_clusters), dtype=int)
    for t, p in zip(y_true, y_pred_cluster):
        confusion[t, p] += 1
    # 匈牙利算法匹配聚类标签与真实标签
    row_ind, col_ind = linear_sum_assignment(-confusion)
    label_map = {col_ind[i]: row_ind[i] for i in range(n_clusters)}
    y_pred = np.array([label_map[p] for p in y_pred_cluster])

    # 计算评估指标
    f1 = f1_score(y_true, y_pred)
    # 提取预测为癌症（1）的概率
    y_score = q[:, label_map[1]] if 1 in label_map else 1 - q[:, label_map[0]]
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return f1, auc, ap


# --------------------------
# 统计计算函数
# --------------------------
def calculate_stats(values):
    """计算均值和标准差"""
    mean = np.mean(values)
    std = np.std(values)
    return mean, std, f"{mean:.4f} ± {std:.4f}"


# --------------------------
# 主函数
# --------------------------
if __name__ == "__main__":
    # 创建保存目录
    save_dir = "../../predict_files/MCGAE/CRC_new/"
    os.makedirs(save_dir, exist_ok=True)

    # 存储所有实验结果
    results = {
        slice_name: {
            seed: {'f1': [], 'auc': [], 'ap': []} for seed in test_seeds
        } for slice_name in slices
    }

    # 遍历所有切片
    for slice_name in slices:
        print(f"\n{'=' * 50}")
        print(f"开始处理切片: {slice_name}")
        print(f"{'=' * 50}")

        # 加载当前切片数据（只加载一次，提高效率）
        try:
            adata, y_true, img_features = load_data(slice_name)
        except Exception as e:
            print(f"加载切片 {slice_name} 数据失败: {e}")
            continue

        # 遍历所有种子
        for seed in test_seeds:
            print(f"\n{'=' * 20} 切片 {slice_name} - 种子 {seed} 的实验 {'=' * 20}")

            # 设置随机种子
            set_seed(seed)

            try:
                # 预处理特征（使用adata副本避免干扰）
                adata_processed = preprocess_features(adata.copy(), img_features, seed)

                # 训练模型
                y_pred_cluster, q = train_mcgae(adata_processed, y_true, seed)

                # 评估并获取指标
                f1, auc, ap = evaluate(y_true, y_pred_cluster, q, n_clusters)

                # 存储结果
                results[slice_name][seed] = {'f1': f1, 'auc': auc, 'ap': ap}

                # 保存当前种子的预测结果
                np.savetxt(f"{save_dir}/{slice_name}_pred_seed={seed}.txt", y_pred_cluster, fmt='%d')

                print(f"切片 {slice_name} - 种子 {seed} 结果:")
                print(f"AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")

            except Exception as e:
                print(f"切片 {slice_name} - 种子 {seed} 实验失败: {e}")
                results[slice_name][seed] = {'f1': np.nan, 'auc': np.nan, 'ap': np.nan}

    # --------------------------
    # 结果汇总与输出
    # --------------------------
    print(f"\n{'=' * 60}")
    print(f"{'=' * 20} 所有实验汇总结果 {'=' * 20}")
    print(f"{'=' * 60}")

    # 创建结果表格
    summary_data = []

    # 计算每个切片的统计量
    slice_stats = {}
    for slice_name in slices:
        # 收集该切片所有种子的指标
        f1_values = [results[slice_name][seed]['f1'] for seed in test_seeds
                     if not np.isnan(results[slice_name][seed]['f1'])]
        auc_values = [results[slice_name][seed]['auc'] for seed in test_seeds
                      if not np.isnan(results[slice_name][seed]['auc'])]
        ap_values = [results[slice_name][seed]['ap'] for seed in test_seeds
                     if not np.isnan(results[slice_name][seed]['ap'])]

        if len(f1_values) > 0:
            f1_mean, f1_std, f1_str = calculate_stats(f1_values)
            auc_mean, auc_std, auc_str = calculate_stats(auc_values)
            ap_mean, ap_std, ap_str = calculate_stats(ap_values)

            slice_stats[slice_name] = {
                'f1': (f1_mean, f1_std), 'auc': (auc_mean, auc_std), 'ap': (ap_mean, ap_std)
            }

            # 输出每个种子的详细结果
            print(f"\n{slice_name} 详细结果:")
            for seed in test_seeds:
                if not np.isnan(results[slice_name][seed]['f1']):
                    print(f"  种子 {seed}: AUC={results[slice_name][seed]['auc']:.4f}, "
                          f"AP={results[slice_name][seed]['ap']:.4f}, "
                          f"F1={results[slice_name][seed]['f1']:.4f}")

            print(f"{slice_name} 汇总: AUC={auc_str}, AP={ap_str}, F1={f1_str}")
            summary_data.append([slice_name, auc_str, ap_str, f1_str])
        else:
            print(f"{slice_name}: 无有效实验结果")
            summary_data.append([slice_name, "N/A", "N/A", "N/A"])

    # 计算总体统计量（所有切片的均值±方差的均值）
    if slice_stats:
        # 收集所有切片的均值
        all_auc_means = [stats['auc'][0] for stats in slice_stats.values()]
        all_ap_means = [stats['ap'][0] for stats in slice_stats.values()]
        all_f1_means = [stats['f1'][0] for stats in slice_stats.values()]

        # 收集所有切片的方差
        all_auc_stds = [stats['auc'][1] for stats in slice_stats.values()]
        all_ap_stds = [stats['ap'][1] for stats in slice_stats.values()]
        all_f1_stds = [stats['f1'][1] for stats in slice_stats.values()]

        # 计算总体均值（各切片均值的均值）
        overall_auc_mean = np.mean(all_auc_means)
        overall_ap_mean = np.mean(all_ap_means)
        overall_f1_mean = np.mean(all_f1_means)

        # 计算总体标准差（各切片标准差的均值）
        overall_auc_std = np.mean(all_auc_stds)
        overall_ap_std = np.mean(all_ap_stds)
        overall_f1_std = np.mean(all_f1_stds)

        overall_auc_str = f"{overall_auc_mean:.4f} ± {overall_auc_std:.4f}"
        overall_ap_str = f"{overall_ap_mean:.4f} ± {overall_ap_std:.4f}"
        overall_f1_str = f"{overall_f1_mean:.4f} ± {overall_f1_std:.4f}"

        summary_data.append(["ALL", overall_auc_str, overall_ap_str, overall_f1_str])

        print(f"\n{'=' * 60}")
        print(f"{'=' * 20} 最终汇总结果 {'=' * 20}")
        print(f"{'=' * 60}")
        print(f"ALL: AUC={overall_auc_str}, AP={overall_ap_str}, F1={overall_f1_str}")

    # 保存汇总结果到CSV文件
    summary_df = pd.DataFrame(summary_data, columns=["Slice", "AUC", "AP", "F1"])
    summary_csv_path = os.path.join(save_dir, "experiment_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n详细结果已保存至: {summary_csv_path}")

    # 打印最终表格
    print(f"\n{'=' * 80}")
    print("最终结果表格:")
    print(f"{'=' * 80}")
    print(summary_df.to_string(index=False))
    print(f"{'=' * 80}")
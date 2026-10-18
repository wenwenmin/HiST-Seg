import argparse
import torch
import numpy as np
from train import get_model_CL as train_load_model_CL
from train import get_model_Rec as train_load_model_Rec
from model import ContrastiveModel, ReconstructionModel
from Dataset import SpotDataset_CL, SpotDataset_Rec
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import evaluate, load_label
import random
import os
import time
import pandas as pd
from datetime import datetime


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_model(
        x, exp, locs, y, prefix, batch_size, epochs, lr, device='cuda', CL_output=512):
    """获取模型，将设备参数转为字符串类型"""
    x = x.copy()
    # 确保设备参数是字符串类型（解决KeyError问题）
    device_str = device if isinstance(device, str) else device.type

    datasets_CA = []
    for i in range(len(x)):
        datasets_CA.append(SpotDataset_CL(x[i], exp[i], locs[i]))

    model_CL, checkpoint_file = train_load_model_CL(
        model_class=ContrastiveModel,
        model_kwargs=dict(
            lr=lr,
            img_dim=x[0].shape[-1],
            gene_dim=exp[0].shape[-1],
            proj_dim=CL_output,
            temperature=0.07),
        datasets=datasets_CA, prefix=prefix,
        batch_size=batch_size, epochs=epochs, device=device_str)  # 使用字符串设备名
    model_CL.eval()
    if device_str == 'cuda':
        torch.cuda.empty_cache()

    datasets_Rec = []
    for i in range(len(x)):
        datasets_Rec.append(SpotDataset_Rec(x[i], exp[i], locs[i], y[i]))

    model_Rec = train_load_model_Rec(
        model_class=ReconstructionModel,
        model_kwargs=dict(
            contrastive_checkpoint_path=checkpoint_file,
            proj_dim=CL_output,
            img_dim=x[0].shape[-1],
            gene_dim=exp[0].shape[-1],
            lr=lr),
        datasets=datasets_Rec, prefix=prefix,
        batch_size=batch_size, epochs=epochs, device=device_str)  # 使用字符串设备名
    model_Rec.eval()
    if device_str == 'cuda':
        torch.cuda.empty_cache()
    return model_Rec


def normalize(data):
    data = data.copy()
    data_min = data.min(0)
    data_max = data.max(0)
    data -= data_min
    data /= (data_max - data_min) + 1e-12
    return data


def training(
        imgs, exps, locs, y, epochs, lr, batch_size, prefix, device='cuda', n_jobs=1, CL_output=512):
    imgs = [img.astype(np.float32) for img in imgs]
    imgs = [normalize(img) for img in imgs]
    # 确保设备参数是字符串类型
    device_str = device if isinstance(device, str) else device.type

    kwargs = dict(x=imgs, exp=exps, locs=locs, y=y, batch_size=batch_size, epochs=epochs, lr=lr,
                  prefix=f'{prefix}states/0/', device=device_str, CL_output=CL_output)  # 使用字符串设备名

    model = get_model(**kwargs)
    return model


def inference(model, imgs, exps, locs, y, batch_size, device='cuda'):
    # 调整设备参数为字符串类型
    device_str = device if isinstance(device, str) else device.type
    trainer = pl.Trainer(
        accelerator=device_str,
        devices=1,
        deterministic=True
    )

    # 确保输入数据格式正确
    imgs = np.array(imgs)
    exps = np.array(exps)
    locs = np.array(locs)
    y = np.array(y)

    # 处理测试数据的特殊格式（去掉多余的维度）
    if len(imgs.shape) == 3 and imgs.shape[0] == 1:
        imgs = imgs[0]  # 从 (1, n, f) 变为 (n, f)
    if len(exps.shape) == 3 and exps.shape[0] == 1:
        exps = exps[0]  # 从 (1, n, f) 变为 (n, f)
    if len(locs.shape) == 3 and locs.shape[0] == 1:
        locs = locs[0]  # 从 (1, n, 2) 变为 (n, 2)

    # 关键修复：确保数据类型正确，从object转换为float32
    imgs = imgs.astype(np.float32)
    exps = exps.astype(np.float32)
    locs = locs.astype(np.float32)
    y = y.astype(np.float32)

    print(f"推理数据形状 - imgs: {imgs.shape}, exps: {exps.shape}, locs: {locs.shape}, y: {y.shape}")
    print(f"推理数据类型 - imgs: {imgs.dtype}, exps: {exps.dtype}, locs: {locs.dtype}, y: {y.dtype}")

    imgs = normalize(imgs)

    model.eval()

    # 创建自定义的collate函数来处理数据
    def custom_collate_fn(batch):
        # 假设batch是列表，每个元素是(img, exp, loc, y)的元组
        imgs_batch = np.array([item[0] for item in batch])
        exps_batch = np.array([item[1] for item in batch])
        locs_batch = np.array([item[2] for item in batch])
        y_batch = np.array([item[3] for item in batch])

        # 转换为tensor
        imgs_batch = torch.from_numpy(imgs_batch).float()
        exps_batch = torch.from_numpy(exps_batch).float()
        locs_batch = torch.from_numpy(locs_batch).float()
        y_batch = torch.from_numpy(y_batch).float()

        return imgs_batch, exps_batch, locs_batch, y_batch

    # 创建数据集和数据加载器
    dataset = SpotDataset_Rec(imgs, exps, locs, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn  # 使用自定义collate函数
    )

    # 测试模型
    trainer.test(model, dataloader)
    results = model.scores

    min_score = results.min()
    max_score = results.max()
    results = (results - min_score) / (max_score - min_score)

    return results


def load_training_data(data_path, label_dir, slice_names):
    """加载训练数据（HER2ST数据集）"""
    print("加载训练数据...")
    data = np.load(data_path, allow_pickle=True)
    all_img = data['img_feats']
    all_expr = data['gene_exprs']
    all_pos = data['coords']

    # 过滤数据，只保留需要的切片
    included_indices = [i for i, name in enumerate(slice_names) if name in slice_names]
    all_img = [all_img[i] for i in included_indices]
    all_expr = [all_expr[i] for i in included_indices]
    all_pos = [all_pos[i] for i in included_indices]

    # 确保训练数据是float32类型
    all_img = [img.astype(np.float32) for img in all_img]
    all_expr = [expr.astype(np.float32) for expr in all_expr]
    all_pos = [pos.astype(np.float32) for pos in all_pos]

    # 加载标签
    all_labels = []
    for s in slice_names:
        label_path = os.path.join(label_dir, f'{s}_labels.txt')
        label = load_label(label_path)
        # 确保标签是numpy数组并转换为float32
        if isinstance(label, list):
            label = np.array(label)
        all_labels.append(label.astype(np.float32))

    return all_img, all_expr, all_pos, all_labels


def load_test_data(data_path, label_path):
    """加载测试数据（Visium_HBC数据集）"""
    print("加载测试数据...")
    data = np.load(data_path, allow_pickle=True)

    # 提取测试数据（注意格式与训练数据不同）
    test_img = data['img_feats']
    test_expr = data['gene_exprs']
    test_pos = data['coords']

    # 处理测试数据的特殊格式（去掉多余的维度）
    if len(test_img.shape) == 3 and test_img.shape[0] == 1:
        test_img = test_img[0]  # 从 (1, n, f) 变为 (n, f)
    if len(test_expr.shape) == 3 and test_expr.shape[0] == 1:
        test_expr = test_expr[0]  # 从 (1, n, f) 变为 (n, f)
    if len(test_pos.shape) == 3 and test_pos.shape[0] == 1:
        test_pos = test_pos[0]  # 从 (1, n, 2) 变为 (n, 2)

    # 确保测试数据是float32类型
    test_img = test_img.astype(np.float32)
    test_expr = test_expr.astype(np.float32)
    test_pos = test_pos.astype(np.float32)

    print(f"测试数据形状 - img: {test_img.shape}, expr: {test_expr.shape}, pos: {test_pos.shape}")
    print(f"测试数据类型 - img: {test_img.dtype}, expr: {test_expr.dtype}, pos: {test_pos.dtype}")

    # 加载测试标签
    test_label = load_label(label_path)
    # 确保标签是numpy数组并转换为float32
    if isinstance(test_label, list):
        test_label = np.array(test_label)
    test_label = test_label.astype(np.float32)

    return test_img, test_expr, test_pos, test_label


def run_single_experiment(train_img, train_expr, train_pos, train_label,
                          test_img, test_expr, test_pos, test_label,
                          epochs, lr, batch_size, prefix, device, CL_output, seed,
                          train_slice_names, test_slice_name):
    """运行单次实验（训练+测试）"""
    set_all_seeds(seed)

    # 训练模型
    print(f"\n开始训练（种子: {seed}）...")
    start_train = time.time()
    model = training(
        imgs=train_img, exps=train_expr, locs=train_pos, y=train_label,
        epochs=epochs, lr=lr, batch_size=batch_size,
        prefix=prefix, device=device, n_jobs=1, CL_output=CL_output
    )
    end_train = time.time()
    train_time = end_train - start_train
    print(f'训练时间: {train_time:.2f}秒')

    # 测试模型
    print(f"开始测试...")
    start_test = time.time()
    scores = inference(model=model, imgs=test_img, exps=test_expr, locs=test_pos,
                       y=test_label, batch_size=batch_size, device=device)
    end_test = time.time()
    test_time = end_test - start_test
    print(f'测试时间: {test_time:.2f}秒')

    # 保存预测结果
    pred_dir = os.path.join("data/Visium_HBC/", "predict_files")
    os.makedirs(pred_dir, exist_ok=True)
    np.savetxt(os.path.join(pred_dir, f"{test_slice_name}_seed{seed}.txt"), scores, fmt='%s')

    # 计算评估指标
    auc, ap, f1 = evaluate(test_label, scores)
    print(f"测试结果 - AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

    return auc, ap, f1, train_time, test_time, scores


def main(args):
    # 设备参数改为字符串类型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 配置参数
    seeds = [24, 42, 123, 345, 567]  # 多个种子用于稳定性测试
    batch_size = 512
    epochs = 100
    lr = 1e-5
    CL_output = 512

    # 数据集配置
    train_data_path = "data/Visium_HBC/train_VisiumHBC_dataset.npz"  # HER2ST训练数据
    train_label_dir = "data/HER2ST/label"  # HER2ST标签目录
    train_slice_names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']  # HER2ST切片名称

    test_data_path = "data/Visium_HBC/test_VisiumHBC_dataset.npz"  # Visium_HBC测试数据
    test_label_path = "data/Visium_HBC/spot_labels.txt"  # Visium_HBC测试标签
    test_slice_name = "Visium_HBC"  # 测试切片名称

    # 记录开始时间
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"实验开始时间: {start_time}")

    # 加载训练数据（HER2ST）
    train_img, train_expr, train_pos, train_label = load_training_data(
        train_data_path, train_label_dir, train_slice_names
    )
    print(f"训练数据加载完成: {len(train_slice_names)}个切片")

    # 加载测试数据（Visium_HBC）
    test_img, test_expr, test_pos, test_label = load_test_data(
        test_data_path, test_label_path
    )
    print(f"测试数据加载完成: 1个切片")

    # 多种子实验
    all_aucs, all_aps, all_f1s = [], [], []
    all_train_times, all_test_times = [], []

    for seed in seeds:
        print(f"\n{'!' * 20} 种子={seed} 实验开始 {'!' * 20}")

        auc, ap, f1, train_time, test_time, scores = run_single_experiment(
            train_img, train_expr, train_pos, train_label,
            test_img, test_expr, test_pos, test_label,
            epochs, lr, batch_size, args.pretrain_dir, device, CL_output, seed,
            train_slice_names, test_slice_name
        )

        all_aucs.append(auc)
        all_aps.append(ap)
        all_f1s.append(f1)
        all_train_times.append(train_time)
        all_test_times.append(test_time)

        print(f"种子 {seed} 结果: AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

    # 输出结果统计
    print("\n" + "=" * 60)
    print("每个种子的单独结果:")
    print("=" * 60)
    for i, seed in enumerate(seeds):
        print(f"种子 {seed}: AUC={all_aucs[i]:.4f}, AP={all_aps[i]:.4f}, F1={all_f1s[i]:.4f}")

    # 计算整体统计
    auc_mean = np.mean(all_aucs)
    auc_std = np.std(all_aucs)
    ap_mean = np.mean(all_aps)
    ap_std = np.std(all_aps)
    f1_mean = np.mean(all_f1s)
    f1_std = np.std(all_f1s)
    train_time_mean = np.mean(all_train_times)
    test_time_mean = np.mean(all_test_times)

    # 输出整体结果
    print("\n" + "=" * 60)
    print("整体结果（均值 ± 标准差）:")
    print("=" * 60)
    print(f"AUC:  {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"AP:   {ap_mean:.4f} ± {ap_std:.4f}")
    print(f"F1:   {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"平均训练时间: {train_time_mean:.2f}秒")
    print(f"平均测试时间: {test_time_mean:.2f}秒")

    # 保存详细结果到CSV
    results_df = pd.DataFrame({
        'Seed': seeds,
        'AUC': all_aucs,
        'AP': all_aps,
        'F1': all_f1s,
        'Train_Time': all_train_times,
        'Test_Time': all_test_times
    })

    # 添加统计行
    stats_row = pd.DataFrame({
        'Seed': ['Mean±Std'],
        'AUC': [f'{auc_mean:.4f}±{auc_std:.4f}'],
        'AP': [f'{ap_mean:.4f}±{ap_std:.4f}'],
        'F1': [f'{f1_mean:.4f}±{f1_std:.4f}'],
        'Train_Time': [f'{train_time_mean:.2f}'],
        'Test_Time': [f'{test_time_mean:.2f}']
    })
    results_df = pd.concat([results_df, stats_row], ignore_index=True)

    results_dir = os.path.join("data/Visium_HBC/", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_HER2ST_to_VisiumHBC_{start_time}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"结果已保存至: {results_file}")

    print(f"峰值内存使用: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dir', type=str, default='data/HER2ST/')
    args = parser.parse_args()

    main(args)
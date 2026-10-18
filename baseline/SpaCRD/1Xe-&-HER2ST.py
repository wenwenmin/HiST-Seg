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

    imgs = normalize(imgs)

    model.eval()

    # 强制重置 scores
    try:
        model.scores = []  # 尝试直接重置
    except:
        # 如果失败，创建一个新的属性
        if hasattr(model, 'scores'):
            delattr(model, 'scores')
        model.scores = []

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
    """加载训练数据（现在使用Xenium_HBC数据集）"""
    print("加载训练数据...")
    data = np.load(data_path, allow_pickle=True)

    # 处理训练数据的特殊格式（去掉多余的维度）
    train_img = data['img_feats']
    train_expr = data['gene_expr']
    train_pos = data['coords']

    if len(train_img.shape) == 3 and train_img.shape[0] == 1:
        train_img = train_img[0]  # 从 (1, n, f) 变为 (n, f)
    if len(train_expr.shape) == 3 and train_expr.shape[0] == 1:
        train_expr = train_expr[0]  # 从 (1, n, f) 变为 (n, f)
    if len(train_pos.shape) == 3 and train_pos.shape[0] == 1:
        train_pos = train_pos[0]  # 从 (1, n, 2) 变为 (n, 2)

    # 确保训练数据是float32类型
    train_img = train_img.astype(np.float32)
    train_expr = train_expr.astype(np.float32)
    train_pos = train_pos.astype(np.float32)

    # 将单个切片包装成列表格式（与原来训练函数期望的格式一致）
    all_img = [train_img]
    all_expr = [train_expr]
    all_pos = [train_pos]

    print(f"训练数据形状 - img: {train_img.shape}, expr: {train_expr.shape}, pos: {train_pos.shape}")

    # 加载标签
    all_labels = []
    for s in slice_names:
        label_path = "data/Xenium_HBC/label.txt"
        if os.path.exists(label_path):
            label = load_label(label_path)
        else:
            # 如果标签文件不存在，尝试直接加载标签文件
            label_path = label_dir
            label = load_label(label_path)

        # 确保标签是numpy数组并转换为float32
        if isinstance(label, list):
            label = np.array(label)
        # 将单个标签包装成列表
        all_labels.append(label.astype(np.float32))

    return all_img, all_expr, all_pos, all_labels


def load_test_data(data_path, label_dir, slice_names):
    """加载测试数据（现在使用HER2ST数据集，保持8个切片独立）"""
    print("加载测试数据...")
    data = np.load(data_path, allow_pickle=True)
    all_img = data['img_feats']
    all_expr = data['gene_expr']
    all_pos = data['coords']

    # 过滤数据，只保留需要的切片
    included_indices = [i for i, name in enumerate(slice_names) if name in slice_names]
    all_img = [all_img[i] for i in included_indices]
    all_expr = [all_expr[i] for i in included_indices]
    all_pos = [all_pos[i] for i in included_indices]

    # 确保测试数据是float32类型
    all_img = [img.astype(np.float32) for img in all_img]
    all_expr = [expr.astype(np.float32) for expr in all_expr]
    all_pos = [pos.astype(np.float32) for pos in all_pos]

    print(f"测试数据包含 {len(all_img)} 个切片")
    for i, (img, expr, pos) in enumerate(zip(all_img, all_expr, all_pos)):
        print(f"切片 {slice_names[i]}: img={img.shape}, expr={expr.shape}, pos={pos.shape}")

    # 加载所有标签
    all_labels = []
    for i, s in enumerate(slice_names):
        label_path = os.path.join(label_dir, f'{s}_labels.txt')
        if os.path.exists(label_path):
            label = load_label(label_path)
        else:
            # 如果文件不存在，尝试其他路径
            label_path = os.path.join(label_dir, f'{s}.txt')
            label = load_label(label_path)

        # 确保标签是numpy数组并转换为float32
        if isinstance(label, list):
            label = np.array(label)
        all_labels.append(label.astype(np.float32))
        print(f"切片 {s} 标签形状: {label.shape}")

    return all_img, all_expr, all_pos, all_labels


def run_single_experiment(train_img, train_expr, train_pos, train_label,
                          test_imgs, test_exprs, test_poss, test_labels,
                          epochs, lr, batch_size, prefix, device, CL_output, seed,
                          train_slice_names, test_slice_names):
    """运行单次实验（训练+测试8个切片）"""
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

    # 分别测试每个切片
    slice_results = {}
    slice_metrics = {}

    for i, slice_name in enumerate(test_slice_names):
        print(f"\n测试切片 {slice_name}...")
        start_test = time.time()
        scores = inference(model=model, imgs=test_imgs[i], exps=test_exprs[i], locs=test_poss[i],
                           y=test_labels[i], batch_size=batch_size, device=device)
        end_test = time.time()
        test_time = end_test - start_test
        # 保存预测结果
        pred_dir = "../../predict_files/SpaCRD/her2st/From_XeniumHBC"
        os.makedirs(pred_dir, exist_ok=True)
        np.savetxt(os.path.join(pred_dir, f"probs_seed={seed}.txt"), scores, fmt='%s')
        # 计算评估指标
        auc, ap, f1 = evaluate(test_labels[i], scores)
        print(f"切片 {slice_name} 结果 - AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}, 测试时间: {test_time:.2f}秒")

        # 保存结果
        slice_results[slice_name] = {
            'scores': scores,
            'auc': auc,
            'ap': ap,
            'f1': f1,
            'test_time': test_time
        }

        slice_metrics[slice_name] = (auc, ap, f1)

    # 计算8个切片的平均指标
    avg_auc = np.mean([metrics[0] for metrics in slice_metrics.values()])
    avg_ap = np.mean([metrics[1] for metrics in slice_metrics.values()])
    avg_f1 = np.mean([metrics[2] for metrics in slice_metrics.values()])
    total_test_time = np.sum([result['test_time'] for result in slice_results.values()])

    print(f"\n8个切片平均指标 - AUC={avg_auc:.4f}, AP={avg_ap:.4f}, F1={avg_f1:.4f}")

    return slice_results, avg_auc, avg_ap, avg_f1, train_time, total_test_time


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
    train_data_path = "../../dataset/test_dataset_XeniumHBC(UNI2-loki+expr).npz"
    train_label_dir = "data/Xenium_HBC/"
    train_slice_names = ['Xenium_HBC']

    test_data_path = "../../dataset/train_dataset_XeniumHBC(UNI2-loki+expr).npz"
    test_label_dir = "data/HER2ST/label"
    test_slice_names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    # 记录开始时间
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"实验开始时间: {start_time}")

    # 加载训练数据
    train_img, train_expr, train_pos, train_label = load_training_data(
        train_data_path, train_label_dir, train_slice_names
    )
    print(f"训练数据加载完成: {len(train_slice_names)}个切片")

    # 加载测试数据（保持8个切片独立）
    test_imgs, test_exprs, test_poss, test_labels = load_test_data(
        test_data_path, test_label_dir, test_slice_names
    )
    print(f"测试数据加载完成: {len(test_slice_names)}个独立切片")

    # 存储所有种子和所有切片的结果
    all_seeds_results = {}

    # 存储每个切片的指标历史（用于计算均值和方差）
    slice_metrics_history = {slice_name: {'aucs': [], 'aps': [], 'f1s': []}
                             for slice_name in test_slice_names}

    # 存储每轮的平均指标历史
    avg_metrics_history = {'aucs': [], 'aps': [], 'f1s': []}

    # 多种子实验
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'!' * 20} 第{seed_idx + 1}轮 种子={seed} 实验开始 {'!' * 20}")

        slice_results, avg_auc, avg_ap, avg_f1, train_time, total_test_time = run_single_experiment(
            train_img, train_expr, train_pos, train_label,
            test_imgs, test_exprs, test_poss, test_labels,
            epochs, lr, batch_size, args.pretrain_dir, device, CL_output, seed,
            train_slice_names, test_slice_names
        )

        # 保存本轮结果
        all_seeds_results[seed] = {
            'slice_results': slice_results,
            'avg_metrics': (avg_auc, avg_ap, avg_f1),
            'train_time': train_time,
            'total_test_time': total_test_time
        }

        # 收集每个切片的指标
        for slice_name in test_slice_names:
            slice_metrics_history[slice_name]['aucs'].append(slice_results[slice_name]['auc'])
            slice_metrics_history[slice_name]['aps'].append(slice_results[slice_name]['ap'])
            slice_metrics_history[slice_name]['f1s'].append(slice_results[slice_name]['f1'])

        # 收集平均指标
        avg_metrics_history['aucs'].append(avg_auc)
        avg_metrics_history['aps'].append(avg_ap)
        avg_metrics_history['f1s'].append(avg_f1)

        # 输出本轮详细结果
        print(f"\n第{seed_idx + 1}轮种子 {seed} 详细结果:")
        print("-" * 60)
        for slice_name in test_slice_names:
            auc = slice_results[slice_name]['auc']
            ap = slice_results[slice_name]['ap']
            f1 = slice_results[slice_name]['f1']
            print(f"切片 {slice_name}: AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")
        print(f"8个切片平均: AUC={avg_auc:.4f}, AP={avg_ap:.4f}, F1={avg_f1:.4f}")

    # 计算并输出最终统计结果
    print("\n" + "=" * 80)
    print("最终统计结果（5轮随机种子）")
    print("=" * 80)

    # 输出每个切片的统计
    print("\n各切片统计结果（均值 ± 标准差）:")
    print("-" * 80)
    for slice_name in test_slice_names:
        auc_mean = np.mean(slice_metrics_history[slice_name]['aucs'])
        auc_std = np.std(slice_metrics_history[slice_name]['aucs'])
        ap_mean = np.mean(slice_metrics_history[slice_name]['aps'])
        ap_std = np.std(slice_metrics_history[slice_name]['aps'])
        f1_mean = np.mean(slice_metrics_history[slice_name]['f1s'])
        f1_std = np.std(slice_metrics_history[slice_name]['f1s'])

        print(f"切片 {slice_name}:")
        print(f"  AUC:  {auc_mean:.4f} ± {auc_std:.4f}")
        print(f"  AP:   {ap_mean:.4f} ± {ap_std:.4f}")
        print(f"  F1:   {f1_mean:.4f} ± {f1_std:.4f}")

    # 输出8个切片平均指标的统计
    print("\n8个切片平均指标统计（均值 ± 标准差）:")
    print("-" * 80)
    avg_auc_mean = np.mean(avg_metrics_history['aucs'])
    avg_auc_std = np.std(avg_metrics_history['aucs'])
    avg_ap_mean = np.mean(avg_metrics_history['aps'])
    avg_ap_std = np.std(avg_metrics_history['aps'])
    avg_f1_mean = np.mean(avg_metrics_history['f1s'])
    avg_f1_std = np.std(avg_metrics_history['f1s'])

    print(f"平均 AUC:  {avg_auc_mean:.4f} ± {avg_auc_std:.4f}")
    print(f"平均 AP:   {avg_ap_mean:.4f} ± {avg_ap_std:.4f}")
    print(f"平均 F1:   {avg_f1_mean:.4f} ± {avg_f1_std:.4f}")

    # 保存详细结果到文件
    save_detailed_results(all_seeds_results, slice_metrics_history, avg_metrics_history, test_slice_names)


def save_detailed_results(all_seeds_results, slice_metrics_history, avg_metrics_history, test_slice_names):
    """保存详细结果到CSV文件"""
    results_dir = "../../predict_files/SpaCRD/her2st/"
    os.makedirs(results_dir, exist_ok=True)

    # 保存每个种子的详细结果
    detailed_data = []
    for seed, results in all_seeds_results.items():
        for slice_name in test_slice_names:
            slice_result = results['slice_results'][slice_name]
            detailed_data.append({
                'seed': seed,
                'slice': slice_name,
                'auc': slice_result['auc'],
                'ap': slice_result['ap'],
                'f1': slice_result['f1'],
                'test_time': slice_result['test_time']
            })

    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(results_dir, "detailed_slice_results.csv"), index=False)

    # # 保存统计结果
    # stats_data = []
    # for slice_name in test_slice_names:
    #     stats_data.append({
    #         'slice': slice_name,
    #         'auc_mean': np.mean(slice_metrics_history[slice_name]['aucs']),
    #         'auc_std': np.std(slice_metrics_history[slice_name]['aucs']),
    #         'ap_mean': np.mean(slice_metrics_history[slice_name]['aps']),
    #         'ap_std': np.std(slice_metrics_history[slice_name]['aps']),
    #         'f1_mean': np.mean(slice_metrics_history[slice_name]['f1s']),
    #         'f1_std': np.std(slice_metrics_history[slice_name]['f1s'])
    #     })
    #
    # # 添加平均统计
    # stats_data.append({
    #     'slice': 'Average',
    #     'auc_mean': np.mean(avg_metrics_history['aucs']),
    #     'auc_std': np.std(avg_metrics_history['aucs']),
    #     'ap_mean': np.mean(avg_metrics_history['aps']),
    #     'ap_std': np.std(avg_metrics_history['aps']),
    #     'f1_mean': np.mean(avg_metrics_history['f1s']),
    #     'f1_std': np.std(avg_metrics_history['f1s'])
    # })
    #
    # stats_df = pd.DataFrame(stats_data)
    # stats_df.to_csv(os.path.join(results_dir, "statistical_results.csv"), index=False)
    #
    # print(f"\n详细结果已保存到: {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dir', type=str, default='data/Xenium_HBC/')
    args = parser.parse_args()

    main(args)
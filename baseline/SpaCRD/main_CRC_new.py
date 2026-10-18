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

        # 修复：将标签从二维压缩到一维
        if y_batch.dim() == 2 and y_batch.size(1) == 1:
            y_batch = y_batch.squeeze(1)

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


def load_training_data(data_path, label_dir):
    """加载训练数据（CRC数据集 - 12个切片）"""
    print("加载CRC训练数据...")
    data = np.load(data_path, allow_pickle=True)

    # 根据fusion_new-CRC_new.py中的训练切片名称
    all_samples = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4', 'SN84_1', 'SN84_2',
                   'SN123_1', 'SN123_2', 'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']

    all_img = []
    all_expr = []
    all_pos = []
    all_labels = []

    for i, sample in enumerate(all_samples):
        # 提取当前切片的数据
        img_feats = data['img_feats'][i]
        gene_exprs = data['gene_expr'][i]
        coords = data['coords'][i]

        # 加载CRC格式的标签
        labels_path = os.path.join(label_dir, f"{sample}-labels3.txt")
        labels = np.loadtxt(labels_path, dtype=int)

        # 应用训练集标签处理：过滤-1
        valid_indices = labels != -1
        img_feats = img_feats[valid_indices]
        gene_exprs = gene_exprs[valid_indices]
        coords = coords[valid_indices]
        labels = labels[valid_indices]

        # 训练集标签转换：0/1 -> 0 (非肿瘤区域), 2 -> 1 (肿瘤区域)
        labels = np.where(labels == 2, 1, 0)

        # 修复：确保标签是一维数组，而不是二维
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.flatten()
        elif labels.ndim == 1:
            # 已经是一维，不需要处理
            pass
        else:
            # 如果标签是多维的，压缩到一维
            labels = labels.reshape(-1)

        # 确保数据是float32类型
        img_feats = img_feats.astype(np.float32)
        gene_exprs = gene_exprs.astype(np.float32)
        coords = coords.astype(np.float32)
        labels = labels.astype(np.float32)

        all_img.append(img_feats)
        all_expr.append(gene_exprs)
        all_pos.append(coords)
        all_labels.append(labels)

        print(f"加载训练切片 {sample}: {len(img_feats)} spots, 标签形状: {labels.shape}")

    return all_img, all_expr, all_pos, all_labels


def load_test_data(data_path, label_base_path):
    """加载测试数据（CRC测试集 - 4个切片）"""
    print("加载CRC测试数据...")
    test_data = np.load(data_path, allow_pickle=True)

    # 测试集数据
    test_img = test_data['img_feats']
    test_expr = test_data['gene_expr']
    test_coords = test_data['coords']

    # 每个切片的spot数量
    spot_counts = [3313, 4174, 4007, 3902]
    total_spots = sum(spot_counts)  # 15396

    # 验证总spot数是否匹配
    if len(test_img) != total_spots:
        print(f"警告：测试集spot数量不匹配。期望: {total_spots}, 实际: {len(test_img)}")
        # 如果数量不匹配，使用实际数量重新计算
        spot_counts = [3313, 4174, 4007, 3902]  # 保持原样，但调整最后一个切片
        spot_counts[-1] = len(test_img) - sum(spot_counts[:-1])
        print(f"调整后的spot数量: {spot_counts}")

    test_datasets = []
    slice_names = ['ST-colon1', 'ST-colon2', 'ST-colon3', 'ST-colon4']

    # 根据实际spot数量划分切片
    start_idx = 0
    for i, (slice_name, spot_count) in enumerate(zip(slice_names, spot_counts)):
        end_idx = start_idx + spot_count

        # 确保不超出范围
        if end_idx > len(test_img):
            end_idx = len(test_img)

        # 提取当前切片的数据
        slice_img = test_img[start_idx:end_idx]
        slice_expr = test_expr[start_idx:end_idx]
        slice_coords = test_coords[start_idx:end_idx]

        # 加载测试集标签
        label_path = label_base_path.format(num=i + 1)
        slice_labels = np.loadtxt(label_path, dtype=int)

        # 确保标签数量与spot数量匹配
        if len(slice_labels) != len(slice_img):
            print(f"警告: 切片 {slice_name} 标签与spot数不匹配！期望: {len(slice_img)}, 实际: {len(slice_labels)}")
            # 如果数量不匹配，截取或填充标签
            if len(slice_labels) > len(slice_img):
                slice_labels = slice_labels[:len(slice_img)]
            else:
                # 填充标签（使用0填充）
                padding = np.zeros(len(slice_img) - len(slice_labels), dtype=int)
                slice_labels = np.concatenate([slice_labels, padding])

        # 修复：确保标签是一维数组，而不是二维
        if slice_labels.ndim == 2 and slice_labels.shape[1] == 1:
            slice_labels = slice_labels.flatten()
        elif slice_labels.ndim == 1:
            # 已经是一维，不需要处理
            pass
        else:
            # 如果标签是多维的，压缩到一维
            slice_labels = slice_labels.reshape(-1)

        # 测试集标签处理：0 -> 0 (非肿瘤区域), 1 -> 1 (肿瘤区域)
        # 不需要转换，直接使用

        # 确保测试数据是float32类型
        slice_img = slice_img.astype(np.float32)
        slice_expr = slice_expr.astype(np.float32)
        slice_coords = slice_coords.astype(np.float32)
        slice_labels = slice_labels.astype(np.float32)

        test_datasets.append({
            'name': slice_name,
            'img': slice_img,
            'expr': slice_expr,
            'coords': slice_coords,
            'labels': slice_labels,
            'spot_count': len(slice_img)
        })

        print(f"加载测试切片 {slice_name}: {len(slice_img)} spots, 标签形状: {slice_labels.shape}")
        start_idx = end_idx

    return test_datasets


def run_single_seed_experiment(seed, train_img, train_expr, train_pos, train_label,
                               test_datasets, batch_size, prefix, device, CL_output,
                               epochs, lr):
    """在单个随机种子下训练模型并测试所有切片"""
    set_all_seeds(seed)

    # 训练模型
    print(f"\n{'=' * 60}")
    print(f"开始训练（种子: {seed}）...")
    print(f"{'=' * 60}")
    start_train = time.time()
    model = training(
        imgs=train_img, exps=train_expr, locs=train_pos, y=train_label,
        epochs=epochs, lr=lr, batch_size=batch_size,
        prefix=prefix, device=device, n_jobs=1, CL_output=CL_output
    )
    end_train = time.time()
    train_time = end_train - start_train
    print(f'训练时间: {train_time:.2f}秒')

    # 测试所有切片
    seed_results = {}
    total_test_time = 0

    for test_data in test_datasets:
        print(f"\n开始测试切片 {test_data['name']}...")
        start_test = time.time()
        scores = inference(model=model, imgs=test_data['img'], exps=test_data['expr'],
                           locs=test_data['coords'], y=test_data['labels'],
                           batch_size=batch_size, device=device)
        end_test = time.time()
        test_time = end_test - start_test
        total_test_time += test_time
        print(f'测试时间: {test_time:.2f}秒')

        # 计算评估指标
        auc, ap, f1 = evaluate(test_data['labels'], scores)
        print(f"测试结果 - AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

        # 保存当前切片结果
        seed_results[test_data['name']] = {
            'auc': auc,
            'ap': ap,
            'f1': f1,
            'test_time': test_time,
            'scores': scores
        }

        # 保存预测结果
        pred_dir = f'../../predict_files/SpaCRD/data/CRC_new/From_CRC/seed_{seed}/'
        os.makedirs(pred_dir, exist_ok=True)
        np.savetxt(os.path.join(pred_dir, f"{test_data['name']}_probs_seed={seed}.txt"), scores, fmt='%s')

    # 计算当前种子下所有切片的平均指标
    avg_auc = np.mean([result['auc'] for result in seed_results.values()])
    avg_ap = np.mean([result['ap'] for result in seed_results.values()])
    avg_f1 = np.mean([result['f1'] for result in seed_results.values()])

    seed_results['overall'] = {
        'auc': avg_auc,
        'ap': avg_ap,
        'f1': avg_f1,
        'train_time': train_time,
        'total_test_time': total_test_time
    }

    print(f"\n种子 {seed} 总体结果:")
    print(f"平均 AUC: {avg_auc:.4f}")
    print(f"平均 AP:  {avg_ap:.4f}")
    print(f"平均 F1:  {avg_f1:.4f}")
    print(f"总训练时间: {train_time:.2f}秒")
    print(f"总测试时间: {total_test_time:.2f}秒")

    return seed_results


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

    # 数据集配置 - 使用CRC数据集
    train_data_path = "../../dataset/train_dataset_CRC_new(UNI2-loki+expr1536).npz"  # CRC训练数据
    train_label_dir = "../../data/CRC/label3"  # CRC标签目录

    test_data_path = "../../dataset/test_dataset_CRC_new(UNI2-loki+expr1536).npz"  # CRC测试数据
    test_label_base_path = "../../data/CRC_new/ST-colon{num}/spot_labels.txt"  # CRC测试标签路径模板

    # 记录开始时间
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"实验开始时间: {start_time}")

    # 加载训练数据（CRC - 12个切片）
    train_img, train_expr, train_pos, train_label = load_training_data(
        train_data_path, train_label_dir
    )
    print(f"训练数据加载完成: 12个CRC切片")

    # 加载测试数据（CRC - 4个切片）
    test_datasets = load_test_data(test_data_path, test_label_base_path)
    print(f"测试数据加载完成: {len(test_datasets)}个CRC切片")

    # 存储所有结果
    all_seed_results = {}

    # 对每个随机种子运行实验
    for seed in seeds:
        seed_results = run_single_seed_experiment(
            seed=seed,
            train_img=train_img,
            train_expr=train_expr,
            train_pos=train_pos,
            train_label=train_label,
            test_datasets=test_datasets,
            batch_size=batch_size,
            prefix=args.pretrain_dir,
            device=device,
            CL_output=CL_output,
            epochs=epochs,
            lr=lr
        )
        all_seed_results[seed] = seed_results

    # 汇总所有种子的结果
    print("\n" + "=" * 80)
    print("所有随机种子实验结果汇总")
    print("=" * 80)

    # 为每个切片和总体指标收集所有种子的结果
    slice_metrics = {}
    overall_metrics = {'auc': [], 'ap': [], 'f1': [], 'train_time': [], 'total_test_time': []}

    # 初始化切片指标字典
    for test_data in test_datasets:
        slice_metrics[test_data['name']] = {'auc': [], 'ap': [], 'f1': [], 'test_time': []}

    # 收集数据
    for seed, results in all_seed_results.items():
        # 收集切片指标
        for slice_name, metrics in results.items():
            if slice_name != 'overall':
                slice_metrics[slice_name]['auc'].append(metrics['auc'])
                slice_metrics[slice_name]['ap'].append(metrics['ap'])
                slice_metrics[slice_name]['f1'].append(metrics['f1'])
                slice_metrics[slice_name]['test_time'].append(metrics['test_time'])

        # 收集总体指标
        overall_metrics['auc'].append(results['overall']['auc'])
        overall_metrics['ap'].append(results['overall']['ap'])
        overall_metrics['f1'].append(results['overall']['f1'])
        overall_metrics['train_time'].append(results['overall']['train_time'])
        overall_metrics['total_test_time'].append(results['overall']['total_test_time'])

    # 计算每个切片的统计结果
    print("\n各切片在5次随机种子下的结果（均值±标准差）:")
    print("-" * 80)

    slice_results = []
    for slice_name, metrics in slice_metrics.items():
        auc_mean = np.mean(metrics['auc'])
        auc_std = np.std(metrics['auc'])
        ap_mean = np.mean(metrics['ap'])
        ap_std = np.std(metrics['ap'])
        f1_mean = np.mean(metrics['f1'])
        f1_std = np.std(metrics['f1'])
        test_time_mean = np.mean(metrics['test_time'])

        print(f"{slice_name}:")
        print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")
        print(f"  AP:  {ap_mean:.4f} ± {ap_std:.4f}")
        print(f"  F1:  {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"  平均测试时间: {test_time_mean:.2f}秒")

        slice_results.append({
            'Slice': slice_name,
            'AUC_Mean': auc_mean, 'AUC_Std': auc_std,
            'AP_Mean': ap_mean, 'AP_Std': ap_std,
            'F1_Mean': f1_mean, 'F1_Std': f1_std,
            'Test_Time_Mean': test_time_mean
        })

    # 计算总体统计结果
    print("\n总体指标在5次随机种子下的结果（均值±标准差）:")
    print("-" * 80)

    overall_auc_mean = np.mean(overall_metrics['auc'])
    overall_auc_std = np.std(overall_metrics['auc'])
    overall_ap_mean = np.mean(overall_metrics['ap'])
    overall_ap_std = np.std(overall_metrics['ap'])
    overall_f1_mean = np.mean(overall_metrics['f1'])
    overall_f1_std = np.std(overall_metrics['f1'])
    train_time_mean = np.mean(overall_metrics['train_time'])
    total_test_time_mean = np.mean(overall_metrics['total_test_time'])

    print(f"平均 AUC: {overall_auc_mean:.4f} ± {overall_auc_std:.4f}")
    print(f"平均 AP:  {overall_ap_mean:.4f} ± {overall_ap_std:.4f}")
    print(f"平均 F1:  {overall_f1_mean:.4f} ± {overall_f1_std:.4f}")
    print(f"平均训练时间: {train_time_mean:.2f}秒")
    print(f"平均总测试时间: {total_test_time_mean:.2f}秒")

    # 保存详细结果到CSV
    results_dir = os.path.join("../../predict_files/SpaCRD/data/CRC_new/From_CRC/", "results")
    os.makedirs(results_dir, exist_ok=True)

    # 创建详细结果DataFrame
    detailed_results = []

    # 添加每个种子的详细结果
    for seed, results in all_seed_results.items():
        for slice_name, metrics in results.items():
            if slice_name != 'overall':
                detailed_results.append({
                    'Seed': seed,
                    'Slice': slice_name,
                    'AUC': metrics['auc'],
                    'AP': metrics['ap'],
                    'F1': metrics['f1'],
                    'Test_Time': metrics['test_time']
                })
            else:
                detailed_results.append({
                    'Seed': seed,
                    'Slice': 'Overall',
                    'AUC': metrics['auc'],
                    'AP': metrics['ap'],
                    'F1': metrics['f1'],
                    'Train_Time': metrics['train_time'],
                    'Total_Test_Time': metrics['total_test_time']
                })

    detailed_df = pd.DataFrame(detailed_results)

    # 添加统计汇总行
    stats_rows = []

    # 切片统计
    for slice_result in slice_results:
        stats_rows.append({
            'Seed': 'Mean±Std',
            'Slice': slice_result['Slice'],
            'AUC': f"{slice_result['AUC_Mean']:.4f}±{slice_result['AUC_Std']:.4f}",
            'AP': f"{slice_result['AP_Mean']:.4f}±{slice_result['AP_Std']:.4f}",
            'F1': f"{slice_result['F1_Mean']:.4f}±{slice_result['F1_Std']:.4f}",
            'Test_Time': f"{slice_result['Test_Time_Mean']:.2f}"
        })

    # 总体统计
    stats_rows.append({
        'Seed': 'Mean±Std',
        'Slice': 'Overall',
        'AUC': f"{overall_auc_mean:.4f}±{overall_auc_std:.4f}",
        'AP': f"{overall_ap_mean:.4f}±{overall_ap_std:.4f}",
        'F1': f"{overall_f1_mean:.4f}±{overall_f1_std:.4f}",
        'Train_Time': f"{train_time_mean:.2f}",
        'Total_Test_Time': f"{total_test_time_mean:.2f}"
    })

    stats_df = pd.DataFrame(stats_rows)
    final_df = pd.concat([detailed_df, stats_df], ignore_index=True)

    results_file = os.path.join(results_dir, f"results_From_CRC_{start_time}.csv")
    final_df.to_csv(results_file, index=False)
    print(f"详细结果已保存至: {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dir', type=str, default='../../predict_files/SpaCRD/data/CRC_new/From_CRC/')
    args = parser.parse_args()

    main(args)
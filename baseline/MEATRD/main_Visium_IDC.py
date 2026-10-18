import argparse
import pandas as pd
import scanpy as sc
import torch
import numpy as np
import random
import dgl
from dgl.data.utils import load_graphs
from finetune import SNNet
from pretrain import pretrain
from utils import evaluate, set_all_seeds
def data_factory(data_path, data_name, is_training=True, label_path=None):
    if is_training:
        label = None
        data = load_graphs(data_path+"ref.bin")[0][0]
    else:
        with open(f"{label_path}", "r") as tf:
            lines = tf.read().split(" ")
        label = [int(i) for i in lines if i != '']
        data = load_graphs(data_path+data_name+"_tgt.bin")[0][0]
    return data, label

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='dataset/IDC/', type=str)
    parser.add_argument('--unet_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--pre_lr', default=5e-4, type=float)
    parser.add_argument('--fine_epochs', default=[10,5], type=list)
    parser.add_argument('--fine_lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()

    data, _ = data_factory(args.data_path, 'Visium_IDC', True)
    #pretrain
    pretrain(data, unet_epochs=args.unet_epochs,
             batch_size=args.batch_size,
             learning_rate=args.pre_lr, Mobile=True)
    #finetune
    model = SNNet(epochs=args.fine_epochs, batch_size=args.batch_size, learning_rate=args.fine_lr,
                  GPU=args.device, Mobile=True)
    model.fit(data)

    return model, args

def run_test(model, args, data_name, label_path):
    data, label = data_factory(args.data_path, data_name, False, label_path)
    score = model.predict(data)
    auc, ap, f1 = evaluate(label, score)
    print(f'Section {data_name}  AUC: {auc:5.3f} AP: {ap:5.3f} F1:{f1:5.3f}')
    return auc, ap, f1

import os
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
def save_model(model, args, seed, save_dir="saved_models"):
    """
    保存模型和训练配置
    :param model: 训练好的模型
    :param args: 训练参数
    :param seed: 随机种子
    :param save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 模型保存路径
    model_path = os.path.join(save_dir, f"model_seed_{seed}.pth")

    # 配置保存路径
    config_path = os.path.join(save_dir, f"config_seed_{seed}.json")

    # 保存模型权重
    torch.save(model.state_dict(), model_path)
    print(f"模型权重已保存至: {model_path}")

    # 保存训练配置
    config = {
        'args': args,
        'seed': seed
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"训练配置已保存至: {config_path}")

    return model_path, config_path

if __name__ == '__main__':
    # HBC IDC 通用 需要改的地方：parser.add_argument('--data_path')  sections = XXX    label_path=XXX
    seeds = [24,42,123,345,567]
    aucss, apss, f1ss = [], [], []
    for seed in seeds:
        print("====================== 开始实验，Seed={} ==========================".format(seed))
        aucs, aps, f1s = [], [], []
        set_all_seeds(seed)
        model, args = train()
        #model_path, config_path = save_model(model, args, seed, "model/")
        sections = ['Visium_IDC']
        for section in sections:
            auc, ap, f1 = run_test(model, args, section, label_path="dataset/IDC/label.txt")
            aucs.append(auc)
            aps.append(ap)
            f1s.append(f1)
        aucss.append(aucs)
        apss.append(aps)
        f1ss.append(f1s)


    aucs_array = np.array(aucss)
    auc_slice_mean = np.mean(aucs_array, axis=0)
    auc_slice_std = np.std(aucs_array, axis=0)
    auc_all_mean = np.mean(aucs_array)
    auc_all_std = np.std(aucs_array)

    aps_array = np.array(apss)
    ap_slice_mean = np.mean(aps_array, axis=0)
    ap_slice_std = np.std(aps_array, axis=0)
    ap_all_mean = np.mean(aps_array)
    ap_all_std = np.std(aps_array)

    f1s_array = np.array(f1ss)
    f1_slice_mean = np.mean(f1s_array, axis=0)
    f1_slice_std = np.std(f1s_array, axis=0)
    f1_all_mean = np.mean(f1s_array)
    f1_all_std = np.std(f1s_array)

    print("\n" + "=" * 60)
    print("Results for Paper (Mean ± Std):")
    print("=" * 60)
    print(f"AUC: {[f'{m:.3f}±{s:.3f}' for m, s in zip(auc_slice_mean, auc_slice_std)]}")
    print(f"AP: {[f'{m:.3f}±{s:.3f}' for m, s in zip(ap_slice_mean, ap_slice_std)]}")
    print(f"F1: {[f'{m:.3f}±{s:.3f}' for m, s in zip(f1_slice_mean, f1_slice_std)]}")

    print("\n" + "=" * 60)
    print("All:")
    print("=" * 60)
    print(f"AUC: {auc_all_mean:.3f} ± {auc_all_std:.3f}")
    print(f"AP:  {ap_all_mean:.3f} ± {ap_all_std:.3f}")
    print(f"F1:  {f1_all_mean:.3f} ± {f1_all_std:.3f}")
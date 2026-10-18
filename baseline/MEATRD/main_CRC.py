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
import warnings
warnings.filterwarnings("ignore")

def data_factory(data_path, data_name, is_training=True):
    if is_training:
        label = None
        data = load_graphs(data_path+"ref.bin")[0][0]
    else:
        with open(f"dataset/CRC/label/{data_name}.txt", "r") as tf:
            lines = tf.read().split(" ")
        label = [int(i) for i in lines if i != '']
        data = load_graphs(data_path+data_name+"_tgt.bin")[0][0]
    return data, label

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='dataset/CRC/', type=str)
    parser.add_argument('--unet_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--pre_lr', default=5e-4, type=float)
    parser.add_argument('--fine_epochs', default=[10,5], type=list)
    parser.add_argument('--fine_lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()

    data, _ = data_factory(args.data_path, None, True)
    #pretrain
    pretrain(data, unet_epochs=args.unet_epochs,
             batch_size=args.batch_size,
             learning_rate=args.pre_lr, Mobile=True)
    #finetune
    model = SNNet(epochs=args.fine_epochs, batch_size=args.batch_size, learning_rate=args.fine_lr, 
                  GPU=args.device, Mobile=True)
    model.fit(data)

    return model, args

import os
def run_test(model, args, data_name):
    data, label = data_factory(args.data_path, data_name, False)
    score = model.predict(data)

    output_str = ' '.join([f'{x:.4f}' for x in score])
    os.makedirs("../../predict_files/MEATRD/CRC", exist_ok=True)
    with open(f'../../predict_files/MEATRD/CRC/{data_name}_scores_seed={seed}.txt', 'w') as f:
        f.write(output_str)

    auc, ap, f1, y_pred = evaluate(label, score, pre=True)
    output_str = ' '.join(map(str, y_pred.tolist()))
    with open(f'../../predict_files/MEATRD/CRC/{data_name}_pred_seed={seed}.txt', 'w') as f:
        f.write(output_str)

    print(f'Section {data_name}  AUC: {auc:5.3f} AP: {ap:5.3f} F1:{f1:5.3f}')

    auc, ap, f1 = evaluate(label, score)
    print(f'Section {data_name}  AUC: {auc:5.3f} AP: {ap:5.3f} F1:{f1:5.3f}')
    return auc, ap, f1


if __name__ == '__main__':
    seeds = [42, 123, 456, 789, 2026]
    aucss, apss, f1ss = [], [], []
    for seed in seeds:
        aucs, aps, f1s = [], [], []
        set_all_seeds(seed)
        model, args = train()
        sections = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4', 'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2', 'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']
        for section in sections:
            auc, ap, f1 = run_test(model, args, section)
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
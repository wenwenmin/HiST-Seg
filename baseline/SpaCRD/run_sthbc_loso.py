# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
# Spot-level protocol:
# Generate spot predictions from the method-native, original-scale H&E/ST data
# (or features prepared from those data). Do not use resized_data/cropped_data
# as model inputs. Those aligned files are used only for pixel-level evaluation.
# Preserve the original AnnData observation order in every prediction file.
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
    'Implementation of get_model for this baseline experiment.'
    x = x.copy()
    
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
        batch_size=batch_size, epochs=epochs, device=device_str)  
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
        batch_size=batch_size, epochs=epochs, device=device_str)  
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
    
    device_str = device if isinstance(device, str) else device.type

    kwargs = dict(x=imgs, exp=exps, locs=locs, y=y, batch_size=batch_size, epochs=epochs, lr=lr,
                  prefix=f'{prefix}states/0/', device=device_str, CL_output=CL_output)  

    model = get_model(**kwargs)
    return model


def inference(model, imgs, exps, locs, y, batch_size, device='cuda'):
    
    device_str = device if isinstance(device, str) else device.type
    trainer = pl.Trainer(
        accelerator=device_str,  
        devices=1,
        deterministic=True
    )

    imgs = imgs.astype(np.float32)
    imgs = normalize(imgs)

    model.eval()

    dataloader = DataLoader(SpotDataset_Rec(imgs, exps, locs, y), batch_size=batch_size, shuffle=False)
    trainer.test(model, dataloader)
    results = model.scores

    min_score = results.min()
    max_score = results.max()
    results = (results - min_score) / (max_score - min_score)

    return results


def run_one_fold(train_indices, test_index, all_img, all_expr, all_pos, all_labels,
                 epoch, lr, batch_size, dir, device, CL_output, seed, slice_names):
    
    train_img = [all_img[i] for i in train_indices]
    train_expr = [all_expr[i] for i in train_indices]
    train_pos = [all_pos[i] for i in train_indices]
    train_label = [all_labels[i] for i in train_indices]

    
    start_train = time.time()
    model = training(
        imgs=train_img, exps=train_expr, locs=train_pos, y=train_label,
        epochs=epoch, lr=lr, batch_size=batch_size,
        prefix=dir, device=device, n_jobs=1, CL_output=CL_output
    )
    end_train = time.time()
    train_time = end_train - start_train
    print(f'Training time:{train_time:.2f}Second')

    
    test_img = all_img[test_index]
    test_expr = all_expr[test_index]
    test_pos = all_pos[test_index]
    test_label = all_labels[test_index]

    
    start_test = time.time()
    scores = inference(model=model, imgs=test_img, exps=test_expr, locs=test_pos,
                       y=test_label, batch_size=batch_size, device=device)
    end_test = time.time()
    test_time = end_test - start_test
    print(f'Test time:{test_time:.2f}Second')

    
    test_name = slice_names[test_index]
    pred_dir = os.path.join(dir, "predict_files")
    os.makedirs(pred_dir, exist_ok=True)
    np.savetxt(os.path.join(pred_dir, f"{test_name}.txt"), scores, fmt='%s')

    
    auc, ap, f1 = evaluate(test_label, scores)
    print(f"test slice{test_name}: AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

    return auc, ap, f1, train_time, test_time


def leave_one_slice_out_cv(all_img, all_expr, all_pos, all_labels, slice_names,
                           epoch, lr, batch_size, dir, device, CL_output, seed):
    set_all_seeds(seed)
    num_slices = len(slice_names)
    fold_metrics = []

    for test_idx in range(num_slices):
        print(f"\n========== Leave-one-out cross-validation: test slices{slice_names[test_idx]} ==========")

        
        train_indices = [i for i in range(num_slices) if i != test_idx]

        
        auc, ap, f1, train_time, test_time = run_one_fold(
            train_indices, test_idx, all_img, all_expr, all_pos, all_labels,
            epoch, lr, batch_size, dir, device, CL_output, seed, slice_names
        )

        
        fold_metrics.append({
            'sample': slice_names[test_idx],
            'auc': auc,
            'ap': ap,
            'f1': f1,
            'train_time': train_time,
            'test_time': test_time
        })

    return fold_metrics


def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Equipment used:{device}")

    
    seeds = [24,42,123,345,567]
    batch_size = 512
    epochs = 100
    lr = 1e-5
    CL_output = 512
    data_path = os.path.join(args.pretrain_dir, 'train_dataset.npz')

    
    all_slices = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']

    
    exclude_slices = []
    included_slices = [s for s in all_slices if s not in exclude_slices]
    included_indices = [all_slices.index(s) for s in included_slices]
    print(f"All slices:{all_slices}")
    print(f"Exclude slices:{exclude_slices}")
    print(f"Slices used for cross-validation:{included_slices}")
    print(f"common{len(included_slices)}slices will be used for leave-one-out cross-validation")

    
    print('Loading data...')
    data = np.load(data_path, allow_pickle=True)
    all_img = data['img_feats']
    all_expr = data['gene_exprs']
    all_pos = data['coords']

    
    all_img = [all_img[i] for i in included_indices]
    all_expr = [all_expr[i] for i in included_indices]
    all_pos = [all_pos[i] for i in included_indices]

    
    all_labels = []
    for s in included_slices:
        label = load_label(f'{args.pretrain_dir}label/{s}_labels.txt')
        all_labels.append(np.array(label))

    
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Experiment start time:{start_time}")

    
    all_aucs, all_aps, all_f1s = [], [], []
    for seed in seeds:
        print(f"\n{'!' * 20}seed ={seed}Cross validation begins{'!' * 20}")
        fold_metrics = leave_one_slice_out_cv(
            all_img, all_expr, all_pos, all_labels, included_slices,
            epochs, lr, batch_size, args.pretrain_dir, device, CL_output, seed
        )

        
        aucs = [m['auc'] for m in fold_metrics]
        aps = [m['ap'] for m in fold_metrics]
        f1s = [m['f1'] for m in fold_metrics]

        all_aucs.append(aucs)
        all_aps.append(aps)
        all_f1s.append(f1s)

        
        print(f"\n seeds{seed}result:")
        for i, sample in enumerate(included_slices):
            print(f"  {sample}: AUC={all_aucs[-1][i]:.4f}, AP={all_aps[-1][i]:.4f}, F1={all_f1s[-1][i]:.4f}")

    
    print("\n" + "=" * 60)
    print('Individual results for each seed:')
    print("=" * 60)
    for i, seed in enumerate(seeds):
        print(
            f"seed{seed:4d}: AUC={[f'{x:.4f}' for x in all_aucs[i]]}, \n"
            f"AP={[f'{x:.4f}' for x in all_aps[i]]},\n "
            f"F1={[f'{x:.4f}' for x in all_f1s[i]]}"
        )

    
    aucs_array = np.array(all_aucs)
    auc_slice_mean = np.mean(aucs_array, axis=0)
    auc_slice_std = np.std(aucs_array, axis=0)

    aps_array = np.array(all_aps)
    ap_slice_mean = np.mean(aps_array, axis=0)
    ap_slice_std = np.std(aps_array, axis=0)

    f1s_array = np.array(all_f1s)
    f1_slice_mean = np.mean(f1s_array, axis=0)
    f1_slice_std = np.std(f1s_array, axis=0)

    
    auc_all_mean = np.mean(aucs_array)
    auc_all_std = np.std(aucs_array)
    ap_all_mean = np.mean(aps_array)
    ap_all_std = np.std(aps_array)
    f1_all_mean = np.mean(f1s_array)
    f1_all_std = np.std(f1s_array)

    
    print("\n" + "=" * 60)
    print('Paper uses results (mean ± standard deviation) per sample:')
    print("=" * 60)
    for i, sample in enumerate(included_slices):
        print(
            f"{sample}: AUC={auc_slice_mean[i]:.4f}±{auc_slice_std[i]:.4f}, "
            f"AP={ap_slice_mean[i]:.4f}±{ap_slice_std[i]:.4f}, "
            f"F1={f1_slice_mean[i]:.4f}±{f1_slice_std[i]:.4f}"
        )

    
    print("\n" + "=" * 60)
    print('Overall results:')
    print("=" * 60)
    print(f"AUC: {auc_all_mean:.3f} ± {auc_all_std:.3f}")
    print(f"AP:  {ap_all_mean:.3f} ± {ap_all_std:.3f}")
    print(f"F1:  {f1_all_mean:.3f} ± {f1_all_std:.3f}")

    
    results_df = pd.DataFrame({
        'Sample': included_slices,
        'AUC_Mean': auc_slice_mean,
        'AUC_Std': auc_slice_std,
        'AP_Mean': ap_slice_mean,
        'AP_Std': ap_slice_std,
        'F1_Mean': f1_slice_mean,
        'F1_Std': f1_slice_std
    })
    results_dir = os.path.join(args.pretrain_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, f"results_HER2ST_{start_time}.csv"), index=False)
    print(f"Results have been saved to{os.path.join(results_dir, f'results_HER2ST_{start_time}.csv')}")

    print(f"Peak memory usage:{torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dir', type=str, default='data/HER2ST/')
    args = parser.parse_args()

    main(args)

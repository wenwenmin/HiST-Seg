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

    
    imgs = np.array(imgs)
    exps = np.array(exps)
    locs = np.array(locs)
    y = np.array(y)

    
    if len(imgs.shape) == 3 and imgs.shape[0] == 1:
        imgs = imgs[0]  
    if len(exps.shape) == 3 and exps.shape[0] == 1:
        exps = exps[0]  
    if len(locs.shape) == 3 and locs.shape[0] == 1:
        locs = locs[0]  

    
    imgs = imgs.astype(np.float32)
    exps = exps.astype(np.float32)
    locs = locs.astype(np.float32)
    y = y.astype(np.float32)

    print(f"Reasoning about data shape - imgs:{imgs.shape}, exps: {exps.shape}, locs: {locs.shape}, y: {y.shape}")
    print(f"Inferred data types - imgs:{imgs.dtype}, exps: {exps.dtype}, locs: {locs.dtype}, y: {y.dtype}")

    imgs = normalize(imgs)

    model.eval()

    
    def custom_collate_fn(batch):
        
        imgs_batch = np.array([item[0] for item in batch])
        exps_batch = np.array([item[1] for item in batch])
        locs_batch = np.array([item[2] for item in batch])
        y_batch = np.array([item[3] for item in batch])

        
        imgs_batch = torch.from_numpy(imgs_batch).float()
        exps_batch = torch.from_numpy(exps_batch).float()
        locs_batch = torch.from_numpy(locs_batch).float()
        y_batch = torch.from_numpy(y_batch).float()

        return imgs_batch, exps_batch, locs_batch, y_batch

    
    dataset = SpotDataset_Rec(imgs, exps, locs, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn  
    )

    
    trainer.test(model, dataloader)
    results = model.scores

    min_score = results.min()
    max_score = results.max()
    results = (results - min_score) / (max_score - min_score)

    return results


def load_training_data(data_path, label_dir, slice_names):
    'Implementation of load_training_data for this baseline experiment.'
    print('Load training data...')
    data = np.load(data_path, allow_pickle=True)
    all_img = data['img_feats']
    all_expr = data['gene_exprs']
    all_pos = data['coords']

    
    included_indices = [i for i, name in enumerate(slice_names) if name in slice_names]
    all_img = [all_img[i] for i in included_indices]
    all_expr = [all_expr[i] for i in included_indices]
    all_pos = [all_pos[i] for i in included_indices]

    
    all_img = [img.astype(np.float32) for img in all_img]
    all_expr = [expr.astype(np.float32) for expr in all_expr]
    all_pos = [pos.astype(np.float32) for pos in all_pos]

    
    all_labels = []
    for s in slice_names:
        label_path = os.path.join(label_dir, f'{s}_labels.txt')
        label = load_label(label_path)
        
        if isinstance(label, list):
            label = np.array(label)
        all_labels.append(label.astype(np.float32))

    return all_img, all_expr, all_pos, all_labels


def load_test_data(data_path, label_path):
    'Implementation of load_test_data for this baseline experiment.'
    print('Load test data...')
    data = np.load(data_path, allow_pickle=True)

    
    test_img = data['img_feats']
    test_expr = data['gene_exprs']
    test_pos = data['coords']

    
    if len(test_img.shape) == 3 and test_img.shape[0] == 1:
        test_img = test_img[0]  
    if len(test_expr.shape) == 3 and test_expr.shape[0] == 1:
        test_expr = test_expr[0]  
    if len(test_pos.shape) == 3 and test_pos.shape[0] == 1:
        test_pos = test_pos[0]  

    
    test_img = test_img.astype(np.float32)
    test_expr = test_expr.astype(np.float32)
    test_pos = test_pos.astype(np.float32)

    print(f"Test data shape - img:{test_img.shape}, expr: {test_expr.shape}, pos: {test_pos.shape}")
    print(f"Test data type - img:{test_img.dtype}, expr: {test_expr.dtype}, pos: {test_pos.dtype}")

    
    test_label = load_label(label_path)
    
    if isinstance(test_label, list):
        test_label = np.array(test_label)
    test_label = test_label.astype(np.float32)

    return test_img, test_expr, test_pos, test_label


def run_single_experiment(train_img, train_expr, train_pos, train_label,
                          test_img, test_expr, test_pos, test_label,
                          epochs, lr, batch_size, prefix, device, CL_output, seed,
                          train_slice_names, test_slice_name):
    'Implementation of run_single_experiment for this baseline experiment.'
    set_all_seeds(seed)

    
    print(f"\nStarting training (seed: {seed})...")
    start_train = time.time()
    model = training(
        imgs=train_img, exps=train_expr, locs=train_pos, y=train_label,
        epochs=epochs, lr=lr, batch_size=batch_size,
        prefix=prefix, device=device, n_jobs=1, CL_output=CL_output
    )
    end_train = time.time()
    train_time = end_train - start_train
    print(f'Training time:{train_time:.2f}Second')

    
    print(f"Start testing...")
    start_test = time.time()
    scores = inference(model=model, imgs=test_img, exps=test_expr, locs=test_pos,
                       y=test_label, batch_size=batch_size, device=device)
    end_test = time.time()
    test_time = end_test - start_test
    print(f'Test time:{test_time:.2f}Second')

    
    pred_dir = os.path.join("data/Visium_HBC/", "predict_files")
    os.makedirs(pred_dir, exist_ok=True)
    np.savetxt(os.path.join(pred_dir, f"{test_slice_name}_seed{seed}.txt"), scores, fmt='%s')

    
    auc, ap, f1 = evaluate(test_label, scores)
    print(f"Test results - AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

    return auc, ap, f1, train_time, test_time, scores


def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Equipment used:{device}")

    
    seeds = [24, 42, 123, 345, 567]  
    batch_size = 512
    epochs = 100
    lr = 1e-5
    CL_output = 512

    
    train_data_path = "data/Xenium_HBC/train_XeniumHBC_dataset.npz"  
    train_label_dir = "data/HER2ST/label"  # Configure this path for your environment; see readme_baseline.txt.
    train_slice_names = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']  

    test_data_path = "data/Xenium_HBC/test_XeniumHBC_dataset.npz"  
    test_label_path = "data/Xenium_HBC/label.txt"  
    test_slice_name = "Xenium_HBC"  

    
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Experiment start time:{start_time}")

    
    train_img, train_expr, train_pos, train_label = load_training_data(
        train_data_path, train_label_dir, train_slice_names
    )
    print(f"Training data loading is completed:{len(train_slice_names)}slices")

    
    test_img, test_expr, test_pos, test_label = load_test_data(
        test_data_path, test_label_path
    )
    print(f"Test data loading completed: 1 slice")

    
    all_aucs, all_aps, all_f1s = [], [], []
    all_train_times, all_test_times = [], []

    for seed in seeds:
        print(f"\n{'!' * 20}seed ={seed}Experiment begins{'!' * 20}")

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

        print(f"seed{seed}Result: AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

    
    print("\n" + "=" * 60)
    print('Individual results for each seed:')
    print("=" * 60)
    for i, seed in enumerate(seeds):
        print(f"seed{seed}: AUC={all_aucs[i]:.4f}, AP={all_aps[i]:.4f}, F1={all_f1s[i]:.4f}")

    
    auc_mean = np.mean(all_aucs)
    auc_std = np.std(all_aucs)
    ap_mean = np.mean(all_aps)
    ap_std = np.std(all_aps)
    f1_mean = np.mean(all_f1s)
    f1_std = np.std(all_f1s)
    train_time_mean = np.mean(all_train_times)
    test_time_mean = np.mean(all_test_times)

    
    print("\n" + "=" * 60)
    print('Overall results (mean ± standard deviation):')
    print("=" * 60)
    print(f"AUC:  {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"AP:   {ap_mean:.4f} ± {ap_std:.4f}")
    print(f"F1:   {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Average training time:{train_time_mean:.2f}Second")
    print(f"Average test time:{test_time_mean:.2f}Second")

    
    results_df = pd.DataFrame({
        'Seed': seeds,
        'AUC': all_aucs,
        'AP': all_aps,
        'F1': all_f1s,
        'Train_Time': all_train_times,
        'Test_Time': all_test_times
    })

    
    stats_row = pd.DataFrame({
        'Seed': ['Mean±Std'],
        'AUC': [f'{auc_mean:.4f}±{auc_std:.4f}'],
        'AP': [f'{ap_mean:.4f}±{ap_std:.4f}'],
        'F1': [f'{f1_mean:.4f}±{f1_std:.4f}'],
        'Train_Time': [f'{train_time_mean:.2f}'],
        'Test_Time': [f'{test_time_mean:.2f}']
    })
    results_df = pd.concat([results_df, stats_row], ignore_index=True)

    results_dir = os.path.join("data/Xenium_HBC/", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_HER2ST_to_XeniumHBC_{start_time}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results have been saved to:{results_file}")

    print(f"Peak memory usage:{torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dir', type=str, default='data/HER2ST/')
    args = parser.parse_args()

    main(args)

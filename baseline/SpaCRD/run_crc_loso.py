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

    
    try:
        model.scores = []  
    except:
        
        if hasattr(model, 'scores'):
            delattr(model, 'scores')
        model.scores = []

    
    def custom_collate_fn(batch):
        
        imgs_batch = np.array([item[0] for item in batch])
        exps_batch = np.array([item[1] for item in batch])
        locs_batch = np.array([item[2] for item in batch])
        y_batch = np.array([item[3] for item in batch])

        
        imgs_batch = torch.from_numpy(imgs_batch).float()
        exps_batch = torch.from_numpy(exps_batch).float()
        locs_batch = torch.from_numpy(locs_batch).float()
        y_batch = torch.from_numpy(y_batch).float()

        
        if y_batch.dim() == 2 and y_batch.size(1) == 1:
            y_batch = y_batch.squeeze(1)

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


def load_training_data(data_path, label_dir):
    'Implementation of load_training_data for this baseline experiment.'
    print('Loading CRC training data...')
    data = np.load(data_path, allow_pickle=True)

    
    all_samples = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4', 'SN84_1', 'SN84_2',
                   'SN123_1', 'SN123_2', 'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']

    all_img = []
    all_expr = []
    all_pos = []
    all_labels = []

    for i, sample in enumerate(all_samples):
        
        img_feats = data['img_feats'][i]
        gene_exprs = data['gene_expr'][i]
        coords = data['coords'][i]

        
        labels_path = os.path.join(label_dir, f"{sample}-labels3.txt")
        labels = np.loadtxt(labels_path, dtype=int)

        
        valid_indices = labels != -1
        img_feats = img_feats[valid_indices]
        gene_exprs = gene_exprs[valid_indices]
        coords = coords[valid_indices]
        labels = labels[valid_indices]

        
        labels = np.where(labels == 2, 1, 0)

        
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.flatten()
        elif labels.ndim == 1:
            
            pass
        else:
            
            labels = labels.reshape(-1)

        
        img_feats = img_feats.astype(np.float32)
        gene_exprs = gene_exprs.astype(np.float32)
        coords = coords.astype(np.float32)
        labels = labels.astype(np.float32)

        all_img.append(img_feats)
        all_expr.append(gene_exprs)
        all_pos.append(coords)
        all_labels.append(labels)

        print(f"Load training slices{sample}: {len(img_feats)}spots, label shape:{labels.shape}")

    return all_img, all_expr, all_pos, all_labels


def load_test_data(data_path, label_base_path):
    'Implementation of load_test_data for this baseline experiment.'
    print('Loading CRC test data...')
    test_data = np.load(data_path, allow_pickle=True)

    
    test_img = test_data['img_feats']
    test_expr = test_data['gene_expr']
    test_coords = test_data['coords']

    
    spot_counts = [3313, 4174, 4007, 3902]
    total_spots = sum(spot_counts)  # 15396

    
    if len(test_img) != total_spots:
        print(f"Warning: The number of spots in the test set does not match. expect:{total_spots}, actual:{len(test_img)}")
        
        spot_counts = [3313, 4174, 4007, 3902]  
        spot_counts[-1] = len(test_img) - sum(spot_counts[:-1])
        print(f"Adjusted spot number:{spot_counts}")

    test_datasets = []
    slice_names = ['ST-colon1', 'ST-colon2', 'ST-colon3', 'ST-colon4']

    
    start_idx = 0
    for i, (slice_name, spot_count) in enumerate(zip(slice_names, spot_counts)):
        end_idx = start_idx + spot_count

        
        if end_idx > len(test_img):
            end_idx = len(test_img)

        
        slice_img = test_img[start_idx:end_idx]
        slice_expr = test_expr[start_idx:end_idx]
        slice_coords = test_coords[start_idx:end_idx]

        
        label_path = label_base_path.format(num=i + 1)
        slice_labels = np.loadtxt(label_path, dtype=int)

        
        if len(slice_labels) != len(slice_img):
            print(f"Warning: slicing{slice_name}The label does not match the number of spots! expect:{len(slice_img)}, actual:{len(slice_labels)}")
            
            if len(slice_labels) > len(slice_img):
                slice_labels = slice_labels[:len(slice_img)]
            else:
                
                padding = np.zeros(len(slice_img) - len(slice_labels), dtype=int)
                slice_labels = np.concatenate([slice_labels, padding])

        
        if slice_labels.ndim == 2 and slice_labels.shape[1] == 1:
            slice_labels = slice_labels.flatten()
        elif slice_labels.ndim == 1:
            
            pass
        else:
            
            slice_labels = slice_labels.reshape(-1)

        
        

        
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

        print(f"Load test slices{slice_name}: {len(slice_img)}spots, label shape:{slice_labels.shape}")
        start_idx = end_idx

    return test_datasets


def run_single_seed_experiment(seed, train_img, train_expr, train_pos, train_label,
                               test_datasets, batch_size, prefix, device, CL_output,
                               epochs, lr):
    'Implementation of run_single_seed_experiment for this baseline experiment.'
    set_all_seeds(seed)

    
    print(f"\n{'=' * 60}")
    print(f"Start training (seed: {seed})...")
    print(f"{'=' * 60}")
    start_train = time.time()
    model = training(
        imgs=train_img, exps=train_expr, locs=train_pos, y=train_label,
        epochs=epochs, lr=lr, batch_size=batch_size,
        prefix=prefix, device=device, n_jobs=1, CL_output=CL_output
    )
    end_train = time.time()
    train_time = end_train - start_train
    print(f'Training time:{train_time:.2f}Second')

    
    seed_results = {}
    total_test_time = 0

    for test_data in test_datasets:
        print(f"\n starts testing slices{test_data['name']}...")
        start_test = time.time()
        scores = inference(model=model, imgs=test_data['img'], exps=test_data['expr'],
                           locs=test_data['coords'], y=test_data['labels'],
                           batch_size=batch_size, device=device)
        end_test = time.time()
        test_time = end_test - start_test
        total_test_time += test_time
        print(f'Test time:{test_time:.2f}Second')

        
        auc, ap, f1 = evaluate(test_data['labels'], scores)
        print(f"Test results - AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")

        
        seed_results[test_data['name']] = {
            'auc': auc,
            'ap': ap,
            'f1': f1,
            'test_time': test_time,
            'scores': scores
        }

        
        pred_dir = f'../../predict_files/SpaCRD/data/CRC_new/From_CRC/seed_{seed}/'
        os.makedirs(pred_dir, exist_ok=True)
        np.savetxt(os.path.join(pred_dir, f"{test_data['name']}_probs_seed={seed}.txt"), scores, fmt='%s')

    
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

    print(f"\n seeds{seed}Overall results:")
    print(f"Average AUC:{avg_auc:.4f}")
    print(f"Average AP:{avg_ap:.4f}")
    print(f"Average F1:{avg_f1:.4f}")
    print(f"Total training time:{train_time:.2f}Second")
    print(f"Total test time:{total_test_time:.2f}Second")

    return seed_results


def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Equipment used:{device}")

    
    seeds = [24, 42, 123, 345, 567]  
    batch_size = 512
    epochs = 100
    lr = 1e-5
    CL_output = 512

    
    train_data_path = "../../dataset/train_dataset_CRC_new(UNI2-loki+expr1536).npz"  
    train_label_dir = "../../data/CRC/label3"  # Configure this path for your environment; see readme_baseline.txt.

    test_data_path = "../../dataset/test_dataset_CRC_new(UNI2-loki+expr1536).npz"  
    test_label_base_path = "../../data/CRC_new/ST-colon{num}/spot_labels.txt"  # Configure this path for your environment; see readme_baseline.txt.

    
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Experiment start time:{start_time}")

    
    train_img, train_expr, train_pos, train_label = load_training_data(
        train_data_path, train_label_dir
    )
    print(f"Training data loading completed: 12 CRC slices")

    
    test_datasets = load_test_data(test_data_path, test_label_base_path)
    print(f"Test data loading completed:{len(test_datasets)}CRC slices")

    
    all_seed_results = {}

    
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

    
    print("\n" + "=" * 80)
    print('Summary of all random seed experiment results')
    print("=" * 80)

    
    slice_metrics = {}
    overall_metrics = {'auc': [], 'ap': [], 'f1': [], 'train_time': [], 'total_test_time': []}

    
    for test_data in test_datasets:
        slice_metrics[test_data['name']] = {'auc': [], 'ap': [], 'f1': [], 'test_time': []}

    
    for seed, results in all_seed_results.items():
        
        for slice_name, metrics in results.items():
            if slice_name != 'overall':
                slice_metrics[slice_name]['auc'].append(metrics['auc'])
                slice_metrics[slice_name]['ap'].append(metrics['ap'])
                slice_metrics[slice_name]['f1'].append(metrics['f1'])
                slice_metrics[slice_name]['test_time'].append(metrics['test_time'])

        
        overall_metrics['auc'].append(results['overall']['auc'])
        overall_metrics['ap'].append(results['overall']['ap'])
        overall_metrics['f1'].append(results['overall']['f1'])
        overall_metrics['train_time'].append(results['overall']['train_time'])
        overall_metrics['total_test_time'].append(results['overall']['total_test_time'])

    
    print('The results of each section under 5 random seeds (mean ± standard deviation):')
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
        print(f"Average test time:{test_time_mean:.2f}Second")

        slice_results.append({
            'Slice': slice_name,
            'AUC_Mean': auc_mean, 'AUC_Std': auc_std,
            'AP_Mean': ap_mean, 'AP_Std': ap_std,
            'F1_Mean': f1_mean, 'F1_Std': f1_std,
            'Test_Time_Mean': test_time_mean
        })

    
    print('The results of the overall index under 5 random seeds (mean ± standard deviation):')
    print("-" * 80)

    overall_auc_mean = np.mean(overall_metrics['auc'])
    overall_auc_std = np.std(overall_metrics['auc'])
    overall_ap_mean = np.mean(overall_metrics['ap'])
    overall_ap_std = np.std(overall_metrics['ap'])
    overall_f1_mean = np.mean(overall_metrics['f1'])
    overall_f1_std = np.std(overall_metrics['f1'])
    train_time_mean = np.mean(overall_metrics['train_time'])
    total_test_time_mean = np.mean(overall_metrics['total_test_time'])

    print(f"Average AUC:{overall_auc_mean:.4f} ± {overall_auc_std:.4f}")
    print(f"Average AP:{overall_ap_mean:.4f} ± {overall_ap_std:.4f}")
    print(f"Average F1:{overall_f1_mean:.4f} ± {overall_f1_std:.4f}")
    print(f"Average training time:{train_time_mean:.2f}Second")
    print(f"Average total test time:{total_test_time_mean:.2f}Second")

    
    results_dir = os.path.join("../../predict_files/SpaCRD/data/CRC_new/From_CRC/", "results")
    os.makedirs(results_dir, exist_ok=True)

    
    detailed_results = []

    
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

    
    stats_rows = []

    
    for slice_result in slice_results:
        stats_rows.append({
            'Seed': 'Mean±Std',
            'Slice': slice_result['Slice'],
            'AUC': f"{slice_result['AUC_Mean']:.4f}±{slice_result['AUC_Std']:.4f}",
            'AP': f"{slice_result['AP_Mean']:.4f}±{slice_result['AP_Std']:.4f}",
            'F1': f"{slice_result['F1_Mean']:.4f}±{slice_result['F1_Std']:.4f}",
            'Test_Time': f"{slice_result['Test_Time_Mean']:.2f}"
        })

    
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
    print(f"Detailed results have been saved to:{results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dir', type=str, default='../../predict_files/SpaCRD/data/CRC_new/From_CRC/')
    args = parser.parse_args()

    main(args)

# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
# Spot-level protocol:
# Generate spot predictions from the method-native, original-scale H&E/ST data
# (or features prepared from those data). Do not use resized_data/cropped_data
# as model inputs. Those aligned files are used only for pixel-level evaluation.
# Preserve the original AnnData observation order in every prediction file.
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
from MCGAE.model import MCGAE  
from MCGAE.utils import set_seed  
import warnings

warnings.filterwarnings("ignore")

# --------------------------
# Configure this path for your environment; see readme_baseline.txt.
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


slices = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']


patch_size = 299  
image_resize = None  


n_latent = 50  
n_clusters = 2  
fusion_mode = "holistic"  
epochs = 30  
lr = 1e-4  
weight_decay = 1e-5  
n_components = 50  


test_seeds = [24, 42, 123, 345, 567]


# --------------------------

# --------------------------
def load_data(slice_name):
    'Implementation of load_data for this baseline experiment.'
    # Configure this path for your environment; see readme_baseline.txt.
    image_path = f"../../data/HB/her2st/{slice_name}/he-raw.jpg"
    adata_path = f"../../data/HB/her2st/{slice_name}/{slice_name}.h5ad"
    label_path = f"../../data/HB/her2st/label/{slice_name}_labels.txt"

    print(f"\n is loading slices{slice_name}data...")
    print(f"Image path:{image_path}")
    print(f"Data path:{adata_path}")
    print(f"Label path:{label_path}")

    
    adata = sc.read_h5ad(adata_path)  
    adata.var_names_make_unique()
    print(f"Loaded adata: {adata.shape} (spots × genes)")
    n_spots = adata.shape[0]  

    
    labels = np.loadtxt(label_path, dtype=int)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    if len(labels) != n_spots:
        raise ValueError(f"Number of tags ({len(labels)}) and the number of spots ({n_spots}) does not match!")
    y_true = labels
    adata.obs["label"] = y_true  

    
    adata = adata[~adata.obs["label"].isna()]
    y_true = adata.obs["label"].astype(int).values
    print(f"Valid spots with labels: {len(y_true)}")

    
    spot_coords = adata.obsm['spatial']  
    print(f"Space coordinate shape:{spot_coords.shape}")

    
    pixel_coords = []
    for (x, y) in spot_coords:
        
        x_transformed = int(x)
        y_transformed = int(y)
        pixel_coords.append([x_transformed, y_transformed])
    pixel_coords = np.array(pixel_coords)
    adata.obsm["pixel_coords"] = pixel_coords

    
    img = Image.open(image_path).convert("RGB")
    if image_resize is not None:
        img = img.resize(image_resize)
    img_array = np.array(img)
    h, w, _ = img_array.shape
    print(f"Loaded image: {h}×{w} pixels")

    
    patch_list = []
    for (x, y) in pixel_coords:
        x1 = max(0, x - patch_size // 2)
        x2 = min(w, x + patch_size // 2)
        y1 = max(0, y - patch_size // 2)
        y2 = min(h, y + patch_size // 2)
        patch = img_array[y1:y2, x1:x2]
        
        pad_h = patch_size - (y2 - y1)
        pad_w = patch_size - (x2 - x1)
        patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        patch_list.append(patch)
    img_features = np.array(patch_list)
    print(f"Extracted {len(patch_list)} image patches")

    return adata, y_true, img_features


# --------------------------

# --------------------------
def preprocess_features(adata, img_features, seed):
    'Implementation of preprocess_features for this baseline experiment.'
    
    
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    expr_orig = adata.X.toarray()
    expr_orig = StandardScaler().fit_transform(expr_orig)
    adata.obsm['feat_orig'] = expr_orig  

    
    np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=0.1, size=expr_orig.shape)  
    expr_corr = expr_orig + noise
    expr_corr = StandardScaler().fit_transform(expr_corr)
    adata.obsm['feat_corr'] = expr_corr  

    
    class SimpleVAE(torch.nn.Module):
        def __init__(self, input_dim, latent_dim=100):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 200),
                torch.nn.ReLU(),
                torch.nn.Linear(200, latent_dim * 2)  
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

    
    input_dim = expr_orig.shape[1]
    vae = SimpleVAE(input_dim=input_dim, latent_dim=100).to(device)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)
    x_tensor = torch.FloatTensor(expr_orig).to(device)
    vae.train()
    for epoch in range(50):  
        x_rec, mu, logvar = vae(x_tensor)
        
        recon_loss = F.mse_loss(x_rec, x_tensor)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        loss = recon_loss + 0.001 * kl_loss
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer_vae.step()
    
    vae.eval()
    with torch.no_grad():
        x_rec, _, _ = vae(x_tensor)
    expr_vae = x_rec.cpu().numpy()
    expr_vae = StandardScaler().fit_transform(expr_vae)
    adata.obsm['feat_vae'] = expr_vae  

    
    pca = PCA(n_components=n_components, random_state=seed)
    adata.obsm['feat_pca_orig'] = pca.fit_transform(expr_orig)  
    adata.obsm['feat_pca_corr'] = pca.transform(expr_corr)  
    adata.obsm['feat_pca_vae'] = pca.transform(expr_vae)  

    
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    cnn = torch.nn.Sequential(*list(cnn.children())[:-1])  
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
    adata.obsm['X_morphology'] = img_features  

    
    adj_orig = build_spatial_adj(adata, k=10)  
    adata.obsm['adj_orig'] = adj_orig
    
    adj_aug = 1 - cdist(expr_orig, expr_orig, metric='cosine')  
    adj_aug[adj_aug < 0.5] = 0  
    adata.obsm['adj_aug'] = adj_aug.astype(np.float32)
    
    adata.obsm['graph_orig'] = adj_orig

    
    labels = adata.obs["label"].values.astype(int)  
    
    one_hot_labels = np.zeros((len(labels), n_clusters), dtype=np.float32)
    one_hot_labels[np.arange(len(labels)), labels] = 1  
    adata.obsm['label_cls'] = one_hot_labels  

    
    print(f"Gene signature (original/perturbed/VAE):{expr_orig.shape}")
    print(f"PCA features:{adata.obsm['feat_pca_orig'].shape}")
    print(f"Image features:{img_features.shape}")
    print(f"Adjacency matrix:{adj_orig.shape}")
    print(f"label_cls shape:{adata.obsm['label_cls'].shape}")
    return adata


# --------------------------

# --------------------------
def build_spatial_adj(adata, k=10):
    'Implementation of build_spatial_adj for this baseline experiment.'
    coords = adata.obsm["pixel_coords"]
    dist_matrix = cdist(coords, coords, metric="euclidean")
    adj = np.zeros_like(dist_matrix)
    for i in range(len(coords)):
        nearest_idx = np.argsort(dist_matrix[i])[1:k + 1]  
        adj[i, nearest_idx] = 1
    adj = (adj + adj.T) / 2  
    return adj.astype(np.float32)


# --------------------------

# --------------------------
def train_mcgae(adata, y_true, seed):
    'Implementation of train_mcgae for this baseline experiment.'
    
    model = MCGAE(
        adata=adata,
        n_latent=n_latent,
        n_components=n_components,
        use_pca=True,
        fusion_mode=fusion_mode,
        random_seed=seed,  
        w_morph=0.5  
    )

    
    model.train(
        max_epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        cluster_method="kmeans",  
        n_clusters=n_clusters,
        cl_start_epoch=50,  
        loss_print_interval=10,
        device=device  
    )

    
    model_output = model.get_model_output()
    y_pred_cluster = model_output['y_pred']

    
    if 'q' in model_output:
        q_tensor = model_output['q']  
        
        if isinstance(q_tensor, torch.Tensor):
            q = q_tensor.cpu().numpy()  
        else:
            q = q_tensor  
    else:
        raise ValueError("'q' is not included in the model output model_output, please check the model output logic")

    return y_pred_cluster, q


# --------------------------

# --------------------------
def evaluate(y_true, y_pred_cluster, q, n_clusters):
    'Implementation of evaluate for this baseline experiment.'
    from scipy.optimize import linear_sum_assignment

    
    confusion = np.zeros((n_clusters, n_clusters), dtype=int)
    for t, p in zip(y_true, y_pred_cluster):
        confusion[t, p] += 1
    
    row_ind, col_ind = linear_sum_assignment(-confusion)
    label_map = {col_ind[i]: row_ind[i] for i in range(n_clusters)}
    y_pred = np.array([label_map[p] for p in y_pred_cluster])

    
    f1 = f1_score(y_true, y_pred)
    
    y_score = q[:, label_map[1]] if 1 in label_map else 1 - q[:, label_map[0]]
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    print("\nEvaluation Results:")
    print(f"AUC: {auc:.4f}")
    print(f"AP: {ap:.4f}")
    print(f"F1: {f1:.4f}")
    return f1, auc, ap  


# --------------------------

# --------------------------
def calculate_stats(values):
    'Implementation of calculate_stats for this baseline experiment.'
    mean = np.mean(values)
    std = np.std(values)
    return mean, std, f"{mean:.4f} ± {std:.4f}"


# --------------------------

# --------------------------
if __name__ == "__main__":
    # Configure this path for your environment; see readme_baseline.txt.
    save_dir = "../../predict_files/MCGAE/her2st/"
    os.makedirs(save_dir, exist_ok=True)

    
    results = {}

    
    for slice_name in slices:
        print(f"\n{'=' * 50}")
        print(f"Start processing slices:{slice_name}")
        print(f"{'=' * 50}")

        
        try:
            adata, y_true, img_features = load_data(slice_name)
        except Exception as e:
            print(f"Load slices{slice_name}Data failed:{e}")
            continue

        
        slice_results = {
            'f1': [],
            'auc': [],
            'ap': []
        }

        
        for seed_idx, seed in enumerate(test_seeds):
            print(f"\n{'=' * 20}slice{slice_name}- seeds{seed} ({seed_idx + 1}/{len(test_seeds)}) {'=' * 20}")

            
            set_seed(seed)

            try:
                
                adata_processed = preprocess_features(adata.copy(), img_features, seed)

                
                y_pred_cluster, q = train_mcgae(adata_processed, y_true, seed)

                
                f1, auc, ap = evaluate(y_true, y_pred_cluster, q, n_clusters)

                
                np.savetxt(f"{save_dir}/{slice_name}_pred_seed={seed}.txt", y_pred_cluster, fmt='%d')

                
                slice_results['f1'].append(f1)
                slice_results['auc'].append(auc)
                slice_results['ap'].append(ap)

            except Exception as e:
                print(f"slice{slice_name}seed{seed}Experiment failed:{e}")
                
                slice_results['f1'].append(np.nan)
                slice_results['auc'].append(np.nan)
                slice_results['ap'].append(np.nan)

        
        results[slice_name] = slice_results

        
        print(f"\n{'=' * 30}slice{slice_name}Statistical results{'=' * 30}")
        for metric in ['auc', 'ap', 'f1']:
            values = slice_results[metric]
            
            valid_values = [v for v in values if not np.isnan(v)]
            if len(valid_values) > 0:
                mean, std, stats_str = calculate_stats(valid_values)
                print(f"{metric.upper()}: {stats_str}")
            else:
                print(f"{metric.upper()}: All experiments failed")

    # --------------------------
    
    # --------------------------
    print(f"\n{'=' * 60}")
    print(f"{' ' * 10}Summary results of five experiments on all slices")
    print(f"{'=' * 60}")

    
    summary_table = []

    
    headers = ['slice', "AUC", "AP", "F1"]
    summary_table.append(headers)

    
    all_auc_means = []
    all_ap_means = []
    all_f1_means = []

    
    for slice_name in slices:
        if slice_name not in results:
            continue

        slice_results = results[slice_name]
        row = [slice_name]

        for metric in ['auc', 'ap', 'f1']:
            values = slice_results[metric]
            
            valid_values = [v for v in values if not np.isnan(v)]
            if len(valid_values) > 0:
                mean, std, stats_str = calculate_stats(valid_values)
                row.append(stats_str)

                
                if metric == 'auc':
                    all_auc_means.append(mean)
                elif metric == 'ap':
                    all_ap_means.append(mean)
                elif metric == 'f1':
                    all_f1_means.append(mean)
            else:
                row.append('Experiment failed')

        summary_table.append(row)

    
    if all_auc_means and all_ap_means and all_f1_means:
        all_auc_mean, all_auc_std, all_auc_str = calculate_stats(all_auc_means)
        all_ap_mean, all_ap_std, all_ap_str = calculate_stats(all_ap_means)
        all_f1_mean, all_f1_std, all_f1_str = calculate_stats(all_f1_means)

        all_row = ["ALL", all_auc_str, all_ap_str, all_f1_str]
        summary_table.append(all_row)

        
        print(f"\n{'=' * 30}Summary results for all slices{'=' * 30}")
        print(f"ALL: AUC={all_auc_str}, AP={all_ap_str}, F1={all_f1_str}")

    
    print('Summary table:')
    print("-" * 80)
    for row in summary_table:
        if row[0] == 'slice':  
            print(f"{row[0]:<8} {row[1]:<20} {row[2]:<20} {row[3]:<20}")
            print("-" * 80)
        elif row[0] == "ALL":  
            print(f"{row[0]:<8} {row[1]:<20} {row[2]:<20} {row[3]:<20}")
            print("-" * 80)
        else:  
            print(f"{row[0]:<8} {row[1]:<20} {row[2]:<20} {row[3]:<20}")

    
    detailed_results = []
    for slice_name in slices:
        if slice_name not in results:
            continue

        slice_results = results[slice_name]
        for seed_idx, seed in enumerate(test_seeds):
            detailed_results.append({
                'Slice': slice_name,
                'Seed': seed,
                'AUC': slice_results['auc'][seed_idx],
                'AP': slice_results['ap'][seed_idx],
                'F1': slice_results['f1'][seed_idx]
            })

    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv(f"{save_dir}/detailed_results.csv", index=False)

    
    df_summary = pd.DataFrame(summary_table[1:], columns=summary_table[0])
    df_summary.to_csv(f"{save_dir}/summary_results.csv", index=False)

    print(f"\n detailed results have been saved to:{save_dir}/detailed_results.csv")
    print(f"Summary results have been saved to:{save_dir}/summary_results.csv")
    print(f"\n experiment completed!")

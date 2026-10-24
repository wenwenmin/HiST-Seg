# Path configuration:
# Replace the repository-relative defaults (data/, outputs/, and workspaces/)
# for your environment. See readme_baseline.txt at the baseline package root.
import os
import sys
import random
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from medpy.metric.binary import hd95
from sklearn.metrics import adjusted_rand_score

try:
    import torch
except Exception:
    torch = None

# ==========================================
# Configure this path for your environment; see readme_baseline.txt.
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
tesla_pkg_path = os.path.join(current_dir, "TESLA_package")
sys.path.insert(0, tesla_pkg_path)

from TESLA.imputation import imputation as run_imputation
from TESLA.annotation import annotation as run_annotation


# ==========================================

# ==========================================
CRC_SLICES = [
    "SN048_1", "SN048_2", "SN048_3", "SN048_4",
    "SN84_1", "SN84_2", "SN123_1", "SN123_2",
    "SN123_4", "SN124_1", "SN124_2", "SN124_4",
]

# Configure this path for your environment; see readme_baseline.txt.
DIR_HE = r"data/CRC/cropped_data/he_cropped/{slice_name}.png"
DIR_LABEL = r"data/CRC/cropped_data/label_cropped/{slice_name}.png"
DIR_TISSUE = r"data/CRC/cropped_data/tissue_mask_cropped/{slice_name}.png"
DIR_ADATA = r"data/CRC/cropped_data/h5ad_cropped/{slice_name}_cropped.h5ad"


SPATIAL_KEY = "spatial_cropped"




MARKER_GENES = [
    "EPCAM", "KRT8", "KRT18", "KRT19",
    "CEACAM5", "CEACAM6", "TACSTD2", "MUC1", "MUC2",
    "CDH17", "CLDN3", "CLDN4", "CLDN7",
    "LGALS4", "FABP1", "OLFM4", "LGR5",
]

# ==========================================

# ==========================================
RESIZE_FACTOR = 0.2
TESLA_RES = 50



USE_OTSU_CLUSTER_SELECTION = False



AUTO_FIX_SPATIAL_XY = True


RANDOM_SEEDS = [42, 123, 567]

# Configure this path for your environment; see readme_baseline.txt.
TESLA_WORKSPACE_DIR = r"workspaces/TESLA/tesla_workspace_crc_multiseed"
METHOD_NAME = "TESLA_CRC"


# ==========================================

# ==========================================
def resize_for_display(img, dsize, is_label=False):
    'Implementation of resize_for_display for this baseline experiment.'
    if is_label:
        return cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)

    src_h, src_w = img.shape[:2]
    dst_w, dst_h = dsize
    shrinking = (dst_w * dst_h) < (src_w * src_h)
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(img, dsize, interpolation=interp)


def resize_binary_mask(mask_bin, target_shape_hw):
    'Implementation of resize_binary_mask for this baseline experiment.'
    target_h, target_w = target_shape_hw
    if mask_bin.shape[:2] == (target_h, target_w):
        out = mask_bin.copy()
    else:
        out = cv2.resize(mask_bin.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8)


def calculate_metrics(pred_mask, gt_mask, tissue_mask):
    'Implementation of calculate_metrics for this baseline experiment.'
    pred = (pred_mask > 0) & (tissue_mask > 0)
    gt = (gt_mask > 0) & (tissue_mask > 0)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)

    smooth = 1e-8
    iou = tp / (tp + fp + fn + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)
    recall = tp / (tp + fn + smooth)

    try:
        if np.sum(pred) > 0 and np.sum(gt) > 0:
            hd95_val = float(hd95(pred, gt))
        else:
            hd95_val = np.nan
    except Exception:
        hd95_val = np.nan

    hd_str = "NaN" if np.isnan(hd95_val) else f"{hd95_val:.4f}"

    return {
        "Target_IoU": float(iou),
        "Dice": float(dice),
        "Recall": float(recall),
        "HD95": hd95_val,
        "HD95_str": hd_str,
    }


def ensure_bgr_image(img_path):
    'Implementation of ensure_bgr_image for this baseline experiment.'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image reading failed:{img_path}")
    return np.ascontiguousarray(img)


def ensure_gray_image(img_path, name="image"):
    'Implementation of ensure_gray_image for this baseline experiment.'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"{name}Read failed:{img_path}")
    return img


def get_crc_paths(slice_name):
    'Implementation of get_crc_paths for this baseline experiment.'
    return {
        "he": DIR_HE.format(slice_name=slice_name),
        "label": DIR_LABEL.format(slice_name=slice_name),
        "tissue": DIR_TISSUE.format(slice_name=slice_name),
        "adata": DIR_ADATA.format(slice_name=slice_name),
    }


def check_input_files(paths, slice_name, seed):
    'Implementation of check_input_files for this baseline experiment.'
    missing = [f"{k}: {v}" for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        print(f"[{slice_name} | seed={seed}] Input file missing, skip this slice:")
        for item in missing:
            print(f"  - {item}")
        return False
    return True


def set_tesla_pixel_coordinates(adata, spatial_coords, img_shape_hw, auto_fix_xy=True):
    'Implementation of set_tesla_pixel_coordinates for this baseline experiment.'
    img_h, img_w = img_shape_hw
    spatial_coords = np.asarray(spatial_coords)
    if spatial_coords.ndim != 2 or spatial_coords.shape[1] < 2:
        raise ValueError(f"Abnormal spatial coordinate dimension: shape={spatial_coords.shape}, should be at least N x 2")

    c0 = np.asarray(spatial_coords[:, 0], dtype=float)
    c1 = np.asarray(spatial_coords[:, 1], dtype=float)

    def in_range(vals, upper):
        return np.nanmin(vals) >= -5 and np.nanmax(vals) <= upper + 5

    if auto_fix_xy:
        
        looks_like_row_col = in_range(c0, img_h) and in_range(c1, img_w)
        
        looks_like_col_row = in_range(c0, img_w) and in_range(c1, img_h)

        if looks_like_col_row and not looks_like_row_col:
            adata.obs["pixel_x"] = c1  # row / y / height
            adata.obs["pixel_y"] = c0  # col / x / width
            print("-> The spatial coordinates are judged as regular (x,y)=(col,row), which have been exchanged to TESLA's pixel_x=row, pixel_y=col")
        else:
            
            adata.obs["pixel_x"] = c0
            adata.obs["pixel_y"] = c1
            if looks_like_col_row and looks_like_row_col:
                print('-> The spatial coordinate range is difficult to distinguish x/y, so keep the original order for now; if the result is overall rotated/dislocated, please manually exchange pixel_x/pixel_y')
            else:
                print('-> The spatial coordinates are written to TESLA in the original order: pixel_x=spatial[:,0], pixel_y=spatial[:,1]')
    else:
        adata.obs["pixel_x"] = c0
        adata.obs["pixel_y"] = c1

    print(
        f"  -> image H,W=({img_h},{img_w}); "
        f"pixel_x range=({adata.obs['pixel_x'].min():.1f},{adata.obs['pixel_x'].max():.1f}); "
        f"pixel_y range=({adata.obs['pixel_y'].min():.1f},{adata.obs['pixel_y'].max():.1f})"
    )

    return adata


def pred_refined_to_2d_map(pred_refined, img_shape_hw, resize_factor):
    'Implementation of pred_refined_to_2d_map for this baseline experiment.'
    img_h, img_w = img_shape_hw
    resize_w = int(img_w * resize_factor)
    resize_h = int(img_h * resize_factor)

    pred_arr = np.asarray(pred_refined).astype(np.int32)
    expected_size = resize_h * resize_w

    if pred_arr.ndim == 2:
        pred_map = pred_arr
    elif pred_arr.ndim == 1:
        if pred_arr.size != expected_size:
            raise ValueError(
                f"pred_refined length does not match, cannot reshape: got{pred_arr.size}, "
                f"expected {expected_size} = {resize_h} * {resize_w}. "
                f"Please check if the annotation's resize_factor is consistent with this."
            )
        pred_map = pred_arr.reshape(resize_h, resize_w)
    else:
        raise ValueError(f"pred_refined dimension exception: ndim={pred_arr.ndim}, shape={pred_arr.shape}")

    return pred_map.astype(np.int32)


def select_target_clusters(tesla_target_clusters, c_m, use_otsu=False):
    'Implementation of select_target_clusters for this baseline experiment.'
    if tesla_target_clusters is None:
        tesla_target_clusters = []

    if not use_otsu:
        return [int(x) for x in tesla_target_clusters]

    if c_m is None or len(c_m) == 0:
        return [int(x) for x in tesla_target_clusters]

    c_m_arr = np.array(c_m)
    if c_m_arr.ndim == 1:
        cluster_ids = np.arange(len(c_m_arr)).astype(int)
        scores = c_m_arr.astype(float)
    elif c_m_arr.ndim == 2 and c_m_arr.shape[1] >= 2:
        cluster_ids = c_m_arr[:, 0].astype(int)
        scores = c_m_arr[:, 1].astype(float)
    else:
        return [int(x) for x in tesla_target_clusters]

    if scores.max() <= scores.min():
        return [int(x) for x in tesla_target_clusters]

    scores_norm = np.uint8(255 * (scores - scores.min()) / (scores.max() - scores.min()))
    thresh_norm, _ = cv2.threshold(scores_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    real_thresh = scores.min() + (float(thresh_norm) / 255.0) * (scores.max() - scores.min())
    print(f"-> Otsu adaptive expression threshold:{real_thresh:.4f}")

    selected = cluster_ids[scores >= real_thresh].tolist()
    return [int(x) for x in selected]


def _safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def _safe_nanstd(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    
    return float(np.nanstd(values, ddof=0))


def _safe_nanvar(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    
    return float(np.nanvar(values, ddof=0))


def _format_mean_std(mean_val, std_val, digits=4):
    if np.isnan(mean_val):
        return "NaN"
    if np.isnan(std_val):
        return f"{mean_val:.{digits}f}±NaN"
    return f"{mean_val:.{digits}f}±{std_val:.{digits}f}"


def build_summary_dataframe(records, slice_order):
    'Implementation of build_summary_dataframe for this baseline experiment.'
    metric_cols = ["Target_IoU", "Dice", "Recall", "HD95"]
    raw_df = pd.DataFrame(records)

    if raw_df.empty:
        return raw_df, pd.DataFrame()

    rows = []
    for s in slice_order:
        part = raw_df[raw_df["Slice"] == s]
        if part.empty:
            continue

        row = {"Slice": s, "Num_Seeds": int(part["Seed"].nunique()) if "Seed" in part.columns else len(part)}
        for m in metric_cols:
            vals = part[m].astype(float).to_numpy()
            row[f"{m}_mean"] = _safe_nanmean(vals)
            row[f"{m}_std"] = _safe_nanstd(vals)
            row[f"{m}_var"] = _safe_nanvar(vals)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        avg_row = {"Slice": "Average", "Num_Seeds": int(raw_df["Seed"].nunique()) if "Seed" in raw_df.columns else np.nan}
        for m in metric_cols:
            slice_means = summary_df[f"{m}_mean"].astype(float).to_numpy()
            avg_row[f"{m}_mean"] = _safe_nanmean(slice_means)
            avg_row[f"{m}_std"] = _safe_nanstd(slice_means)
            avg_row[f"{m}_var"] = _safe_nanvar(slice_means)
        summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)

    return raw_df, summary_df


def summary_dataframe_to_table(summary_df, method_name="TESLA_CRC"):
    'Implementation of summary_dataframe_to_table for this baseline experiment.'
    if summary_df.empty:
        return '[Summary] Unable to generate summary table without successfully completed slicing.'

    display_rows = []
    for _, row in summary_df.iterrows():
        display_rows.append({
            "Slice": str(row["Slice"]),
            "Target_IoU": _format_mean_std(row["Target_IoU_mean"], row["Target_IoU_std"], digits=4),
            "Dice": _format_mean_std(row["Dice_mean"], row["Dice_std"], digits=4),
            "Recall": _format_mean_std(row["Recall_mean"], row["Recall_std"], digits=4),
            "HD95 (px)": _format_mean_std(row["HD95_mean"], row["HD95_std"], digits=2),
        })

    widths = {
        "Slice": max(8, max(len(r["Slice"]) for r in display_rows)),
        "Target_IoU": max(17, max(len(r["Target_IoU"]) for r in display_rows)),
        "Dice": max(17, max(len(r["Dice"]) for r in display_rows)),
        "Recall": max(17, max(len(r["Recall"]) for r in display_rows)),
        "HD95 (px)": max(14, max(len(r["HD95 (px)"]) for r in display_rows)),
    }

    def border_line():
        return (
            "|"
            + "-" * (widths["Slice"] + 2)
            + "|"
            + "-" * (widths["Target_IoU"] + 2)
            + "|"
            + "-" * (widths["Dice"] + 2)
            + "|"
            + "-" * (widths["Recall"] + 2)
            + "|"
            + "-" * (widths["HD95 (px)"] + 2)
            + "|"
        )

    def row_line(row):
        return (
            f"| {row['Slice']:<{widths['Slice']}} "
            f"| {row['Target_IoU']:<{widths['Target_IoU']}} "
            f"| {row['Dice']:<{widths['Dice']}} "
            f"| {row['Recall']:<{widths['Recall']}} "
            f"| {row['HD95 (px)']:<{widths['HD95 (px)']}} |"
        )

    title = f"📋 {method_name}- CRC slice-level multi-seed statistical summary table (mean ± standard deviation) ({method_name})"
    table_width = max(115, len(title) + 4)
    sep = "=" * table_width

    lines = [
        "",
        sep,
        title,
        sep,
        (
            f"| {'Slice':<{widths['Slice']}} "
            f"| {'Target_IoU':<{widths['Target_IoU']}} "
            f"| {'Dice':<{widths['Dice']}} "
            f"| {'Recall':<{widths['Recall']}} "
            f"| {'HD95 (px)':<{widths['HD95 (px)']}} |"
        ),
        border_line(),
    ]

    for r in display_rows:
        if r["Slice"] == "Average":
            lines.append(border_line())
        lines.append(row_line(r))

    lines.append(sep)
    return "\n".join(lines)


def save_and_print_summary(records, slice_order, metrics_save_dir, method_name="TESLA_CRC"):
    'Implementation of save_and_print_summary for this baseline experiment.'
    os.makedirs(metrics_save_dir, exist_ok=True)

    raw_df, summary_df = build_summary_dataframe(records, slice_order)

    raw_csv = os.path.join(metrics_save_dir, f"{method_name.lower()}_raw_metrics.csv")
    summary_csv = os.path.join(metrics_save_dir, f"{method_name.lower()}_summary_metrics.csv")
    table_txt = os.path.join(metrics_save_dir, f"{method_name.lower()}_summary_table.txt")

    if not raw_df.empty:
        raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    if not summary_df.empty:
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    table_str = summary_dataframe_to_table(summary_df, method_name=method_name)
    print(table_str)

    with open(table_txt, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    print('[Summary] Summary file saved:')
    print(f"  raw metrics   -> {raw_csv}")
    print(f"  summary csv   -> {summary_csv}")
    print(f"  summary table -> {table_txt}")


def set_global_seed(seed):
    'Implementation of set_global_seed for this baseline experiment.'
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def resolve_marker_genes(adata, marker_genes):
    'Implementation of resolve_marker_genes for this baseline experiment.'
    var_names = list(map(str, adata.var_names))
    exact_set = set(var_names)
    upper_to_actual = {}
    for g in var_names:
        upper_to_actual.setdefault(g.upper(), g)

    resolved = []
    seen = set()
    missing = []
    for g in marker_genes:
        if g in exact_set:
            actual = g
        else:
            actual = upper_to_actual.get(g.upper())

        if actual is None:
            missing.append(g)
            continue

        if actual not in seen:
            resolved.append(actual)
            seen.add(actual)

    if missing:
        print(f"-> The following marker genes are not in adata.var_names and have been automatically ignored:{missing}")

    return resolved


def filter_nonconstant_genes(adata_sub, genes, context="raw"):
    'Implementation of filter_nonconstant_genes for this baseline experiment.'
    kept = []
    for idx, g in enumerate(genes):
        vals = np.asarray(adata_sub.X[:, idx]).reshape(-1)
        if np.nanmax(vals) > np.nanmin(vals):
            kept.append(g)
        else:
            print(f"  -> {context}: marker gene {g}Expressed as a constant, ignored")
    return kept


def print_dataset_config():
    print("\n" + "=" * 100)
    print('TESLA CRC cropped_data multiple random seed segmentation script')
    print("=" * 100)
    print(f"Number of slices:{len(CRC_SLICES)}")
    print(f"Slice list:{CRC_SLICES}")
    print(f"Spatial coordinate key: adata.obsm['{SPATIAL_KEY}']")
    print(f"Random seed:{RANDOM_SEEDS}")
    print(f"Output directory:{TESLA_WORKSPACE_DIR}")
    print(f"marker genes: {MARKER_GENES}")
    print("=" * 100 + "\n")


def main():
    slices = CRC_SLICES
    marker_genes = MARKER_GENES

    tesla_workspace_dir = TESLA_WORKSPACE_DIR
    vis_root_dir = os.path.join(tesla_workspace_dir, "visualizations")
    mask_root_dir = os.path.join(tesla_workspace_dir, "pred_masks")
    cluster_root_dir = os.path.join(tesla_workspace_dir, "cluster_maps")
    metrics_save_dir = os.path.join(tesla_workspace_dir, "metrics")
    os.makedirs(vis_root_dir, exist_ok=True)
    os.makedirs(mask_root_dir, exist_ok=True)
    os.makedirs(cluster_root_dir, exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)

    print_dataset_config()
    metric_records = []

    for seed in RANDOM_SEEDS:
        set_global_seed(seed)
        print("\n" + "#" * 100)
        print(f"# Start random seed experiment: seed={seed}")
        print("#" * 100)

        vis_save_dir = os.path.join(vis_root_dir, f"seed_{seed}")
        mask_save_dir = os.path.join(mask_root_dir, f"seed_{seed}")
        cluster_save_dir = os.path.join(cluster_root_dir, f"seed_{seed}")
        os.makedirs(vis_save_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)
        os.makedirs(cluster_save_dir, exist_ok=True)

        for s in slices:
            print(f"\n========== Start processing CRC slices:{s} | Seed: {seed} ==========")
            paths = get_crc_paths(s)
            if not check_input_files(paths, s, seed):
                continue

            try:
                img_raw = ensure_bgr_image(paths["he"])
                img_tesla = img_raw.copy()
                img_vis = img_raw.copy()

                adata = sc.read_h5ad(paths["adata"])
                adata.var_names_make_unique()
                gt_img = ensure_gray_image(paths["label"], name="GT label")
                tissue_mask_raw = ensure_gray_image(paths["tissue"], name="Tissue mask")
            except Exception as e:
                print(f"[{s} | seed={seed}] File reading failed:{e}")
                continue

            img_h, img_w = img_tesla.shape[:2]
            gt_mask = (gt_img > 127).astype(np.uint8)
            tissue_mask_raw_bin = (tissue_mask_raw > 127).astype(np.uint8)

            
            tissue_mask_tesla = resize_binary_mask(tissue_mask_raw_bin, (img_h, img_w))
            
            tissue_mask_eval = resize_binary_mask(tissue_mask_raw_bin, gt_mask.shape[:2])

            valid_genes = resolve_marker_genes(adata, marker_genes)
            if not valid_genes:
                print(f"[{s} | seed={seed}] No valid marker genes found, skipped.")
                print(f"-> adata.var_names Example:{list(map(str, adata.var_names[:30]))}")
                continue

            adata_sub = adata[:, valid_genes].copy()
            if hasattr(adata_sub.X, "toarray"):
                adata_sub.X = adata_sub.X.toarray()

            valid_genes = filter_nonconstant_genes(adata_sub, valid_genes, context="raw")
            if not valid_genes:
                print(f"[{s} | seed={seed}] marker genes are all expressed constant in this slice and are skipped.")
                continue
            adata_sub = adata_sub[:, valid_genes].copy()

            
            if SPATIAL_KEY not in adata_sub.obsm:
                print(f"[{s} | seed={seed}] There is no ' in adata.obsm{SPATIAL_KEY}',jump over.")
                print(f"-> Currently available obsm keys:{list(adata_sub.obsm.keys())}")
                continue

            spatial_coords = np.asarray(adata_sub.obsm[SPATIAL_KEY])
            print(f"-> Use spatial coordinates: adata.obsm['{SPATIAL_KEY}']")
            adata_sub = set_tesla_pixel_coordinates(
                adata=adata_sub,
                spatial_coords=spatial_coords,
                img_shape_hw=(img_h, img_w),
                auto_fix_xy=AUTO_FIX_SPATIAL_XY,
            )

            
            contours, _ = cv2.findContours(tissue_mask_tesla, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print(f"[{s} | seed={seed}] No contour found in tissue mask, skipped.")
                continue
            cnt = max(contours, key=cv2.contourArea)

            # ---------------------------------------------------------
            
            # ---------------------------------------------------------
            print(f"[{s} | seed={seed}] Gene expression superpixel enhancement (imputation) in progress...")
            try:
                set_global_seed(seed)
                sudo_adata = run_imputation(
                    img=img_tesla,
                    raw=adata_sub,
                    cnt=cnt,
                    genes=valid_genes,
                    res=TESLA_RES,
                )
            except Exception as e:
                print(f"[{s} | seed={seed}] TESLA imputation failed:{e}")
                continue

            valid_imputed_genes = []
            for g in valid_genes:
                if g in sudo_adata.var_names:
                    g_idx = sudo_adata.var_names.get_loc(g)
                    g_vals = np.asarray(sudo_adata.X[:, g_idx]).reshape(-1)
                    if np.nanmax(g_vals) > np.nanmin(g_vals):
                        valid_imputed_genes.append(g)
                    else:
                        print(f"  -> imputed: marker gene {g}Expressed as a constant, ignored")

            if not valid_imputed_genes:
                print(f"[{s} | seed={seed}] There are no valid marker genes after imputation, so skip.")
                continue

            print(f"-> Marker genes ultimately used for TESLA annotation:{valid_imputed_genes}")
            print(f"[{s} | seed={seed}] Using CNN for joint segmentation annotation...")
            try:
                
                
                set_global_seed(seed)
                pred_refined, tesla_target_clusters, c_m = run_annotation(
                    img=img_tesla,
                    sudo_adata=sudo_adata,
                    genes=valid_imputed_genes,
                    resize_factor=RESIZE_FACTOR,
                    binary=tissue_mask_tesla,
                    res=TESLA_RES,
                    min_UMI=0,
                    rseed=seed,
                    tseed=seed,
                    nseed=seed,
                )
            except Exception as e:
                print(f"[{s} | seed={seed}] TESLA annotation failed:{e}")
                continue

            # ---------------------------------------------------------
            
            # ---------------------------------------------------------
            try:
                pred_refined_map = pred_refined_to_2d_map(
                    pred_refined=pred_refined,
                    img_shape_hw=(img_h, img_w),
                    resize_factor=RESIZE_FACTOR,
                )
            except Exception as e:
                print(f"[{s} | seed={seed}] pred_refined reshape failed:{e}")
                continue

            pred_refined_resized = resize_for_display(
                pred_refined_map,
                (gt_mask.shape[1], gt_mask.shape[0]),
                is_label=True,
            ).astype(np.int32)

            dynamic_target_clusters = select_target_clusters(
                tesla_target_clusters=tesla_target_clusters,
                c_m=c_m,
                use_otsu=USE_OTSU_CLUSTER_SELECTION,
            )
            print(f"-> TESLA original target clusters:{tesla_target_clusters}")
            print(f"-> Final selected cancer cluster IDs:{dynamic_target_clusters}")

            # ---------------------------------------------------------
            
            # ---------------------------------------------------------
            current_pred = np.zeros_like(gt_mask, dtype=np.uint8)
            for cid in dynamic_target_clusters:
                current_pred[pred_refined_resized == cid] = 1

            
            current_pred = (current_pred & tissue_mask_eval).astype(np.uint8)

            
            valid_pred_raw = pred_refined_resized[tissue_mask_eval > 0]
            valid_gt = gt_mask[tissue_mask_eval > 0]
            if valid_pred_raw.size > 0 and valid_gt.size > 0:
                ari_score = adjusted_rand_score(valid_gt, valid_pred_raw)
            else:
                ari_score = float("nan")

            metrics = calculate_metrics(current_pred, gt_mask, tissue_mask_eval)
            hd_str = metrics["HD95_str"]

            print(f"[{s} | seed={seed}] Evaluation Results:")
            print(f"  Unsupervised ARI: {ari_score:.4f}(Clustering boundary quality, for reference only)")
            print(
                f"  Target_IoU: {metrics['Target_IoU']:.4f} | "
                f"Dice: {metrics['Dice']:.4f} | "
                f"Recall: {metrics['Recall']:.4f} | "
                f"HD95: {hd_str}px\n"
            )

            
            metric_records.append({
                "Slice": s,
                "Seed": seed,
                "ARI": float(ari_score) if not np.isnan(ari_score) else np.nan,
                "Target_IoU": float(metrics["Target_IoU"]),
                "Dice": float(metrics["Dice"]),
                "Recall": float(metrics["Recall"]),
                "HD95": float(metrics["HD95"]) if not np.isnan(metrics["HD95"]) else np.nan,
            })

            # ---------------------------------------------------------
            
            # ---------------------------------------------------------
            pred_mask_path = os.path.join(mask_save_dir, f"{s}_seed{seed}_tesla_crc_pred_mask.png")
            cv2.imwrite(pred_mask_path, (current_pred * 255).astype(np.uint8))

            
            cluster_map_path = os.path.join(cluster_save_dir, f"{s}_seed{seed}_tesla_crc_cluster_map.png")
            cv2.imwrite(cluster_map_path, pred_refined_resized.astype(np.uint16))

            he_img = resize_for_display(img_vis, (gt_mask.shape[1], gt_mask.shape[0]))
            overlay = he_img.copy()
            cancer_display_mask = (current_pred == 1) & (tissue_mask_eval > 0)
            overlay[cancer_display_mask] = [0, 0, 255]

            alpha = 0.6
            vis_img = cv2.addWeighted(overlay, alpha, he_img, 1 - alpha, 0)
            vis_path = os.path.join(vis_save_dir, f"{s}_seed{seed}_tesla_crc_cancer_prediction.jpg")
            cv2.imwrite(vis_path, vis_img)

            print(f"[{s} | seed={seed}] Saved:")
            print(f"  mask       -> {pred_mask_path}")
            print(f"  cluster map-> {cluster_map_path}")
            print(f"  overlay    -> {vis_path}")

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    save_and_print_summary(
        records=metric_records,
        slice_order=slices,
        metrics_save_dir=metrics_save_dir,
        method_name=METHOD_NAME,
    )

    print('[Note] The console table displays mean±std; the summary CSV additionally contains the *_var variance column.')
    print(f"[Note] Random seed list:{RANDOM_SEEDS}")
    print(f"[Note] CRC coordinate reading key: adata.obsm['{SPATIAL_KEY}']")


if __name__ == "__main__":
    main()

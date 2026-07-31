import os
import cv2
import numpy as np
import scanpy as sc
from tqdm import tqdm


def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(file_path):
        return None
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)


def cv_imwrite(save_path, img):
    ext = os.path.splitext(save_path)[1]
    is_success, im_buf_arr = cv2.imencode(ext, img)
    if is_success:
        im_buf_arr.tofile(save_path)


def crop_and_resize_data(
    slices,
    he_dir,
    label_dir,
    tissue_mask_dir,
    h5ad_dir,
    cropped_output_dir,
    resized_output_dir,
    padding=200,
    max_size=6000,
):
    out_he_dir = os.path.join(cropped_output_dir, "he_cropped")
    out_label_dir = os.path.join(cropped_output_dir, "label_cropped")
    out_tissue_dir = os.path.join(cropped_output_dir, "tissue_mask_cropped")

    os.makedirs(out_he_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_tissue_dir, exist_ok=True)
    os.makedirs(resized_output_dir, exist_ok=True)

    for slice_name in tqdm(slices, desc="Processing Slices"):
        he_path = os.path.join(he_dir, slice_name, "he-raw.jpg")
        label_path = os.path.join(label_dir, f"{slice_name}_label.png")
        tissue_path = os.path.join(tissue_mask_dir, f"{slice_name}.png")
        h5ad_path = os.path.join(h5ad_dir, slice_name, f"{slice_name}.h5ad")

        he_img = cv_imread(he_path, cv2.IMREAD_COLOR)
        label_img = cv_imread(label_path, cv2.IMREAD_GRAYSCALE)
        tissue_img = cv_imread(tissue_path, cv2.IMREAD_GRAYSCALE)

        if he_img is None or label_img is None or tissue_img is None:
            continue

        try:
            adata = sc.read_h5ad(h5ad_path)
        except Exception:
            continue

        coords = adata.obsm['spatial']
        min_x, max_x = int(np.min(coords[:, 0])), int(np.max(coords[:, 0]))
        min_y, max_y = int(np.min(coords[:, 1])), int(np.max(coords[:, 1]))

        h, w = he_img.shape[:2]
        start_x = max(0, min_x - padding)
        end_x = min(w, max_x + padding)
        start_y = max(0, min_y - padding)
        end_y = min(h, max_y + padding)

        he_cropped = he_img[start_y:end_y, start_x:end_x]
        label_cropped = label_img[start_y:end_y, start_x:end_x]
        tissue_cropped = tissue_img[start_y:end_y, start_x:end_x]

        new_coords = coords.copy()
        new_coords[:, 0] = new_coords[:, 0] - start_x
        new_coords[:, 1] = new_coords[:, 1] - start_y

        adata_cropped = adata.copy()
        adata_cropped.obsm['spatial_cropped'] = new_coords

        cv_imwrite(os.path.join(out_he_dir, f"{slice_name}_he_cropped.jpg"), he_cropped)
        cv_imwrite(os.path.join(out_label_dir, f"{slice_name}_label_cropped.png"), label_cropped)
        cv_imwrite(os.path.join(out_tissue_dir, f"{slice_name}.png"), tissue_cropped)

        crop_h, crop_w = he_cropped.shape[:2]
        longest_side = max(crop_h, crop_w)

        if longest_side > max_size:
            scale = max_size / longest_side
        else:
            scale = 1.0

        new_w = max(1, round(crop_w * scale))
        new_h = max(1, round(crop_h * scale))

        if scale < 1.0:
            he_resized = cv2.resize(he_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
            label_resized = cv2.resize(label_cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_resized = cv2.resize(tissue_cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            he_resized = he_cropped
            label_resized = label_cropped
            mask_resized = tissue_cropped

        slice_resized_dir = os.path.join(resized_output_dir, slice_name)
        os.makedirs(slice_resized_dir, exist_ok=True)

        cv_imwrite(os.path.join(slice_resized_dir, "he_resized.png"), he_resized)
        cv_imwrite(os.path.join(slice_resized_dir, "label_resized.png"), label_resized)
        cv_imwrite(os.path.join(slice_resized_dir, "mask_resized.png"), mask_resized)

        adata_resized = adata_cropped.copy()
        spatial_resized = new_coords.copy() * scale
        adata_resized.obsm['spatial_resized'] = spatial_resized

        if 'spatial' in adata_resized.uns:
            for library_id in adata_resized.uns['spatial'].keys():
                sf = adata_resized.uns['spatial'][library_id].get('scalefactors', {})
                if 'spot_diameter_fullres' in sf:
                    sf['spot_diameter_resized'] = sf['spot_diameter_fullres'] * scale

        adata_resized.write_h5ad(os.path.join(slice_resized_dir, "adata_resized.h5ad"))

        with open(os.path.join(slice_resized_dir, "scalefactors.txt"), "w", encoding="utf-8") as f:
            f.write(f"scale_factor: {scale:.6f}\n")
            f.write(f"resized: {'yes' if scale < 1.0 else 'no'}\n")
            f.write(f"original_size_wh: {crop_w}x{crop_h}\n")
            f.write(f"resized_size_wh: {new_w}x{new_h}\n")


if __name__ == "__main__":
    slices = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
    base_dir = "./data/STHBC"

    he_dir = os.path.join(base_dir, "her2st")
    label_dir = os.path.join(base_dir, "label_img")
    tissue_mask_dir = os.path.join(base_dir, "mask")
    h5ad_dir = os.path.join(base_dir, "her2st")

    cropped_output_dir = os.path.join(base_dir, "cropped_data")
    resized_output_dir = os.path.join(base_dir, "resized_data")

    crop_and_resize_data(
        slices,
        he_dir,
        label_dir,
        tissue_mask_dir,
        h5ad_dir,
        cropped_output_dir,
        resized_output_dir,
        padding=200,
        max_size=6000,
    )
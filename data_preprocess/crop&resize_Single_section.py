import os
import cv2
import numpy as np
import scanpy as sc


def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(file_path):
        return None
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)


def cv_imwrite(save_path, img):
    ext = os.path.splitext(save_path)[1]
    is_success, im_buf_arr = cv2.imencode(ext, img)
    if is_success:
        im_buf_arr.tofile(save_path)


def process_vihbc(
    visium_dir,
    he_path,
    label_path,
    mask_path,
    cropped_dir,
    adata_cropped_path,
    resized_dir,
    adata_resized_path,
    padding=200,
    max_size=6000,
):
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(resized_dir, exist_ok=True)

    adata = sc.read_visium(visium_dir)

    coords = adata.obsm['spatial'].copy()
    min_x, max_x = int(np.min(coords[:, 0])), int(np.max(coords[:, 0]))
    min_y, max_y = int(np.min(coords[:, 1])), int(np.max(coords[:, 1]))

    he_img = cv_imread(he_path, cv2.IMREAD_COLOR)
    label_img = cv_imread(label_path, cv2.IMREAD_GRAYSCALE)
    mask_img = cv_imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if he_img is None or label_img is None or mask_img is None:
        raise FileNotFoundError("Image, label, or mask file not found.")

    if he_img.shape[:2] != mask_img.shape[:2]:
        print("Warning: Image and mask dimensions do not match.")

    # 1. Crop white borders
    h, w = he_img.shape[:2]
    start_x = max(0, min_x - padding)
    end_x = min(w, max_x + padding)
    start_y = max(0, min_y - padding)
    end_y = min(h, max_y + padding)

    he_cropped = he_img[start_y:end_y, start_x:end_x]
    label_cropped = label_img[start_y:end_y, start_x:end_x]
    mask_cropped = mask_img[start_y:end_y, start_x:end_x]

    out_he_path = os.path.join(cropped_dir, "he_cropped.jpg")
    out_label_path = os.path.join(cropped_dir, "label_cropped.png")
    out_mask_path = os.path.join(cropped_dir, "mask_cropped.png")

    cv_imwrite(out_he_path, he_cropped)
    cv_imwrite(out_label_path, label_cropped)
    cv_imwrite(out_mask_path, mask_cropped)

    new_coords = coords - np.array([start_x, start_y])
    adata_cropped = adata.copy()
    adata_cropped.obsm['spatial_cropped'] = new_coords

    save_dir = os.path.dirname(adata_cropped_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    adata_cropped.write_h5ad(adata_cropped_path)

    # 2. Resize image if max dimension exceeds max_size
    crop_h, crop_w = he_cropped.shape[:2]
    longest_side = max(crop_h, crop_w)

    if longest_side > max_size:
        scale = max_size / longest_side
    else:
        scale = 1.0

    new_w = max(1, round(crop_w * scale))
    new_h = max(1, round(crop_h * scale))

    he_resized = cv2.resize(he_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    label_resized = cv2.resize(label_cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask_resized = cv2.resize(mask_cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    out_he_resized_path = os.path.join(resized_dir, "he_resized.png")
    out_label_resized_path = os.path.join(resized_dir, "label_resized.png")
    out_mask_resized_path = os.path.join(resized_dir, "mask_resized.png")

    cv_imwrite(out_he_resized_path, he_resized)
    cv_imwrite(out_label_resized_path, label_resized)
    cv_imwrite(out_mask_resized_path, mask_resized)

    out_scalefactors_path = os.path.join(resized_dir, "scalefactors.txt")
    with open(out_scalefactors_path, "w", encoding="utf-8") as f:
        f.write(f"scale_factor: {scale:.6f}\n")
        f.write(f"original_size_wh: {crop_w}x{crop_h}\n")
        f.write(f"resized_size_wh: {new_w}x{new_h}\n")

    adata_resized = adata_cropped.copy()
    spatial_resized = new_coords.copy() * scale
    adata_resized.obsm['spatial_resized'] = spatial_resized

    if 'spatial' in adata_resized.uns:
        for library_id in adata_resized.uns['spatial'].keys():
            sf = adata_resized.uns['spatial'][library_id].get('scalefactors', {})
            if 'spot_diameter_fullres' in sf:
                sf['spot_diameter_resized'] = sf['spot_diameter_fullres'] * scale

    resized_save_dir = os.path.dirname(adata_resized_path)
    if resized_save_dir:
        os.makedirs(resized_save_dir, exist_ok=True)
    adata_resized.write_h5ad(adata_resized_path)


if __name__ == "__main__":
    data_dir = "./data/DuCIS" #ViHBC/XeHBC
    he_path = os.path.join(data_dir, "he-raw.jpg")
    label_path = os.path.join(data_dir, "cancer_label_resized.png")
    mask_path = os.path.join(data_dir, "DuCIS_mask.png")

    cropped_dir = os.path.join(data_dir, "cropped_data")
    adata_cropped_path = os.path.join(cropped_dir, "adata_cropped.h5ad")

    resized_dir = os.path.join(data_dir, "resized_data")
    adata_resized_path = os.path.join(resized_dir, "adata_resized.h5ad")

    process_vihbc(
        data_dir,
        he_path,
        label_path,
        mask_path,
        cropped_dir,
        adata_cropped_path,
        resized_dir,
        adata_resized_path,
        padding=200,
        max_size=6000,
    )
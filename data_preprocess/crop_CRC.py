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


def crop_and_align_data_crc(slices, he_dir, label_dir, tissue_mask_dir, adata_dir, output_dir, padding=200):
    out_he_dir = os.path.join(output_dir, "he_cropped")
    out_label_dir = os.path.join(output_dir, "label_cropped")
    out_tissue_dir = os.path.join(output_dir, "tissue_mask_cropped")
    out_h5ad_dir = os.path.join(output_dir, "h5ad_cropped")

    os.makedirs(out_he_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_tissue_dir, exist_ok=True)
    os.makedirs(out_h5ad_dir, exist_ok=True)

    def get_coordinate_transform(s_name):
        if s_name == "SN048_1":
            return lambda x, y: (int(x * 2.18) + 480, int(y * 2.18) + 230)
        elif s_name == "SN048_2":
            return lambda x, y: (int(x * 2.18) + 300, int(y * 2.18) + 160)
        elif s_name == "SN048_3":
            return lambda x, y: (int(x * 2.18) + 420, int(y * 2.18) + 80)
        elif s_name == "SN048_4":
            return lambda x, y: (int(x * 2.18) + 200, int(y * 2.18) + 80)
        elif s_name == "SN84_1":
            return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 200)
        elif s_name in ["SN84_2", "SN124_4"]:
            return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 240)
        elif s_name in ["SN123_1", "SN123_2", "SN123_4"]:
            return lambda x, y: (int(x * 2.18) + 270, int(y * 2.18) + 140)
        elif s_name in ["SN124_1", "SN124_2"]:
            return lambda x, y: (int(x * 2.18) + 220, int(y * 2.18) + 280)
        else:
            return lambda x, y: (int(x * 2.18), int(y * 2.18))

    for slice_name in tqdm(slices, desc="Processing CRC Slices"):
        he_path = os.path.join(he_dir, f"{slice_name}.png")
        label_path = os.path.join(label_dir, f"{slice_name}1.png")
        tissue_path = os.path.join(tissue_mask_dir, f"{slice_name}_tissue_mask1.png")
        adata_path = os.path.join(adata_dir, slice_name)

        he_img = cv_imread(he_path, cv2.IMREAD_COLOR)
        label_img = cv_imread(label_path, cv2.IMREAD_GRAYSCALE)
        tissue_img = cv_imread(tissue_path, cv2.IMREAD_GRAYSCALE)

        if he_img is None or label_img is None or tissue_img is None:
            print(f"Warning: missing image for {slice_name}, skipping.")
            continue

        try:
            adata = sc.read_visium(adata_path)
        except Exception as e:
            print(f"Warning: cannot read adata for {slice_name} ({e}), skipping.")
            continue

        raw_coords = adata.obsm['spatial']
        transform_func = get_coordinate_transform(slice_name)

        real_coords = np.zeros_like(raw_coords, dtype=np.float32)
        for i in range(len(raw_coords)):
            x, y = transform_func(raw_coords[i, 0], raw_coords[i, 1])
            real_coords[i, 0] = x
            real_coords[i, 1] = y

        min_x, max_x = int(np.min(real_coords[:, 0])), int(np.max(real_coords[:, 0]))
        min_y, max_y = int(np.min(real_coords[:, 1])), int(np.max(real_coords[:, 1]))

        h, w = he_img.shape[:2]
        start_x = max(0, min_x - padding)
        end_x = min(w, max_x + padding)
        start_y = max(0, min_y - padding)
        end_y = min(h, max_y + padding)

        he_cropped = he_img[start_y:end_y, start_x:end_x]
        label_cropped = label_img[start_y:end_y, start_x:end_x]
        tissue_cropped = tissue_img[start_y:end_y, start_x:end_x]

        new_coords = real_coords.copy()
        new_coords[:, 0] = new_coords[:, 0] - start_x
        new_coords[:, 1] = new_coords[:, 1] - start_y

        adata.obsm['spatial_cropped'] = new_coords

        cv_imwrite(os.path.join(out_he_dir, f"{slice_name}.png"), he_cropped)
        cv_imwrite(os.path.join(out_label_dir, f"{slice_name}.png"), label_cropped)
        cv_imwrite(os.path.join(out_tissue_dir, f"{slice_name}.png"), tissue_cropped)

        out_h5ad_path = os.path.join(out_h5ad_dir, f"{slice_name}_cropped.h5ad")
        adata.write_h5ad(out_h5ad_path)

        print(f"{slice_name}: original {w}x{h} -> cropped {he_cropped.shape[1]}x{he_cropped.shape[0]}")


if __name__ == "__main__":
    SLICES = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4',
              'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2',
              'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']

    HE_DIR = "./data/CRC/slices_image"
    LABEL_DIR = "./data/CRC/label"
    TISSUE_MASK_DIR = "./data/CRC/tissue_masks"
    ADATA_DIR = "./data/CRC"
    OUTPUT_DIR = "./data/CRC/cropped_data"

    crop_and_align_data_crc(SLICES, HE_DIR, LABEL_DIR, TISSUE_MASK_DIR, ADATA_DIR, OUTPUT_DIR, padding=200)

import os
import cv2
import numpy as np
import scanpy as sc
from tqdm import tqdm


# --- 必备的安全读取与保存函数（应对中文路径） ---
def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    """支持中文路径的图像读取"""
    if not os.path.exists(file_path):
        return None
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)


def cv_imwrite(save_path, img):
    """支持中文路径的图像保存"""
    ext = os.path.splitext(save_path)[1]
    is_success, im_buf_arr = cv2.imencode(ext, img)
    if is_success:
        im_buf_arr.tofile(save_path)


def crop_and_align_data_crc(slices, he_dir, label_dir, tissue_mask_dir, adata_dir, output_dir, padding=200):
    """
    针对 CRC 数据集：同步裁剪 H&E 图、标签图以及组织掩码图，并保存更新坐标后的 h5ad。
    """
    # 1. 创建输出目录结构
    out_he_dir = os.path.join(output_dir, "he_cropped")
    out_label_dir = os.path.join(output_dir, "label_cropped")
    out_tissue_dir = os.path.join(output_dir, "tissue_mask_cropped")
    out_h5ad_dir = os.path.join(output_dir, "h5ad_cropped")  # 专门存裁剪后的adata

    os.makedirs(out_he_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_tissue_dir, exist_ok=True)
    os.makedirs(out_h5ad_dir, exist_ok=True)

    # CRC特有的坐标转换函数
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
        # 2. 构建符合 CRC 命名习惯的文件路径
        he_path = os.path.join(he_dir, f"{slice_name}.png")
        label_path = os.path.join(label_dir, f"{slice_name}1.png")
        tissue_path = os.path.join(tissue_mask_dir, f"{slice_name}_tissue_mask1.png")
        adata_path = os.path.join(adata_dir, slice_name)

        # 3. 读取图像与数据
        he_img = cv_imread(he_path, cv2.IMREAD_COLOR)
        label_img = cv_imread(label_path, cv2.IMREAD_GRAYSCALE)
        tissue_img = cv_imread(tissue_path, cv2.IMREAD_GRAYSCALE)

        if he_img is None or label_img is None or tissue_img is None:
            print(f"⚠️ 警告: 找不到 {slice_name} 的某张图像，跳过。")
            continue

        try:
            # 采用 read_visium 读取原始文件夹
            adata = sc.read_visium(adata_path)
        except Exception as e:
            print(f"⚠️ 警告: 无法读取 {slice_name} 的 adata ({e})，跳过。")
            continue

        # 4. 获取原始坐标并转换为图像真实坐标
        raw_coords = adata.obsm['spatial']
        transform_func = get_coordinate_transform(slice_name)

        # 初始化一个数组存放转换后的真实像素坐标
        real_coords = np.zeros_like(raw_coords, dtype=np.float32)
        for i in range(len(raw_coords)):
            x, y = transform_func(raw_coords[i, 0], raw_coords[i, 1])
            real_coords[i, 0] = x
            real_coords[i, 1] = y

        # 基于真实像素坐标计算 Bounding Box
        min_x, max_x = int(np.min(real_coords[:, 0])), int(np.max(real_coords[:, 0]))
        min_y, max_y = int(np.min(real_coords[:, 1])), int(np.max(real_coords[:, 1]))

        # 5. 应用 Padding 并确保不越界
        h, w = he_img.shape[:2]
        start_x = max(0, min_x - padding)
        end_x = min(w, max_x + padding)
        start_y = max(0, min_y - padding)
        end_y = min(h, max_y + padding)

        # 6. 执行同步裁剪
        he_cropped = he_img[start_y:end_y, start_x:end_x]
        label_cropped = label_img[start_y:end_y, start_x:end_x]
        tissue_cropped = tissue_img[start_y:end_y, start_x:end_x]

        # 7. 坐标系平移 (Shift Coordinates) 以对齐裁剪后的图像
        # 将转换好的真实坐标减去裁剪框的左上角起点，得到相对裁剪后图像的坐标
        new_coords = real_coords.copy()
        new_coords[:, 0] = new_coords[:, 0] - start_x
        new_coords[:, 1] = new_coords[:, 1] - start_y

        # 存入 adata 备用
        adata.obsm['spatial_cropped'] = new_coords

        # 8. 保存裁剪后的数据
        cv_imwrite(os.path.join(out_he_dir, f"{slice_name}.png"), he_cropped)
        cv_imwrite(os.path.join(out_label_dir, f"{slice_name}.png"), label_cropped)
        cv_imwrite(os.path.join(out_tissue_dir, f"{slice_name}.png"), tissue_cropped)

        # 将包含新坐标系的 adata 保存为 h5ad 文件，避免破坏原始 Visium 文件夹结构
        out_h5ad_path = os.path.join(out_h5ad_dir, f"{slice_name}_cropped.h5ad")
        adata.write_h5ad(out_h5ad_path)

        print(f"✅ {slice_name}: 原尺寸 {w}x{h} -> 裁剪后 {he_cropped.shape[1]}x{he_cropped.shape[0]}")


if __name__ == "__main__":
    # CRC 样本列表
    SLICES = ['SN048_1', 'SN048_2', 'SN048_3', 'SN048_4',
              'SN84_1', 'SN84_2', 'SN123_1', 'SN123_2',
              'SN123_4', 'SN124_1', 'SN124_2', 'SN124_4']

    # --- 路径配置 (完全基于对话历史中的路径结构) ---
    HE_DIR = r"D:\DNM_project(cancer detect)\data\CRC\slices_image"
    LABEL_DIR = r"D:\Project_2\Pixel_Seg\data\CRC\label"
    TISSUE_MASK_DIR = r"D:\Project_2\Pixel_Seg\data\CRC\tissue_masks"
    ADATA_DIR = r"D:\DNM_project(cancer detect)\data\CRC"
    OUTPUT_DIR = r"D:\Project_2\Pixel_Seg\data\CRC\cropped_data"

    # 执行裁剪逻辑 (设定200像素的安全边距，防止边缘细胞被切掉)
    crop_and_align_data_crc(SLICES, HE_DIR, LABEL_DIR, TISSUE_MASK_DIR, ADATA_DIR, OUTPUT_DIR, padding=200)
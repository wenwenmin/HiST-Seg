import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import timm
from timm.data import create_transform, resolve_data_config
from PIL import Image
from huggingface_hub import login
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2

class ROIDataset(Dataset):
    def __init__(self, img_list, transform):
        super().__init__()
        self.images_lst = img_list
        self.transform = transform

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        pil_image = Image.fromarray(self.images_lst[idx].astype('uint8'))
        image = self.transform(pil_image)
        return image


class UNIExtractor:
    def __init__(self, batch_size=512, device=None):
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        login('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')  # token
        print("login huggingface success!")

        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True
        )
        self.model.eval().to(self.device)

        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    # def crop_image(self, img, x, y, crop_size):
    #     left = x - int(crop_size // 2)
    #     top = y - int(crop_size // 2)
    #     right = left + crop_size
    #     bottom = top + crop_size
    #     return img[top:bottom, left:right]

    def crop_image(self, img, x, y, crop_size):
        img_h, img_w = img.shape[:2]

        # 将坐标转换为整数
        x = int(x)
        y = int(y)

        # 计算裁剪边界
        half_size = crop_size // 2
        left = max(0, x - half_size)
        top = max(0, y - half_size)
        right = min(img_w, x + half_size)
        bottom = min(img_h, y + half_size)

        # 裁剪图像
        crop = img[top:bottom, left:right]

        # 如果裁剪尺寸不对，进行填充
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            print("crop size error! padding....")
            # 创建正确尺寸的黑色背景
            fixed_crop = np.zeros((crop_size, crop_size, 3), dtype=img.dtype)
            # 将实际裁剪的内容放到正确位置
            h, w = crop.shape[:2]
            fixed_crop[:h, :w] = crop
            return fixed_crop

        return crop

    def extract(self, img_path, spatial_coords, crop_size=300):
        img = cv2.imread(img_path)
        img = np.array(img)

        sub_images = [
            self.crop_image(img, int(x), int(y), crop_size)
            for x, y in spatial_coords
        ]

        dataset = ROIDataset(sub_images, self.transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        feature_embs = []

        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self.device)
                emb = self.model(batch)
                feature_embs.append(emb.cpu().numpy())

        feature_embs = np.concatenate(feature_embs, axis=0)
        return feature_embs
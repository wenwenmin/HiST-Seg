import os
from torch.utils.data import Dataset
from skimage import io


class DataSegmentCamelyon16(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        all_patches = sorted(os.listdir(image_dir))
        self.images = all_patches

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index]).replace(".tif", "_mask.tif")
        filenames = {'image': img_path, 'mask': mask_path}
        image = io.imread(img_path)

        mask = io.imread(mask_path)
        # 0.0 255
        # so that the range is 0 to 1
        mask[mask == 255.0] = 1.0
        augmentations = self.transform(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]

        return image, mask, filenames


class DualDataSegmentCamelyon16(Dataset):
    def __init__(self, image_dir_detail, image_dir_context ,mask_dir_detail, transform=None):
        self.image_dir_detail = image_dir_detail
        self.image_dir_context = image_dir_context
        self.mask_dir_detail = mask_dir_detail
        self.transform = transform
        # https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
        self.images_detail = sorted(os.listdir(image_dir_detail))
        print("the total number of images is ", len(self.images_detail))

    def __len__(self):
        return len(self.images_detail)

    def __getitem__(self, index):
        # img_path detail is for the level 2
        img_path_detail = os.path.join(self.image_dir_detail, self.images_detail[index])
        # the img_context has the exact same filename expect for the folder with level three
        img_path_context = os.path.join(self.image_dir_context, self.images_detail[index])
        mask_path_detail = os.path.join(self.mask_dir_detail, self.images_detail[index]).replace(".tif", "_mask.tif")

        filenames = {'image_detail': img_path_detail, 'image_context': img_path_context, 'mask_detail': mask_path_detail}
        image_detail = io.imread(img_path_detail)
        image_context = io.imread(img_path_context)

        mask_detail = io.imread(mask_path_detail)
        mask_detail[mask_detail == 255.0] = 1.0

        augmentations = self.transform(image=image_detail, image1=image_context, mask=mask_detail)
        image_detail = augmentations["image"]
        image_context = augmentations["image1"]
        mask_detail = augmentations["mask"]
        return image_detail, image_context, mask_detail, filenames

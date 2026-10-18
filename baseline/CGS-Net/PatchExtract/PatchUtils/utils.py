import glob
import os
from tqdm import tqdm
from PatchExtract.parameters import cf


def rename_patches(mask_dir, image_dir):
    """
    This is function is for renaming patches in the train_images folders for the level 3 patches. The problem is that
    the patches have the full naming convention for the level 3 mask patches but not the RGB patches (images).
    The mask_dir has the patches with the correct name.
    the image_dir has the patches with the truncated name.
    """
    mask_files = glob.glob(os.path.join(mask_dir, "*.tif"))
    image_files = glob.glob(os.path.join(image_dir, "*.tif"))

    for image in image_files:
        # the image file name is grabbed by the first spilt, the second spilt grabs everything but the file extention
        # we are finding the corresponding mask file
        image_match_mask = [s for s in mask_files if (image.split("/")[-1]).split('.')[0] in s]
        assert len(image_match_mask) == 1  # there should only be one match per patch
        # note the image_match_mask contains the fulle file path, we only want the file name without the mask part
        image_match_mask_name = (image_match_mask[0]).split("/")[-1].replace('_mask.tif', '.tif')
        # raise Exception("see comments")
        os.rename(image, os.path.join(image_dir, image_match_mask_name))


def get_slidelist():
    TUMOR = 1
    NONTUMOR = 0
    TUMOR_PATCHES_ONLY = 1
    ALL_PATCHES = 0

    tumor_list_names = glob.glob(os.path.join(cf.tumor_slide_path, "*.tif"))
    tumor_list_names = [tumorfn.split("/")[-1] for tumorfn in tumor_list_names]

    nontumor_list_names = glob.glob(os.path.join(cf.nontumor_slide_path, "*.tif"))
    nontumor_list_names = [normfn.split("/")[-1] for normfn in nontumor_list_names]
    slidelist = []

    for tumor_slide in tumor_list_names:
        # all the tumor slides that can contribute tumor patches only. Because some regions of the tumor slides
        # are tumor and not marked.
        if tumor_slide == "tumor_092.tif" or tumor_slide == "tumor_010.tif" or tumor_slide == "tumor_018.tif" or tumor_slide == "tumor_020.tif" or tumor_slide == "tumor_025.tif" or tumor_slide == "tumor_044.tif" or tumor_slide == "tumor_046.tif" or tumor_slide == "tumor_051.tif" or tumor_slide == "tumor_054.tif" or tumor_slide == "tumor_056.tif" or tumor_slide == "tumor_067.tif" or tumor_slide == "tumor_033.tif" or tumor_slide == "tumor_085.tif" or tumor_slide == "tumor_079.tif" or tumor_slide == "tumor_029.tif" or tumor_slide == "tumor_034.tif" or tumor_slide == "tumor_055.tif" or tumor_slide == "tumor_015.tif" or tumor_slide == "tumor_110.tif" or tumor_slide == "tumor_095.tif":
            slidelist.append((tumor_slide, TUMOR, TUMOR_PATCHES_ONLY))
        else:
            slidelist.append((tumor_slide, TUMOR, ALL_PATCHES))

    for normal_slide in nontumor_list_names:
        slidelist.append((normal_slide, NONTUMOR, ALL_PATCHES))

    return slidelist


def get_unqiue_tumor_slides():
    TUMOR = 1
    TUMOR_PATCHES_ONLY = 1
    ALL_PATCHES = 0
    tumor_list_names = glob.glob(os.path.join(cf.tumor_slide_path, "*.tif"))
    tumor_list_names = [tumorfn.split("/")[-1] for tumorfn in tumor_list_names]

    tumor_unique = []
    for tumor_slide in tumor_list_names:
        if tumor_slide == 'tumor_058.tif' or tumor_slide == 'tumor_078.tif' or tumor_slide == 'tumor_092.tif' or tumor_slide == 'tumortest_001.tif' or tumor_slide == 'tumortest_040.tif' or tumor_slide == 'tumor_082.tif' or tumor_slide == 'tumor_029.tif' or tumor_slide == 'tumortest_016.tif' or tumor_slide == 'tumor_076.tif' or tumor_slide == 'tumor_056.tif' or tumor_slide == 'tumortest_073.tif' or tumor_slide == 'tumor_036.tif' or tumor_slide == 'tumor_085.tif' or tumor_slide == 'tumor_064.tif' or tumor_slide == 'tumor_105.tif' or tumor_slide == 'tumor_021.tif' or tumor_slide == 'tumor_038.tif' or tumor_slide == 'tumor_052.tif' or tumor_slide == 'tumor_051.tif' or tumor_slide == 'tumor_016.tif' or tumor_slide == 'tumor_061.tif' or tumor_slide == 'tumortest_113.tif' or tumor_slide == 'tumor_072.tif' or tumor_slide == 'tumor_033.tif' or tumor_slide == 'tumor_026.tif' or tumor_slide == 'tumor_069.tif' or tumor_slide == 'tumortest_071.tif' or tumor_slide == 'tumor_068.tif' or tumor_slide == 'tumor_039.tif' or tumor_slide == 'tumor_104.tif' or tumor_slide == 'tumor_089.tif' or tumor_slide == 'tumortest_002.tif' or tumor_slide == 'tumor_031.tif' or tumor_slide == 'tumor_101.tif' or tumor_slide == 'tumortest_021.tif' or tumor_slide == 'tumor_091.tif' or tumor_slide == 'tumor_055.tif' or tumor_slide == 'tumor_004.tif' or tumor_slide == 'tumor_079.tif' or tumor_slide == 'tumor_014.tif' or tumor_slide == 'tumortest_090.tif' or tumor_slide == 'tumor_071.tif' or tumor_slide == 'tumortest_104.tif' or tumor_slide == 'tumor_037.tif' or tumor_slide == 'tumortest_082.tif' or tumor_slide == 'tumor_109.tif' or tumor_slide == 'tumortest_094.tif' or tumor_slide == 'tumor_106.tif' or tumor_slide == 'tumor_047.tif' or tumor_slide == 'tumor_042.tif' or tumor_slide == 'tumor_058.tif' or \
                tumor_slide == 'tumortest_075.tif' or tumor_slide == 'tumortest_051.tif' or tumor_slide == 'tumortest_026.tif' or tumor_slide == 'tumortest_121.tif' or tumor_slide == 'tumor_110.tif' or tumor_slide == 'tumor_054.tif' or tumor_slide == 'tumor_075.tif' or tumor_slide == 'tumor_009.tif' or tumor_slide == 'tumortest_068.tif' or tumor_slide == 'tumortest_092.tif' or tumor_slide == 'tumor_044.tif' or tumor_slide == 'tumor_083.tif' or tumor_slide == 'tumor_025.tif' or tumor_slide == 'tumor_011.tif' or tumor_slide == 'tumortest_108.tif' or tumor_slide == 'tumor_108.tif' or tumor_slide == 'tumor_095.tif' or tumor_slide == 'tumor_090.tif' or tumor_slide == 'tumor_020.tif' or tumor_slide == 'tumor_018.tif' or tumor_slide == 'tumor_062.tif' or tumor_slide == 'tumor_034.tif' or tumor_slide == 'tumortest_105.tif' or tumor_slide == 'tumortest_027.tif' or tumor_slide == 'tumor_046.tif':
            if tumor_slide == "tumor_092.tif" or tumor_slide == "tumor_010.tif" or tumor_slide == "tumor_018.tif" or tumor_slide == "tumor_020.tif" or tumor_slide == "tumor_025.tif" or tumor_slide == "tumor_044.tif" or tumor_slide == "tumor_046.tif" or tumor_slide == "tumor_051.tif" or tumor_slide == "tumor_054.tif" or tumor_slide == "tumor_056.tif" or tumor_slide == "tumor_067.tif" or tumor_slide == "tumor_033.tif" or tumor_slide == "tumor_085.tif" or tumor_slide == "tumor_079.tif" or tumor_slide == "tumor_029.tif" or tumor_slide == "tumor_034.tif" or tumor_slide == "tumor_055.tif" or tumor_slide == "tumor_015.tif" or tumor_slide == "tumor_110.tif" or tumor_slide == "tumor_095.tif":
                tumor_unique.append((tumor_slide, TUMOR, TUMOR_PATCHES_ONLY))
            else:
                tumor_unique.append((tumor_slide, TUMOR, ALL_PATCHES))
    return tumor_unique


class MlFilepaths:
    def __init__(self, TOP_DIR_DATA_FINAL, level=2):
        if not os.path.isdir(TOP_DIR_DATA_FINAL):
            os.makedirs(TOP_DIR_DATA_FINAL)

        self.train_images_dir = os.path.join(TOP_DIR_DATA_FINAL, "train_images_lv" + str(level))
        self.train_masks_dir = os.path.join(TOP_DIR_DATA_FINAL, "train_masks_lv" + str(level))
        self.val_images_dir = os.path.join(TOP_DIR_DATA_FINAL, "val_images_lv" + str(level))
        self.val_masks_dir = os.path.join(TOP_DIR_DATA_FINAL, "val_masks_lv" + str(level))

        self.test_images_dir = os.path.join(TOP_DIR_DATA_FINAL, "test_images_lv" + str(level))
        self.test_masks_dir = os.path.join(TOP_DIR_DATA_FINAL, "test_masks_lv" + str(level))

        if not os.path.isdir(self.train_images_dir):
            os.makedirs(self.train_images_dir)

        if not os.path.isdir(self.train_masks_dir):
            os.makedirs(self.train_masks_dir)

        if not os.path.isdir(self.val_images_dir):
            os.makedirs(self.val_images_dir)

        if not os.path.isdir(self.val_masks_dir):
            os.makedirs(self.val_masks_dir)

        if not os.path.isdir(self.test_images_dir):
            os.makedirs(self.test_images_dir)

        if not os.path.isdir(self.test_masks_dir):
            os.makedirs(self.test_masks_dir)

        print("train_images_dir", self.train_images_dir)
        print("train_masks_dir", self.train_masks_dir)
        print("val_images_dir", self.val_images_dir)
        print("val_masks_dir", self.val_masks_dir)
        print("test_images_dir", self.test_images_dir)
        print("test_masks_dir", self.test_masks_dir)


def main():
    print("The get_slide_stats was not executed")


def get_slide_stats(TOP_DIR_DATA_FINAL):
    # Here we will glob all the patches in all splits and find the number and type of patches contributed by each slide
    filepaths_ml = MlFilepaths(TOP_DIR_DATA_FINAL=TOP_DIR_DATA_FINAL)
    all_train_imgs = glob.glob(os.path.join(filepaths_ml.train_images_dir, "*.tif"))
    all_val_imgs = glob.glob(os.path.join(filepaths_ml.val_images_dir, "*.tif"))
    all_test_imgs = glob.glob(os.path.join(filepaths_ml.test_images_dir, "*.tif"))

    all_slide_names = get_slidelist()
    # slide is a tuple (name,..) we only want the name and also to remove the file extension
    all_slide_names = [slide[0].split('.')[0] for slide in all_slide_names]
    f = open("slide_stats.txt", "w")
    f.write('slidename,split,normal_from_normal,normal_from_tumor,tumor_from_tumor, boundary \n')
    for slide in tqdm(all_slide_names):
        train_matching = [patch for patch in all_train_imgs if slide in patch]
        val_matching = [patch for patch in all_val_imgs if slide in patch]
        test_matching = [patch for patch in all_test_imgs if slide in patch]
        if len(train_matching) == 0 and len(val_matching) == 0 and len(test_matching) == 0:
            raise Exception("slide not found in any split")
        if len(train_matching) != 0 and len(val_matching) != 0 and len(test_matching) != 0:
            raise Exception("slide found in all splits")

        if len(train_matching) != 0:
            split = 'train'
            slide_patches = train_matching[:]
        elif len(val_matching) != 0:
            split = 'val'
            slide_patches = val_matching[:]
        elif len(test_matching) != 0:
            split = 'test'
            slide_patches = test_matching[:]
        else:
            raise Exception("slide not found in any split")

        normal_from_normal = 0
        normal_from_tumor = 0
        tumor_from_tumor = 0
        boundary = 0
        for patch in slide_patches:
            if 'normal_from_normal' in patch:
                normal_from_normal += 1
            elif 'normal_from_tumor' in patch:
                normal_from_tumor += 1
            elif 'tumor_from_tumor' in patch:
                tumor_from_tumor += 1
            elif 'boundary' in patch:
                boundary += 1
            else:
                raise Exception("patch not found in any category")
        f.write(slide + ',' + split + ',' + str(normal_from_normal) + ',' + str(normal_from_tumor) + ',' + str(
            tumor_from_tumor) + ',' + str(boundary) + '\n')

    # done with all the slides and patches
    f.close()


if __name__ == "__main__":
    main()
    # print('running get slide patch stats')
    # get_slide_stats(TOP_DIR_DATA_FINAL="/home/jjuybari/data/c16_data/ml_split_refine/")

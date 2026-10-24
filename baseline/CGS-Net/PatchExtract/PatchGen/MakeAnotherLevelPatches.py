import glob
import os
from PatchExtract.parameters import cf, hp
import ExtractPatches
import PatchMaskGen
import random
from PatchExtract.PatchUtils.utils import MlFilepaths
from multiprocessing import Pool
from functools import partial
from PatchExtract.PatchUtils.utils import rename_patches
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--ml_dir", type=str,
                    help="the directory path to the patch folder.",
                    required=True)
parser.add_argument("--lvl3_maskpath", type=str,
                    help="the final location for the patch spilts",
                    required=True)
parser.add_argument("--num_threads", type=int,
                    help="number of threads", required=True)

args = parser.parse_args()


def main():
    random.seed(a=48)
    filepaths_ml = MlFilepaths(TOP_DIR_DATA_FINAL=args.ml_dir, level=2)
    print("---------train--------")
    make_diff_lvlpatches_entire_slide(folder=filepaths_ml.train_images_dir, groupname="train")
    print("---------val---------")
    make_diff_lvlpatches_entire_slide(folder=filepaths_ml.val_images_dir, groupname="val")
    print("---------test---------")
    make_diff_lvlpatches_entire_slide(folder=filepaths_ml.test_images_dir, groupname="test")
    print("finished")


def make_diff_lvlpatches_entire_slide(folder, groupname):
    """
    folder is going to be either train_images_lv2 or val_images_lv2 or test_images_lv2. The code will first glob all the patches
    in that directory

    groupname is going to be either train, val, or test. This is just for the naming of the patch masks
    """
    all_patches = glob.glob(os.path.join(folder, '*.tif'))  # full file patch to patch
    all_patches = [patch.split("/")[-1] for patch in all_patches]  # simply the filenames of the patches
    slide_names = ['_'.join(patch.split("_")[0:2]) for patch in all_patches]  # the slide name prefix for each patch 12K

    slide_names = list(set(slide_names))  # a list of all the unique slides

    with Pool(args.num_threads) as p:
        p.map(partial(make_patch_new_lvl, group_name=groupname, folder=folder), slide_names)


def make_patch_new_lvl(slide, group_name, folder):
    level_2_patches = glob.glob(os.path.join(folder, slide + "*.tif"))
    center_cords = []
    normal = False
    if "normal" in slide:
        normal = True
    for level2patch in level_2_patches:
        just_patch_name = level2patch.split("/")
        ci = just_patch_name[-1].split("_")[2]
        ci = int(ci)
        cj = just_patch_name[-1].split("_")[3]
        cj = int(cj)
        center_cords.append((ci, cj))
    # Now we have all the center cords for all the given slide and the slide in the train slide
    # now call get_patch_lower_level
    # also note you are passing the level 2 slide map which is want you want. the slide map is only used for visually
    # to track all the patches extracted.
    ExtractPatches.get_patch_lower_level(slide_name=slide, current_center_cords_list=center_cords, current_level=2,
                                         lower_level=3, downsample_current=hp.resolution_levels[2],
                                         downsample_factor_lower_level=hp.resolution_levels[3], normal=normal,
                                         group=group_name, ml_split_dir=args.ml_dir)
    # after this function call you will now have all the level 3 patches. you now need to make the patch level masks
    # ground truths for all the patches. This function assumes nothing
    # it globs the patches and all that. you will already have the patches in this for loop
    PatchMaskGen.make_patch_mask(slide=slide + '.tif', mask_path=args.lvl3_maskpath, patch_source=folder,
                                 current_cord_lvl=hp.resolution_levels[2], patch_size=224,
                                 patch_dir=cf.patch_path,
                                 downsample_factor=hp.resolution_levels[3], group=group_name, final_dir=args.ml_dir)
    print("slide completed", slide)


if __name__ == "__main__":
    # the main generates the different level patches and the masks for the different level
    main()
    # the full file name of the patch gets lost and these functions ensure that the file paths for the
    # level 2 and level 3 match correctly
    filepaths_ml_level3 = MlFilepaths(TOP_DIR_DATA_FINAL=args.ml_dir, level=3)
    rename_patches(mask_dir=filepaths_ml_level3.train_masks_dir, image_dir=filepaths_ml_level3.train_images_dir)
    rename_patches(mask_dir=filepaths_ml_level3.val_masks_dir, image_dir=filepaths_ml_level3.val_images_dir)
    rename_patches(mask_dir=filepaths_ml_level3.test_masks_dir, image_dir=filepaths_ml_level3.test_images_dir)

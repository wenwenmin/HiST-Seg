import os
import numpy as np
import glob
from shutil import copy2
import argparse
import math
from utils import get_slidelist, MlFilepaths, get_unqiue_tumor_slides
from PatchExtract.parameters import cf
from multiprocessing import Pool
from functools import partial


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--final_dir", type=str,
                        help="the final location for the patch spilts",
                        required=True)
    parser.add_argument("--num_threads", type=int,
                        help="number of threads", required=True)

    args = parser.parse_args()

    TOP_DIR_DATA_FINAL = args.final_dir
    val_spilt = 0.20
    test_spilt = 0.20
    slidelist = get_slidelist()

    # we need to separate the slides into tumor and normal
    tumor_slides = []
    # note these are just the names of the slides
    tumor_unique_slides = get_unqiue_tumor_slides()
    normal_slides = []

    for slide in slidelist:
        if "tumor" in slide[0]:
            tumor_slides.append(slide)
        else:
            normal_slides.append(slide)
    num_all_tumor_slides = len(tumor_slides)
    # we now need to remove the unique tumor slides from tumor_slides
    tumor_set = set(tumor_slides)
    tumor_unique_set = set(tumor_unique_slides)
    tumor_slides = list(tumor_set - tumor_unique_set)

    assert num_all_tumor_slides == len(tumor_slides) + len(tumor_unique_slides)

    # we are calculating the number of slides that will be used for validation
    num_tumor_slides = len(tumor_slides)
    num_tumorunique_slides = len(tumor_unique_slides)
    num_normal_slides = len(normal_slides)
    # these two variables are for how many slides we use for validation
    int_tumor_val = math.ceil(num_tumor_slides * val_spilt)
    int_tumorunqiue_val = math.ceil(num_tumorunique_slides * val_spilt)
    int_normal_val = math.ceil(num_normal_slides * val_spilt)

    int_tumor_test = math.ceil(num_tumor_slides * test_spilt)
    int_tumorunqiue_test = math.ceil(num_tumorunique_slides * test_spilt)
    int_normal_test = math.ceil(num_normal_slides * test_spilt)

    int_normal_train = num_normal_slides - int_normal_val - int_normal_test
    int_tumor_train = num_tumor_slides - int_tumor_val - int_tumor_test
    int_tumorunqiue_train = num_tumorunique_slides - int_tumorunqiue_val - int_tumorunqiue_test

    filepaths_ml = MlFilepaths(TOP_DIR_DATA_FINAL=TOP_DIR_DATA_FINAL)

    # validation

    normal_val_list = np.random.choice(num_normal_slides, int_normal_val, replace=False)
    tumor_val_list = np.random.choice(num_tumor_slides, int_tumor_val, replace=False)
    tumorunqiue_val_list = np.random.choice(num_tumorunique_slides, int_tumorunqiue_val, replace=False)

    normal_val_list = [normal_slides[i] for i in normal_val_list]
    tumor_val_list = [tumor_slides[i] for i in tumor_val_list]
    tumorunqiue_val_list = [tumor_unique_slides[i] for i in tumorunqiue_val_list]


    for normal_val in normal_val_list:
        normal_slides.remove(normal_val)
        # removing the validation normal tissue from the main list (to prevent information leakage)

    for tumor_val in tumor_val_list:
        tumor_slides.remove(tumor_val)  # all the training dataset will be tumor list names and normal list names

    for tumorunqiue_val in tumorunqiue_val_list:
        tumor_unique_slides.remove(tumorunqiue_val)

    normal_test_list = np.random.choice(num_normal_slides - int_normal_val, int_normal_test, replace=False)
    tumor_test_list = np.random.choice(num_tumor_slides - int_tumor_val, int_tumor_test, replace=False)
    tumor_testunique_list = np.random.choice(num_tumorunique_slides - int_tumorunqiue_val, int_tumorunqiue_test,
                                             replace=False)

    normal_test_list = [normal_slides[i] for i in normal_test_list]
    tumor_test_list = [tumor_slides[i] for i in tumor_test_list]
    tumor_testunique_list = [tumor_unique_slides[i] for i in tumor_testunique_list]

    for normal_test in normal_test_list:
        normal_slides.remove(normal_test)

    for tumor_test_slide in tumor_test_list:
        tumor_slides.remove(tumor_test_slide)

    for tumorunqiue_test in tumor_testunique_list:
        tumor_unique_slides.remove(tumorunqiue_test)

    # dataset validation
    # train
    print("----train----")
    print("int_normal_train ", int_normal_train)
    print("int_tumor_train ", int_tumor_train)
    print("int_tumorunqiue_train ", int_tumorunqiue_train)
    # validation
    print("----val----")
    print("int_normal_val ", int_normal_val)
    print("int_tumor_val ", int_tumor_val)
    print("int_tumorunqiue_val ", int_tumorunqiue_val)
    # test
    print("----test----")
    print("int_normal_test ", int_normal_test)
    print("int_tumor_test ", int_tumor_test)
    print("int_tumorunqiue_test ", int_tumorunqiue_test)
    print("-------------------this is for normal from normal train")
    with Pool(args.num_threads) as p:
        p.map(partial(move_normal_from_normal_patches, split_image_dir=filepaths_ml.train_images_dir,
                      split_mask_dir=filepaths_ml.train_masks_dir), normal_slides)
    print("-------------------this is for normal from normal validation")
    with Pool(args.num_threads) as p:
        p.map(partial(move_normal_from_normal_patches, split_image_dir=filepaths_ml.val_images_dir,
                      split_mask_dir=filepaths_ml.val_masks_dir), normal_val_list)
    print("-------------------this is for normal from normal test")
    with Pool(args.num_threads) as p:
        p.map(partial(move_normal_from_normal_patches, split_image_dir=filepaths_ml.test_images_dir,
                      split_mask_dir=filepaths_ml.test_masks_dir), normal_test_list)
    # normal from tumor unique all splits
    print("-------------------this is for normal from tumor unique train")
    with Pool(args.num_threads) as p:
        p.map(partial(move_normal_from_tumor_patches, split_image_dir=filepaths_ml.train_images_dir,
                      split_mask_dir=filepaths_ml.train_masks_dir), tumor_unique_slides)
    print("-------------------this is for normal from tumor unique validation")
    with Pool(args.num_threads) as p:
        p.map(partial(move_normal_from_tumor_patches, split_image_dir=filepaths_ml.val_images_dir,
                      split_mask_dir=filepaths_ml.val_masks_dir), tumorunqiue_val_list)
    print("-------------------this is for normal from tumor unique test")
    with Pool(args.num_threads) as p:
        p.map(partial(move_normal_from_tumor_patches, split_image_dir=filepaths_ml.test_images_dir,
                      split_mask_dir=filepaths_ml.test_masks_dir), tumor_testunique_list)

    # Normal from tumor all three splits
    print("-------------------this is for normal from tumor train")
    with Pool(args.num_threads) as p:
        p.map(partial(move_normal_from_tumor_patches, split_image_dir=filepaths_ml.train_images_dir,
                      split_mask_dir=filepaths_ml.train_masks_dir), tumor_slides)
    print("-------------------this is for normal from tumor validation")
    with Pool(args.num_threads) as p:
        p.map(partial(move_normal_from_tumor_patches, split_image_dir=filepaths_ml.val_images_dir,
                      split_mask_dir=filepaths_ml.val_masks_dir), tumor_val_list)
    print("-------------------this is for normal from tumor test")
    with Pool(args.num_threads) as p:
        p.map(partial(move_normal_from_tumor_patches, split_image_dir=filepaths_ml.test_images_dir,
                      split_mask_dir=filepaths_ml.test_masks_dir), tumor_test_list)
    # tumor unique all three splits
    print("-------------------this is for tumor unique training")
    with Pool(args.num_threads) as p:
        p.map(partial(move_tumor_patches, split_image_dir=filepaths_ml.train_images_dir,
                      split_mask_dir=filepaths_ml.train_masks_dir), tumor_unique_slides)
    print("-------------------this is for tumor unique validation")
    with Pool(args.num_threads) as p:
        p.map(partial(move_tumor_patches, split_image_dir=filepaths_ml.val_images_dir,
                      split_mask_dir=filepaths_ml.val_masks_dir), tumorunqiue_val_list)
    print("-------------------this is for tumor unique test")
    with Pool(args.num_threads) as p:
        p.map(partial(move_tumor_patches, split_image_dir=filepaths_ml.test_images_dir,
                      split_mask_dir=filepaths_ml.test_masks_dir), tumor_testunique_list)

    # Tumor all three splits
    print("-------------------this is for tumor training")
    with Pool(args.num_threads) as p:
        p.map(partial(move_tumor_patches, split_image_dir=filepaths_ml.train_images_dir,
                      split_mask_dir=filepaths_ml.train_masks_dir), tumor_slides)
    print("-------------------this is for tumor validation")
    with Pool(args.num_threads) as p:
        p.map(partial(move_tumor_patches, split_image_dir=filepaths_ml.val_images_dir,
                      split_mask_dir=filepaths_ml.val_masks_dir), tumor_val_list)
    print("-------------------this is for tumor test")
    with Pool(args.num_threads) as p:
        p.map(partial(move_tumor_patches, split_image_dir=filepaths_ml.test_images_dir,
                      split_mask_dir=filepaths_ml.test_masks_dir), tumor_test_list)


def move_selected_patches(selected_patches, image_dir, mask_dir):
    for patch in selected_patches:
        just_patch_name = patch.split("/")
        mask_patch = patch.replace(just_patch_name[-2], just_patch_name[-2] + "_mask")
        mask_fname = (just_patch_name[-1]).replace(".tif", "_mask.tif")
        mask_patch = mask_patch.replace(just_patch_name[-1], mask_fname)
        dest_string_path = image_dir + "/" + just_patch_name[-1]
        dest_string_path_mask = mask_dir + "/" + mask_fname
        copy2(patch, dest_string_path)  # https://stackoverflow.com/questions/123198/how-to-copy-files
        copy2(mask_patch, dest_string_path_mask)


def move_normal_from_normal_patches(norm_slide, split_image_dir, split_mask_dir):
    normal_fname = norm_slide[0].replace('.tif', '')  # we should have the name of the file without .svs
    glob_name = os.path.join(cf.normal_patch_path, normal_fname + "_*")

    raw_normal_match = glob.glob(glob_name)
    array_glob = np.array(raw_normal_match)
    pick_normal_match = array_glob
    # here patch includes the full path name
    move_selected_patches(selected_patches=pick_normal_match, image_dir=split_image_dir,
                          mask_dir=split_mask_dir)


def move_normal_from_tumor_patches(tum_slide, split_image_dir, split_mask_dir):
    tumor_fname = tum_slide[0].replace('.tif', '')  # we should have the name of the file without .tif
    glob_name = os.path.join(cf.normal_patch_path, tumor_fname + "_*")  # key part here

    raw_tumor_match = glob.glob(glob_name)

    array_glob = np.array(raw_tumor_match)
    pick_tumor_match = array_glob
    # here patch includes the full path name
    move_selected_patches(selected_patches=pick_tumor_match, image_dir=split_image_dir,
                          mask_dir=split_mask_dir)


def move_tumor_patches(tum_slide, split_image_dir, split_mask_dir):
    tumor_train_fname = tum_slide[0].replace('.tif', '')  # we should have the name of the file without .svs
    glob_name = os.path.join(cf.tumor_patch_path, tumor_train_fname + "_*")
    raw_tumor_match = glob.glob(glob_name)

    array_glob = np.array(raw_tumor_match)
    pick_tumor_match = array_glob
    # here patch includes the full path name
    move_selected_patches(selected_patches=pick_tumor_match, image_dir=split_image_dir,
                          mask_dir=split_mask_dir)


if __name__ == "__main__":
    main()

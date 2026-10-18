#!/usr/bin/env bash


echo "Running script"
THREAD_COUNT=1

# to run the full patch generation code on a single level for level 2
time python ./top_caller_patches.py --make_slide_mask 1 --make_patch 1 --make_patch_mask 1 --num_threads $THREAD_COUNT --level 2

# now we can move the patches for level 2 into train, val and test.
FINAL_DIR="./newpatches/" # this is the file path for the dataset splits
time python ./PatchExtract/PatchUtils/Patch_mover_first_pass.py --final_dir $FINAL_DIR --num_threads $THREAD_COUNT

# keep in mind we need the level 3 masks (mainly tumor masks) to generate the level 3 patch masks.
# the reason why we don't need anything for the rgb patches is that they are read directly from the slide file in level 0 coordinates.
time python ./top_caller_patches.py --make_slide_mask 1 --make_patch 0 --make_patch_mask 0 --num_threads $THREAD_COUNT --level 3

# run the code on a different patch level. Note the data needed to already been spilt into train, test, val
# Note the file paths need to be updated
MASK_LVL3="/path/here/patchrun/mask_lv3/" # this needs to match base_dir_output in parameters.py
time python ./PatchExtract/PatchGen/MakeAnotherLevelPatches.py --ml_dir $FINAL_DIR --lvl3_maskpath $MASK_LVL3 --num_threads $THREAD_COUNT

# make the combined trainval dir for level 2
mkdir ${FINAL_DIR}trainval_images_lv2/
mkdir ${FINAL_DIR}trainval_masks_lv2/
mkdir ${FINAL_DIR}trainval_images_lv3/
mkdir ${FINAL_DIR}trainval_masks_lv3/


cp -a ${FINAL_DIR}train_images_lv2/. ${FINAL_DIR}trainval_images_lv2/
cp -a ${FINAL_DIR}val_images_lv2/. ${FINAL_DIR}trainval_images_lv2/
cp -a ${FINAL_DIR}train_masks_lv2/. ${FINAL_DIR}trainval_masks_lv2/.
cp -a ${FINAL_DIR}val_masks_lv2/. ${FINAL_DIR}trainval_masks_lv2/.
# make the combined trainval for level 3
cp -a ${FINAL_DIR}train_images_lv3/. ${FINAL_DIR}trainval_images_lv3/.
cp -a ${FINAL_DIR}val_images_lv3/. ${FINAL_DIR}trainval_images_lv3/.
cp -a ${FINAL_DIR}train_masks_lv3/. ${FINAL_DIR}trainval_masks_lv3/.
cp -a ${FINAL_DIR}val_masks_lv3/. ${FINAL_DIR}trainval_masks_lv3/.
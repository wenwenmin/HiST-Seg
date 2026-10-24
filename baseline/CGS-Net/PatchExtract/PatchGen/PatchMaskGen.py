import glob
import os
from PatchExtract.parameters import cf
import tifffile


def make_patch_mask(slide, mask_path, patch_source, current_cord_lvl, patch_size, patch_dir,
                    downsample_factor, group=None, final_dir=None):
    """
    group is test/valid/train
    downsample_factor_cord is the the downsample factor that the patches were used to make the
    patches.
    downsample_factor can be higher or lower. this means we can go up a level or down a level.
    if we want to make patch masks for the same level that they were generated. ????
    cordinate level is the level of the pa. the patch names store the center cords in level 0
    mask_level is the level at which the patches are stored in. For example c16_data_new/patch/tumor_lv3/
    means that the tumor patches are in level 3 and the masks will then need to be level 3. so mask_level needs
    to match the patch level
    """
    half_p_size = patch_size // 2

    if group is None:
        # when running this code from DataGenFlow we do not need a group. The group is needed when
        # we run the code for Another level
        dir_name_mask = patch_source.split("/")[-1] + "_mask"
        patch_path_mask = os.path.join(patch_dir, dir_name_mask)
        print(patch_path_mask)

        patch_path_mask_group = patch_path_mask
    else:
        # hardcoded level 3 for now.
        dir_name_mask = group + '_masks_lv' + str(3)
        patch_path_mask_group = os.path.join(final_dir, dir_name_mask)
    if not os.path.isdir(patch_path_mask_group):
        os.makedirs(patch_path_mask_group)

    downsample = current_cord_lvl / downsample_factor

    slide_fname = slide
    # the spilt is for only grabbing the filename without the extention.
    list_all_patches = glob.glob(os.path.join(patch_source, slide_fname.split('.')[0]) + "_*")
    # need to load in the mask Even the normal slides have a tumor_mask_ file that is all black, this
    # is becuase I want to be constant with white meaning tumor and black meaning normal tissue
    tumor_mask = os.path.join(mask_path, 'tumor_mask_' + slide)
    mask_img = tifffile.imread(tumor_mask)
    for patch in list_all_patches:
        patch_name = patch.split("/")[-1]

        patch_name_no_type = patch_name.split(".")[0]
        # tumor_014_11437_18651_nontumor_boundary_mask.tif
        # the above is an example of a patch name
        # the code breaks it up and we divide up the name by "_"
        # so we have tumor 014 11437 18651 in list
        i_index = int(patch_name_no_type.split("_")[2])
        j_index = int(patch_name_no_type.split("_")[3])
        assert i_index > 0 and j_index > 0
        # potentially need to downsample
        i = int(i_index * downsample)
        j = int(j_index * downsample)
        a = j - half_p_size
        b = i - half_p_size

        mask_patch = mask_img[a:a + patch_size, b:b + patch_size]
        patch_fn = patch_path_mask_group + "/" + patch_name_no_type + "_mask" + cf.patch_file_type
        tifffile.imwrite(patch_fn, data=mask_patch)

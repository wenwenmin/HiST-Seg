import PatchExtract.SlideMapsMasks.MapMaskHelpers as MapMaskHelpers
from PatchExtract.SlideMapsMasks.MapMaskHelpers import read_xml_c16 as read_xml
import os
import tiffslide
from PatchExtract.parameters import cf, hp
import numpy as np
import cv2
import random
from PIL import Image


def extract_tumor_patch_from_tumor_slide(slide_name, mask_level, downsample_factor, patch_kept_center_cords,
                                         more_patches, current_num_patches_picked):
    """ Extract tumor patches using tumor slides
    Args:
        :param slide_name: The name of the slide
        :param downsample_factor: this is the factor that corresponds to the downsamply for the given mask level
        for instance at level 3 the downsample_factor is ~ 32. We need this because near the end of this function
        slide.read_region needs the top left corner cordinates in level 0 resolution. hence we have to upsample
        :param mask_level: level of mask should be int hryf
        :param patch_kept_center_cords: This is a list of all the center cordinates that we kept
        this is to keep track of what we have in patches so we dont get duplicate patches and
        to make sure that patch overlap is not too large.
        :returns a list of all the center cordinates of the patches.
    """
    slide_name_no_ext = slide_name.replace(cf.image_file_type, '')
    dir_name = 'tumor_lv' + str(mask_level)
    tumor_patch_path = os.path.join(cf.patch_path, dir_name)
    if not os.path.isdir(tumor_patch_path):
        os.makedirs(tumor_patch_path)

    slide, slide_map, tissue_mask = MapMaskHelpers.return_slides_masks(slide_name=slide_name,binary_tumor=1)
    # one since a tumor
    tumor_mask = MapMaskHelpers.get_slide_tumor_mask(slide_name)
    xml_fname = os.path.join(cf.xml_path, slide_name).replace(cf.image_file_type, '.xml')
    xml_cords_boundary_tumor, _ = read_xml(xml_fname, downsample_factor=downsample_factor)

    t_cnt = 0
    num_tries = 0
    print('{}: extracting tumor patches'.format(slide_name_no_ext))
    # progress_bar = tqdm(total=cf.tumor_patches_per_slide + 1)
    patch_tumor_threshold = 0.75
    patch_tumor_threshold_less_restrictive = 0.45
    while (t_cnt != (cf.tumor_patches_per_slide + more_patches)) and (num_tries <= hp.max_tries):
        """
        This for loop will randomly pick the center pixels. Then adjustments will be made to get the patch
        """
        num_tries += 1
        num_splines = len(xml_cords_boundary_tumor)-1
        rand_spline_idx = random.randint(0, num_splines)
        # rand_idx = num_tries % len(xml_cords_boundary_tumor[rand_spline_idx])
        x_cords = [ij[0] for ij in xml_cords_boundary_tumor[rand_spline_idx]]
        y_cords = [ij[1] for ij in xml_cords_boundary_tumor[rand_spline_idx]]

        bound_x = (max(x_cords) + min(x_cords)) // 2
        bound_y = (max(y_cords) + min(y_cords)) // 2

        # Here we are determinnig the range (max-min) for both the x and y directions and using the range
        # to determine the values to sample from.
        x_free = max(x_cords) - min(x_cords)
        y_free = max(y_cords) - min(y_cords)
        try:
            assert x_free > 0
            assert y_free > 0
        except AssertionError:
            print("something is wrong with the xml cordinates the range is negative")
            raise

        ci = random.randint(-x_free, x_free)  # randint(a, b): a <= N <= b
        cj = random.randint(-y_free, y_free)  # randint(a, b): a <= N <= b

        ci = abs(int(bound_x) + ci)
        cj = abs(int(bound_y) + cj)

        # (ci,cj) are the randomly selected center pixel coordinates for the patch
        # ci corresponds to the x-axis
        # cj corresponds to the y-axis
        # we will take those center pixels and make a square patch
        patch_tissue_ratio, patch_tumor_ratio = MapMaskHelpers.return_ratios(ci, cj, binary_tissue=0,
                                                                      tissue_mask=tissue_mask, mask=tumor_mask)
        use_patch_cords = False
        if patch_tumor_ratio > patch_tumor_threshold and patch_tissue_ratio > hp.patch_tissue_threshold:
            use_patch_cords = MapMaskHelpers.check_overlap(ci=ci,cj=cj,previous_center_cords=patch_kept_center_cords,
                                                           overlap=1)

        # here we are losenning the requirements for the tissue requirements
        # this only happens if we have gone through half of our attempts
        # Note that we are not changing the overlap percentage which is still at 1 i.e. 50% overlap
        if num_tries >= (hp.max_tries_boundary / 2):
            if patch_tumor_ratio > patch_tumor_threshold_less_restrictive and patch_tissue_ratio > hp.patch_tissue_threshold:
                use_patch_cords = MapMaskHelpers.check_overlap(ci=ci, cj=cj,
                                                               previous_center_cords=patch_kept_center_cords,
                                                               overlap=1)

        if use_patch_cords:
            patch_kept_center_cords.append((ci, cj))
            patch_name = slide_name_no_ext + '_' + str(ci) + '_' + str(cj) + '_tumor_from_tumor' + cf.patch_file_type
            patch_name = os.path.join(tumor_patch_path, patch_name)
            patch_pil_img, top_left, bottom_right = MapMaskHelpers.return_patch_from_slide(slide=slide,
                                        mask_level=mask_level, ci=ci, cj=cj, downsample_factor=downsample_factor)
            Image.fromarray(patch_pil_img).save(patch_name)

            cv2.rectangle(slide_map, top_left, bottom_right, hp.red, hp.patch_thickness)
            t_cnt += 1
            # progress_bar.update(1)
    MapMaskHelpers.write_slide_post_map(slide_name, slide_map)
    print('{}: extracted: tumor = {}'.format(slide_name_no_ext, t_cnt))
    # this is where the orginal function ended the code below oversamples patches from tumor slides that do not contribute a minimum of 80 patches per slide
    # note that this is to keep all slides contributing roughly the same number of patches. Look at paramters.py and see that the
    # number of patches per tumor slides is 100. 80 was selected so that all contribute roughly the same.
    current_num_patches_picked += t_cnt
    while current_num_patches_picked < 80:
        xml_cord = random.choice(xml_cords_boundary_tumor)
        rand_idx = random.randint(0, len(xml_cord) - 1)
        xml_cords = xml_cord[rand_idx]
        # print("xml cords is")
        # print(xml_cord)
        ci = xml_cords[0]
        cj = xml_cords[1]
        patch_kept_center_cords.append((ci, cj))
        patch_name = slide_name_no_ext + '_' + str(ci) + '_' + str(cj) + '_tumor_boundary' + cf.patch_file_type
        patch_name = os.path.join(tumor_patch_path, patch_name)
        patch_pil_img, top_left, bottom_right = MapMaskHelpers.return_patch_from_slide(slide=slide,
                                                                                       mask_level=mask_level,
                                                                                       ci=ci, cj=cj,
                                                                                       downsample_factor=downsample_factor)

        Image.fromarray(patch_pil_img).save(patch_name)
        current_num_patches_picked += 1
        cv2.rectangle(slide_map, top_left, bottom_right, hp.brown, hp.patch_thickness)
    MapMaskHelpers.write_slide_post_map(slide_name, slide_map)
    print('{}: extracted: tumor = {}'.format(slide_name_no_ext, t_cnt))
    return patch_kept_center_cords, t_cnt


def extract_normal_patch_from_tumor_slide(slide_name, mask_level, downsample_factor):
    """ Extract tumor patches using tumor slides
    Args:
        slide_num (int): number of slide used
        :param downsample_factor this is the factor that corresponds to the downsamply for the given mask level
        for instance at level 3 the downsample_factor is ~ 32. We need this because near the end of this function
        slide.read_region needs the top left corner cordinates in level 0 resolution. hence we have to upsample
        mask_level (int): level of mask
        :returns a list of all the center cordinates of the patches.
    """

    patch_kept_center_cords = []
    slide_name_no_ext = slide_name.replace(cf.image_file_type, '')
    dir_name = 'normal_lv' + str(mask_level)
    tumor_patch_path = os.path.join(cf.patch_path, dir_name)
    # print("the tumor patch path is ", tumor_patch_path)
    if not os.path.isdir(tumor_patch_path):
        os.makedirs(tumor_patch_path)

    slide, slide_map, tissue_mask = MapMaskHelpers.return_slides_masks(slide_name=slide_name, binary_tumor=1)
    # one since a tumor slide
    normal_mask = MapMaskHelpers.get_slide_normal_mask(slide_name)
    """
    The way that the normal mask is generated is using the tumor mask which means that coarse grain
    will get passed along so repeat the same procedure
    """

    width, height = np.array(slide.level_dimensions[mask_level])

    t_cnt = 0
    num_tries = 0

    print('{}: extracting normal patches from tumor slide'.format(slide_name_no_ext))
    # progress_bar = tqdm(total=cf.normal_patches_per_tumor_slide + 1)
    while (t_cnt != cf.normal_patches_per_tumor_slide) and (num_tries <= hp.max_tries):
        """
        This for loop will randomly pick the center pixels. Then adjustments will be made to get the patch
        """
        # print("do I make it here")
        num_tries += 1
        ci = random.randint(0, width - 2)  # randint(a, b): a <= N <= b
        cj = random.randint(0, height - 2)  # randint(a, b): a <= N <= b
        # (ci,cj) are the randomly selected center pixel cordinates for the patch
        # ci corresponds to the x-axis
        # cj corresponds to the y-axis
        # we will take those center pixels and make a square patch

        patch_tissue_ratio, patch_normal_ratio = MapMaskHelpers.return_ratios(ci, cj, binary_tissue=0,
                                                                              tissue_mask=tissue_mask, mask=normal_mask)
        use_patch_cords = False
        # print("the patch normal ratio is ", patch_normal_ratio)
        # print("the patch tissue ratio is ", patch_tissue_ratio)
        if patch_normal_ratio > hp.normal_patch_normal_ratio_threshold and patch_tissue_ratio > hp.patch_tissue_threshold:
            use_patch_cords = MapMaskHelpers.check_overlap(ci=ci,cj=cj,previous_center_cords=patch_kept_center_cords,
                                                           overlap=1)

        if use_patch_cords:
            patch_kept_center_cords.append((ci, cj))
            patch_name = slide_name_no_ext + '_' + str(ci) + '_' + str(cj) + '_normal_from_tumor' + cf.patch_file_type
            # print("patch_name for normal patch from tumor slide")
            # print(patch_name)
            patch_name = os.path.join(tumor_patch_path, patch_name)
            patch_pil_img, top_left, bottom_right = MapMaskHelpers.return_patch_from_slide(slide=slide,
                                        mask_level=mask_level, ci=ci, cj=cj, downsample_factor=downsample_factor)
            Image.fromarray(patch_pil_img).save(patch_name)

            cv2.rectangle(slide_map, top_left, bottom_right, hp.green, hp.patch_thickness)
            t_cnt += 1
            # progress_bar.update(1)

    MapMaskHelpers.write_slide_post_map(slide_name, slide_map)
    print('{}: extracted: normal = {}'.format(slide_name_no_ext, t_cnt))
    # progress_bar.close()
    return patch_kept_center_cords


def extract_normal_patch_from_normal_slide(slide_name, mask_level, downsample_factor):
    """Extract normal patches using normal slides
    Args:
        :param slide_name: number of slide used
        :param downsample_factor this is the factor that corresponds to the downsamply for the given mask level
        for instance at level 3 the downsample_factor is ~ 32. We need this because near the end of this function
        slide.read_region needs the top left corner cordinates in level 0 resolution. hence we have to upsample
        :param mask_level: level of mask should be an int
        :returns a list of all the center coordinates of the patches.
    """

    patch_kept_center_cords = []
    slide_name_no_ext = slide_name.replace(cf.image_file_type, '')
    dir_name = 'normal_lv' + str(mask_level)
    tumor_patch_path = os.path.join(cf.patch_path, dir_name)
    if not os.path.isdir(tumor_patch_path):
        os.makedirs(tumor_patch_path)

    # I only need the tissue mask since these are normal slides and the normal mask corresponds to the tissue mask
    # there are no tumors
    slide, slide_map, tissue_mask = MapMaskHelpers.return_slides_masks(slide_name=slide_name, binary_tumor=0)
    width, height = np.array(slide.level_dimensions[mask_level])

    t_cnt = 0
    num_tries = 0

    print('{}: extracting normal patches'.format(slide_name_no_ext))
    # progress_bar = tqdm(total=cf.normal_patches_per_normal_slide + 1)
    while (t_cnt != cf.normal_patches_per_normal_slide) and (num_tries <= hp.max_tries):
        """
        This for loop will randomly pick the center pixels. Then adjustments will be made to get the patch
        """
        num_tries += 1
        ci = random.randint(0, width - 2)  # randint(a, b): a <= N <= b
        cj = random.randint(0, height - 2)  # randint(a, b): a <= N <= b
        # (ci,cj) are the randomly selected center pixel coordinates for the patch
        # ci corresponds to the x-axis
        # cj corresponds to the y-axis
        # we will take those center pixels and make a square patch

        # recall this is a normal slide is the tissue mask is the normal mask
        patch_normal_ratio = MapMaskHelpers.return_ratios(ci, cj, binary_tissue=1,
                                                          tissue_mask=tissue_mask)
        use_patch_cords= False
        if patch_normal_ratio > hp.normal_patch_normal_ratio_threshold:
            # with boundary patches this line needs to be changed the number was 0.90

            use_patch_cords = MapMaskHelpers.check_overlap(ci=ci,cj=cj,previous_center_cords=patch_kept_center_cords,
                                                           overlap=1)

        if use_patch_cords:
            patch_kept_center_cords.append((ci, cj))
            patch_name = slide_name_no_ext + '_' + str(ci) + '_' + str(cj) + '_normal_from_normal' + cf.patch_file_type
            patch_name = os.path.join(tumor_patch_path, patch_name)
            patch_pil_img, top_left, bottom_right = MapMaskHelpers.return_patch_from_slide(slide=slide,
                                                                                           mask_level=mask_level,
                                                                                           ci=ci, cj=cj,
                                                                                           downsample_factor=downsample_factor)
            Image.fromarray(patch_pil_img).save(patch_name)

            cv2.rectangle(slide_map, top_left, bottom_right, hp.green, hp.patch_thickness)
            t_cnt += 1
            # progress_bar.update(1)

    MapMaskHelpers.write_slide_post_map(slide_name, slide_map)
    print('{}: extracted: normal = {}'.format(slide_name_no_ext, t_cnt))
    # progress_bar.close()
    return patch_kept_center_cords


def get_patch_lower_level(slide_name, current_center_cords_list, current_level, lower_level, downsample_current,
                          downsample_factor_lower_level, normal, group, ml_split_dir):
    BLUE = (255, 0, 0)
    THICKNESS = 1

    slide_map = MapMaskHelpers.get_slide_post_map(slide_name+'.tif')
    # print("the slide name path is ", slide_name+'.tif', flush=True)
    slide_name_no_ext = slide_name.replace(cf.image_file_type, '')
    # print("the slide name  with no ext is ", slide_name_no_ext)
    # hard_coded_patch_path = "/home/jjuybari/data/c16_data/ml_july16/"
    dir_name = group + '_images_lv' + str(lower_level)
    patch_path = os.path.join(ml_split_dir, dir_name)
    if not os.path.isdir(patch_path):
        os.makedirs(patch_path)
    if normal:
        # dir_name = 'normal_lv' + str(lower_level)
        # patch_path = os.path.join(hard_coded_patch_path, dir_name)
        slide_fullpath = os.path.join(cf.nontumor_slide_path, slide_name + cf.image_file_type)
        # if not os.path.isdir(patch_path):
        #     os.makedirs(patch_path)
    else:
        # dir_name = 'tumor_lv' + str(lower_level)
        slide_fullpath = os.path.join(cf.tumor_slide_path, slide_name + cf.image_file_type)

    # slide_fullpath = os.path.join(cf.slide_path, slide_name)
    # print("from function call lower_level",slide_fullpath, flush=True)
    slide = tiffslide.open_slide(slide_fullpath)

    for current_center_cords in current_center_cords_list:
        """
        An example here will be helpful. So you currently have a map at level 2 with pacthes at level 2 you want patches at level 1,
         the downsample from level 0 to level 2 is 16 while level 1 has a downsample factor have 4. First we want
        to get the cordinates in the lower level. We do this in steps 1. get center cords in level 0 by multiplying by 16 (downsample_current) 
        2. convert to lower_level cords by downsampling (divide by 4 downsample_factor_next_level)
        3. We then get the top_left and bottom_right cords for the lower_level (level 1) for the patch, here our patch size is 256 by 256
          0 / 0 - - - - - - - - - - - - -  
            | 
            | top left
            | (ci-128,cj-128)                                                     
            |                             
            |        (ci,cj)                   
            |                     (ci+128,cj+128)  bottom right of patch          
            v                              
        4. We know put the top left and bottom right cordinates of the patch in level 0 so multiply by the downsample
            factor
        5. We then get the patch region using read_region
        6. convert from RGBA to RGB
        7. The patch at the lower level is save with the cordinates from the higher level
            so it is easy to identify which patchs have the same center pixels across levels
        """

        # level 0 center cords
        center_level_zero_cords = (current_center_cords[0] * downsample_current,
                                   current_center_cords[1] * downsample_current)  # this puts the cords in level 0
        # lower (next level) center cords
        center_level_lower_cords = (center_level_zero_cords[0] / downsample_factor_lower_level,
                                    center_level_zero_cords[1] / downsample_factor_lower_level)
        # top left corener in next leve cords
        top_left_lower_level_cords = (
            center_level_lower_cords[0] - hp.half_p_size, center_level_lower_cords[1] - hp.half_p_size)
        # bottom right corner in next level cords
        bottom_right_lower_level_cords = (
            center_level_lower_cords[0] + hp.half_p_size, center_level_lower_cords[1] + hp.half_p_size)
        # Put lower level top left coordinates into level 0
        top_left_lv0 = (int(top_left_lower_level_cords[0] * downsample_factor_lower_level),
                        int(top_left_lower_level_cords[1] * downsample_factor_lower_level))
        # get patch name
        patch_name = slide_name_no_ext + '_' + str(current_center_cords[0]) + '_' + str(
            current_center_cords[1]) + cf.patch_file_type
        patch_name = os.path.join(patch_path, patch_name)
        # the read_region must be in level 0 coordinates
        patch = slide.read_region(top_left_lv0, lower_level, (hp.patch_size, hp.patch_size))
        # the following code is to make sure we save the patch only with the RGB
        patch_pil_img = np.array(patch)
        patch_pil_img = patch_pil_img[:, :, :3]  # we only want the RGB from the RGBA image
        Image.fromarray(patch_pil_img).save(patch_name)
        # here the cords all must be in the current level so it easy to see patches across different levels
        # We are doing the sampling at once here mutilpying by the lower_level downsample factor and dividing
        # by the current downsample factor
        top_left_current_level = (
            int(top_left_lower_level_cords[0] * (downsample_factor_lower_level / downsample_current)),
            int(top_left_lower_level_cords[1] * (downsample_factor_lower_level / downsample_current)))
        bottom_right_current_level = (
            int(bottom_right_lower_level_cords[0] * (downsample_factor_lower_level / downsample_current)),
            int(bottom_right_lower_level_cords[1] * (downsample_factor_lower_level / downsample_current)))
        cv2.rectangle(slide_map, top_left_current_level, bottom_right_current_level, BLUE, THICKNESS)


def extract_tumor_boundary_patch_from_tumor_slide(slide_name, mask_level, downsample_factor,patch_kept_center_cords, more_patches):
    """ Extract tumor patches using tumor slides
    Args:
        :param slide_name: name of slide used
        :param downsample_factor this is the factor that corresponds to the downsamply for the given mask level
        for instance at level 3 the downsample_factor is ~ 32. We need this because near the end of this function
        slide.read_region needs the top left corner cordinates in level 0 resolution. hence we have to upsample
        :param mask_level: level of mask should be an int
        :param xml_data (xml_file): this is xml dataset that denotes where the tumor is in the tumor slide
        note, that you can only get boundary patches from tumor slides becuase there is tumor and normal dataset.
        :returns a list of all the center cordinates of the patches.
    Main idea:
        we import the xml dataset and use cordinates as the center pixels for patch generation. We could even use
        random sampling from the xml cordinates to make the boundar patches. The purpose here was to get more patches
        along the boundary for tumor slides to increase the model accuracy. Becuase boundary patches would be the most
        difficult for the model to predict
    """

    # print("the downsample factor is ", downsample_factor)
    # print("the width is ", width)
    # print("the height is ", height)
    # read in the xml cordinates
    xml_fname = os.path.join(cf.xml_path, slide_name).replace(cf.image_file_type, '.xml')
    print(" the xml file path is ", xml_fname)
    xml_cords_boundary_tumor, _ = read_xml(xml_fname, downsample_factor=downsample_factor)
    # at this point xml_cords should contain what I want then just need to sample from there
    # the main while loop needs to change. something like use the first xml center cord as a base randomly
    # sample some center pixels get the patches then move on to the next xml_center cord. this should increase
    # the number of boundary patches
    slide_name_no_ext = slide_name.replace(cf.image_file_type, '')
    dir_name = 'tumor_lv' + str(mask_level)
    tumor_patch_path = os.path.join(cf.patch_path, dir_name)
    if not os.path.isdir(tumor_patch_path):
        os.makedirs(tumor_patch_path)

    slide, slide_map, tissue_mask = MapMaskHelpers.return_slides_masks(slide_name=slide_name,binary_tumor=1)
    # one since a tumor
    tumor_mask = MapMaskHelpers.get_slide_tumor_mask(slide_name)

    t_cnt = 0
    print('{}: extracting tumor boundary tumor patches'.format(slide_name_no_ext))
    num_tries = 0
    patch_tumor_threshold_upper = 0.75
    patch_tumor_threshold_lower = 0.40 # 0.20 for less restrive
    patch_tumor_threshold_lower_less_restrictive = 0.20 # 0.20 for less restrive

    while (t_cnt != (cf.tumor_boundary_patches_per_slide + more_patches) and (num_tries <= hp.max_tries_boundary)):
        """
        This for loop will randomly pick the center pixels. Then adjustments will be made to get the patch. 
        This includes generating random numbers between -1500, 1500 and adding them to the xml cord which is the
        boundary. 1500 was selected so that we remain near the boundary. Later in the while loop we check for previous
        patch overlap, calculate the amount of tissue and the amount of tumor and these need to meet certain threshold
        in order the patch to be selected. I.e we do not want a patch that is just background. 
        """
        num_tries += 1
        # this number is to keep the patch close to the boundary if we increase this then we can
        # deviate further from the boundary which is not desired
        ci = random.randint(-hp.boundary_range,
                            hp.boundary_range)  # randint(a, b): a <= N <= b ci = random.randint(0, width - 2)
        cj = random.randint(-hp.boundary_range,
                            hp.boundary_range)  # randint(a, b): a <= N <= b cj = random.randint(0, height - 2)

        if num_tries >= (hp.max_tries_boundary * 0.80):
            ci = random.randint(-10,10)  # randint(a, b): a <= N <= b ci = random.randint(0, width - 2)
            cj = random.randint(-10,10)
        # (ci,cj) are the randomly selected center pixel cordinates for the patch
        # ci corresponds to the x-axis
        # cj corresponds to the y-axis
        # we will take those center pixels and make a square patch

        # Here we are picking random xml coordinates for the tumor segmentation

        # print("length of xml_cords_boundary is: ",len(xml_cords_boundary[0]))
        # print(xml_cords_boundary)

        # we have a list of lists of cords one for each tumor
        num_splines = len(xml_cords_boundary_tumor)-1
        rand_spline_idx = random.randint(0, num_splines)
        # rand_idx = num_tries % len(xml_cords_boundary_tumor[rand_spline_idx])
        rand_idx = random.randint(0, len(xml_cords_boundary_tumor[rand_spline_idx]) -1)
        xml_cord = xml_cords_boundary_tumor[rand_spline_idx][rand_idx]
        # print("xml cords is")
        # print(xml_cord)
        bound_x = xml_cord[0]
        bound_y = xml_cord[1]
        # print(rand_idx)
        # print("example of bound x", bound_x)
        # print("example of bound y", bound_y)
        # exit()
        # print("example of ci", ci)
        # print("example of cj", cj)
        ci = abs(int(bound_x) + ci)
        cj = abs(int(bound_y) + cj) # need absolute so do not have a negative pixel cordinate value
        # print("example of bound x", bound_x)
        # print("example of bound y", bound_y)

        patch_tissue_ratio, patch_tumor_ratio = MapMaskHelpers.return_ratios(ci, cj, binary_tissue=0,
                                                                             tissue_mask=tissue_mask, mask=tumor_mask)
        use_patch_cords = False  # flag variable,
        if patch_tumor_threshold_lower < patch_tumor_ratio < patch_tumor_threshold_upper and patch_tissue_ratio > hp.patch_tissue_threshold:
            """
            the thresholds are a best guess and are hyper parameters for patch extraction
            """
            use_patch_cords = MapMaskHelpers.check_overlap(ci=ci,cj=cj,previous_center_cords=patch_kept_center_cords,
                                                           overlap=1)

        # here we are losenning the requirements for the tissue requirements
        if num_tries >= (hp.max_tries_boundary / 2):
            if patch_tumor_threshold_lower_less_restrictive < patch_tumor_ratio < patch_tumor_threshold_upper and patch_tissue_ratio > hp.patch_tissue_threshold:
                use_patch_cords = MapMaskHelpers.check_overlap(ci=ci, cj=cj,
                                                               previous_center_cords=patch_kept_center_cords,
                                                               overlap=1)

        if num_tries >= (hp.max_tries_boundary * 0.85):
            if 0.10 < patch_tumor_ratio < 0.90 and patch_tissue_ratio > 0.25:
                use_patch_cords = MapMaskHelpers.check_overlap(ci=ci, cj=cj,
                                                               previous_center_cords=patch_kept_center_cords,
                                                               overlap=1)

        if use_patch_cords:
            # print("xml cords is ")
            # print(xml_cord)
            # print(" example of ci used", ci)
            # print(" example of cj used", cj)
            # print("example of bound x", bound_x)
            # print("example of bound y", bound_y)

            patch_kept_center_cords.append((ci, cj))
            patch_name = slide_name_no_ext + '_' + str(ci) + '_' + str(cj) + '_tumor_boundary' + cf.patch_file_type
            patch_name = os.path.join(tumor_patch_path, patch_name)
            patch_pil_img, top_left, bottom_right = MapMaskHelpers.return_patch_from_slide(slide=slide,
                                                                                           mask_level=mask_level,
                                                                                           ci=ci, cj=cj,
                                                                                           downsample_factor=downsample_factor)

            Image.fromarray(patch_pil_img).save(patch_name)

            cv2.rectangle(slide_map, top_left, bottom_right, hp.purple, hp.patch_thickness)
            t_cnt += 1
            # progress_bar.update(1)
    extra_count = 0
    for xml_cord in xml_cords_boundary_tumor:
        if extra_count == 10:
            break
        rand_idx = random.randint(0, len(xml_cord) - 1)
        xml_cords = xml_cord[rand_idx]
        # print("xml cords is")
        # print(xml_cord)
        ci = xml_cords[0]
        cj = xml_cords[1]
        patch_kept_center_cords.append((ci, cj))
        patch_name = slide_name_no_ext + '_' + str(ci) + '_' + str(cj) + '_tumor_boundary' + cf.patch_file_type
        patch_name = os.path.join(tumor_patch_path, patch_name)
        patch_pil_img, top_left, bottom_right = MapMaskHelpers.return_patch_from_slide(slide=slide,
                                                                                       mask_level=mask_level,
                                                                                       ci=ci, cj=cj,
                                                                                       downsample_factor=downsample_factor)

        Image.fromarray(patch_pil_img).save(patch_name)
        extra_count +=1
        cv2.rectangle(slide_map, top_left, bottom_right, hp.brown, hp.patch_thickness)

    MapMaskHelpers.write_slide_post_map(slide_name, slide_map)
    print('{}: extracted: boundary tumor = {}'.format(slide_name_no_ext, t_cnt))
    # progress_bar.close()
    return patch_kept_center_cords, t_cnt


def extract_nontumor_boundary_patch_from_tumor_slide(slide_name, mask_level, downsample_factor,patch_kept_center_cords):
    """ Extract tumor patches using tumor slides
    Args:
        :param slide_name: name of slide used
        :param downsample_factor this is the factor that corresponds to the downsamply for the given mask level
        for instance at level 3 the downsample_factor is ~ 32. We need this because near the end of this function
        slide.read_region needs the top left corner cordinates in level 0 resolution. hence we have to upsample
        :param mask_level: level of mask should be an int
        :param xml_data (xml_file): this is xml dataset that denotes where the tumor is in the tumor slide
        note, that you can only get boundary patches from tumor slides becuase there is tumor and normal dataset.
        :returns a list of all the center cordinates of the patches.
    Main idea:
        we import the xml dataset and use cordinates as the center pixels for patch generation. We could even use
        random sampling from the xml cordinates to make the boundar patches. The purpose here was to get more patches
        along the boundary for tumor slides to increase the model accuracy. Becuase boundary patches would be the most
        difficult for the model to predict
    """

    # read in the xml cordinates
    xml_fname = os.path.join(cf.xml_path, slide_name).replace(cf.image_file_type, '.xml')
    _, xml_cords_boundary_nontumor = read_xml(xml_fname, downsample_factor=downsample_factor)
    # at this point xml_cords should contain what I want then just need to sample from there
    # the main while loop needs to change. something like use the first xml center cord as a base randomly
    # sample some center pixels get the patches then move on to the next xml_center cord. this should increase
    # the number of boundary patches
    # patch_kept_center_cords = []

    # print(xml_cords_boundary_nontumor)
    slide_name_no_ext = slide_name.replace(cf.image_file_type, '')
    dir_name = 'tumor_lv' + str(mask_level)
    tumor_patch_path = os.path.join(cf.patch_path, dir_name)
    if not os.path.isdir(tumor_patch_path):
        os.makedirs(tumor_patch_path)

    slide, slide_map, tissue_mask = MapMaskHelpers.return_slides_masks(slide_name=slide_name,binary_tumor=1)
    # one since a tumor
    tumor_mask = MapMaskHelpers.get_slide_tumor_mask(slide_name)

    t_cnt = 0

    print('{}: extracting notumor boundary tumor patches'.format(slide_name_no_ext))
    num_tries = 0
    # progress_bar = tqdm(total=cf.nontumor_boundary_patches_per_slide + 1)
    patch_tumor_threshold_upper = 0.75
    patch_tumor_threshold_lower = 0.40
    patch_tumor_threshold_lower_less_restrictive = 0.20

    while (t_cnt != cf.nontumor_boundary_patches_per_slide) and (num_tries <= hp.max_tries_boundary):
        """
        This for loop will randomly pick the center pixels. Then adjustments will be made to get the patch. 
        This includes generating random numbers between -1500, 1500 and adding them to the xml cord which is the
        boundary. 1500 was selected so that we remain near the boundary. Later in the while loop we check for previous
        patch overlap, calculate the amount of tissue and the amount of tumor and these need to meet certain threshold
        in order the patch to be selected. I.e we do not want a patch that is just background. 
        """
        num_tries += 1
        ci = random.randint(-hp.boundary_range,
                            hp.boundary_range)  # randint(a, b): a <= N <= b ci = random.randint(0, width - 2)
        cj = random.randint(-hp.boundary_range,
                            hp.boundary_range)  # randint(a, b): a <= N <= b cj = random.randint(0, height - 2)
        # this number is to keep the patch close to the boundary if we increase this then we can
        # deviate further from the boundary which is not desired

        # (ci,cj) are the randomly selected center pixel cordinates for the patch
        # ci corresponds to the x-axis
        # cj corresponds to the y-axis
        # we will take those center pixels and make a square patch

        # Here we are picking random xml coordinates for the tumor segmentation

        # print("length of xml_cords_boundary is: ",len(xml_cords_boundary[0]))
        # print(xml_cords_boundary)

        # we have a list of lists of cords one for each tumor
        num_splines = len(xml_cords_boundary_nontumor)-1
        rand_spline_idx = random.randint(0, num_splines)
        # rand_idx = num_tries % len(xml_cords_boundary_tumor[rand_spline_idx])
        rand_idx = random.randint(0, len(xml_cords_boundary_nontumor[rand_spline_idx]) -1)
        xml_cord = xml_cords_boundary_nontumor[rand_spline_idx][rand_idx]
        # print("xml cords is")
        # print(xml_cord)
        bound_x = xml_cord[0]
        bound_y = xml_cord[1]
        # print(rand_idx)
        # print("example of bound x", bound_x)
        # print("example of bound y", bound_y)
        # exit()

        ci = abs(int(bound_x) + ci)
        cj = abs(int(bound_y) + cj)

        patch_tissue_ratio, patch_tumor_ratio = MapMaskHelpers.return_ratios(ci, cj, binary_tissue=0,
                                                                             tissue_mask=tissue_mask, mask=tumor_mask)
        use_patch_cords = False  # flag variable, we will not use the patch cords if it results in 50% overlap with existing patches.
        if patch_tumor_threshold_lower < patch_tumor_ratio < patch_tumor_threshold_upper and patch_tissue_ratio > hp.patch_tissue_threshold:

            use_patch_cords = MapMaskHelpers.check_overlap(ci=ci,cj=cj,previous_center_cords=patch_kept_center_cords,
                                                           overlap=1)

        # here we are losenning the requirements for the tissue requirements
        if num_tries >= (hp.max_tries_boundary / 2):
            if patch_tumor_threshold_lower_less_restrictive < patch_tumor_ratio < patch_tumor_threshold_upper and patch_tissue_ratio > hp.patch_tissue_threshold:
                use_patch_cords = MapMaskHelpers.check_overlap(ci=ci, cj=cj,
                                                               previous_center_cords=patch_kept_center_cords,
                                                               overlap=1)

        if use_patch_cords:
            patch_kept_center_cords.append((ci, cj))
            patch_name = slide_name_no_ext + '_' + str(ci) + '_' + str(cj) + '_nontumor_boundary'+cf.patch_file_type
            patch_name = os.path.join(tumor_patch_path, patch_name)
            patch_pil_img, top_left, bottom_right = MapMaskHelpers.return_patch_from_slide(slide=slide,
                                                                                           mask_level=mask_level,
                                                                                           ci=ci, cj=cj,
                                                                                           downsample_factor=downsample_factor)

            Image.fromarray(patch_pil_img).save(patch_name)

            cv2.rectangle(slide_map, top_left, bottom_right, hp.black, hp.patch_thickness)
            t_cnt += 1
            # progress_bar.update(1)

    MapMaskHelpers.write_slide_post_map(slide_name, slide_map)
    print('{}: extracted: boundary nontumor = {}'.format(slide_name_no_ext, t_cnt))
    return patch_kept_center_cords, t_cnt



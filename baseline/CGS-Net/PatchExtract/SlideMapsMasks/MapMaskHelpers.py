"""
This script contains functions that just get the masks and maps
"""
from PatchExtract.parameters import cf, hp
import os
from xml.etree.ElementTree import parse
import tiffslide
import numpy as np
import tifffile


def read_xml_emmc(target_xml_path, downsample_factor):
    """
    Description: a mask_level = 0 means that we are using the full slide image with no
    down sampling. A mask_level = 1 means that we downsample by 2
    mask_level = 2 means that we downsample by 4
    :param target_xml_path: the file location of the xml dataRaw
    :param downsample_factor: for WSI images this number corresponds to the slide.level_downsamples
    mask_level: the resolution detail, 0 means full resolution higher numbers means more coarse resolution
    :return: the downsampled cordinate list
    """
    # xml stuff open the xml file in a broswer to get a better understanding
    tree = parse(target_xml_path)
    root = tree.getroot()
    # this holds all the xml cordinates of the anotated region of the slide
    coors_list = []

    for areas in root.iter('Coordinates'):  # see xml file this a the tag we want
        coors = []  # temp list to hold modified xml cordinates
        for area in areas:
            x = round(float(area.attrib["X"]) / downsample_factor)  # the actual downsampling
            y = round(float(area.attrib["Y"]) / downsample_factor)
            coors.append([x, y])
        coors_list.append(coors)
    return coors_list, None


def read_xml_c16(target_xml_path, downsample_factor):
    """
    Description: a mask_level = 0 means that we are using the full slide image with no
    down sampling.
    :param target_xml_path: the file location of the xml dataRaw
    :param downsample_factor: for WSI images this number corresponds to the slide.level_downsamples
    mask_level: the resolution detail, 0 means full resolution higher numbers means more coarse resolution
    :return: the downsampled cordinate list

    Note that the xml files for c16 have three groups. groups _0 and _1 mark cancer regions while
    group _2 are nontumor regions with the tumor regions. So we basically repeat the read xml fucntion twice
    one for nontumor and one for tumor. we collect both sets of cords and return them
    """
    # xml stuff open the xml file in a broswer to get a better understanding
    tree = parse(target_xml_path)
    root = tree.getroot()
    # this holds all the xml cordinates of the anotated region of the slide
    coors_list_tumor = []
    coors_list_nontumor = []
    root_anno = list(root)[0]
    for annotation in root_anno:
        # print(annotation.attrib['PartOfGroup'])
        if annotation.attrib['PartOfGroup'] == "_0" or annotation.attrib['PartOfGroup'] == "_1" or annotation.attrib['PartOfGroup'] == "Tumor":
            for areas in annotation.iter('Coordinates'):
                coors = []  # temp list to hold modified xml cordinates
                # print(areas)
                for area in areas:
                    # print(area.attrib["PartOfGroup"])
                    x = round(float(area.attrib["X"]) / downsample_factor)  # the actual downsampling
                    y = round(float(area.attrib["Y"]) / downsample_factor)
                    coors.append([x, y])
                coors_list_tumor.append(coors)
        elif annotation.attrib['PartOfGroup'] == "_2" or annotation.attrib['PartOfGroup'] == "Exclusion":
            for areas in annotation.iter('Coordinates'):
                coors = []  # temp list to hold modified xml cordinates
                # print(areas)
                for area in areas:
                    # print(area.attrib["PartOfGroup"])
                    x = round(float(area.attrib["X"]) / downsample_factor)  # the actual downsampling
                    y = round(float(area.attrib["Y"]) / downsample_factor)
                    coors.append([x, y])
                coors_list_nontumor.append(coors)
    return coors_list_tumor, coors_list_nontumor


def get_slide_pre_map(slide_name, path=None):
    if path is None:
        path = cf.map_path
    # slide_name_no_ext = slide_name.replace('.tif', '')
    map_fname = os.path.join(path, 'map_' + slide_name)
    slide_map = "unboundlocalError"
    if not os.path.isfile(map_fname):
        print('Error: file {} does not exist'.format(map_fname))
    else:
        slide_map = tifffile.imread(map_fname)
        # slide_map = cv2.imread(map_fname, -1)  # -1 loads the image "as is"
    return slide_map


def get_slide_post_map(slide_name, path=None):
    if path is None:
        path = cf.map_path
    # slide_name_no_ext = slide_name.replace('.tif', '')
    map_fname = os.path.join(path, 'post_map_' + slide_name)
    if not os.path.isfile(map_fname):
        slide_map = get_slide_pre_map(slide_name)
    else:
        slide_map = tifffile.imread(map_fname)
        # slide_map = cv2.imread(map_fname, -1)  # -1 loads the image "as is"
    return slide_map


def get_slide_normal_mask(slide_name, path=None):
    if path is None:
        path = cf.mask_path
    # slide_name_no_ext = slide_name.replace('.tif', '')
    normal_mask_fname = os.path.join(path, 'normal_mask_' + slide_name)
    if not os.path.isfile(normal_mask_fname):
        print('Error: file {} does not exist'.format(normal_mask_fname))
    else:
        normal_mask = tifffile.imread(normal_mask_fname)
        #TODO: there is an error here with the normal mask, it is not being read correctly instead of 225 for the mask,
        # -1 is being loaded. Hence my function return_ratios, is incocrrect. A hack fix is just checking if the
        # the sum of the mask is negat then multiplying by -255. This is a hack fix and should be fixed. See
        # return_ratios for the hack fix.
        # normal_mask = cv2.imread(normal_mask_fname, cv2.IMREAD_GRAYSCALE)  # reads image as grayscale
        return normal_mask


def get_slide_tumor_mask(slide_name, path=None):
    if path is None:
        path = cf.mask_path

    # slide_name_no_ext = slide_name.replace('.tif', '')
    tumor_mask_fname = os.path.join(path, 'tumor_mask_' + slide_name)
    if not os.path.isfile(tumor_mask_fname):
        print('Error: file {} does not exist'.format(tumor_mask_fname))
    else:
        tumor_mask = tifffile.imread(tumor_mask_fname)
        # tumor_mask = cv2.imread(tumor_mask_fname, cv2.IMREAD_GRAYSCALE)  # reads image as grayscale
        return tumor_mask


def get_slide_tissue_mask(slide_name, path=None):
    if path is None:
        path = cf.mask_path
    # slide_name_no_ext = slide_name.replace('.tif', '')
    tissue_mask_fname = os.path.join(path, 'tissue_mask_' + slide_name)
    if not os.path.isfile(tissue_mask_fname):
        print('Error: file {} does not exist'.format(tissue_mask_fname))
    else:
        tissue_mask = tifffile.imread(tissue_mask_fname)
        # tissue_mask = cv2.imread(tissue_mask_fname, cv2.IMREAD_GRAYSCALE)  # reads image as grayscale
        return tissue_mask


def write_slide_post_map(slide_name, slide_map):
    # slide_name_no_ext = slide_name.replace('.tif', '')
    map_fname = os.path.join(cf.map_path, 'post_map_' + slide_name)
    tifffile.imwrite(map_fname, data=slide_map)
    # cv2.imwrite(map_fname, slide_map)


def return_slides_masks(slide_name, binary_tumor):
    """
    Do not put the tumor mask or normal mask in this function. It is very easy
    for there to be an error and provide the wrong mask. The three items returned here
    are always the same regardless of the sitatuion. They only depend on if the slide is
    a tumor or not.
    :param slide_name:
    :param binary_tumor:
    :return:
    """

    if binary_tumor:
        slide_fullpath = os.path.join(cf.tumor_slide_path, slide_name)
        # print("bin_tumor = true and the slide path is ", slide_fullpath)
        slide = tiffslide.open_slide(slide_fullpath)
        slide_map = get_slide_post_map(slide_name)
        tissue_mask = get_slide_tissue_mask(slide_name)
    else:
        slide_fullpath = os.path.join(cf.nontumor_slide_path, slide_name)
        # print("bin_tumor = False and the slide path is ", slide_fullpath)
        slide = tiffslide.open_slide(slide_fullpath)
        slide_map = get_slide_post_map(slide_name)
        tissue_mask = get_slide_tissue_mask(slide_name)

    return slide, slide_map, tissue_mask


def return_ratios(ci, cj, binary_tissue, tissue_mask, mask=0):
    # Here we're getting the column pixel where the patch starts
    start_col = ci - hp.half_p_size
    # The column pixel where the patch ends
    end_col = ci + hp.half_p_size
    # start and end row follow the same idea
    start_row = cj - hp.half_p_size
    end_row = cj + hp.half_p_size

    mask_max = hp.patch_size * hp.patch_size * 255  # this is what a mask that is all one class
    patch_tissue_mask_sum = tissue_mask[start_row:end_row, start_col:end_col].sum()
    patch_tissue_ratio = patch_tissue_mask_sum / mask_max
    if binary_tissue:
        return patch_tissue_ratio
    else:
        # now with these coordinates I want to determine how much of the patch is actually tumor or normal
        assert mask.size != 0
        patch_mask_sum = mask[start_row:end_row,
                         start_col:end_col].sum()  # here every white pixel is 255 so the max
        patch_ratio = patch_mask_sum / mask_max

        if patch_mask_sum < 0:
            patch_mask_sum = patch_mask_sum * -255
            patch_ratio = patch_mask_sum / mask_max
        ### debug
        # if patch_tissue_ratio > 0.5:
        #     print(mask[start_row:end_row, start_col:end_col])
        #     print("patch mask sum is ", patch_mask_sum)
        #     print("the patch ratio is ", patch_ratio)
        return patch_tissue_ratio, patch_ratio


def return_patch_from_slide(slide, mask_level, ci, cj, downsample_factor):
    """

    :param slide: the slide which you want to extract a patch from
    :param mask_level: the level of the patch that you want
    :param ci: proposed center pixel
    :param cj: same
    :param downsample_factor: from level 0 reference frame
    :return:

    """
    # downsample_factor =1
    start_col = ci - hp.half_p_size
    end_col = ci + hp.half_p_size
    start_row = cj - hp.half_p_size
    end_row = cj + hp.half_p_size
    top_left = (start_col, start_row)  # (X, Y) coordinates, not matrix row-col indices
    bottom_right = (end_col, end_row)
    # Always fixed to zero this always get the patches from the highest resolution.
    # I change the parameter here to get patch at different level
    # The tuple for top_left needs to be the top left in the 0 reference frame

    start_col_lv0 = start_col * downsample_factor
    start_row_lv0 = start_row * downsample_factor
    try:
        assert isinstance(start_col_lv0, int) is True and isinstance(start_row_lv0, int) == True
    except AssertionError:
        print("the top left cords are not integers check the parameters.py")
        print("Also check this function, the slide_region only accepts ints")
        raise

    top_left_lv0 = (start_col_lv0, start_row_lv0)
    patch = slide.read_region(top_left_lv0, mask_level, (hp.patch_size, hp.patch_size))
    # at this point patch is RGBA and PIL would have trouble saving the patch because of the A
    # sometimes the patch was checkboard (transparent) other times not.
    # The code below removes the Alpha from patch and then saves it
    patch_pil_img = np.array(patch)
    patch_pil_img = patch_pil_img[:, :, :3]  # we only want the RGB from the RGBA image
    return patch_pil_img, top_left, bottom_right


def check_overlap(ci, cj, previous_center_cords, overlap=1):
    use_patch_cords = False  # flag variable, we will not use the patch cords if it
    # results in 50% overlap with existing patches
    if not previous_center_cords:  # check if the list is empty
        use_patch_cords = True
    # here we are checking the overlap by comparing center pixels with previous center pixels
    # if the distance in any direction is greater than half the patch size
    # then the maximum overlap is a little under 50%
    for prev_ci, prev_cj in previous_center_cords:
        if abs(ci - prev_ci) > (hp.half_p_size * overlap):
            use_patch_cords = True
        elif abs(cj - prev_cj) > (hp.half_p_size * overlap):
            use_patch_cords = True
        else:
            use_patch_cords = False
            break
    return use_patch_cords


def close_slide(slide):
    """

    :param slide: This needs to be an OpenSlide object
    :return: nothing just closes the OpenSlide object
    """
    slide.close()

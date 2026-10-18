from PatchExtract.SlideMapsMasks.MapMaskGen import main as make_map_mask
from PatchExtract.PatchGen.ExtractPatches import extract_tumor_patch_from_tumor_slide as collect_tumor_patch
from PatchExtract.parameters import cf, hp
from PatchExtract.PatchGen.ExtractPatches import \
    extract_normal_patch_from_tumor_slide as collect_normal_patch_from_tumor_slide
from PatchExtract.PatchGen.ExtractPatches import extract_normal_patch_from_normal_slide as collect_normal_patch
from PatchExtract.PatchGen.ExtractPatches import \
    extract_tumor_boundary_patch_from_tumor_slide as collect_tumor_boundary_patch
from PatchExtract.PatchGen.ExtractPatches import \
    extract_nontumor_boundary_patch_from_tumor_slide as collect_nontumor_boundary_patch
from PatchExtract.PatchGen.PatchMaskGen import make_patch_mask
import os
from PatchExtract.SlideMapsMasks.MapMaskHelpers import read_xml_c16 as read_xml


def main(slide, FLAG_MASKS, FLAG_PATCH, FLAG_PATCH_MASK) -> None:
    """
    This is what Caller.py calls, bin_tumor is the binary tumor. with the flags for the other operations
    The first if statement is to generate the slide level masks for each slide (very large files). There
    are three types of masks, a tumor, normal and tissue masks.
    """
    slide_name = slide[0]
    bin_tumor = slide[1]
    ONLY_TUMOR_PATCH = slide[2]
    if FLAG_MASKS:
        print("working on the mask for ", slide_name, flush=True)
        make_map_mask(slide_name, bin_tumor, hp.resolution_levels[hp.level], hp.level)
    if FLAG_PATCH:
        # this is the only section that depends on only Tumor patches or not
        # This section needed the ONLY_TUMOR_PATCH flag since there are some tumor slides that have tumor regions
        # that are not marked by the pathologist. So if we extract a nontumor patch, we could actually be extracting
        # a tumor patch since all the tumors are not marked.
        print("working on the patches for ", slide_name, flush=True)
        make_patches(slide_name, bin_tumor, only_tumor=ONLY_TUMOR_PATCH)
    if FLAG_PATCH_MASK:
        # here we are just creating the patch level masks. this is the simplest section of the three.
        print("working on the patch masks for ", slide_name, flush=True)
        make_patch_masks(slide_name, bin_tumor)


def make_patches(slide_name, bin_tumor, only_tumor=False) -> None:
    """
        Parameters:
            slide_name str: the name of the slide
            bin_tumor int: a binary variable specifying is the slide is a tumor slide (one yes and 0 no)
            only_tumor bool: only run code to make tumor patches from the slide.

        Returns:
            None

        This function makes the patches per slide.

    """
    if only_tumor:
        # see earlier comments about only_tumor. only one type of patches generated for these slides.
        tumor_patches(slide_name=slide_name)
        return

    if bin_tumor:
        # we are dealing with a tumor slide and need to grab tumor patches.
        tumor_patches(slide_name=slide_name)
        # Get the normal patches from a tumor slide
        collect_normal_patch_from_tumor_slide(slide_name, mask_level=hp.level,
                                              downsample_factor=hp.resolution_levels[hp.level])

    else:
        # from a nontumor slide we only grab normal patches
        collect_normal_patch(slide_name, mask_level=hp.level,
                             downsample_factor=hp.resolution_levels[hp.level])


def make_patch_masks(slide_name, bin_tumor) -> None:
    if bin_tumor:
        # now make the patch level maks for the tumor patches from the tumor slide
        make_patch_mask(slide=slide_name, mask_path=cf.mask_path, patch_source=cf.tumor_patch_path,
                        current_cord_lvl=hp.resolution_levels[hp.level],
                        patch_size=hp.patch_size, patch_dir=cf.patch_path,
                        downsample_factor=hp.resolution_levels[hp.level])
        # make the patch masks for the normal patches from the tumor slide
        make_patch_mask(slide=slide_name, mask_path=cf.mask_path, patch_source=cf.normal_patch_path,
                        current_cord_lvl=hp.resolution_levels[hp.level],
                        patch_size=hp.patch_size, patch_dir=cf.patch_path,
                        downsample_factor=hp.resolution_levels[hp.level])
        # note the coordinate_lvl is not changing throughout this process. We are generating the patch masks
        # for only a single level of patches.
    else:
        # This is for a nontumor slide
        make_patch_mask(slide=slide_name, mask_path=cf.mask_path, patch_source=cf.normal_patch_path,
                        current_cord_lvl=hp.resolution_levels[hp.level],
                        patch_size=hp.patch_size, patch_dir=cf.patch_path,
                        downsample_factor=hp.resolution_levels[hp.level])


def tumor_patches(slide_name) -> None:
    total_tumor_patches_collected = 0
    nontumor_boundary_patches_collected = 0
    center_cords_temp = []  # this list contains the center cords for every single patch that was selected
    # We are checking if the tumor slide has _02 group markings. If it does not then we skip it
    xml_fname = os.path.join(cf.xml_path, slide_name).replace(cf.image_file_type, '.xml')
    _, xml_cords_boundary_nontumor = read_xml(xml_fname, downsample_factor=hp.resolution_levels[hp.level])
    boundary_center_cords_nontumor = []
    if bool(xml_cords_boundary_nontumor):
        boundary_center_cords_nontumor, nontumor_boundary_patches_collected = collect_nontumor_boundary_patch(
            slide_name, mask_level=hp.level,
            downsample_factor=hp.resolution_levels[
                hp.level],
            patch_kept_center_cords=boundary_center_cords_nontumor)
    # Now we check how many patches we got compare to desired and try to get more on the next pass
    more_patches = cf.nontumor_boundary_patches_per_slide - nontumor_boundary_patches_collected
    total_tumor_patches_collected += nontumor_boundary_patches_collected
    if more_patches <= 0:
        more_patches = 0
    assert more_patches >= 0
    center_cords_temp = center_cords_temp + boundary_center_cords_nontumor
    boundary_center_cords_tumor, tumor_boundary_patches_collected = collect_tumor_boundary_patch(slide_name,
                                                                                                 mask_level=hp.level,
                                                                                                 downsample_factor=
                                                                                                 hp.resolution_levels[
                                                                                                     hp.level],
                                                                                                 patch_kept_center_cords=center_cords_temp,
                                                                                                 more_patches=more_patches)

    more_patches = cf.tumor_boundary_patches_per_slide - tumor_boundary_patches_collected
    total_tumor_patches_collected += tumor_boundary_patches_collected
    if more_patches <= 0:
        more_patches = 0
    assert more_patches >= 0
    center_cords_temp = center_cords_temp + boundary_center_cords_tumor
    # we add the center cords from the boundary patches
    collect_tumor_patch(slide_name, mask_level=hp.level,
                        downsample_factor=hp.resolution_levels[hp.level],
                        patch_kept_center_cords=center_cords_temp, more_patches=more_patches, current_num_patches_picked=total_tumor_patches_collected)

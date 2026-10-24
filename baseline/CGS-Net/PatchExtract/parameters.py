import os
import shutil


#  Note that some functions in utils.py have level in them.

class cf:
    """
    This class holds all correct file paths and needs to be setup before running the code, this class will be imported
    to all other files.
    Note that these are class variables.
    The file paths need to be full otherwise the code does not work
    """
    base_dir_output = os.path.join("/path/here/","patchrun")
    tumor_slide_path = "./slidedata/tumor/"
    nontumor_slide_path = "./slidedata/normal/"

    xml_path = "./slidedata/annotations"
    map_path = base_dir_output + "/map_lv2"
    mask_path = base_dir_output + "/mask_lv2"
    patch_path = base_dir_output + "/patch"
    tumor_patch_path = base_dir_output + "/patch/tumor_lv2"
    normal_patch_path = base_dir_output + "/patch/normal_lv2"
    normal_patch_mask_path = normal_patch_path + "_mask"
    tumor_patch_mask_path = tumor_patch_path + "_mask"

    # these variables determine how many patches we get WSI
    normal_patches_per_tumor_slide = 50
    normal_patches_per_normal_slide = 25
    tumor_patches_per_slide = 25
    tumor_boundary_patches_per_slide = 50
    nontumor_boundary_patches_per_slide = 25
    image_file_type = ".tif"
    patch_file_type = ".tif"

    @classmethod
    def rm_dir(cls, folder):
        """
        This is used when wanting to remove a folder that is filled with dataset when I want to rerun the code.
        The flag is so that I only have to change 1 variable to either remove the dirs or to keep them
        """
        for f in folder:
            shutil.rmtree(f)

    @classmethod
    def make_folder(cls, folder):
        """
        to just make the folders in the properties, note this can make folders outside the properites ie. cf.map_path
        """
        if not os.path.isdir(folder):
            os.makedirs(folder)

    @classmethod
    def check_folders(cls):
        """
        Checks if the map, mask, and patch folders all exist.
        """
        cls.make_folder(cls.map_path)
        cls.make_folder(cls.mask_path)
        cls.make_folder(cls.patch_path)
        cls.make_folder(cls.base_dir_output)

    @classmethod
    def update_cf(cls, new_lvl):
        """
        The point of this function is to switch between and higher and lower_levels, for example
        /dataRaw/testpatch/tumor_lv1 to /dataRaw/testpatch/tumor_lv0 and then back to /dataRaw/testpatch/tumor_lv1
        this makes it simple and I don't have to change much of the code in the functions because they are all
        calling this file
        """
        #
        base_map_path = cls.map_path.rpartition("/")[0]
        cls.map_path = os.path.join(base_map_path, "map_lv" + str(new_lvl))
        if not os.path.isdir(cls.map_path):
            os.makedirs(cls.map_path)
        #
        base_mask_path = cls.mask_path.rpartition("/")[0]
        cls.mask_path = os.path.join(base_mask_path, "mask_lv" + str(new_lvl))
        if not os.path.isdir(cls.mask_path):
            os.makedirs(cls.mask_path)
        #
        base_tumor_patch_path = cls.tumor_patch_path.rpartition("/")[0]
        cls.tumor_patch_path = os.path.join(base_tumor_patch_path, "tumor_lv" + str(new_lvl))
        if not os.path.isdir(cls.tumor_patch_path):
            os.makedirs(cls.tumor_patch_path)
        #
        base_normal_patch_path = cls.normal_patch_path.rpartition("/")[0]
        cls.normal_patch_path = os.path.join(base_normal_patch_path, "normal_lv" + str(new_lvl))
        if not os.path.isdir(cls.normal_patch_path):
            os.makedirs(cls.normal_patch_path)


class hp:
    """
    This class defines all the hyperparameters for the code
    These are class variables
    """

    patch_size = 224
    level = 2
    resolution_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    max_tries = 25000
    max_tries_boundary = 25000
    half_p_size = patch_size // 2
    boundary_range = half_p_size  # this is the number of pixels that we can deviate from in x and y directions for
    patch_tissue_threshold = 0.50
    normal_patch_normal_ratio_threshold = 0.75
    patch_thickness = 2  # for drawing on the map
    red = (0, 0, 255)  # for tumor patches
    green = (0, 255, 0)  # for normal patches
    purple = (191, 64, 191)  # for tumor boundary
    black = (0, 0, 0)  # for nontumor boundary within tumor
    brown = (128, 0, 0)  # for single patches sampled from xml with no requirements
    blue = (255, 0, 0)

    @classmethod
    def update_level(cls, new_level):
        cls.level = new_level

    @classmethod
    def check(cls):
        """
        This class checks the factors and makes sure that they are ints if not errors out
        :return:
        """
        try:
            for down_factor in cls.resolution_levels:
                assert isinstance(down_factor, int) == True
        except AssertionError:
            print("the factors are not integers check the parameters.py")
            raise

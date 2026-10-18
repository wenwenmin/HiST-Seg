import time
import numpy as np
from PatchExtract.parameters import cf, hp
from .MapMaskHelpers import read_xml_c16 as read_xml  # c16 because we are doing c16 dataset
from .MapMaskHelpers import close_slide
import os
import tiffslide
import tifffile
import cv2
from scipy import ndimage as ndi


def main(slidename, binary_tumor, downsamplefactor, mask_level):
    make_mask_timed(slidename, mask_level, binary_tumor, downsamplefactor)
    print("Done with the masks and maps for ", slidename)


def make_mask_timed(slide_name, mask_level, binary_tumor, downsamplefactor):
    """
    :param slide_name: the slide name
    :param mask_level: the level that corresponds to the downsampling 0 no downsamply 3 the most integer
    :param binary_tumor: True or False is a tumor slide
    :param downsamplefactor: the level of downsampling
    :return:
    """
    start = time.time()
    make_mask(slide_name, mask_level, binary_tumor, downsamplefactor)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{}: Masking time: {:0>2}:{:0>2}:{:05.2f}".format(slide_name, int(hours), int(minutes), seconds), flush=True)


def make_mask(slide_name, mask_level, binary_tumor, downsample_factor):
    """
    make tumor, normal, tissue mask using xml files and otsu threshold
    """

    # slide loading
    try:
        if binary_tumor:
            slide_fullpath = os.path.join(cf.tumor_slide_path, slide_name)
        else:
            slide_fullpath = os.path.join(cf.nontumor_slide_path, slide_name)
        print(slide_fullpath)
        slide = tiffslide.open_slide(slide_fullpath)  # this opens the slide but we need to close it
    except Exception:
        print("Unable to load slide check file path")
        raise

    # xml loading
    try:
        if binary_tumor:
            xml_fname = os.path.join(cf.xml_path, slide_name).replace(cf.image_file_type, '.xml')
            tumor_coors_list, nontumor_coors_list = read_xml(xml_fname, downsample_factor)
            print('{}: {} annotations'.format(slide_name, len(tumor_coors_list)))
        else:
            print("Normal selected")
            tumor_coors_list = []
            nontumor_coors_list = []
    except FileNotFoundError:
        print("check the path to the xml files, at this point the error is probably in read_xml")
        print(os.path.join(cf.xml_path, slide_name).replace(cf.image_file_type, '.xml'))  # xml_fname
        raise

    # check if map directory exists if not error out
    try:
        assert os.path.isdir(cf.map_path) == True
    except AssertionError:
        print("the map dir does not exist")
        raise

    # keep in mind that the map is the same size of the slide so that is why I'm getting the slide dimensions
    # to make the slide_map variable.
    map_fname = os.path.join(cf.map_path, 'map_' + slide_name)
    print(f"Making the map: {map_fname}")
    if not os.path.isfile(map_fname):  # if there is no file then make it
        # following line gets RGB thumbnail of dimension (width, height) belonging to a specific level
        # https://openslide.org/api/python/
        # slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[hp.map_level]))
        slide_map = np.float32(slide.get_thumbnail(slide.level_dimensions[hp.level]))
        print("the map was made ")
        # print("the shape of the slide map after create, ", slide_map.shape)
        # Check the earlier code, the coors_list is empty for normal slide becuase there are no tumor regions to
        # segment. The contours are the lines that mark where the tumor is on the slide.
        for coors in tumor_coors_list:
            # slide_map is the image, np.array([coors]) is the list of contours to draw
            # 255 is the color of the contours
            # 1 is the thickness of the contours
            cv2.drawContours(slide_map, np.array([coors]), -1, 255, 1)

        for coors in nontumor_coors_list:
            # slide_map is the image, np.array([coors]) is the list of contours to draw
            # 0 is the color of the contours
            # 1 is the thickness of the contours
            cv2.drawContours(slide_map, np.array([coors]), -1, 0, 1)

        slide_map = np.array(slide_map, dtype=np.uint8)
        tifffile.imwrite(map_fname, data=slide_map)
    else:
        print("the map already existed and was not remade")

    # check tumor mask / draw tumor mask
    # check if mask directory exists if not error out
    try:
        assert os.path.isdir(cf.mask_path) == True
    except AssertionError:
        print("the mask dir does not exist")
        raise
    tumor_mask_fname = os.path.join(cf.mask_path, 'tumor_mask_' + slide_name)

    if binary_tumor and not os.path.isfile(
            tumor_mask_fname):  # if the mask already exists then don't need to make it again

        print(f'working on the tumor mask: {tumor_mask_fname}', flush=True)
        tumor_mask = np.zeros(slide.level_dimensions[mask_level][::-1], dtype=np.int16)
        # slide.level_dimensions[mask_level] gets the pixel count for row and column
        # [::-1] reverses the list

        # the default is too large float64
        # if the code error outs here then try adding dtype=np.float32 to reduce the amount of memory needed
        # if float32 doesnot work try using np.int32
        # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
        for coors in tumor_coors_list:
            cv2.drawContours(tumor_mask, np.array([coors]), -1, 255, -1)

        for coors in nontumor_coors_list:
            cv2.drawContours(tumor_mask, np.array([coors]), -1, 0, -1)
        # tumor_mask.astype(np.uint8)
        # cv2.imwrite(tumor_mask_fname, tumor_mask)
        tifffile.imwrite(tumor_mask_fname, data=tumor_mask)

    # check tissue mask / draw tissue mask
    tissue_mask_fname = os.path.join(cf.mask_path, 'tissue_mask_' + slide_name)
    print(f'working on the tissue mask: {tissue_mask_fname}', flush=True)

    # this code always has to run so that we have the tissue_mask ready for the following chunck
    if not os.path.isfile(tissue_mask_fname):
        slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
        slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
        slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
        slide_lv = slide_lv[:, :, 1]
        _, tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # code here is for fat cells
        dilation = cv2.dilate(tissue_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)), iterations=1)
        tissue_mask = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100)))
        # we need to make the tissue mask boolean becuase thats what the following functions exptect
        # and they do not work otherwise
        tissue_mask = np.array(tissue_mask, dtype=bool)
        remove_small_objects(ar=tissue_mask, min_size=1000000, connectivity=1, out=tissue_mask)
        remove_small_holes(ar=tissue_mask, area_threshold=100000, connectivity=1, out=tissue_mask)
        tissue_mask = np.array(tissue_mask, dtype=np.uint8)
        tissue_mask = tissue_mask * 255
        tifffile.imwrite(tissue_mask_fname, data=tissue_mask)
        print("Everything done for tissue mask", flush=True)
    else:
        print(tissue_mask_fname, " already existed in the dir, we did not make it.")
        tissue_mask = tifffile.imread(tissue_mask_fname)

    normal_mask_fname = os.path.join(cf.mask_path, 'normal_mask_' + slide_name)
    if not os.path.isfile(normal_mask_fname):
        if binary_tumor:
            print(f'working on the normal mask: {normal_mask_fname}', flush=True)
            tumor_mask = tifffile.imread(tumor_mask_fname)

            tissue_mask = np.where(tumor_mask == 255, 0, tissue_mask)
            normal_mask = np.array(tissue_mask, dtype=np.int8)
            tifffile.imwrite(normal_mask_fname, data=normal_mask)
        else:
            tumor_mask = np.zeros(slide.level_dimensions[mask_level][::-1], dtype=np.int8)
            tifffile.imwrite(tumor_mask_fname, data=tumor_mask)

    close_slide(slide)


# the following two functions are taken from https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/morphology/misc.py#L141-L221
# and are needed for the tissue mask that includes fat cells
def remove_small_objects(ar, min_size=64, connectivity=1, *, out=None):
    """Remove objects smaller than the specified size.

    Expects ar to be an array with labeled objects, and removes objects
    smaller than min_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type is
        int, the ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `ar` is bool.
    out : ndarray
        Array of the same shape as `ar`, into which the output is
        placed. By default, a new array is created.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.

    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> d = morphology.remove_small_objects(a, 6, out=a)
    >>> d is a
    True

    """
    # Raising type error if not int or bool
    # _check_dtype_supported(ar)

    if out is None:
        out = ar.copy()
    else:
        out[:] = ar

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    if len(component_sizes) == 2 and out.dtype != bool:
        raise Exception("Only one label was provided to `remove_small_objects`. "
                        "Did you mean to use a boolean array?")

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def remove_small_objects(ar, min_size=64, connectivity=1, *, out=None):
    """Remove objects smaller than the specified size.

    Expects ar to be an array with labeled objects, and removes objects
    smaller than min_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type is
        int, the ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `ar` is bool.
    out : ndarray
        Array of the same shape as `ar`, into which the output is
        placed. By default, a new array is created.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.

    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> d = morphology.remove_small_objects(a, 6, out=a)
    >>> d is a
    True

    """
    # Raising type error if not int or bool
    # _check_dtype_supported(ar)

    if out is None:
        out = ar.copy()
    else:
        out[:] = ar

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    if len(component_sizes) == 2 and out.dtype != bool:
        raise Exception("Only one label was provided to `remove_small_objects`. "
                        "Did you mean to use a boolean array?")

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def remove_small_holes(ar, area_threshold=64, connectivity=1, *, out=None):
    """Remove contiguous holes smaller than the specified size.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the connected components of interest.
    area_threshold : int, optional (default: 64)
        The maximum area, in pixels, of a contiguous hole that will be filled.
        Replaces `min_size`.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel.
    out : ndarray
        Array of the same shape as `ar` and bool dtype, into which the
        output is placed. By default, a new array is created.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small holes within connected components removed.

    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[1, 1, 1, 1, 1, 0],
    ...               [1, 1, 1, 0, 1, 0],
    ...               [1, 0, 0, 1, 1, 0],
    ...               [1, 1, 1, 1, 1, 0]], bool)
    >>> b = morphology.remove_small_holes(a, 2)
    >>> b
    array([[ True,  True,  True,  True,  True, False],
           [ True,  True,  True,  True,  True, False],
           [ True, False, False,  True,  True, False],
           [ True,  True,  True,  True,  True, False]])
    >>> c = morphology.remove_small_holes(a, 2, connectivity=2)
    >>> c
    array([[ True,  True,  True,  True,  True, False],
           [ True,  True,  True, False,  True, False],
           [ True, False, False,  True,  True, False],
           [ True,  True,  True,  True,  True, False]])
    >>> d = morphology.remove_small_holes(a, 2, out=a)
    >>> d is a
    True

    Notes
    -----
    If the array type is int, it is assumed that it contains already-labeled
    objects. The labels are not kept in the output image (this function always
    outputs a bool image). It is suggested that labeling is completed after
    using this function.

    """
    # _check_dtype_supported(ar)

    # Creates warning if image is an integer image
    if ar.dtype != bool:
        raise Exception("Any labeled images will be returned as a boolean array. "
                        "Did you mean to use a boolean array?", UserWarning)

    if out is not None:
        if out.dtype != bool:
            raise TypeError("out dtype must be bool")
    else:
        out = ar.astype(bool, copy=True)

    # Creating the inverse of ar
    np.logical_not(ar, out=out)

    # removing small objects from the inverse of ar
    out = remove_small_objects(out, area_threshold, connectivity, out=out)

    np.logical_not(out, out=out)

    return out

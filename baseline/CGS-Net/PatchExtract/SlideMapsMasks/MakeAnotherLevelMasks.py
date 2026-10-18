#readData OpenCV TIFF: TIFFRGBAImageOK: Sorry, can not handle images with 64-bit samples
# global /tmp/pip-req-build-ms668fyv/opencv/modules/imgcodecs/src/grfmt_tiff.cpp (457)
from PatchExtract.parameters import cf, hp
from tqdm import tqdm
import MapMaskGen
import random
import argparse

random.seed(a=48)

parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int,
                    help="slide to start for manual multithreading", required=True)
parser.add_argument("--end_idx", type=int,
                    help="slide to end for manual multithreading", required=True)


args = parser.parse_args()
start_idx = args.start_idx
end_idx = args.end_idx

# double check that the folders are generated
cf.check_folders()

# the handcoded slidelist below is becuase C16 had some tumor slides that could only contribute tumor slides
##################
massiveslidelist = [('tumor_061.tif', 1, 0), ('tumor_058.tif', 1, 0), ('tumor_072.tif', 1, 0), ('tumor_035.tif', 1, 0),
              ('tumor_078.tif', 1, 0), ('tumor_092.tif', 1, 1), ('tumor_033.tif', 1, 1), ('tumor_008.tif', 1, 0),
              ('tumor_026.tif', 1, 0), ('tumor_053.tif', 1, 0), ('tumor_098.tif', 1, 0), ('tumor_050.tif', 1, 0),
              ('tumor_093.tif', 1, 0), ('tumor_103.tif', 1, 0), ('tumor_068.tif', 1, 0), ('tumor_039.tif', 1, 0),
              ('tumor_007.tif', 1, 0), ('tumor_104.tif', 1, 0), ('tumor_089.tif', 1, 0), ('tumor_065.tif', 1, 0),
              ('tumor_057.tif', 1, 0), ('tumor_082.tif', 1, 0), ('tumor_012.tif', 1, 0), ('tumor_029.tif', 1, 1),
              ('tumor_030.tif', 1, 0), ('tumor_062.tif', 1, 0), ('tumor_003.tif', 1, 0), ('tumor_034.tif', 1, 1),
              ('tumor_066.tif', 1, 0), ('tumor_031.tif', 1, 0), ('tumor_076.tif', 1, 0), ('tumor_099.tif', 1, 0),
              ('tumor_101.tif', 1, 0), ('tumor_027.tif', 1, 0), ('tumor_091.tif', 1, 0), ('tumor_055.tif', 1, 1),
              ('tumor_004.tif', 1, 0), ('tumor_107.tif', 1, 0), ('tumor_015.tif', 1, 1), ('tumor_097.tif', 1, 0),
              ('tumor_056.tif', 1, 1), ('tumor_079.tif', 1, 1), ('tumor_002.tif', 1, 0), ('tumor_087.tif', 1, 0),
              ('tumor_102.tif', 1, 0), ('tumor_036.tif', 1, 0), ('tumor_014.tif', 1, 0), ('tumor_071.tif', 1, 0),
              ('tumor_013.tif', 1, 0), ('tumor_049.tif', 1, 0), ('tumor_085.tif', 1, 1), ('tumor_096.tif', 1, 0),
              ('tumor_037.tif', 1, 0), ('tumor_045.tif', 1, 0), ('tumor_041.tif', 1, 0), ('tumor_074.tif', 1, 0),
              ('tumor_023.tif', 1, 0), ('tumor_019.tif', 1, 0), ('tumor_064.tif', 1, 0), ('tumor_105.tif', 1, 0),
              ('tumor_059.tif', 1, 0), ('tumor_109.tif', 1, 0), ('tumor_080.tif', 1, 0), ('tumor_084.tif', 1, 0),
              ('tumor_106.tif', 1, 0), ('tumor_043.tif', 1, 0), ('tumor_047.tif', 1, 0), ('tumor_042.tif', 1, 0),
              ('tumor_021.tif', 1, 0), ('tumor_094.tif', 1, 0), ('tumor_005.tif', 1, 0), ('tumor_038.tif', 1, 0),
              ('tumor_022.tif', 1, 0), ('tumor_048.tif', 1, 0), ('tumor_063.tif', 1, 0), ('tumor_060.tif', 1, 0),
              ('tumor_028.tif', 1, 0), ('tumor_040.tif', 1, 0), ('tumor_088.tif', 1, 0), ('tumor_006.tif', 1, 0),
              ('tumor_081.tif', 1, 0), ('tumor_067.tif', 1, 1), ('tumor_110.tif', 1, 1), ('tumor_054.tif', 1, 1),
              ('tumor_075.tif', 1, 0), ('tumor_009.tif', 1, 0), ('tumor_052.tif', 1, 0), ('tumor_077.tif', 1, 0),
              ('tumor_044.tif', 1, 1), ('tumor_083.tif', 1, 0), ('tumor_025.tif', 1, 0), ('tumor_032.tif', 1, 0),
              ('tumor_024.tif', 1, 0), ('tumor_011.tif', 1, 0), ('tumor_017.tif', 1, 0), ('tumor_069.tif', 1, 0),
              ('tumor_001.tif', 1, 0), ('tumor_010.tif', 1, 1), ('tumor_111.tif', 1, 0), ('tumor_051.tif', 1, 1),
              ('tumor_100.tif', 1, 0), ('tumor_108.tif', 1, 0), ('tumor_070.tif', 1, 0), ('tumor_095.tif', 1, 1),
              ('tumor_016.tif', 1, 0), ('tumor_090.tif', 1, 0), ('tumor_020.tif', 1, 1), ('tumor_073.tif', 1, 0),
              ('tumor_086.tif', 1, 0), ('tumor_046.tif', 1, 1), ('tumor_018.tif', 1, 1), ('normal_053.tif', 0, 0),
              ('normal_115.tif', 0, 0), ('normal_001.tif', 0, 0), ('normal_121.tif', 0, 0), ('normal_058.tif', 0, 0),
              ('normal_072.tif', 0, 0), ('normal_024.tif', 0, 0), ('normal_118.tif', 0, 0), ('normal_057.tif', 0, 0),
              ('normal_088.tif', 0, 0), ('normal_064.tif', 0, 0), ('normal_127.tif', 0, 0), ('normal_160.tif', 0, 0),
              ('normal_071.tif', 0, 0), ('normal_067.tif', 0, 0), ('normal_059.tif', 0, 0), ('normal_096.tif', 0, 0),
              ('normal_026.tif', 0, 0), ('normal_044.tif', 0, 0), ('normal_102.tif', 0, 0), ('normal_002.tif', 0, 0),
              ('normal_101.tif', 0, 0), ('normal_010.tif', 0, 0), ('normal_080.tif', 0, 0), ('normal_117.tif', 0, 0),
              ('normal_027.tif', 0, 0), ('normal_133.tif', 0, 0), ('normal_108.tif', 0, 0), ('normal_085.tif', 0, 0),
              ('normal_045.tif', 0, 0), ('normal_082.tif', 0, 0), ('normal_140.tif', 0, 0), ('normal_065.tif', 0, 0),
              ('normal_135.tif', 0, 0), ('normal_063.tif', 0, 0), ('normal_069.tif', 0, 0), ('normal_007.tif', 0, 0),
              ('normal_155.tif', 0, 0), ('normal_150.tif', 0, 0), ('normal_128.tif', 0, 0), ('normal_120.tif', 0, 0),
              ('normal_030.tif', 0, 0), ('normal_111.tif', 0, 0), ('normal_131.tif', 0, 0), ('normal_087.tif', 0, 0),
              ('normal_105.tif', 0, 0), ('normal_146.tif', 0, 0), ('normal_097.tif', 0, 0), ('normal_060.tif', 0, 0),
              ('normal_148.tif', 0, 0), ('normal_051.tif', 0, 0), ('normal_094.tif', 0, 0), ('normal_008.tif', 0, 0),
              ('normal_076.tif', 0, 0), ('normal_158.tif', 0, 0), ('normal_029.tif', 0, 0), ('normal_126.tif', 0, 0),
              ('normal_025.tif', 0, 0), ('normal_048.tif', 0, 0), ('normal_106.tif', 0, 0), ('normal_056.tif', 0, 0),
              ('normal_035.tif', 0, 0), ('normal_046.tif', 0, 0), ('normal_022.tif', 0, 0), ('normal_139.tif', 0, 0),
              ('normal_020.tif', 0, 0), ('normal_011.tif', 0, 0), ('normal_003.tif', 0, 0), ('normal_084.tif', 0, 0),
              ('normal_033.tif', 0, 0), ('normal_062.tif', 0, 0), ('normal_132.tif', 0, 0), ('normal_145.tif', 0, 0),
              ('normal_040.tif', 0, 0), ('normal_043.tif', 0, 0), ('normal_083.tif', 0, 0), ('normal_081.tif', 0, 0),
              ('normal_037.tif', 0, 0), ('normal_122.tif', 0, 0), ('normal_018.tif', 0, 0), ('normal_119.tif', 0, 0),
              ('normal_137.tif', 0, 0), ('normal_114.tif', 0, 0), ('normal_077.tif', 0, 0), ('normal_103.tif', 0, 0),
              ('normal_075.tif', 0, 0), ('normal_156.tif', 0, 0), ('normal_049.tif', 0, 0), ('normal_095.tif', 0, 0),
              ('normal_089.tif', 0, 0), ('normal_032.tif', 0, 0), ('normal_042.tif', 0, 0), ('normal_141.tif', 0, 0),
              ('normal_021.tif', 0, 0), ('normal_154.tif', 0, 0), ('normal_112.tif', 0, 0), ('normal_151.tif', 0, 0),
              ('normal_091.tif', 0, 0), ('normal_104.tif', 0, 0), ('normal_129.tif', 0, 0), ('normal_079.tif', 0, 0),
              ('normal_143.tif', 0, 0), ('normal_100.tif', 0, 0), ('normal_147.tif', 0, 0), ('normal_134.tif', 0, 0),
              ('normal_109.tif', 0, 0), ('normal_006.tif', 0, 0), ('normal_041.tif', 0, 0), ('normal_013.tif', 0, 0),
              ('normal_004.tif', 0, 0), ('normal_074.tif', 0, 0), ('normal_124.tif', 0, 0), ('normal_005.tif', 0, 0),
              ('normal_061.tif', 0, 0), ('normal_130.tif', 0, 0), ('normal_153.tif', 0, 0), ('normal_047.tif', 0, 0),
              ('normal_157.tif', 0, 0), ('normal_107.tif', 0, 0), ('normal_066.tif', 0, 0), ('normal_092.tif', 0, 0),
              ('normal_098.tif', 0, 0), ('normal_116.tif', 0, 0), ('normal_023.tif', 0, 0), ('normal_036.tif', 0, 0),
              ('normal_031.tif', 0, 0), ('normal_093.tif', 0, 0), ('normal_019.tif', 0, 0), ('normal_152.tif', 0, 0),
              ('normal_054.tif', 0, 0), ('normal_142.tif', 0, 0), ('normal_052.tif', 0, 0), ('normal_070.tif', 0, 0),
              ('normal_039.tif', 0, 0), ('normal_113.tif', 0, 0), ('normal_055.tif', 0, 0), ('normal_050.tif', 0, 0),
              ('normal_123.tif', 0, 0), ('normal_015.tif', 0, 0), ('normal_138.tif', 0, 0), ('normal_110.tif', 0, 0),
              ('normal_038.tif', 0, 0), ('normal_090.tif', 0, 0), ('normal_159.tif', 0, 0), ('normal_014.tif', 0, 0),
              ('normal_016.tif', 0, 0), ('normal_017.tif', 0, 0), ('normal_099.tif', 0, 0), ('normal_073.tif', 0, 0),
              ('normal_125.tif', 0, 0), ('normal_078.tif', 0, 0), ('normal_012.tif', 0, 0), ('normal_136.tif', 0, 0),
              ('normal_068.tif', 0, 0), ('normal_009.tif', 0, 0), ('normal_149.tif', 0, 0), ('normal_034.tif', 0, 0),
              ('normal_028.tif', 0, 0), ('normal_144.tif', 0, 0)]

#############

slidelist=massiveslidelist[start_idx:end_idx]
# grab tumor slides
#tumor_slides = []
#normal_slides = []
#for slide in slidelist:
    #if "tumor" in slide[0]:
     #   tumor_slides.append(slide)
    #else:
     #   normal_slides.append(slide)

count=0
for slide_name, binary_tumor, only_tumor_patches in tqdm(slidelist):
    print(slide_name)
#     here to clarify we alreadly have the tissue,tumor, and nontumor slide level masks for the level 2 resolution
#     we now want all of these for level 1
    # here we can use a lot of the code from MapMaskGen which does this for us
    print(cf.map_path)
    MapMaskGen.main(slidename=slide_name, binary_tumor=binary_tumor, downsamplefactor=hp.resolution_levels[3], mask_level=3)
    count += 1
    print("number of slides completed: ", count)
    print("the code has finished running")
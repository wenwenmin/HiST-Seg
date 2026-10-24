"""
This script sets up all the variables and file lists with all the normal and tumor slides. This script then
calls DataGenFlow.py for each slide in the slidelist.
"""
from PatchExtract.parameters import cf, hp
from PatchExtract.DataGenFlow import main as data_main
from PatchExtract.PatchUtils.utils import get_slidelist
import argparse
import random
from multiprocessing import Pool
from functools import partial

random.seed(a=48)

parser = argparse.ArgumentParser()
parser.add_argument("--make_slide_mask", type=int,
                    help="1 for do the operation 0 to skip", required=True)
parser.add_argument("--make_patch", type=int,
                    help="1 for do the operation 0 to skip", required=True)
parser.add_argument("--make_patch_mask", type=int,
                    help="1 for do the operation 0 to skip", required=True)
parser.add_argument("--num_threads", type=int,
                    help="number of threads", required=True)
parser.add_argument("--level", type=int, help="what level are we using for the patches, maps, masks", required=True)


args = parser.parse_args()
FLAG_MASKS = args.make_slide_mask
FLAG_PATCH = args.make_patch
FLAG_PATCH_MASK = args.make_patch_mask
which_level = args.level
# double check that the folders are generated
cf.check_folders()
if which_level==3:
    cf.update_cf(which_level)
    hp.update_level(which_level)
slidelist = get_slidelist()

TUMOR = 1
NONTUMOR = 0
TUMOR_PATCHES_ONLY = 1
ALL_PATCHES = 0

with Pool(args.num_threads) as p:
    p.map(partial(data_main, FLAG_MASKS=FLAG_MASKS, FLAG_PATCH=FLAG_PATCH,FLAG_PATCH_MASK=FLAG_PATCH_MASK), slidelist)

print("the code has finished running")

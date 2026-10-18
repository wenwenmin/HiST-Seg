import os
from sklearn.metrics import auc
import numpy as np

def make_AUC(folders):
    aucs = []
    for folder in folders:
        fpr = np.loadtxt(os.path.join(folder, "test_fprs.txt"))
        tpr = np.loadtxt(os.path.join(folder, "test_tprs.txt"))
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)

    std = np.std(aucs)
    mean = np.mean(aucs)
    return mean, std


def paper_results():
    ## Swins
    base_path_swin = "/path/results/paper/swin"
    swin_tiny_path_1 = os.path.join(base_path_swin, "swin_lv2_tiny_r00")
    swin_tiny_path_2 = os.path.join(base_path_swin, "swin_lv2_tiny_best_r00_v2_dgx")
    swin_tiny_path_3 = os.path.join(base_path_swin, "swin_lv2_tiny_best_r00_v3_dgx")

    swin_small_path_1 = os.path.join(base_path_swin, "swin_lv2_small_r00_gpu")
    swin_small_path_2 = os.path.join(base_path_swin, "swin_lv2_small_best_r00_v2_gpu")
    swin_small_path_3 = os.path.join(base_path_swin, "swin_lv2_small_best_r00_v3_dgx_full")

    dual_tiny_1 = os.path.join(base_path_swin, "dual_swin_tiny_r00_dgx_v2_again_b64")
    dual_tiny_2 = os.path.join(base_path_swin, "dual_swin_tiny_r00_gpu_v2")
    dual_tiny_3 = os.path.join(base_path_swin, "dual_swin_tiny_r00_gpu_v3")

    dual_small_1 = os.path.join(base_path_swin, "dual_swin_small_r00_dgx_b32_v1_again")
    dual_small_2 = os.path.join(base_path_swin, "dual_swin_small_r00_dgx_b32_v2_again")
    dual_small_3 = os.path.join(base_path_swin, "dual_swin_small_r00_dgx_b32_v3_again")

    dual_smalls = [dual_small_1, dual_small_2, dual_small_3]
    dual_tinys = [dual_tiny_1, dual_tiny_2, dual_tiny_3]
    swin_tinys = [swin_tiny_path_1, swin_tiny_path_2, swin_tiny_path_3]
    swin_smalls = [swin_small_path_1, swin_small_path_2, swin_small_path_3]

    print(" the results for dual_smalls ", make_AUC(dual_smalls))
    print(" the results for dual_tinys ", make_AUC(dual_tinys))
    print(" the results for swin_tinys ", make_AUC(swin_tinys))
    print(" the results for swin_smalls ", make_AUC(swin_smalls))
    # Mit
    b1_base_path = "/path/results/paper/b1"
    mit_b1_1 = os.path.join(b1_base_path, "mit_b1_lv2_best_v1")
    mit_b1_2 = os.path.join(b1_base_path, "mit_b1_lv2_best_v2")
    mit_b1_3 = os.path.join(b1_base_path, "mit_b1_lv2_best_v3_gpu")

    dual_b1_1 = os.path.join(b1_base_path, "dual_b1_best_v1")
    dual_b1_2 = os.path.join(b1_base_path, "dual_b1_best_v2_dgx")
    dual_b1_3 = os.path.join(b1_base_path, "dual_b1_best_v3_dgx_fullrun")

    b2_base_path = "/path/results/paper/b2"

    mit_b2_1 = os.path.join(b2_base_path, "mit_b2_lv2_best_v1")
    mit_b2_2 = os.path.join(b2_base_path, "mit_b2_lv2_best_v2_gpu")
    mit_b2_3 = os.path.join(b2_base_path, "mit_b2_lv2_best_v3_gpu")

    dual_b2_1 = os.path.join(b2_base_path, "dual_b2_r01_fullrun")
    dual_b2_2 = os.path.join(b2_base_path, "dual_b2_best_v2_r01_dgx")
    dual_b2_3 = os.path.join(b2_base_path, "dual_b2_best_v3_r01_dgx")

    mit_b1s = [mit_b1_1, mit_b1_2, mit_b1_3]
    dual_b1s = [dual_b1_1, dual_b1_2, dual_b1_3]
    mit_b2s = [mit_b2_1, mit_b2_2, mit_b2_3]
    dual_b2s = [dual_b2_1, dual_b2_2, dual_b2_3]

    print(" the results for dual_b1s ", make_AUC(dual_b1s))
    print(" the results for dual_b2s ", make_AUC(dual_b2s))
    print(" the results for mit_b1s ", make_AUC(mit_b1s))
    print(" the results for mit_b2s ", make_AUC(mit_b2s))

if __name__ == "__main__":
    paper_results()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

path="/your/path/to/results"

dual_best_mit_b1_fpr = np.loadtxt(f"{path}/b1/dual_b1_best_v1/test_fprs.txt")
dual_best_mit_b1_tpr = np.loadtxt(f"{path}/b1/dual_b1_best_v1/test_tprs.txt")

single_best_mit_b1_fpr = np.loadtxt(f"{path}/b1/mit_b1_lv2_best_v1/test_fprs.txt")
single_best_mit_b1_tpr = np.loadtxt(f"{path}/b1/mit_b1_lv2_best_v1/test_tprs.txt")


dual_best_mit_b2_fpr = np.loadtxt(f"{path}/b2/dual_b2_best_v2_r01_dgx/test_fprs.txt")
dual_best_mit_b2_tpr = np.loadtxt(f"{path}/b2/dual_b2_best_v2_r01_dgx/test_tprs.txt")

single_best_mit_b2_fpr = np.loadtxt(f"{path}/b2/mit_b2_lv2_best_v1/test_fprs.txt")
single_best_mit_b2_tpr = np.loadtxt(f"{path}/b2/mit_b2_lv2_best_v1/test_tprs.txt")

dual_best_swin_small_fpr = np.loadtxt(f"{path}/swin/dual_swin_small_r00_dgx_b32_v1_again/test_fprs.txt")
dual_best_swin_small_tpr = np.loadtxt(f"{path}/swin/dual_swin_small_r00_dgx_b32_v1_again/test_tprs.txt")

single_best_swin_small_fpr = np.loadtxt(f"{path}/swin/swin_lv2_small_r0a_gpu/test_fprs.txt")
single_best_swin_small_tpr = np.loadtxt(f"{path}/swin/swin_lv2_small_r0a_gpu/test_tprs.txt")


dual_best_swin_tiny_fpr = np.loadtxt(f"{path}/swin/dual_swin_tiny_r00_gpu_v3/test_fprs.txt")
dual_best_swin_tiny_tpr = np.loadtxt(f"{path}/swin/dual_swin_tiny_r00_gpu_v3/test_tprs.txt")

single_best_swin_tiny_fpr = np.loadtxt(f"{path}/swin/swin_lv2_tiny_best_r00_v3_dgx/test_fprs.txt")
single_best_swin_tiny_tpr = np.loadtxt(f"{path}/swin/swin_lv2_tiny_best_r00_v3_dgx/test_tprs.txt")


auc_single_mit_b1 = auc(single_best_mit_b1_fpr,single_best_mit_b1_tpr)
auc_single_mit_b2 = auc(single_best_mit_b2_fpr,single_best_mit_b2_tpr)
auc_dual_mit_b2 = auc(dual_best_mit_b2_fpr,dual_best_mit_b2_tpr)
auc_dual_mit_b1 = auc(dual_best_mit_b1_fpr,dual_best_mit_b1_tpr)


auc_single_swin_tiny = auc(single_best_swin_tiny_fpr,single_best_swin_tiny_tpr)
auc_single_swin_small = auc(single_best_swin_small_fpr,single_best_swin_small_tpr)
auc_dual_swin_tiny = auc(dual_best_swin_tiny_fpr,dual_best_swin_tiny_tpr)
auc_dual_swin_small = auc(dual_best_swin_small_fpr,dual_best_swin_small_tpr)


plt.figure(figsize=(10, 10))
plt.plot(single_best_mit_b1_fpr, single_best_mit_b1_tpr, label='Single MiT-B1 (AUC = {:.4f})'.format(auc_single_mit_b1))
plt.plot(single_best_mit_b2_fpr, single_best_mit_b2_tpr, label='Single MiT-B2 (AUC = {:.4f})'.format(auc_single_mit_b2))

plt.plot(dual_best_mit_b2_fpr, dual_best_mit_b2_tpr, label='CGS-Net MiT-B2 (AUC = {:.4f})'.format(auc_dual_mit_b2))
plt.plot(dual_best_mit_b1_fpr, dual_best_mit_b1_tpr, label='CGS-Net MiT-B1 (AUC = {:.4f})'.format(auc_dual_mit_b1))


plt.plot(single_best_swin_tiny_fpr, single_best_swin_tiny_tpr, label='Single Swin V2 Tiny (AUC = {:.4f})'.format(auc_single_swin_tiny))
plt.plot(single_best_swin_small_fpr, single_best_swin_small_tpr, label='Single Swin V2 Small (AUC = {:.4f})'.format(auc_single_swin_small))

plt.plot(dual_best_swin_tiny_fpr, dual_best_swin_tiny_tpr, label='CGS-Net Swin V2 Tiny (AUC = {:.4f})'.format(auc_dual_swin_tiny))
plt.plot(dual_best_swin_small_fpr, dual_best_swin_small_tpr, label='CGS-Net Swin V2 Small (AUC = {:.4f})'.format(auc_dual_swin_small))


plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlabel('False Positive Rate', fontsize=24)
plt.ylabel('True Positive Rate', fontsize=24)
plt.title('ROC Curve', fontsize=24)
# plt.legend(loc="lower right")
plt.tick_params(axis='x', labelsize=22)
plt.tick_params(axis='y', labelsize=22)
legend = plt.legend(loc='lower right', prop={'size': 18})
legend.get_title().set_fontsize('24')
plt.tight_layout()
plt.savefig('./roc.png', bbox_inches='tight')
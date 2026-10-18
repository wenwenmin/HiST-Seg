import dgl
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from sklearn import metrics
from PIL import Image
import os
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)


def evaluate(y_true, y_score, pre=False):
    """calculate evaluation metrics"""
    y_true = pd.Series(y_true)
    y_score = pd.Series(y_score)

    roc_auc = metrics.roc_auc_score(y_true, y_score)
    ap = metrics.average_precision_score(y_true, y_score)

    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thres = np.percentile(y_score, ratio)
    y_pred = (y_score >= 0.5).astype(int)
    y_true = y_true.astype(int)
    _, _, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')
    if pre:
        return roc_auc, ap, f1, y_pred
    else:
        return roc_auc, ap, f1
def load_image(filename):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)  # AAAI需要
    os.environ['PYTHONHASHSEED'] = str(seed)



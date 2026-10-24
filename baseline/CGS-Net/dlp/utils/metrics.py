import torch.nn.functional as F
import torch
import os
import torch.nn as nn


class ModelMetrics:
    def __init__(self, results_dir, loss_fn_name: str):
        self.results_dir = results_dir
        self.loss_name = loss_fn_name
        # file paths
        self.train_loss = os.path.join(self.results_dir, self.loss_name + "_loss_train.txt")
        self.trainval_loss = os.path.join(self.results_dir, self.loss_name + "_loss_trainval.txt")
        self.val_loss = os.path.join(self.results_dir, self.loss_name + "_loss_val.txt")
        self.test_loss = os.path.join(self.results_dir, self.loss_name + "_loss_test.txt")
        self.lr_decoder = os.path.join(self.results_dir, "lr_decoder.txt")
        self.lr_backbone = os.path.join(self.results_dir, "lr_backbone.txt")
        self.lr_mca = os.path.join(self.results_dir, "lr_mca.txt")
        # cancer
        self.train_jacc_cancer = os.path.join(self.results_dir, "jacc_loss_train_cancer.txt")
        self.trainval_jacc_cancer = os.path.join(self.results_dir, "jacc_loss_trainval_cancer.txt")
        self.val_jacc_cancer = os.path.join(self.results_dir, "jacc_loss_val_cancer.txt")
        self.test_jacc_cancer = os.path.join(self.results_dir, "jacc_loss_test_cancer.txt")
        self.train_dice_cancer = os.path.join(self.results_dir, "dice_score_train_cancer.txt")
        self.trainval_dice_cancer = os.path.join(self.results_dir, "dice_score_trainval_cancer.txt")
        self.val_dice_cancer = os.path.join(self.results_dir, "dice_score_val_cancer.txt")
        self.test_dice_cancer = os.path.join(self.results_dir, "dice_score_test_cancer.txt")
        # noncancer
        self.train_jacc_noncancer = os.path.join(self.results_dir, "jacc_loss_train_noncancer.txt")
        self.trainval_jacc_noncancer = os.path.join(self.results_dir, "jacc_loss_trainval_noncancer.txt")
        self.val_jacc_noncancer = os.path.join(self.results_dir, "jacc_loss_val_noncancer.txt")
        self.test_jacc_noncancer = os.path.join(self.results_dir, "jacc_loss_test_noncancer.txt")
        self.train_dice_noncancer = os.path.join(self.results_dir, "dice_score_train_noncancer.txt")
        self.trainval_dice_noncancer = os.path.join(self.results_dir, "dice_score_trainval_noncancer.txt")
        self.val_dice_noncancer = os.path.join(self.results_dir, "dice_score_val_noncancer.txt")
        self.test_dice_noncancer = os.path.join(self.results_dir, "dice_score_test_noncancer.txt")
        ### fp, fn, tp, tn
        # tp
        self.train_tp = os.path.join(self.results_dir, "tp_train.txt")
        self.trainval_tp = os.path.join(self.results_dir, "tp_trainval.txt")
        self.val_tp = os.path.join(self.results_dir, "tp_val.txt")
        self.test_tp = os.path.join(self.results_dir, "tp_test.txt")
        # tn
        self.train_tn = os.path.join(self.results_dir, "tn_train.txt")
        self.trainval_tn = os.path.join(self.results_dir, "tn_trainval.txt")
        self.val_tn = os.path.join(self.results_dir, "tn_val.txt")
        self.test_tn = os.path.join(self.results_dir, "tn_test.txt")
        # fp
        self.train_fp = os.path.join(self.results_dir, "fp_train.txt")
        self.trainval_fp = os.path.join(self.results_dir, "fp_trainval.txt")
        self.val_fp = os.path.join(self.results_dir, "fp_val.txt")
        self.test_fp = os.path.join(self.results_dir, "fp_test.txt")
        # fn
        self.train_fn = os.path.join(self.results_dir, "fn_train.txt")
        self.trainval_fn = os.path.join(self.results_dir, "fn_trainval.txt")
        self.val_fn = os.path.join(self.results_dir, "fn_val.txt")
        self.test_fn = os.path.join(self.results_dir, "fn_test.txt")

    @staticmethod
    def write_list_to_file(file_path_and_name, list_to_write) -> None:
        textfile = open(file_path_and_name, "w")
        for element in list_to_write:
            textfile.write(str(element) + "\n")
        textfile.close()

    @staticmethod
    def write_to_file(file_name, value, epoch):
        """
        The epoch==0 block makes a new file, this is important if running the same code using the same directories.
        Otherwise the new values will append to the old values.
        """
        if epoch == 0:
            textfile = open(file_name, "w")
        else:
            textfile = open(file_name, "a")
        textfile.write(str(value) + "\n")
        textfile.close()


def pixel_class_accuracy(prediction: torch.Tensor, target: torch.Tensor):
    """
    the target (ground truth) is a binary mask where 0 means noncancer and 1 means cancer
    https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
    """
    neg_count = int((target == 0).sum())
    pos_count = int((target == 1).sum())
    trueneg_count = neg_correct = int(((prediction == 0) & (target == 0)).sum())
    truepos_count = pos_correct = int(((prediction == 1) & (target == 1)).sum())
    falsepos_count = neg_count - neg_correct
    falseneg_count = pos_count - pos_correct

    return trueneg_count, truepos_count, falsepos_count, falseneg_count


def jacc_loss(prediction, target):
    """
    Returning the jaccard loss from calculation of the dice score
    """
    dice_score = (2 * (prediction * target).sum()) / ((prediction + target).sum() + 1e-8)
    return 1.0 - (dice_score / (2. - dice_score))


def dice_coeff(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dice_score = (2 * (prediction * target).sum()) / ((prediction + target).sum() + 1e-8)
    return dice_score


def dice_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert prediction.shape == target.shape
    dice_score = (2 * (prediction * target).sum()) / ((prediction + target).sum() + 1e-8)
    return 1 - dice_score


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Note you pass the raw inputs into this function. Do not pass anything through a sigmoid as inputs
    This loss function idea was taken from https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
    """
    gamma = 2  # from the focal loss paper
    alpha = 0.25  # also from the paper https://arxiv.org/pdf/1708.02002.pdf
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    # other options include sum
    return loss.mean()


def combined_focal_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    sig_input = torch.sigmoid(inputs)  # the inputs come directly from the model and have not been
    # passed through a sigmoid layer
    preds = (sig_input > 0.5).float()
    # note the focal loss does not get preds as inputs because we are using bce with logit loss which includes
    # a sigmoid layer. The dice loss function does not include a sigmoid layer so we pass the preds to the dice loss
    return 0.95 * focal_loss(inputs=inputs, targets=targets) + 0.05 * dice_loss(prediction=preds, target=targets)


def make_loss_fn(name: str):
    if name == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
        loss_fn.name = "bce"
    elif name == "focal":
        loss_fn = combined_focal_dice_loss
        loss_fn.name = "focal"
    else:
        raise Exception("wrong value to loss function passed")

    return loss_fn


if __name__ == "__main__":
    print("metrics.py was executed")

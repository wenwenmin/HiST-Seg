import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from numpy import loadtxt
from sklearn.metrics import auc


def make_graph_single(metric_name, file_save_path, data_list):
    plt.title(metric_name + " for Training,Validation, and Test")
    label = metric_name
    plt.plot(data_list, label=label)
    plt.xlabel('Epochs')
    plt.legend()
    file_path = file_save_path
    plt.savefig(file_path)
    plt.close()


def make_precision_recall_curves(file_save_path, tp_list, fp_list, fn_list, epoch_skip, min_focal_val_epoch,
                                 extra_title=""):
    """
    https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
    """
    plt.title("Precision and Recall curves " + extra_title)
    assert len(tp_list) == len(fp_list) == len(fn_list)

    precision_list = np.nan_to_num(tp_list / (tp_list + fp_list))
    recall_list = np.nan_to_num(tp_list / (tp_list + fn_list))

    precision_label = "precision score:" + str(round(precision_list[min_focal_val_epoch], 3)) + " at " + str(
        min_focal_val_epoch)
    recall_label = "recall score:" + str(round(recall_list[min_focal_val_epoch], 3)) + " at " + str(min_focal_val_epoch)
    plt.plot(precision_list, label=precision_label)
    plt.plot(recall_list, label=recall_label)
    max_pixelnum = np.max(np.maximum.reduce([precision_list, recall_list]))
    min_pixelnum = max_pixelnum * -0.1
    plt.ylim((min_pixelnum, max_pixelnum * 1.1))
    if epoch_skip < len(tp_list):
        # the max and min functions will return a list when passed a list and hence need to have [0]
        max_y = np.max(np.maximum.reduce([precision_list[epoch_skip:], recall_list[epoch_skip:]]))
        min_y = max_y * -0.1
        plt.ylim((min_y, max_y * 1.1))
    plt.xlabel('Epochs')
    plt.ylabel('Precision / recall')
    plt.legend()
    file_path = os.path.join(file_save_path)
    plt.savefig(file_path)
    plt.close()


def make_pixelclass_graph(file_save_path, tp_list, fp_list, fn_list, tn_list, epoch_skip, extra_title=""):
    """
    idea for this graph came form colab for star dist
    """
    assert len(tp_list) == len(fp_list) == len(fn_list) == len(tn_list)
    plt.title("pixel class graph for " + extra_title)
    plt.plot(tp_list, label='tp')
    plt.plot(fp_list, label='fp')
    plt.plot(fn_list, label='fn')
    plt.plot(tn_list, label='tn')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel number')
    max_pixelnum = np.max(np.maximum.reduce([tp_list, fp_list, fn_list, tn_list]))
    min_pixelnum = max_pixelnum * -0.1
    plt.ylim((min_pixelnum, max_pixelnum * 1.1))
    if epoch_skip < len(tp_list):
        # the max and min functions will return a list when passed a list and hence need to have [0]
        max_y = np.max(
            np.maximum.reduce([tp_list[epoch_skip:], fp_list[epoch_skip:], fn_list[epoch_skip:], tn_list[epoch_skip:]]))
        min_y = max_y * -0.1
        plt.ylim((min_y, max_y * 1.1))
    plt.legend()
    file_path = os.path.join(file_save_path)
    plt.savefig(file_path)
    plt.close()


def make_graph_train_and_val(metric_name, file_save_path, data_list_val, data_list_train, epoch_skip,
                             epoch_min_val_loss):
    epoch_skip = int(epoch_skip)
    plt.title(metric_name + " for Train and Val")
    val_num = data_list_val[epoch_min_val_loss]
    train_num = data_list_train[epoch_min_val_loss]
    val_label = "Val " + metric_name + " " + str(round(val_num, 3)) + " at " + str(epoch_min_val_loss)
    train_label = "Train " + metric_name + " " + str(round(train_num, 3)) + " at " + str(epoch_min_val_loss)
    assert len(data_list_val) == len(data_list_train)
    length = len(data_list_train) - 1

    plt.plot(data_list_val, label=val_label)
    plt.plot(data_list_train, label=train_label)
    # gets the max for the following lists for plotting ylimits when epoch skip is set
    # this code will error out if the epoch skip is larger then the list length
    if epoch_skip < len(data_list_train):
        # the max and min functions will return a list when passed a list and hence need to have [0]
        max_y = max(max(data_list_train[epoch_skip:]), max(data_list_val[epoch_skip:]))
        min_y = min(min(data_list_train[epoch_skip:]), min(data_list_val[epoch_skip:]))
        plt.ylim((min_y, max_y))
    plt.xlim((epoch_skip, length))
    plt.xlabel('Epochs')
    plt.legend()
    file_path = file_save_path
    plt.savefig(file_path)
    plt.close()


def make_graph_trainval(metric_name, file_save_path, data_list_train, epoch_skip, epoch_min_val_loss):
    epoch_skip = int(epoch_skip)
    plt.title(metric_name + " for Trainval (combined datasets)")
    train_num = data_list_train[epoch_min_val_loss]
    train_label = "TrainVal " + metric_name + " " + str(round(train_num, 3)) + " at " + str(
        epoch_min_val_loss)
    length = len(data_list_train) - 1

    plt.plot(data_list_train, label=train_label)
    # gets the max for the following lists for plotting ylimits when epoch skip is set
    # this code will error out if the epoch skip is larger then the list length
    if epoch_skip < len(data_list_train):
        # the max and min functions will return a list when passed a list and hence need to have [0]
        max_y = max(data_list_train[epoch_skip:])
        min_y = min(data_list_train[epoch_skip:])
        plt.ylim((min_y, max_y))
    plt.xlim((epoch_skip, length))
    plt.xlabel('Epochs')
    plt.legend()
    file_path = file_save_path
    plt.savefig(file_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data", type=str,
                        help="the full path to the csv file with the dataset", required=True)
    parser.add_argument("--model_type", type=str,
                        help="seg for segmentation model or class for classification model", required=True)
    parser.add_argument("--loss_name", type=str, help="this is commonly bce_loss but could be anything check train.py",
                        required=True)
    parser.add_argument('--testing', type=int,
                        help='testing means that we are only plotting a single curve for the combined train and '
                             'validation dataset'
                             'when this is false then we have two curves one for train and one for val', required=True)
    args = parser.parse_args()

    RESULTS_DIR = args.source_data
    testing = args.testing
    if not os.path.isdir(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    file_path_data = args.source_data
    num_skip = 75  # this how many epochs to skip for the graphing with skipped epochs

    # need to find the
    # epoch where the minimum val focal loss occurs and then we can use that epoch to find the
    # values of the other statistics at that epoch.
    if not testing:
        # loading the data for the loss function (that used to update the wights)
        file_one = os.path.join(file_path_data, args.loss_name + "_loss_train.txt")
        file_two = os.path.join(file_path_data, args.loss_name + "_loss_val.txt")
        file_one_list_train = loadtxt(file_one, comments="#", delimiter=",", unpack=False)
        file_two_list_val = loadtxt(file_two, comments="#", delimiter=",", unpack=False)

        # finding the epoch where the minimum val focal loss occurs
        val_index_min_focal = np.where(file_two_list_val == min(file_two_list_val))[0][0]
        print("epoch where val min loss occurs", val_index_min_focal)
        final_pdf = os.path.join(RESULTS_DIR, args.loss_name + "_image.pdf")
        final_pdf_adj = os.path.join(RESULTS_DIR, args.loss_name + "_image_adj.pdf")
        make_graph_train_and_val(args.loss_name, final_pdf, file_two_list_val, file_one_list_train,
                                 epoch_skip=0, epoch_min_val_loss=val_index_min_focal)
        make_graph_train_and_val(args.loss_name, final_pdf_adj, file_two_list_val, file_one_list_train,
                                 epoch_skip=num_skip, epoch_min_val_loss=val_index_min_focal)

        # cancer files
        file_one_cancer = os.path.join(file_path_data, "jacc_loss_train_cancer.txt")
        file_two_cancer = os.path.join(file_path_data, "jacc_loss_val_cancer.txt")
        file_one_list_train_cancer = loadtxt(file_one_cancer, comments="#", delimiter=",", unpack=False)
        file_two_list_val_cancer = loadtxt(file_two_cancer, comments="#", delimiter=",", unpack=False)
        # noncancer files
        file_one_noncancer = os.path.join(file_path_data, "jacc_loss_train_noncancer.txt")
        file_two_noncancer = os.path.join(file_path_data, "jacc_loss_val_noncancer.txt")
        file_one_list_train_noncancer = loadtxt(file_one_noncancer, comments="#", delimiter=",", unpack=False)
        file_two_list_val_noncancer = loadtxt(file_two_noncancer, comments="#", delimiter=",", unpack=False)

        # cancer graphs
        final_pdf_cancer = os.path.join(RESULTS_DIR, "jacc_loss_image_cancer.pdf")
        final_pdf_adj_cancer = os.path.join(RESULTS_DIR, "jacc_loss_image_adj_cancer.pdf")
        make_graph_train_and_val("jacc loss cancer", final_pdf_cancer, data_list_val=file_two_list_val_cancer,
                                 data_list_train=file_one_list_train_cancer,
                                 epoch_skip=0, epoch_min_val_loss=val_index_min_focal)
        make_graph_train_and_val("jacc loss cancer", final_pdf_adj_cancer,
                                 data_list_val=file_two_list_val_cancer, data_list_train=file_one_list_train_cancer,
                                 epoch_skip=num_skip, epoch_min_val_loss=val_index_min_focal)
        # noncancer graphs
        final_pdf_noncancer = os.path.join(RESULTS_DIR, "jacc_loss_image_noncancer.pdf")
        final_pdf_adj_noncancer = os.path.join(RESULTS_DIR, "jacc_loss_image_adj_noncancer.pdf")
        make_graph_train_and_val("jacc loss noncancer", final_pdf_noncancer,
                                 data_list_val=file_two_list_val_noncancer,
                                 data_list_train=file_one_list_train_noncancer,
                                 epoch_skip=0, epoch_min_val_loss=val_index_min_focal)
        make_graph_train_and_val("jacc loss noncancer", final_pdf_adj_noncancer,
                                 data_list_val=file_two_list_val_noncancer,
                                 data_list_train=file_one_list_train_noncancer,
                                 epoch_skip=num_skip, epoch_min_val_loss=val_index_min_focal)
        # intersection over union calculation for cancer class
        iou_train_cancer = [1 - i for i in file_one_list_train_cancer]  # i is the jaccard loss
        iou_val_cancer = [1 - i for i in file_two_list_val_cancer]  # i is the jaccard loss
        final_pdf_cancer = os.path.join(RESULTS_DIR, "iou_image_cancer.pdf")
        final_pdf_adj_cancer = os.path.join(RESULTS_DIR, "iou_image_adj_cancer.pdf")
        make_graph_train_and_val("iou_cancer", final_pdf_cancer, iou_val_cancer, iou_train_cancer, epoch_skip=0,
                                 epoch_min_val_loss=val_index_min_focal)
        make_graph_train_and_val("iou_cancer", final_pdf_adj_cancer, iou_val_cancer, iou_train_cancer,
                                 epoch_skip=num_skip, epoch_min_val_loss=val_index_min_focal)
        # IOU for noncancer
        iou_train_noncancer = [1 - i for i in file_one_list_train_noncancer]  # i is the jaccard loss
        iou_val_noncancer = [1 - i for i in file_two_list_val_noncancer]  # i is the jaccard loss
        final_pdf_noncancer = os.path.join(RESULTS_DIR, "iou_image_noncancer.pdf")
        final_pdf_adj_noncancer = os.path.join(RESULTS_DIR, "iou_image_adj_noncancer.pdf")
        make_graph_train_and_val("iou_noncancer", final_pdf_noncancer, iou_val_noncancer, iou_train_noncancer,
                                 epoch_skip=0, epoch_min_val_loss=val_index_min_focal)
        make_graph_train_and_val("iou_noncancer", final_pdf_adj_noncancer, iou_val_noncancer,
                                 iou_train_noncancer, epoch_skip=num_skip,
                                 epoch_min_val_loss=val_index_min_focal)

        # mean iou graphs
        mean_iou_train = [(i + j) / 2 for i, j in zip(iou_train_cancer, iou_train_noncancer)]
        mean_iou_val = [(i + j) / 2 for i, j in zip(iou_val_cancer, iou_val_noncancer)]
        final_pdf_mean = os.path.join(RESULTS_DIR, "mean_iou_image.pdf")
        final_pdf_adj_mean = os.path.join(RESULTS_DIR, "mean_iou_image_adj.pdf")
        make_graph_train_and_val("mean_iou", final_pdf_mean, mean_iou_val, mean_iou_train, epoch_skip=0,
                                 epoch_min_val_loss=val_index_min_focal)
        make_graph_train_and_val("mean_iou", final_pdf_adj_mean, mean_iou_val, mean_iou_train,
                                 epoch_skip=num_skip, epoch_min_val_loss=val_index_min_focal)

        # dice score calculation for cancer class
        file_one_cancer = os.path.join(file_path_data, "dice_score_train_cancer.txt")
        file_two_cancer = os.path.join(file_path_data, "dice_score_val_cancer.txt")
        file_one_list_train_cancer = loadtxt(file_one_cancer, comments="#", delimiter=",", unpack=False)
        file_two_list_val_cancer = loadtxt(file_two_cancer, comments="#", delimiter=",", unpack=False)
        final_pdf_cancer = os.path.join(RESULTS_DIR, "f1_image_cancer.pdf")
        final_pdf_adj_cancer = os.path.join(RESULTS_DIR, "f1_image_adj_cancer.pdf")
        make_graph_train_and_val("dice score (f1) cancer", final_pdf_cancer, file_two_list_val_cancer,
                                 file_one_list_train_cancer,
                                 epoch_skip=0, epoch_min_val_loss=val_index_min_focal)
        make_graph_train_and_val("dice score (f1) cancer", final_pdf_adj_cancer, file_two_list_val_cancer,
                                 file_one_list_train_cancer,
                                 epoch_skip=num_skip, epoch_min_val_loss=val_index_min_focal)
        # dice score calculation for noncancer class
        file_one_noncancer = os.path.join(file_path_data, "dice_score_train_noncancer.txt")
        file_two_noncancer = os.path.join(file_path_data, "dice_score_val_noncancer.txt")
        file_one_list_train_noncancer = loadtxt(file_one_noncancer, comments="#", delimiter=",", unpack=False)
        file_two_list_val_noncancer = loadtxt(file_two_noncancer, comments="#", delimiter=",", unpack=False)
        final_pdf_noncancer = os.path.join(RESULTS_DIR, "f1_image_noncancer.pdf")
        final_pdf_adj_noncancer = os.path.join(RESULTS_DIR, "f1_image_adj_noncancer.pdf")
        make_graph_train_and_val("dice score (f1) noncancer", final_pdf_noncancer, file_two_list_val_noncancer,
                                 file_one_list_train_noncancer,
                                 epoch_skip=0, epoch_min_val_loss=val_index_min_focal)
        make_graph_train_and_val("dice score (f1) noncancer", final_pdf_adj_noncancer, file_two_list_val_noncancer,
                                 file_one_list_train_noncancer,
                                 epoch_skip=num_skip, epoch_min_val_loss=val_index_min_focal)
        # mean dice score calculation
        mean_dice_train = [(i + j) / 2 for i, j in zip(file_one_list_train_cancer, file_one_list_train_noncancer)]
        mean_dice_val = [(i + j) / 2 for i, j in zip(file_two_list_val_cancer, file_two_list_val_noncancer)]
        final_pdf_mean = os.path.join(RESULTS_DIR, "f1_image_mean.pdf")
        final_pdf_adj_mean = os.path.join(RESULTS_DIR, "f1_image_adj_mean.pdf")
        make_graph_train_and_val("dice score (f1) mean", final_pdf_mean, mean_dice_val, mean_dice_train,
                                 epoch_skip=0, epoch_min_val_loss=val_index_min_focal)
        make_graph_train_and_val("dice score (f1) mean", final_pdf_adj_mean, mean_dice_val, mean_dice_train,
                                 epoch_skip=num_skip, epoch_min_val_loss=val_index_min_focal)

        file_one = os.path.join(file_path_data, "lr_decoder.txt")
        file_one_list = loadtxt(file_one, comments="#", delimiter=",", unpack=False)
        final_pdf = os.path.join(RESULTS_DIR, "lr_image_decoder.pdf")
        make_graph_single("learning_rate for decoder", file_save_path=final_pdf, data_list=file_one_list)

        file_one = os.path.join(file_path_data, "lr_backbone.txt")
        file_one_list = loadtxt(file_one, comments="#", delimiter=",", unpack=False)
        final_pdf = os.path.join(RESULTS_DIR, "lr_image_backbone.pdf")
        make_graph_single("learning_rate for backbone", file_save_path=final_pdf, data_list=file_one_list)
        try:
            file_one = os.path.join(file_path_data, "lr_mca.txt")
            file_one_list = loadtxt(file_one, comments="#", delimiter=",", unpack=False)
            final_pdf = os.path.join(RESULTS_DIR, "lr_image_mca.pdf")
            make_graph_single("learning_rate for mca", file_save_path=final_pdf, data_list=file_one_list)
        except:
            print("the model executed was not a dual")

        ### make graphs tp, fp, fn, tn
        # train
        tp_train = os.path.join(file_path_data, "tp_train.txt")
        fp_train = os.path.join(file_path_data, "fp_train.txt")
        fn_train = os.path.join(file_path_data, "fn_train.txt")
        tn_train = os.path.join(file_path_data, "tn_train.txt")
        tp_train = loadtxt(tp_train, comments="#", delimiter=",", unpack=False)
        fp_train = loadtxt(fp_train, comments="#", delimiter=",", unpack=False)
        fn_train = loadtxt(fn_train, comments="#", delimiter=",", unpack=False)
        tn_train = loadtxt(tn_train, comments="#", delimiter=",", unpack=False)
        final_pdf = os.path.join(RESULTS_DIR, "tp_fp_fn_tn_train.pdf")
        final_pdf_adj = os.path.join(RESULTS_DIR, "tp_fp_fn_tn_train_adj.pdf")
        make_pixelclass_graph(file_save_path=final_pdf, tp_list=tp_train, fp_list=fp_train, fn_list=fn_train,
                              tn_list=tn_train, epoch_skip=0, extra_title="train")
        make_pixelclass_graph(file_save_path=final_pdf_adj, tp_list=tp_train, fp_list=fp_train, fn_list=fn_train,
                              tn_list=tn_train, epoch_skip=num_skip, extra_title="train")
        # precision and recall
        final_pdf = os.path.join(RESULTS_DIR, "precsion_recall_train.pdf")
        final_pdf_adj = os.path.join(RESULTS_DIR, "precsion_recall_train_adj.pdf")
        make_precision_recall_curves(file_save_path=final_pdf, tp_list=tp_train, fp_list=fp_train, fn_list=fn_train,
                                     epoch_skip=0, min_focal_val_epoch=val_index_min_focal, extra_title="train")
        make_precision_recall_curves(file_save_path=final_pdf_adj, tp_list=tp_train, fp_list=fp_train, fn_list=fn_train,
                                     epoch_skip=num_skip, min_focal_val_epoch=val_index_min_focal, extra_title="train")
        # val
        tp_val = os.path.join(file_path_data, "tp_val.txt")
        fp_val = os.path.join(file_path_data, "fp_val.txt")
        fn_val = os.path.join(file_path_data, "fn_val.txt")
        tn_val = os.path.join(file_path_data, "tn_val.txt")
        tp_val = loadtxt(tp_val, comments="#", delimiter=",", unpack=False)
        fp_val = loadtxt(fp_val, comments="#", delimiter=",", unpack=False)
        fn_val = loadtxt(fn_val, comments="#", delimiter=",", unpack=False)
        tn_val = loadtxt(tn_val, comments="#", delimiter=",", unpack=False)
        final_pdf = os.path.join(RESULTS_DIR, "tp_fp_fn_tn_val.pdf")
        final_pdf_adj = os.path.join(RESULTS_DIR, "tp_fp_fn_tn_val_adj.pdf")
        make_pixelclass_graph(file_save_path=final_pdf, tp_list=tp_val, fp_list=fp_val, fn_list=fn_val, tn_list=tn_val,
                              epoch_skip=0, extra_title="val")
        make_pixelclass_graph(file_save_path=final_pdf_adj, tp_list=tp_val, fp_list=fp_val, fn_list=fn_val,
                              tn_list=tn_val, epoch_skip=num_skip, extra_title="val")
        # precision and recall
        final_pdf = os.path.join(RESULTS_DIR, "precsion_recall_val.pdf")
        final_pdf_adj = os.path.join(RESULTS_DIR, "precsion_recall_val_adj.pdf")
        make_precision_recall_curves(file_save_path=final_pdf, tp_list=tp_val, fp_list=fp_val, fn_list=fn_val,
                                     epoch_skip=0, min_focal_val_epoch=val_index_min_focal, extra_title="val")
        make_precision_recall_curves(file_save_path=final_pdf_adj, tp_list=tp_val, fp_list=fp_val, fn_list=fn_val,
                                     epoch_skip=num_skip, min_focal_val_epoch=val_index_min_focal, extra_title="val")
        # roc plot val
        fpr_file = os.path.join(RESULTS_DIR, "val_fprs.txt")
        tpr_file = os.path.join(RESULTS_DIR, "val_tprs.txt")
        final_pdf = os.path.join(RESULTS_DIR, "roc_plot_val.pdf")
        make_roc_graph(fprs=fpr_file, tprs=tpr_file, file_save_path=final_pdf, extra_name="val")
    elif testing:
        # this means that we only have a single curve for the combined train and validation dataset
        # loading the data for the loss function (that used to update the wights)
        file_one = os.path.join(file_path_data, args.loss_name + "_loss_trainval.txt")
        file_one_list_train = loadtxt(file_one, comments="#", delimiter=",", unpack=False)

        # finding the epoch where the minimum val focal loss occurs
        trainval_index_min_focal = np.where(file_one_list_train == min(file_one_list_train))[0][0]
        print("the epoch where the val min loss occurs", trainval_index_min_focal)
        final_pdf = os.path.join(RESULTS_DIR, args.loss_name + "_image_trainval.pdf")
        final_pdf_adj = os.path.join(RESULTS_DIR, args.loss_name + "_image_adj_trainval.pdf")
        make_graph_trainval(args.loss_name, final_pdf, file_one_list_train,
                            epoch_skip=0, epoch_min_val_loss=trainval_index_min_focal)
        make_graph_trainval(args.loss_name, final_pdf_adj, file_one_list_train,
                            epoch_skip=num_skip, epoch_min_val_loss=trainval_index_min_focal)

        # cancer files
        file_one_cancer = os.path.join(file_path_data, "jacc_loss_trainval_cancer.txt")
        file_one_list_train_cancer = loadtxt(file_one_cancer, comments="#", delimiter=",", unpack=False)
        # noncancer files
        file_one_noncancer = os.path.join(file_path_data, "jacc_loss_trainval_noncancer.txt")
        file_one_list_train_noncancer = loadtxt(file_one_noncancer, comments="#", delimiter=",", unpack=False)

        # cancer graphs
        final_pdf_cancer = os.path.join(RESULTS_DIR, "jacc_loss_image_cancer_trainval.pdf")
        final_pdf_adj_cancer = os.path.join(RESULTS_DIR, "jacc_loss_image_adj_cancer_trainval.pdf")
        make_graph_trainval("jacc loss cancer", final_pdf_cancer,
                            data_list_train=file_one_list_train_cancer,
                            epoch_skip=0, epoch_min_val_loss=trainval_index_min_focal)
        make_graph_trainval("jacc loss cancer", final_pdf_adj_cancer,
                            data_list_train=file_one_list_train_cancer,
                            epoch_skip=num_skip, epoch_min_val_loss=trainval_index_min_focal)
        # noncancer graphs
        final_pdf_noncancer = os.path.join(RESULTS_DIR, "jacc_loss_image_noncancer_trainval.pdf")
        final_pdf_adj_noncancer = os.path.join(RESULTS_DIR, "jacc_loss_image_adj_noncancer_trainval.pdf")
        make_graph_trainval("jacc loss noncancer", final_pdf_noncancer,
                            data_list_train=file_one_list_train_noncancer,
                            epoch_skip=0, epoch_min_val_loss=trainval_index_min_focal)
        make_graph_trainval("jacc loss noncancer", final_pdf_adj_noncancer,
                            data_list_train=file_one_list_train_noncancer,
                            epoch_skip=num_skip, epoch_min_val_loss=trainval_index_min_focal)
        # intersection over union calculation for cancer class
        iou_train_cancer = [1 - i for i in file_one_list_train_cancer]  # i is the jaccard loss
        final_pdf_cancer = os.path.join(RESULTS_DIR, "iou_image_cancer_trainval.pdf")
        final_pdf_adj_cancer = os.path.join(RESULTS_DIR, "iou_image_adj_cancer_trainval.pdf")
        make_graph_trainval("iou_cancer", final_pdf_cancer, iou_train_cancer, epoch_skip=0,
                            epoch_min_val_loss=trainval_index_min_focal)
        make_graph_trainval("iou_cancer", final_pdf_adj_cancer, iou_train_cancer,
                            epoch_skip=num_skip, epoch_min_val_loss=trainval_index_min_focal)
        # IOU for noncancer
        iou_train_noncancer = [1 - i for i in file_one_list_train_noncancer]  # i is the jaccard loss
        final_pdf_noncancer = os.path.join(RESULTS_DIR, "iou_image_noncancer_trainval.pdf")
        final_pdf_adj_noncancer = os.path.join(RESULTS_DIR, "iou_image_adj_noncancer_trainval.pdf")
        make_graph_trainval("iou_noncancer", final_pdf_noncancer, iou_train_noncancer,
                            epoch_skip=0, epoch_min_val_loss=trainval_index_min_focal)
        make_graph_trainval("iou_noncancer", final_pdf_adj_noncancer,
                            iou_train_noncancer, epoch_skip=num_skip,
                            epoch_min_val_loss=trainval_index_min_focal)

        # mean iou graphs
        mean_iou_train = [(i + j) / 2 for i, j in zip(iou_train_cancer, iou_train_noncancer)]
        final_pdf_mean = os.path.join(RESULTS_DIR, "mean_iou_image_trainval.pdf")
        final_pdf_adj_mean = os.path.join(RESULTS_DIR, "mean_iou_image_adj_trainval.pdf")
        make_graph_trainval("mean_iou", final_pdf_mean, mean_iou_train, epoch_skip=0,
                            epoch_min_val_loss=trainval_index_min_focal)
        make_graph_trainval("mean_iou", final_pdf_adj_mean, mean_iou_train,
                            epoch_skip=num_skip, epoch_min_val_loss=trainval_index_min_focal)

        # dice score calculation for cancer class
        file_one_cancer = os.path.join(file_path_data, "dice_score_trainval_cancer.txt")
        file_one_list_train_cancer = loadtxt(file_one_cancer, comments="#", delimiter=",", unpack=False)
        final_pdf_cancer = os.path.join(RESULTS_DIR, "f1_image_cancer_trainval.pdf")
        final_pdf_adj_cancer = os.path.join(RESULTS_DIR, "f1_image_adj_cancer_trainval.pdf")
        make_graph_trainval("dice score (f1) cancer", final_pdf_cancer,
                            file_one_list_train_cancer,
                            epoch_skip=0, epoch_min_val_loss=trainval_index_min_focal)
        make_graph_trainval("dice score (f1) cancer", final_pdf_adj_cancer,
                            file_one_list_train_cancer,
                            epoch_skip=num_skip, epoch_min_val_loss=trainval_index_min_focal)
        # dice score calculation for noncancer class
        file_one_noncancer = os.path.join(file_path_data, "dice_score_trainval_noncancer.txt")
        file_one_list_train_noncancer = loadtxt(file_one_noncancer, comments="#", delimiter=",", unpack=False)
        final_pdf_noncancer = os.path.join(RESULTS_DIR, "f1_image_noncancer_trainval.pdf")
        final_pdf_adj_noncancer = os.path.join(RESULTS_DIR, "f1_image_adj_noncancer_trainval.pdf")
        make_graph_trainval("dice score (f1) noncancer", final_pdf_noncancer,
                            file_one_list_train_noncancer,
                            epoch_skip=0, epoch_min_val_loss=trainval_index_min_focal)
        make_graph_trainval("dice score (f1) noncancer", final_pdf_adj_noncancer,
                            file_one_list_train_noncancer,
                            epoch_skip=num_skip, epoch_min_val_loss=trainval_index_min_focal)
        # mean dice score calculation
        mean_dice_train = [(i + j) / 2 for i, j in zip(file_one_list_train_cancer, file_one_list_train_noncancer)]
        final_pdf_mean = os.path.join(RESULTS_DIR, "f1_image_mean_trainval.pdf")
        final_pdf_adj_mean = os.path.join(RESULTS_DIR, "f1_image_adj_mean_trainval.pdf")
        make_graph_trainval("dice score (f1) mean", final_pdf_mean, mean_dice_train,
                            epoch_skip=0, epoch_min_val_loss=trainval_index_min_focal)
        make_graph_trainval("dice score (f1) mean", final_pdf_adj_mean, mean_dice_train,
                            epoch_skip=num_skip, epoch_min_val_loss=trainval_index_min_focal)

        ### make graphs tp, fp, fn, tn
        # train
        tp_train = os.path.join(file_path_data, "tp_trainval.txt")
        fp_train = os.path.join(file_path_data, "fp_trainval.txt")
        fn_train = os.path.join(file_path_data, "fn_trainval.txt")
        tn_train = os.path.join(file_path_data, "tn_trainval.txt")
        tp_train = loadtxt(tp_train, comments="#", delimiter=",", unpack=False)
        fp_train = loadtxt(fp_train, comments="#", delimiter=",", unpack=False)
        fn_train = loadtxt(fn_train, comments="#", delimiter=",", unpack=False)
        tn_train = loadtxt(tn_train, comments="#", delimiter=",", unpack=False)
        final_pdf = os.path.join(RESULTS_DIR, "tp_fp_fn_tn_trainval.pdf")
        final_pdf_adj = os.path.join(RESULTS_DIR, "tp_fp_fn_tn_trainval_adj.pdf")
        make_pixelclass_graph(file_save_path=final_pdf, tp_list=tp_train, fp_list=fp_train, fn_list=fn_train,
                              tn_list=tn_train, epoch_skip=0, extra_title="trainval")
        make_pixelclass_graph(file_save_path=final_pdf_adj, tp_list=tp_train, fp_list=fp_train, fn_list=fn_train,
                              tn_list=tn_train, epoch_skip=num_skip, extra_title="trainval")
        # precision and recall
        final_pdf = os.path.join(RESULTS_DIR, "precsion_recall_trainval.pdf")
        final_pdf_adj = os.path.join(RESULTS_DIR, "precsion_recall_trainval_adj.pdf")
        make_precision_recall_curves(file_save_path=final_pdf, tp_list=tp_train, fp_list=fp_train, fn_list=fn_train,
                                     epoch_skip=0, min_focal_val_epoch=trainval_index_min_focal, extra_title="trainval")
        make_precision_recall_curves(file_save_path=final_pdf_adj, tp_list=tp_train, fp_list=fp_train, fn_list=fn_train,
                                     epoch_skip=num_skip, min_focal_val_epoch=trainval_index_min_focal,
                                     extra_title="trainval")
        # roc plot test
        fpr_file = os.path.join(RESULTS_DIR, "test_fprs.txt")
        tpr_file = os.path.join(RESULTS_DIR, "test_tprs.txt")
        final_pdf = os.path.join(RESULTS_DIR, "roc_plot_test.pdf")
        make_roc_graph(fprs=fpr_file, tprs=tpr_file, file_save_path=final_pdf, extra_name="test")


def make_roc_graph(fprs, tprs, file_save_path, extra_name='') -> None:
    """
    fprs is the file paths to the fprs something like os.path.join(RESULTS_DIR, 'val_tprs.txt')
    """
    fpr_load = np.loadtxt(fprs)
    tpr_load = np.loadtxt(tprs)
    auc_value = auc(fpr_load, tpr_load)
    plt.title("ROC curve for: " + extra_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr_load, tpr_load, label='AUC (area = {:.3f})'.format(auc_value))
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.legend(loc="lower right")
    file_path = os.path.join(file_save_path)
    plt.savefig(file_path)
    plt.close()


if __name__ == "__main__":
    main()

import torch
import random
import torch.optim as optim
import numpy as np
import os
from numpy import loadtxt
import sys
from datetime import datetime
from utils.config import make_model_options
from models.model import ModelMaker
from utils.utils import make_lr_schedulers
from dataset.data import DataMaker
from utils.metrics import ModelMetrics, make_loss_fn
from sklearn.metrics import auc


def main():
    random.seed(a=48)
    model_options = make_model_options()
    if model_options['model_type'] == "segment":
        from utils.train_fns import train_fn_segment as train_fn_one_epoch
        from utils.train_fns import check_accuracy_segment as check_accuracy
        from utils.utils import save_predictions_per_epoch_segment as save_preds_per_epoch
        from utils.utils import save_predictions_as_imgs_segment as save_best_preds
    elif model_options['model_type'] == "segmentdual":
        from dlp.utils.train_fns import train_fn_segmentdual as train_fn_one_epoch
        from dlp.utils.train_fns import check_accuracy_segmentdual as check_accuracy
        from dlp.utils.utils import save_predictions_per_epoch_segmentdual as save_preds_per_epoch
        from dlp.utils.utils import save_predictions_as_imgs_segmentdual as save_best_preds
    else:
        raise Exception("incorrect model type passed")

    device = torch.device("cuda:" + str(model_options['gpu_id']) if torch.cuda.is_available() else "error")

    print("clearing gpu cache")
    torch.cuda.empty_cache()
    model_obj = ModelMaker(model_params=model_options)
    model = model_obj.make_model()
    model_obj.print_model_params(model=model)
    model.to(device)
    train_transform = model_obj.make_train_transform()
    # the test_transform is the same as the val transform.
    test_transform = model_obj.make_val_transform()
    if model_options['model_type'] == "segmentdual":
        train_transform.add_targets({'image1': 'image'})
        test_transform.add_targets({'image1': 'image'})

    if model_options['model_type'] == "segmentdual":
        optimizer_decoder = optim.Adam(model.decoder.lv2decoder.parameters(), lr=model_options['lr_decoder'])
        optimizer_backbone = optim.Adam(model.encoder.encoder_context.parameters(), lr=model_options['lr_backbone'])
        optimizer_backbone_second = optim.Adam(model.encoder.encoder_detail.parameters(),
                                               lr=model_options['lr_backbone'])
        optimizer_mca = optim.Adam(model.decoder.mca.parameters(), lr=model_options['lr_mca'])
    else:
        optimizer_decoder = optim.Adam(model.decoder.parameters(), lr=model_options['lr_decoder'])
        optimizer_backbone = optim.Adam(model.encoder.parameters(), lr=model_options['lr_backbone'])
        optimizer_backbone_second = None
        optimizer_mca = None

    model_options['optimizers'] = {'optimizer_decoder': optimizer_decoder, 'optimizer_backbone': optimizer_backbone,
                                   'optimizer_backbone_second': optimizer_backbone_second, 'optimizer_mca': optimizer_mca}

    make_lr_schedulers(model_params=model_options)

    data_obj = DataMaker(model_params=model_options)
    loss_fn = make_loss_fn(name=model_options["loss_fn_name"])
    metrics = ModelMetrics(results_dir=model_options['results_dir'], loss_fn_name=loss_fn.name)

    # Note that method call returns train_loader, val_loader, test_loader in that order
    # we want to switch the val and test datasets, and so I am doing that here
    train_loader, _, test_loader = data_obj.get_loaders(model_params=model_options,
                                                        train_transform=train_transform,
                                                        val_transform=test_transform,
                                                        test_transform=test_transform, trainval=True)

    scaler = torch.cuda.amp.GradScaler()
    print(model_options)
    # now we need to find the epoch where the minimum val loss occurred, this is how long we train
    # the model for on the combined train and val set.
    file_two = os.path.join(model_options['results_dir'], model_options['loss_fn_name'] + "_loss_val.txt")
    file_two_list_val = loadtxt(file_two, comments="#", delimiter=",", unpack=False)
    val_index_min_focal = np.where(file_two_list_val == min(file_two_list_val))[0][0]
    print(" the epoch where the minimum val loss occurs is ", val_index_min_focal)
    for epoch in range(val_index_min_focal+1):
        if epoch == model_options['unfreeze']:
            model_options['update_backbone_weights'] = True
            if model_options['model_type'] == "segmentdual":
                for param in model.encoder.encoder_context.parameters():
                    param.requires_grad = True
                for param in model.encoder.encoder_detail.parameters():
                    param.requires_grad = True
                for param in model.decoder.lv2decoder.parameters():
                    param.requires_grad = True
            else:
                # this is the case where there is just a single encoder branch
                for param in model.encoder.parameters():
                    param.requires_grad = True

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Epoch number: ", epoch, " time is ", current_time)
        train_loss_jaccard_cancer, train_dice_cancer, train_loss_jaccard_noncancer, train_dice_noncancer, train_loss, train_tp, train_fp, train_tn, train_fn = train_fn_one_epoch(
            train_loader, model, loss_fn=loss_fn, scaler=scaler,
            device=device, model_hyperparams=model_options)
        if model_options['lr_schedule_type'] != "None":
            if model_options['model_type'] == "segmentdual":
                model_options['lr_scheduler']['scheduler_mca'].step()
            else:
                model_options['lr_scheduler']['scheduler_decoder'].step()
        if model_options['update_backbone_weights']:
            if model_options['model_type'] == "segmentdual":
                model_options['lr_scheduler']['scheduler_backbone'].step()
                model_options['lr_scheduler']['scheduler_backbone_second'].step()
                model_options['lr_scheduler']['scheduler_decoder'].step()
            else:
                model_options['lr_scheduler']['scheduler_backbone'].step()
                model_options['lr_scheduler']['scheduler_decoder'].step()

        metrics.write_to_file(file_name=metrics.trainval_loss, value=train_loss, epoch=epoch)
        metrics.write_to_file(file_name=metrics.trainval_jacc_cancer, value=train_loss_jaccard_cancer, epoch=epoch)
        metrics.write_to_file(file_name=metrics.trainval_dice_cancer, value=train_dice_cancer, epoch=epoch)
        metrics.write_to_file(file_name=metrics.trainval_jacc_noncancer, value=train_loss_jaccard_noncancer,
                              epoch=epoch)
        metrics.write_to_file(file_name=metrics.trainval_dice_noncancer, value=train_dice_noncancer, epoch=epoch)
        # fp, tp, fn, tn
        metrics.write_to_file(file_name=metrics.trainval_tp, value=train_tp, epoch=epoch)
        metrics.write_to_file(file_name=metrics.trainval_fp, value=train_fp, epoch=epoch)
        metrics.write_to_file(file_name=metrics.trainval_tn, value=train_tn, epoch=epoch)
        metrics.write_to_file(file_name=metrics.trainval_fn, value=train_fn, epoch=epoch)
        if epoch % 50 == 0:
            save_preds_per_epoch(num_samples=model_options['batch_size']-1, loader=train_loader, model=model,
                                 epoch=epoch, metric1_name="trainval_" + loss_fn.name, m1_num=train_loss,
                                 metric2_name="trainval jaccard cancer", m2_num=train_loss_jaccard_cancer,
                                 folder=os.path.join(model_options['results_dir'], "epoch_perdict/"), file_pref="trainval")
    # at this point we have trained on the combined dataset for the number of epochs that the minimum val loss occurred
    # on just the training data
    print('saved the weights after doing all the trainval data')
    torch.save(model.state_dict(), model_options['trainval_weight_path'])
    model.load_state_dict(torch.load(model_options['trainval_weight_path']))
    model.eval()
    #

    (test_loss_jaccard_cancer, test_dice_cancer, test_loss_jaccard_noncancer, test_dice_noncancer, test_loss, test_tp,
     test_fp, test_tn, test_fn, fprs, tprs, thresholds) = check_accuracy(
        test_loader, model, device=device, fn_loss=loss_fn)
    save_best_preds(model_options['batch_size']-1, test_loader, model, name="test", folder=model_options['results_dir'], device=device)
    # writing for the roc
    print('writing test fprs')
    file = open(os.path.join(model_options['results_dir'], 'test_fprs.txt'), 'w')
    for fpr in fprs:
        file.write(str(fpr) + "\n")
    file.close()
    print('writing test tprs')
    file = open(os.path.join(model_options['results_dir'], 'test_tprs.txt'), 'w')
    for tpr in tprs:
        file.write(str(tpr) + "\n")
    file.close()
    print('writing test thresholds')
    file = open(os.path.join(model_options['results_dir'], 'test_thresholds.txt'), 'w')
    for threshold in thresholds:
        file.write(str(threshold) + "\n")
    file.close()

    print("the test jaccard loss for cancer is: ", test_loss_jaccard_cancer)
    print("the test dice for cancer is: ", test_dice_cancer)
    print("the test jaccard loss for noncancer is: ", test_loss_jaccard_noncancer)
    print("the test dice for noncancer is: ", test_dice_noncancer)
    print("the test mean dsc: ", (test_dice_cancer + test_dice_noncancer) / 2)
    print("the test mean jaccard loss: ", (test_loss_jaccard_cancer + test_loss_jaccard_noncancer) / 2)
    print("the test loss is: ", test_loss)
    print("the test tp is: ", test_tp)
    print("the test fp is: ", test_fp)
    print("the test tn is: ", test_tn)
    print("the test fn is: ", test_fn)
    print("the test AUC is: ", auc(np.loadtxt(os.path.join(model_options['results_dir'], 'test_fprs.txt')),
                                   np.loadtxt(os.path.join(model_options['results_dir'], 'test_tprs.txt'))))
    print("finished everything, script is complete")
    sys.exit(0)


def only_test_data():
    random.seed(a=48)
    model_options = make_model_options()
    if model_options['model_type'] == "segment":
        from utils.train_fns import train_fn_segment as train_fn_one_epoch
        from utils.train_fns import check_accuracy_segment as check_accuracy
        from utils.utils import save_predictions_per_epoch_segment as save_preds_per_epoch
        from utils.utils import save_predictions_as_imgs_segment as save_best_preds
    elif model_options['model_type'] == "segmentdual":
        from dlp.utils.train_fns import train_fn_segmentdual as train_fn_one_epoch
        from dlp.utils.train_fns import check_accuracy_segmentdual as check_accuracy
        from dlp.utils.utils import save_predictions_per_epoch_segmentdual as save_preds_per_epoch
        from dlp.utils.utils import save_predictions_as_imgs_segmentdual as save_best_preds
    else:
        raise Exception("incorrect model type passed")

    device = torch.device("cuda:" + str(model_options['gpu_id']) if torch.cuda.is_available() else "error")

    print("clearing gpu cache")
    torch.cuda.empty_cache()
    model_obj = ModelMaker(model_params=model_options)
    model = model_obj.make_model()
    model_obj.print_model_params(model=model)
    model.to(device)

    train_transform = model_obj.make_train_transform()
    # the test_transform is the same as the val transform.
    test_transform = model_obj.make_val_transform()
    if model_options['model_type'] == "segmentdual":
        train_transform.add_targets({'image1': 'image'})
        test_transform.add_targets({'image1': 'image'})

    data_obj = DataMaker(model_params=model_options)
    loss_fn = make_loss_fn(name=model_options["loss_fn_name"])
    # metrics = ModelMetrics(results_dir=model_options['results_dir'], loss_fn_name=loss_fn.name)

    # Note that method call returns train_loader, val_loader, test_loader in that order
    # we want to switch the val and test datasets, and so I am doing that here
    train_loader, _, test_loader = data_obj.get_loaders(model_params=model_options,
                                                        train_transform=train_transform,
                                                        val_transform=test_transform,
                                                        test_transform=test_transform, trainval=True)

    # scaler = torch.cuda.amp.GradScaler()
    print(model_options)

    (test_loss_jaccard_cancer, test_dice_cancer, test_loss_jaccard_noncancer, test_dice_noncancer, test_loss, test_tp,
     test_fp, test_tn, test_fn, fprs, tprs, thresholds) = check_accuracy(
        test_loader, model, device=device, fn_loss=loss_fn)
    save_best_preds(model_options['batch_size']-1, test_loader, model, name="test", folder=model_options['results_dir'], device=device)
    # writing for the roc
    print('writing test fprs')
    file = open(os.path.join(model_options['results_dir'], 'test_fprs.txt'), 'w')
    for fpr in fprs:
        file.write(str(fpr) + "\n")
    file.close()
    print('writing test tprs')
    file = open(os.path.join(model_options['results_dir'], 'test_tprs.txt'), 'w')
    for tpr in tprs:
        file.write(str(tpr) + "\n")
    file.close()
    print('writing test thresholds')
    file = open(os.path.join(model_options['results_dir'], 'test_thresholds.txt'), 'w')
    for threshold in thresholds:
        file.write(str(threshold) + "\n")
    file.close()

    print("the test jaccard loss for cancer is: ", test_loss_jaccard_cancer)
    print("the test dice for cancer is: ", test_dice_cancer)
    print("the test jaccard loss for noncancer is: ", test_loss_jaccard_noncancer)
    print("the test dice for noncancer is: ", test_dice_noncancer)
    print("the test mean dsc: ", (test_dice_cancer + test_dice_noncancer) / 2)
    print("the test mean jaccard loss: ", (test_loss_jaccard_cancer + test_loss_jaccard_noncancer) / 2)
    print("the test loss is: ", test_loss)
    print("the test tp is: ", test_tp)
    print("the test fp is: ", test_fp)
    print("the test tn is: ", test_tn)
    print("the test fn is: ", test_fn)
    print("the test AUC is: ", auc(np.loadtxt(os.path.join(model_options['results_dir'], 'test_fprs.txt')),
                                   np.loadtxt(os.path.join(model_options['results_dir'], 'test_tprs.txt'))))
    print("finished everything, script is complete")
    sys.exit(0)


if __name__ == "__main__":
    # main()
    only_test_data()

import sys
import os
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import random
from sklearn.metrics import auc
from models.model import ModelMaker
from utils.config import make_model_options
from dataset.data import DataMaker
from utils.metrics import ModelMetrics, make_loss_fn
from utils.utils import make_lr_schedulers


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
        optimizer_backbone = optim.Adam(model.encoder.encoder_context.parameters(), lr=model_options['lr_backbone'])
        optimizer_backbone_second = optim.Adam(model.encoder.encoder_detail.parameters(),
                                               lr=model_options['lr_backbone'])
        optimizer_decoder = optim.Adam(model.decoder.lv2decoder.parameters(), lr=model_options['lr_decoder'])
        optimizer_mca = optim.Adam(model.decoder.mca.parameters(), lr=model_options['lr_mca'])
    else:
        optimizer_decoder = optim.Adam(model.decoder.parameters(), lr=model_options['lr_decoder'])
        optimizer_backbone = optim.Adam(model.encoder.parameters(), lr=model_options['lr_backbone'])
        optimizer_backbone_second = None
        optimizer_mca = None

    model_options['optimizers'] = {'optimizer_decoder': optimizer_decoder, 'optimizer_backbone': optimizer_backbone,
                                   'optimizer_backbone_second': optimizer_backbone_second,
                                   'optimizer_mca': optimizer_mca}

    make_lr_schedulers(model_params=model_options)
    data_obj = DataMaker(model_params=model_options)
    loss_fn = make_loss_fn(name=model_options["loss_fn_name"])
    metrics = ModelMetrics(results_dir=model_options['results_dir'], loss_fn_name=loss_fn.name)

    # Note that method call returns train_loader, val_loader, test_loader in that order
    train_loader, val_loader, _ = data_obj.get_loaders(model_params=model_options,
                                                       train_transform=train_transform,
                                                       val_transform=test_transform,
                                                       test_transform=test_transform, trainval=False)

    scaler = torch.cuda.amp.GradScaler()
    lowest_val_loss = 5  # this is an arbitrary value for initialization
    print(model_options)
    for epoch in range(model_options['num_epochs']):
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
        (train_loss_jaccard_cancer, train_dice_cancer, train_loss_jaccard_noncancer, train_dice_noncancer, train_loss,
         train_tp, train_fp, train_tn, train_fn) = train_fn_one_epoch(
            train_loader, model, loss_fn=loss_fn, scaler=scaler,
            device=device, model_hyperparams=model_options)

        (val_loss_jaccard_cancer, val_dice_cancer, val_loss_jaccard_noncancer, val_dice_noncancer, val_loss, val_tp,
         val_fp, val_tn, val_fn, _, _, _) = check_accuracy(
            val_loader, model, device=device, fn_loss=loss_fn)
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

        # This is part is to save the model weights only if the validation loss improved
        if val_loss < lowest_val_loss:
            print("saving weights")
            lowest_val_loss = val_loss
            torch.save(model.state_dict(), model_options['lowest_val_loss_weights_path'])

        # loss used to update the weights usually bce or focal
        metrics.write_to_file(file_name=metrics.train_loss, value=train_loss, epoch=epoch)
        metrics.write_to_file(file_name=metrics.val_loss, value=val_loss, epoch=epoch)
        # jaccard loss. used to calculate the iou
        metrics.write_to_file(file_name=metrics.train_jacc_cancer, value=train_loss_jaccard_cancer, epoch=epoch)
        metrics.write_to_file(file_name=metrics.val_jacc_cancer, value=val_loss_jaccard_cancer, epoch=epoch)
        # dice score aka f1 score
        metrics.write_to_file(file_name=metrics.train_dice_cancer, value=train_dice_cancer, epoch=epoch)
        metrics.write_to_file(file_name=metrics.val_dice_cancer, value=val_dice_cancer, epoch=epoch)
        # noncancer iou and f1 score
        # jaccard loss. used to calculate the iou
        metrics.write_to_file(file_name=metrics.train_jacc_noncancer, value=train_loss_jaccard_noncancer, epoch=epoch)
        metrics.write_to_file(file_name=metrics.val_jacc_noncancer, value=val_loss_jaccard_noncancer, epoch=epoch)
        # dice score aka f1 score
        metrics.write_to_file(file_name=metrics.train_dice_noncancer, value=train_dice_noncancer, epoch=epoch)
        metrics.write_to_file(file_name=metrics.val_dice_noncancer, value=val_dice_noncancer, epoch=epoch)

        metrics.write_to_file(file_name=metrics.lr_decoder,
                              value=optimizer_decoder.state_dict()['param_groups'][0]['lr'], epoch=epoch)
        metrics.write_to_file(file_name=metrics.lr_backbone,
                              value=optimizer_backbone.state_dict()['param_groups'][0]['lr'], epoch=epoch)
        if model_options['model_type'] == "segmentdual":
            metrics.write_to_file(file_name=metrics.lr_mca,
                                  value=optimizer_mca.state_dict()['param_groups'][0]['lr'], epoch=epoch)
        # fp, tp, fn, tn
        metrics.write_to_file(file_name=metrics.train_tp, value=train_tp, epoch=epoch)
        metrics.write_to_file(file_name=metrics.train_fp, value=train_fp, epoch=epoch)
        metrics.write_to_file(file_name=metrics.train_tn, value=train_tn, epoch=epoch)
        metrics.write_to_file(file_name=metrics.train_fn, value=train_fn, epoch=epoch)
        # val
        metrics.write_to_file(file_name=metrics.val_tp, value=val_tp, epoch=epoch)
        metrics.write_to_file(file_name=metrics.val_fp, value=val_fp, epoch=epoch)
        metrics.write_to_file(file_name=metrics.val_tn, value=val_tn, epoch=epoch)
        metrics.write_to_file(file_name=metrics.val_fn, value=val_fn, epoch=epoch)

        if epoch % 50 == 0:
            save_preds_per_epoch(num_samples=model_options['batch_size']-1, loader=train_loader, model=model,
                                 epoch=epoch, metric1_name="train_" + loss_fn.name, m1_num=train_loss,
                                 metric2_name="train jaccard cancer", m2_num=train_loss_jaccard_cancer,
                                 folder=os.path.join(model_options['results_dir'], "epoch_perdict/"), file_pref="train")
            save_preds_per_epoch(num_samples=model_options['batch_size']-1, loader=val_loader, model=model,
                                 epoch=epoch, metric1_name="val_" + loss_fn.name, m1_num=val_loss,
                                 metric2_name="val jaccard cancer", m2_num=val_loss_jaccard_cancer,
                                 folder=os.path.join(model_options['results_dir'], "epoch_perdict/"), file_pref="val")

    # load the weights that lead to lowest val bce
    model.load_state_dict(torch.load(model_options['lowest_val_loss_weights_path']))
    model.eval()
    save_best_preds(model_options['batch_size']-1, val_loader, model, name="val", folder=model_options['results_dir'], device=device)
    (val_loss_jaccard_cancer, val_dice_cancer, val_loss_jaccard_noncancer, val_dice_noncancer, val_loss, val_tp,
     val_fp, val_tn, val_fn, val_fpr, val_tpr, val_thresholds) = check_accuracy(
        val_loader, model, device=device, fn_loss=loss_fn)
    print('statistics for the val set using the model that gave the lowest val loss')
    print("the val jaccard loss for cancer is: ", val_loss_jaccard_cancer)
    print("the val dice for cancer is: ", val_dice_cancer)
    print("the val jaccard loss for noncancer is: ", val_loss_jaccard_noncancer)
    print("the val dice for noncancer is: ", val_dice_noncancer)
    print("the val mean dsc: ", (val_dice_cancer + val_dice_noncancer) / 2)
    print("the val mean jaccard loss: ", (val_loss_jaccard_cancer + val_loss_jaccard_noncancer) / 2)
    print("the val loss is: ", val_loss)
    print("the val tp is: ", val_tp)
    print("the val fp is: ", val_fp)
    print("the val tn is: ", val_tn)
    print("the val fn is: ", val_fn)
    print("finished everything, script is complete")
    # writing for the roc
    print('writing val fprs')
    file = open(os.path.join(model_options['results_dir'], 'val_fprs.txt'), 'w')
    for fpr in val_fpr:
        file.write(str(fpr) + "\n")
    file.close()
    print('writing val tprs')
    file = open(os.path.join(model_options['results_dir'], 'val_tprs.txt'), 'w')
    for tpr in val_tpr:
        file.write(str(tpr) + "\n")
    file.close()
    print('writing val thresholds')
    file = open(os.path.join(model_options['results_dir'], 'val_thresholds.txt'), 'w')
    for threshold in val_thresholds:
        file.write(str(threshold) + "\n")
    file.close()

    print("the val AUC is: ", auc(np.loadtxt(os.path.join(model_options['results_dir'], 'val_fprs.txt')),
                                   np.loadtxt(os.path.join(model_options['results_dir'], 'val_tprs.txt'))))
    print("finished everything, script is complete")
    sys.exit(0)


if __name__ == "__main__":
    main()

import torch
import matplotlib.pyplot as plt
import os
import torchvision
import matplotlib.image as mpimg
from torch.optim.lr_scheduler import LambdaLR


def make_lr_schedulers(model_params) -> None:
    if model_params['lr_schedule_type'] == "simple":
        lambda_decoder = lambda epoch: 0.25 if epoch > 200 else 0.5 if epoch > 100 else 1
        lambda_backbone = lambda epoch: 0.25 if epoch > 100 else 1
        lambda_mca = lambda epoch: 0.25 if epoch > 200 else 0.50 if epoch > 150 else 0.75 if epoch > 100 else 1
        # remember that the backbone is frozen so this will
        # take effect after the backbone unfreezees. So it really says unfreeze+100 when the LR drop will occur
        scheduler_backbone = LambdaLR(model_params['optimizers']['optimizer_backbone'], lr_lambda=lambda_backbone)
        scheduler_decoder = LambdaLR(model_params['optimizers']['optimizer_decoder'], lr_lambda=lambda_decoder)
        scheduler_backbone_second = None
        scheduler_mca = None
        if model_params['model_type'] == "segmentdual":
            scheduler_backbone_second = LambdaLR(model_params['optimizers']['optimizer_backbone_second'],
                                                 lr_lambda=lambda_backbone)
            scheduler_mca = LambdaLR(model_params['optimizers']['optimizer_mca'], lr_lambda=lambda_mca)
        model_params['lr_scheduler'] = {'scheduler_backbone': scheduler_backbone,
                                        'scheduler_decoder': scheduler_decoder,
                                        'scheduler_backbone_second': scheduler_backbone_second,
                                        'scheduler_mca': scheduler_mca}

    elif model_params['lr_schedule_type'] is None:
        pass
    else:
        raise Exception("the lr_scheduler_type is not supported")

    return


def save_predictions_per_epoch_segmentdual(
        num_samples, loader, model, epoch, metric1_name, m1_num, metric2_name, m2_num, file_pref: str,
        folder="saved_images/", device="cuda"
):
    model.eval()
    if not os.path.isdir(folder):
        os.makedirs(folder)
    count = 0
    for idx, (x_detail, x_context, y_detail, fnames) in enumerate(loader):
        count += 1
        if count == 20:
            break
        x_detail = x_detail.float().to(device=device)
        x_context = x_context.float().to(device=device)
        y_detail = y_detail.float().unsqueeze(1).to(device=device)  # just added might throw an error
        with torch.no_grad():
            preds = torch.sigmoid(model(x_detail, x_context))
            preds = (preds > 0.5).float()
            plt.figure(figsize=(14, 30))

            title_name = "Epoch: " + str(epoch) + " " + metric1_name + ": " + str(
                m1_num) + "\n" + metric2_name + ": " + str(
                m2_num) + "\n" + "order is detail, context, detail gt, detail pred"
            plt.suptitle(title_name)
            for i in range(num_samples - 1):
                torchvision.utils.save_image(x_detail[i], f"{folder}/input1detail.png")
                torchvision.utils.save_image(x_context[i], f"{folder}/input1context.png")
                torchvision.utils.save_image(y_detail[i], f"{folder}/input1gtdetail.png")
                torchvision.utils.save_image(preds[i], f"{folder}/input1preddetail.png")
                # now make the plot from the above saved images
                plt.subplot(num_samples, 4, 4 * i + 1)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1detail.png")
                plt.imshow(img)
                detail_image_name = fnames['image_detail'][i].split('/')[-1].split('_')
                plt.title('_'.join(detail_image_name[0:4]).ljust(27) + '\n' + '_'.join(detail_image_name[4:]).ljust(27))

                plt.subplot(num_samples, 4, 4 * i + 2)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1context.png")
                plt.imshow(img)
                context_image_name = fnames['image_context'][i].split('/')[-1].split('_')
                plt.title(
                    '_'.join(context_image_name[0:4]).ljust(27) + '\n' + '_'.join(context_image_name[4:]).ljust(27))

                plt.subplot(num_samples, 4, 4 * i + 3)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1gtdetail.png")
                plt.imshow(img)
                mask_name = fnames['mask_detail'][i].split('/')[-1].split('_')
                plt.title('_'.join(mask_name[0:4]).ljust(27) + '\n' + '_'.join(mask_name[4:]).ljust(27))

                plt.subplot(num_samples, 4, 4 * i + 4)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1preddetail.png")
                plt.imshow(img)

            file_path = folder + file_pref + "_samplePred_epoch" + str(epoch) + ".pdf"
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

    model.train()


def save_predictions_per_epoch_segment(
        num_samples, loader, model, epoch, metric1_name, m1_num, metric2_name, m2_num, file_pref: str,
        folder="saved_images/", device="cuda"

):
    model.eval()
    if not os.path.isdir(folder):
        os.makedirs(folder)
    count = 0
    for idx, (x, y, fnames) in enumerate(loader):
        count += 1
        if count == 20:
            break
        x = x.float().to(device=device)
        y = y.float().unsqueeze(1).to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            plt.figure(figsize=(7, 15))

            title_name = "Epoch: " + str(epoch) + " " + metric1_name + ": " + str(
                m1_num) + "\n" + metric2_name + ": " + str(m2_num) + "\n" + "order is input, gt, pred"
            plt.suptitle(title_name)
            for i in range(num_samples - 1):
                torchvision.utils.save_image(x[i], f"{folder}/input1.png")
                torchvision.utils.save_image(y[i], f"{folder}/input1gt.png")
                torchvision.utils.save_image(preds[i], f"{folder}/input1pred.png")
                # now make the plot from the above saved images
                plt.subplot(num_samples, 3, 3 * i + 1)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1.png")
                plt.imshow(img)
                # this is the entire file path, we only want the file name
                image_name = fnames['image'][i].split('/')[-1].split('_')

                plt.title('_'.join(image_name[0:4]).ljust(27) + '\n' + '_'.join(image_name[4:]).ljust(27))

                plt.subplot(num_samples, 3, 3 * i + 2)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1gt.png")
                plt.imshow(img)
                mask_name = fnames['mask'][i].split('/')[-1].split('_')
                plt.title('_'.join(mask_name[0:4]).ljust(27) + '\n' + '_'.join(mask_name[4:]).ljust(27))

                plt.subplot(num_samples, 3, 3 * i + 3)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1pred.png")
                plt.imshow(img)

            file_path = folder + file_pref + "_samplePred_epoch" + str(epoch) + ".pdf"
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

    model.train()


def save_predictions_as_imgs_segmentdual(
        num_samples, loader, model, name, folder="saved_images/", device="cuda") -> None:
    if not os.path.isdir(folder):
        os.makedirs(folder)
    count = 0

    for idx, (x_detail, x_context, y_detail, fnames) in enumerate(loader):
        count += 1
        if count == 20:
            break
        x_detail = x_detail.float().to(device=device)
        x_context = x_context.float().to(device=device)
        y_detail = y_detail.float().unsqueeze(1).to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x_detail, x_context))
            preds = (preds > 0.5).float()
            plt.figure(figsize=(12, 30))

            title_name = "input, gt, pred".rjust(20)
            plt.suptitle(title_name)
            for i in range(num_samples - 1):
                torchvision.utils.save_image(x_detail[i], f"{folder}/input1detail.png")
                torchvision.utils.save_image(x_context[i], f"{folder}/input1context.png")
                torchvision.utils.save_image(y_detail[i], f"{folder}/input1gtdetail.png")
                torchvision.utils.save_image(preds[i], f"{folder}/input1preddetail.png")
                # now make the plot from the above saved images
                plt.subplot(num_samples, 4, 4 * i + 1)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1detail.png")
                plt.imshow(img)
                detail_image_name = fnames['image_detail'][i].split('/')[-1].split('_')
                plt.title('_'.join(detail_image_name[0:4]).ljust(27) + '\n' + '_'.join(detail_image_name[4:]).ljust(27))

                plt.subplot(num_samples, 4, 4 * i + 2)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1context.png")
                plt.imshow(img)
                context_image_name = fnames['image_context'][i].split('/')[-1].split('_')
                plt.title(
                    '_'.join(context_image_name[0:4]).ljust(27) + '\n' + '_'.join(context_image_name[4:]).ljust(27))

                plt.subplot(num_samples, 4, 4 * i + 3)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1gtdetail.png")
                plt.imshow(img)
                mask_name = fnames['mask_detail'][i].split('/')[-1].split('_')
                plt.title('_'.join(mask_name[0:4]).ljust(27) + '\n' + '_'.join(mask_name[4:]).ljust(27))

                plt.subplot(num_samples, 4, 4 * i + 4)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1preddetail.png")
                plt.imshow(img)

            file_path = os.path.join(folder, name + "_batch" + str(idx) + "BestModelPredictions.pdf")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

    model.train()


def save_predictions_as_imgs_segment(
        num_samples, loader, model, name, folder="saved_images/", device="cuda") -> None:
    if not os.path.isdir(folder):
        os.makedirs(folder)
    count = 0
    for idx, (x, y, fnames) in enumerate(loader):
        count += 1
        if count == 20:
            break
        x = x.float().to(device=device)
        y = y.float().unsqueeze(1).to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            plt.figure(figsize=(14, 30))
            title_name = "input det, context, det gt, pred".rjust(20)
            plt.suptitle(title_name)
            for i in range(num_samples - 1):
                torchvision.utils.save_image(x[i], f"{folder}/input1.png")
                torchvision.utils.save_image(y[i], f"{folder}/input1gt.png")
                torchvision.utils.save_image(preds[i], f"{folder}/input1pred.png")
                # now make the plot from the above saved images
                plt.subplot(num_samples, 3, 3 * i + 1)  # I had 3 * i+1
                plt.axis('off')
                img = mpimg.imread(folder + "/input1.png")
                plt.imshow(img)
                image_name = fnames['image'][i].split('/')[-1].split('_')
                plt.title('_'.join(image_name[0:4]).ljust(27) + '\n' + '_'.join(image_name[4:]).ljust(27))

                plt.subplot(num_samples, 3, 3 * i + 2)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1gt.png")
                plt.imshow(img)
                mask_name = fnames['mask'][i].split('/')[-1].split('_')
                plt.title('_'.join(mask_name[0:4]).ljust(27) + '\n' + '_'.join(mask_name[4:]).ljust(27))

                plt.subplot(num_samples, 3, 3 * i + 3)
                plt.axis('off')
                img = mpimg.imread(folder + "/input1pred.png")
                plt.imshow(img)

        file_path = os.path.join(folder, name + "_batch" + str(idx) + "BestModelPredictions.pdf")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    pass

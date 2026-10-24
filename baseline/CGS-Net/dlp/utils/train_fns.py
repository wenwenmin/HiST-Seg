import torch
from .metrics import jacc_loss, dice_coeff, pixel_class_accuracy
from sklearn import metrics


def train_fn_segment(loader, model, loss_fn, scaler, device, model_hyperparams):
    # will do one epoch of training
    loss_score = 0
    jac_loss_score_cancer = 0
    jac_loss_score_noncancer = 0
    dice_score_cancer = 0
    dice_score_noncancer = 0
    tp_score = 0
    fp_score = 0
    tn_score = 0
    fn_score = 0

    for batch_idx, (data, targets, _) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        l1_norm = 0
        l2_norm = 0
        if model_hyperparams['l2weight'] != 0 or model_hyperparams['l1weight'] != 0:
            # the idea here is that we will always have l2 applied to the decoder since the encoder is frozen

            l1_norm = sum(p.abs().sum() for p in model.decoder.parameters())
            l2_norm = sum(p.pow(2.0).sum() for p in model.decoder.parameters())

        loss = loss + model_hyperparams['l1weight'] * l1_norm + model_hyperparams['l2weight'] * l2_norm
        loss_score += loss.detach().item()  # the .item does not do gradients

        # backward-- update weights
        # note in the segment one there should be no second backbone weights. they were removed
        if model_hyperparams['update_backbone_weights']:
            model_hyperparams['optimizers']['optimizer_backbone'].zero_grad()
        model_hyperparams['optimizers']['optimizer_decoder'].zero_grad()

        scaler.scale(loss).backward()

        if model_hyperparams['update_backbone_weights']:
            scaler.step(model_hyperparams['optimizers']['optimizer_backbone'])

        scaler.step(model_hyperparams['optimizers']['optimizer_decoder'])
        scaler.update()

        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(predictions) > 0.5).float()
            tn, tp, fp, fn = pixel_class_accuracy(prediction=preds, target=targets)
            tp_score += tp
            fp_score += fp
            tn_score += tn
            fn_score += fn

            jac_loss_score_cancer += jacc_loss(preds, targets).item()
            dice_score_cancer += dice_coeff(preds, targets).item()
            # what we just got was the cancer class jac and dice scores. So now we need to do the noncancer class (
            # normal) For this we will swap 0 and 1 in the targets and preds and then do the same thing as above.
            # Recall that the original encoding is 0 for noncancer and 1 for cancer. So if we swap the 0 and 1 then
            # we can get the noncancer class metrics
            targets_noncancer = (targets == 0).float()  # this will swap the 0 and 1
            preds_noncancer = (preds == 0).float()
            jac_loss_score_noncancer += jacc_loss(preds_noncancer, targets_noncancer).item()
            dice_score_noncancer += dice_coeff(preds_noncancer, targets_noncancer).item()

            # Very important distinction here the loss_fn has a built-in sigmoid, and so we
            # do not need the preds line otherwise we have two sigmoids. So we use the
            # predictions from above. the preds is only for extra metrics.
        model.train()

    return (jac_loss_score_cancer / len(loader)), dice_score_cancer / len(loader), (
            jac_loss_score_noncancer / len(loader)), dice_score_noncancer / len(loader), (
            loss_score / len(loader)), tp_score, fp_score, tn_score, fn_score


def train_fn_segmentdual(loader, model, loss_fn, scaler, device, model_hyperparams):
    """
    The idea here is that this code is adapted from the single input version to handle the dual inputs.
    """
    # will do one epoch of training
    loss_score = 0
    jac_loss_score_cancer = 0
    jac_loss_score_noncancer = 0
    dice_score_cancer = 0
    dice_score_noncancer = 0
    tp_score = 0
    fp_score = 0
    tn_score = 0
    fn_score = 0

    for batch_idx, (x_detail, x_context, targets, _) in enumerate(loader):
        x_detail = x_detail.to(device=device)
        x_context = x_context.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        predictions = model(x_detail, x_context)
        loss = loss_fn(predictions, targets)

        l1_norm = 0
        l2_norm = 0
        if model_hyperparams['l2weight'] != 0 or model_hyperparams['l1weight'] != 0:
            l1_norm = sum(p.abs().sum() for p in model.decoder.mca.parameters())
            l2_norm = sum(p.pow(2.0).sum() for p in model.decoder.mca.parameters())

        loss = loss + model_hyperparams['l1weight'] * l1_norm + model_hyperparams['l2weight'] * l2_norm
        loss_score += loss.item()  # the .item does not do gradients

        # backward-- update weights
        if model_hyperparams['update_backbone_weights']:
            model_hyperparams['optimizers']['optimizer_backbone'].zero_grad()
            model_hyperparams['optimizers']['optimizer_backbone_second'].zero_grad()
            model_hyperparams['optimizers']['optimizer_decoder'].zero_grad()

        model_hyperparams['optimizers']['optimizer_mca'].zero_grad()

        scaler.scale(loss).backward()

        if model_hyperparams['update_backbone_weights']:
            scaler.step(model_hyperparams['optimizers']['optimizer_backbone'])
            scaler.step(model_hyperparams['optimizers']['optimizer_backbone_second'])
            scaler.step(model_hyperparams['optimizers']['optimizer_decoder'])

        scaler.step(model_hyperparams['optimizers']['optimizer_mca'])
        scaler.update()

        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(predictions) > 0.5).float()
            tn, tp, fp, fn = pixel_class_accuracy(prediction=preds, target=targets)
            tp_score += tp
            fp_score += fp
            tn_score += tn
            fn_score += fn

            jac_loss_score_cancer += jacc_loss(preds, targets).item()
            dice_score_cancer += dice_coeff(preds, targets).item()

            targets_noncancer = (targets == 0).float()  # this will swap the 0 and 1
            preds_noncancer = (preds == 0).float()
            jac_loss_score_noncancer += jacc_loss(preds_noncancer, targets_noncancer).item()
            dice_score_noncancer += dice_coeff(preds_noncancer, targets_noncancer).item()
        model.train()
    return ((jac_loss_score_cancer / len(loader)), dice_score_cancer / len(loader),
            (jac_loss_score_noncancer / len(loader)), dice_score_noncancer / len(loader), (loss_score / len(loader)),
            tp_score, fp_score, tn_score, fn_score)


def check_accuracy_segment(loader, model, device, fn_loss):
    loss_score = 0
    jac_loss_score_cancer = 0
    jac_loss_score_noncancer = 0
    dice_score_cancer = 0
    dice_score_noncancer = 0
    tp_score = 0
    fp_score = 0
    tn_score = 0
    fn_score = 0
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            x = x.float()
            model_pred = model(x)

            y = y.float().to(device).unsqueeze(1)  # the label does not have a channel
            loss_score += fn_loss(model_pred, y).item()

            preds = torch.sigmoid(model_pred)
            all_probs.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())
            # remember each pixel here is a probability of belonging to the class. Each prob greater than 0.5 gets
            # converted to 1 else 0. This determining whether the model is confident
            preds = (preds > 0.5).float()
            tn, tp, fp, fn = pixel_class_accuracy(prediction=preds, target=y)
            tp_score += tp
            fp_score += fp
            tn_score += tn
            fn_score += fn
            jac_loss_score_cancer += jacc_loss(preds, y).item()
            dice_score_cancer += dice_coeff(preds, y).item()
            # for comments see the train function
            targets_noncancer = (y == 0).float()  # this will swap the 0 and 1
            preds_noncancer = (preds == 0).float()
            jac_loss_score_noncancer += jacc_loss(preds_noncancer, targets_noncancer).item()
            dice_score_noncancer += dice_coeff(preds_noncancer, targets_noncancer).item()

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_probs, drop_intermediate=False)
    model.train()

    return (
        (jac_loss_score_cancer / len(loader)), dice_score_cancer / len(loader),
        (jac_loss_score_noncancer / len(loader)),
        dice_score_noncancer / len(loader), loss_score / len(loader), tp_score, fp_score, tn_score, fn_score, fpr, tpr,
        thresholds)


def check_accuracy_segmentdual(loader, model, device, fn_loss):
    loss_score = 0
    jac_loss_score_cancer = 0
    jac_loss_score_noncancer = 0
    dice_score_cancer = 0
    dice_score_noncancer = 0
    all_probs = []
    all_labels = []
    tp_score = 0
    fp_score = 0
    tn_score = 0
    fn_score = 0
    model.eval()
    with torch.no_grad():
        for x_detail, x_context, y, _ in loader:
            x_detail = x_detail.float().to(device)
            x_context = x_context.float().to(device)
            model_pred = model(x_detail, x_context)
            y = y.float().to(device).unsqueeze(1)  # the label does not have a channel
            loss_score += fn_loss(model_pred, y).item()

            preds = torch.sigmoid(model_pred)
            # with these preds we need to track the roc data across all the batches
            all_probs.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())
            # remember each pixel here is a probability of belonging to the class. Each prob greater than 0.5 gets
            # converted to 1 else 0. This determining whether the model is confident
            preds = (preds > 0.5).float()
            tn, tp, fp, fn = pixel_class_accuracy(prediction=preds, target=y)
            tp_score += tp
            fp_score += fp
            tn_score += tn
            fn_score += fn
            jac_loss_score_cancer += jacc_loss(preds, y).item()
            dice_score_cancer += dice_coeff(preds, y).item()
            # for comments see the train function
            targets_noncancer = (y == 0).float()  # this will swap the 0 and 1
            preds_noncancer = (preds == 0).float()
            jac_loss_score_noncancer += jacc_loss(preds_noncancer, targets_noncancer).item()
            dice_score_noncancer += dice_coeff(preds_noncancer, targets_noncancer).item()

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_probs, drop_intermediate=False)
    model.train()

    return (jac_loss_score_cancer / len(loader)), dice_score_cancer / len(loader), (
            jac_loss_score_noncancer / len(loader)), dice_score_noncancer / len(loader), loss_score / len(
        loader), tp_score, fp_score, tn_score, fn_score, fpr, tpr, thresholds

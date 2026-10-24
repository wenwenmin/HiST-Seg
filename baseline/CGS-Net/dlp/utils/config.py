import argparse
import os


def get_cmdargs_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        help="the model name to use", required=True)
    parser.add_argument("--model_type", type=str,
                        help="segment or classify", default="segmentdual")
    parser.add_argument("--result_name", type=str,
                        help="the name to store the results", required=True)
    parser.add_argument("--lr_schedule", type=str,
                        help="Use a Learning Rate scheduler 1 for Yes, 0 for none", required=False, default="None")
    parser.add_argument("--data_dir", type=str,
                        help="", required=True)
    parser.add_argument("--gpu_id", type=int,
                        help="this should be the id of the free gpu that has been assigned to you", required=True)
    parser.add_argument("--epochs_num", type=int,
                        help="How many epochs are you running for?", required=False, default=500)
    parser.add_argument("--lone_weight", type=float,
                        help="the weight for L1", required=False, default=0.0)
    parser.add_argument("--ltwo_weight", type=float,
                        help="the weight for l two", required=True)
    parser.add_argument("--aug_type", type=str,
                        help="the weight for l one", required=True)
    parser.add_argument("--unfreeze", type=int,
                        help="the epoch to unfreeze the weights", required=True)
    parser.add_argument("--data_name", type=str,
                        help="which dataset are we using c16_lv2, c16_lv3, c16 for both levels", required=False,
                        default="c16")
    parser.add_argument("--loss_name", type=str,
                        help="bce or focal", required=True)
    parser.add_argument("--lr_decoder", type=float,
                        help="the lr for the decoder of the neural network", default=0.0003, required=False)
    parser.add_argument("--lr_backbone", type=float,
                        help="the lr for the backbone of the neural network", default=0.0003, required=False)
    parser.add_argument("--lr_mca", type=float,
                        help="LR for the MCa part", default=0.0003, required=False)
    parser.add_argument("--dropout", type=float,
                        help="The rate for dropout, note that all models may not support dropout", required=False)
    parser.add_argument("--weights", type=str,
                        help="which weights to use with the dual encoder", required=False)
    parser.add_argument("--batch_size", type=int,
                        help="batch size", required=True)

    return parser.parse_args()


def make_model_options():
    """

    The following describe the roles of the different key-value pairs:
    update_backbone_weights:  this variable is for updating the weights in the encoder (backbones). It starts out False
     because the encoder weights are frozen while the decoder weights are unfrozen. After training the decoder
     then the encoder weights are unfrozen.
    """
    # The idea here is to use the command line args and return a dictionary currently in train.py
    # this will be dynamic with defaults setup for the model_options dictionary. allowing less cmdline args
    # for example l1 always equals 0 so make 0 default and put 0 into dictionary. if the -l1_weight is present then
    # change the value
    cmdargs = get_cmdargs_train()
    lowest_val_loss_weight_path = os.path.join(cmdargs.result_name, 'lowest_val_loss.pth')
    trainval_weight_path = os.path.join(cmdargs.result_name, 'trainval.pth')
    model_options = {'update_backbone_weights': False, 'l1weight': cmdargs.lone_weight,
                     'l2weight': cmdargs.ltwo_weight, 'optimizers': None, 'input_img_size': 224,
                     'model_type': cmdargs.model_type, 'gpu_id': cmdargs.gpu_id, 'model_name': cmdargs.model_name,
                     'aug_type': cmdargs.aug_type,
                     'batch_size': cmdargs.batch_size, 'num_workers': 4, 'pin_mem': True,
                     'data_name': cmdargs.data_name,
                     'lowest_val_loss_weights_path': lowest_val_loss_weight_path, 'results_dir': cmdargs.result_name,
                     'data_dir': cmdargs.data_dir, 'lr_decoder': cmdargs.lr_decoder, 'lr_backbone': cmdargs.lr_backbone,
                     'lr_schedule_type': cmdargs.lr_schedule, 'lr_scheduler': None,
                     'num_epochs': cmdargs.epochs_num, 'unfreeze': cmdargs.unfreeze,
                     'trainval_weight_path': trainval_weight_path,
                     'loss_fn_name': cmdargs.loss_name,
                     'dropout': cmdargs.dropout, 'weights': cmdargs.weights, 'lr_mca': cmdargs.lr_mca}

    if not os.path.isdir(model_options['results_dir']):
        os.makedirs(model_options['results_dir'])

    return model_options


if __name__ == "__main__":
    print("You executed python config.py.")

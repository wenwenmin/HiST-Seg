#!/usr/bin/env bash

pwd; hostname; date

# File paths for results, data
modelfile="mit_b1_lv2"
DATA_DIR="./data"
RESULT_NAME="./results/${modelfile}/"
DATA_NAME="c16_lv2" #level 2 or level 3
WEIGHTS="trainval_lv2"

# Model parameters
MODEL="mit_b1"
AUGMENT="moreagg"
LOSSNAME="focal"
LR_SCHEDULER="simple"

L1=0.0006
L2=0.006
LR_DECODER=0.000003
LR_BACKBONE=0.0000001875
DROPOUT=0.125
BATCHSIZE=8

UNFREEZE=600
NUM_EPOCHS=1000

# scripts running
echo "Running script"
CUDA_ME=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs)

MTYPE="segment"
echo "the epoch where we unfreeze the weights is $UNFREEZE"
echo "the l1 weight regularization is $L1"
echo "the l2 weight regularization is $L2"
echo "the type of data augmentation is $AUGMENT"
echo "The dataset path is $DATA_DIR"
echo "The model selection is $MODEL"
echo "The number of epochs is $NUM_EPOCHS"
echo "The image size is $IMG_SIZE"
echo "The name of the results are $RESULT_NAME"
echo "Are you using a LR scheduler? $LR_SCHEDULER"
echo "The LR for the backbone is $LR_BACKBONE"
echo "The LR for the decoder is $LR_DECODER"
echo "The dropout is $DROPOUT"
echo "The loss function is $LOSSNAME"
echo "The data name is $DATA_NAME"
echo "The patch type is $PATCHES"
echo "The model type is $MTYPE"


#python ./dlp/train.py \
#--model_name $MODEL --model_type $MTYPE --result_name $RESULT_NAME --lr_schedule $LR_SCHEDULER --data_dir $DATA_DIR \
#--gpu_id $CUDA_ME --epochs_num $NUM_EPOCHS \
#--lone_weight $L1 --ltwo_weight $L2 --aug_type $AUGMENT --unfreeze $UNFREEZE \
#--data_name $DATA_NAME --loss_name $LOSSNAME \
#--lr_decoder $LR_DECODER --lr_backbone $LR_BACKBONE --dropout $DROPOUT --batch_size $BATCHSIZE --weights $WEIGHTS
#
#echo "Now doing the graphing for training and validation"
#
#python ./dlp/utils/graphing.py --source_data $RESULT_NAME --model_type $MTYPE --loss_name $LOSSNAME --testing 0

echo "now running the test analysis"
python ./dlp/test_analysis.py \
--model_name $MODEL --model_type $MTYPE --result_name $RESULT_NAME --lr_schedule $LR_SCHEDULER --data_dir $DATA_DIR \
--gpu_id $CUDA_ME --epochs_num $NUM_EPOCHS \
--lone_weight $L1 --ltwo_weight $L2 --aug_type $AUGMENT --unfreeze $UNFREEZE \
--data_name $DATA_NAME --loss_name $LOSSNAME \
--lr_decoder $LR_DECODER --lr_backbone $LR_BACKBONE --dropout $DROPOUT --batch_size $BATCHSIZE --weights $WEIGHTS

#python ./dlp/utils/graphing.py --source_data $RESULT_NAME --model_type $MTYPE --loss_name $LOSSNAME --testing 1
#echo "script has finished"
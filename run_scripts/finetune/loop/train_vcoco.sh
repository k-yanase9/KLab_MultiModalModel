#!/bin/bash

#引数を一つとる
# $1: bachsize

# 4090-

# Linear
batch_size=$1
dataset="vcoco"
model=$2
epoch=50
enc=2
dec=12
lr=1e-4


 if [ "$model" = "random" ]; then
                cp /home/k-yanase/src/KLab_MultiModalModel/pretrain.pth task_train.pth
        else    
                cp /home/k-yanase/qnap5/k-yanase/Klab_MultiModalModel/task_train_model/$model/epoch_50.pth task_train.pth
        fi


    
torchrun --nnodes=1 --nproc_per_node=1 finetune.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --transformer_model_init pretrain \
        --max_source_length 64 \
        --max_target_length 12 \
        --stage finetune \
        --loss CrossEntropy \
        --lr $lr \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs $epoch \
        --warmup_rate 0.01 \
        --datasets $dataset \
        --is_tgt_id \
        --root_dir /data01/ \
        --result_dir results/finetune/$dataset/$model/$1

torchrun --nnodes=1 --nproc_per_node=1 finetune_score.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --transformer_model_init pretrain \
        --max_source_length 64 \
        --max_target_length 12 \
        --stage finetune \
        --loss CrossEntropy \
        --lr $lr \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs $epoch \
        --warmup_rate 0.01 \
        --datasets $dataset \
        --is_tgt_id \
        --root_dir /data01/ \
        --result_dir results/finetune/$dataset/$model/$1

rm task_train.pth
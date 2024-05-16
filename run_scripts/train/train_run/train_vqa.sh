# 1ノード8GPU

# Linear
batch_size=300
dataset="vqa"

epoch=1

enc=2
dec=12
lr=1e-4

# python nodetest.py \
#         --transformer_num_layers $enc \
#         --transformer_num_decoder_layers $dec \
#         --transformer_model_init pretrain \
#         --stage train \
#         --loss CrossEntropy \
#         --lr $lr \
#         --lr_scheduler LinearWarmup \
#         -b $batch_size \
#         --start_epoch 0 \
#         --num_epochs $epoch \
#         --warmup_rate 0.01 \
#         --datasets $dataset \
#         --root_dir /home/data \
#         --uncalc_val \
#         --save_interval 1 \
#         --result_dir results/A100_40/node1_nomulti
torchrun --nnodes=1 --nproc_per_node=4 nodetest.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --transformer_model_init pretrain \
        --stage train \
        --loss CrossEntropy \
        --lr $lr \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --start_epoch 0 \
        --num_epochs $epoch \
        --warmup_rate 0.01 \
        --datasets $dataset \
        --root_dir /home/data \
        --uncalc_val \
        --save_interval 1 \
        --result_dir results/A100_40/node1_nomulti
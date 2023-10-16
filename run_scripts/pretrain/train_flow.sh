# 1ノード2GPU

batch_size=200
# dataset="cc3m cc12m imagenet imagenet21k places365 redcaps sun397"
dataset="redcaps"

enc=2
dec=12

/opt/conda/envs/mmm/bin/torchrun --nnodes=1 --nproc_per_node=2 train.py \
        --float_type float16 \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase pretrain \
        --loss CrossEntropy \
        --lr 1e-5 \
        --lr_scheduler CosineAnnealingLR \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs 1 \
        --datasets $dataset \
        --root_dir /data/group1/z40441a/ \
        --result_dir results/pretrain/$dataset/enc$enc\_dec$dec/

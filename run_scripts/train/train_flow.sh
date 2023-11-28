# 1ノード2GPU

# Linear
batch_size=64
dataset="all"

epoch=20

enc=2
dec=12

/opt/conda/envs/mmm/bin/torchrun --nnodes=1 --nproc_per_node=2 multi_task_train.py \
        --float_type float16 \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --stage train \
        --loss CrossEntropy \
        --lr 1e-4 \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --num_epochs $epoch \
        --warmup_rate 0.001 \
        --datasets $dataset \
        --root_dir /data/group1/z40441a/ \
        --save_interval 1 \
        --result_dir results/train/all/enc$enc\_dec$dec/Linear$epoch/

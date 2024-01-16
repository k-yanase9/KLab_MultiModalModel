# 1ノード8GPU H100

# Linear
batch_size=128
dataset="all"

epoch=20

enc=2
dec=12

torchrun --nnodes=1 --nproc_per_node=8 multi_task_pretrain.py \
        --transformer_d_ff 2048 \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --stage pretrain \
        --loss CrossEntropy \
        --lr 1e-4 \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs $epoch \
        --warmup_rate 0.001 \
        --datasets $dataset \
        --root_dir /local/ \
        --uncalc_val \
        --save_interval 1 \
        --result_dir results/pretrain/all_equal/enc$enc\_dec$dec/Linear$epoch/

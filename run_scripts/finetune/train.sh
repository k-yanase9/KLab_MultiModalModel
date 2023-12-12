# 1ノード2GPU

# Linear
batch_size=160
dataset="hico"

epoch=50

enc=2
dec=12
lr=1e-4

torchrun --nnodes=1 --nproc_per_node=1 finetune.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --max_source_length 64 \
        --max_target_length 2 \
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
        --root_dir /local/ \
        --result_dir results/finetune/hico/tgt_true/Linear$epoch\_$lr/

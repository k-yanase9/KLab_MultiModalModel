# Goal: 32768
dataset="imagenet"
src_len=7
tgt_len=74

for batch_size in 127; do
torchrun --nnodes=1 --nproc_per_node=4 train_dummy.py \
        --max_source_length $src_len \
        --max_target_length $tgt_len \
        --phase train \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --num_epochs 1 \
        --datasets $dataset \
        --root_dir /data01/ \
        --result_dir results/check_batch/$dataset/
done
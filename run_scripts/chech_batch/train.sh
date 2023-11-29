batch_size=64
torchrun --nnodes=1 --nproc_per_node=8 check_batch_size.py \
        --stage train \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --num_epochs 1 \
        --datasets "all" \
        --root_dir /local/ \
        --result_dir results/check_batch/$dataset/
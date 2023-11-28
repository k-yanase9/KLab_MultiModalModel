batch_size=64
torchrun --nnodes=1 --nproc_per_node=2 check_batch_size.py \
        --float_type float16 \
        --stage train \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --num_epochs 1 \
        --datasets "all" \
        --root_dir /data01/ \
        --result_dir results/check_batch/$dataset/
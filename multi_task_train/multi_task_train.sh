torchrun --nnodes=1 --nproc_per_node=4 multi_task_train.py \
        --result_dir results/multi_task_train/ \
        --num_epochs 2 \
        --lr 0.01 \
        --optimizer AdamW \

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25670 multi_task_train_multi_node.py \
        --result_dir results/multi_task_train/ \
        --num_epochs 2 \
        --lr 0.01 \
        --optimizer AdamW \

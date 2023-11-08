torchrun --nnodes=1 --nproc_per_node=4 --master_port 29600 multi_task_train2.py \
        --result_dir results/multi_task_train2/ \
        --num_epochs 2 \
        --lr 0.01 \
        --optimizer AdamW \

torchrun --nnodes=1 --nproc_per_node=4 train.py \
    --image_model_train \
    --accumulation_steps 2 \
    --num_steps 10000 \
    --data_dir /data/dataset/redcaps/ \
    --result_dir results/pretrain/with_swin/
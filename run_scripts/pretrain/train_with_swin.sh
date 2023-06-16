for model in "t5-base"
do
torchrun --nnodes=1 --nproc_per_node=4 train.py \
        --image_model_train \
        --ffn \
        --transformer_model_name $model \
        --lr 0.001 \
        --optimizer AdamW \
        --num_epochs 5 \
        --save_interval 1 \
        --data_dir /data/dataset/redcaps/ \
        --result_dir results/pretrain/with_swin/$model/
done

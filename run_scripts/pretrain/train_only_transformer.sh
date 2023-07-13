batch_size=64
for model in "google/flan-t5-small"
do
torchrun --nnodes=1 --nproc_per_node=8 train.py \
        --language_model_name google/flan-t5-base \
        --ffn \
        --transformer_model_name $model \
        --pretrain \
        --lr 0.001 \
        --optimizer AdamW \
        --batch_size $batch_size \
        --num_epochs 50 \
        --save_interval 1 \
        --data_dir /local/redcaps/ \
        --result_dir results/pretrain/redcaps/only_transformer/$model/
done

batch_size=256
for model in "google/flan-t5-small"
do
torchrun --nnodes=1 --nproc_per_node=4 train.py \
        --language_model_name google/flan-t5-base \
        --ffn \
        --transformer_model_name $model \
        --lr 0.001 \
        --optimizer AdamW \
        --batch_size $batch_size \
        --num_epochs 50 \
        --save_interval 50 \
        --data_dir /user/data/imagenet/ \
        --result_dir results/image_classify/only_transformer/$model/
done

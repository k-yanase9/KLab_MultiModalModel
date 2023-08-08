batch_size=128
for model in "google/flan-t5-small"
do
python test.py \
        --image_model_train \
        --language_model_name google/flan-t5-base \
        --ffn \
        --transformer_model_name $model \
        --lr 0.001 \
        --optimizer AdamW \
        --batch_size $batch_size \
        --num_epochs 50 \
        --save_interval 50 \
        --data_dir /data/dataset/imagenet_2012/ \
        --result_dir results/image_classify/with_swin/$model/
done
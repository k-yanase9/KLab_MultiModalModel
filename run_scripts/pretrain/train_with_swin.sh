batch_size=64
dataset="redcaps places365 sun397 inaturalist"

for enc in 4
do
        for dec in 4
        do
        torchrun --nnodes=1 --nproc_per_node=4 train.py \
                --language_model_name google/flan-t5-base \
                --image_model_train \
                --image_model_name microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft \
                --ffn \
                --transformer_num_layers $enc \
                --transformer_num_decoder_layers $dec \
                --pretrain \
                --lr 0.01 \
                --optimizer AdamW \
                --lr_scheduler StepLR \
                --batch_size $batch_size \
                --num_epochs 50 \
                --root_dir /local/ \
                --dataset $dataset \
                --result_dir results/pretrain_with_swin/redcaps_places365_sun397_inaturalist/enc$enc\_dec$dec/
        done
done
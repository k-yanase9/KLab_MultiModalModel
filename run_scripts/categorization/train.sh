batch_size=512
dataset="openimage"

epoch=50

dec=0
lr=1e-5
for enc in 2 4; do
torchrun --nnodes=1 --nproc_per_node=4 train.py \
        --language_model_name google/flan-t5-xxl \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase classify \
        --loss CrossEntropy \
        --lr $lr \
        --lr_scheduler CosineAnnealingLR \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs $epoch \
        --datasets $dataset \
        --root_dir /local/ \
        --result_dir results/localization/$dataset/enc$enc\_dec$dec\_xxl/cos$epoch\_$lr/
done
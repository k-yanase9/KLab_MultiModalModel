batch_size=512
dataset="openimage"

dec=0

for enc in 2; do
torchrun --nnodes=1 --nproc_per_node=4 train.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase classify \
        --loss CrossEntropy \
        --lr 0.01 \
        --lr_scheduler StepLR \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs 100 \
        --datasets $dataset \
        --root_dir /local/ \
        --result_dir results/localization/$dataset/enc$enc\_dec$dec/
done
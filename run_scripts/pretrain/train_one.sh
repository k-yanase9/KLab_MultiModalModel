batch_size=2

enc=2
dec=12

for dataset in "imagenet21k"; do
python train_one.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase pretrain \
        --loss CrossEntropy \
        --lr 0.01 \
        --lr_scheduler StepLR \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs 500 \
        --datasets $dataset \
        --root_dir /data01/ \
        --result_dir results/pretrain/$dataset\_one/enc$enc\_dec$dec/
done
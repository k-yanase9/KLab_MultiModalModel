batch_size=1
dataset="openimage"

enc=2
dec=0

python train_one.py \
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
        --root_dir /data/dataset/ \
        --result_dir results/localization/$dataset/enc$enc\_dec$dec\_one/

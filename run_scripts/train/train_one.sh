batch_size=2
dataset="sun397"

enc=2
dec=2

python train_one.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase train \
        --loss CrossEntropy \
        --lr 1e-5 \
        --lr_scheduler LambdaLR \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs 500 \
        --datasets $dataset \
        --root_dir /data01/ \
        --result_dir results/train/$dataset\_one/enc$enc\_dec$dec/
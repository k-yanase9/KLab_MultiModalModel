batch_size=1
dataset="sun397"

enc=2
dec=12

for dataset in "redcaps" "cc3m" "cc12m"; do
python train_one.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase train \
        --loss CrossEntropy \
        --lr 0.01 \
        --lr_scheduler StepLR \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs 100 \
        --datasets $dataset \
        --root_dir /data01/ \
        --result_dir results/caption/$dataset\_one/enc$enc\_dec$dec/
done
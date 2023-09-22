batch_size=1
dataset="sun397"

enc=2
dec=12

for dataset in "redcaps"; do
        python train_one.py \
                --transformer_num_layers $enc \
                --transformer_num_decoder_layers $dec \
                --phase train \
                --loss CrossEntropy \
                --lr 0.01 \
                --lr_scheduler StepLR \
                -b $batch_size \
                --start_epoch 1 \
                --num_epochs 50 \
                --save_interval 50 \
                --datasets $dataset \
                --root_dir /data01/ \
                --result_dir results/caption/$dataset\_one/enc$enc\_dec$dec/

        python train_one.py \
                --transformer_num_layers $enc \
                --transformer_num_decoder_layers $dec \
                --phase train \
                --loss CrossEntropy \
                --lr 0.01 \
                --lr_scheduler StepLR \
                -b $batch_size \
                --start_epoch 51 \
                --num_epochs 100 \
                --save_interval 50 \
                --datasets $dataset \
                --root_dir /data01/ \
                --result_dir results/caption/$dataset\_one/enc$enc\_dec$dec/
done
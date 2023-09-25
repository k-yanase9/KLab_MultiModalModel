batch_size=64
dataset="cc3m cc12m imagenet imagenet21k places365 redcaps sun397"

enc=2
dec=12

torchrun --nnodes=1 --nproc_per_node=8 train.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase pretrain \
        --loss CrossEntropy \
        --lr 0.01 \
        --lr_scheduler StepLR \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs 50 \
        --datasets $dataset \
        --root_dir /local/ \
        --save_interval 1 \
        --result_dir results/pretrain/cc3m_cc12m_imagenet_imagenet21k_places365_redcaps_sun397/enc$enc\_dec$dec/

enc=2
dec=12

python check_model_size.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --transformer_model_init random \
        --stage train \
        --loss CrossEntropy \
        --lr_scheduler LinearWarmup \
        --start_epoch 1 \
        --warmup_rate 0.01 \
        --root_dir /home/data/ \
        --save_interval 2 
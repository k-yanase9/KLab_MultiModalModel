# 1ノード8GPU

batch_size=64
dataset="cc3m imagenet sun397"

epoch=5

enc=2
dec=12

torchrun --nnodes=1 --nproc_per_node=1 multi_task_train_p1.py \
        --float_type float16 \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --stage train \
        --loss CrossEntropy \
        --lr 1e-4 \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --num_epochs $epoch \
        --warmup_rate 0.001 \
        --datasets $dataset \
        --root_dir /data/group1/z40441a/ \
        --save_interval 1 \
        --result_dir results/train/cc3m_imagenet_sun397/enc$enc\_dec$dec/Linear$epoch/

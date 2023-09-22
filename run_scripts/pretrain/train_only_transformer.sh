batch_size=50
dataset="cc3m cc12m imagenet inaturalist places365 redcaps sun397"

d_model=768
d_ff=3072
d_kv=64
num_heads=12
enc=2
dec=12

torchrun --nnodes=1 --nproc_per_node=8 train.py \
        -l google/flan-t5-large \
        -i microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft \
        --ffn \
        --transformer_d_model $d_model \
        --transformer_d_ff $d_ff \
        --transformer_d_kv $d_kv \
        --transformer_num_heads $num_heads \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --image_vocab_size 16384 \
        --loc_vocab_size 1000 \
        --pretrain \
        --lr 0.001 \
        --optimizer AdamW \
        --lr_scheduler StepLR \
        -b $batch_size \
        --start_epoch 21 \
        --num_epochs 40 \
        --root_dir /local/ \
        --save_interval 1 \
        --dataset $dataset \
        --result_dir results/pretrain/cc3m_cc12m_imagenet_inaturalist_places365_redcaps_sun397/enc$enc\_dec$dec/

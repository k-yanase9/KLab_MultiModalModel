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
        
# batch_size=2
# dataset="cc3m cc12m imagenet inaturalist places365 redcaps sun397"

# d_model=64
# d_ff=64
# d_kv=64
# num_heads=12
# enc=2
# dec=2

# torchrun --nnodes=1 --nproc_per_node=4 "multi_task_train.py" \
#         -l google/flan-t5-large \
#         -i microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft \
#         --ffn \
#         --transformer_d_model $d_model \
#         --transformer_d_ff $d_ff \
#         --transformer_d_kv $d_kv \
#         --transformer_num_heads $num_heads \
#         --transformer_num_layers $enc \
#         --transformer_num_decoder_layers $dec \
#         --pretrain \
#         --lr 0.01 \
#         --optimizer AdamW \
#         --lr_scheduler StepLR \
#         -b $batch_size \
#         --num_epochs 1 \
#         --root_dir /home/omote/gpu-node-data01 \
#         --dataset $dataset \
#         --result_dir results/pretrain/cc3m_cc12m_imagenet_inaturalist_places365_redcaps_sun397/enc$enc\_dec$dec/ \
        # --vae_ckpt_path ""

# 1ノード8GPU RTX6000Ada

# Linear
batch_size=32
dataset="cc3m vg_vqa vqa2 imagenet sun397"

epoch=20

enc=2
dec=12

torchrun --nnodes=1 --nproc_per_node=2 multi_task_train_p4.py \
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
        --result_dir results/train/cc3m_vg_vqa_vqa2_imagenet_sun397/enc$enc\_dec$dec/Linear$epoch/

# # Cosine
# torchrun --nnodes=1 --nproc_per_node=8 train.py \
#         --id t2vu6fas \
#         --transformer_num_layers $enc \
#         --transformer_num_decoder_layers $dec \
#         --phase pretrain \
#         --loss CrossEntropy \
#         --lr 1e-4 \
#         --lr_scheduler Cosine \
#         -b $batch_size \
#         --start_epoch 2 \
#         --num_epochs $epoch \
#         --warmup_rate 0.001 \
#         --datasets $dataset \
#         --root_dir /local/ \
#         --save_interval 1 \
#         --result_dir results/pretrain_Ada/cc3m_cc12m_imagenet_imagenet21k_places365_redcaps_sun397/enc$enc\_dec$dec/Cos$epoch/

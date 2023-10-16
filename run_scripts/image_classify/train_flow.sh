# 1ノード2GPU
batch_size=256
dataset="imagenet"

epoch=100

enc=2
dec=0
lr=1e-5
torchrun --nnodes=1 --nproc_per_node=2 train.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase classify \
        --loss CrossEntropy \
        --lr $lr \
        --lr_scheduler LambdaLR \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs $epoch \
        --datasets $dataset \
        --root_dir /data/group1/z40441a/ \
        --result_dir results/image_classify/$dataset/enc$enc\_dec$dec/Lambda$epoch\_$lr/

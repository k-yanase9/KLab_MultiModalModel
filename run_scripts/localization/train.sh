batch_size=512
dataset="openimage"

d_model=768
d_ff=3072
d_kv=64
num_heads=12
dec=0

for enc in 1 2; do
torchrun --nnodes=1 --nproc_per_node=4 train.py \
        -l google/flan-t5-large \
        -i microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft \
        --ffn \
        --transformer_d_model $d_model \
        --transformer_d_ff $d_ff \
        --transformer_d_kv $d_kv \
        --transformer_num_heads $num_heads \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --loss CrossEntropy \
        --lr 0.01 \
        --optimizer AdamW \
        --lr_scheduler StepLR \
        -b $batch_size \
        --num_epochs 100 \
        --root_dir /local/ \
        --dataset $dataset \
        --result_dir results/localization/$dataset/enc$enc\_dec$dec/
done
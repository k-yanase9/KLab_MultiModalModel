# Goal: 32768
dataset="imagenet"
src_len=7
tgt_len=74
batch_size=168

dataset="vg_det"
src_len=8
tgt_len=256
batch_size=57

dataset="vg_loc"
src_len=25
tgt_len=126
batch_size=108

dataset="cc3m"
src_len=7
tgt_len=256
batch_size=56

dataset="vg_rcap"
src_len=20
tgt_len=256
batch_size=55

dataset="vg_refexp"
src_len=256
tgt_len=120
batch_size=79

dataset="vg_cat"
src_len=22
tgt_len=17
batch_size=246

dataset="vg_rel"
src_len=50
tgt_len=25
batch_size=247

dataset="visual7w_vqa"
src_len=125
tgt_len=128
batch_size=93

dataset="vcr"
src_len=256
tgt_len=103
batch_size=86

torchrun --nnodes=1 --nproc_per_node=2 train_dummy.py \
        --float_type float16 \
        --max_source_length $src_len \
        --max_target_length $tgt_len \
        --stage train \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --num_epochs 1 \
        --datasets $dataset \
        --root_dir /data01/ \
        --result_dir results/check_batch/$dataset/
# a100-40

# Linear
batch_size=160
dataset="vcoco"
enc=8
dec=12
lr=1e-4
modelsize="enc${enc}_dec${dec}"
model="alltask_random"
basedir="/home/k-yanase/Klab_MultiModalModel/model_size_results/$modelsize/$model"
# best.pth ファイルの存在を確認してコピー
if [ -f "$basedir/multitask/best.pth" ]; then
    cp "$basedir/multitask/best.pth" task_train.pth
    echo "Copied $basedir/multitask/best.pth to task_train.pth"
else
    echo "Error: $basedir/multitask/best.pth not found."
    exit 1
fi

epoch=20


torchrun --nnodes=1 --nproc_per_node=4 finetune.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --transformer_model_init pretrain \
        --max_source_length 64 \
        --max_target_length 12 \
        --stage finetune \
        --loss CrossEntropy \
        --lr $lr \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs $epoch \
        --warmup_rate 0.01 \
        --datasets $dataset \
        --is_tgt_id \
        --root_dir /home/data/ \
        --result_dir $basedir/finetune/$dataset/

torchrun --nnodes=1 --nproc_per_node=1 finetune_score.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --transformer_model_init pretrain \
        --max_source_length 64 \
        --max_target_length 12 \
        --stage finetune \
        --loss CrossEntropy \
        --lr $lr \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs $epoch \
        --warmup_rate 0.01 \
        --datasets $dataset \
        --is_tgt_id \
        --root_dir /home/data/ \
        --result_dir $basedir/finetune/$dataset/


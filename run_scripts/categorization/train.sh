batch_size=512
dataset="openimage"

epoch=50

enc=2
dec=0
lr=1e-4

for model in "large" "xl" "xxl"; do
torchrun --nnodes=1 --nproc_per_node=4 train.py \
        --language_model_name google/flan-t5-$model \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --phase classify \
        --loss CrossEntropy \
        --lr $lr \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --num_epochs $epoch \
        --warmup_rate 0.001 \
        --root_dir /local/ \
        --datasets $dataset \
        --save_interval 10 \
        --result_dir results/categorization/$dataset/enc$enc\_dec$dec\_$model/Linear$epoch\_$lr/
done
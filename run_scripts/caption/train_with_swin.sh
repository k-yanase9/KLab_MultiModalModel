torchrun --nnodes=1 --nproc_per_node=4 train.py \
    --image_model_train \
    --num_epochs 50 \
    --save_interval 5 \
    --data_dir /user/data/mscoco2017/ \
    --result_dir results/caption/with_swin/
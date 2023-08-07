torchrun --nnodes=1 --nproc_per_node=4 train.py \
 --data_dir /data/dataset/openimage  \
 --image_model_train \
 --result_dir results/caption/oid \
 --batch_size 8 \
 --num_epochs 30 \
 --save_interval 5
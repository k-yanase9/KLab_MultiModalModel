torchrun --nnodes=1 --nproc_per_node=4 train.py \
 --data_dir /data/dataset/imSitu  \
 --image_model_train \
 --result_dir results/qva/imSitu/ \
 --batch_size 32 \
 --num_epochs 30 \
 --save_interval 5 \
 --ffn
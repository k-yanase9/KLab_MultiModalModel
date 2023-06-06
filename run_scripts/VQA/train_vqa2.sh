torchrun --nnodes=1 --nproc_per_node=4 train.py \
 --data_dir /data/dataset/vqa2  \
 --image_model_train \
 --result_dir results/qva/vqa2/ \
 --batch_size 64 \
 --num_epochs 30 \
 --save_interval 5 \
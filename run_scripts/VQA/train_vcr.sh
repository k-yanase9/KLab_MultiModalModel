torchrun --nnodes=1 --nproc_per_node=4 train_temp.py --data_dir /data/dataset/vcr  --image_model_train --result_dir results/qva/vcr/ --batch_size 8
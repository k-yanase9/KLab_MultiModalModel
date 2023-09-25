#!/bin/bash
# export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
# numactl -l $1

# . /opt/conda/etc/profile.d/conda.sh
# conda activate ./.conda_env
#pipenv shell
# pip freeze > world_rank_${OMPI_COMM_WORLD_RANK}.log
source ./.venv/bin/activate
python /home/w49009a/omote/KLab_MultiModalModel/multi_node.py

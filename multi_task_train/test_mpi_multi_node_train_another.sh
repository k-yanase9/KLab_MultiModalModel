#!/bin/bash -x
#PJM -L rscgrp=cx-workshop
#PJM -L node=2
#PJM -L jobenv=singularity
#PJM -L elapse=00:10:00
#PJM -j
#PJM -S

module load cuda
module load openmpi_cuda
module load gcc python cudnn nccl singularity
export OMPI_MCA_btl_tcp_if_include=ib0
mpirun -n 8 -machinefile $PJM_O_NODEINF -report-bindings -map-by ppr:4:node singularity exec --nv /home/w49009a/pytorch_omote.sif bash /home/w49009a/omote/KLab_MultiModalModel/multi_task_train/test_each_node_task_train.sh
#mpirun -n 8 -machinefile $PJM_O_NODEINF -x MASTER_PORT=25978 -report-bindings -map-by ppr:2:socket singularity exec --nv /home/w49009a/pytorch_omote.sif bash /home/w49009a/omote/KLab_MultiModalModel/multi_task_train/test_each_node_task_train.sh
# bash
# conda activate ./.conda_env
# mpirun -n 8 -machinefile $PJM_O_NODEINF -report-bindings -map-by ppr:2:socket bash ./test_each_node_task_python.sh

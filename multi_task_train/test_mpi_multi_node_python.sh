#!/bin/bash -x
#PJM -L rscgrp=cx-workshop
#PJM -L node=2
#PJM -L jobenv=singularity
#PJM -L elapse=00:10:00
#PJM -j
#PJM -S

module load cuda
module load openmpi_cuda
module load singularity
mpirun -n 8 -machinefile $PJM_O_NODEINF -report-bindings -map-by ppr:2:socket singularity exec --nv /home/w49009a/pytorch_omote.sif bash /home/w49009a/omote/KLab_MultiModalModel/test_each_node_task_python.sh
# bash
# conda activate ./.conda_env
# mpirun -n 8 -machinefile $PJM_O_NODEINF -report-bindings -map-by ppr:2:socket bash ./test_each_node_task_python.sh

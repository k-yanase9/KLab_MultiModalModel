#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
conda activate ./.conda_env
mpirun -n 8 -machinefile $PJM_O_NODEINF -report-bindings -map-by ppr:2:socket bash ./test_each_node_task_python.sh

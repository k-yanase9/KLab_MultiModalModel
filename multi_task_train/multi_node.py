import os 
import torch
import datasets
import transformers

node_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
with open(f"python_result_rank_{node_rank}.log",mode="w") as f:
    f.write(str(torch.cuda.is_available()))    

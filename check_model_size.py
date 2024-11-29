import os
import pkgutil
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

from data import *
from data.multi_task_dataloader import MultiTaskDataLoader4, get_multi_task_data
from models.model import MyModel
from modules import *

def count_parameters(model):
    # 総パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    # 訓練可能なパラメータ数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def check_model_size():
    args = parse_arguments()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # create model
    model = MyModel(args)
    total,trainable = count_parameters(model)
    print(f"total_param:{total},trainable:{trainable}")

    # M単位での表示
    total_m = total / 1e6  # 総パラメータ数 (M単位)
    trainable_m = trainable / 1e6  # 訓練可能パラメータ数 (M単位)
    
    print(f"Total Parameters: {total_m:.2f}M")
    print(f"Trainable Parameters: {trainable_m:.2f}M")
    
if __name__ == "__main__":
    check_model_size()

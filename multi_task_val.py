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
from models.model import MyModel
from modules import *

FULL_DATASET_NAME_DICT = {
    "caption": ["redcaps", "cc3m", "cc12m"], 
    "relation":["vg_rel", "openimage_rel"], 
    "rcap": ["grit20m_rcap", "vg_rcap"],
    "refexp": ["grit20m_refexp", "vg_refexp"],
    "det": ["vg_det", "openimage_det", "objects365_det"],
    "cat": ["vg_cat", "openimage_cat", "objects365_cat"],
    "loc": ["vg_loc", "openimage_loc", "objects365_loc"],
    "vqa": ["vg_vqa", "vqa2", "tdiuc", "imSitu", "visual7w_vqa"], 
    "gvqa": ["vcr", "visual7w_gvqa"],
    "classify": ["imagenet", "imagenet21k", "places365", "sun397"]}
# Flow
TASK_SAMPLE_NUM_DICT = {"caption": 6, "relation":2, "rcap":6, "refexp":4, "det":6, "cat":2, "loc":3, "vqa": 4, "gvqa":1, "classify": 2} #何回タスクごとにバッチを取得するか
# H100
TASK_SAMPLE_NUM_DICT = {"caption": 6, "relation":2, "rcap":6, "refexp":4, "det":6, "cat":2, "loc":3, "vqa": 4, "gvqa":1, "classify": 2} #何回タスクごとにバッチを取得するか
# General
SRC_LEN_DICT = {"caption": 7, "relation":50, "rcap":20, "refexp":184, "det":8, "cat":22, "loc":25, "vqa": 125, "gvqa":256, "classify": 7}
TGT_LEN_DICT = {"caption": 256, "relation":25, "rcap":256, "refexp":120, "det":256, "cat":17, "loc":126, "vqa": 128, "gvqa":103, "classify": 74}

#勾配をスケールする関数
def multiply_grad(optimizer, multiplier):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.mul_(multiplier)
                

use_wandb = False
if pkgutil.find_loader("wandb") is not None:
    import wandb

    use_wandb = True


def train():
    args = parse_arguments()
    if args.multinode:
        port_num = 27971
        host_list_file = os.environ["PJM_O_NODEINF"]
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        with open(host_list_file) as f:
            host = f.readlines()
        host[0] = host[0].rstrip("\n")
        dist_url = "tcp://" + host[0] + ":" + str(port_num)
        dist.init_process_group(backend="nccl", init_method=dist_url, rank=world_rank, world_size=args.world_size)
    else:
        dist.init_process_group(backend="nccl")
        args.world_size = torch.cuda.device_count()  # GPU数
        world_rank = dist.get_rank()
        local_rank = world_rank % args.world_size
        dist_url = "env://"

    if world_rank == 0:
        os.makedirs(args.result_dir, exist_ok=True)
        if use_wandb:
            wandb_init(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if world_rank == 0:
        logger = get_logger(args, log_name='val.log')

    # create model
    model = MyModel(args).to(local_rank)
    model = DDP(model, device_ids=[local_rank])#,find_unused_parameters=True)

    torch.cuda.empty_cache()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tgt_tokenizer = AutoTokenizer.from_pretrained(
        args.language_model_name,
        model_max_length=args.max_target_length,
        use_fast=True,
        extra_ids=0,
        additional_special_tokens=[f"<extra_id_{i}>" for i in range(100)]
        + [f"<loc_{i}>" for i in range(args.loc_vocab_size)]
        + [f"<add_{i}>" for i in range(args.additional_vocab_size)],
    )
    if args.language_model_train:
        src_tokenizer = tgt_tokenizer
    else:
        src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=args.max_source_length, use_fast=True)

    # データの設定
    # 引数の設定
    if args.datasets[0] == 'all':
        train_dataset_name_dict = FULL_DATASET_NAME_DICT
        train_task_sample_num_dict = TASK_SAMPLE_NUM_DICT
        train_src_len_dict = SRC_LEN_DICT
        train_tgt_len_dict = TGT_LEN_DICT
        args.datasets = []
        for v in train_dataset_name_dict.values():
            args.datasets.extend(v)
    else:
        train_task_sample_num_dict = {}
        train_src_len_dict = {}
        train_tgt_len_dict = {}
        for task, dataset_names in FULL_DATASET_NAME_DICT.items():
            for dataset_name in dataset_names:
                if dataset_name in args.datasets:
                    train_task_sample_num_dict[task] = TASK_SAMPLE_NUM_DICT[task]
                    train_src_len_dict[task] = SRC_LEN_DICT[task]
                    train_tgt_len_dict[task] = TGT_LEN_DICT[task]

    src_len_list = []
    tgt_len_list = []
    for task, sample in train_task_sample_num_dict.items():
        src_len_list.extend([train_src_len_dict[task]] * sample)
        tgt_len_list.extend([train_tgt_len_dict[task]] * sample)
    
    val_dataset = get_data(args, "val", src_tokenizer, tgt_tokenizer, max(src_len_list), max(tgt_len_list))
    if world_rank == 0:
        logger.info(f"val_dataset:{len(val_dataset)}")

    val_loader = get_distributed_dataloader(args, val_dataset, shuffle=False)

    loss_counter = LossCounter()
    with open(os.path.join(args.result_dir, 'train.log'), 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Train' in line:
                epoch = int(line.split('/')[0].split('(')[-1])
                loss = float(line.split(',')[1].split(':')[-1].strip())
                loss_counter.add("train", loss)
                if world_rank == 0 and use_wandb:
                    wandb.log({"epoch": epoch, "train/loss": loss})
    if args.start_epoch > 1:
        with open(os.path.join(args.result_dir, 'val.log'), 'r') as f:
            for line in f:
                if 'Epoch' in line and 'Val' in line:
                    loss_counter.add("val", float(line.split(',')[1].split(':')[-1].strip()))
        min_val_loss = min(loss_counter.losses['val'])
        if world_rank == 0:
            logger.info(f'[Loaded] Best Val loss : {min_val_loss}')
    else:
        min_val_loss = 100

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        if not os.path.exists(os.path.join(args.result_dir, f'epoch_{epoch}.pth')):
            if world_rank == 0:
                logger.info(f'epoch_{epoch}.pth is not found. Skip this epoch.')
            continue
        model.module.load(result_name=f'epoch_{epoch}.pth')
        torch.cuda.empty_cache()
        if world_rank == 0:
            logger.info(f'epoch_{epoch}.pth is loaded.')
        # 検証ループ
        if args.language_model_train:
            model.module.language_model.eval()
        if args.image_model_train:
            model.module.image_model.eval()
        model.module.transformer.eval()
        loss_per_step = torch.tensor(0.0).to(local_rank)
        accumulation_sample_size = torch.tensor(0).to(local_rank)
        val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(world_rank != 0))
        for i, (src_images, _, src_texts, tgt_texts) in enumerate(val_loop):
            #勾配更新の前準備
            with torch.no_grad():
                src_images = src_images.to(local_rank, non_blocking=True)
                src_texts = src_texts.to(local_rank, non_blocking=True)
                tgt_texts = tgt_texts.to(local_rank, non_blocking=True)

                loss, preds, sample_size = model(src_images, src_texts, None, tgt_texts, None)

                loss_per_step += loss.item()
                accumulation_sample_size += sample_size

        # 他のノードから集める
        dist.all_reduce(loss_per_step, op=dist.ReduceOp.SUM)
        dist.all_reduce(accumulation_sample_size, op=dist.ReduceOp.SUM)

        if world_rank == 0:
            val_loss = (loss_per_step / accumulation_sample_size).cpu().numpy().copy()
            loss_counter.add("val", val_loss)
            logger.info(f'[Epoch ({epoch}/{args.num_epochs}) Val] Loss : {val_loss}')
            if use_wandb:
                wandb.log({"epoch": epoch, "val/loss": val_loss})

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                logger.info('Best Model')
            
    if world_rank == 0: 
        if use_wandb:
            wandb.finish()
        loss_counter.plot_loss(args.result_dir, val_show=not args.uncalc_val)

def wandb_init(args):
    name = f'worldsize{args.world_size}_' + ' '.join(args.datasets)
    if args.id is None:
        args.id = wandb.util.generate_id()
    wandb.init(
        id=args.id,
        project=f"{args.stage}_val", 
        name=name,
        config=args,
        resume=True if args.start_epoch > 1 else False
    )
    wandb.define_metric("epoch")
    wandb.define_metric("iter")
    wandb.define_metric("iter/*", step_metric="iter")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")


if __name__ == "__main__":
    train()

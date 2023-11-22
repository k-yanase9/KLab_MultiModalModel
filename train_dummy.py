import os
import pkgutil
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from data import *
from models.model import MyModel
from modules import *

use_wandb = False
use_nvidia_smi = False
if pkgutil.find_loader("wandb") is not None:
    import wandb
    use_wandb = True
if pkgutil.find_loader("nvidia_smi") is not None:
    import nvidia_smi
    use_nvidia_smi = True

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
            if use_nvidia_smi:
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(local_rank)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                wandb.log({f"MemTotal{local_rank}":info.total // (1024**2)})

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if world_rank == 0:
        logger = get_logger(args)

    # create model
    model = MyModel(args).to(local_rank)
    model = DDP(model, device_ids=[local_rank])#,find_unused_parameters=True)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True if args.float_type == 'float16' else False)
    optimizer = get_optimizer(model, args)

    # データの設定
    image_size = 256
    src_vocab_size = 32128
    tgt_vocab_size = 32128 + args.loc_vocab_size + args.additional_vocab_size
    train_iter_per_epoch = 20
    val_iter_per_epoch = 10

    if 'Warmup' in args.lr_scheduler and args.num_steps is None:
        args.num_steps = args.num_epochs * train_iter_per_epoch
    scheduler = get_scheduler(args, optimizer)

    loss_counter = LossCounter()
    steps = 0
    min_val_loss = 100

    epoch = 1
    # 学習ループ
    if args.language_model_train: model.module.language_model.train()
    if args.image_model_train: model.module.image_model.train()
    model.module.transformer.train()
    train_loss = torch.tensor(0.0).to(local_rank)
    train_count = torch.tensor(0).to(local_rank)
    pbar = tqdm(total=train_iter_per_epoch, desc='Train', disable=(world_rank != 0))
    for i in range(train_iter_per_epoch):
        src_images = torch.randn(args.batch_size, 3, image_size, image_size, device=local_rank)

        src_texts = torch.randint(1, src_vocab_size, (args.batch_size, args.max_source_length), device=local_rank)
        tgt_texts = torch.randint(1, tgt_vocab_size, (args.batch_size, args.max_target_length), device=local_rank)
        tgt_attention_masks = torch.ones_like(tgt_texts, device=local_rank, dtype=torch.bool)
        src_attention_masks = torch.ones_like(src_texts, device=local_rank, dtype=torch.bool)

        loss, preds = model(src_images, src_texts, None, tgt_texts, tgt_attention_masks)
        loss /= args.accumulation_steps
        scaler.scale(loss).backward()

        train_loss += loss.item() * src_images.shape[0]
        train_count += src_images.shape[0]

        # args.accumulation_steps回の勾配を蓄積してから、optimizer.step()を呼び出す
        if (i + 1) % args.accumulation_steps == 0 or i + 1 == train_iter_per_epoch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
            if world_rank == 0:
                steps += 1
                if use_nvidia_smi:
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(local_rank)
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    wandb.log({"train_step":steps, f"train/MemUsed{local_rank}":info.used // (1024**2)})
            if args.num_steps is not None:
                scheduler.step()

    # 他のノードから集める
    dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
    pbar.close()

    if world_rank == 0:
        train_loss /= train_count
        loss_counter.add("train", train_loss.cpu().numpy().copy())
        logger.info(f'[Epoch ({epoch}/{args.num_epochs}) Train] Loss : {train_loss}, Steps : {steps}, LR : {optimizer.param_groups[0]["lr"]}')

    if args.lr_scheduler != '' and args.num_steps is None:
        scheduler.step()

    # 検証ループ
    if args.language_model_train:
        model.module.language_model.eval()
    if args.image_model_train:
        model.module.image_model.eval()
    model.module.transformer.eval()
    val_loss = torch.tensor(0.0).to(local_rank)
    val_count = torch.tensor(0).to(local_rank)
    pbar = tqdm(total=val_iter_per_epoch, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(world_rank != 0))
    steps=0
    for i in range(val_iter_per_epoch):
        with torch.no_grad():
            src_images = torch.randn(args.batch_size, 3, image_size, image_size, device=local_rank)

            src_texts = torch.randint(1, src_vocab_size, (args.batch_size, args.max_source_length), device=local_rank)
            tgt_texts = torch.randint(1, tgt_vocab_size, (args.batch_size, args.max_target_length), device=local_rank)
            tgt_attention_masks = torch.ones_like(tgt_texts, device=local_rank, dtype=torch.bool)
            src_attention_masks = torch.ones_like(src_texts, device=local_rank, dtype=torch.bool)

            loss, preds = model(src_images, src_texts, src_attention_masks, tgt_texts, tgt_attention_masks)

            val_loss += loss.item() * src_images.shape[0]
            val_count += src_images.shape[0]
            pbar.update(1)
            if world_rank == 0:
                steps += 1
                if use_nvidia_smi:
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(local_rank)
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    wandb.log({"val_step":steps,f"val/MemUsed{local_rank}":info.used // (1024**2)})
    pbar.close()

    # 他のノードから集める
    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_count, op=dist.ReduceOp.SUM)

    if world_rank == 0:
        val_loss /= val_count
        loss_counter.add("val", val_loss.cpu().numpy().copy())
        logger.info(f'[Epoch ({epoch}/{args.num_epochs}) Val] Loss : {val_loss}')

        if val_loss < min_val_loss:
            min_val_loss = val_loss

    if world_rank == 0: 
        if use_wandb:
            wandb.finish()
            
def wandb_init(args):
    name = f"{' '.join(args.datasets)}_{args.batch_size}"
    if args.id is None:
        args.id = 'batch_check_'+wandb.util.generate_id()
    wandb.init(
        id=args.id,
        project='batch_check', 
        name=name,
        config=args,
        resume=True if args.start_epoch > 1 else False
    )
    wandb.define_metric("train_step")
    wandb.define_metric("val_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("val/*", step_metric="val_step")
if __name__ == "__main__":
    train()

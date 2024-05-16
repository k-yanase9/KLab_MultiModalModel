import os
import socket
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
    "classify": ["imagenet", "imagenet21k", "places365", "sun397"]
}
# # Flow
# ONE_GPU_BATCH_DICT = {"caption": 48, "relation":144, "rcap":48, "refexp":72, "det":48, "cat":72, "loc":96, "vqa": 72, "gvqa":48, "classify": 144} #1gpuのバッチサイズ
# TASK_SAMPLE_NUM_DICT = {"caption": 6, "relation":2, "rcap":6, "refexp":4, "det":6, "cat":2, "loc":3, "vqa": 4, "gvqa":1, "classify": 2} #何回タスクごとにバッチを取得するか
# NUM_STEP_PER_EPOCH_MAX = 5120
# # H100
# ONE_GPU_BATCH_DICT = {"caption": 120, "relation":360, "rcap":90, "refexp":180, "det":120, "cat":360, "loc":240, "vqa": 180, "gvqa":125, "classify": 360} #1gpuのバッチサイズ
# TASK_SAMPLE_NUM_DICT = {"caption": 6, "relation":2, "rcap":6, "refexp":4, "det":6, "cat":2, "loc":3, "vqa": 4, "gvqa":1, "classify": 2} #何回タスクごとにバッチを取得するか
# NUM_STEP_PER_EPOCH_MAX = 2400
# # A100 40
ONE_GPU_BATCH_DICT = {"caption": 60, "relation":180, "rcap":45, "refexp":90, "det":60, "cat":180, "loc":120, "vqa": 90, "gvqa":60, "classify": 180} #1gpuのバッチサイズ
TASK_SAMPLE_NUM_DICT = {"caption": 6, "relation":2, "rcap":6, "refexp":4, "det":6, "cat":2, "loc":3, "vqa": 4, "gvqa":1, "classify": 2} #何回タスクごとにバッチを取得するか
NUM_STEP_PER_EPOCH_MAX = 4800

# 4090
# ONE_GPU_BATCH_DICT = {"caption": 20, "relation":60, "rcap":15, "refexp":30, "det":20, "cat":60, "loc":40, "vqa": 30, "gvqa":20, "classify": 60} #1gpuのバッチサイズ
# TASK_SAMPLE_NUM_DICT = {"caption": 6, "relation":2, "rcap":6, "refexp":4, "det":6, "cat":2, "loc":3, "vqa": 4, "gvqa":1, "classify": 2} #何回タスクごとにバッチを取得するか
# NUM_STEP_PER_EPOCH_MAX = 13200


# General
SRC_LEN_DICT = {"caption": 7, "relation":50, "rcap":20, "refexp":184, "det":8, "cat":22, "loc":25, "vqa": 125, "gvqa":256, "classify": 7}
TGT_LEN_DICT = {"caption": 256, "relation":25, "rcap":256, "refexp":120, "det":256, "cat":17, "loc":126, "vqa": 128, "gvqa":103, "classify": 18}

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
        world_rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])
        port_num = 27971
        os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12345')

        hostname = socket.gethostname()
    
        # ホスト名に基づいてNCCLのネットワークインターフェース名を設定
        if hostname == "a100-40gbx4-01":
            nccl_ifname = "eno1"
        elif hostname == "a100-40gbx4-02":
            nccl_ifname = "enp193s0f0"
        else:
            nccl_ifname = "default"

        os.environ["NCCL_SOCKET_IFNAME"] = nccl_ifname
        dist.init_process_group("nccl", rank=world_rank, world_size=args.world_size)
        local_rank = int(os.getenv('SLURM_LOCALID', '0'))
        # torch.cuda.set_device(local_rank)

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
        logger = get_logger(args)

    # create model
    model = MyModel(args).to(local_rank)
    if args.start_epoch > 1:
        model.load(result_name=f'epoch_{args.start_epoch-1}.pth' if args.save_interval is not None else 'best.pth')
        if world_rank == 0:
            logger.info(f'epoch_{args.start_epoch-1}.pth loaded')
    elif args.transformer_model_init == 'pretrain':
        model.load(result_name=f'pretrain.pth', result_path='.')
        if world_rank == 0:
            logger.info('Pretrained model loaded')
    elif 't5' in args.transformer_model_init:
        model.load_from_original(args.transformer_model_init)
        if world_rank == 0:
            logger.info(f'{args.transformer_model_init} loaded')
    model = DDP(model, device_ids=[local_rank])#,find_unused_parameters=True)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True if args.float_type == 'float16' else False)
    optimizer = get_optimizer(model, args)
    if args.start_epoch > 1:
        optimizer.load_state_dict(torch.load(os.path.join(args.result_dir, f'epoch_{args.start_epoch-1}.optimizer' if args.save_interval is not None else 'best.optimizer')))
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
    if args.datasets[0] == 'all':
        train_dataset_name_dict = FULL_DATASET_NAME_DICT
        train_task_sample_num_dict = TASK_SAMPLE_NUM_DICT
        train_one_gpu_batch_dict = ONE_GPU_BATCH_DICT
        train_src_len_dict = SRC_LEN_DICT
        train_tgt_len_dict = TGT_LEN_DICT
        args.datasets = []
        for v in train_dataset_name_dict.values():
            args.datasets.extend(v)
    else:
        train_dataset_name_dict = {}
        train_task_sample_num_dict = {}
        train_one_gpu_batch_dict = {}
        train_src_len_dict = {}
        train_tgt_len_dict = {}
        for task, dataset_names in FULL_DATASET_NAME_DICT.items():
            if task in args.datasets:
                train_dataset_name_dict[task] = FULL_DATASET_NAME_DICT[task]
                train_task_sample_num_dict[task] = TASK_SAMPLE_NUM_DICT[task]
                train_one_gpu_batch_dict[task] = ONE_GPU_BATCH_DICT[task]
                train_src_len_dict[task] = SRC_LEN_DICT[task]
                train_tgt_len_dict[task] = TGT_LEN_DICT[task]
                continue
            for dataset_name in dataset_names:
                if dataset_name in args.datasets:
                    if task in train_dataset_name_dict.keys():
                        train_dataset_name_dict[task].append(dataset_name)
                    else:
                        train_dataset_name_dict[task] = [dataset_name]
                        train_task_sample_num_dict[task] = TASK_SAMPLE_NUM_DICT[task]
                        train_one_gpu_batch_dict[task] = ONE_GPU_BATCH_DICT[task]
                        train_src_len_dict[task] = SRC_LEN_DICT[task]
                        train_tgt_len_dict[task] = TGT_LEN_DICT[task]
    sum_task_sample_num = sum(train_task_sample_num_dict.values())
    num_steps_per_epoch = NUM_STEP_PER_EPOCH_MAX // args.world_size

    src_len_list = []
    tgt_len_list = []
    for task, sample in train_task_sample_num_dict.items():
        src_len_list.extend([train_src_len_dict[task]] * sample)
        tgt_len_list.extend([train_tgt_len_dict[task]] * sample)
    
    if world_rank == 0:
        logger.info(f"target_DataSet:{train_dataset_name_dict}")
        logger.info(f"num_steps_per_epoch:{num_steps_per_epoch}")
    
    train_dataset_dict = get_multi_task_data(args, train_dataset_name_dict, "train", src_tokenizer, tgt_tokenizer, train_src_len_dict, train_tgt_len_dict)
    for task, dataset in train_dataset_dict.items():
        if world_rank == 0:
            logger.info(f"train_dataset ({task}):{len(dataset)}")
    train_loader = MultiTaskDataLoader4(
        train_dataset_dict,
        batch_size_dict=train_one_gpu_batch_dict,
        each_task_sample_num_dict=train_task_sample_num_dict,
        is_ddp=True,
        seed=args.seed,
        loader_drop_last=True,
        sampler_drop_last=True,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    
    if not args.uncalc_val:
        val_dataset = get_data(args, "val", src_tokenizer, tgt_tokenizer, max(src_len_list), max(tgt_len_list))
        if world_rank == 0:
            logger.info(f"val_dataset:{len(val_dataset)}")
        val_loader = get_distributed_dataloader(args, val_dataset, shuffle=False)

    if 'Warmup' in args.lr_scheduler and args.num_steps is None:
        args.num_steps = args.num_epochs * len(train_loader)
    scheduler = get_scheduler(args, optimizer)

    loss_counter = LossCounter()
    min_val_loss = 100
    if args.start_epoch > 1:
        with open(os.path.join(args.result_dir, 'train.log'), 'r') as f:
            for line in f:
                if 'Epoch' in line:
                    if 'Train' in line:
                        loss_counter.add("train", float(line.split(',')[1].split(':')[-1].strip()))
                        steps = int(line.split(',')[2].split(':')[-1].strip())
                    elif 'Val' in line and not args.uncalc_val:
                        loss_counter.add("val", float(line.split(',')[1].split(':')[-1].strip()))
        if not args.uncalc_val: min_val_loss = min(loss_counter.losses['val'])
        if world_rank == 0:
            logger.info(f'[Loaded] steps : {steps}')
            if not args.uncalc_val:
                logger.info(f'Best Val loss : {min_val_loss}')
        if 'Warmup' in args.lr_scheduler:
            for _ in range(steps):
                scheduler.step()
        else:
            for _ in range(args.start_epoch - 1):
                scheduler.step()
    else:
        steps = 0
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        # 学習ループ
        train_loader.set_epoch(epoch)
        if args.language_model_train: model.module.language_model.train()
        if args.image_model_train: model.module.image_model.train()
        model.module.transformer.train()
        train_loss = 0.0
        train_count = 0
        
        for i, samples in enumerate(train_loader):
            if i >= num_steps_per_epoch:
                break
            accumulation_sample_size = torch.tensor(0).to(local_rank)
            loss_per_step = torch.tensor(0.0).to(local_rank)
            pbar = tqdm(total=sum_task_sample_num, desc=f'Train Ep:{epoch} ({i+1}/{num_steps_per_epoch})', disable=(world_rank != 0))
            #累積数分の使用するデータをモデルに通して、勾配累積
            for j, (src_images, _, src_texts, tgt_texts) in enumerate(samples):
                src_images = src_images.to(local_rank, non_blocking=True)
                src_texts = src_texts.to(local_rank, non_blocking=True)
                tgt_texts = tgt_texts.to(local_rank, non_blocking=True)

                loss, preds, sample_size = model(src_images, src_texts, None, tgt_texts, None)
                loss_per_step += loss.item()
                accumulation_sample_size += sample_size.item()
                scaler.scale(loss).backward()

                pbar.update(1)

            # if (i + 1) % args.accumulation_steps == 0 or i + 1 == len(train_loader):
            #sum_loss/num_tokens
            
            #勾配更新の前準備
            dist.all_reduce(accumulation_sample_size, op=dist.ReduceOp.SUM)
            grad_scale = args.world_size / accumulation_sample_size
            multiply_grad(optimizer, grad_scale)
            
            #記録準備
            dist.all_reduce(loss_per_step, op=dist.ReduceOp.SUM)
            loss_per_step /= accumulation_sample_size

            train_loss += loss_per_step.cpu().numpy().copy()
            train_count += 1
            
            #勾配更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            #記録
            pbar.close()
            if world_rank == 0:
                steps += 1
                if use_wandb:
                    wandb.log({"iter": steps, "iter/loss": loss_per_step.item(), "iter/lr": optimizer.param_groups[0]["lr"]})
            if args.num_steps is not None:
                scheduler.step()

        if world_rank == 0:
            train_loss /= train_count
            loss_counter.add("train", train_loss)
            logger.info(f'[Epoch ({epoch}/{args.num_epochs}) Train] Loss : {train_loss}, Steps : {steps}, LR : {optimizer.param_groups[0]["lr"]}')
            if use_wandb:
                wandb.log({"epoch": epoch, "train/loss": train_loss, "train/lr": optimizer.param_groups[0]["lr"]})

        if args.lr_scheduler != '' and args.num_steps is None:
            scheduler.step()

        # 検証ループ
        if not args.uncalc_val:
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
                    print('Best Model and Optimizer saving...')
                    model.module.save()
                    torch.save(optimizer.state_dict(), os.path.join(args.result_dir, 'best.optimizer'))
                    logger.info('Best Model and Optimizer saved')

        if world_rank == 0 and args.save_interval is not None:
            if (epoch) % args.save_interval == 0:
                print(f'Model and Optimizer {epoch} saving...')
                model.module.save(result_name=f'epoch_{epoch}.pth')
                torch.save(optimizer.state_dict(), os.path.join(args.result_dir, f'epoch_{epoch}.optimizer'))
                print(f'Model and Optimizer {epoch} saved')

        if epoch == args.stop_epoch:
            if world_rank == 0: 
                logger.info(f'Train stoped at {epoch} epoch')
            break
            
    if world_rank == 0: 
        if use_wandb:
            wandb.finish()
        loss_counter.plot_loss(args.result_dir, val_show=not args.uncalc_val)

def wandb_init(args):
    # name = f'{args.stage}_{"_".join(args.datasets)}_{args.transformer_model_init}_worldsize{args.world_size}'
    name = f'worldsize{args.world_size}_nomuiti'
    if args.id is None:
        args.id = wandb.util.generate_id()
    wandb.init(
        id=args.id,
        project=f"gfm1.0_nodetest", 
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

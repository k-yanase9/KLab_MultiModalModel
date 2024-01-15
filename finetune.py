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
        logger = get_logger(args)

    # create model
    model = MyModel(args).to(local_rank)
    if args.start_epoch > 1:
        model.load(result_name=f'epoch_{args.start_epoch-1}.pth' if args.save_interval is not None else 'best.pth')
        if world_rank == 0:
            logger.info(f'epoch_{args.start_epoch-1}.pth loaded')
    elif args.transformer_model_init == 'pretrain':
        model.load(result_name=f'task_train.pth', result_path='.')
        if world_rank == 0:
            logger.info('Task trained model loaded')
    elif args.transformer_model_init == 'few':
        model.load(result_name=f'few_train.pth', result_path='.')
        if world_rank == 0:
            logger.info('Few Task trained model loaded')

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
    train_dataset = get_dataset(args.root_dir, args.datasets[0], args.stage, is_tgt_id=args.is_tgt_id, phase="train", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer, src_len=args.max_source_length, tgt_len=args.max_target_length)
    if world_rank == 0:
        logger.info(f"train_dataset:{len(train_dataset)}")
    train_loader = get_distributed_dataloader(args, train_dataset, shuffle=True)
    
    if not args.uncalc_val:
        val_dataset = get_dataset(args.root_dir, args.datasets[0], args.stage, is_tgt_id=args.is_tgt_id, phase="val", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer, src_len=args.max_source_length, tgt_len=args.max_target_length)
        if world_rank == 0:
            logger.info(f"val_dataset:{len(val_dataset)}")
        val_loader = get_distributed_dataloader(args, val_dataset, shuffle=False, drop_last=False)

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
        train_loader.sampler.set_epoch(epoch)
        if args.language_model_train: model.module.language_model.train()
        if args.image_model_train: model.module.image_model.train()
        model.module.transformer.train()
        train_loss = 0.0
        train_count = 0
        
        loop = tqdm(train_loader, desc=f'Train Ep({epoch}/{args.num_epochs})', disable=(world_rank != 0))
        for i, (src_images, _, src_texts, tgt_texts) in enumerate(loop):
            accumulation_sample_size = torch.tensor(0).to(local_rank)
            loss_per_step = torch.tensor(0.0).to(local_rank)
            
            #累積数分の使用するデータをモデルに通して、勾配累積
            src_images = src_images.to(local_rank, non_blocking=True)
            src_texts = src_texts.to(local_rank, non_blocking=True)
            tgt_texts = tgt_texts.to(local_rank, non_blocking=True)

            loss, preds, sample_size = model(src_images, src_texts, None, tgt_texts, None)
            loss_per_step += loss.item()
            accumulation_sample_size += sample_size.item()
            scaler.scale(loss).backward()
            
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
            loop = tqdm(val_loader, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(world_rank != 0))
            for i, (src_images, _, src_texts, tgt_texts) in enumerate(loop):
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
    name = f'{args.stage}_{args.transformer_model_init}_worldsize{args.world_size}'
    if args.id is None:
        args.id = wandb.util.generate_id()
    wandb.init(
        id=args.id,
        project=f"{args.stage}_"+"_".join(args.datasets), 
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

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
    model = DDP(model, device_ids=[local_rank])#,find_unused_parameters=True)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True if args.float_type == 'float16' else False)
    optimizer = get_optimizer(model, args)
    if args.start_epoch > 1:
        optimizer.load_state_dict(torch.load(os.path.join(args.result_dir, f'epoch_{args.start_epoch-1}.optimizer' if args.save_interval is not None else 'best.optimizer')))

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
    train_dataset, val_dataset = get_data(args, src_tokenizer, tgt_tokenizer)
    if world_rank == 0:
        logger.info(f'Train Dataset : {len(train_dataset)}, Val Dataset : {len(val_dataset)}')
    train_loader = get_distributed_dataloader(args, train_dataset, shuffle=True)
    val_loader = get_distributed_dataloader(args, val_dataset, shuffle=False)

    if 'Warmup' in args.lr_scheduler and args.num_steps is None:
        args.num_steps = args.num_epochs * len(train_loader)
    scheduler = get_scheduler(args, optimizer)

    loss_counter = LossCounter()
    if args.start_epoch > 1:
        with open(os.path.join(args.result_dir, 'train.log'), 'r') as f:
            for line in f:
                if 'Epoch' in line:
                    if 'Train' in line:
                        loss_counter.add("train", float(line.split(',')[1].split(':')[-1].strip()))
                        if args.phase == 'classify':
                            steps = int(line.split(',')[3].split(':')[-1].strip())
                        else:
                            steps = int(line.split(',')[2].split(':')[-1].strip())
                    elif 'Val' in line:
                        loss_counter.add("val", float(line.split(',')[1].split(':')[-1].strip()))
        min_val_loss = min(loss_counter.losses['val'])
        if world_rank == 0:
            logger.info(f'[Loaded] steps : {steps}, Best Val loss : {min_val_loss}')
        if 'Warmup' in args.lr_scheduler:
            for _ in range(steps):
                scheduler.step()
        else:
            for _ in range(args.start_epoch - 1):
                scheduler.step()
    else:
        steps = 0
        min_val_loss = 100
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        # 学習ループ
        train_loader.sampler.set_epoch(epoch)
        if args.language_model_train: model.module.language_model.train()
        if args.image_model_train: model.module.image_model.train()
        model.module.transformer.train()
        train_loss = torch.tensor(0.0).to(local_rank)
        if args.phase == 'classify':
            train_acc = torch.tensor(0.0).to(local_rank)
        train_count = torch.tensor(0).to(local_rank)
        pbar = tqdm(total=int(np.ceil(len(train_loader) / args.accumulation_steps)), desc=f'Train (Epoch {epoch}/{args.num_epochs})', disable=(world_rank != 0))
        for i, (src_images, tgt_images, src_texts, tgt_texts) in enumerate(train_loader):
            src_images = src_images.to(local_rank, non_blocking=True)
            # if args.phase == 'pretrain':
            #     tgt_images = tgt_images.to(local_rank)
            #     tgt_texts, _ = model.module.image_to_z(tgt_images)

            if args.phase == 'pretrain':
                src_texts = src_texts.to(local_rank, non_blocking=True)
                tgt_texts = tgt_texts.to(local_rank, non_blocking=True)
                tgt_attention_masks = torch.ones_like(tgt_texts, device=local_rank, dtype=torch.bool)
                tgt_attention_masks[tgt_texts == 0] = 0
            else:
                src_inputs = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt') # ['pt', 'tf', 'np', 'jax']
                src_texts = src_inputs['input_ids'].to(local_rank, non_blocking=True)
                if args.phase == 'classify':
                    tgt_texts = tgt_texts.to(local_rank, non_blocking=True)
                    tgt_attention_masks = None
                else:
                    tgt_inputs = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')
                    tgt_texts = tgt_inputs['input_ids'].to(local_rank, non_blocking=True)
                    tgt_attention_masks = tgt_inputs['attention_mask'].to(local_rank, non_blocking=True) 
            src_attention_masks = torch.ones_like(src_texts, device=local_rank, dtype=torch.bool)
            src_attention_masks[src_texts == 0] = 0

            loss, preds = model(src_images, src_texts, None, tgt_texts, tgt_attention_masks)
            loss /= args.accumulation_steps
            scaler.scale(loss).backward()

            train_loss += loss.item() * src_images.shape[0]
            if args.phase == 'classify':
                train_acc += torch.sum(preds == tgt_texts)
            train_count += src_images.shape[0]

            # args.accumulation_steps回の勾配を蓄積してから、optimizer.step()を呼び出す
            if (i + 1) % args.accumulation_steps == 0 or i + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pbar.update(1)
                if world_rank == 0:
                    steps += 1
                    if use_wandb:
                        wandb.log({"iter": steps, "iter/loss": loss.item(), "iter/lr": optimizer.param_groups[0]["lr"]})
                if args.num_steps is not None:
                    scheduler.step()

        # 他のノードから集める
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        if args.phase == 'classify':
            dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
        pbar.close()

        if world_rank == 0:
            train_loss /= train_count
            loss_counter.add("train", train_loss.cpu().numpy().copy())
            if args.phase == 'classify':
                train_acc /= train_count
                logger.info(
                    f'[Epoch ({epoch}/{args.num_epochs}) Train] Loss : {train_loss}, Acc : {train_acc}, Steps : {steps}, LR : {optimizer.param_groups[0]["lr"]}'
                )
                if use_wandb:
                    wandb.log({"epoch": epoch, "train/loss": train_loss, "train/acc": train_acc, "train/lr": optimizer.param_groups[0]["lr"]})
            else:
                logger.info(f'[Epoch ({epoch}/{args.num_epochs}) Train] Loss : {train_loss}, Steps : {steps}, LR : {optimizer.param_groups[0]["lr"]}')
                if use_wandb:
                    wandb.log({"epoch": epoch, "train/loss": train_loss, "train/lr": optimizer.param_groups[0]["lr"]})

        if args.lr_scheduler != '' and args.num_steps is None:
            scheduler.step()

        # 検証ループ
        if args.language_model_train:
            model.module.language_model.eval()
        if args.image_model_train:
            model.module.image_model.eval()
        model.module.transformer.eval()
        val_loss = torch.tensor(0.0).to(local_rank)
        if args.phase == 'classify':
            val_acc = torch.tensor(0.0).to(local_rank)
        val_count = torch.tensor(0).to(local_rank)
        val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(world_rank != 0))
        for src_images, tgt_images, src_texts, tgt_texts in val_loop:
            with torch.no_grad():
                src_images = src_images.to(local_rank, non_blocking=True)
                # if args.phase == 'pretrain':
                #    tgt_images = tgt_images.to(local_rank)
                #    tgt_texts, _ = model.module.image_to_z(tgt_images)
                if args.phase == 'pretrain':
                    src_texts = src_texts.to(local_rank, non_blocking=True)
                    tgt_texts = tgt_texts.to(local_rank, non_blocking=True)
                    tgt_attention_masks = torch.ones_like(tgt_texts, device=local_rank, dtype=torch.bool)
                    tgt_attention_masks[tgt_texts == 0] = 0
                else:
                    src_inputs = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt') # ['pt', 'tf', 'np', 'jax']
                    src_texts = src_inputs['input_ids'].to(local_rank, non_blocking=True)
                    if args.phase == 'classify':
                        tgt_texts = tgt_texts.to(local_rank, non_blocking=True)
                        tgt_attention_masks = None
                    else:
                        tgt_inputs = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')
                        tgt_texts = tgt_inputs['input_ids'].to(local_rank, non_blocking=True)
                        tgt_attention_masks = tgt_inputs['attention_mask'].to(local_rank, non_blocking=True)
                src_attention_masks = torch.ones_like(src_texts, device=local_rank, dtype=torch.bool)
                src_attention_masks[src_texts == 0] = 0

                loss, preds = model(src_images, src_texts, src_attention_masks, tgt_texts, tgt_attention_masks)

                val_loss += loss.item() * src_images.shape[0]
                if args.phase == 'classify':
                    val_acc += torch.sum(preds == tgt_texts)
                val_count += src_images.shape[0]

        # 他のノードから集める
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        if args.phase == 'classify':
            dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count, op=dist.ReduceOp.SUM)

        if world_rank == 0:
            val_loss /= val_count
            loss_counter.add("val", val_loss.cpu().numpy().copy())
            if args.phase == 'classify':
                val_acc /= val_count
                logger.info(f'[Epoch ({epoch}/{args.num_epochs}) Val] Loss : {val_loss}, Acc : {val_acc}')
                if use_wandb:
                    wandb.log({"epoch": epoch, "val/loss": val_loss, "val/acc": val_acc})
            else:
                logger.info(f'[Epoch ({epoch}/{args.num_epochs}) Val] Loss : {val_loss}')
                if use_wandb:
                    wandb.log({"epoch": epoch, "val/loss": val_loss})

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print('Best Model and Optimizer saving...')
                model.module.save()
                torch.save(optimizer.state_dict(), os.path.join(args.result_dir, 'best.optimizer'))
                logger.info('Best Model and Optimizer saved')

            if args.save_interval is not None:
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
        loss_counter.plot_loss(args.result_dir)
        if use_wandb:
            wandb.finish()


def wandb_init(args):
    if args.phase == 'classify':
        name = f'enc{args.transformer_num_layers}_{args.language_model_name.split("/")[-1]}'
    else:
        name = f'enc{args.transformer_num_layers}_dec{args.transformer_num_decoder_layers}_worldsize{args.world_size}'
    if args.id is None:
        args.id = wandb.util.generate_id()
    wandb.init(
        id=args.id,
        project=f"{args.phase}_"+"_".join(args.datasets), 
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

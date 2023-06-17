import os
import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from tqdm import tqdm

from data import *
from modules import *
from models.model import MyModel

def train():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    args = parse_arguments()
    args.world_size = torch.cuda.device_count() # GPU数
    device_id = rank % args.world_size

    if rank == 0: os.makedirs(args.result_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if rank == 0: logger = get_logger(args)

    # create model
    model = MyModel(args).to(device_id)
    model = DDP(model, device_ids=[device_id])
    
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(args, optimizer)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=512, use_fast=True)

    # データの設定
    train_loader, val_loader = get_data(args, rank=rank)

    if args.num_epochs is None:
        args.num_epochs = int(args.num_steps / len(train_loader)) + 1
    steps = 0
    min_val_loss = 100
    loss_counter = LossCounter()
    for epoch in range(1, args.num_epochs+1):
        # 学習ループ
        if args.image_model_train:
            model.module.image_model.train()
        model.module.transformer.train()
        train_loss = torch.tensor(0.0).to(device_id)
        train_count = torch.tensor(0).to(device_id)
        pbar = tqdm(total=int(np.ceil(len(train_loader)/args.accumulation_steps)), desc=f'Train (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
        for i, (images, src_texts, tgt_texts) in enumerate(train_loader):
            if i % args.accumulation_steps == 0:
                optimizer.zero_grad()
            images = images.to(device_id)
            src_texts = tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(device_id) # ['pt', 'tf', 'np', 'jax']
            tgt_texts = tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(device_id) # ['pt', 'tf', 'np', 'jax']
            loss = model(images, src_texts, tgt_texts)

            loss /= args.accumulation_steps
            loss.backward()

            train_loss += loss.item() * images.shape[0]
            train_count += images.shape[0]

            # args.accumulation_steps回の勾配を蓄積してから、optimizer.step()を呼び出す
            if (i + 1) % args.accumulation_steps == 0 or i + 1 == len(train_loader):
                optimizer.step()
                pbar.update(1)
                if rank == 0: steps += 1

        # 他のノードから集める
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count, op=dist.ReduceOp.SUM)

        if rank == 0:
            train_loss /= train_count
            loss_counter.add("train", train_loss.cpu().numpy().copy())

        if args.lr_scheduler != '':
            scheduler.step()
        pbar.close()
        # 検証ループ
        if args.image_model_train:
            model.module.image_model.eval()
        model.module.transformer.eval()
        val_loss = torch.tensor(0.0).to(device_id)
        val_count = torch.tensor(0).to(device_id)
        val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
        for images, src_texts, tgt_texts in val_loop:
            with torch.no_grad():
                images = images.to(device_id)
                src_texts = tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(device_id) # ['pt', 'tf', 'np', 'jax']
                tgt_texts = tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(device_id) # ['pt', 'tf', 'np', 'jax']
                loss = model(images, src_texts, tgt_texts)
                
                val_loss += loss.item() * images.shape[0]
                val_count += images.shape[0]

        # 他のノードから集める
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count, op=dist.ReduceOp.SUM)

        if rank == 0:
            val_loss /= val_count
            loss_counter.add("val", val_loss.cpu().numpy().copy())
            logger.info(f'[Epoch ({epoch}/{args.num_epochs})] Train loss : {train_loss}, Val loss : {val_loss}')
        
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print('Best Model saving...')
                model.module.save()
                logger.info('Best Model saved')

            if args.save_interval is not None:
                if args.num_steps is None:
                    if (epoch) % args.save_interval == 0:
                        print(f'Model {epoch} saving...')
                        model.module.save(result_name=f'epoch_{epoch}.pth')
                        print(f'Model {epoch} saved')
                else:
                    if steps % args.save_interval == 0:
                        print(f'Model {steps} saving...')
                        model.module.save(result_name=f'step_{steps}.pth')
                        print(f'Model {steps} saved')
            
    if rank == 0: loss_counter.plot_loss(args.result_dir)

if __name__=="__main__":
    train()
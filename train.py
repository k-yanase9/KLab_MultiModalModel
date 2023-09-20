import os
import random
import numpy as np
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
    args.gpu_nums = torch.cuda.device_count() # GPU数
    device_id = rank % args.gpu_nums

    if rank == 0: os.makedirs(args.result_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if rank == 0: logger = get_logger(args)

    # create model
    model = MyModel(args).to(device_id)
    if args.start_epoch > 1:
        model.load(result_name='best.pth')
    model = DDP(model, device_ids=[device_id])#,find_unused_parameters=True)
    
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(args, optimizer)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True)
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(args.loc_vocab_size)] + [f"<img_{i}>" for i in range(args.image_vocab_size)])

    # データの設定
    train_dataset, val_dataset = get_data(args, src_tokenizer, tgt_tokenizer)
    train_loader = get_distributed_dataloader(args, train_dataset, shuffle=True)
    val_loader = get_distributed_dataloader(args, val_dataset, shuffle=False)

    if args.num_epochs is None:
        args.num_epochs = int(args.num_steps / len(train_loader)) + 1

    loss_counter = LossCounter()
    if args.start_epoch > 1:
        with open(os.path.join(args.result_dir, 'train.log'), 'r') as f:
            for line in f:
                if 'Epoch' in line:
                    loss_counter.add("train", float(line.split(',')[1].split(':')[-1].strip()))
                    loss_counter.add("val", float(line.split(',')[2].split(':')[-1].strip()))
                    steps = int(line.split(',')[3].split(':')[-1].strip())
        min_val_loss = min(loss_counter.losses['val'])
        if rank == 0: logger.info(f'[Loaded] steps : {steps}, Best Val loss : {min_val_loss}')
    else:
        steps = 0
        min_val_loss = 100
    for epoch in range(args.start_epoch, args.num_epochs+1):
        # 学習ループ
        image_mask_ratio = 0.0
        if args.image_model_train:
            model.module.image_model.train()
        model.module.transformer.train()
        train_loss = torch.tensor(0.0).to(device_id)
        if not args.pretrain: train_acc = torch.tensor(0.0).to(device_id)
        train_count = torch.tensor(0).to(device_id)
        pbar = tqdm(total=int(np.ceil(len(train_loader)/args.accumulation_steps)), desc=f'Train (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
        for i, (src_images, tgt_images, src_texts, tgt_texts) in enumerate(train_loader):
            if i % args.accumulation_steps == 0:
                optimizer.zero_grad()
            src_images = src_images.to(device_id, non_blocking=True)
            # if args.pretrain:
            #     tgt_images = tgt_images.to(device_id)
            #     tgt_texts, _ = model.module.image_to_z(tgt_images)
            src_texts = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(device_id, non_blocking=True) # ['pt', 'tf', 'np', 'jax']
            if args.pretrain:
                tgt_texts = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(device_id, non_blocking=True) # ['pt', 'tf', 'np', 'jax']
            else:
                tgt_texts = tgt_texts.to(device_id, non_blocking=True)

            loss, preds = model(src_images, src_texts, tgt_texts, image_mask_ratio=image_mask_ratio)
            loss /= args.accumulation_steps
            loss.backward()

            train_loss += loss.item() * src_images.shape[0]
            if not args.pretrain: train_acc += torch.sum(preds == tgt_texts)
            train_count += src_images.shape[0]

            # args.accumulation_steps回の勾配を蓄積してから、optimizer.step()を呼び出す
            if (i + 1) % args.accumulation_steps == 0 or i + 1 == len(train_loader):
                optimizer.step()
                pbar.update(1)
                if rank == 0: steps += 1
                if args.num_steps is not None:
                    scheduler.step()

        # 他のノードから集める
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        if not args.pretrain: dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
        pbar.close()

        if rank == 0:
            train_loss /= train_count
            loss_counter.add("train", train_loss.cpu().numpy().copy())
            if not args.pretrain:
                train_acc /= train_count
                logger.info(f'[Epoch ({epoch}/{args.num_epochs}) Train] Loss : {train_loss}, Acc : {train_acc}, Steps : {steps}')

        if args.lr_scheduler != '' and args.num_steps is None:
            scheduler.step()
        # 検証ループ
        if args.image_model_train:
            model.module.image_model.eval()
        model.module.transformer.eval()
        val_loss = torch.tensor(0.0).to(device_id)
        if not args.pretrain: val_acc = torch.tensor(0.0).to(device_id)
        val_count = torch.tensor(0).to(device_id)
        val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
        for src_images, tgt_images, src_texts, tgt_texts in val_loop:
            with torch.no_grad():
                src_images = src_images.to(device_id, non_blocking=True)
                # if args.pretrain:
                #    tgt_images = tgt_images.to(device_id)
                #    tgt_texts, _ = model.module.image_to_z(tgt_images)
                src_texts = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(device_id, non_blocking=True) # ['pt', 'tf', 'np', 'jax']
                if args.pretrain:
                    tgt_texts = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(device_id, non_blocking=True) # ['pt', 'tf', 'np', 'jax']
                else:
                    tgt_texts = tgt_texts.to(device_id, non_blocking=True)
                
                loss, preds = model(src_images, src_texts, tgt_texts)
                
                val_loss += loss.item() * src_images.shape[0]
                if not args.pretrain: val_acc += torch.sum(preds == tgt_texts)
                val_count += src_images.shape[0]

        # 他のノードから集める
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        if not args.pretrain: dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count, op=dist.ReduceOp.SUM)

        if rank == 0:
            val_loss /= val_count
            loss_counter.add("val", val_loss.cpu().numpy().copy())
            if args.pretrain: 
                logger.info(f'[Epoch ({epoch}/{args.num_epochs})] Train loss : {train_loss}, Val loss : {val_loss}, Steps : {steps}')
            else:
                val_acc /= val_count
                logger.info(f'[Epoch ({epoch}/{args.num_epochs}) Val] Loss : {val_loss}, Acc : {val_acc}')
        
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
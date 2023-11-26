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
ONE_GPUT_BATCH_DICT = {"caption": 54, "relation":216, "rcap":54, "refexp":81, "det":54, "cat":216, "loc":108, "vqa": 81, "gvqa":54, "classify": 162} #1gpuのバッチサイズ
TASK_SAMPLE_NUM_DICT = {"caption": 12, "relation":3, "rcap":12, "refexp":8, "det":12, "cat":3, "loc":6, "vqa": 8, "gvqa":2, "classify": 4} #何回タスクごとにバッチを取得するか

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
    # 引数の設定
    if args.datasets[0] == 'all':
        train_dataset_name_dict = FULL_DATASET_NAME_DICT
        args.datasets = []
        for v in train_dataset_name_dict.values():
            args.datasets.extend(v)
    else:
        train_dataset_name_dict = {}
        for task, dataset_names in FULL_DATASET_NAME_DICT.items():
            for dataset_name in dataset_names:
                if dataset_name in args.datasets:
                    if task in train_dataset_name_dict.keys():
                        train_dataset_name_dict[task].append(dataset_name)
                    else:
                        train_dataset_name_dict[task] = [dataset_name]
    num_steps_per_epoch = 2560 // args.world_size #使用するデータ数の上限、subsetで上限最大値caption80000, classify160000
    
    if world_rank == 0:
        logger.info(f"target_DataSet:{train_dataset_name_dict}")
        logger.info(f"num_steps_per_epoch:{num_steps_per_epoch}")
    
    train_dataset_dict = get_multi_task_data(args, train_dataset_name_dict, "train", src_tokenizer, tgt_tokenizer)
    for task, dataset in train_dataset_dict.items():
        if world_rank == 0:
            logger.info(f"task:{task} train_dataset:{len(dataset)}")
    val_dataset = get_data(args, "val", src_tokenizer, tgt_tokenizer)
    
    train_loader = MultiTaskDataLoader4(
        train_dataset_dict,
        batch_size_dict=ONE_GPUT_BATCH_DICT,
        each_task_sample_num_dict=TASK_SAMPLE_NUM_DICT,
        is_ddp=True,
        seed=args.seed,
        loader_drop_last=True,
        sampler_drop_last=True,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
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
                        if args.stage == 'classify':
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
        train_loader.set_epoch(epoch)
        if args.language_model_train: model.module.language_model.train()
        if args.image_model_train: model.module.image_model.train()
        model.module.transformer.train()
        train_loss = torch.tensor(0.0).to(local_rank)
        train_count = torch.tensor(0).to(local_rank)
        pbar = tqdm(total=num_steps_per_epoch, desc=f'Train (Epoch {epoch}/{args.num_epochs})', disable=(world_rank != 0))
        
        for i, samples in enumerate(train_loader):
            if i >= num_steps_per_epoch:
                break
            accumulation_sample_size = torch.tensor(0).long().to(local_rank)
            loss_per_step = 0
            #累積数分の使用するデータをモデルに通して、勾配累積
            for src_images, _, src_texts, tgt_texts in samples:
                src_images = src_images.to(local_rank, non_blocking=True)

                if args.stage == 'pretrain':
                    src_texts = src_texts.to(local_rank, non_blocking=True)
                    tgt_texts = tgt_texts.to(local_rank, non_blocking=True)
                    tgt_attention_masks = torch.ones_like(tgt_texts, device=local_rank, dtype=torch.bool)
                    tgt_attention_masks[tgt_texts == 0] = 0
                else:
                    src_inputs = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt') # ['pt', 'tf', 'np', 'jax']
                    src_texts = src_inputs['input_ids'].to(local_rank, non_blocking=True)
                    tgt_inputs = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')
                    tgt_texts = tgt_inputs['input_ids'].to(local_rank, non_blocking=True)
                    tgt_attention_masks = tgt_inputs['attention_mask'].to(local_rank, non_blocking=True)
                if world_rank == 0:
                    logger.info(f'b{src_images.shape[0]} src{src_texts.shape[1]} tgt{tgt_texts.shape[1]}')
                src_attention_masks = torch.ones_like(src_texts, device=local_rank, dtype=torch.bool)
                src_attention_masks[src_texts == 0] = 0

                loss, preds,sample_size = model(src_images, src_texts, None, tgt_texts, tgt_attention_masks)
                loss_per_step += loss.item()
                accumulation_sample_size += sample_size
                scaler.scale(loss).backward()

                train_loss += loss.item() #loss.item() * src_images.shape[0]
                if args.stage == 'classify':
                    train_acc += torch.sum(preds == tgt_texts)
                #train_count += src_images.shape[0]

            # if (i + 1) % args.accumulation_steps == 0 or i + 1 == len(train_loader):
            #sum_loss/num_tokens
            
            #勾配更新の前準備
            dist.all_reduce(accumulation_sample_size, op=dist.ReduceOp.SUM)
            grad_scale = args.world_size / accumulation_sample_size
            multiply_grad(optimizer, grad_scale)
            
            #記録準備
            loss_per_step = torch.tensor(loss_per_step).to(local_rank)
            dist.all_reduce(loss_per_step, op=dist.ReduceOp.SUM)
            train_count += accumulation_sample_size
            
            #勾配更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            #記録
            pbar.update(1)
            if world_rank == 0:
                steps += 1
                if use_wandb:
                    wandb.log({"iter": steps, "iter/loss": loss_per_step.item()/accumulation_sample_size.item(), "iter/lr": optimizer.param_groups[0]["lr"]})
            if args.num_steps is not None:
                scheduler.step()

        # 他のノードから集める
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        # dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
        pbar.close()

        if world_rank == 0:
            train_loss /= train_count
            loss_counter.add("train", train_loss.cpu().numpy().copy())
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
        val_count = torch.tensor(0).to(local_rank)
        val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(world_rank != 0))
        for src_images, _, src_texts, tgt_texts in val_loop:
            #勾配更新の前準備
            accumulation_sample_size = torch.tensor(0).long().to(local_rank)
            with torch.no_grad():
                src_images = src_images.to(local_rank, non_blocking=True)
                if args.stage == 'pretrain':
                    src_texts = src_texts.to(local_rank, non_blocking=True)
                    tgt_texts = tgt_texts.to(local_rank, non_blocking=True)
                    tgt_attention_masks = torch.ones_like(tgt_texts, device=local_rank, dtype=torch.bool)
                    tgt_attention_masks[tgt_texts == 0] = 0
                else:
                    src_inputs = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt') # ['pt', 'tf', 'np', 'jax']
                    src_texts = src_inputs['input_ids'].to(local_rank, non_blocking=True)
                    tgt_inputs = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')
                    tgt_texts = tgt_inputs['input_ids'].to(local_rank, non_blocking=True)
                    tgt_attention_masks = tgt_inputs['attention_mask'].to(local_rank, non_blocking=True)
                src_attention_masks = torch.ones_like(src_texts, device=local_rank, dtype=torch.bool)
                src_attention_masks[src_texts == 0] = 0

                loss, preds,sample_size = model(src_images, src_texts, src_attention_masks, tgt_texts, tgt_attention_masks)

                val_loss += loss.item()#loss.item() * src_images.shape[0]
                val_count = sample_size
                #val_count += src_images.shape[0]
                    
            dist.all_reduce(accumulation_sample_size, op=dist.ReduceOp.SUM)

        # 他のノードから集める
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        #dist.all_reduce(val_count, op=dist.ReduceOp.SUM)

        if world_rank == 0:
            val_loss /= val_count
            loss_counter.add("val", val_loss.cpu().numpy().copy())
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
    name = f'enc{args.transformer_num_layers}_dec{args.transformer_num_decoder_layers}_worldsize{args.world_size}'
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

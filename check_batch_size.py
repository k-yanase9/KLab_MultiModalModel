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

# Flow
ONE_GPU_BATCH_DICT = {"caption": 48, "relation":192, "rcap":48, "refexp":72, "det":48, "cat":192, "loc":96, "vqa": 72, "gvqa":48, "classify": 144} #1gpuのバッチサイズ
TASK_SAMPLE_NUM_DICT = {"caption": 12, "relation":3, "rcap":12, "refexp":8, "det":12, "cat":3, "loc":6, "vqa": 8, "gvqa":2, "classify": 4} #何回タスクごとにバッチを取得するか
# H100
ONE_GPU_BATCH_DICT = {"caption": 120, "relation":480, "rcap":120, "refexp":180, "det":120, "cat":480, "loc":240, "vqa": 180, "gvqa":210, "classify": 360} #1gpuのバッチサイズ
TASK_SAMPLE_NUM_DICT = {"caption": 12, "relation":3, "rcap":12, "refexp":8, "det":12, "cat":3, "loc":6, "vqa": 8, "gvqa":1, "classify": 4} #何回タスクごとにバッチを取得するか
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
        logger = get_logger(args)

    # create model
    model = MyModel(args).to(local_rank)
    model = DDP(model, device_ids=[local_rank])#,find_unused_parameters=True)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True if args.float_type == 'float16' else False)
    optimizer = get_optimizer(model, args)

    image_size = 256
    src_vocab_size = 32128
    tgt_vocab_size = 32128 + args.loc_vocab_size + args.additional_vocab_size
    val_iter_per_epoch = 10

    if 'Warmup' in args.lr_scheduler and args.num_steps is None:
        args.num_steps = args.num_epochs * 1
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
    epoch = 1
    # 学習ループ
    if args.language_model_train: model.module.language_model.train()
    if args.image_model_train: model.module.image_model.train()
    model.module.transformer.train()
    train_loss = torch.tensor(0.0).to(local_rank)
    train_count = torch.tensor(0).to(local_rank)
    
    for iter in range(5):
        accumulation_sample_size = torch.tensor(0).long().to(local_rank)
        for task in ONE_GPU_BATCH_DICT.keys():
            loss_per_step = 0
            #累積数分の使用するデータをモデルに通して、勾配累積
            batch_size = ONE_GPU_BATCH_DICT[task]
            max_src_len = SRC_LEN_DICT[task]
            max_tgt_len = TGT_LEN_DICT[task]
            train_loop = tqdm(range(1, TASK_SAMPLE_NUM_DICT[task]+1), desc=f'Train ({task})', disable=(world_rank != 0))
            for i in train_loop:
                src_images = torch.randn(batch_size, 3, image_size, image_size, device=local_rank)

                src_texts = torch.randint(1, src_vocab_size, (batch_size, max_src_len), device=local_rank)
                tgt_texts = torch.randint(1, tgt_vocab_size, (batch_size, max_tgt_len), device=local_rank)
                tgt_attention_masks = torch.ones_like(tgt_texts, device=local_rank, dtype=torch.bool)
                src_attention_masks = torch.ones_like(src_texts, device=local_rank, dtype=torch.bool)

                loss, preds, sample_size = model(src_images, src_texts, src_attention_masks, tgt_texts, tgt_attention_masks)
                loss_per_step += loss.item()
                accumulation_sample_size += sample_size
                scaler.scale(loss).backward()

                train_loss += loss.item() #loss.item() * src_images.shape[0]
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
        if world_rank == 0:
            logger.info('backward')
        
        #記録
        if world_rank == 0:
            steps += 1
            if use_wandb:
                wandb.log({"iter": steps, "iter/loss": loss_per_step.item()/accumulation_sample_size.item(), "iter/lr": optimizer.param_groups[0]["lr"]})
        if args.num_steps is not None:
            scheduler.step()

    # 他のノードから集める
    dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
    # dist.all_reduce(train_count, op=dist.ReduceOp.SUM)

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
    val_loop = tqdm(range(val_iter_per_epoch), desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(world_rank != 0))
    for i in val_loop:
        #勾配更新の前準備
        with torch.no_grad():
            src_images = torch.randn(args.batch_size, 3, image_size, image_size, device=local_rank)
            
            src_texts = torch.randint(1, src_vocab_size, (args.batch_size, args.max_source_length), device=local_rank)
            tgt_texts = torch.randint(1, tgt_vocab_size, (args.batch_size, args.max_target_length), device=local_rank)
            tgt_attention_masks = torch.ones_like(tgt_texts, device=local_rank, dtype=torch.bool)
            src_attention_masks = torch.ones_like(src_texts, device=local_rank, dtype=torch.bool)

            loss, preds, sample_size = model(src_images, src_texts, src_attention_masks, tgt_texts, tgt_attention_masks)

            val_loss += loss.item()
            val_count += sample_size

    # 他のノードから集める
    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_count, op=dist.ReduceOp.SUM)

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

    if world_rank == 0: 
        if use_wandb:
            wandb.finish()


def wandb_init(args):
    name = f'enc{args.transformer_num_layers}_dec{args.transformer_num_decoder_layers}_worldsize{args.world_size}'
    if args.id is None:
        args.id = wandb.util.generate_id()
    wandb.init(
        id=args.id,
        project='batch_check', 
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

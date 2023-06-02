import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoImageProcessor, AutoTokenizer
from tqdm import tqdm

from modules import *
from models.model import MyModel

def train():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()

    args = parse_arguments()
    os.makedirs(args.result_dir, exist_ok=True)

    logger = get_logger(args)
    if rank == 0: logger.info(args)

    # create model
    model = MyModel(args).to(device_id)
    model = DDP(model, device_ids=[device_id])
    
    optimizer = torch.optim.Adam(model.module.transformer.parameters(), lr=args.lr)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    image_processor = AutoImageProcessor.from_pretrained(args.image_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=512)

    # データの設定
    train_loader, val_loader = get_dataloader(args, phase="train", rank=rank), get_dataloader(args, phase="val", rank=rank)

    min_val_loss = 100
    loss_counter = LossCounter(len(train_loader), len(val_loader))
    for epoch in range(1, args.num_epochs+1):
        # 学習ループ
        model.module.transformer.train()
        train_loop = tqdm(train_loader, desc=f'Train (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
        for images, src_texts, tgt_texts in train_loop:
            images = image_processor(images, return_tensors="pt").to(device_id)
            source_encoding = tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt').to(device_id) # ['pt', 'tf', 'np', 'jax']
            target_encoding = tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt').to(device_id) # ['pt', 'tf', 'np', 'jax']
            loss = model(images, source_encoding, target_encoding)

            # パラメータの更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_counter.add_loss('train', loss.item())

        # 検証ループ
        model.module.transformer.eval()
        val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
        for images, src_texts, tgt_texts in val_loop:
            with torch.no_grad():
                images = image_processor(images, return_tensors="pt").to(device_id)
                source_encoding = tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt').to(device_id) # ['pt', 'tf', 'np', 'jax']
                target_encoding = tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt').to(device_id) # ['pt', 'tf', 'np', 'jax']
                loss = model(images, source_encoding, target_encoding)
                loss_counter.add_loss('val', loss.item())

        if rank == 0:
            train_loss, val_loss = loss_counter.count_and_get_loss()
            logger.info(f'[Epoch ({epoch}/{args.num_epochs})] Train loss : {train_loss}, Val loss : {val_loss}')
        
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print('Best Model saving...')
                model.module.save()
                logger.info('Best Model saved')

            if (epoch) % args.save_interval == 0:
                print(f'Model {epoch} saving...')
                model.module.save(result_name=f'epoch_{epoch}.pth')
                print(f'Model {epoch} saved')
            
    if rank == 0: loss_counter.plot_loss(args.result_dir)

if __name__=="__main__":
    train()
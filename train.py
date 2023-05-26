import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoImageProcessor, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.config import parse_arguments
from modules.models import *
from modules.loader import DatasetLoader

def train(rank, world_size):
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    args = parse_arguments()
    if rank == 0:
        os.makedirs(args.result_dir, exist_ok=True)

    # create local model
    model = MyModel(args).to(rank)
    # construct DDP model
    model = DDP(model, device_ids=[rank])
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    image_processor = AutoImageProcessor.from_pretrained(args.image_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=512)

    # データローダーの設定
    train_dataset = DatasetLoader(args, phase="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True, shuffle=True, drop_last=True)
    val_dataset = DatasetLoader(args, phase="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True, shuffle=False, drop_last=False)

    min_val_loss = 100
    val_loss_list = []
    for epoch in range(args.num_epochs):
        # 学習ループ
        train_loop = tqdm(train_loader, desc=f'Train (Epoch {epoch+1}/{args.num_epochs})')
        model.train()
        for images, src_texts, tgt_texts in train_loop:
            images = image_processor(images, return_tensors="pt").to(rank)
            source_encoding = tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt').to(rank) # ['pt', 'tf', 'np', 'jax']
            target_encoding = tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt').to(rank) # ['pt', 'tf', 'np', 'jax']
            loss = model(images, source_encoding, target_encoding)

            # 勾配の計算とパラメータの更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 検証ループ
        val_losses = []
        model.eval()

        val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch+1}/{args.num_epochs})')
        for images, src_texts, tgt_texts in val_loop:
            with torch.no_grad():
                loss = model(images, src_texts, tgt_texts)
                val_losses.append(loss)

        if rank == 0:
            val_loss = torch.mean(torch.tensor(val_losses))
            val_loss_list.append(val_loss)
        
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                model.save(args.result_dir)
                print('Model saved')
            print(f'{epoch+1}: {val_loss}')

    if rank == 0:
        # Plot the loss values.
        plt.plot(val_loss_list)

        # Set the title and axis labels.
        plt.title('Val Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Show the plot.
        plt.savefig(f"{args.result_dir}/val_loss.png")

def main():
    world_size = 4
    mp.spawn(train,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
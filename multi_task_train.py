import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

import data as data_module
from data import *
from models.model import MyModel
from modules import *


def train():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    args = parse_arguments()
    args.gpu_nums = torch.cuda.device_count()  # GPU数
    device_id = rank % args.gpu_nums

    if rank == 0:
        os.makedirs(args.result_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if rank == 0:
        logger = get_logger(args)

    # create model
    model = MyModel(args).to(device_id)
    model = DDP(model, device_ids=[device_id])

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(args, optimizer)
    # if rank == 0:
    #     logger.info(scheduler.step_size)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True)
    tgt_tokenizer = AutoTokenizer.from_pretrained(
        args.language_model_name,
        model_max_length=256,
        use_fast=True,
        extra_ids=0,
        additional_special_tokens=[f"<extra_id_{i}>" for i in range(100)]
        + [f"<loc_{i}>" for i in range(args.loc_vocab_size)]
        + [f"<img_{i}>" for i in range(args.image_vocab_size)],
    )

    # データの設定
    batch_size_list = [5, 4, 21]
    coco_dataset = data_module.caption.COCODatasetLoader("/home/omote/gpu-node/mscoco2017", phase="val")  # データ数4
    oidv7_caption_dataset = data_module.caption.OpenImageDataset_Caption("/home/omote/gpu-node/openimage", phase="val")  # データ数3
    detection_dataset = data_module.detection.oidv7_detection.OpenImageDataset_detection("/home/omote/gpu-node/openimage", phase="val")  # データ数3 add transforms
    vqa_dataset = data_module.vqa.vqa2.Vqa2dataset("/home/omote/gpu-node/vqa2", phase="val")  # データ数3 add transforms
    dataset_list = [
        data_module.MultiChainDataset(dataset_list=[coco_dataset, oidv7_caption_dataset], key_list=[[0, 2, 3], [0, 1, 2]]),
        detection_dataset,
        vqa_dataset,
    ]

    additional_transforms = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    distributed_samplers = [
        torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=True)
        for dataset in dataset_list
    ]
    loaders = [
        torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=distributed_sampler, num_workers=4, pin_memory=True)
        for dataset, batch_size, distributed_sampler in zip(dataset_list, batch_size_list, distributed_samplers)
    ]

    val_sampler = torch.utils.data.distributed.DistributedSampler(coco_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    val_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=10, sampler=val_sampler, num_workers=4, pin_memory=True)

    # if args.num_epochs is None:
    #     args.num_epochs = int(args.num_steps / len(train_loader)) + 1

    min_step = min([len(loader) for loader in loaders])
    max_step = max([len(loader) for loader in loaders])
    if rank == 0:
        logger.info(f"min_step: {min_step}, max_step: {max_step}")
        logger.info(f"batch_size_list: {batch_size_list}")
        logger.info(f"len(dataset_list): {[len(dataset) for dataset in dataset_list]}")
        logger.info(f"len(loaders): {[len(loader) for loader in loaders]}")

    steps = 0
    min_val_loss = 100
    loss_counter = LossCounter()
    for epoch in range(1, args.num_epochs + 1):
        # すべてのiteratorを初期化する
        iterators = [iter(loader) for loader in loaders]

        # 学習ループ
        image_mask_ratio = 0.0
        if args.image_model_train:
            model.module.image_model.train()
        model.module.transformer.train()
        train_loss = torch.tensor(0.0).to(device_id)
        train_count = torch.tensor(0).to(device_id)
        pbar = tqdm(total=int(np.ceil(min_step / args.accumulation_steps)), desc=f'Train (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
        for i in range(min_step):
            # すべてのiteratorからデータを取得し結合する
            image_list = []
            src_text_list = []
            tgt_text_list = []
            caption_data = next(iterators[0])
            detection_data = next(iterators[1])
            vqa_data = next(iterators[2])
            image_list.append(caption_data[0])
            image_list.append(additional_transforms(detection_data[0]))
            image_list.append(additional_transforms(vqa_data[0]))
            src_text_list.extend(caption_data[1])
            src_text_list.extend(detection_data[1])
            src_text_list.extend(vqa_data[1])
            tgt_text_list.extend(caption_data[2])
            tgt_text_list.extend(detection_data[2])
            tgt_text_list.extend(vqa_data[2])
            src_images = torch.cat(image_list, dim=0)
            src_texts = src_text_list
            tgt_texts = tgt_text_list

            if i % args.accumulation_steps == 0:
                optimizer.zero_grad()
            src_images = src_images.to(device_id, non_blocking=True)
            # if args.pretrain:
            #     tgt_images = tgt_images.to(device_id)
            #     tgt_texts, _ = model.module.image_to_z(tgt_images)
            src_texts = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(
                device_id, non_blocking=True
            )  # ['pt', 'tf', 'np', 'jax']
            tgt_texts = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(
                device_id, non_blocking=True
            )  # ['pt', 'tf', 'np', 'jax']

            loss = model(src_images, src_texts, tgt_texts, image_mask_ratio=image_mask_ratio)

            loss /= args.accumulation_steps
            loss.backward()

            train_loss += loss.item() * src_images.shape[0]
            train_count += src_images.shape[0]

            # args.accumulation_steps回の勾配を蓄積してから、optimizer.step()を呼び出す
            if (i + 1) % args.accumulation_steps == 0 or i + 1 == min_step:
                optimizer.step()
                pbar.update(1)
                if rank == 0:
                    steps += 1
                if args.num_epochs is None:
                    scheduler.step()

        # 他のノードから集める
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count, op=dist.ReduceOp.SUM)

        if rank == 0:
            train_loss /= train_count
            loss_counter.add("train", train_loss.cpu().numpy().copy())

        if args.lr_scheduler != '' and args.num_steps is None:
            scheduler.step()
        pbar.close()
        # 検証ループ
        if args.image_model_train:
            model.module.image_model.eval()
        model.module.transformer.eval()
        val_loss = torch.tensor(0.0).to(device_id)
        val_count = torch.tensor(0).to(device_id)
        val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
        for src_images, tgt_images, src_texts, tgt_texts in val_loop:
            with torch.no_grad():
                src_images = src_images.to(device_id, non_blocking=True)
                # if args.pretrain:
                #    tgt_images = tgt_images.to(device_id)
                #    tgt_texts, _ = model.module.image_to_z(tgt_images)
                src_texts = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(
                    device_id, non_blocking=True
                )  # ['pt', 'tf', 'np', 'jax']
                tgt_texts = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(
                    device_id, non_blocking=True
                )  # ['pt', 'tf', 'np', 'jax']

                loss = model(src_images, src_texts, tgt_texts)

                val_loss += loss.item() * src_images.shape[0]
                val_count += src_images.shape[0]

        # 他のノードから集める
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count, op=dist.ReduceOp.SUM)

        if rank == 0:
            val_loss /= val_count
            loss_counter.add("val", val_loss.cpu().numpy().copy())
            logger.info(
                f'[Epoch ({epoch}/{args.num_epochs})] Train loss : {train_loss}, Val loss : {val_loss}, Steps : {steps}, Image Mask Ratio : {image_mask_ratio}'
            )

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

    if rank == 0:
        loss_counter.plot_loss(args.result_dir)


if __name__ == "__main__":
    train()

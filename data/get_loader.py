import os
import torch
from torch.utils.data import random_split, DataLoader, distributed, ConcatDataset
from .caption import *
from .image_classify import *
from .vqa import *
from .pretrain import *

def get_data(args):
    train_datasets, val_datasets = [], []
    for dataset_name in args.datasets:
        if dataset_name in ['redcaps', 'sun397']:
            dataset = get_dataset(args, dataset_name)
            val_rate = 0.1
            val_size = int(len(dataset) * val_rate)
            train_size = len(dataset) - val_size

            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
            )
            
        else:
            train_dataset = get_dataset(args, dataset_name, phase="train")
            val_dataset = get_dataset(args, dataset_name, phase="val")
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    if len(args.datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)

    elif len(args.datasets) == 0:
        raise NotImplementedError
    
    return train_dataset, val_dataset

def get_distributed_dataloader(args, dataset, num_workers=4, shuffle=True):
    sampler = distributed.DistributedSampler(dataset, drop_last=True, shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, sampler=sampler)
    return dataloader

def get_dataloader(args, dataset, num_workers=4, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=shuffle)
    return dataloader

def get_dataset(args, dataset_name, phase="train"):
    data_dir = os.path.join(args.root_dir, dataset_name)
    if args.pretrain: # 事前学習だったら
        if 'redcaps' == dataset_name:
            dataset = RedCapsPretrainDatasetLoader(data_dir)
        elif 'imagenet' == dataset_name:
            dataset = ImageNetPretrainDatasetLoader(data_dir, phase=phase)
        elif 'places365' == dataset_name:
            if phase == "train":
                dataset = Places365PretrainDatasetLoader(root=data_dir, split="train-standard", small=True)
            elif phase == "val":
                dataset = Places365PretrainDatasetLoader(root=data_dir, split="val", small=True)
        elif 'sun397' == dataset_name:
            dataset = SUN397PretrainDatasetLoader(root=data_dir)
        elif 'inaturalist' == dataset_name:
            if phase == "train":
                dataset = INaturalistPretrainDatasetLoader(root=data_dir, version=f'2021_train')
            elif phase == "val":
                dataset = INaturalistPretrainDatasetLoader(root=data_dir, version=f'2021_valid')
        else:
            raise NotImplementedError
    else:
        if 'mscoco' == dataset_name:
            dataset = COCODatasetLoader(data_dir, phase)
        elif 'redcaps' == dataset_name:
            dataset = RedCapsDatasetLoader(data_dir)
        elif 'vcr' == dataset_name:
            dataset = Vcrdataset(data_dir, phase=phase)
        elif 'vqa2' == dataset_name:
            dataset = Vqa2dataset(data_dir, phase=phase)
        elif 'imsitu' == dataset_name:
            dataset = imSituDataset(data_dir, phase=phase)
        elif 'imagenet' == dataset_name:
            dataset = ImageNetDatasetLoader(data_dir, phase=phase)
        else:
            raise NotImplementedError
    return dataset
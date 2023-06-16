import os
import torch
from .redcaps import RedCapsDatasetLoader
from .coco import COCODatasetLoader
from .vcr import Vcrdataset
from .vqa2 import Vqa2dataset
from .imSitu import imSituDataset

def get_data(args, rank):
    if 'redcaps' in args.data_dir.lower():
        dataset = get_dataset(args)
        val_rate = 0.1
        val_size = int(len(dataset) * val_rate)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
    else:
        train_dataset = get_dataset(args, phase="train")
        val_dataset = get_dataset(args, phase="val")

    return get_dataloader(args, train_dataset, rank), get_dataloader(args, val_dataset, rank)
    
def get_dataloader(args, dataset, rank):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=torch.cuda.device_count(), rank=rank, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=os.cpu_count()//torch.cuda.device_count(), pin_memory=True, sampler=sampler)
    return dataloader

def get_dataset(args, phase="train"):
    if 'mscoco' in args.data_dir.lower():
        dataset = COCODatasetLoader(args.data_dir, phase)
    elif 'redcaps' in args.data_dir.lower():
        dataset = RedCapsDatasetLoader(args.data_dir)
    elif 'vcr' in args.data_dir.lower():
        dataset = Vcrdataset(args.data_dir,phase=phase)
    elif 'vqa2' in args.data_dir.lower():
        dataset = Vqa2dataset(args.data_dir,phase=phase)
    elif 'imsitu' in args.data_dir.lower():
        dataset = imSituDataset(args.data_dir,phase=phase)
    else:
        raise NotImplementedError
    return dataset
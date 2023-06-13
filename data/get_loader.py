import os
import torch
from .redcaps import RedCapsDatasetLoader
from .coco import COCODatasetLoader
from .vcr import Vcrdataset
from .vqa2 import Vqa2dataset
from .imSitu import imSituDataset

def get_dataloader(args, phase, rank):
    dataset = get_dataset(args,phase)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=torch.cuda.device_count(), rank=rank, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=os.cpu_count()//4, pin_memory=True, sampler=sampler)
    return dataloader

def get_dataset(args, phase):
    if 'mscoco' in args.data_dir.lower():
        dataset = COCODatasetLoader(args.data_dir, phase)
    elif 'redcaps' in args.data_dir.lower():
        dataset = RedCapsDatasetLoader(args.data_dir, phase)
    elif 'vcr' in args.data_dir.lower():
        dataset = Vcrdataset(args.data_dir,phase=phase)
    elif 'vqa2' in args.data_dir.lower():
        dataset = Vqa2dataset(args.data_dir,phase=phase)
    elif 'imsitu' in args.data_dir.lower():
        dataset = imSituDataset(args.data_dir,phase=phase)
    else:
        raise NotImplementedError
    return dataset
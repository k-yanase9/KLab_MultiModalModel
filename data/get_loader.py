import torch
from .caption import *
from .image_classify import *
from .vqa import *
from .pretrain import *

def get_data(args):
    if 'redcaps' in args.data_dir.lower():
        dataset = get_dataset(args)
        val_rate = 0.1
        val_size = int(len(dataset) * val_rate)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
        )
        
    else:
        train_dataset = get_dataset(args, phase="train")
        val_dataset = get_dataset(args, phase="val")

    return get_dataloader(args, train_dataset), get_dataloader(args, val_dataset)
    
def get_dataloader(args, dataset):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=16, pin_memory=True, sampler=sampler)
    return dataloader

def get_dataset(args, phase="train"):
    if args.pretrain: # 事前学習だったら
        if 'redcaps' in args.data_dir.lower():
            dataset = RedCapsPretrainDatasetLoader(args.data_dir)
        elif 'imagenet' in args.data_dir.lower():
            dataset = ImageNetPretrainDatasetLoader(args.data_dir)
        else:
            raise NotImplementedError
    else:
        if 'mscoco' in args.data_dir.lower():
            dataset = COCODatasetLoader(args.data_dir, phase)
        elif 'redcaps' in args.data_dir.lower():
            dataset = RedCapsDatasetLoader(args.data_dir)
        elif 'vcr' in args.data_dir.lower():
            dataset = Vcrdataset(args.data_dir, phase=phase)
        elif 'vqa2' in args.data_dir.lower():
            dataset = Vqa2dataset(args.data_dir, phase=phase)
        elif 'imsitu' in args.data_dir.lower():
            dataset = imSituDataset(args.data_dir, phase=phase)
        elif 'imagenet' in args.data_dir.lower():
            dataset = ImageNetDatasetLoader(args.data_dir, phase=phase)
        else:
            raise NotImplementedError
    return dataset
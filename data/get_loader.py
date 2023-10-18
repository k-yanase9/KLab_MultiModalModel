import os
from torch.utils.data import DataLoader, distributed, ConcatDataset
from .caption import *
from .image_classify import *
from .vqa import *
from .pretrain import *
from .relationship import *
from .categorization import *
from .detection import *
from .localization import *

def get_data(args, src_tokenizer=None, tgt_tokenizer=None):
    train_datasets, val_datasets = [], []
    for dataset_name in args.datasets:
        train_dataset = get_dataset(args, dataset_name, phase="train", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        val_dataset = get_dataset(args, dataset_name, phase="val", src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    if len(args.datasets) == 0:
        raise ValueError
    
    return train_dataset, val_dataset

def get_distributed_dataloader(args, dataset, num_workers=4, shuffle=True):
    sampler = distributed.DistributedSampler(dataset, drop_last=True, shuffle=shuffle)
    if args.phase == 'pretrain': 
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.datasets[0].collate_fn, num_workers=num_workers, pin_memory=True, sampler=sampler, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, sampler=sampler, drop_last=True)
    return dataloader

def get_dataloader(args, dataset, num_workers=4, shuffle=False):
    if args.phase == 'pretrain': 
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.datasets[0].collate_fn, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=shuffle)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=shuffle)
    return dataloader

def get_dataset(args, dataset_name, phase="train", src_tokenizer=None, tgt_tokenizer=None):
    data_dir = os.path.join(args.root_dir, dataset_name)
    if args.phase == 'pretrain': # 事前学習だったら
        if src_tokenizer is None or tgt_tokenizer is None:
            raise NotImplementedError
        if 'redcaps' == dataset_name:
            dataset = RedCaps_Pretrain(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'imagenet' == dataset_name:
            dataset = ImageNet_Pretrain(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'imagenet21k' == dataset_name:
            dataset = ImageNet21k_Pretrain(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'places365' == dataset_name:
            dataset = Places365_Pretrain(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'sun397' == dataset_name:
            dataset = SUN397_Pretrain(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'inaturalist' == dataset_name:
            dataset = INaturalist_Pretrain(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'cc3m' == dataset_name:
            dataset = CC3M_Pretrain(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        elif 'cc12m' == dataset_name:
            dataset = CC12M_Pretrain(args, data_dir, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
        else:
            raise NotImplementedError
    elif args.phase == 'classify':
        if 'sun397' == dataset_name:
            dataset = SUN397_Classify(data_dir, phase=phase, is_tgt_id=True)
        elif 'openimage' == dataset_name:
            dataset = OpenImageDataset_Categorization(data_dir, phase=phase, is_tgt_id=True)
        else:
            raise NotImplementedError
    else:
        if 'mscoco' == dataset_name:
            dataset = COCO_Caption(data_dir, phase)
        elif 'redcaps' == dataset_name:
            dataset = RedCaps_Caption(data_dir, phase)
        elif 'cc3m' == dataset_name:
            dataset = CC3M_Caption(data_dir, phase)
        elif 'cc12m' == dataset_name:
            dataset = CC12M_Caption(data_dir, phase)
        elif 'vcr' == dataset_name:
            dataset = Vcrdataset(data_dir, phase=phase)
        elif 'vqa2' == dataset_name:
            dataset = Vqa2dataset(data_dir, phase=phase)
        elif 'imsitu' == dataset_name:
            dataset = imSituDataset(data_dir, phase=phase)
        elif 'imagenet' == dataset_name:
            dataset = ImageNet_Classify(data_dir, phase=phase)
        elif 'imagenet21k' == dataset_name:
            dataset = ImageNet21k_Classify(data_dir, phase=phase)
        elif 'sun397' == dataset_name:
            dataset = SUN397_Classify(data_dir, phase=phase)
        else:
            raise NotImplementedError
    return dataset
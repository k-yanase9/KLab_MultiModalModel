import itertools
import os

from torch.utils.data import ConcatDataset, DataLoader, distributed
from multiprocessing import Pool

from .caption import *
from .categorization import *
from .detection import *
from .gvqa import *
from .image_classify import *
from .localization import *
from .pretrain import *
from .relationship import *
from .vqa import *
from .hoi import *
from .ocr import *

def get_dataset_process(item):
    dataset_name, args, phase, src_tokenizer, tgt_tokenizer, src_len, tgt_len = item
    dataset = get_dataset(args.root_dir, dataset_name, args.stage, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer, src_len=src_len, tgt_len=tgt_len)
    return dataset

def get_data(args, phase="train", src_tokenizer=None, tgt_tokenizer=None, src_len=None, tgt_len=None):
    datasets = []
    with Pool(8) as p:
        for dataset in p.imap(get_dataset_process, zip(args.datasets, itertools.repeat(args), itertools.repeat(phase), itertools.repeat(src_tokenizer), itertools.repeat(tgt_tokenizer), itertools.repeat(src_len), itertools.repeat(tgt_len))):
            datasets.append(dataset)

    dataset = ConcatDataset(datasets)

    if len(args.datasets) == 0:
        raise ValueError

    return dataset

def get_distributed_dataloader(args, dataset, num_workers=4, shuffle=True, drop_last=True):
    sampler = distributed.DistributedSampler(dataset, drop_last=True, shuffle=shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=drop_last,
    )
    return dataloader


def get_dataloader(args, dataset, num_workers=4, shuffle=False, drop_last=True):
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=drop_last, 
        shuffle=shuffle)
    
    return dataloader


def get_dataset(root_dir="/data01", dataset_name="cc3m", stage="pretrain", **kwargs):
    data_dir = os.path.join(root_dir, dataset_name)
    if stage == 'pretrain':  # 事前学習だったら
        if 'redcaps' == dataset_name:
            dataset = RedCaps_Pretrain(data_dir, **kwargs)
        elif 'imagenet' == dataset_name:
            dataset = ImageNet_Pretrain(data_dir, **kwargs)
        elif 'imagenet21k' == dataset_name:
            dataset = ImageNet21k_Pretrain(data_dir, **kwargs)
        elif 'places365' == dataset_name:
            dataset = Places365_Pretrain(data_dir, **kwargs)
        elif 'sun397' == dataset_name:
            dataset = SUN397_Pretrain(data_dir, **kwargs)
        elif 'inaturalist' == dataset_name:
            dataset = INaturalist_Pretrain(data_dir, **kwargs)
        elif 'cc3m' == dataset_name:
            dataset = CC3M_Pretrain(data_dir, **kwargs)
        elif 'cc12m' == dataset_name:
            dataset = CC12M_Pretrain(data_dir, **kwargs)
        else:
            raise NotImplementedError
    elif stage == 'classify':
        if 'sun397' == dataset_name:
            dataset = SUN397_Classify(data_dir, is_tgt_id=True, **kwargs)
        elif 'openimage' == dataset_name:
            dataset = OpenImage_Categorization(data_dir, is_tgt_id=True, **kwargs)
        elif 'mscoco' == dataset_name:
            dataset = COCO_Categorization(data_dir, is_tgt_id=True, **kwargs)
        else:
            raise NotImplementedError
    elif stage == 'finetune':
        if 'hico' == dataset_name:
            dataset = HICO_HOI(data_dir, **kwargs)
        elif 'vcoco' == dataset_name:
            dataset = VCOCO_HOI(data_dir, **kwargs)
        elif 'icdar' in dataset_name:
            data_dir = os.path.join(root_dir, 'ICDAR2013')
            if 'loc' in dataset_name:
                dataset = ICDAR_Localization(data_dir, **kwargs)
            elif 'read' in dataset_name:
                dataset = ICDAR_Read(data_dir, **kwargs)
            else:
                raise NotImplementedError
        elif 'deepfashion2' in dataset_name:
            data_dir = os.path.join(root_dir, 'DeepFashion2')
            if 'cat' in dataset_name.lower():
                dataset = DeepFashion2_Categorization(data_dir, **kwargs)
            elif 'loc' in dataset_name.lower():
                dataset = DeepFashion2_Localization(data_dir, **kwargs)
            elif 'det' in dataset_name.lower():
                dataset = DeepFashion2_Detection(data_dir, **kwargs)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        # caption
        if 'mscoco' == dataset_name:
            dataset = COCO_Caption(data_dir, **kwargs)
        elif 'redcaps' == dataset_name:
            dataset = RedCaps_Caption(data_dir, **kwargs)
        elif 'cc3m' == dataset_name:
            dataset = CC3M_Caption(data_dir, **kwargs)
        elif 'cc12m' == dataset_name:
            dataset = CC12M_Caption(data_dir, **kwargs)
        elif 'grit20m' in dataset_name:
            data_dir = os.path.join(root_dir, 'grit20m')
            if 'rcap' in dataset_name.lower():
                dataset = Grit20M_RegionCaption(data_dir, **kwargs)
            elif 'refexp' in dataset_name.lower():
                dataset = Grit20M_RefExp(data_dir, **kwargs)
        # categorization&detection
        elif 'openimage' in dataset_name:
            data_dir = os.path.join(root_dir, 'openimage')
            if 'cat' in dataset_name.lower():
                dataset = OpenImage_Categorization(data_dir, **kwargs)
            elif 'det' in dataset_name.lower():
                dataset = OpenImage_Detection(data_dir, **kwargs)
            elif 'loc' in dataset_name.lower():
                dataset = OpenImage_Localization(data_dir, **kwargs)
            elif 'rel' in dataset_name.lower():
                dataset = OpenImage_Relation(data_dir, **kwargs)
            else:
                raise NotImplementedError
        elif 'objects365' in dataset_name:
            data_dir = os.path.join(root_dir, 'objects365')
            if 'cat' in dataset_name.lower():
                dataset = Objects365_Categorization(data_dir, **kwargs)
            elif 'det' in dataset_name.lower():
                dataset = Objects365_Detection(data_dir, **kwargs)
            elif 'loc' in dataset_name.lower():
                dataset = Objects365_Localization(data_dir, **kwargs)
            else:
                raise NotImplementedError
        elif 'vg' in dataset_name:
            data_dir = os.path.join(root_dir, 'visual_genome')
            if 'cat' in dataset_name.lower():
                dataset = VisualGenome_Categorization(data_dir, **kwargs)
            elif 'det' in dataset_name.lower():
                dataset = VisualGenome_Detection(data_dir, **kwargs)
            elif 'loc' in dataset_name.lower():
                dataset = VisualGenome_Localization(data_dir, **kwargs)
            elif 'rel' in dataset_name.lower():
                dataset = VisualGenome_Relation(data_dir, **kwargs)
            elif 'vqa' in dataset_name.lower():
                dataset = VisualGenome_VQA(data_dir, **kwargs)
            elif 'rcap' in dataset_name.lower():
                dataset = VisualGenome_RegionCaption(data_dir, **kwargs)
            elif 'refexp' in dataset_name.lower():
                dataset = VisualGenome_RefExp(data_dir, **kwargs)
            else:
                raise NotImplementedError
        # vqa & gvqa
        elif 'vcr' == dataset_name:
            dataset = Vcrdataset(data_dir, **kwargs)
        elif 'vqa2' == dataset_name:
            dataset = Vqa2dataset(data_dir, **kwargs)
        elif 'imSitu' == dataset_name:
            dataset = imSituDataset(data_dir, **kwargs)
        elif 'tdiuc' == dataset_name:
            dataset = Tdiucdataset(data_dir, **kwargs)
        elif 'visual7w' in dataset_name:
            data_dir = os.path.join(root_dir, 'visual7w')
            if 'gvqa' in dataset_name.lower():
                dataset = Visual7W_GVQA(data_dir, **kwargs)
            elif 'vqa' in dataset_name.lower():
                dataset = Visual7W_VQA(data_dir, **kwargs)
            else:
                raise NotImplementedError
        # classify
        elif 'imagenet' == dataset_name:
            dataset = ImageNet_Classify(data_dir, **kwargs)
        elif 'imagenet21k' == dataset_name:
            dataset = ImageNet21k_Classify(data_dir, **kwargs)
        elif 'sun397' == dataset_name:
            dataset = SUN397_Classify(data_dir, **kwargs)
        elif 'places365' == dataset_name:
            dataset = Places365_Classify(data_dir, **kwargs)
        else:
            print(dataset_name)
            raise NotImplementedError
    return dataset

import os
import argparse
from transformers import AutoTokenizer
from data import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("target_dataset_name", type=str, choices=['redcaps', 'imagenet', 'imagenet21k', 'places365', 'inaturalist', 'cc3m', 'cc12m', 'sun397', 'mscoco', 'vcr', 'vqa2', 'imsitu', 'visual7w', 'imagenet', 'openimage'])
parser.add_argument("--language_model_name", type=str, default="google/flan-t5-large")
parser.add_argument("--loc_vocab_size", type=int, default=1600)
parser.add_argument("--additional_vocab_size", type=int, default=10000)
parser.add_argument("--max_source_length", type=int, default=512)
parser.add_argument("--max_target_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--phase", type=str, default="train")
parser.add_argument("--root_dir", type=str, default="/data01/")
args = parser.parse_args()
print(args)

target_dataset_name = args.target_dataset_name
print(f"Target dataset: {target_dataset_name}")

src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True)
tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(args.loc_vocab_size)] + [f"<add_{i}>" for i in range(args.additional_vocab_size)])

datasets = {}
datasets['train'] = get_dataset(args, dataset_name=target_dataset_name, phase="train")
datasets['val'] = get_dataset(args, dataset_name=target_dataset_name, phase="val")

def draw_hist(score, title='', save_path='result.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    max_score = max(score)
    plt.hist(score, bins=max_score)
    plt.title(f'{title} (max:{str(max_score)})')
    plt.yscale('log')
    plt.savefig(save_path)
    plt.clf()
    plt.close()

for phase, dataset in datasets.items():
    print(f"\nPhase: {phase}")
    
    src_dataloader = np.array_split(dataset.src_texts, len(dataset.src_texts)//args.batch_size)
    tgt_dataloader = np.array_split(dataset.tgt_texts, len(dataset.tgt_texts)//args.batch_size)
    print('example(src):')
    print(src_dataloader[0][0])
    print(src_tokenizer(src_dataloader[0][0], padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'])
    print('example(tgt):')
    print(tgt_dataloader[0][0])
    print(tgt_tokenizer(tgt_dataloader[0][0], padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'])

    loop = tqdm(zip(src_dataloader, tgt_dataloader), total=len(src_dataloader), desc=f"{target_dataset_name} {phase}")
    src_counts = []
    tgt_counts = []
    for src_texts, tgt_texts in loop:
        src_inputs = src_tokenizer(src_texts.tolist(), padding="longest", max_length=args.max_source_length, return_tensors='pt') # ['pt', 'tf', 'np', 'jax']
        src_texts = src_inputs['input_ids']
        tgt_inputs = tgt_tokenizer(tgt_texts.tolist(), padding="longest", max_length=args.max_target_length, return_tensors='pt') # ['pt', 'tf', 'np', 'jax']
        tgt_texts = tgt_inputs['input_ids']

        src_count = torch.sum(src_texts!=0, dim=1)
        src_counts.extend(src_count.tolist())
        tgt_count = torch.sum(tgt_texts!=0, dim=1)
        tgt_counts.extend(tgt_count.tolist())
    print('max src: ', max(src_counts))
    print('max tgt: ', max(tgt_counts))
    draw_hist(src_counts, f'{target_dataset_name} {phase} src', f'results/token/{target_dataset_name}/{phase}_src.png')
    draw_hist(tgt_counts, f'{target_dataset_name} {phase} tgt', f'results/token/{target_dataset_name}/{phase}_tgt.png')
    print('histogram saved')
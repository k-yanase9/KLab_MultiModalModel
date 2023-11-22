import os
import argparse
from modules import get_logger
from transformers import AutoTokenizer
from data import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("target_dataset_name", type=str, choices=[
    'imagenet', 'imagenet21k', 'inaturalist', 'places365', 'sun397', 
    'redcaps', 'cc3m', 'cc12m', 'mscoco', 'grit20m_rcap', 'grit20m_refexp',
    'vcr', 'vqa2', 'imSitu', 'tdiuc', 'visual7w_vqa', 'visual7w_gvqa', 
    'openimage_cat', 'openimage_det', 'openimage_loc', 'openimage_rel', 
    'objects365_cat', 'objects365_det', 'objects365_loc', 
    'vg_cat', 'vg_det', 'vg_loc', 'vg_rel', 'vg_vqa', 'vg_rcap', 'vg_refexp'
])
parser.add_argument("--language_model_name", type=str, default="google/flan-t5-large")
parser.add_argument("--loc_vocab_size", type=int, default=1600)
parser.add_argument("--additional_vocab_size", type=int, default=10000)
parser.add_argument("--max_source_length", type=int, default=512)
parser.add_argument("--max_target_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--phase", type=str, default="train")
parser.add_argument("--root_dir", type=str, default="/data01/")
args = parser.parse_args()
target_dataset_name = args.target_dataset_name
args.result_dir = f'results/token/{target_dataset_name}'
args.start_epoch = 1
os.makedirs(args.result_dir, exist_ok=True)

logger = get_logger(args, log_name='token_size.log')
logger.info(args)

logger.info(f"Target dataset: {target_dataset_name}")

src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True)
tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(args.loc_vocab_size)] + [f"<add_{i}>" for i in range(args.additional_vocab_size)])

datasets = {}
datasets['train'] = get_dataset(args, dataset_name=target_dataset_name, phase="train")
datasets['val'] = get_dataset(args, dataset_name=target_dataset_name, phase="val")

def draw_hist(score, title='', save_path='result.png'):
    max_score = max(score)
    counts = np.bincount(score)[1:max_score+1]
    plt.bar(range(1, max_score+1), counts)
    # plt.xlim(250,260)
    plt.title(f'{title} (max:{str(max_score)})')
    plt.yscale('log')
    plt.savefig(save_path)
    plt.clf()
    plt.close()

for phase, dataset in datasets.items():
    logger.info(f"\nPhase: {phase} ({len(dataset)} samples)")
    
    if 'gvqa' in target_dataset_name:
        tmp = []
        for que, loc in zip(dataset.src_texts, dataset.locs):
            src = f'{que} choices: {",".join(loc)}'
            tmp.append(src)
        dataset.src_texts = tmp
    src_dataloader = np.array_split(dataset.src_texts, len(dataset.src_texts)//args.batch_size)
    tgt_dataloader = np.array_split(dataset.tgt_texts, len(dataset.tgt_texts)//args.batch_size)
    # logger.info(f'example(src): {src_dataloader[0][0]}')
    # encoded = src_tokenizer(src_dataloader[0][0], padding="longest", max_length=args.max_source_length, return_tensors="pt")
    # logger.info(src_tokenizer.batch_decode(encoded['input_ids'][0]))
    # logger.info(f'example(tgt): {tgt_dataloader[0][0]}')
    # encoded = tgt_tokenizer(tgt_dataloader[0][0], padding="longest", max_length=args.max_target_length, return_tensors="pt")
    # logger.info(tgt_tokenizer.batch_decode(encoded['input_ids'][0]))

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
    for index in np.where(np.array(src_counts) > 256)[0]:
        text_256 = dataset.src_texts[index]
        logger.info(f'src({index}): {text_256}')
        encoded = src_tokenizer(text_256, padding="longest", max_length=args.max_target_length, return_tensors="pt")
        logger.info(src_tokenizer.batch_decode(encoded['input_ids'][0]))
        text_256 = dataset.tgt_texts[index]
        logger.info(f'tgt({index}): {text_256}')
        encoded = tgt_tokenizer(text_256, padding="longest", max_length=args.max_target_length, return_tensors="pt")
        logger.info(tgt_tokenizer.batch_decode(encoded['input_ids'][0]))

    index = np.argmax(src_counts)
    max_text = dataset.src_texts[index]
    logger.info(f'max(src): {max_text}')
    encoded = src_tokenizer(max_text, padding="longest", max_length=args.max_source_length, return_tensors="pt")
    logger.info(src_tokenizer.batch_decode(encoded['input_ids'][0]))
    index = np.argmax(tgt_counts)
    max_text = dataset.tgt_texts[index]
    logger.info(f'max(tgt): {max_text}')
    encoded = tgt_tokenizer(max_text, padding="longest", max_length=args.max_target_length, return_tensors="pt")
    logger.info(tgt_tokenizer.batch_decode(encoded['input_ids'][0]))
    logger.info(f'{max(src_counts)}, {max(tgt_counts)}')
    draw_hist(src_counts, f'{target_dataset_name} {phase} src', f'{args.result_dir}/{phase}_src.png')
    draw_hist(tgt_counts, f'{target_dataset_name} {phase} tgt', f'{args.result_dir}/{phase}_tgt.png')
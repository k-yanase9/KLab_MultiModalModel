import os
import random

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from data import *
from metrics import *
from models.model import MyModel
from modules import *

INPUT_DIR = "/home/jikuya/mAP/inputs/"
GT_DIR = os.path.join(INPUT_DIR, "ground-truth")
RESULT_DIR = os.path.join(INPUT_DIR, "detection-results")

os.makedirs(GT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def train():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel(args).to(device)
    torch.cuda.empty_cache()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=args.max_source_length, use_fast=True)
    tgt_tokenizer = AutoTokenizer.from_pretrained(
        args.language_model_name,
        model_max_length=args.max_target_length,
        use_fast=True,
        extra_ids=0,
        additional_special_tokens=[f"<extra_id_{i}>" for i in range(100)]
        + [f"<loc_{i}>" for i in range(args.loc_vocab_size)]
        + [f"<add_{i}>" for i in range(args.additional_vocab_size)],
    )

    val_dataset = get_dataset(args.root_dir, args.datasets[0], args.stage, is_tgt_id=args.is_tgt_id, phase="val", return_img_path=True)
    if 'deepfashion2' in args.datasets[0]:
        with open(os.path.join(args.root_dir, 'DeepFashion2', 'category.tsv')) as f:
            categories = f.readlines()
        # key value
        categories = {int(c.split('\t')[0]): c.split('\t')[1].rstrip() for c in categories}

    model.load(result_name=f'best.pth')
    torch.cuda.empty_cache()
    # 検証ループ
    if args.image_model_train:
        model.image_model.eval()
    model.transformer.eval()
    val_loader = get_dataloader(args, val_dataset, shuffle=False, drop_last=False)
    random.seed(999)
    torch.manual_seed(999)
    val_loop = tqdm(val_loader, desc=f'Val {" ".join(args.datasets)}')
    scale = 256 / 40
    bias = scale
    for src_images, img_paths, src_texts, tgt_texts in val_loop:
        for tgt_text, img_path in zip(tgt_texts, img_paths):
            gt_str = ''
            if '<loc_' in tgt_text:
                locs = tgt_text.split('> ')
                for loc in locs:
                    label, l1, l2 = loc.split('<loc_')
                    label_id = label.removeprefix('<add_').removesuffix('>')
                    label = categories[int(label_id)]
                    left_top, right_bottom = int(l1.rstrip('>')), int(l2.rstrip('>'))
                    gt_str += f'{label} {left_top%40*scale+bias} {left_top//40*scale+bias} {right_bottom%40*scale+bias} {right_bottom//40*scale+bias}\n'
            with open(os.path.join(GT_DIR, os.path.basename(img_path).replace('.jpg', '.txt')), 'w') as f:
                f.write(gt_str)

        with torch.no_grad():
            src_images = src_images.to(device, non_blocking=True)
            encoded_src_texts = src_tokenizer(src_texts, padding="max_length", max_length=args.max_source_length, return_tensors="pt", return_attention_mask=False)["input_ids"].to(device, non_blocking=True)

            generates, scores = model(src_images, encoded_src_texts, None, return_loss=False, return_score=True, num_beams=4)
            generated_tokens = generates.sequences[:, 1:].cpu()
            scores = scores.cpu().numpy()

            for img_path, generated_token, score in zip(img_paths, generated_tokens, scores):
                result_str = ''
                decoded_token = tgt_tokenizer.batch_decode(generated_token)
                # print(generated_token, decoded_token)
                for i in range(0, len(score), 4):
                    if len(decoded_token[i:i+4]) < 4:
                        continue
                    label, l1, l2, _ = decoded_token[i:i+4]
                    if label == '<pad>':
                        break
                    if '<add_' not in label:
                        continue
                    label_id = label.removeprefix('<add_').removesuffix('>')
                    label = categories[int(label_id)]
                    conf = np.mean(score[i])
                    conf = np.exp(conf)
                    left_top = int(l1.removeprefix('<loc_').removesuffix('>'))
                    right_bottom = int(l2.removeprefix('<loc_').removesuffix('>'))
                    result_str += f'{label} {conf} {left_top%40*scale+bias} {left_top//40*scale+bias} {right_bottom%40*scale+bias} {right_bottom//40*scale+bias}\n'
                with open(os.path.join(RESULT_DIR, os.path.basename(img_path).replace('.jpg', '.txt')), 'w') as f:
                    f.write(result_str)

if __name__ == "__main__":
    train()

import os
import pkgutil
import random
import gc

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data import *
from metrics import *
from models.model import MyModel
from modules import *

FULL_DATASET_NAME_DICT = {
    "caption": ["redcaps", "cc3m", "cc12m"], 
    "relation":["vg_rel", "openimage_rel"], 
    "rcap": ["grit20m_rcap", "vg_rcap"],
    "refexp": ["grit20m_refexp", "vg_refexp"],
    "det": ["vg_det", "openimage_det", "objects365_det"],
    "cat": ["vg_cat", "openimage_cat", "objects365_cat"],
    "loc": ["vg_loc", "openimage_loc", "objects365_loc"],
    "vqa": ["vg_vqa", "vqa2", "tdiuc", "imSitu", "visual7w_vqa"], 
    "gvqa": ["vcr", "visual7w_gvqa"],
    "classify": ["imagenet", "imagenet21k", "places365", "sun397"]}

use_wandb = False
if pkgutil.find_loader("wandb") is not None:
    import wandb

    use_wandb = True


def train():
    args = parse_arguments()
    if use_wandb: wandb_init(args)
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

    val_dataset = get_data(args, "val")

    for epoch in range(args.start_epoch, args.num_epochs + 1, args.save_interval):
        if not os.path.exists(os.path.join(args.result_dir, f'epoch_{epoch}.pth')):
            print(f'epoch_{epoch}.pth is not found. Skip this epoch.')
        model.load(result_name=f'epoch_{epoch}.pth')
        torch.cuda.empty_cache()
        print(f'epoch_{epoch}.pth is loaded.')
        # 検証ループ
        if args.image_model_train:
            model.image_model.eval()
        model.transformer.eval()
        val_loader = get_dataloader(args, val_dataset, shuffle=False, drop_last=False)
        random.seed(999)
        torch.manual_seed(999)
        inputs = []
        preds = []
        gts = []
        val_loop = tqdm(val_loader, desc=f'{" ".join(args.datasets)} (Epoch {epoch}/{args.num_epochs})')
        for src_images, _, src_texts, tgt_texts in val_loop:
            inputs.extend(src_texts)
            gts.extend(tgt_texts)
            with torch.no_grad():
                src_images = src_images.to(device, non_blocking=True)
                encoded_src_texts = src_tokenizer(src_texts, padding="max_length", max_length=args.max_source_length, return_tensors="pt", return_attention_mask=False)["input_ids"].to(device, non_blocking=True)

                outputs = model(src_images, encoded_src_texts, return_loss=False, num_beams=4)
                outputs = outputs[:, 1:]
                pred_texts = tgt_tokenizer.batch_decode(outputs)
                preds.extend([pred_text.replace("<pad>", "").replace("</s>", "") for pred_text in pred_texts])

        result, results = evaluate_score(gts, preds)
        result['epoch'] = int(epoch)

        if args.datasets[0] in FULL_DATASET_NAME_DICT['loc']:
            score, scores = calc_loc_score(preds, gts)
            result['loc_score'] = score
        if use_wandb:
            my_table = wandb.Table(columns=["id", "Inputs", "Ground Truth", "Prediction"]+list(results.keys()))
            for i, contents in enumerate(zip(inputs, gts, preds, *results.values())):
                my_table.add_data(i+1, *contents)
            wandb.log({f"results_ep{epoch}": my_table})
        wandb.log(result)

        del src_images, src_texts, outputs
        gc.collect()
        torch.cuda.empty_cache()

    wandb.finish()

def wandb_init(args):
    name = ' '.join(args.datasets)
    if args.id is None:
        args.id = wandb.util.generate_id()
    wandb.init(
        id=args.id,
        project=f"{args.stage}_score", 
        name=name,
        config=args,
        resume=True if args.start_epoch > 1 else False
    )
    wandb.define_metric("epoch")
    wandb.define_metric("iter")
    wandb.define_metric("iter/*", step_metric="iter")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")


if __name__ == "__main__":
    train()

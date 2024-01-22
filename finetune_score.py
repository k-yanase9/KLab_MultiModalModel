import os
import pkgutil
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data import *
from metrics import *
from models.model import MyModel
from modules import *

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

    val_dataset = get_dataset(args.root_dir, args.datasets[0], args.stage, is_tgt_id=args.is_tgt_id, phase="val", return_img_path=True)
    test_dataset = get_dataset(args.root_dir, args.datasets[0], args.stage, is_tgt_id=args.is_tgt_id, phase="test", return_img_path=True)

    model.load(result_name=f'best.pth')
    torch.cuda.empty_cache()
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
    val_loop = tqdm(val_loader, desc=f'Val {" ".join(args.datasets)}')
    for src_images, img_paths, src_texts, tgt_texts in val_loop:
        inputs.extend(src_texts)
        gts.extend(tgt_texts)
        with torch.no_grad():
            src_images = src_images.to(device, non_blocking=True)
            encoded_src_texts = src_tokenizer(src_texts, padding="max_length", max_length=args.max_source_length, return_tensors="pt", return_attention_mask=False)["input_ids"].to(device, non_blocking=True)

            generates, _ = model(src_images, encoded_src_texts, return_loss=False, num_beams=4)
            outputs = generates.sequences[:, 1:].cpu()
            pred_texts = tgt_tokenizer.batch_decode(outputs)
            preds.extend([pred_text.replace("<pad>", "").replace("</s>", "") for pred_text in pred_texts])

    if '_loc' in args.datasets[0]:
        score, scores = calc_loc_score(preds, gts, split_word='<loc_')
        print("Loc Score:", score)
    if use_wandb:
        if '_loc' in args.datasets[0]:
            my_table = wandb.Table(columns=["id", "Img Path", "Src Text", "Ground Truth", "Prediction", "Loc Score"])
            for i, contents in enumerate(zip(img_paths, inputs, gts, preds, scores)):
                my_table.add_data(i+1, *contents)
            wandb.log({"Val Loc":score})
        else:
            my_table = wandb.Table(columns=["id", "Img Path", "Src Text", "Ground Truth", "Prediction"])
            for i, contents in enumerate(zip(img_paths, inputs, gts, preds)):
                my_table.add_data(i+1, *contents)
        wandb.log({f"Val Results": my_table})

    total = 0
    correct = 0
    for gt, pred in zip(gts, preds):
        if gt == pred:
            correct += 1
        total += 1
    acc = correct / total * 100
        
    print(f"Val Accuracy: {acc}")
    wandb.log({"Val accuracy":acc})

    test_loader = get_dataloader(args, test_dataset, shuffle=False, drop_last=False)
    random.seed(999)
    torch.manual_seed(999)
    image_paths = []
    inputs = []
    preds = []
    gts = []
    test_loop = tqdm(test_loader, desc=f'Test {" ".join(args.datasets)}')
    for src_images, img_paths, src_texts, tgt_texts in test_loop:
        image_paths.extend(img_paths)
        inputs.extend(src_texts)
        gts.extend(tgt_texts)
        with torch.no_grad():
            src_images = src_images.to(device, non_blocking=True)
            encoded_src_texts = src_tokenizer(src_texts, padding="max_length", max_length=args.max_source_length, return_tensors="pt", return_attention_mask=False)["input_ids"].to(device, non_blocking=True)

            outputs = model(src_images, encoded_src_texts, return_loss=False, num_beams=4)
            outputs = outputs[:, 1:]
            pred_texts = tgt_tokenizer.batch_decode(outputs)
            preds.extend([pred_text.replace("<pad>", "").replace("</s>", "") for pred_text in pred_texts])

    if '_loc' in args.datasets[0]:
        score, scores = calc_loc_score(preds, gts, split_word='<loc_')
        print("Loc Score:", score)
    if use_wandb:
        if '_loc' in args.datasets[0]:
            my_table = wandb.Table(columns=["id", "Img Path", "Src Text", "Ground Truth", "Prediction", "Loc Score"])
            for i, contents in enumerate(zip(image_paths, inputs, gts, preds, scores)):
                my_table.add_data(i+1, *contents)
            wandb.log({"Test Loc":score})
        else:
            my_table = wandb.Table(columns=["id", "Img Path", "Src Text", "Ground Truth", "Prediction"])
            for i, contents in enumerate(zip(image_paths, inputs, gts, preds)):
                my_table.add_data(i+1, *contents)
        wandb.log({f"Test Results": my_table})
    
    total = 0
    correct = 0
    for gt, pred in zip(gts, preds):
        if gt == pred:
            correct += 1
        total += 1
    acc = correct / total * 100
        
    print(f"Test Accuracy: {acc}")
    if use_wandb:
        wandb.log({"Test accuracy":acc})

    print("Writing results.tsv")
    write_str = 'img_path\tsrc\tans\tpred\n'
    for img_path, src, ans, pred in zip(image_paths, inputs, gts, preds):
        write_str += f'{img_path}\t{src}\t{ans}\t{pred}\n'
    with open(os.path.join(args.result_dir, 'results.tsv'), 'w') as f:
        f.write(write_str)
    print("Done")

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

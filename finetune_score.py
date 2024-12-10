import os
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data import *
from metrics import *
from models.model import MyModel
from modules import *

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

try:
    import wandb
    use_wandb = True
except ImportError:
    use_wandb = False

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
    print(f"Val Dataset: {len(val_dataset)}")
    if args.stage=="zeroshot":
        model.load(result_name=f'task_train.pth',result_path="./")
    else:
        model.load(result_name=f'best.pth')
    torch.cuda.empty_cache()
    # 検証ループ
    if args.image_model_train:
        model.image_model.eval()
    model.transformer.eval()
    val_loader = get_dataloader(args, val_dataset, shuffle=False, drop_last=False)
    random.seed(999)
    torch.manual_seed(999)
    paths = []
    inputs = []
    preds = []
    gts = []
    val_loop = tqdm(val_loader, desc=f'Val {" ".join(args.datasets)}')
    for src_images, img_paths, src_texts, tgt_texts in val_loop:
        paths.extend(img_paths)
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

    print("Writing results.tsv")
    if '_loc' in args.datasets[0]:
        with open(os.path.join(args.result_dir, 'results.tsv'), 'w') as f:
            write_str = 'img_path\tsrc\tGT\tpred\tLoc_score\n'
            for img_path, src, ans, pred, loc in zip(paths, inputs, gts, preds, scores):
                write_str += f'{img_path}\t{src}\t{ans}\t{pred}\t{loc}\n'
            f.write(write_str)
    else:
        with open(os.path.join(args.result_dir, 'results.tsv'), 'w') as f:
            write_str = 'img_path\tsrc\tGT\tpred\n'
            for img_path, src, ans, pred in zip(paths, inputs, gts, preds):
                write_str += f'{img_path}\t{src}\t{ans}\t{pred}\n'
            f.write(write_str)
    print("Done")
        
    if use_wandb:
        if '_loc' in args.datasets[0]:
            my_table = wandb.Table(columns=["id", "Img Path", "Src Text", "Ground Truth", "Prediction", "Loc Score"])
            for i, contents in enumerate(zip(paths, inputs, gts, preds, scores)):
                my_table.add_data(i+1, *contents)
            wandb.log({"Val Loc":score})
        else:
            my_table = wandb.Table(columns=["id", "Img Path", "Src Text", "Ground Truth", "Prediction"])
            for i, contents in enumerate(zip(paths, inputs, gts, preds)):
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
    if use_wandb:
        wandb.log({"Val accuracy":acc})

    if '_loc' not in args.datasets[0]:
        mlb = MultiLabelBinarizer()
        gts_tmp = []
        preds_tmp = []
        for gt, pred in zip(gts, preds):
            tmp = []
            for i, num in enumerate(gt.split("<add_")):
                if i == 0:
                    continue
                tmp.append(num.rstrip('>'))
            gts_tmp.append(tmp)
            tmp = []
            for i, num in enumerate(pred.split("<add_")):
                if i == 0:
                    continue
                tmp.append(num.rstrip('>'))
            preds_tmp.append(tmp)

        gts_binarized = mlb.fit_transform(gts_tmp)
        preds_binarized = mlb.transform(preds_tmp)
        micro_f1 = f1_score(gts_binarized, preds_binarized, average='micro')*100
        macro_f1 = f1_score(gts_binarized, preds_binarized, average='macro')*100
        with open(os.path.join(args.result_dir, 'score.txt'), 'w') as f:
            f.write(f"Val Micro F1: {micro_f1}\n")
            f.write(f"Val Macro F1: {macro_f1}")
        print(f"Val Micro F1: {micro_f1}")
        print(f"Val Macro F1: {macro_f1}")
        if use_wandb:
            wandb.log({"Val Micro F1":micro_f1})
            wandb.log({"Val Macro F1":macro_f1})

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

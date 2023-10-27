import os
import random
import pkgutil
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from data import *
from modules import *
from models.model import MyModel

use_wandb = False
if pkgutil.find_loader("wandb") is not None:
    import wandb
    use_wandb = True

def test():
    args = parse_arguments()
    args.data_phase = 'train'
    if use_wandb: wandb_init(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel(args).to(device)
    src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=args.max_source_length)
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=args.max_target_length, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(args.loc_vocab_size)] + [f"<add_{i}>" for i in range(args.additional_vocab_size)])
    checkpoints_names = [file_name for file_name in os.listdir(args.result_dir) if file_name.endswith('.pth') and file_name.startswith('epoch_')]
    # logger = get_logger(args, f'test_{args.datasets[0]}.log')
    dataset = get_dataset(args, dataset_name=args.datasets[0], phase=args.data_phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    for checkpoints_name in checkpoints_names:
        epoch = checkpoints_name.split('_')[1].split('.')[0]
        print(f'loading {checkpoints_name}...', end='')
        model.load(result_name=checkpoints_name)
        print(f'done')
        
        dataloader = get_dataloader(args, dataset, num_workers=4, shuffle=False)
        random.seed(999)
        torch.manual_seed(999)
        srcs = []
        preds = []
        gts = []
        str_preds = []
        str_gts = []
        for src_images, tgt_images, src_texts, tgt_texts in tqdm(dataloader, desc=f'{epoch}/{len(checkpoints_names)}'):
            with torch.no_grad():
                src_images = src_images.to(device)
                src_texts = src_texts.to(device)
                src_attention_masks = torch.ones_like(src_texts, device=device)
                src_attention_masks[src_texts==0] = 0

                outputs = model(src_images, src_texts, src_attention_masks, return_loss=False, num_beams=4)
                outputs = outputs[:,1:]
                for src, gt, pred in zip(src_texts.cpu().numpy(), tgt_texts.numpy(), outputs.cpu().numpy()):
                    # 5トークン目以降の<pad>と<eos>を除去
                    if args.data_phase != 'train': srcs.append(src[src!=0])
                    if 0 in gt[:4]:
                        gts.append(gt[:4])
                    else:
                        gts.append(gt[gt!=0])
                    pred2 = []
                    for p in pred:
                        pred2.append(p)
                        if 1 in pred2:
                            pred2 += [0] * max(4-len(pred2),0)
                            break
                    if args.data_phase != 'train': preds.append(pred2)
                    # 文字列に変換
                    str_pred = ' '.join(map(str, pred2))
                    str_gt = ' '.join(map(str, gts[-1]))
                    str_preds.append(str_pred)
                    str_gts.append(str_gt)
        result, results = evaluate_score(str_gts, str_preds)
        if args.data_phase != 'train':
            srcs = src_tokenizer.batch_decode(srcs)
            preds = tgt_tokenizer.batch_decode(preds)
            gts = tgt_tokenizer.batch_decode(gts)
        result['epoch'] = int(epoch)
        if use_wandb:
            if args.data_phase != 'train':
                my_table = wandb.Table(columns=["id", "Input", "Ground Truth", "Prediction", "Ground Truth (num)", "Prediction (num)"]+list(results.keys()))
                for id, contents in enumerate(zip(srcs, gts, preds, str_gts, str_preds, *results.values())):
                    my_table.add_data(id+1, *contents)
                wandb.log({"results_ep"+epoch: my_table})
            wandb.log(result)
    wandb.finish()

def wandb_init(args):
    wandb.init(
        project=f"pretrain_test", 
        name=f"{args.datasets[0]}_{args.data_phase}_b{args.batch_size}",
        config=args,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("Bleu_*", step_metric="epoch")
    wandb.define_metric("CIDEr", step_metric="epoch")
    wandb.define_metric("ROUGE_L", step_metric="epoch")
    wandb.define_metric("results", step_metric="epoch")

if __name__ == '__main__':
    test()
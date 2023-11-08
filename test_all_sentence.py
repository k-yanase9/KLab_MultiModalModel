import os
import random
import pkgutil
import torch
import numpy as np
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
    args.data_phase = 'val'
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
        preds = []
        gts = []
        for src_images, tgt_images, src_texts, tgt_texts in tqdm(dataloader, desc=f'{epoch}/{len(checkpoints_names)}'):
            with torch.no_grad():
                src_images = src_images.to(device)
                src_text_ids = src_texts.to(device)
                src_attention_masks = torch.ones_like(src_text_ids, device=device)
                src_attention_masks[src_text_ids==0] = 0

                outputs = model(src_images, src_text_ids, src_attention_masks, return_loss=False, num_beams=4)
                outputs = outputs[:,1:]
                for src, gt, pred in zip(src_texts.numpy(), tgt_texts.numpy(), outputs.cpu().numpy()):
                    completed_sentence = []
                    src = src[src!=0]
                    for s in src:
                        if s in src_tokenizer.additional_special_tokens_ids:
                            for g in gt[gt!=0]:
                                gt = np.delete(gt, 0)
                                if g == s:
                                    continue
                                if g == s -1 or g == 1:
                                    break
                                completed_sentence.append(g)
                        else:
                            completed_sentence.append(s)
                    gts.append(completed_sentence)

                    completed_sentence = []
                    for s in src:
                        if s in src_tokenizer.additional_special_tokens_ids:
                            for p in pred[pred!=0]:
                                pred = np.delete(pred, 0)
                                if p == s:
                                    continue
                                if p == s -1 or p == 1:
                                    break
                                completed_sentence.append(p)
                        else:
                            completed_sentence.append(s)
                    preds.append(completed_sentence)

        result, results = evaluate_score(gts, preds)
        if args.data_phase != 'train':
            preds = tgt_tokenizer.batch_decode(preds)
            gts = tgt_tokenizer.batch_decode(gts)
        result['epoch'] = int(epoch)
        if use_wandb:
            if args.data_phase != 'train':
                my_table = wandb.Table(columns=["id", "Ground Truth", "Prediction"]+list(results.keys()))
                for id, contents in enumerate(zip(gts, preds, *results.values())):
                    my_table.add_data(id+1, *contents)
                wandb.log({"results_ep"+epoch: my_table})
            wandb.log(result)
    wandb.finish()

def wandb_init(args):
    wandb_id = wandb.util.generate_id()
    wandb.init(
        id=f'test_all_sentence_{wandb_id}',
        project=f"pretrain_test_all_sentence", 
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
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
    if use_wandb: wandb_init(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel(args).to(device)
    src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=args.max_source_length)
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=args.max_target_length, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(args.loc_vocab_size)] + [f"<add_{i}>" for i in range(args.additional_vocab_size)])
    checkpoints_names = [file_name for file_name in os.listdir(args.result_dir) if file_name.endswith('.pth') and file_name.startswith('epoch_')]
    # logger = get_logger(args, f'test_{args.datasets[0]}.log')
    dataset = get_dataset(args, dataset_name=args.datasets[0], phase='val', src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
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
        for src_images, tgt_images, src_texts, tgt_texts in tqdm(dataloader, desc=epoch):
            with torch.no_grad():
                src_images = src_images.to(device)
                src_texts = src_texts.to(device)
                src_attention_masks = torch.ones_like(src_texts).to(device)
                src_attention_masks[src_texts==0] = 0

                outputs = model(src_images, src_texts, src_attention_masks, return_loss=False, num_beams=4)
                for gt, pred in zip(tgt_texts.numpy(), outputs[:,1:].cpu().numpy()):
                    pred = ' '.join(map(str, pred))
                    gt = ' '.join(map(str, gt))
                    preds.append(pred)
                    gts.append(gt)
        result = evaluate_score(gts, preds)
        result['epoch'] = int(epoch)
        print(result)
        if use_wandb:
            wandb.log(result)
    wandb.finish()

def wandb_init(args):
    wandb.init(
        project=f"pretrain_test", 
        name=args.datasets[0],
        config=args,
    )
    wandb.define_metric("epoch")
    wandb.define_metric("Bleu_*", step_metric="epoch")
    wandb.define_metric("CIDEr", step_metric="epoch")

if __name__ == '__main__':
    test()
import os
import random
import torch
import numpy as np
from transformers import AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt

from data import *
from modules import *
from models.model import MyModel

def train():
    args = parse_arguments()

    os.makedirs(args.result_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger = get_logger(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model
    model = MyModel(args).to(device)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(args, optimizer)

    if args.start_epoch != 1:
        model.load(f'epoch_{args.start_epoch-1}.pth')
        optimizer.load_state_dict(torch.load(os.path.join(args.result_dir, f'epoch_{args.start_epoch-1}.opt')))

    src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True)
    # tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(args.loc_vocab_size)] + [f"<img_{i}>" for i in range(args.image_vocab_size)])
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(args.loc_vocab_size)] + [f"<add_{i}>" for i in range(args.additional_vocab_size)])
    
    # データの設定
    train_dataset, val_dataset = get_data(args, src_tokenizer, tgt_tokenizer)
    train_loader = get_dataloader(args, train_dataset, num_workers=1, shuffle=False)

    if args.num_epochs is None:
        logger.info("This code only supports num_epochs mode.")
        exit()

    steps = 0
    loss_counter = LossCounter()

    data_iter = iter(train_loader)
    src_images, tgt_images, src_texts, tgt_texts = data_iter.__next__()

    src_images = src_images.to(device)

    # print("src_images.shape", src_images.shape)
    # print("tgt_images.shape", tgt_images.shape)
    logger.info(f"src_texts: {src_texts}")
    logger.info(f"tgt_texts: {tgt_texts}")
    src_inputs = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt') # ['pt', 'tf', 'np', 'jax']
    src_texts = src_inputs['input_ids'].to(device)
    src_attention_masks = src_inputs['attention_mask'].to(device)
    if args.phase == 'classify':
        tgt_texts = tgt_texts.to(device)
        tgt_attention_masks = None
    else:
        tgt_inputs = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')
        tgt_texts = tgt_inputs['input_ids'].to(device)
        tgt_attention_masks = tgt_inputs['attention_mask'].to(device)

    for epoch in range(args.start_epoch, args.num_epochs+1):
        # 学習ループ
        if args.image_model_train:
            model.image_model.train()
        model.transformer.train()
        
        optimizer.zero_grad()

        loss, preds = model(src_images, src_texts, src_attention_masks, tgt_texts, tgt_attention_masks, image_mask_ratio=0.0)

        loss.backward()

        train_loss = loss.item()

        optimizer.step()
        # pbar.update(1)
        steps += 1

        loss_counter.add("train", train_loss)

        if args.lr_scheduler != '':
            scheduler.step()

        logger.info(f'[Epoch ({epoch}/{args.num_epochs})] Train loss : {train_loss}, Steps : {steps}, LR : {optimizer.param_groups[0]["lr"]}')

        if (epoch) % 50 == 0:
            with torch.no_grad():
                if args.phase != 'classify':
                    preds = tgt_tokenizer.batch_decode(preds)
                logger.info(f"Pred: {preds}")
    
        if args.save_interval is not None:
            if (epoch) % args.save_interval == 0:
                print(f'Model and Optimizer {epoch} saving...')
                model.save(result_name=f'epoch_{epoch}.pth')
                torch.save(optimizer.state_dict(), os.path.join(args.result_dir, f'epoch_{epoch}.opt'))
                logger.info(f'Model and Optimizer {epoch} saved')
            
    loss_counter.plot_loss(args.result_dir)

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

if __name__=="__main__":
    train()
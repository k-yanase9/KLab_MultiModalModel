import torch
# import torch.distributed as dist
# import numpy as np
# from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from tqdm import tqdm

from data import *
from modules import *
from models.model import MyModel

def test():
    args = parse_arguments()
    args.gpu_nums = torch.cuda.device_count() # GPU数

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel(args).to(device)
    model.load(result_name='best.pth')

    logger = get_logger(args, 'test.log')

    tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True)

    # データの設定
    dataset = get_dataset(args, phase='val')
    dataset_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=False, drop_last=False)

    results = {}
    for images, src_texts, gt_texts in tqdm(dataloader):
        with torch.no_grad():
            images = images.to(device)
            src_texts = tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(device)
            outputs = model(images, src_texts, return_loss=False)
            pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for gt_text, pred_text in zip(gt_texts, pred_texts):
                if gt_text not in results:
                    results[gt_text] = 0
                if gt_text == pred_text:
                    results[gt_text] += 1
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (gt_text, correct) in enumerate(sorted_results[:5]):
        logger.info(f'TOP{i}: {gt_text}: {correct}')
    for i, (gt_text, correct) in enumerate(sorted_results[-5:]):
        logger.info(f'WORST{i}: {gt_text}: {correct}')
    corrects = sum(results.values())
    logger.info(f'Accuracy: {corrects / dataset_size * 100}% ({corrects}/{dataset_size})')

if __name__ == '__main__':
    test()
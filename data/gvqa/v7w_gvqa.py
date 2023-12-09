import os
import torch
import random
from PIL import Image
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class Visual7W_GVQA(DatasetLoader):
    """Visual7Wのデータセット
    """    
    def __init__(self,data_dir:str="/data01/visual7w",phase:str="train", **kwargs):
        super().__init__(**kwargs)
        self.locs = []
        tsv_path = os.path.join(data_dir, f'{phase}_gvqa_loc40.tsv')

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        if phase=='val':
            lines = lines[:MAX_VAL_DATA_SIZE]

        for line in lines:
            image_name, question, answer_name, answer_loc, dummy1_name, dummy1_loc, dummy2_name, dummy2_loc, dummy3_name, dummy3_loc = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, 'images', image_name)
            self.images.append(img_path)
            self.locs.append([answer_name+answer_loc,dummy1_name+dummy1_loc, dummy2_name+dummy2_loc, dummy3_name+dummy3_loc])
            self.src_texts.append(question)
            self.tgt_texts.append(answer_name+answer_loc)

    def __getitem__(self, idx):
        image, question, tgt_text, locs = self.images[idx], self.src_texts[idx], self.tgt_texts[idx], self.locs[idx]
        image = Image.open(image).convert('RGB')
        random.shuffle(locs)
        src_text = f'{question} choices: {",".join(locs)}'
        if self.src_tokenizer is not None:
            src_text = self.src_tokenizer(src_text, max_length=self.src_len, padding='max_length', return_tensors='pt')['input_ids'][0]
        if self.tgt_tokenizer is not None:
            tgt_text = self.tgt_tokenizer(tgt_text, max_length=self.tgt_len, padding='max_length', return_tensors='pt')['input_ids'][0]
        src_image = self.src_transforms(image)
        tgt_image = torch.zeros(1)

        return src_image, tgt_image, src_text, tgt_text
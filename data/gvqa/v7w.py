import os
import torch
import random
from PIL import Image
from ..dataset_loader import DatasetLoader

class Visual7W_GVQA(DatasetLoader):
    """Visual7Wのデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/Visual7w",phase:str="train"):
        super().__init__()
        self.locs = []

        with open(os.path.join(data_dir, f'{phase}_gvqa_loc40.tsv')) as f:
            items = f.readlines()
        for item in items[1:]:
            item = item.rstrip()
            image_name, question, answer_name, answer_loc, dummy1_name, dummy1_loc, dummy2_name, dummy2_loc, dummy3_name, dummy3_loc = item.split('\t')
            self.images.append(os.path.join(data_dir, 'images', image_name))
            self.locs.append([answer_loc, dummy1_loc, dummy2_loc, dummy3_loc])
            self.src_texts.append(question)
            self.tgt_texts.append(f'{answer_name} {answer_loc}')

    def __getitem__(self, idx):
        image, question, tgt_text, locs = self.images[idx], self.src_texts[idx], self.tgt_texts[idx], self.locs[idx]
        image = Image.open(image).convert('RGB')
        locs = random.shuffle(locs)
        src_text = f'{question} choices: {",".join(locs)}'
        src_image = self.src_transforms(image)
        tgt_image = torch.zeros(1)

        return src_image, tgt_image, src_text, tgt_text
import json
import os

import torch
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader


class Grit20m_Caption(DatasetLoader):
    def __init__(self,data_dir="/data/dataset/grit20m",phase="train"):
        super().__init__()
        with open(os.path.join(data_dir,f"{phase}_caption.tsv")) as f:
            items = f.read()
        
        items = items.split("\n")
        items = [item.split("\t") for item in items]
        items = items[1:]


        self.tgt_texts = [item[1] for item in items]
        self.src_texts =  ["What does the image describe?"]*len(items)
        self.images = [os.path.join(data_dir,item[0]) for item in items]
    


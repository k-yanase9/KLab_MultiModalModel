import json
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader

class CC3M_Caption(DatasetLoader):
    def __init__(self,data_dir="/dada01/cc3m",phase="train",imagesize=(256,256)):
        super().__init__()
        
        with open(os.path.join(data_dir,f"{phase}.tsv"),"r") as f:
            items = f.read()

        items = items.split("\n")
        items = [item.split("\t") for item in items]

        items = items[1:]

        self.tgt_texts = [item[1] for item in items]
        self.src_texts = ["What does the image describe?"]*len(items)
        self.images = [os.path.join(data_dir,phase,item[0]) for item in items]
        
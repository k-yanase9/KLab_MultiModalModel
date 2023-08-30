import os
import json
from ..dataset_loader import DatasetLoader

class CC12M_Caption(DatasetLoader):
    def __init__(self, data_dir='/data01/cc12m', phase='train'):
        super().__init__()
        
        with open(os.path.join(data_dir,f"text_{phase}.tsv"), 'r') as f:
            text_items = f.read()
        text_items = text_items.split('\n')
        text_items = [item.split('\t') for item in text_items]

        text_items = text_items[1:]

        with open(os.path.join(data_dir,f"img_{phase}.tsv"), 'r') as f:
            img_items = f.read()
        img_items = img_items.split('\n')
        img_items = [item.split('\t') for item in img_items]

        img_items = img_items[1:]
        items = text_items + img_items
        
        self.tgt_texts = [item[1] for item in items]
        self.src_texts = ["What does the image describe?"]*len(items)
        self.images = [os.path.join(data_dir,"images",item[0]) for item in items]


        
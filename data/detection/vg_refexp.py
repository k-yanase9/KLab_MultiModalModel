import os
from copy import deepcopy

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader

class VisualGenome_RefExp(DatasetLoader):
    """VisualGenomeのReferring Expressionデータセット
    """    
    def __init__(self,data_dir:str="/data01/visual_genome/", phase:str="train"):
        super().__init__()        
        
        with open(os.path.join(data_dir, f"{phase}_ref_exp.tsv")) as f:
            items = f.readlines()

        for item in items[1:]:
            item = item.rstrip().split("\t")
            if len(item) < 3:
                continue
            image_id, caption, loc = item
            self.images.append(os.path.join(data_dir,"images_256",f"{image_id}.png"))
            self.src_texts.append(f'Which regions does the text \"{caption}\" describe?')
            self.tgt_texts.append(loc)

    

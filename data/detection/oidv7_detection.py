import os
import pandas as pd
from PIL import Image
from copy import deepcopy
import torch
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader

#存在しない画像を除外するためのリスト
dropimageidlist = ["7f1934f5884fad79"]

class OpenImageDataset_detection(DatasetLoader):
    """openimageのdetectionデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/openimage/",phase:str="train",imagesize:tuple[int,int]=(256,256)):
        super().__init__()        
        if phase=="val":
            phase = "validation"

        with open(os.path.join(data_dir,"bbox",f"{phase}_detection_40.csv")) as f:
            items = f.read()

        items = items.split("\n")
        items = [item.split(",") for item in items]
        items = items[1:-1]
        self.tgt_texts = [item[1] for item in items]
        self.src_texts = ["What objects are in the image?"]*len(items)
        self.images = [os.path.join(data_dir,phase,f"{item[0]}.jpg") for item in items]

        #dropimageidlistに含まれる画像と対応するテキストを除外する
        for drop_id in dropimageidlist:
            drop_path = os.path.join(data_dir,phase,f"{drop_id}.jpg")
            if drop_path in self.images:
                drop_index = self.images.index(drop_path)
                self.tgt_texts.pop(drop_index)
                self.src_texts.pop(drop_index)
                self.images.pop(drop_index)
    
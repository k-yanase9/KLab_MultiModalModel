import json
import os

import torch
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader

#存在しない画像を除外するためのリスト
dropimageidlist =["7f1934f5884fad79","429019e83c1c2c94","4f818c006da84c9e","5b86e93f8654118a","673d74b7d39741c3","6dcd3ce37a17f2be","805baf9650a12710"
                   ,"98ac2996fc46b56d","a46a248a39f2d97c"]

class OpenImageDataset_Caption(DatasetLoader):
    def __init__(self,data_dir="/data/dataset/openimage",phase="train",imagesize=(256,256)):
        super().__init__()
        if phase =="val":
            phase = "validation"
    
        with open(os.path.join(data_dir,"caption",f"{phase}_caption.jsonl"),"r") as f:
            self.items = [json.loads(s) for s in f]

        self.tgt_texts = [item["caption"] for item in self.items]
        self.src_texts = ["What does the image describe?"]*len(self.items)
        self.images = [os.path.join(data_dir,f"{phase}_256",f"{item['image_id']}.jpg") for item in self.items]

        #dropimageidlistに含まれる画像と対応するキャプションを除外する
        for drop_id in dropimageidlist:  
            drop_path = os.path.join(data_dir,f"{phase}_256",f"{drop_id}.jpg")
            if drop_path in self.images:
                drop_index = self.images.index(drop_path)
                self.tgt_texts.pop(drop_index)
                self.src_texts.pop(drop_index)
                self.images.pop(drop_index)
    


if __name__ == "__main__":
    dataset = OpenImageDataset_Caption(phase="val")
    data = dataset[0]
    print(data)

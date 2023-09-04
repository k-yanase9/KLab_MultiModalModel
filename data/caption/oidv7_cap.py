import json
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader

#存在しない画像を除外するためのリスト
dropimageidlist = ["7f1934f5884fad79"]

class OpenImageDataset_Caption(DatasetLoader):
    def __init__(self,data_dir="/data/dataset/openimage",phase="train",imagesize=(256,256)):
        super().__init__()
        if phase =="val":
            phase = "validation"
    
        with open(os.path.join(data_dir,"caption",f"{phase}_caption.jsonl"),"r") as f:
            self.items = [json.loads(s) for s in f]

        self.tgt_texts = [item["caption"] for item in self.items]
        self.src_texts = ["What does the image describe?"]*len(self.items)
        self.images = [os.path.join(data_dir,phase,f"{item['image_id']}.jpg") for item in self.items]

        #dropimageidlistに含まれる画像と対応するキャプションを除外する
        for drop_id in dropimageidlist:  
            drop_path = os.path.join(data_dir,phase,f"{drop_id}.jpg")
            if drop_path in self.images:
                drop_index = self.images.index(drop_path)
                self.tgt_texts.pop(drop_index)
                self.src_texts.pop(drop_index)
                self.images.pop(drop_index)
    

    def __len__(self):
        return len(self.items)



if __name__ =="__main__":
    dataset = OpenImageDataset_Caption(phase="val")
    data = dataset[0]
    print(data)
    
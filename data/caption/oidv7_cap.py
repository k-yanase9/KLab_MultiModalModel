import json
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader

class OpenImageDataset_Caption(DatasetLoader):
    def __init__(self,data_dir="/data/dataset/openimage",phase="train",imagesize=(256,256)):
        super().__init__()
        if phase =="val":
            self.phase = "validation"
        else:
            self.phase = phase
        self.data_dir = data_dir
        self.transform = ToTensor()
        self.imagesize = imagesize

        with open(os.path.join(self.data_dir,"caption",f"{self.phase}_caption.jsonl"),"r") as f:
            self.items = [json.loads(s) for s in f]


        self.tgt_texts = [item["caption"] for item in self.items]
        self.src_texts = ["What does the image describe?"]*len(self.items)
        self.images = [os.path.join(self.data_dir,self.phase,f"{item['image_id']}.jpg") for item in self.items]

    def __len__(self):
        return len(self.items)



if __name__ =="__main__":
    dataset = OpenImageDataset_Caption(phase="val")
    data = dataset[0]
    print(data)
    
import json
from copy import deepcopy
from PIL import Image
import torch
from torchvision.transforms import ToTensor,functional
import os

class imSituDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir="/data/dataset/imSitu",phase="train",imagesize=(256,256)):
        if phase =="val":
            self.phase = "dev"
        else:
            self.phase = phase
        self.data_dir = data_dir
        self.transform = ToTensor()
        self.imagesize = imagesize

        with open(os.path.join(self.data_dir,"imSituVQA.json")) as f:
            items = json.load(f)

        self.items = items[self.phase]
                
    def __getitem__(self,idx):
        src_text = self.items["question"][idx]
        tgt_text = self.items["answer"][idx]
        imgpath = os.path.join(self.data_dir,"of500_images_resized",self.items["image_file"][idx])
        image = Image.open(imgpath).convert("RGB").resize(self.imagesize)
        image = self.transform(image)
        return image,src_text,tgt_text

    def __len__(self):
        return len(self.items["question"])

if __name__ =="__main__":
    dataset = imSituDataset()
    print(dataset[0])
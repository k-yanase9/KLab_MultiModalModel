import json
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor

class OpenImageDataset_Caption(torch.utils.data.Dataset):
    def __init__(self,data_dir="/data/dataset/openimage",phase="train",imagesize=(256,256)):
        if phase =="val":
            self.phase = "validation"
        else:
            self.phase = phase
        self.data_dir = data_dir
        self.transform = ToTensor()
        self.imagesize = imagesize

        with open(os.path.join(self.data_dir,"caption",f"{self.phase}_caption.jsonl"),"r") as f:
            self.items = [json.loads(s) for s in f]
        
    def __getitem__(self,idx):
        src_text = "What does the image describe?"
        tgt_text = self.items[idx]["caption"]
        imgpath = os.path.join(self.data_dir,self.phase,f"{self.items[idx]['image_id']}.jpg")
        image = Image.open(imgpath).convert("RGB").resize(self.imagesize)
        image = self.transform(image)
        return image,src_text,tgt_text

    def __len__(self):
        return len(self.items)



if __name__ =="__main__":
    dataset = OpenImageDataset_Caption(phase="val")
    data = dataset[0]
    print(data)
    
import json
from copy import deepcopy
from PIL import Image
import torch
from torchvision.transforms import ToTensor,functional
import os
from ..dataset_loader import DatasetLoader

class imSituDataset(DatasetLoader):
    def __init__(self,data_dir="/data/dataset/imSitu",phase="train",imagesize=(256,256)):
        super().__init__()
        if phase =="val":
            phase = "dev"
        self.data_dir = data_dir
        self.transform = ToTensor()
        self.imagesize = imagesize

        with open(os.path.join(data_dir,"imSituVQA.json")) as f:
            items = json.load(f)

        items = items[phase]

        self.tgt_texts = [item for item in items["answer"]]
        self.src_texts = [item for item in items["question"]]
        self.images = [os.path.join(data_dir,"of500_images_resized",item) for item in items["image_file"]]
                
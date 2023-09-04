import os
from copy import deepcopy

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from ..dataset_loader import DatasetLoader

# 存在しない画像を除外するためのリスト
dropimageidlist = [
    "7f1934f5884fad79",
    "429019e83c1c2c94",
    "4f818c006da84c9e",
    "5b86e93f8654118a",
    "673d74b7d39741c3",
    "6dcd3ce37a17f2be",
    "805baf9650a12710",
    "98ac2996fc46b56d",
    "a46a248a39f2d97c",
    "9316d4095eab6d10",
    "9ee38bb2e69da0ac",
    "37625d59d0e0782a",
]

class OpenImageDataset_detection(torch.utils.data.Dataset):
    """openimageのdetectionデータセット"""

    def __init__(self, data_dir: str = "/data/dataset/openimage/", phase: str = "train", imagesize: tuple[int, int] = (256, 256)):
        if phase == "val":
            self.phase = "validation"
        else:
            self.phase = phase

        self.data_dir = data_dir
        self.imagesize = imagesize
        self.transform = ToTensor()

        datapath = os.path.join(data_dir, "bbox", f"{self.phase}_detection_40.csv")
        self.df = pd.read_csv(datapath)
        # dropimageidlistに含まれる画像を除外する
        self.df = self.df[self.df["imageID"].isin(dropimageidlist) == False]
        leabelpath = os.path.join(data_dir, "oidv7-class-descriptions.csv")
        self.labels = pd.read_csv(leabelpath)

    # def _return_loc(self,imsize:tuple[int,int],bbox: list[float,float,float,float])->tuple[int,int,int,int]:
    #     """locationを返す

    #     Parameters
    #     ----------
    #     imsize : tuple[int,int]
    #         画像のサイズ
    #     bbox : list[float,float,float,float]
    #         bboxの情報

    #     Returns
    #     -------
    #     x1,x2,y1,y2 : int,int,int,int
    #         locationのタプル
    #     """
    #     x1 = int(imsize[0]*bbox[0])
    #     x2 = int(imsize[0]*bbox[2])
    #     y1 = int(imsize[1]*bbox[1])
    #     y2 = int(imsize[1]*bbox[3])
    #     return x1,x2,y1,y2

    def __getitem__(self, index):
        data = self.df.iloc[index]
        imagepath = os.path.join(self.data_dir, self.phase, data["imageID"] + ".jpg")
        image = self.transform(Image.open(imagepath).convert("RGB").resize(self.imagesize))
        src_text = "What objects are in the image?"
        tgt_text = data["text"]
        return image, src_text, tgt_text

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    dataset = OpenImageDataset_detection(data_dir="/local_data1/openimage", phase="test")
    data = dataset[0]
    print(data)
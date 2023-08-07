import os
import pandas as pd
from PIL import Image
from copy import deepcopy
import torch
from torchvision.transforms import ToTensor
import random


class OpenImageDataset_relation(torch.utils.data.Dataset):
    """openimageのrelathionship用データセット
    """
    def __init__(self,data_dir:str="/data/dataset/openimage/",phase:str="train",imagesize:tuple[int,int]=(256,256)):
        """コンストラクタ

        Parameters
        ----------
        data_dir : str, optional
            データのrootパス, by default "/data/dataset/openimage/"
        phase : str, optional
            フェイズ train val test, by default "train"
        imagesize : tuple[int,int], optional
            transformするイメージのサイズ, by default (256,256)
        """        

        #valでくるとデータパスでエラーがでるので回避
        if phase =="val":
            self.phase = "validation"
        else:
            self.phase = phase
        self.data_dir = data_dir
        self.transform = ToTensor()
        self.imagesize = imagesize
        anopath = os.path.join(self.data_dir,"relation",f"oidv6-{self.phase}-annotations-vrd-fix.csv")
        rel = pd.read_csv(anopath)
        #isは関係が崩壊しているので除外
        self.items = rel[rel['RelationshipLabel']!="is"] 
        labels_path = os.path.join(self.data_dir,"oidv7-class-descriptions.csv")
        self.labels = pd.read_csv(labels_path)
        
    def _get_location(self,item:dict)->list[tuple[int,int,int,int],tuple[int,int,int,int]]:
        """物のロケーションを取得する

        Parameters
        ----------
        item : dict
            画像の情報

        Returns
        -------
        list[tuple[int,int,int,int],tuple[int,int,int,int]]
            Labelname1の座標、Labelname1の座標
        """        
        lb1_x1 = int(self.imagesize[0]*item['XMin1'])
        lb1_x2 = int(self.imagesize[0]*item['XMax1'])
        lb1_y1 = int(self.imagesize[1]*item['YMin1'])
        lb1_y2 = int(self.imagesize[1]*item['YMax1'])
        lb2_x1 = int(self.imagesize[0]*item['XMin2'])
        lb2_x2 = int(self.imagesize[0]*item['XMax2'])
        lb2_y1 = int(self.imagesize[1]*item['YMin2'])
        lb2_y2 = int(self.imagesize[1]*item['YMax2'])
        return (lb1_x1,lb1_y1,lb1_x2,lb1_y2),(lb2_x1,lb2_y1,lb2_x2,lb2_y2)

    def __getitem__(self,idx):
        image,src_text,tgt_text,_,_ = self.get_all(idx)
        return image,src_text,tgt_text

    def get_all(self,idx:int):
        item= deepcopy(self.items.iloc[idx])
        image = Image.open(os.path.join(f'{self.data_dir}',self.phase,f"{item['ImageID']}.jpg")).convert("RGB").resize(self.imagesize)
        image = self.transform(image)
        #MIDからオブジェクトラベルを取得
        Label1 = self.labels[self.labels.LabelName==item["LabelName1"]].iloc[0,1]
        Label2 = self.labels[self.labels.LabelName==item["LabelName2"]].iloc[0,1]
        loc1,loc2 = self._get_location(item)
        lb1 = f"\'loc{loc1[0]} loc{loc1[1]} loc{loc1[2]} loc{loc1[3]} {Label1}"
        lb2 = f"\'loc{loc2[0]} loc{loc2[1]} loc{loc2[2]} loc{loc2[3]} {Label2}"
        src_text = f"What is the relationship between {lb1} and {lb2}"
        tgt_text = f"{Label1} {item['RelationshipLabel']} {Label2}"
        return image,src_text,tgt_text,loc1,loc2

    def __len__(self):
        return len(self.items)



if __name__ =="__main__":
    from PIL import ImageDraw
    import torchvision
    index = 2
    dataset = OpenImageDataset_relation(is_mask=True)
    print(dataset.get_all(1))
    # data = dataset.get_all(idx=index)
    # p = torchvision.transforms.functional.to_pil_image(data[0])
    # draw = ImageDraw.Draw(p)
    # draw.rectangle(data[3])
    # p.show()
    # print(f"q:{data[1]}")
    # print(f"a:{data[2]}")
    
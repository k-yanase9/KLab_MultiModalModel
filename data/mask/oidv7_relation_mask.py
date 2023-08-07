import os
import pandas as pd
from PIL import Image
from copy import deepcopy
import torch
from torchvision.transforms import ToTensor
import random
from .utils import make_mask_textpair
# import utils


class OpenImageDataset_relation_mask(torch.utils.data.Dataset):
    """openimageのrelathionshipをマスクしたデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/openimage/",phase:str="train",imagesize:tuple[int,int]=(256,256)):
        """コンストラクタ

        Parameters
        ----------
        data_dir : str, optional
            データセットのrootパス, by default "/data/dataset/openimage/"
        phase : str, optional
            フェイズ train,val,testから選択 , by default "train"
        imagesize : tuple, optional
            transformする画像のサイズ, by default (256,256)
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
        #relationとしてはisは関係が崩壊しているが、maskでは問題ないのでそのままにする
        self.items = rel
        labels_path = os.path.join(self.data_dir,"oidv7-class-descriptions.csv")
        self.labels = pd.read_csv(labels_path)

    def __getitem__(self,idx):
        item= deepcopy(self.items.iloc[idx])
        image = Image.open(os.path.join(f'{self.data_dir}',self.phase,f"{item['ImageID']}.jpg")).convert("RGB").resize(self.imagesize)
        image = self.transform(image)
        Label1 = self.labels[self.labels.LabelName==item["LabelName1"]].iloc[0,1]
        Label2 = self.labels[self.labels.LabelName==item["LabelName2"]].iloc[0,1]
        #フルのリレーションからmaskされたソースとターゲットを作成
        src_text,tgt_text = make_mask_textpair(f"{Label1} {item['RelationshipLabel']} {Label2}.")
        # src_text,tgt_text = utils.make_mask_textpair(f"{Label1} {item['RelationshipLabel']} {Label2}.")
        return image,src_text,tgt_text

    def __len__(self):
        return len(self.items)



if __name__ =="__main__":
    #ローカルで動かす場合は、.utilsをコメントアウトして、import utilsを有効にする
    from PIL import ImageDraw
    import torchvision
    dataset = OpenImageDataset_relation_mask("/local_data1/openimage",phase="train")
    for i in range(210,len(dataset)):
        print(f"{i}:{dataset[i][1]}")


    
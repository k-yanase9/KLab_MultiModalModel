import os
from ..dataset_loader import DatasetLoader, HOI_SRC_TEXT
import random
import copy
import glob
import numpy as np
from PIL import Image
import torch


class stanford_cars(DatasetLoader):
    """stanford cars dataset
    """
    def __init__(self, data_dir:str="/data01/Stanford_Cars", phase:str="train", **kwargs):
        super().__init__(**kwargs)
        
        with open(os.path.join(data_dir, f'anno_{phase}.csv')) as f:
            self.data = f.read().splitlines()
        self.data = [x.split(",") for x in self.data]
        with open(os.path.join(data_dir, f'names.csv')) as f:
            self.labels = f.read().splitlines()

        self.data_dir = data_dir
        self.phase = phase

    def return_loc(self,bbox:list[float,float,float,float],locsize=40):
        """locationを返す

        Parameters
        ----------
        bbox : list[float,float,float,float]
            bboxの情報

        Returns
        -------
        tuple[int,int]
            lu(左上)rd(右下)のタプル

        Notes
        -------
        
        """         

        ##bboxの右上と左下の座標を返す 特にy座標は注意！！先に0-39で確定させてからlocsize倍する
        lu = int(bbox[0]*(locsize-1))+int(bbox[1]*(locsize-1))*locsize
        rd = int(bbox[2]*(locsize-1))+int(bbox[3]*(locsize-1))*locsize
        return lu,rd
    
    def get_img_path(self,dir_name):
        """
        指定されたディレクト以下の画像パスから一つランダムに返す"""
        img_paths = glob.glob(dir_name + '/*.jpg')
        return random.choice(img_paths)     

    def train_data_proc(self,d):
        label = self.labels[int(d[5]) - 1]
        
        #対象ラベルを省いて残りから3つ選ぶ
        sub_label_list = copy.deepcopy(self.labels)
        sub_label_list.remove(label)
        sub_labels = random.sample(sub_label_list,3)

        #画像読み込み
        main_img_path = os.path.join(self.data_dir, "car_data/car_data", self.phase,label.replace("/", "-"),d[0])
        sub_img_paths = [self.get_img_path(os.path.join(self.data_dir, "car_data/car_data", self.phase,sub_label.replace("/","-"))) for sub_label in sub_labels]
        main_img = Image.open(main_img_path).convert('RGB')
        main_w, main_h = main_img.size
        main_img = np.array(main_img.resize((128,128)))
        imgs = [main_img]
        for sub_img_path in sub_img_paths:
            _img = np.array(Image.open(sub_img_path).convert('RGB').resize((128,128)))
            imgs.append(_img)

        #画像合成
        img = np.zeros((256,256,3))
        id_list = [0,1,2,3] #0がmain   
        random.shuffle(id_list)
        img[:128,:128,:] = imgs[id_list[0]]
        img[:128,128:,:] = imgs[id_list[1]]
        img[128:,:128,:] = imgs[id_list[2]]
        img[128:,128:,:] = imgs[id_list[3]]
        img = Image.fromarray(np.uint8(img))
        #loc作成
        w,h = img.size
        offset_dict = [[0,0],[0.55,0],[0,0.55],[0.55,0.55]] #40で区切る関係で中心が0.5にならない。0.55は暫定値
        x1,y1,x2,y2 = d[1:5]

        #存在位置によるオフセット処理
        x1 = float(x1)/float(2.0*main_w) +offset_dict[id_list.index(0)][0]  
        y1 = float(y1)/float(2.0*main_h) +offset_dict[id_list.index(0)][1]
        x2 = float(x2)/float(2.0*main_w) +offset_dict[id_list.index(0)][0]
        y2 = float(y2)/float(2.0*main_h) +offset_dict[id_list.index(0)][1]
        
        if x2>1.0:
            x2 = 1.0
        if y2>1.0:
            y2 = 1.0
        locs = self.return_loc([x1,y1,x2,y2])
        label_id = int(d[5]) - 1
        return locs,img,label_id
        
    def __getitem__(self, idx):
        d = self.data[idx] #d[0]:画像名 d[1:5]:bbox d[5]:label
        locs,img,label_id = self.train_data_proc(d)
        
        #正規の処理
        src_image = self.src_transforms(img)
        tgt_image = torch.zeros(1)
        src_text = f"What are the make, model and year of the cars in the area <loc_{locs[0]}><loc_{locs[1]}>?"
        tgt_text = f"<add_{label_id}>"
        if self.src_tokenizer is not None:
            src_text = self.src_tokenizer(src_text, max_length=self.src_len, padding='max_length', return_tensors='pt')['input_ids'][0]
        if self.tgt_tokenizer is not None:
            tgt_text = self.tgt_tokenizer(tgt_text, max_length=self.tgt_len, padding='max_length', return_tensors='pt')['input_ids'][0]
        return src_image, tgt_image, src_text, tgt_text
    
    def __len__(self):
        return len(self.data)
    
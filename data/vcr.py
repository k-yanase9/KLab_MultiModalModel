import os
import json
import random
from copy import deepcopy
from PIL import Image
import torch
from torchvision.transforms import ToTensor,functional


_GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Skyler', 'Frankie', 'Pat', 'Quinn', 'Harper', 'Michael', 'Logan', 'Noah', 'James', 'Evelyn',
                        'Avery', 'Madison', 'Riley', 'Jayden', 'Ainsley', 'Arden', 'Dakota', 'Finley', 'Hayden', 'Lennox',
                        'Lindsey','Robin','Parker' 'River', 'Rowan', 'Avey', 'Jordan', 'Cameron', 'Angel', 'Carter','Ryan',
                        'Dylan', 'Ezra', 'Emery', 'Hunter', 'Kai', 'Nova', 'Ollie']

class Vcrdataset(torch.utils.data.Dataset):
    def __init__(self,data_dir="/data/dataset/vcr",mode="vqa",phase="train",imagesize=(256,256)):
        self.mode = mode
        self.phase = phase
        self.data_dir = data_dir
        self.transform = ToTensor()
        self.imagesize = imagesize

        with open(os.path.join(self.data_dir,"vcr1annots",f"{self.phase}.jsonl"),"r") as f:
            self.items = [json.loads(s) for s in f]


    def _make_text(self,source_txt,objects,is_q=True):
        for idx,txt in enumerate(source_txt):
            if type(txt)==list: #オブジェクトはリストで与えられる
                source_txt[idx] = ""
                for obj in txt:
                    #質問にしかlocationを入れない
                    if is_q:
                        source_txt[idx] += f"loc{objects[obj]['bbox'][0]} loc{objects[obj]['bbox'][1]} loc{objects[obj]['bbox'][2]} loc{objects[obj]['bbox'][3]} {objects[obj]['name']}, " 
                    else:
                        source_txt[idx] += objects[obj]['name'] + ", "
                #一番最後のカンマを消す
                source_txt[idx] = source_txt[idx][:-2]
        
        #最後のピリオドや?をスペースで分けないために、ひとつ前に加えてから、一番最後をpopする
        source_txt[-2] = source_txt[-2] + source_txt[-1]
        source_txt = source_txt[:-1]

        return " ".join(source_txt).replace(" ' ","'")
    
    def _get_boxes(self,box,xrasio,yrasio):
        x1,y1,x2,y2,_ = box
        x1 = int(x1*xrasio)
        x2 = int(x2*xrasio)
        y1 = int(y1*yrasio)
        y2 = int(y2*yrasio)
        
        return (x1,y1,x2,y2)

    def __getitem__(self,idx):
        item= deepcopy(self.items[idx])
        image = Image.open(os.path.join(f'{self.data_dir}',"vcr1images",f'{item["img_fn"]}')).convert("RGB").resize(self.imagesize)
        image = self.transform(image)
        question = item["question"]
        answer = item["answer_choices"][item["answer_label"]]
        namelist = random.sample(_GENDER_NEUTRAL_NAMES,item["objects"].count('person'))
        with open(os.path.join(f'{self.data_dir}',"vcr1images",f'{item["metadata_fn"]}')) as f:
            metadata = json.load(f)
        xrasio = self.imagesize[0]/metadata["width"]
        yrasio = self.imagesize[0]/metadata["height"]
        objects = []
        per_count = 0
        for index,obj in enumerate(item["objects"]):
            bbox = self._get_boxes(metadata["boxes"][index],xrasio,yrasio)
            if obj=="person":
                objects.append({"name":namelist[per_count],"bbox":bbox})
                per_count+=1
            else:
                objects.append({"name":obj,"bbox":bbox})
        
        src_text = self._make_text(question,objects)
        tgt_text = self._make_text(answer,objects,is_q=False)
        return image,src_text,tgt_text

    def get_all(self,idx):
        item= deepcopy(self.items[idx])
        image = Image.open(os.path.join(f'{self.data_dir}',"vcr1images",f'{item["img_fn"]}')).convert("RGB").resize(self.imagesize)
        image = self.transform(image)
        question = item["question"]
        answer = item["answer_choices"][item["answer_label"]]
        namelist = random.sample(_GENDER_NEUTRAL_NAMES,item["objects"].count('person'))
        with open(os.path.join(f'{self.data_dir}',"vcr1images",f'{item["metadata_fn"]}')) as f:
            metadata = json.load(f)
        xrasio = self.imagesize[0]/metadata["width"]
        yrasio = self.imagesize[0]/metadata["height"]
        objects = []
        per_count = 0
        for index,obj in enumerate(item["objects"]):
            bbox = self._get_boxes(metadata["boxes"][index],xrasio,yrasio)
            if obj=="person":
                objects.append({"name":namelist[per_count],"bbox":bbox})
                per_count+=1
            else:
                objects.append({"name":obj,"bbox":bbox})
        
        src_text = self._make_text(question,objects)
        tgt_text = self._make_text(answer,objects,is_q=False)
        return image,src_text,tgt_text,objects


    def __len__(self):
        return len(self.items)

if __name__ =="__main__":
    _DATADIR = "/data/dataset/vcr"
    dataset = Vcrdataset(_DATADIR)
    data = dataset[11]
    print(data)
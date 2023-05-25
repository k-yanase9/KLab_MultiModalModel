import os
import torch
from torchvision.transforms import ToTensor
from pycocotools.coco import COCO
from PIL import Image

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, args, phase):
        self.images, self.tgt_texts = [], []
        self.transform = ToTensor()

        anno_path = os.path.join(args.data_dir, 'annotations', f'captions_{phase}2014.json')
        coco = COCO(anno_path)
        img_dir = os.path.join(args.data_dir, 'images', f'{phase}2014')

        for id, value in coco.imgs.items():
            img_name = value['file_name']

            img_path = os.path.join(img_dir, img_name)
            
            try:
                caption = coco.loadAnns(id)[0]['caption']
            except:
                continue
            self.images.append(img_path)
            self.tgt_texts.append(caption)

    def __getitem__(self, idx):
        image, tgt_text = self.images[idx], self.tgt_texts[idx]
        src_text = 'What does th image describe ?'
        image = Image.open(image).convert('RGB').resize((256,256))
        image = self.transform(image)

        return image, src_text, tgt_text
    
    def __len__(self):
        return len(self.images)
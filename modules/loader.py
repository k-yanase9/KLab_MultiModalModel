import os
import torch
from torchvision.transforms import ToTensor
from .coco import SilentCOCO
from PIL import Image

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, args, phase):
        self.images, self.tgt_texts = [], []
        self.transform = ToTensor()

        anno_path = os.path.join(args.data_dir, 'annotations', f'captions_{phase}2017.json')
        coco = SilentCOCO(anno_path)
        img_dir = os.path.join(args.data_dir, f'{phase}2017')

        for image_id in coco.getImgIds():
            image_info = coco.loadImgs(image_id)[0]
            img_name = image_info['file_name']
            img_path = os.path.join(img_dir, img_name)
            
            caption = coco.loadAnns(coco.getAnnIds(image_id))[0]['caption']
            
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
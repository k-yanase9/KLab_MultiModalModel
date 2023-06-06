import os
import json
import torch
from torchvision.transforms import ToTensor
from .coco import SilentCOCO
from PIL import Image
from .vcrloader import Vcrdataset
from .vqa2loader import Vqa2dataset

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self):
        self.images, self.tgt_texts, self.src_texts = [], [], []
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image, src_text, tgt_text = self.images[idx], self.src_texts[idx], self.tgt_texts[idx]
        image = Image.open(image).convert('RGB').resize((256,256))
        image = self.transform(image)

        return image, src_text, tgt_text
    
    def __len__(self):
        return len(self.images)

class COCODatasetLoader(DatasetLoader):
    def __init__(self, data_dir='/data/datatset/mscoco2017', phase='train'):
        super().__init__()
        anno_path = os.path.join(data_dir, 'annotations', f'captions_{phase}2017.json')
        coco = SilentCOCO(anno_path)
        img_dir = os.path.join(data_dir, f'{phase}2017')

        for image_id in coco.getImgIds():
            image_info = coco.loadImgs(image_id)[0]
            img_name = image_info['file_name']
            img_path = os.path.join(img_dir, img_name)
            
            caption = coco.loadAnns(coco.getAnnIds(image_id))[0]['caption']
            
            self.images.append(img_path)
            self.src_texts.append('What does th image describe ?')
            self.tgt_texts.append(caption)

class RedCapsDatasetLoader(DatasetLoader):
    def __init__(self, data_dir='/data/dataset/redcaps', phase='train'):
        super().__init__()
        anno_dir = os.path.join(data_dir, 'annotations')
        img_dir = os.path.join(data_dir, 'images')

        for annotations_file_name in os.listdir(anno_dir):
            annotations_filepath = os.path.join(anno_dir, annotations_file_name)
            annotations = json.load(open(annotations_filepath))
            
            for ann in annotations["annotations"]:
                img_path = os.path.join(img_dir, ann["subreddit"], f"{ann['image_id']}.jpg")
                self.images.append(img_path)
                self.src_texts.append(ann['raw_caption'])

    def __getitem__(self, idx):
        image, src_text = self.images[idx], self.src_texts[idx]
        src_text = src_text.replace('.', ' .').replace(',', ' ,').replace('!', ' !').replace('?', ' ?') # ,.!?の前にスペースを挿入
        src_text = src_text.split() # 単語に分割
        # len(src_text) * 0.15だけランダムにsrc_textのインデックスを取得
        mask_idx = torch.randperm(len(src_text))[:int(len(src_text) * 0.15)+1]

        tgt_text = ['<extra_id_0>']
        j = 0
        for i in range(len(src_text)):
            if i in mask_idx:
                tgt_text.append(src_text[i])
                tgt_text.append(f'<extra_id_{j+1}>')
                src_text[i] = f'<extra_id_{j}>'
                j += 1
        src_text = ' '.join(src_text)
        tgt_text = ' '.join(tgt_text)

        image = Image.open(image).convert('RGB').resize((256,256))
        image = self.transform(image)

        return image, src_text, tgt_text
    



def get_dataloader(args, phase, rank):
    if 'mscoco' in args.data_dir.lower():
        dataset = COCODatasetLoader(args.data_dir, phase)
    elif 'redcaps' in args.data_dir.lower():
        dataset = RedCapsDatasetLoader(args.data_dir, phase)
    elif 'vrc' in args.data_dir.lower():
        dataset = Vcrdataset(args.data_dir,phase=phase)
    elif 'vqa2' in args.data_dir.lower():
        dataset = Vqa2dataset(args.data_dir,phase=phase)
    else:
        raise NotImplementedError
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=torch.cuda.device_count(), rank=rank, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=os.cpu_count()//4, pin_memory=True, sampler=sampler)
    return dataloader
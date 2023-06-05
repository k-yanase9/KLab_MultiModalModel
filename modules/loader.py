import os
import torch
from torchvision.transforms import ToTensor
from .coco import SilentCOCO
from PIL import Image

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
    def __init__(self, data_dir, phase):
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
            self.tgt_texts.append(caption)
            self.src_texts.append('What does th image describe ?')

def get_dataloader(args, phase, rank):
    if 'mscoco' in args.data_dir.lower():
        dataset = COCODatasetLoader(args.data_dir, phase)
    else:
        raise NotImplementedError
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=torch.cuda.device_count(), rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=os.cpu_count()//4, pin_memory=True, sampler=sampler)
    return dataloader
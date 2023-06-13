import os
import json
import torch
from PIL import Image
from .dataset_loader import DatasetLoader

class RedCapsDatasetLoader(DatasetLoader):
    def __init__(self, data_dir='/data/dataset/redcaps', phase='train'):
        super().__init__()
        anno_dir = os.path.join(data_dir, 'removed_annotations')
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

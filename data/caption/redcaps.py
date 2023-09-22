import os
from ..dataset_loader import DatasetLoader, CAPTION_SRC_TEXT

class RedCaps_Caption(DatasetLoader):
    def __init__(self, data_dir='/data01/redcaps', phase='train', resize=256):
        super().__init__(resize)
        text_tsv_path = os.path.join(data_dir, f'{phase}_text.tsv')
        img_tsv_path = os.path.join(data_dir, f'{phase}_img.tsv')
        
        with open(text_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, class_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, 'images', img_name)
            self.images.append(img_path)
            self.src_texts.append(CAPTION_SRC_TEXT)
            self.tgt_texts.append(caption)

        with open(img_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, class_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, 'images', img_name)
            self.images.append(img_path)
            self.src_texts.append(CAPTION_SRC_TEXT)
            self.tgt_texts.append(caption)
        
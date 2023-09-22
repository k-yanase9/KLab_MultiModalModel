import os
from ..dataset_loader import DatasetLoader, CAPTION_SRC_TEXT

class CC3M_Caption(DatasetLoader):
    def __init__(self,data_dir="/data01/cc3m", phase="train", resize=256):
        super().__init__(resize)
        tsv_path = os.path.join(data_dir, f'{phase}.tsv')
        
        with open(tsv_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines[1:]:
            img_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, phase, img_name)
            self.images.append(img_path)
            self.src_texts.append(CAPTION_SRC_TEXT)
            self.tgt_texts.append(caption)
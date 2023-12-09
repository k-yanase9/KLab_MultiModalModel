import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class Grit20M_RegionCaption(DatasetLoader):
    def __init__(self,data_dir="/data/dataset/grit20m",phase="train", **kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir, f'{phase}_region_caption.tsv')

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        if phase=='val':
            lines = lines[:MAX_VAL_DATA_SIZE]

        for line in lines:
            img_name, src, tgt = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
            self.src_texts.append(src)
            self.tgt_texts.append(tgt)
import os
from ..dataset_loader import DatasetLoader

class ICDAR_Read(DatasetLoader):
    def __init__(self, data_dir:str="/data01/ICDAR2013/", phase:str="train", is_tgt_id=False, **kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir, f"{phase}_loc40.tsv")

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]

        for line in lines:
            img_name, label, locs = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
            self.src_texts.append(f"What is the OCR of the region {locs}?")
            self.tgt_texts.append(label)

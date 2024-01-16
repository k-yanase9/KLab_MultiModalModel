import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class Vcrdataset(DatasetLoader):
    """Vcrのデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/vcr",phase:str="train",**kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir, f'{phase}_vqa_fix_cut.tsv')

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        if phase=='val':
            lines = lines[:MAX_VAL_DATA_SIZE]

        for line in lines:
            img_name, question, answer = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
            self.src_texts.append(question)
            self.tgt_texts.append(answer)

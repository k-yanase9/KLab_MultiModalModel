import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class Tdiucdataset(DatasetLoader):
    """Vcrのデータセット
    """    
    def __init__(self,data_dir:str="/data01/tdiuc",phase:str="train",**kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir, f'{phase}_pngfix.tsv')

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        if phase=='val':
            lines = lines[:MAX_VAL_DATA_SIZE]

        for line in lines:
            image_name, question, answer = line.removesuffix('\n').split('\t')
            self.images.append(os.path.join(data_dir, image_name))
            self.src_texts.append(question)
            self.tgt_texts.append(answer)

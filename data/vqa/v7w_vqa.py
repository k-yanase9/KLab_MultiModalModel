import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class Visual7W_VQA(DatasetLoader):
    """Visual7Wのデータセット
    """    
    def __init__(self,data_dir:str="/data01/visual7w",phase:str="train", **kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir, f'{phase}_vqa.tsv')

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        if phase=='val':
            lines = lines[:MAX_VAL_DATA_SIZE]

        for line in lines:
            image_name, question, answer, dummy1, dummy2, dummy3 = line.removesuffix('\n').split('\t')
            self.images.append(os.path.join(data_dir, 'images', image_name))
            self.src_texts.append(question)
            self.tgt_texts.append(answer)
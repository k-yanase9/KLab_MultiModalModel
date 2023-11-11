import os
from ..dataset_loader import DatasetLoader

class Visual7W_VQA(DatasetLoader):
    """Visual7Wのデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/Visual7w",phase:str="train"):
        super().__init__()

        with open(os.path.join(data_dir, f'{phase}_vqa.tsv')) as f:
            items = f.readlines()
        for item in items[1:]:
            item = item.rstrip()
            image_name, question, answer, dummy1, dummy2, dummy3 = item.split('\t')
            self.images.append(os.path.join(data_dir, 'images', image_name))
            self.src_texts.append(question)
            self.tgt_texts.append(answer)
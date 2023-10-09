import os
from ..dataset_loader import DatasetLoader


class Vcrdataset(DatasetLoader):
    """Vcrのデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/vcr",phase:str="train",imagesize:tuple[int,int]=(256,256)):
        super().__init__()

        with open(os.path.join(data_dir,f"{phase}_vqa_loc40_pngfix.tsv")) as f:
            items = f.read()

        items = items.split("\n")
        items = [item.split("\t") for item in items]

        items = items[1:-1]

        self.tgt_texts = [item[2] for item in items]
        self.src_texts = [item[1] for item in items]
        self.images = [os.path.join(data_dir,item[0]) for item in items]


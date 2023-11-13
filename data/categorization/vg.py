import os
from ..dataset_loader import DatasetLoader

class VisualGenome_Categorization(DatasetLoader):
    """VisualGenomeのcategorizationデータセット
    """    
    def __init__(self, data_dir:str="/data01/visual_genome/", phase:str="train", is_tgt_id:bool=False):
        super().__init__()        

        with open(os.path.join(data_dir, f"{phase}_categorization.tsv")) as f:
            items = f.readlines()
        items = [item.rstrip().split("\t") for item in items]
        items = items[1:]

        if is_tgt_id:
            raise NotImplementedError
        else:
            self.tgt_texts = [item[2] for item in items]

        self.src_texts = [f"What is the category of the region {item[1]}?" for item in items]
        self.images = [os.path.join(data_dir,"images_256",f"{item[0]}.png") for item in items]
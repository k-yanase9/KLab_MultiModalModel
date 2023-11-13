import os
from ..dataset_loader import DatasetLoader

class VisualGenome_Relation(DatasetLoader):
    """VisualGenomeのrelationデータセット
    """    
    def __init__(self, data_dir:str="/data01/visual_genome/", phase:str="train"):
        super().__init__()        

        with open(os.path.join(data_dir,f"{phase}_relation.tsv")) as f:
            items = f.readlines()
        tmp = []
        for item in items:
            item = item.rstrip().split("\t")
            if len(item) < 6:
                continue
            tmp.append(item)
        items = tmp[1:]

        self.tgt_texts = [f"{item[1]} {item[5]} {item[3]}" for item in items]
        self.src_texts = [f"What is the relationship between {item[1]}{item[2]} and {item[3]}{item[4]}?" for item in items]
        self.images = [os.path.join(data_dir,"images_256",f"{item[0]}.png") for item in items]

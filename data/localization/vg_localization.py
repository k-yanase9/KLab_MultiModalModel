import os
from ..dataset_loader import DatasetLoader

class VisualGenome_Localization(DatasetLoader):
    """VisualGenomeのlocalizationデータセット
    """    
    def __init__(self, data_dir:str="/data01/visual_genome/", phase:str="train"):
        super().__init__()        

        with open(os.path.join(data_dir,f"{phase}_localize.tsv")) as f:
            items = f.readlines()
        items = [item.rstrip().split("\t") for item in items]
        items = items[1:]

        self.tgt_texts = [item[2] for item in items]
        self.src_texts = [f"Which regions does the text \"{item[1]}\" describe?" for item in items]
        self.images = [os.path.join(data_dir,"images_256",f"{item[0]}.png") for item in items]

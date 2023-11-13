import os
from ..dataset_loader import DatasetLoader, DETECTION_SRC_TEXT

class VisualGenome_Detection(DatasetLoader):
    """VisualGenomeのdetectionデータセット
    """    
    def __init__(self, data_dir:str="/data01/visual_genome/", phase:str="train"):
        super().__init__()        

        with open(os.path.join(data_dir,f"{phase}_detect.tsv")) as f:
            items = f.readlines()
        items = [item.rstrip().split("\t") for item in items]
        items = items[1:]

        self.tgt_texts = [item[1] for item in items]
        self.src_texts = [DETECTION_SRC_TEXT]*len(items)
        self.images = [os.path.join(data_dir,"images_256",f"{item[0]}.png") for item in items]

import os
from ..dataset_loader import DatasetLoader

class VisualGenome_VQA(DatasetLoader):
    """VisualGenomeのVQAデータセット
    """    
    def __init__(self, data_dir:str="/data01/visual_genome/", phase:str="train"):
        super().__init__()        

        with open(os.path.join(data_dir, f"{phase}_qa.tsv")) as f:
            items = f.readlines()

        for item in items[1:]:
            image_id, question, answer = item.rstrip().split("\t")
            self.images.append(os.path.join(data_dir,"images_256",f"{image_id}.png"))
            self.src_texts.append(question)
            self.tgt_texts.append(answer)
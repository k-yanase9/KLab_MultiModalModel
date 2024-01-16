import os
from ..dataset_loader import DatasetLoader

class DeepFashion2_Localization(DatasetLoader):
    """DeepFashion2のLocalizationタスク用のDatasetLoader
    """    
    def __init__(self, data_dir:str="/data01/DeepFashion2/", phase:str="train", is_tgt_id:bool=False, **kwargs):
        super().__init__(**kwargs)
        if phase=="val":
            phase = "validation"
        tsv_path = os.path.join(data_dir, f"{phase}_loc40.tsv")

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]

        for line in lines:
            line = line.removesuffix('\n').split('\t')
            img_name, cat_id, cat_name, loc  = line
            img_path = os.path.join(data_dir, phase, img_name)
            self.images.append(img_path)
            self.src_texts.append(f'Which regions does the text "{cat_name}" describe?')
            self.tgt_texts.append(loc)
import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class VisualGenome_Categorization(DatasetLoader):
    """VisualGenomeのcategorizationデータセット
    """    
    def __init__(self, data_dir:str="/data01/visual_genome/", phase:str="train", is_tgt_id:bool=False, **kwargs):
        super().__init__(**kwargs)        
        tsv_path = os.path.join(data_dir, f"{phase}_categorization.tsv")

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        if phase=='val':
            lines = lines[:MAX_VAL_DATA_SIZE]

        for line in lines:
            line = line.removesuffix('\n').split('\t')
            img_name, loc, cat_name, cat_id = line
            img_path = os.path.join(data_dir, f"images_256", f"{img_name}.png")
            self.images.append(img_path)
            self.src_texts.append(f"What is the category of the region {loc}?")
            if is_tgt_id:
                self.tgt_texts.append(int(cat_id))
            else:
                self.tgt_texts.append(cat_name)

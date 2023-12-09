import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class Grit20M_RefExp(DatasetLoader):
    """Grit20mのReferring Expressionデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/grit20m/",phase:str="train", **kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir,  f"{phase}_ref_exp_cut.tsv")
            
        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        count = 0

        for line in lines:
            if count >= MAX_VAL_DATA_SIZE and phase == 'val':
                break
            line = line.removesuffix('\n').split('\t')
            if len(line) < 3:
                continue
            img_name, src, tgt = line
            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
            self.src_texts.append(src)
            self.tgt_texts.append(tgt)
            count += 1

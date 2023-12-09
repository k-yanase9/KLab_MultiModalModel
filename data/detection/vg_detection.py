import os
from ..dataset_loader import DatasetLoader, DETECTION_SRC_TEXT, MAX_VAL_DATA_SIZE

class VisualGenome_Detection(DatasetLoader):
    """VisualGenomeのdetectionデータセット
    """    
    def __init__(self, data_dir:str="/data01/visual_genome/", phase:str="train", **kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir, f"{phase}_detect_fix_cut_max_tokens.tsv")

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        if phase=='val':
            lines = lines[:MAX_VAL_DATA_SIZE]

        for line in lines:
            img_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, f"images_256", f"{img_name}.png")
            self.images.append(img_path)
            self.src_texts.append(DETECTION_SRC_TEXT)
            self.tgt_texts.append(caption)

import os
from ..dataset_loader import DatasetLoader, DETECTION_SRC_TEXT

class DeepFashion2_Detection(DatasetLoader):
    """DeepFashion2のDetectionタスク用のDatasetLoader
    """    
    def __init__(self, data_dir:str="/data01/DeepFashion2/", phase:str="train", is_tgt_id:bool=False, **kwargs):
        super().__init__(**kwargs)
        if phase=="val":
            phase = "validation"
        if is_tgt_id:
            tsv_path = os.path.join(data_dir, f"{phase}_det_id.tsv")
        else:
            tsv_path = os.path.join(data_dir, f"{phase}_det_words.tsv")

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]

        for line in lines:
            line = line.removesuffix('\n').split('\t')
            img_name, det = line
            if img_name == '':
                continue
            img_path = os.path.join(data_dir, phase, img_name)
            self.images.append(img_path)
            self.src_texts.append(DETECTION_SRC_TEXT)
            self.tgt_texts.append(det)
import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class VisualGenome_RefExp(DatasetLoader):
    """VisualGenomeのReferring Expressionデータセット
    """    
    def __init__(self,data_dir:str="/data01/visual_genome/", phase:str="train", **kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir,  f"{phase}_ref_exp.tsv")
        
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
            image_id, caption, loc = line
            img_path = os.path.join(data_dir, 'images_256', f'{image_id}.png')
            self.images.append(img_path)
            self.src_texts.append(f'Which regions does the text \"{caption}\" describe?')
            self.tgt_texts.append(loc)
            count += 1
    

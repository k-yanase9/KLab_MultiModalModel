import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class VisualGenome_RegionCaption(DatasetLoader):
    """VisualGenomeのRegionCaptionデータセット
    """    
    def __init__(self,data_dir:str="/data01/visual_genome/", phase:str="train", **kwargs):
        super().__init__(**kwargs)        
        
        with open(os.path.join(data_dir, f"{phase}_ref_exp.tsv")) as f:
            lines = f.readlines()
        lines = lines[1:]
        count = 0

        for line in lines:
            if count >= MAX_VAL_DATA_SIZE and phase == 'val':
                break
            line = line.removesuffix('\n').split('\t')
            if len(line) < 3:
                continue
            image_id, caption, locs = line
            for loc in locs.split():
                if count >= MAX_VAL_DATA_SIZE and phase == 'val':
                    break
                self.images.append(os.path.join(data_dir,"images_256",f"{image_id}.png"))
                self.src_texts.append(f'What does the region {loc} describe?')
                self.tgt_texts.append(caption)
                count += 1

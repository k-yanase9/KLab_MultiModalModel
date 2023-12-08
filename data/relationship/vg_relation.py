import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class VisualGenome_Relation(DatasetLoader):
    """VisualGenomeのrelationデータセット
    """    
    def __init__(self, data_dir:str="/data01/visual_genome/", phase:str="train", **kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir, f"{phase}_relation.tsv")
        
        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        count = 0

        for line in lines:
            if count >= MAX_VAL_DATA_SIZE and phase == 'val':
                break
            line = line.removesuffix('\n').split('\t')
            if len(line) < 6:
                continue
            img_name, obj1, loc1, obj2, loc2, relation = line
            img_path = os.path.join(data_dir, f"images_256", f"{img_name}.png")
            self.images.append(img_path)
            self.src_texts.append(f"What is the relationship between {obj1}{loc1} and {obj2}{loc2}?")
            self.tgt_texts.append(f"{obj1} {relation} {obj2}")
            count += 1

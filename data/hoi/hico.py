import os
from ..dataset_loader import DatasetLoader, HOI_SRC_TEXT

class HICO_HOI(DatasetLoader):
    """HICOデータセット
    """
    def __init__(self, data_dir:str="/data01/hico_det/", phase:str="train", is_tgt_id=False, **kwargs):
        super().__init__(**kwargs)
        if is_tgt_id:
            tsv_path = os.path.join(data_dir, f"{phase}_loc40_wo_no_interact.tsv")
        else:
            tsv_path = os.path.join(data_dir, f"{phase}_loc40.tsv")
        
        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        
        for line in lines:
            img_name, anns = line.removesuffix('\n').split("\t")
            tgt_text = []
            for ann in anns.split("&&"):
                human_id, human_name, human_loc, hoi_id, hoi_name, obj_id, obj_name, obj_loc = ann.split(',')
                tgt_text.append(f"{human_name}{human_loc} {hoi_name} {obj_name}{obj_loc}")
                if is_tgt_id:
                    break

            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
            if is_tgt_id:
                self.src_texts.append(f'What is the interaction between {human_name}{human_loc} and {obj_name}{obj_loc}?')
                self.tgt_texts.append(f'<add_{hoi_id}>')
            else:
                self.src_texts.append(HOI_SRC_TEXT)
                self.tgt_texts.append(','.join(tgt_text))
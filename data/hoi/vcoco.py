import os
from ..dataset_loader import DatasetLoader, HOI_SRC_TEXT

class VCOCO_HOI(DatasetLoader):
    """VCOCOデータセット
    """
    def __init__(self, data_dir:str="/data01/vcoco/", phase:str="train", is_tgt_id=False, **kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir, f"{phase}_loc40.tsv")
        
        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        
        for line in lines:
            img_name, anns = line.removesuffix('\n').split("\t")
            # tgt_text = []
            for ann in anns.split("&&"):
                human_id, human_name, human_loc, hoi_id, hoi_name, obj_id, obj_name, obj_loc = ann.split(',')
                hoi_names = hoi_name.split('%')
                hoi_ids = hoi_id.split('%')

                img_path = os.path.join(data_dir, img_name)
                self.images.append(img_path)
                self.src_texts.append(f'What is the interaction between {human_name}{human_loc} and {obj_name}{obj_loc}?')
                if is_tgt_id:
                    hoi_ids = [f'<add_{hoi_id}>' for hoi_id in hoi_ids]
                    self.tgt_texts.append(''.join(hoi_ids))
                else:
                    self.tgt_texts.append(','.join(hoi_names))

                # tgt_text.append(f"{human_name}{human_loc} {','.join(hoi_names)} {obj_name}{obj_loc}")

            if False:
                img_path = os.path.join(data_dir, img_name)
                self.images.append(img_path)
                self.src_texts.append(HOI_SRC_TEXT)
                self.tgt_texts.append(','.join(tgt_text))
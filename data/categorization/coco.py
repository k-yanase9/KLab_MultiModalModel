import os
from ..dataset_loader import DatasetLoader

class COCO_Categorization(DatasetLoader):
    def __init__(self, data_dir='/data/mscoco2017/', phase='train', is_tgt_id=False):
        super().__init__()
        tsv_path = os.path.join(data_dir, f'{phase}.tsv')

        with open(tsv_path) as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_path, loc, cat_id, cat_name = line.split("\t")
            img_path = os.path.join(data_dir, img_path)
            self.images.append(img_path)
            self.src_texts.append(f'What is the category of the region {loc}?')
            if is_tgt_id:
                self.tgt_texts.append(int(cat_id))
            else:
                self.tgt_texts.append(cat_name.strip())
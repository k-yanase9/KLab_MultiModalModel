import os

from ..dataset_loader import DatasetLoader, CLASSIFY_SRC_TEXT, MAX_VAL_DATA_SIZE


class Places365_Classify(DatasetLoader):
    def __init__(self,data_dir='/data01/places365', phase='train', **kwargs):
        super().__init__(**kwargs)
        text_tsv_path = os.path.join(data_dir, f'{phase}_text.tsv')
        img_tsv_path = os.path.join(data_dir, f'{phase}_img.tsv')

        with open(text_tsv_path, 'r') as f:
            lines = f.readlines()

        with open(img_tsv_path, 'r') as f:
            tmp = f.readlines()

        lines = lines[1:] + tmp[1:]
        if phase == 'val':
            lines = lines[:MAX_VAL_DATA_SIZE]

        for line in lines:
            img_name, class_name = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
            self.src_texts.append(CLASSIFY_SRC_TEXT)
            self.tgt_texts.append(class_name.strip())

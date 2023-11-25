import os

from ..dataset_loader import CLASSIFY_SRC_TEXT, DatasetLoader


class Places365_Classify(DatasetLoader):
    def __init__(self,data_dir='/data01/places365', phase='train'):
        super().__init__()
        text_tsv_path = os.path.join(data_dir, f'{phase}_text.tsv')
        img_tsv_path = os.path.join(data_dir, f'{phase}_img.tsv')

        with open(text_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, class_name = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            self.src_texts.append(CLASSIFY_SRC_TEXT)
            self.images.append(img_path)
            self.tgt_texts.append(class_name.strip())

        with open(img_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, class_name = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            self.src_texts.append(CLASSIFY_SRC_TEXT)
            self.images.append(img_path)
            self.tgt_texts.append(class_name.strip())

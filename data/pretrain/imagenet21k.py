import os
from .pretrain import ClassifyPretrainDatasetLoader

class ImageNet21k_Pretrain(ClassifyPretrainDatasetLoader):
    def __init__(self, data_dir='/data01/imagenet21k/', phase='train', **kwargs):
        super().__init__(**kwargs)
        text_tsv_path = os.path.join(data_dir, f'text_{phase}_256fix.tsv')
        img_tsv_path = os.path.join(data_dir, f'img_{phase}_256fix.tsv')

        with open(text_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, label = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            class_name = label.split()[0].replace('_', ' ')
            self.images.append(img_path)
            self.src_texts.append(class_name.strip())

        with open(img_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, label = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            class_name = label.split()[0].replace('_', ' ')
            self.images.append(img_path)
            self.src_texts.append(class_name.strip())
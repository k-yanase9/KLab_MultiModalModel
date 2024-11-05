import os
from .pretrain import PretrainDatasetLoader

class CC3M_Pretrain(PretrainDatasetLoader):
    def __init__(self, data_dir='/home/data/cc3m', phase='train', **kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir, f'{phase}.tsv')

        with open(tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, phase, img_name)
            self.images.append(img_path)
            self.src_texts.append(caption)

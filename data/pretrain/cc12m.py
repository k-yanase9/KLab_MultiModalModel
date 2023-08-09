import os
from .pretrain import PretrainDatasetLoader

class CC12MPretrainDatasetLoader(PretrainDatasetLoader):
    def __init__(self, args, data_dir='/data/datatset/cc12m', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)
        tsv_path = os.path.join(data_dir, 'train.tsv')

        with open(tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, 'train', img_name)
            self.images.append(img_path)
            self.src_texts.append(caption)
import os
from .pretrain import PretrainDatasetLoader

class CC12M_Pretrain(PretrainDatasetLoader):
    def __init__(self, args, data_dir='/data01/cc12m/', phase='train', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)
        text_tsv_path = os.path.join(data_dir, f'text_{phase}.tsv')
        img_tsv_path = os.path.join(data_dir, f'img_{phase}.tsv')

        with open(text_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, 'images', img_name)
            self.images.append(img_path)
            self.src_texts.append(caption)

        with open(img_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, 'images', img_name)
            self.images.append(img_path)
            self.src_texts.append(caption)
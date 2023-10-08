from .pretrain import ClassifyPretrainDatasetLoader
import os

class Places365_Pretrain(ClassifyPretrainDatasetLoader):
    def __init__(self, args, data_dir='/data01/places365', phase='train', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)
        text_tsv_path = os.path.join(data_dir, f'{phase}_text.tsv')
        img_tsv_path = os.path.join(data_dir, f'{phase}_img.tsv')

        with open(text_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
            self.src_texts.append(caption)

        with open(img_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, class_name = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
            self.src_texts.append(class_name.strip())
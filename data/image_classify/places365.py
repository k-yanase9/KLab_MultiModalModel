from ..dataset_loader import DatasetLoader
import os

class Places365_Classify(DatasetLoader):
    def __init__(self, args, data_dir='/data01/places365', phase='train', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)
        text_tsv_path = os.path.join(data_dir, f'{phase}_text.tsv')
        img_tsv_path = os.path.join(data_dir, f'{phase}_img.tsv')

        with open(os.path.join(data_dir, 'categories_places365.txt'), 'r') as f:
            self.classes = [c.split()[0] for c in f]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        with open(text_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            class_name = '/'.join(img_name.split('/')[1:3])
            self.images.append(img_path)
            self.src_texts.append(self.class_to_idx[class_name])

        with open(img_tsv_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, caption = line.removesuffix('\n').split('\t')
            img_path = os.path.join(data_dir, img_name)
            class_name = '/'.join(img_name.split('/')[1:3])
            self.images.append(img_path)
            self.src_texts.append(self.class_to_idx[class_name])
import os
from ..dataset_loader import DatasetLoader, CAPTION_SRC_TEXT

class RedCaps_Caption(DatasetLoader):
    def __init__(self, data_dir='/data01/redcaps', phase='train', resize=256):
        super().__init__(resize)
        
        with open(os.path.join(data_dir, f'{phase}.tsv'), 'r') as f:
            items = f.read().split('\n')

        items = items[1:]
        items = [item.split('\t') for item in items]

        self.images = [os.path.join(data_dir, 'images', item[0]) for item in items]
        self.src_texts = [CAPTION_SRC_TEXT for _ in range(len(items))]
        self.tgt_texts = [item[2] for item in items]

        
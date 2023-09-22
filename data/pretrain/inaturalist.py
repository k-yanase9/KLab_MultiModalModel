import os
from .pretrain import ClassifyPretrainDatasetLoader
from torchvision.datasets import INaturalist

class INaturalist_Pretrain(ClassifyPretrainDatasetLoader):
    def __init__(self, args, data_dir='/data01/inaturalist', phase='train', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)
        if phase == 'train':
            version = '2021_train'
        elif phase == 'val':
            version = '2021_valid'
        else:
            raise NotImplementedError
        
        dataset = INaturalist(root=data_dir, version=version)
        for cat_id, fname in dataset.index:
            img = os.path.join(data_dir, version, dataset.all_categories[cat_id], fname)
            self.images.append(img)

            target = ' '.join(dataset.all_categories[cat_id].split('_')[1:]) + '.'
            self.src_texts.append(target)

        del dataset
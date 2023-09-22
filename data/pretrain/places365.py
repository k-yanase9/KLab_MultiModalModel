from .pretrain import ClassifyPretrainDatasetLoader
from torchvision.datasets import Places365

class Places365_Pretrain(ClassifyPretrainDatasetLoader):
    def __init__(self, args, data_dir='/data01/places365', phase='train', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)
        if phase == 'train':
            split = 'train-standard'
        elif phase == 'val':
            split = 'val'
        else:
            raise NotImplementedError
        
        dataset = Places365(root=data_dir, split=split, small=True)
        for img, label in dataset.imgs:
            self.images.append(img)
            self.src_texts.append(f'{dataset.classes[label].split("/")[2].replace("_", " ")}.')
        del dataset
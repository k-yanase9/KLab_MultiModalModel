import os
import tarfile
from PIL import Image
from .pretrain import ClassifyPretrainDatasetLoader

class ImageNet21kPretrainDatasetLoader(ClassifyPretrainDatasetLoader):
    def __init__(self, args, data_dir='/data01/imagenet_21k/', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)

        # Load class names
        ids_txt_path = os.path.join(data_dir, 'imagenet21k_wordnet_ids.txt')
        class_name_txt_path = os.path.join(data_dir, 'imagenet21k_wordnet_lemmas.txt')

        with open(ids_txt_path, 'r') as f:
            ids = [line.strip() for line in f.readlines()]

        with open(class_name_txt_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

        imagenet_classes = dict(zip(ids, class_names))

        img_folder_path = os.path.join(data_dir, 'images/')
        folders = os.listdir(img_folder_path)
        for folder in folders:
            folder_path = os.path.join(img_folder_path, folder)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    self.images.append(img_path)
                    self.src_texts.append(imagenet_classes[folder])
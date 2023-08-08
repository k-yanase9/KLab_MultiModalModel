import os
import tarfile
from PIL import Image
from .pretrain import PretrainDatasetLoader
from ..mask.utils import make_mask_textpair

class ImageNet21kPretrainDatasetLoader(PretrainDatasetLoader):
    def __init__(self, args, data_dir='/data01/imagenet_21k/', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, data_dir, resize, src_tokenizer, tgt_tokenizer, mask_probability)

        # Load class names
        ids_txt_path = os.path.join(data_dir, 'imagenet21k_wordnet_ids.txt')
        class_name_txt_path = os.path.join(data_dir, 'imagenet21k_wordnet_lemmas.txt')

        with open(ids_txt_path, 'r') as f:
            ids = [line.strip() for line in f.readlines()]

        with open(class_name_txt_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

        img_folder_path = os.path.join(data_dir, 'tar/')
        imagenet_classes = dict(zip(ids, class_names))

        tar_files = os.listdir(img_folder_path)
        for tar_file in tar_files:
            tar = tarfile.open(img_folder_path + tar_file)
            img_names = tar.getnames()

            for name in img_names:
                class_id = name.split("_")[0]
                img_name = tar.extractfile(name)
                img = Image.open(img_name).resize((resize, resize))
                self.images.append(img)
                self.src_texts.append(f'a photo of {imagenet_classes[class_id]}')
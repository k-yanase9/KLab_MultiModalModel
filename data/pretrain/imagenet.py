import os
from .pretrain import PretrainDatasetLoader

class ImageNetPretrainDatasetLoader(PretrainDatasetLoader):
    def __init__(self, data_dir='/data/datatset/imagenet_2012', resize=256):
        super().__init__(data_dir, resize)
        img_folder_path = os.path.join(data_dir, "train")

        # Load class names
        map_clsloc_path = os.path.join(data_dir, 'map_clsloc.txt')
        imagenet_classes = {}
        with open(map_clsloc_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                imagenet_classes[line[0]] = line[2]

        for class_folder, class_name in imagenet_classes.items():
            class_folder_path = os.path.join(img_folder_path, class_folder)
            class_name = class_name.replace('_', ' ')
            for img in os.listdir(class_folder_path):
                img_path = os.path.join(class_folder_path, img)
                self.images.append(img_path)
                self.src_texts.append(f'a photo of {class_name}')

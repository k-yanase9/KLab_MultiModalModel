import os

from ..dataset_loader import CLASSIFY_SRC_TEXT, DatasetLoader


class ImageNet_Classify(DatasetLoader):
    def __init__(self, data_dir='/data01/imagenet', phase='train', is_tgt_id=False):
        super().__init__()
        img_folder_path = os.path.join(data_dir, f'{phase}_256')

        # Load class names
        map_clsloc_path = os.path.join(data_dir, 'map_clsloc.txt')
        imagenet_classes = {}
        
        class_folder_to_id = {}
        with open(map_clsloc_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split(' ')
                imagenet_classes[line[0]] = line[2]
                class_folder_to_id[line[0]] = i

        for class_folder, class_name in imagenet_classes.items():
            class_folder_path = os.path.join(img_folder_path, class_folder)
            class_name = class_name.replace('_', ' ')
            for img in os.listdir(class_folder_path):
                img_path = os.path.join(class_folder_path, img)
                self.images.append(img_path)
                self.src_texts.append(CLASSIFY_SRC_TEXT)
                if is_tgt_id:
                    self.tgt_texts.append(class_folder_to_id[class_folder])
                else:
                    self.tgt_texts.append(class_name)

import os
import json
from .pretrain import PretrainDatasetLoader

class RedCapsPretrainDatasetLoader(PretrainDatasetLoader):
    def __init__(self, data_dir='/data/dataset/redcaps', resize=256):
        super().__init__(data_dir, resize)
        anno_dir = os.path.join(data_dir, 'removed_annotations')
        img_dir = os.path.join(data_dir, 'images')

        for annotations_file_name in os.listdir(anno_dir):
            if "2019" in annotations_file_name: # 2019年のデータのみ使用
                annotations_filepath = os.path.join(anno_dir, annotations_file_name)
                annotations = json.load(open(annotations_filepath))
                
                for ann in annotations["annotations"]:
                    img_path = os.path.join(img_dir, ann["subreddit"], f"{ann['image_id']}.jpg")
                    self.images.append(img_path)
                    self.src_texts.append(ann['raw_caption'])

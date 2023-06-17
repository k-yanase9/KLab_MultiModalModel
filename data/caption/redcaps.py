import os
import json
from ..dataset_loader import DatasetLoader

class RedCapsDatasetLoader(DatasetLoader):
    def __init__(self, data_dir='/data/dataset/redcaps'):
        super().__init__()
        anno_dir = os.path.join(data_dir, 'removed_annotations')
        img_dir = os.path.join(data_dir, 'images')

        for annotations_file_name in os.listdir(anno_dir):
            if "2020" in annotations_file_name: # 2020年のデータのみ使用
                annotations_filepath = os.path.join(anno_dir, annotations_file_name)
                annotations = json.load(open(annotations_filepath))
                
                for ann in annotations["annotations"]:
                    img_path = os.path.join(img_dir, ann["subreddit"], f"{ann['image_id']}.jpg")
                    tgt_text = ann['raw_caption'].replace('.', ' .').replace(',', ' ,').replace('!', ' !').replace('?', ' ?') # ,.!?の前にスペースを挿入

                    self.images.append(img_path)
                    self.src_texts.append('What does the image describe ?')
                    self.tgt_texts.append(tgt_text)
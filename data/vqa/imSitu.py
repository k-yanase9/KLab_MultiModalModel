import json
import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

class imSituDataset(DatasetLoader):
    def __init__(self,data_dir="/data/dataset/imSitu",phase="train",**kwargs):
        super().__init__(**kwargs)
        if phase =="val":
            phase = "dev"
        json_path = os.path.join(data_dir,"imSituVQA.json")

        with open(json_path) as f:
            lines = json.load(f)
        lines = lines[phase]
        count = 0

        for img_name, question, answer in zip(lines["image_file"],lines["question"],lines["answer"]):
            if count >= MAX_VAL_DATA_SIZE and phase == "dev":
                break
            img_path = os.path.join(data_dir,"of500_images_resized",img_name)
            self.images.append(img_path)
            self.src_texts.append(question)
            self.tgt_texts.append(answer)
            count += 1

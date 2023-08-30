import json
from ..dataset_loader import DatasetLoader

class Vqa2dataset(DatasetLoader):
    def __init__(self,data_dir="/data/dataset/vqa2",phase="train",imagesize=(256,256)):
        super().__init__()
        quetions = json.load(open(f'{data_dir}/v2_OpenEnded_mscoco_{phase}2014_questions.json'))
        answers = json.load(open(f'{data_dir}/v2_mscoco_{phase}2014_annotations.json'))

        self.src_texts = [item["question"] for item in quetions['questions']]
        self.tgt_texts = [item["multiple_choice_answer"] for item in answers['annotations']]
        self.images = [f'{data_dir}/{phase}2014/COCO_{phase}2014_{str(item["image_id"]).zfill(12)}.jpg' for item in quetions['questions']]

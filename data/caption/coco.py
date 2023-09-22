import os
import json
from pycocotools.coco import COCO
from collections import defaultdict
from ..dataset_loader import DatasetLoader, CAPTION_SRC_TEXT

class SilentCOCO(COCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

class COCO_Caption(DatasetLoader):
    def __init__(self, data_dir='/data/dataset/mscoco2017', phase='train'):
        super().__init__()
        anno_path = os.path.join(data_dir, 'annotations', f'captions_{phase}2017.json')
        coco = SilentCOCO(anno_path)
        img_dir = os.path.join(data_dir, f'{phase}2017')

        for image_id in coco.getImgIds():
            image_info = coco.loadImgs(image_id)[0]
            img_name = image_info['file_name']
            img_path = os.path.join(img_dir, img_name)
            
            caption = coco.loadAnns(coco.getAnnIds(image_id))[0]['caption']
            
            self.images.append(img_path)
            self.src_texts.append(CAPTION_SRC_TEXT)
            self.tgt_texts.append(caption)
import os
from ..dataset_loader import DatasetLoader, DETECTION_SRC_TEXT, MAX_VAL_DATA_SIZE

class Objects365_Detection(DatasetLoader):
    """openimageのdetectionデータセット
    """    
    def __init__(self,data_dir:str="/data01/objects365/",phase:str="train", **kwargs):
        super().__init__(**kwargs)
        tsv_path = os.path.join(data_dir,f"{phase}_dec_cut_max_tokens.tsv")
        #存在しない画像を除外するためのリスト
        dropimageidlist =['patch16_256/objects365_v2_00908726.png','patch6_256/objects365_v1_00320532.png','patch6_256/objects365_v1_00320534.png']
        dropimageidlist = [os.path.join("processed_data","images",phase,drop_id) for drop_id in dropimageidlist]

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        count = 0

        for line in lines:
            if count >= MAX_VAL_DATA_SIZE and phase == 'val':
                break
            img_name, caption = line.removesuffix('\n').split('\t')
            if img_name in dropimageidlist:
                continue
            img_path = os.path.join(data_dir, img_name)
            self.images.append(img_path)
            self.src_texts.append(DETECTION_SRC_TEXT)
            self.tgt_texts.append(caption)
            count += 1

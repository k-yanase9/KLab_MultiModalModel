import os
from ..dataset_loader import DatasetLoader, MAX_VAL_DATA_SIZE

#存在しない画像を除外するためのリスト
dropimageidlist =["7f1934f5884fad79","429019e83c1c2c94","4f818c006da84c9e","5b86e93f8654118a","673d74b7d39741c3","6dcd3ce37a17f2be","805baf9650a12710"
                   ,"98ac2996fc46b56d","a46a248a39f2d97c","9316d4095eab6d10","9ee38bb2e69da0ac","37625d59d0e0782a"]

class OpenImage_Categorization(DatasetLoader):
    """openimageのcategorizationデータセット
    """    
    def __init__(self,data_dir:str="/data/dataset/openimage/",phase:str="train",is_tgt_id:bool=False, **kwargs):
        super().__init__(**kwargs)        
        if phase=="val":
            phase = "validation"
        tsv_path = os.path.join(data_dir,"tsv_new",f"{phase}_40_cat.tsv")

        with open(tsv_path) as f:
            lines = f.readlines()
        lines = lines[1:]
        count = 0
        
        for line in lines:
            img_name, loc, cat_name, cat_id  = line.removesuffix('\n').split('\t')
            if img_name in dropimageidlist:
                continue
            img_path = os.path.join(data_dir,f"{phase}_256_png",f"{img_name}.png")
            self.images.append(img_path)
            self.src_texts.append(f"What is the category of the region {loc}?")
            if is_tgt_id:
                self.tgt_texts.append(int(cat_id.split(',')[0]))
            else:
                self.tgt_texts.append(cat_name)
            count += 1

import os

from ..dataset_loader import DatasetLoader

#存在しない画像を除外するためのリスト
dropimageidlist =['patch16_256/objects365_v2_00908726.png','patch6_256/objects365_v1_00320532.png','patch6_256/objects365_v1_00320534.png']

class Objects365_Localization(DatasetLoader):
    """openimageのlocalizationデータセット
    """    
    def __init__(self,data_dir:str="/data01/objects365/",phase:str="train"):
        super().__init__()        
        with open(os.path.join(data_dir,f"{phase}_loc_cut_max_tokens.tsv")) as f:
            items = f.read()

        items = items.split("\n")
        items = [item.split("\t") for item in items]
        items = items[1:]
        self.tgt_texts = [item[3] for item in items]
        self.src_texts = [f"Which regions does the text \"{item[1]}\" describe?" for item in items]
        self.images = [os.path.join(data_dir,item[0]) for item in items]

        #dropimageidlistに含まれる画像と対応するテキストを除外する
        for drop_id in dropimageidlist:
            drop_path = os.path.join(data_dir,"processed_data","images",phase,drop_id)
            while drop_path in self.images:
                drop_index = self.images.index(drop_path)
                self.tgt_texts.pop(drop_index)
                self.src_texts.pop(drop_index)
                self.images.pop(drop_index)
        

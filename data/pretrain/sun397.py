from pathlib import Path
from .pretrain import ClassifyPretrainDatasetLoader
# from torchvision.datasets import SUN397

class SUN397_Pretrain(ClassifyPretrainDatasetLoader):
    def __init__(self, args, data_dir='/data01/sun397', phase='train', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)
        self.data_dir = Path(data_dir) / "SUN397_256"

        with open(self.data_dir / f"{phase}.tsv") as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_path, class_name = line.split("\t")
            img_path = self.data_dir / img_path
            self.images.append(img_path)
            self.src_texts.append(f'{class_name} .')
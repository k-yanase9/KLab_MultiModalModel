from pathlib import Path
from .pretrain import ClassifyPretrainDatasetLoader
# from torchvision.datasets import SUN397

class SUN397_Pretrain(ClassifyPretrainDatasetLoader):
    def __init__(self, args, data_dir='/data01/sun397', resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)
        self.data_dir = Path(data_dir) / "SUN397"

        with open(self.data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self.images = list(self.data_dir.rglob("sun_*.jpg"))
        self.src_texts = [" ".join(path.relative_to(self.data_dir).parts[1:-1]).replace("_"," ")+"." for path in self.images]
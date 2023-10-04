from pathlib import Path
from ..dataset_loader import DatasetLoader

class SUN397_Classify(DatasetLoader):
    def __init__(self, data_dir='/data01/sun397', phase='train'):
        super().__init__()
        self.data_dir = Path(data_dir) / "SUN397_256"

        with open(self.data_dir / "ClassName.txt") as f:
            self.classes = [c.split()[0] for c in f]

        with open(self.data_dir / f"{phase}.tsv") as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_path, _ = line.split("\t")
            label = '/' + '/'.join(img_path.split('/')[:-1])
            img_path = self.data_dir / img_path
            self.images.append(img_path)
            self.src_texts.append('What does the image describe ?')
            self.tgt_texts.append(self.classes.index(label))

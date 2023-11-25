from pathlib import Path

from ..dataset_loader import CLASSIFY_SRC_TEXT, DatasetLoader


class SUN397_Classify(DatasetLoader):
    def __init__(self, data_dir='/data01/sun397', phase='train', is_tgt_id=False):
        super().__init__()
        self.data_dir = Path(data_dir) / "SUN397_256"

        with open(self.data_dir / "ClassName.txt") as f:
            self.classes = [c.split()[0] for c in f]

        with open(self.data_dir / f"{phase}.tsv") as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_path, class_name = line.split("\t")
            label = '/' + '/'.join(img_path.split('/')[:-1])
            img_path = self.data_dir / img_path
            self.images.append(img_path)
            self.src_texts.append(CLASSIFY_SRC_TEXT)
            if is_tgt_id:
                self.tgt_texts.append(self.classes.index(label))
            else:
                self.tgt_texts.append(class_name.strip())

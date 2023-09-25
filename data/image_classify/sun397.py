from pathlib import Path
from ..dataset_loader import DatasetLoader

class SUN397_Classify(DatasetLoader):
    def __init__(self, data_dir='/data01/sun397'):
        super().__init__()
        self.data_dir = Path(data_dir) / "SUN397"

        with open(self.data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]
        
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for image in self.data_dir.rglob("sun_*.jpg"):
            self.images.append(image)
            class_name = '/'.join(image.relative_to(self.data_dir).parts[1:-1])
            self.src_texts.append('What does the image describe ?')
            self.tgt_texts.append(self.class_to_idx[class_name])
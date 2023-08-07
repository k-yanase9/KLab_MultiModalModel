from PIL import Image
from ..dataset_loader import DatasetLoader
from ..mask.utils import make_mask_textpair

class PretrainDatasetLoader(DatasetLoader):
    def __init__(self, data_dir='/data/dataset/redcaps', resize=256):
        super().__init__(resize)

    def __getitem__(self, idx):
        image, text = self.images[idx], self.src_texts[idx]
        src_text, tgt_text = make_mask_textpair(text)

        image = Image.open(image).convert('RGB')#.resize((256,256))
        src_image = self.src_transforms(image)
        tgt_image = self.tgt_transforms(image)
        tgt_image = 2.*tgt_image-1.

        return src_image, tgt_image, src_text, tgt_text
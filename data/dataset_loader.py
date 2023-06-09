import torch
from torchvision.transforms import ToTensor
from PIL import Image

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self):
        self.images, self.tgt_texts, self.src_texts = [], [], []
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image, src_text, tgt_text = self.images[idx], self.src_texts[idx], self.tgt_texts[idx]
        image = Image.open(image).convert('RGB').resize((256,256))
        image = self.transform(image)

        return image, src_text, tgt_text
    
    def __len__(self):
        return len(self.images)

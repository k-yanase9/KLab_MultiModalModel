from torchvision import transforms
from torchvision.datasets import Places365

class Places365PretrainDatasetLoader(Places365):
    def __init__(self, resize=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src_transforms = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.tgt_transforms = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image, src_text = super().__getitem__(index)
        tgt_text = ''

        src_image = self.src_transforms(image)
        tgt_image = self.tgt_transforms(image)
        tgt_image = 2.*tgt_image-1.

        return src_image, tgt_image, src_text, tgt_text

    def __len__(self):
        return super().__len__()
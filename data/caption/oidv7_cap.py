import json
import os

import torch
from PIL import Image
from torchvision import transforms

# 存在しない画像を除外するためのリスト
dropimageidlist = [
    "7f1934f5884fad79",
    "429019e83c1c2c94",
    "4f818c006da84c9e",
    "5b86e93f8654118a",
    "673d74b7d39741c3",
    "6dcd3ce37a17f2be",
    "805baf9650a12710",
    "98ac2996fc46b56d",
    "a46a248a39f2d97c",
    "9316d4095eab6d10",
    "9ee38bb2e69da0ac",
    "37625d59d0e0782a",
]


class OpenImageDataset_Caption(torch.utils.data.Dataset):
    def __init__(self, data_dir="/data/dataset/openimage", phase="train", imagesize=(256, 256)):
        if phase == "val":
            self.phase = "validation"
        else:
            self.phase = phase
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.imagesize = imagesize

        with open(os.path.join(self.data_dir, "caption", f"{self.phase}_caption.jsonl"), "r") as f:
            self.items = [json.loads(s) for s in f]

        items = []
        for item in self.items:
            if item["image_id"] not in dropimageidlist:
                items.append(item)
        self.items = items

    def __getitem__(self, idx):
        src_text = "What does the image describe?"
        tgt_text = self.items[idx]["caption"]
        imgpath = os.path.join(self.data_dir, self.phase, f"{self.items[idx]['image_id']}.jpg")
        image = Image.open(imgpath).convert("RGB").resize(self.imagesize)
        image = self.transform(image)
        return image, src_text, tgt_text

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    dataset = OpenImageDataset_Caption(phase="val")
    data = dataset[0]
    print(data)

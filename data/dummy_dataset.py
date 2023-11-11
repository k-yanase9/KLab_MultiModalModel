import torch
from torch.utils.data import Dataset

from .pretrain.pretrain import PretrainDatasetLoader


class DummyDataset(PretrainDatasetLoader):
    def __init__(self, image_w_h, src_dict_dim, tgt_dict_dim, src_max_length, tgt_max_length, length):
        self.image_w_h = image_w_h
        self.src_dict_dim = src_dict_dim
        self.tgt_dict_dim = tgt_dict_dim
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.length = length

    def __getitem__(self, idx):
        image = torch.rand(3, self.image_w_h[0], self.image_w_h[1])
        src_text = torch.randint(0, self.src_dict_dim - 1, (self.src_max_length,))
        tgt_text = torch.randint(0, self.tgt_dict_dim - 1, (self.tgt_max_length,))

        return image, image, src_text, tgt_text

    def __len__(self):
        return self.length

    def collate_fn(self, batch):
        # バッチ内のテンソルをパッドする
        src_images, tgt_images, src_texts, tgt_texts = [], [], [], []
        for src_image, tgt_image, src_text, tgt_text in batch:
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            src_texts.append(src_text)
            tgt_texts.append(tgt_text)

        src_images = torch.stack(src_images)
        tgt_images = torch.stack(tgt_images)
        src_texts = torch.stack(src_texts)
        tgt_texts = torch.stack(tgt_texts)

        return src_images, tgt_images, src_texts, tgt_texts

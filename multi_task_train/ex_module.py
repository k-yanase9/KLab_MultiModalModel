import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# テスト用のモデル
class ExModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.image_fc = nn.Linear(1, 1)
        self.in_fc = nn.Linear(1, 1)

    def forward(self, image, in_data) -> torch.Tensor:
        """_summary_

        Args:
            image (_type_): (B, 1)
            in_data (_type_): (B, 1)

        Returns:
            torch.Tensor: (B, 1)
        """
        image = self.image_fc(image)
        in_data = self.in_fc(in_data)
        return image + in_data


# csvファイルからデータを読み込むためのDataset
# task_kind = 1なら (10.1234, 11,1234, 12.1234) のようなデータを返す
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return [torch.tensor([float(data)]) for data in self.data[idx]]


class MyChainDataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.length = sum([len(dataset) for dataset in dataset_list])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        for dataset in self.dataset_list:
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx -= len(dataset)
        raise IndexError

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
    def __init__(self, dataset_list, key_list = None):
        """_summary_
        dataset_listにはDatasetのリストを入れる
        key_listはdataset_listの各データセットから取り出すデータのキーのリストを入れる
        例えばkey_list = None とすると, dataset_list[0][idx]から全てのデータを, dataset_list[1][idx]から全てのデータを取り出す
        key_list = [[0, 1], [0, 2]] とすると, dataset_list[0][idx]から0番目と1番目のデータを, dataset_list[1][idx]から0番目と2番目のデータを取り出す
        key_list = [["a", "b"], ["a", "c"]] とすると, dataset_list[0][idx]から"a"と"b"のデータを, dataset_list[1][idx]から"a"と"c"のデータを取り出す

        Args:
            dataset_list (list[Dataset]): データセットのリスト
            key_list (list[list[int  |  str]], optional): データセットから取り出すデータのキーのリスト. Defaults to None.
        """
        self.dataset_list = dataset_list
        self.key_list = key_list
        self.length = sum([len(dataset) for dataset in dataset_list])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        for j, dataset in enumerate(self.dataset_list):
            if idx < len(dataset):
                if self.key_list is None:
                    return dataset[idx]
                else:
                    data = dataset[idx]
                    return [data[key] for key in self.key_list[j]]
            else:
                idx -= len(dataset)
        raise IndexError

import torch
from PIL import Image
from torchvision import transforms

CAPTION_SRC_TEXT = "What does the image describe ?"

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, resize=256):
        self.images, self.tgt_texts, self.src_texts = [], [], []
        self.src_transforms = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tgt_transforms = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        image, src_text, tgt_text = self.images[idx], self.src_texts[idx], self.tgt_texts[idx]
        image = Image.open(image).convert('RGB')
        src_image = self.src_transforms(image)
        tgt_image = torch.zeros(1)

        return src_image, tgt_image, src_text, tgt_text

    def __len__(self):
        return len(self.images)


class MultiChainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list: list[torch.utils.data.Dataset], key_list: list[list[int | str]] = None):
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

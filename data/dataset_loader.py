import torch
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
CLASSIFY_SRC_TEXT = "What is in this image?"
CAPTION_SRC_TEXT = "What does the image describe?"
DETECTION_SRC_TEXT = "What objects are in the image?"
HOI_SRC_TEXT = "What interactions are in the image?"
MAX_VAL_DATA_SIZE = 50000

class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, src_tokenizer=None, tgt_tokenizer=None, src_len=None, tgt_len=None, resize=256, return_img_path=False):
        self.images, self.tgt_texts, self.src_texts = [], [], []
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.return_img_path = return_img_path
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
        image_path, src_text, tgt_text = self.images[idx], self.src_texts[idx], self.tgt_texts[idx]
        image = Image.open(image_path).convert('RGB')
        src_image = self.src_transforms(image)
        tgt_image = torch.zeros(1)
        if self.src_tokenizer is not None:
            src_text = self.src_tokenizer(src_text, max_length=self.src_len, padding='max_length', return_tensors='pt')['input_ids'][0]
        if self.tgt_tokenizer is not None:
            tgt_text = self.tgt_tokenizer(tgt_text, max_length=self.tgt_len, padding='max_length', return_tensors='pt')['input_ids'][0]
            
        if self.return_img_path:
            return src_image, image_path, src_text, tgt_text
        else:
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

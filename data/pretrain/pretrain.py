from PIL import Image
from ..dataset_loader import DatasetLoader

class PretrainDatasetLoader(DatasetLoader):
    def __init__(self, data_dir='/data/dataset/redcaps', resize=256):
        super().__init__(resize)

    def __getitem__(self, idx):
        image, src_text = self.images[idx], self.src_texts[idx]
        # src_text = src_text.replace('.', ' .').replace(',', ' ,').replace('!', ' !').replace('?', ' ?') # ,.!?の前にスペースを挿入
        # src_text = src_text.split() # 単語に分割
        # len(src_text) * 0.15だけランダムにsrc_textのインデックスを取得
        # mask_idx = torch.randperm(len(src_text))[:int(len(src_text) * 0.15)+1]

        # tgt_text = ['<extra_id_0>']
        # j = 0
        # for i in range(len(src_text)):
        #     if i in mask_idx:
        #         tgt_text.append(src_text[i])
        #         tgt_text.append(f'<extra_id_{j+1}>')
        #         src_text[i] = f'<extra_id_{j}>'
        #         j += 1
        # src_text = ' '.join(src_text)
        # tgt_text = ' '.join(tgt_text)
        tgt_text = ''

        image = Image.open(image).convert('RGB')#.resize((256,256))
        src_image = self.src_transforms(image)
        tgt_image = self.tgt_transforms(image)
        tgt_image = 2.*tgt_image-1.

        return src_image, tgt_image, src_text, tgt_text
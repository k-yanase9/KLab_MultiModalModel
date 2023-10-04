import torch
import random
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from ..dataset_loader import DatasetLoader

class PretrainDatasetLoader(DatasetLoader):
    def __init__(self, args, resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(resize)
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length 
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.mask_tokens = src_tokenizer.additional_special_tokens_ids
        self.mask_prob = mask_probability

    def __getitem__(self, idx):
        image, text = self.images[idx], self.src_texts[idx]
        src_text = self.tgt_tokenizer.encode_plus(text, return_attention_mask=False, verbose=False)["input_ids"][:-1]
        tgt_text = self.generate_target_ids(src_text)
        src_text = torch.tensor(src_text)
        tgt_text = torch.tensor(tgt_text)

        image = Image.open(image).convert('RGB')#.resize((256,256))
        src_image = self.src_transforms(image)
        # tgt_image = self.tgt_transforms(image)
        # tgt_image = 2.*tgt_image-1.
        tgt_image = torch.zeros(1)

        return src_image, tgt_image, src_text, tgt_text
    
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
        src_texts = pad_sequence(src_texts, batch_first=True, padding_value=self.src_tokenizer.pad_token_id)
        tgt_texts = pad_sequence(tgt_texts, batch_first=True, padding_value=self.tgt_tokenizer.pad_token_id)
        src_attention_masks = torch.ones_like(src_texts)
        src_attention_masks[src_texts == self.src_tokenizer.pad_token_id] = 0
        tgt_attention_masks = torch.ones_like(tgt_texts)
        tgt_attention_masks[tgt_texts == self.tgt_tokenizer.pad_token_id] = 0
        src_inputs = {"input_ids": src_texts, "attention_mask": src_attention_masks}
        tgt_inputs = {"input_ids": tgt_texts, "attention_mask": tgt_attention_masks}

        return src_images, tgt_images, src_inputs, tgt_inputs
    
    def generate_target_ids(self, input_id):
        target_id = []
        masked_indexes = sorted(random.sample(range(0, len(input_id)),  # sample a word index in sentence
                                                min(max(int(self.mask_prob * len(input_id)),1),  # number of tokens masked
                                                    len(self.mask_tokens) - 1)))  # but never more than special tokens available
        mask = [(i in masked_indexes)  # this is True or False
                for i in range(len(input_id))]
        i = 0
        end = len(input_id)
        masked_spans_counter = 0
        while i < end:
            if mask[i]:
                current_words_masked = [input_id[i]]
                input_id[i] = self.mask_tokens[masked_spans_counter]
                masked_spans_counter += 1
                while i + 1 < end and mask[i + 1]:
                    current_words_masked.append(input_id[i + 1])
                    del input_id[i + 1]
                    del mask[i + 1]
                    end -= 1
                target_id.extend(current_words_masked)
            else:
                if len(target_id) == 0 or target_id[-1] != self.mask_tokens[masked_spans_counter]:
                    target_id.append(self.mask_tokens[masked_spans_counter])
            i += 1
        return target_id
    
class ClassifyPretrainDatasetLoader(PretrainDatasetLoader):
    def __init__(self, args, resize=256, src_tokenizer=None, tgt_tokenizer=None, mask_probability=0.15):
        super().__init__(args, resize, src_tokenizer, tgt_tokenizer, mask_probability)

    def __getitem__(self, idx):
        image, text = self.images[idx], self.src_texts[idx]
        rate = random.random()
        period = '.'# if rate * 10 % 2 == 0 else ' .'
        if rate < 0.25:
            text = 'A short image description: ' + text + period
        elif rate < 0.5:
            text = 'An image of ' + text + period
        elif rate < 0.75:
            text = 'A photo of ' + text + period
        else:
            text = 'An image that shows ' + text + period

        src_text = self.tgt_tokenizer.encode_plus(text, return_attention_mask=False, verbose=False)["input_ids"][:-1]
        tgt_text = self.generate_target_ids(src_text)
        src_text = torch.tensor(src_text)
        tgt_text = torch.tensor(tgt_text)

        image = Image.open(image).convert('RGB')#.resize((256,256))
        src_image = self.src_transforms(image)
        # tgt_image = self.tgt_transforms(image)
        # tgt_image = 2.*tgt_image-1.
        tgt_image = torch.zeros(1)

        return src_image, tgt_image, src_text, tgt_text
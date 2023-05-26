import torch
from torch import nn
from transformers import AutoTokenizer, AutoImageProcessor, T5EncoderModel, Swinv2Model, T5ForConditionalGeneration

# モデルの定義
class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.max_source_length = 256
        self.max_target_length = 128
        
        self.image_processor = AutoImageProcessor.from_pretrained(args.image_model_name)
        self.image_model = Swinv2Model.from_pretrained(args.image_model_name).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=512)
        self.language_model = T5EncoderModel.from_pretrained(args.language_model_name).to(self.device) # device_map="auto"

        self.transformer = T5ForConditionalGeneration.from_pretrained(args.language_model_name).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, images, src_texts, tgt_texts):
        with torch.no_grad():
            images = self.image_processor(images, return_tensors="pt").to(self.device)

            image_embeddings = self.image_model(**images).last_hidden_state

            source_encoding = self.tokenizer(src_texts, padding="longest", max_length=self.max_source_length, return_tensors='pt').to(self.device) # ['pt', 'tf', 'np', 'jax']
            language_embeddings = self.language_model(source_encoding['input_ids']).last_hidden_state

            concated_embeddings = torch.cat((image_embeddings,language_embeddings), dim=1)

        target_encoding = self.tokenizer(tgt_texts, padding="longest", max_length=self.max_target_length, return_tensors='pt').to(self.device) # ['pt', 'tf', 'np', 'jax']

        loss = self.transformer(inputs_embeds=concated_embeddings, labels=target_encoding['input_ids']).loss

        return loss
    
    def save(self, result_dir):
        torch.save(self.transformer.state_dict(), f"{result_dir}/best.pth")

    def load(self, result_dir):
        self.transformer.load_state_dict(torch.load(f"{result_dir}/best.pth"))
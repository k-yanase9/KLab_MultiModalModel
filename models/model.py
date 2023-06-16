import os
import torch
from torch import nn
from transformers import T5EncoderModel, Swinv2Model, T5ForConditionalGeneration, logging, ResNetModel
logging.set_verbosity_error()

# モデルの定義
class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.result_dir = args.result_dir
        
        self.language_model = T5EncoderModel.from_pretrained(args.language_model_name).requires_grad_(False) # device_map="auto"
        self.language_model.eval()

        if "resnet" in args.image_model_name:
            self.image_model = ResNetModel.from_pretrained(args.image_model_name).requires_grad_(args.image_model_train)
        elif "swinv2" in args.image_model_name:
            self.image_model = Swinv2Model.from_pretrained(args.image_model_name).requires_grad_(args.image_model_train)

        self.transformer = T5ForConditionalGeneration.from_pretrained(args.transformer_model_name)

        if args.ffn:
            self.language_ffn = nn.Linear(self.language_model.config.d_model, self.transformer.config.d_model)
            self.image_ffn = nn.Linear(self.image_model.num_features, self.transformer.config.d_model)

    def forward(self, images, src_texts, tgt_texts=None, return_loss=True):
        with torch.no_grad():
            language_embeddings = self.language_model(src_texts).last_hidden_state
        image_embeddings = self.image_model(images).last_hidden_state

        if self.args.ffn:
            language_embeddings = self.language_ffn(language_embeddings)
            image_embeddings = self.image_ffn(image_embeddings)

        concated_embeddings = torch.cat((image_embeddings,language_embeddings), dim=1)

        if return_loss:
            return self.transformer(inputs_embeds=concated_embeddings, labels=tgt_texts).loss
        else:
            return self.transformer.generate(inputs_embeds=concated_embeddings)
    
    def save(self, result_name="best.pth"):
        result_path = os.path.join(self.args.result_dir, result_name)
        checkpoints = {'transformer': self.transformer.state_dict()}
        if self.args.image_model_train:
            checkpoints['image_model'] = self.image_model.state_dict()
        if self.args.ffn:
            checkpoints['language_ffn'] = self.language_ffn.state_dict()
            checkpoints['image_ffn'] = self.image_ffn.state_dict()
        torch.save(checkpoints, result_path)

    def load(self, result_name="best.pth"):
        result_path = os.path.join(self.args.result_dir, result_name)
        checkpoints = torch.load(result_path)
        self.transformer.load_state_dict(checkpoints['transformer'])
        if self.args.image_model_train:
            self.image_model.load_state_dict(checkpoints['image_model'])
        if self.args.ffn:
            self.language_ffn.load_state_dict(checkpoints['language_ffn'])
            self.image_ffn.load_state_dict(checkpoints['image_ffn'])
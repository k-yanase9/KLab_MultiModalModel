import os
import torch
from torch import nn
from transformers import T5EncoderModel, Swinv2Model, T5ForConditionalGeneration, logging
logging.set_verbosity_error()

# モデルの定義
class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.result_dir = args.result_dir
        
        self.language_model = T5EncoderModel.from_pretrained(args.language_model_name).requires_grad_(False) # device_map="auto"
        self.image_model = Swinv2Model.from_pretrained(args.image_model_name).requires_grad_(args.image_model_train)

        self.transformer = T5ForConditionalGeneration.from_pretrained(args.transformer_model_name)

    def forward(self, images, source_encoding, target_encoding=None, return_loss=True):
        with torch.no_grad():
            language_embeddings = self.language_model(source_encoding['input_ids']).last_hidden_state
        image_embeddings = self.image_model(**images).last_hidden_state
        concated_embeddings = torch.cat((image_embeddings,language_embeddings), dim=1)

        if return_loss:
            return self.transformer(inputs_embeds=concated_embeddings, labels=target_encoding['input_ids']).loss
        else:
            return self.transformer.generate(inputs_embeds=concated_embeddings)
    
    def save(self, result_name="best.pth"):
        result_path = os.path.join(self.args.result_dir, result_name)
        checkpoints = {'transformer': self.transformer.state_dict()}
        if self.args.image_model_train:
            checkpoints['image_model'] = self.image_model.state_dict()
        torch.save(checkpoints, result_path)

    def load(self, result_name="best.pth"):
        result_path = os.path.join(self.args.result_dir, result_name)
        checkpoints = torch.load(result_path)
        self.transformer.load_state_dict(checkpoints['transformer'])
        if self.args.image_model_train:
            self.image_model.load_state_dict(checkpoints['image_model'])
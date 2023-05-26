import torch
from torch import nn
from transformers import T5EncoderModel, Swinv2Model, T5ForConditionalGeneration, logging
logging.set_verbosity_error()

# モデルの定義
class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.image_model = Swinv2Model.from_pretrained(args.image_model_name).requires_grad_(False)

        self.language_model = T5EncoderModel.from_pretrained(args.language_model_name).requires_grad_(False) # device_map="auto"

        self.transformer = T5ForConditionalGeneration.from_pretrained(args.transformer_model_name)

    def forward(self, images, source_encoding, target_encoding=None, return_loss=True):
        with torch.no_grad():
            image_embeddings = self.image_model(**images).last_hidden_state
            language_embeddings = self.language_model(source_encoding['input_ids']).last_hidden_state
            concated_embeddings = torch.cat((image_embeddings,language_embeddings), dim=1)

        if return_loss:
            return self.transformer(inputs_embeds=concated_embeddings, labels=target_encoding['input_ids']).loss
        else:
            return self.transformer.generate(inputs_embeds=concated_embeddings)
    
    def save(self, result_dir):
        torch.save(self.transformer.state_dict(), f"{result_dir}/best.pth")

    def load(self, result_dir):
        self.transformer.load_state_dict(torch.load(f"{result_dir}/best.pth"))
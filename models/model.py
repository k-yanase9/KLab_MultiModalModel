import os
import torch
import numpy as np
from torch import nn
from transformers import T5EncoderModel, Swinv2Model, T5Config, logging, ResNetModel, T5ForConditionalGeneration
from models.vqgan import VQModel
logging.set_verbosity_error()

# モデルの定義
class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.result_dir = args.result_dir
        
        self.vae = VQModel(ckpt_path=args.vae_ckpt_path).requires_grad_(False)
        self.vae.eval()
        self.language_model = T5EncoderModel.from_pretrained(args.language_model_name).requires_grad_(False) # device_map="auto"
        self.language_model.eval()

        if "resnet" in args.image_model_name: # 事前学習用に書き換えたのでおそらく動かない
            self.image_model = ResNetModel.from_pretrained(args.image_model_name).requires_grad_(args.image_model_train)
        elif "swinv2" in args.image_model_name:
            self.image_model = Swinv2Model.from_pretrained(args.image_model_name, use_mask_token=args.pretrain).requires_grad_(args.image_model_train)
            self.num_patches = (self.image_model.config.image_size // self.image_model.config.patch_size) ** 2

        transformer_config = T5Config(
            vocab_size=32128+args.loc_vocab_size+args.image_vocab_size, 
            d_model=args.transformer_d_model,
            d_ff=args.transformer_d_ff,
            d_kv=args.transformer_d_kv,
            num_heads=args.transformer_num_heads,
            num_layers=args.transformer_num_layers,
            num_decoder_layers=args.transformer_num_decoder_layers,
            decoder_start_token_id=0,
            max_length=args.max_target_length,
        )
        self.transformer = T5ForConditionalGeneration(transformer_config)

        # self.transformer = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").requires_grad_(True)
        # vocab_size = 32128+args.loc_vocab_size+args.image_vocab_size
        # self.transformer.shared = nn.Embedding(vocab_size, self.transformer.model_dim)
        # self.transformer.encoder.embed_tokens = self.transformer.shared
        # self.transformer.decoder.embed_tokens = self.transformer.shared
        # self.transformer.lm_head = nn.Linear(self.transformer.model_dim, vocab_size, bias=False)

        if args.ffn:
            self.language_ffn = nn.Linear(self.language_model.config.d_model, self.transformer.config.d_model)
            self.image_ffn = nn.Linear(self.image_model.num_features, self.transformer.config.d_model)

    def forward(self, images, src_texts, tgt_texts=None, return_loss=True, num_beams=1, num_return_sequences=1, do_sample=False):
        with torch.no_grad():
            language_embeddings = self.language_model(src_texts).last_hidden_state

        if self.args.pretrain: # 事前学習だったら画像パッチにマスクをかける
            bool_masked_pos = self.random_patch_masking(len(images))
        else:
            bool_masked_pos = None
        image_embeddings = self.image_model(pixel_values=images, bool_masked_pos=bool_masked_pos).last_hidden_state

        if self.args.ffn:
            language_embeddings = self.language_ffn(language_embeddings)
            image_embeddings = self.image_ffn(image_embeddings)

        concated_embeddings = torch.cat((image_embeddings,language_embeddings), dim=1)

        if return_loss:
            return self.transformer(inputs_embeds=concated_embeddings, labels=tgt_texts).loss
        else:
            return self.transformer.generate(inputs_embeds=concated_embeddings, num_beams=num_beams, num_return_sequences=num_return_sequences, do_sample=do_sample, max_length=self.args.max_target_length)
    
    def random_patch_masking(self, batch_size):
        len_keep = int(self.num_patches * (1 - self.args.image_mask_ratio))
        noise = torch.rand(batch_size, self.num_patches, device=self.image_model.device)

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([batch_size, self.num_patches], device=self.image_model.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask
    
    def image_to_z(self, images):
        z = self.vae.get_codebook_indices(images) # VAEで中間表現を得る
        z_text = z.cpu().numpy().astype(str) # 文字列に変換
        z_text = np.char.add(np.char.add('<img_', z_text), '>') # <extra_id_0>のようにする
        z_text = [''.join(b) for b in z_text]
        return z_text, z
    
    def z_to_image(self, z):
        x = self.vae.decode_code(z)
        return x

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
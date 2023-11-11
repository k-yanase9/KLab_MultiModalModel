import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.pop('vocab_size', 10000)
        self.d_ff = kwargs.pop('d_ff', 2048)
        self.d_model = kwargs.pop('d_model', 512)
        self.dropput_rate = kwargs.pop('dropout_rate', 0.1)
        self.max_length = kwargs.pop('max_length', 512)
        self.num_heads = kwargs.pop('num_heads', 8)
        self.num_layers = kwargs.pop('num_layers', 6)

class PositionalEncoding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(PositionalEncoding, self).__init__()
        self.d_model = config.d_model
        self.max_length = config.max_length
        self.pos_encoding = self._get_pos_encoding()

    def _get_angles(self, pos, i):
        angles = 1 / torch.pow(10000, (2 * (i // 2)) / self.d_model)
        return pos * angles
    
    def _get_pos_encoding(self):
        pos_encoding = self._get_angles(torch.arange(self.max_length).unsqueeze(1), torch.arange(self.d_model).unsqueeze(0))
        pos_encoding[:, 0::2] = torch.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(pos_encoding[:, 1::2])
        return pos_encoding.unsqueeze(0)
    
    def forward(self, x):
        pos_encoding = self.pos_encoding[:, :x.size(1), :].to(x.device)
        return x + pos_encoding

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        assert config.d_model % self.num_heads == 0
        self.d_k = config.d_model // self.num_heads
        
        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)

    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)
        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.dropout = nn.Dropout(config.dropput_rate)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, x, attention_mask=None):
        # Self Attention
        attention_output = self.attention(x, x, x, attention_mask)
        x = self.norm1(attention_output + x)
        
        # Feed Forward
        ff_output = self.ff(x)
        output = self.norm2(ff_output + x)
        
        return output

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, use_embedding=False):
        super(Transformer, self).__init__()
        self.config = config
        if use_embedding:
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config)
        self.encoder = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        # デコーダの実装も必要に応じて

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=inputs_embeds.device)
        inputs_embeds = self.positional_encoding(inputs_embeds)
        for layer in self.encoder:
            inputs_embeds = layer(inputs_embeds, attention_mask)
        return [inputs_embeds]